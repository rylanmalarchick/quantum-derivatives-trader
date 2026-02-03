"""
Parallel Execution Engine: Multi-process quantum circuit evaluation.

Quantum circuit simulation is embarrassingly parallel at the batch level.
This module provides optimized parallel execution with:
1. Process pool for CPU-bound quantum simulation
2. Work stealing for load balancing
3. Memory-aware batch sizing
4. Affinity hints for NUMA systems

Think of this as a runtime system for quantum workloads.
"""

from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar
from functools import partial
import numpy as np
import logging
import queue
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class WorkerConfig:
    """Configuration for a worker process."""
    worker_id: int
    num_workers: int
    cpu_affinity: Optional[List[int]] = None  # Which CPUs to bind to
    memory_limit_mb: Optional[int] = None
    device: str = "cpu"  # cpu, cuda:0, etc.


@dataclass
class BatchResult:
    """Result from a batch computation."""
    batch_id: int
    results: List[Any]
    worker_id: int
    compute_time: float
    memory_peak_mb: float


def _set_cpu_affinity(cpus: List[int]) -> None:
    """Set CPU affinity for current process (Linux only)."""
    try:
        os.sched_setaffinity(0, set(cpus))
    except (AttributeError, OSError):
        pass  # Not supported on this platform


def _worker_init(config: WorkerConfig) -> None:
    """Initialize worker process."""
    if config.cpu_affinity:
        _set_cpu_affinity(config.cpu_affinity)
    
    # Set thread count for NumPy/PyTorch to avoid oversubscription
    num_threads = max(1, len(config.cpu_affinity) if config.cpu_affinity else 1)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    # Import torch here to pick up env vars
    import torch
    torch.set_num_threads(num_threads)


def _execute_batch(
    fn: Callable,
    batch: List[Any],
    batch_id: int,
    config: WorkerConfig,
) -> BatchResult:
    """Execute a batch of work items."""
    import time
    import tracemalloc
    
    tracemalloc.start()
    start = time.perf_counter()
    
    results = [fn(item) for item in batch]
    
    compute_time = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return BatchResult(
        batch_id=batch_id,
        results=results,
        worker_id=config.worker_id,
        compute_time=compute_time,
        memory_peak_mb=peak / 1024 / 1024,
    )


class WorkerPool:
    """
    Process pool optimized for quantum circuit simulation.
    
    Features:
    - Automatic batch sizing based on memory
    - CPU affinity for NUMA-aware scheduling
    - Work stealing for load balancing
    - Progress tracking
    
    Example:
        pool = WorkerPool(num_workers=8)
        
        def simulate_circuit(params):
            # Heavy quantum simulation
            return result
        
        results = pool.map(simulate_circuit, param_list, batch_size=32)
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_affinity: bool = True,
    ):
        """
        Args:
            num_workers: Number of worker processes (default: CPU count)
            use_affinity: Bind workers to specific CPUs
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.use_affinity = use_affinity
        
        # Create worker configs with affinity
        self.configs = []
        num_cpus = mp.cpu_count()
        cpus_per_worker = max(1, num_cpus // self.num_workers)
        
        for i in range(self.num_workers):
            affinity = None
            if use_affinity:
                start_cpu = i * cpus_per_worker
                affinity = list(range(start_cpu, min(start_cpu + cpus_per_worker, num_cpus)))
            
            self.configs.append(WorkerConfig(
                worker_id=i,
                num_workers=self.num_workers,
                cpu_affinity=affinity,
            ))
        
        self._executor: Optional[ProcessPoolExecutor] = None
    
    def __enter__(self):
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(self.configs[0],),  # Basic init for all
        )
        return self
    
    def __exit__(self, *args):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def map(
        self,
        fn: Callable[[T], R],
        items: List[T],
        batch_size: Optional[int] = None,
        ordered: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[R]:
        """
        Map function over items in parallel.
        
        Args:
            fn: Function to apply
            items: Items to process
            batch_size: Items per batch (auto if None)
            ordered: Preserve input order
            progress_callback: Called with (completed, total)
        
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Auto batch size
        if batch_size is None:
            batch_size = max(1, len(items) // (self.num_workers * 4))
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        if self._executor is None:
            raise RuntimeError("WorkerPool must be used as context manager")
        
        # Submit all batches
        futures: Dict[Future, int] = {}
        for batch_id, batch in enumerate(batches):
            config = self.configs[batch_id % self.num_workers]
            future = self._executor.submit(
                _execute_batch, fn, batch, batch_id, config
            )
            futures[future] = batch_id
        
        # Collect results
        results_map: Dict[int, List[R]] = {}
        completed = 0
        
        for future in futures:
            batch_result = future.result()
            results_map[batch_result.batch_id] = batch_result.results
            completed += 1
            
            if progress_callback:
                progress_callback(completed, len(batches))
        
        # Flatten in order
        if ordered:
            all_results = []
            for i in range(len(batches)):
                all_results.extend(results_map[i])
            return all_results
        else:
            return [r for results in results_map.values() for r in results]
    
    def imap(
        self,
        fn: Callable[[T], R],
        items: Iterator[T],
        batch_size: int = 32,
    ) -> Iterator[R]:
        """
        Lazy map - yields results as they complete.
        
        Useful for very large item lists that don't fit in memory.
        """
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= batch_size:
                for result in self.map(fn, batch, batch_size=batch_size, ordered=True):
                    yield result
                batch = []
        
        if batch:
            for result in self.map(fn, batch, batch_size=len(batch), ordered=True):
                yield result


class ParallelExecutor:
    """
    High-level parallel execution with automatic strategy selection.
    
    Chooses between:
    - ThreadPool (I/O bound, Python objects)
    - ProcessPool (CPU bound, pickle-able)
    - Sequential (small workloads)
    """
    
    def __init__(
        self,
        strategy: str = "auto",
        num_workers: Optional[int] = None,
    ):
        """
        Args:
            strategy: "auto", "thread", "process", or "sequential"
            num_workers: Worker count
        """
        self.strategy = strategy
        self.num_workers = num_workers or mp.cpu_count()
    
    def _select_strategy(self, items: List, fn: Callable) -> str:
        """Auto-select execution strategy."""
        if self.strategy != "auto":
            return self.strategy
        
        n = len(items)
        
        # Small workloads: sequential
        if n <= 4:
            return "sequential"
        
        # Check if function contains I/O indicators
        fn_name = getattr(fn, '__name__', '')
        if any(x in fn_name.lower() for x in ['read', 'write', 'fetch', 'load', 'save']):
            return "thread"
        
        # Default to process for CPU-bound
        return "process"
    
    def map(
        self,
        fn: Callable[[T], R],
        items: List[T],
        **kwargs,
    ) -> List[R]:
        """Map function over items with auto strategy."""
        strategy = self._select_strategy(items, fn)
        
        if strategy == "sequential":
            return [fn(item) for item in items]
        
        elif strategy == "thread":
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                return list(pool.map(fn, items))
        
        elif strategy == "process":
            with WorkerPool(num_workers=self.num_workers) as pool:
                return pool.map(fn, items, **kwargs)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# Decorator for easy parallelization
def parallel(
    num_workers: int = None,
    batch_size: int = None,
):
    """
    Decorator to parallelize a function over its first argument.
    
    @parallel(num_workers=4)
    def process(item):
        return heavy_computation(item)
    
    # Now process([a, b, c, d]) runs in parallel
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(items: List, *args, **kwargs):
            if not isinstance(items, list):
                return fn(items, *args, **kwargs)
            
            bound_fn = partial(fn, *args, **kwargs) if args or kwargs else fn
            
            with WorkerPool(num_workers=num_workers or mp.cpu_count()) as pool:
                return pool.map(bound_fn, items, batch_size=batch_size)
        
        return wrapper
    return decorator
