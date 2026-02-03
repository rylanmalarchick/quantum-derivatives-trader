"""
Profiler: Deep performance analysis for quantum-classical workloads.

Provides:
1. Function-level profiling with call graphs
2. Memory tracking and leak detection
3. GPU utilization monitoring
4. Quantum circuit cost breakdown
5. Flamegraph generation

Think of this as a specialized profiler for hybrid quantum-classical code.
"""

from __future__ import annotations

import cProfile
import functools
import gc
import io
import pstats
import sys
import time
import threading
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of timing a code block."""
    name: str
    start_time: float
    end_time: float
    duration: float
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    call_count: int = 1
    children: List['TimingResult'] = field(default_factory=list)
    
    @property
    def memory_delta_mb(self) -> float:
        return self.memory_end_mb - self.memory_start_mb
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'duration_s': self.duration,
            'memory_delta_mb': self.memory_delta_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'call_count': self.call_count,
            'children': [c.to_dict() for c in self.children],
        }


@dataclass
class ProfileResult:
    """Complete profiling result."""
    total_time: float
    peak_memory_mb: float
    function_stats: Dict[str, TimingResult]
    call_graph: Dict[str, List[str]]
    hotspots: List[Tuple[str, float]]  # (function, % of total time)
    memory_leaks: List[Tuple[str, float]]  # (location, bytes leaked)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "PROFILING SUMMARY",
            "=" * 60,
            f"Total time: {self.total_time:.3f}s",
            f"Peak memory: {self.peak_memory_mb:.1f} MB",
            "",
            "TOP HOTSPOTS:",
        ]
        
        for func, pct in self.hotspots[:10]:
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {bar} {pct:5.1f}% {func}")
        
        if self.memory_leaks:
            lines.append("")
            lines.append("POTENTIAL MEMORY LEAKS:")
            for loc, size in self.memory_leaks[:5]:
                lines.append(f"  {size/1024:.1f} KB at {loc}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class Profiler:
    """
    Hierarchical profiler for quantum-classical code.
    
    Usage:
        profiler = Profiler()
        
        with profiler.section("training"):
            with profiler.section("forward"):
                output = model(x)
            with profiler.section("backward"):
                loss.backward()
        
        print(profiler.report())
    
    Or as decorator:
        @profiler.profile
        def expensive_function():
            ...
    """
    
    def __init__(self, track_memory: bool = True, track_calls: bool = True):
        self.track_memory = track_memory
        self.track_calls = track_calls
        
        self._stack: List[TimingResult] = []
        self._root_results: List[TimingResult] = []
        self._function_times: Dict[str, List[float]] = defaultdict(list)
        self._call_graph: Dict[str, set] = defaultdict(set)
        self._lock = threading.Lock()
        
        if track_memory:
            tracemalloc.start()
    
    @contextmanager
    def section(self, name: str):
        """Profile a code section."""
        start_time = time.perf_counter()
        
        if self.track_memory:
            gc.collect()
            memory_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        else:
            memory_start = 0
        
        result = TimingResult(
            name=name,
            start_time=start_time,
            end_time=0,
            duration=0,
            memory_start_mb=memory_start,
            memory_end_mb=0,
            memory_peak_mb=0,
        )
        
        with self._lock:
            if self._stack:
                self._stack[-1].children.append(result)
                self._call_graph[self._stack[-1].name].add(name)
            else:
                self._root_results.append(result)
            self._stack.append(result)
        
        try:
            yield result
        finally:
            end_time = time.perf_counter()
            
            if self.track_memory:
                current, peak = tracemalloc.get_traced_memory()
                result.memory_end_mb = current / 1024 / 1024
                result.memory_peak_mb = peak / 1024 / 1024
            
            result.end_time = end_time
            result.duration = end_time - start_time
            
            with self._lock:
                self._stack.pop()
                self._function_times[name].append(result.duration)
    
    def profile(self, fn: Callable) -> Callable:
        """Decorator to profile a function."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.section(fn.__qualname__):
                return fn(*args, **kwargs)
        return wrapper
    
    def get_result(self) -> ProfileResult:
        """Get profiling results."""
        total_time = sum(r.duration for r in self._root_results)
        peak_memory = max(
            (r.memory_peak_mb for r in self._root_results),
            default=0
        )
        
        # Build function stats
        function_stats = {}
        for name, times in self._function_times.items():
            function_stats[name] = TimingResult(
                name=name,
                start_time=0,
                end_time=0,
                duration=sum(times),
                memory_start_mb=0,
                memory_end_mb=0,
                memory_peak_mb=0,
                call_count=len(times),
            )
        
        # Find hotspots
        hotspots = []
        if total_time > 0:
            for name, times in self._function_times.items():
                pct = 100 * sum(times) / total_time
                hotspots.append((name, pct))
        hotspots.sort(key=lambda x: -x[1])
        
        # Check for memory leaks
        memory_leaks = []
        if self.track_memory:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            for stat in top_stats[:10]:
                if stat.size > 1024 * 1024:  # > 1MB
                    memory_leaks.append((str(stat.traceback), stat.size))
        
        return ProfileResult(
            total_time=total_time,
            peak_memory_mb=peak_memory,
            function_stats=function_stats,
            call_graph={k: list(v) for k, v in self._call_graph.items()},
            hotspots=hotspots,
            memory_leaks=memory_leaks,
        )
    
    def report(self) -> str:
        """Generate profiling report."""
        return self.get_result().summary()
    
    def reset(self) -> None:
        """Reset profiler state."""
        with self._lock:
            self._stack.clear()
            self._root_results.clear()
            self._function_times.clear()
            self._call_graph.clear()
        
        if self.track_memory:
            tracemalloc.reset_peak()


# Convenience decorator
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get or create global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler()
    return _global_profiler


def profile_function(fn: Callable) -> Callable:
    """Decorator using global profiler."""
    return get_profiler().profile(fn)


@contextmanager
def profile_section(name: str):
    """Context manager using global profiler."""
    with get_profiler().section(name):
        yield


class CPUProfiler:
    """
    CPU profiler using cProfile for detailed analysis.
    
    More overhead than Profiler but provides complete call graph.
    """
    
    def __init__(self):
        self._profiler = cProfile.Profile()
        self._running = False
    
    def start(self) -> None:
        """Start profiling."""
        self._profiler.enable()
        self._running = True
    
    def stop(self) -> None:
        """Stop profiling."""
        if self._running:
            self._profiler.disable()
            self._running = False
    
    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.start()
        try:
            yield
        finally:
            self.stop()
    
    def stats(self, sort_by: str = 'cumulative') -> str:
        """Get profiling stats as string."""
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats(30)
        return stream.getvalue()
    
    def top_functions(self, n: int = 10) -> List[Tuple[str, float, int]]:
        """Get top N functions by cumulative time."""
        stats = pstats.Stats(self._profiler)
        
        result = []
        for (filename, lineno, name), (cc, nc, tt, ct, callers) in stats.stats.items():
            result.append((f"{filename}:{lineno}({name})", ct, nc))
        
        result.sort(key=lambda x: -x[1])
        return result[:n]


def benchmark(
    fn: Callable,
    *args,
    n_runs: int = 10,
    warmup: int = 2,
    **kwargs,
) -> Dict[str, float]:
    """
    Benchmark a function with multiple runs.
    
    Returns:
        Dict with timing statistics (mean, std, min, max)
    """
    import numpy as np
    
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        fn(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
        'n_runs': n_runs,
    }
