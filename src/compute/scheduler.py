"""
Task Scheduler: Priority-based scheduling with dependency resolution.

Implements a DAG-based scheduler that:
1. Topologically sorts tasks by dependencies
2. Schedules based on priority and estimated cost
3. Maximizes parallelism while respecting constraints
4. Handles failures with retry logic

Think of this as a mini-compiler pass scheduler.
"""

from __future__ import annotations

import heapq
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from queue import PriorityQueue
import logging

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Task priority levels (lower = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskState(IntEnum):
    """Task execution state."""
    PENDING = 0
    READY = 1      # All dependencies satisfied
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5


@dataclass
class ComputeTask:
    """
    A unit of computation with dependencies.
    
    Attributes:
        name: Unique identifier
        fn: Callable to execute
        args: Positional arguments
        kwargs: Keyword arguments
        priority: Scheduling priority
        dependencies: Set of task names that must complete first
        estimated_cost: Relative cost estimate (for scheduling heuristics)
        timeout: Maximum execution time in seconds
        retries: Number of retry attempts on failure
    """
    name: str
    fn: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    dependencies: Set[str] = field(default_factory=set)
    estimated_cost: float = 1.0
    timeout: Optional[float] = None
    retries: int = 0
    
    # Internal state
    _state: TaskState = field(default=TaskState.PENDING, repr=False)
    _result: Any = field(default=None, repr=False)
    _error: Optional[Exception] = field(default=None, repr=False)
    _start_time: Optional[float] = field(default=None, repr=False)
    _end_time: Optional[float] = field(default=None, repr=False)
    
    @property
    def state(self) -> TaskState:
        return self._state
    
    @property
    def result(self) -> Any:
        if self._state != TaskState.COMPLETED:
            raise RuntimeError(f"Task {self.name} not completed")
        return self._result
    
    @property
    def duration(self) -> Optional[float]:
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return None
    
    def __hash__(self):
        return hash(self.name)
    
    def __lt__(self, other):
        """Compare by priority for heap operations."""
        return self.priority < other.priority


@dataclass
class SchedulerStats:
    """Statistics from scheduler execution."""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total_time: float = 0.0
    max_parallelism: int = 0
    avg_queue_depth: float = 0.0


class TaskScheduler:
    """
    DAG-based task scheduler with parallel execution.
    
    Features:
    - Dependency resolution via topological sort
    - Priority-based scheduling
    - Parallel execution with configurable workers
    - Failure handling and retries
    - Real-time progress tracking
    
    Example:
        scheduler = TaskScheduler(max_workers=4)
        
        scheduler.add_task(ComputeTask(
            name="load_data",
            fn=load_data,
            priority=Priority.HIGH,
        ))
        
        scheduler.add_task(ComputeTask(
            name="preprocess",
            fn=preprocess,
            dependencies={"load_data"},
        ))
        
        scheduler.add_task(ComputeTask(
            name="train",
            fn=train,
            dependencies={"preprocess"},
            estimated_cost=10.0,
        ))
        
        results = scheduler.run()
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        fail_fast: bool = False,
        progress_callback: Optional[Callable[[str, TaskState], None]] = None,
    ):
        """
        Args:
            max_workers: Maximum concurrent tasks
            fail_fast: Stop all tasks on first failure
            progress_callback: Called on each task state change
        """
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.progress_callback = progress_callback
        
        self.tasks: Dict[str, ComputeTask] = {}
        self.dependents: Dict[str, Set[str]] = defaultdict(set)  # task -> tasks that depend on it
        
        self._lock = threading.Lock()
        self._stats = SchedulerStats()
    
    def add_task(self, task: ComputeTask) -> None:
        """Add a task to the schedule."""
        if task.name in self.tasks:
            raise ValueError(f"Task {task.name} already exists")
        
        self.tasks[task.name] = task
        
        # Register dependencies
        for dep in task.dependencies:
            self.dependents[dep].add(task.name)
    
    def add_tasks(self, tasks: List[ComputeTask]) -> None:
        """Add multiple tasks."""
        for task in tasks:
            self.add_task(task)
    
    def _validate_dag(self) -> None:
        """Verify no cycles exist in the dependency graph."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)
            
            for dep in self.tasks[name].dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Task {name} depends on unknown task {dep}")
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(name)
            return False
        
        for name in self.tasks:
            if name not in visited:
                if has_cycle(name):
                    raise ValueError("Cycle detected in task dependencies")
    
    def _get_ready_tasks(self, completed: Set[str]) -> List[ComputeTask]:
        """Get tasks whose dependencies are all satisfied."""
        ready = []
        for name, task in self.tasks.items():
            if task._state == TaskState.PENDING:
                if task.dependencies.issubset(completed):
                    task._state = TaskState.READY
                    ready.append(task)
        return ready
    
    def _execute_task(self, task: ComputeTask) -> Tuple[str, bool, Any]:
        """Execute a single task with error handling."""
        task._state = TaskState.RUNNING
        task._start_time = time.perf_counter()
        
        if self.progress_callback:
            self.progress_callback(task.name, TaskState.RUNNING)
        
        attempts = task.retries + 1
        last_error = None
        
        for attempt in range(attempts):
            try:
                result = task.fn(*task.args, **task.kwargs)
                task._result = result
                task._state = TaskState.COMPLETED
                task._end_time = time.perf_counter()
                
                if self.progress_callback:
                    self.progress_callback(task.name, TaskState.COMPLETED)
                
                return (task.name, True, result)
            
            except Exception as e:
                last_error = e
                if attempt < attempts - 1:
                    logger.warning(f"Task {task.name} failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        # All retries exhausted
        task._state = TaskState.FAILED
        task._error = last_error
        task._end_time = time.perf_counter()
        
        if self.progress_callback:
            self.progress_callback(task.name, TaskState.FAILED)
        
        logger.error(f"Task {task.name} failed after {attempts} attempts: {last_error}")
        return (task.name, False, last_error)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute all tasks respecting dependencies.
        
        Returns:
            Dict mapping task names to their results
        """
        if not self.tasks:
            return {}
        
        self._validate_dag()
        
        start_time = time.perf_counter()
        completed: Set[str] = set()
        failed: Set[str] = set()
        results: Dict[str, Any] = {}
        
        self._stats = SchedulerStats(total_tasks=len(self.tasks))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[Future, ComputeTask] = {}
            pending_queue: List[ComputeTask] = []
            
            # Initial ready tasks
            ready = self._get_ready_tasks(completed)
            heapq.heapify(ready)
            pending_queue = ready
            
            while pending_queue or futures:
                # Submit ready tasks
                while pending_queue and len(futures) < self.max_workers:
                    task = heapq.heappop(pending_queue)
                    future = executor.submit(self._execute_task, task)
                    futures[future] = task
                
                # Track max parallelism
                self._stats.max_parallelism = max(
                    self._stats.max_parallelism, len(futures)
                )
                
                if not futures:
                    break
                
                # Wait for at least one to complete
                done_futures = []
                try:
                    for future in as_completed(futures, timeout=1.0):
                        done_futures.append(future)
                        break
                except TimeoutError:
                    # Still waiting, loop back and check
                    continue
                
                if not done_futures:
                    continue
                
                for future in done_futures:
                    task = futures.pop(future)
                    name, success, result = future.result()
                    
                    if success:
                        completed.add(name)
                        results[name] = result
                        self._stats.completed += 1
                        
                        # Check for newly ready tasks
                        for dependent_name in self.dependents[name]:
                            dep_task = self.tasks[dependent_name]
                            if dep_task._state == TaskState.PENDING:
                                if dep_task.dependencies.issubset(completed):
                                    dep_task._state = TaskState.READY
                                    heapq.heappush(pending_queue, dep_task)
                    else:
                        failed.add(name)
                        self._stats.failed += 1
                        
                        if self.fail_fast:
                            # Cancel pending tasks
                            for t in self.tasks.values():
                                if t._state in (TaskState.PENDING, TaskState.READY):
                                    t._state = TaskState.CANCELLED
                                    self._stats.cancelled += 1
                            pending_queue.clear()
                            
                            # Cancel running futures
                            for f in futures:
                                f.cancel()
                            futures.clear()
        
        self._stats.total_time = time.perf_counter() - start_time
        
        return results
    
    def get_stats(self) -> SchedulerStats:
        """Get execution statistics."""
        return self._stats
    
    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order (for visualization)."""
        self._validate_dag()
        
        in_degree = {name: len(task.dependencies) for name, task in self.tasks.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # Sort by priority within same level
            queue.sort(key=lambda n: self.tasks[n].priority)
            name = queue.pop(0)
            result.append(name)
            
            for dependent in self.dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result
    
    def visualize(self) -> str:
        """Return ASCII visualization of task DAG."""
        lines = ["Task DAG:"]
        order = self.get_execution_order()
        
        for name in order:
            task = self.tasks[name]
            deps = ", ".join(task.dependencies) if task.dependencies else "(none)"
            state = task._state.name
            dur = f"{task.duration:.3f}s" if task.duration else "pending"
            lines.append(f"  [{task.priority.name}] {name} <- {deps} | {state} | {dur}")
        
        return "\n".join(lines)


# Convenience function for quick parallel execution
def parallel_map(
    fn: Callable,
    items: List[Any],
    max_workers: int = 4,
    priority: Priority = Priority.NORMAL,
) -> List[Any]:
    """
    Map a function over items in parallel.
    
    Like multiprocessing.Pool.map but uses TaskScheduler.
    """
    scheduler = TaskScheduler(max_workers=max_workers)
    
    for i, item in enumerate(items):
        scheduler.add_task(ComputeTask(
            name=f"item_{i}",
            fn=fn,
            args=(item,),
            priority=priority,
        ))
    
    results = scheduler.run()
    return [results[f"item_{i}"] for i in range(len(items))]
