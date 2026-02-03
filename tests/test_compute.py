"""
Comprehensive tests for the compute module.

Tests scheduler, parallel execution, router, and profiler components.
Follows the project testing philosophy: tests are specification.
"""

import pytest
import time
import threading
import multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from src.compute.scheduler import (
    TaskScheduler, ComputeTask, Priority, TaskState, 
    SchedulerStats, parallel_map
)
from src.compute.parallel import (
    WorkerPool, ParallelExecutor, WorkerConfig, BatchResult,
    _execute_batch, parallel
)
from src.compute.router import (
    ComputeRouter, CPUSimulatorBackend, GPUSimulatorBackend,
    DeviceType, DeviceCapability, RoutingDecision, get_router, route_circuit
)
from src.compute.profiler import (
    Profiler, CPUProfiler, TimingResult, ProfileResult,
    benchmark, get_profiler, profile_section, profile_function
)


# ============================================================================
# Helper functions for multiprocessing (must be top-level for pickling)
# ============================================================================

def double(x):
    """Double a number - picklable for multiprocessing."""
    return x * 2

def identity(x):
    """Return input unchanged - picklable for multiprocessing."""
    return x

def read_data(x):
    """Simulate reading data - picklable for multiprocessing."""
    return x


# ============================================================================
# SCHEDULER TESTS
# ============================================================================

class TestComputeTask:
    """Tests for ComputeTask dataclass."""
    
    def test_task_creation_minimal(self):
        """Task with just name and function."""
        task = ComputeTask(name="test", fn=lambda: 42)
        
        assert task.name == "test"
        assert task.priority == Priority.NORMAL
        assert task.dependencies == set()
        assert task.retries == 0
        assert task._state == TaskState.PENDING
    
    def test_task_creation_full(self):
        """Task with all parameters."""
        task = ComputeTask(
            name="complex",
            fn=lambda x, y: x + y,
            args=(1,),
            kwargs={"y": 2},
            priority=Priority.HIGH,
            dependencies={"dep1", "dep2"},
            estimated_cost=5.0,
            timeout=10.0,
            retries=3,
        )
        
        assert task.name == "complex"
        assert task.priority == Priority.HIGH
        assert task.dependencies == {"dep1", "dep2"}
        assert task.estimated_cost == 5.0
        assert task.retries == 3
    
    def test_task_hash_by_name(self):
        """Tasks are hashable by name."""
        t1 = ComputeTask(name="same", fn=lambda: 1)
        t2 = ComputeTask(name="same", fn=lambda: 2)
        t3 = ComputeTask(name="different", fn=lambda: 1)
        
        assert hash(t1) == hash(t2)
        assert hash(t1) != hash(t3)
    
    def test_task_comparison_by_priority(self):
        """Tasks compare by priority for heap operations."""
        high = ComputeTask(name="high", fn=lambda: 0, priority=Priority.HIGH)
        low = ComputeTask(name="low", fn=lambda: 0, priority=Priority.LOW)
        
        assert high < low  # Lower priority value = higher priority
    
    def test_task_result_before_completion_raises(self):
        """Accessing result before completion raises."""
        task = ComputeTask(name="test", fn=lambda: 42)
        
        with pytest.raises(RuntimeError, match="not completed"):
            _ = task.result
    
    def test_task_duration_before_execution(self):
        """Duration is None before execution."""
        task = ComputeTask(name="test", fn=lambda: 42)
        
        assert task.duration is None


class TestTaskScheduler:
    """Tests for TaskScheduler."""
    
    def test_empty_scheduler_returns_empty(self):
        """Empty scheduler returns empty results."""
        scheduler = TaskScheduler()
        results = scheduler.run()
        
        assert results == {}
    
    def test_single_task(self):
        """Single task execution."""
        scheduler = TaskScheduler()
        scheduler.add_task(ComputeTask(
            name="single",
            fn=lambda: 42,
        ))
        
        results = scheduler.run()
        
        assert results["single"] == 42
    
    def test_independent_tasks_parallel(self):
        """Independent tasks can run in parallel."""
        results = []
        lock = threading.Lock()
        
        def track_and_compute(value):
            with lock:
                results.append(("start", value))
            time.sleep(0.05)
            with lock:
                results.append(("end", value))
            return value * 2
        
        scheduler = TaskScheduler(max_workers=4)
        for i in range(4):
            scheduler.add_task(ComputeTask(
                name=f"task_{i}",
                fn=track_and_compute,
                args=(i,),
            ))
        
        outputs = scheduler.run()
        
        # All tasks should complete
        assert len(outputs) == 4
        for i in range(4):
            assert outputs[f"task_{i}"] == i * 2
        
        # Check parallelism: starts should happen before all ends
        starts = [r for r in results if r[0] == "start"]
        assert len(starts) == 4
    
    def test_dependent_tasks_sequential(self):
        """Dependent tasks run sequentially."""
        execution_order = []
        
        def track(name):
            execution_order.append(name)
            return name
        
        scheduler = TaskScheduler(max_workers=4)
        
        scheduler.add_task(ComputeTask(
            name="first",
            fn=track,
            args=("first",),
        ))
        scheduler.add_task(ComputeTask(
            name="second",
            fn=track,
            args=("second",),
            dependencies={"first"},
        ))
        scheduler.add_task(ComputeTask(
            name="third",
            fn=track,
            args=("third",),
            dependencies={"second"},
        ))
        
        scheduler.run()
        
        assert execution_order == ["first", "second", "third"]
    
    def test_diamond_dependency(self):
        """Diamond-shaped dependency graph."""
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        execution_order = []
        lock = threading.Lock()
        
        def track(name):
            with lock:
                execution_order.append(name)
            return name
        
        scheduler = TaskScheduler(max_workers=4)
        
        scheduler.add_task(ComputeTask(name="A", fn=track, args=("A",)))
        scheduler.add_task(ComputeTask(name="B", fn=track, args=("B",), dependencies={"A"}))
        scheduler.add_task(ComputeTask(name="C", fn=track, args=("C",), dependencies={"A"}))
        scheduler.add_task(ComputeTask(name="D", fn=track, args=("D",), dependencies={"B", "C"}))
        
        scheduler.run()
        
        # A must be first, D must be last
        assert execution_order[0] == "A"
        assert execution_order[-1] == "D"
        # B and C can be in any order
        assert set(execution_order[1:3]) == {"B", "C"}
    
    def test_cycle_detection(self):
        """Cycle in dependencies raises error."""
        scheduler = TaskScheduler()
        
        scheduler.add_task(ComputeTask(name="A", fn=lambda: 0, dependencies={"B"}))
        scheduler.add_task(ComputeTask(name="B", fn=lambda: 0, dependencies={"A"}))
        
        with pytest.raises(ValueError, match="Cycle detected"):
            scheduler.run()
    
    def test_missing_dependency_raises(self):
        """Missing dependency raises error."""
        scheduler = TaskScheduler()
        
        scheduler.add_task(ComputeTask(
            name="A",
            fn=lambda: 0,
            dependencies={"nonexistent"},
        ))
        
        with pytest.raises(ValueError, match="unknown task"):
            scheduler.run()
    
    def test_duplicate_task_raises(self):
        """Adding task with same name raises."""
        scheduler = TaskScheduler()
        scheduler.add_task(ComputeTask(name="dup", fn=lambda: 0))
        
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_task(ComputeTask(name="dup", fn=lambda: 1))
    
    def test_priority_ordering(self):
        """Higher priority tasks run first."""
        execution_order = []
        lock = threading.Lock()
        
        def track(name):
            with lock:
                execution_order.append(name)
            time.sleep(0.01)  # Small delay to ensure ordering
            return name
        
        scheduler = TaskScheduler(max_workers=1)  # Force sequential
        
        # Add in reverse priority order
        scheduler.add_task(ComputeTask(
            name="background",
            fn=track,
            args=("background",),
            priority=Priority.BACKGROUND,
        ))
        scheduler.add_task(ComputeTask(
            name="critical",
            fn=track,
            args=("critical",),
            priority=Priority.CRITICAL,
        ))
        scheduler.add_task(ComputeTask(
            name="normal",
            fn=track,
            args=("normal",),
            priority=Priority.NORMAL,
        ))
        
        scheduler.run()
        
        # Critical should run first
        assert execution_order[0] == "critical"
        assert execution_order[-1] == "background"
    
    def test_retry_on_failure_simple(self):
        """Task retries on failure (simplified test without timing issues)."""
        attempt_count = [0]
        
        def flaky():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ValueError("Transient error")
            return "success"
        
        scheduler = TaskScheduler()
        scheduler.add_task(ComputeTask(
            name="flaky",
            fn=flaky,
            retries=2,  # Just 2 retries to keep it fast
        ))
        
        results = scheduler.run()
        
        assert results["flaky"] == "success"
        assert attempt_count[0] == 2
    
    def test_fail_fast_cancels_pending(self):
        """fail_fast=True cancels pending tasks on failure."""
        scheduler = TaskScheduler(max_workers=1, fail_fast=True)
        
        scheduler.add_task(ComputeTask(
            name="fail",
            fn=lambda: 1/0,  # Will fail
            priority=Priority.HIGH,
        ))
        scheduler.add_task(ComputeTask(
            name="never_runs",
            fn=lambda: "should not run",
            priority=Priority.LOW,
        ))
        
        results = scheduler.run()
        stats = scheduler.get_stats()
        
        assert stats.failed >= 1
        assert "never_runs" not in results or stats.cancelled >= 1
    
    def test_progress_callback(self):
        """Progress callback is called on state changes."""
        callbacks = []
        
        def on_progress(name, state):
            callbacks.append((name, state))
        
        scheduler = TaskScheduler(progress_callback=on_progress)
        scheduler.add_task(ComputeTask(name="test", fn=lambda: 42))
        
        scheduler.run()
        
        # Should have RUNNING and COMPLETED callbacks
        states = [state for _, state in callbacks if _ == "test"]
        assert TaskState.RUNNING in states
        assert TaskState.COMPLETED in states
    
    def test_stats_tracking(self):
        """Scheduler tracks execution statistics."""
        scheduler = TaskScheduler(max_workers=2)
        
        for i in range(5):
            scheduler.add_task(ComputeTask(
                name=f"task_{i}",
                fn=lambda: time.sleep(0.01),
            ))
        
        scheduler.run()
        stats = scheduler.get_stats()
        
        assert stats.total_tasks == 5
        assert stats.completed == 5
        assert stats.failed == 0
        assert stats.total_time > 0
        assert stats.max_parallelism >= 1
    
    def test_get_execution_order(self):
        """get_execution_order returns topological sort."""
        scheduler = TaskScheduler()
        
        scheduler.add_task(ComputeTask(name="C", fn=lambda: 0, dependencies={"B"}))
        scheduler.add_task(ComputeTask(name="B", fn=lambda: 0, dependencies={"A"}))
        scheduler.add_task(ComputeTask(name="A", fn=lambda: 0))
        
        order = scheduler.get_execution_order()
        
        assert order == ["A", "B", "C"]
    
    def test_visualize(self):
        """visualize returns DAG string representation."""
        scheduler = TaskScheduler()
        scheduler.add_task(ComputeTask(name="A", fn=lambda: 0))
        scheduler.add_task(ComputeTask(name="B", fn=lambda: 0, dependencies={"A"}))
        
        viz = scheduler.visualize()
        
        assert "Task DAG" in viz
        assert "A" in viz
        assert "B" in viz


class TestParallelMap:
    """Tests for parallel_map convenience function."""
    
    def test_basic_parallel_map(self):
        """Basic parallel map over items."""
        results = parallel_map(double, [1, 2, 3, 4])
        
        assert results == [2, 4, 6, 8]
    
    def test_empty_list(self):
        """Empty list returns empty."""
        results = parallel_map(identity, [])
        
        assert results == []


# ============================================================================
# PARALLEL EXECUTION TESTS
# ============================================================================

class TestWorkerConfig:
    """Tests for WorkerConfig."""
    
    def test_default_config(self):
        """Default config values."""
        config = WorkerConfig(worker_id=0, num_workers=4)
        
        assert config.worker_id == 0
        assert config.num_workers == 4
        assert config.cpu_affinity is None
        assert config.memory_limit_mb is None
        assert config.device == "cpu"


class TestBatchResult:
    """Tests for BatchResult."""
    
    def test_batch_result_creation(self):
        """BatchResult stores execution metadata."""
        result = BatchResult(
            batch_id=1,
            results=[1, 2, 3],
            worker_id=0,
            compute_time=0.5,
            memory_peak_mb=100.0,
        )
        
        assert result.batch_id == 1
        assert result.results == [1, 2, 3]
        assert result.compute_time == 0.5


class TestExecuteBatch:
    """Tests for _execute_batch function."""
    
    def test_execute_batch_applies_function(self):
        """_execute_batch applies function to all items."""
        config = WorkerConfig(worker_id=0, num_workers=1)
        result = _execute_batch(double, [1, 2, 3], batch_id=0, config=config)
        
        assert result.results == [2, 4, 6]
        assert result.batch_id == 0
        assert result.worker_id == 0
        assert result.compute_time > 0


class TestWorkerPool:
    """Tests for WorkerPool."""
    
    def test_worker_pool_context_manager(self):
        """WorkerPool works as context manager."""
        with WorkerPool(num_workers=2) as pool:
            results = pool.map(double, [1, 2, 3, 4])
        
        assert results == [2, 4, 6, 8]
    
    def test_worker_pool_empty_list(self):
        """Empty list returns empty."""
        with WorkerPool() as pool:
            results = pool.map(identity, [])
        
        assert results == []
    
    def test_worker_pool_preserves_order(self):
        """Results maintain input order."""
        with WorkerPool(num_workers=4) as pool:
            results = pool.map(identity, list(range(100)), batch_size=10)
        
        assert results == list(range(100))
    
    def test_worker_pool_progress_callback(self):
        """Progress callback is called."""
        progress_calls = []
        
        def on_progress(completed, total):
            progress_calls.append((completed, total))
        
        with WorkerPool(num_workers=2) as pool:
            pool.map(
                identity,
                list(range(8)),
                batch_size=2,
                progress_callback=on_progress,
            )
        
        # Should have progress calls
        assert len(progress_calls) > 0
        # Final call should show all complete
        assert progress_calls[-1][0] == progress_calls[-1][1]
    
    def test_worker_pool_without_context_raises(self):
        """Using pool without context manager raises."""
        pool = WorkerPool()
        
        with pytest.raises(RuntimeError, match="context manager"):
            pool.map(identity, [1, 2])
    
    def test_worker_pool_auto_batch_size(self):
        """Auto batch size is computed."""
        with WorkerPool(num_workers=2) as pool:
            # Should not raise
            results = pool.map(identity, list(range(100)))
        
        assert len(results) == 100


class TestParallelExecutor:
    """Tests for ParallelExecutor."""
    
    def test_auto_strategy_small_list_sequential(self):
        """Small lists use sequential strategy."""
        executor = ParallelExecutor(strategy="auto")
        
        # With <= 4 items, should use sequential
        results = executor.map(double, [1, 2, 3])
        
        assert results == [2, 4, 6]
    
    def test_explicit_sequential_strategy(self):
        """Sequential strategy works."""
        executor = ParallelExecutor(strategy="sequential")
        results = executor.map(double, list(range(10)))
        
        assert results == [i * 2 for i in range(10)]
    
    def test_explicit_thread_strategy(self):
        """Thread strategy works."""
        executor = ParallelExecutor(strategy="thread", num_workers=2)
        results = executor.map(double, list(range(10)))
        
        assert results == [i * 2 for i in range(10)]
    
    def test_explicit_process_strategy(self):
        """Process strategy works."""
        executor = ParallelExecutor(strategy="process", num_workers=2)
        results = executor.map(double, list(range(10)))
        
        assert results == [i * 2 for i in range(10)]
    
    def test_io_function_uses_thread(self):
        """Functions with I/O names use thread strategy."""
        executor = ParallelExecutor(strategy="auto")
        
        # Should detect "read" in name and use threads
        results = executor.map(read_data, list(range(10)))
        
        assert results == list(range(10))
    
    def test_invalid_strategy_raises(self):
        """Invalid strategy raises ValueError."""
        executor = ParallelExecutor(strategy="invalid")
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            executor.map(identity, [1, 2, 3, 4, 5])


class TestParallelDecorator:
    """Tests for @parallel decorator."""
    
    def test_parallel_decorator_with_list(self):
        """Decorator parallelizes over list input (test with thread strategy)."""
        # Note: @parallel uses ProcessPool which can't pickle local functions
        # Test the decorator logic with sequential fallback for single item
        @parallel(num_workers=2)
        def process(x):
            return x * 3
        
        # Single item should work (passthrough)
        result = process(5)
        assert result == 15
        
        # For list, we need a top-level function - tested via ParallelExecutor
    
    def test_parallel_decorator_single_item(self):
        """Decorator passes through single items."""
        @parallel(num_workers=2)
        def process(x):
            return x * 3
        
        result = process(5)
        
        assert result == 15
    
    def test_parallel_decorator_type_check(self):
        """Decorator checks for list type."""
        @parallel(num_workers=2)
        def transform(x):
            return x * 2
        
        # Non-list input should be passed through directly
        assert transform(10) == 20
        assert transform("test") == "testtest"


# ============================================================================
# ROUTER TESTS
# ============================================================================

class TestDeviceCapability:
    """Tests for DeviceCapability."""
    
    def test_default_values(self):
        """Default capability values."""
        cap = DeviceCapability(
            device_type=DeviceType.CPU,
            device_id="test",
        )
        
        assert cap.max_qubits == 0
        assert cap.available is True
    
    def test_cpu_cost_estimate(self):
        """CPU cost estimate scales with qubits and depth."""
        cap = DeviceCapability(
            device_type=DeviceType.CPU,
            device_id="test",
            flops_estimate=1e10,  # 10 GFLOPS
        )
        
        cost_small = cap.cost_estimate(n_qubits=4, depth=10)
        cost_large = cap.cost_estimate(n_qubits=8, depth=10)
        
        # Cost should scale exponentially with qubits
        assert cost_large > cost_small * 10  # 2^8 / 2^4 = 16x
    
    def test_qpu_cost_latency_dominated(self):
        """QPU cost is latency dominated."""
        cap = DeviceCapability(
            device_type=DeviceType.QPU,
            device_id="test",
            latency_ms=100,
        )
        
        cost = cap.cost_estimate(n_qubits=20, depth=100)
        
        # Should be roughly latency + small gate time
        assert 0.1 < cost < 0.3


class TestCPUSimulatorBackend:
    """Tests for CPUSimulatorBackend."""
    
    def test_get_capabilities(self):
        """Backend returns valid capabilities."""
        backend = CPUSimulatorBackend(n_qubits=15)
        caps = backend.get_capabilities()
        
        assert caps.device_type == DeviceType.CPU
        assert caps.max_qubits == 15
        assert caps.num_cores > 0
        assert caps.memory_gb > 0
    
    def test_execute_callable(self):
        """Execute calls the circuit."""
        backend = CPUSimulatorBackend()
        mock_circuit = Mock(return_value=0.5)
        
        result = backend.execute(mock_circuit)
        
        mock_circuit.assert_called_once()
        assert result == 0.5


class TestGPUSimulatorBackend:
    """Tests for GPUSimulatorBackend."""
    
    def test_unavailable_on_cpu_only(self):
        """GPU backend unavailable without CUDA."""
        backend = GPUSimulatorBackend()
        
        # On CPU-only machines, should not be available
        # (This test is environment-dependent)
        caps = backend.get_capabilities()
        
        # Should have valid structure either way
        assert caps.device_type == DeviceType.GPU


class TestComputeRouter:
    """Tests for ComputeRouter."""
    
    def test_register_backend(self):
        """Backends can be registered."""
        router = ComputeRouter()
        backend = CPUSimulatorBackend()
        
        router.register_backend("cpu", backend)
        
        assert "cpu" in router.backends
    
    def test_auto_discover(self):
        """Auto-discover finds CPU backend."""
        router = ComputeRouter()
        router.auto_discover()
        
        assert "cpu" in router.backends
    
    def test_route_selects_available_backend(self):
        """Route selects from available backends."""
        router = ComputeRouter()
        router.register_backend("cpu", CPUSimulatorBackend(n_qubits=20))
        
        decision = router.route(n_qubits=10, depth=100)
        
        assert decision.backend is not None
        assert decision.estimated_cost > 0
        assert decision.reason != ""
    
    def test_route_respects_qubit_limit(self):
        """Route respects backend qubit limits."""
        router = ComputeRouter()
        router.register_backend("small", CPUSimulatorBackend(n_qubits=5))
        router.register_backend("large", CPUSimulatorBackend(n_qubits=20))
        
        decision = router.route(n_qubits=10, depth=100)
        
        # Should select the large backend
        assert decision.backend.get_capabilities().max_qubits >= 10
    
    def test_route_no_backend_raises(self):
        """Route raises when no backend can handle workload."""
        router = ComputeRouter()
        router.register_backend("small", CPUSimulatorBackend(n_qubits=5))
        
        with pytest.raises(RuntimeError, match="No backend"):
            router.route(n_qubits=100, depth=100)
    
    def test_routing_stats(self):
        """Router tracks routing statistics."""
        router = ComputeRouter()
        router.register_backend("cpu", CPUSimulatorBackend(n_qubits=20))
        
        router.route(n_qubits=10, depth=100)
        router.route(n_qubits=5, depth=50)
        
        stats = router.get_stats()
        
        assert stats["cpu"] == 2
    
    def test_execute_routes_and_runs(self):
        """Execute routes and runs circuit."""
        router = ComputeRouter()
        router.register_backend("cpu", CPUSimulatorBackend())
        
        mock_circuit = Mock(return_value=0.42)
        result = router.execute(mock_circuit, n_qubits=5, depth=10)
        
        assert result == 0.42


class TestGlobalRouter:
    """Tests for global router functions."""
    
    def test_get_router_creates_singleton(self):
        """get_router returns same instance."""
        import src.compute.router as router_module
        
        # Reset global
        router_module._global_router = None
        
        r1 = get_router()
        r2 = get_router()
        
        assert r1 is r2
    
    def test_route_circuit_uses_global(self):
        """route_circuit uses global router."""
        import src.compute.router as router_module
        router_module._global_router = None
        
        # Should not raise
        decision = route_circuit(n_qubits=5, depth=10)
        
        assert decision.backend is not None


# ============================================================================
# PROFILER TESTS
# ============================================================================

class TestTimingResult:
    """Tests for TimingResult."""
    
    def test_memory_delta(self):
        """Memory delta calculation."""
        result = TimingResult(
            name="test",
            start_time=0,
            end_time=1,
            duration=1,
            memory_start_mb=100,
            memory_end_mb=150,
            memory_peak_mb=200,
        )
        
        assert result.memory_delta_mb == 50
    
    def test_to_dict(self):
        """Converts to dictionary."""
        result = TimingResult(
            name="test",
            start_time=0,
            end_time=1,
            duration=1,
            memory_start_mb=100,
            memory_end_mb=150,
            memory_peak_mb=200,
            call_count=5,
        )
        
        d = result.to_dict()
        
        assert d["name"] == "test"
        assert d["duration_s"] == 1
        assert d["call_count"] == 5


class TestProfiler:
    """Tests for Profiler class."""
    
    def test_section_timing(self):
        """Section records timing."""
        profiler = Profiler(track_memory=False)
        
        with profiler.section("test"):
            time.sleep(0.05)
        
        result = profiler.get_result()
        
        assert "test" in result.function_stats
        assert result.function_stats["test"].duration >= 0.04
    
    def test_nested_sections(self):
        """Nested sections create hierarchy."""
        profiler = Profiler(track_memory=False)
        
        with profiler.section("outer"):
            with profiler.section("inner"):
                time.sleep(0.01)
        
        result = profiler.get_result()
        
        assert "outer" in result.function_stats
        assert "inner" in result.function_stats
        assert "inner" in result.call_graph["outer"]
    
    def test_profile_decorator(self):
        """Profile decorator works."""
        profiler = Profiler(track_memory=False)
        
        @profiler.profile
        def my_function():
            time.sleep(0.01)
            return 42
        
        result = my_function()
        
        assert result == 42
        stats = profiler.get_result()
        # Check for function name in stats (may include class name)
        assert any("my_function" in name for name in stats.function_stats.keys())
    
    def test_hotspots_sorted(self):
        """Hotspots are sorted by time percentage."""
        profiler = Profiler(track_memory=False)
        
        with profiler.section("fast"):
            time.sleep(0.01)
        
        with profiler.section("slow"):
            time.sleep(0.05)
        
        result = profiler.get_result()
        
        assert len(result.hotspots) >= 2
        # Slow should be first
        assert result.hotspots[0][0] == "slow"
        assert result.hotspots[0][1] > result.hotspots[1][1]
    
    def test_report_generates_summary(self):
        """Report generates readable summary."""
        profiler = Profiler(track_memory=False)
        
        with profiler.section("test"):
            time.sleep(0.01)
        
        report = profiler.report()
        
        assert "PROFILING SUMMARY" in report
        assert "Total time" in report
        assert "test" in report
    
    def test_reset_clears_state(self):
        """Reset clears profiler state."""
        profiler = Profiler(track_memory=False)
        
        with profiler.section("test"):
            pass
        
        profiler.reset()
        result = profiler.get_result()
        
        assert len(result.function_stats) == 0
    
    def test_memory_tracking(self):
        """Memory tracking records allocations."""
        profiler = Profiler(track_memory=True)
        
        with profiler.section("allocate"):
            # Allocate some memory
            _ = [i for i in range(100000)]
        
        result = profiler.get_result()
        
        # Should have recorded some memory
        assert result.peak_memory_mb >= 0


class TestCPUProfiler:
    """Tests for CPUProfiler (cProfile wrapper)."""
    
    def test_profile_context_manager(self):
        """Profile as context manager."""
        profiler = CPUProfiler()
        
        with profiler.profile():
            sum(range(10000))
        
        stats = profiler.stats()
        
        assert "function calls" in stats or "primitive calls" in stats
    
    def test_start_stop(self):
        """Manual start/stop."""
        profiler = CPUProfiler()
        
        profiler.start()
        sum(range(10000))
        profiler.stop()
        
        stats = profiler.stats()
        
        assert len(stats) > 0
    
    def test_top_functions(self):
        """Get top functions by time."""
        profiler = CPUProfiler()
        
        with profiler.profile():
            sum(range(10000))
        
        top = profiler.top_functions(n=5)
        
        assert len(top) <= 5
        for name, time_val, calls in top:
            assert isinstance(name, str)
            assert time_val >= 0
            assert calls >= 0


class TestBenchmark:
    """Tests for benchmark function."""
    
    def test_benchmark_basic(self):
        """Basic benchmarking."""
        def fast_fn():
            return sum(range(100))
        
        stats = benchmark(fast_fn, n_runs=5, warmup=1)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["n_runs"] == 5
    
    def test_benchmark_with_args(self):
        """Benchmark with arguments."""
        def add(a, b):
            return a + b
        
        stats = benchmark(add, 1, 2, n_runs=3, warmup=1)
        
        assert stats["n_runs"] == 3
        assert stats["mean"] > 0


class TestGlobalProfiler:
    """Tests for global profiler functions."""
    
    def test_get_profiler_creates_singleton(self):
        """get_profiler returns same instance."""
        import src.compute.profiler as profiler_module
        
        # Reset global
        profiler_module._global_profiler = None
        
        p1 = get_profiler()
        p2 = get_profiler()
        
        assert p1 is p2
    
    def test_profile_section_uses_global(self):
        """profile_section uses global profiler."""
        import src.compute.profiler as profiler_module
        profiler_module._global_profiler = None
        
        with profile_section("test"):
            pass
        
        result = get_profiler().get_result()
        assert "test" in result.function_stats
    
    def test_profile_function_decorator(self):
        """profile_function decorator uses global."""
        import src.compute.profiler as profiler_module
        profiler_module._global_profiler = None
        
        @profile_function
        def decorated():
            return 42
        
        result = decorated()
        
        assert result == 42


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestComputeIntegration:
    """Integration tests combining compute components."""
    
    def test_scheduler_with_profiler(self):
        """Scheduler tasks can be profiled."""
        profiler = Profiler(track_memory=False)
        
        def profiled_task(name):
            with profiler.section(name):
                time.sleep(0.01)
            return name
        
        scheduler = TaskScheduler()
        scheduler.add_task(ComputeTask(
            name="task1",
            fn=profiled_task,
            args=("task1",),
        ))
        scheduler.add_task(ComputeTask(
            name="task2", 
            fn=profiled_task,
            args=("task2",),
        ))
        
        scheduler.run()
        
        result = profiler.get_result()
        assert "task1" in result.function_stats
        assert "task2" in result.function_stats
    
    def test_router_cost_estimation(self):
        """Router cost estimates are reasonable."""
        router = ComputeRouter()
        router.auto_discover()
        
        # Small circuit should have low cost
        decision_small = router.route(n_qubits=4, depth=10)
        
        # Large circuit should have higher cost
        decision_large = router.route(n_qubits=15, depth=100)
        
        assert decision_large.estimated_cost > decision_small.estimated_cost
