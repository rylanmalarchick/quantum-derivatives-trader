#!/usr/bin/env python3
"""
Demo: Compute Infrastructure for Quantum-Classical Hybrid Workloads

This script demonstrates the compute optimization infrastructure:
1. Task scheduling with DAG dependencies
2. Parallel quantum circuit evaluation  
3. Intelligent backend routing
4. Performance profiling

Run with: python scripts/demo_compute.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.compute.scheduler import TaskScheduler, ComputeTask, Priority, parallel_map
from src.compute.parallel import WorkerPool, ParallelExecutor
from src.compute.router import ComputeRouter, get_router, route_circuit
from src.compute.profiler import Profiler, benchmark, profile_section


def demo_task_scheduler():
    """Demonstrate DAG-based task scheduling."""
    print("\n" + "=" * 60)
    print("DEMO 1: Task Scheduler with Dependencies")
    print("=" * 60)
    
    # Simulate a typical ML pipeline
    def load_data():
        time.sleep(0.1)
        return {"data": np.random.randn(1000, 10)}
    
    def preprocess(data):
        time.sleep(0.05)
        return {"processed": data["data"] * 2}
    
    def train_model(data):
        time.sleep(0.2)
        return {"model": "trained"}
    
    def evaluate(model, data):
        time.sleep(0.1)
        return {"accuracy": 0.95}
    
    def save_results(results):
        time.sleep(0.05)
        return {"saved": True}
    
    # Create scheduler
    scheduler = TaskScheduler(max_workers=4)
    
    # Add tasks with dependencies
    scheduler.add_task(ComputeTask(
        name="load_data",
        fn=load_data,
        priority=Priority.HIGH,
    ))
    
    scheduler.add_task(ComputeTask(
        name="preprocess",
        fn=lambda: preprocess(scheduler.tasks["load_data"].result),
        dependencies={"load_data"},
        priority=Priority.NORMAL,
    ))
    
    scheduler.add_task(ComputeTask(
        name="train",
        fn=lambda: train_model(scheduler.tasks["preprocess"].result),
        dependencies={"preprocess"},
        priority=Priority.NORMAL,
        estimated_cost=5.0,  # Most expensive
    ))
    
    scheduler.add_task(ComputeTask(
        name="evaluate",
        fn=lambda: evaluate(
            scheduler.tasks["train"].result,
            scheduler.tasks["preprocess"].result
        ),
        dependencies={"train", "preprocess"},
    ))
    
    scheduler.add_task(ComputeTask(
        name="save",
        fn=lambda: save_results(scheduler.tasks["evaluate"].result),
        dependencies={"evaluate"},
        priority=Priority.LOW,
    ))
    
    # Visualize DAG
    print("\nTask DAG:")
    print(scheduler.visualize())
    
    # Execute
    print("\nExecuting...")
    start = time.perf_counter()
    results = scheduler.run()
    elapsed = time.perf_counter() - start
    
    # Stats
    stats = scheduler.get_stats()
    print(f"\nCompleted {stats.completed}/{stats.total_tasks} tasks in {elapsed:.3f}s")
    print(f"Max parallelism achieved: {stats.max_parallelism}")
    print(f"Final result: {results.get('save')}")


def demo_parallel_execution():
    """Demonstrate parallel workload execution."""
    print("\n" + "=" * 60)
    print("DEMO 2: Parallel Quantum Circuit Simulation")
    print("=" * 60)
    
    import pennylane as qml
    
    # Create a parameterized quantum circuit
    n_qubits = 4
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits):
            qml.RZ(params[n_qubits + i], wires=i)
        return qml.expval(qml.PauliZ(0))
    
    # Generate random parameters
    n_circuits = 100
    all_params = [np.random.randn(2 * n_qubits) for _ in range(n_circuits)]
    
    # Sequential execution
    print(f"\nEvaluating {n_circuits} circuits...")
    
    start = time.perf_counter()
    sequential_results = [circuit(p) for p in all_params]
    sequential_time = time.perf_counter() - start
    print(f"Sequential: {sequential_time:.3f}s ({n_circuits/sequential_time:.1f} circuits/s)")
    
    # Parallel execution
    start = time.perf_counter()
    parallel_results = parallel_map(circuit, all_params, max_workers=4)
    parallel_time = time.perf_counter() - start
    print(f"Parallel:   {parallel_time:.3f}s ({n_circuits/parallel_time:.1f} circuits/s)")
    
    speedup = sequential_time / parallel_time
    print(f"Speedup:    {speedup:.2f}x")
    
    # Verify results match
    if np.allclose(sequential_results, parallel_results):
        print("✓ Results match")
    else:
        print("✗ Results differ!")


def demo_compute_router():
    """Demonstrate intelligent backend routing."""
    print("\n" + "=" * 60)
    print("DEMO 3: Compute Router")
    print("=" * 60)
    
    router = get_router()
    
    # Show available backends
    print("\nDiscovered backends:")
    for name, backend in router.backends.items():
        caps = backend.get_capabilities()
        print(f"  {name}: {caps.device_type.name}")
        print(f"    Max qubits: {caps.max_qubits}")
        print(f"    Cores: {caps.num_cores}")
        print(f"    Memory: {caps.memory_gb:.1f} GB")
        print(f"    FLOPS: {caps.flops_estimate:.2e}")
    
    # Route different workloads
    print("\nRouting decisions:")
    workloads = [
        (4, 10, "Small circuit"),
        (10, 50, "Medium circuit"),
        (16, 100, "Large circuit"),
        (20, 200, "Very large circuit"),
    ]
    
    for n_qubits, depth, desc in workloads:
        try:
            decision = router.route(n_qubits, depth)
            caps = decision.backend.get_capabilities()
            print(f"  {desc} ({n_qubits}q, depth={depth}):")
            print(f"    → {caps.device_id}")
            print(f"    Estimated time: {decision.estimated_cost:.4f}s")
        except RuntimeError as e:
            print(f"  {desc} ({n_qubits}q, depth={depth}): {e}")


def demo_profiler():
    """Demonstrate performance profiling."""
    print("\n" + "=" * 60)
    print("DEMO 4: Performance Profiler")
    print("=" * 60)
    
    profiler = Profiler()
    
    def matrix_ops(n: int):
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        return np.linalg.svd(A @ B)
    
    def quantum_sim():
        import pennylane as qml
        dev = qml.device("lightning.qubit", wires=8)
        
        @qml.qnode(dev)
        def circuit():
            for i in range(8):
                qml.Hadamard(wires=i)
            for i in range(7):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(8)]
        
        return circuit()
    
    # Profile a simulated training loop
    print("\nProfiling simulated training loop...")
    
    for epoch in range(3):
        with profiler.section(f"epoch_{epoch}"):
            with profiler.section("forward"):
                quantum_sim()
            with profiler.section("classical"):
                matrix_ops(500)
            with profiler.section("backward"):
                time.sleep(0.1)
    
    # Print report
    print(profiler.report())
    
    # Benchmark a specific function
    print("\nBenchmarking quantum circuit:")
    stats = benchmark(quantum_sim, n_runs=5, warmup=1)
    print(f"  Mean: {stats['mean']*1000:.2f}ms ± {stats['std']*1000:.2f}ms")
    print(f"  Min:  {stats['min']*1000:.2f}ms")
    print(f"  Max:  {stats['max']*1000:.2f}ms")


def main():
    """Run all demos."""
    print("╔" + "═" * 58 + "╗")
    print("║" + " Quantum-Classical Compute Infrastructure Demo ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    demos = [
        ("Task Scheduler", demo_task_scheduler),
        ("Parallel Execution", demo_parallel_execution),
        ("Compute Router", demo_compute_router),
        ("Profiler", demo_profiler),
    ]
    
    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\n⚠ Demo '{name}' failed: {e}")
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
