"""
Compute optimization module.

High-performance infrastructure for quantum-classical hybrid computations:
- Parallel task scheduling and execution
- Intelligent workload routing (CPU/GPU/QPU)
- Profiling and bottleneck analysis
- Memory-efficient batch processing
"""

from .scheduler import TaskScheduler, ComputeTask, Priority
from .parallel import ParallelExecutor, WorkerPool
from .router import ComputeRouter, DeviceCapability
from .profiler import Profiler, profile_function

__all__ = [
    'TaskScheduler',
    'ComputeTask', 
    'Priority',
    'ParallelExecutor',
    'WorkerPool',
    'ComputeRouter',
    'DeviceCapability',
    'Profiler',
    'profile_function',
]
