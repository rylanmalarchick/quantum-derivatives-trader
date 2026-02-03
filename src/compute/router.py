"""
Compute Router: Intelligent workload routing across heterogeneous devices.

Routes computation to optimal backend based on:
1. Workload characteristics (qubit count, circuit depth)
2. Available hardware (CPU cores, GPU memory, QPU access)
3. Current load and queue depth
4. Cost/latency tradeoffs

This is essentially a JIT compiler's backend selector for quantum workloads.
"""

from __future__ import annotations

import os
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of compute devices."""
    CPU = auto()
    GPU = auto()        # CUDA/ROCm
    QPU = auto()        # Real quantum hardware
    SIMULATOR = auto()  # Quantum simulator (statevector, MPS, etc.)
    TPU = auto()        # Tensor processing unit


@dataclass
class DeviceCapability:
    """Capabilities and current state of a compute device."""
    device_type: DeviceType
    device_id: str
    
    # Static capabilities
    max_qubits: int = 0
    max_circuit_depth: int = 0
    supports_mid_circuit_measurement: bool = False
    native_gates: List[str] = field(default_factory=list)
    
    # For classical devices
    num_cores: int = 0
    memory_gb: float = 0.0
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0
    
    # Dynamic state
    current_load: float = 0.0  # 0-1
    queue_depth: int = 0
    available: bool = True
    
    # Performance model
    flops_estimate: float = 0.0  # Estimated FLOPS
    latency_ms: float = 0.0      # Round-trip latency
    
    def cost_estimate(self, n_qubits: int, depth: int) -> float:
        """Estimate cost (time) for a circuit execution."""
        if self.device_type in (DeviceType.CPU, DeviceType.GPU):
            # Simulation cost: O(2^n * depth)
            return (2 ** n_qubits) * depth / max(self.flops_estimate, 1e9)
        elif self.device_type == DeviceType.QPU:
            # Real hardware: mostly latency dominated
            return self.latency_ms / 1000 + depth * 0.001  # ~1us per gate
        else:
            return float('inf')


class Backend(ABC):
    """Abstract backend for executing quantum circuits."""
    
    @abstractmethod
    def execute(self, circuit: Any, shots: int = 1000) -> Any:
        """Execute a circuit."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> DeviceCapability:
        """Get device capabilities."""
        pass


class CPUSimulatorBackend(Backend):
    """PennyLane lightning.qubit backend."""
    
    def __init__(self, n_qubits: int = 20):
        self.n_qubits = n_qubits
        self._caps = None
    
    def execute(self, circuit: Any, shots: int = 1000) -> Any:
        # Circuit is already a PennyLane QNode, just call it
        return circuit()
    
    def get_capabilities(self) -> DeviceCapability:
        if self._caps is None:
            import multiprocessing as mp
            import psutil
            
            self._caps = DeviceCapability(
                device_type=DeviceType.CPU,
                device_id="lightning.qubit",
                max_qubits=self.n_qubits,
                max_circuit_depth=10000,
                num_cores=mp.cpu_count(),
                memory_gb=psutil.virtual_memory().total / 1e9,
                flops_estimate=mp.cpu_count() * 10e9,  # ~10 GFLOPS per core
                latency_ms=0.1,
            )
        return self._caps


class GPUSimulatorBackend(Backend):
    """PennyLane lightning.gpu backend (if available)."""
    
    def __init__(self):
        self._available = False
        self._caps = None
        
        try:
            import pennylane as qml
            import torch
            if torch.cuda.is_available():
                self._available = True
        except ImportError:
            pass
    
    def execute(self, circuit: Any, shots: int = 1000) -> Any:
        if not self._available:
            raise RuntimeError("GPU backend not available")
        return circuit()
    
    def get_capabilities(self) -> DeviceCapability:
        if self._caps is None:
            import torch
            
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self._caps = DeviceCapability(
                    device_type=DeviceType.GPU,
                    device_id=f"cuda:0 ({props.name})",
                    max_qubits=28,  # GPU can handle more qubits
                    max_circuit_depth=10000,
                    has_gpu=True,
                    gpu_memory_gb=props.total_memory / 1e9,
                    flops_estimate=props.multi_processor_count * 128 * 1.5e9,
                    latency_ms=1.0,
                )
            else:
                self._caps = DeviceCapability(
                    device_type=DeviceType.GPU,
                    device_id="unavailable",
                    available=False,
                )
        return self._caps


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    backend: Backend
    reason: str
    estimated_cost: float
    alternatives: List[Tuple[Backend, float]] = field(default_factory=list)


class ComputeRouter:
    """
    Routes quantum workloads to optimal backend.
    
    Example:
        router = ComputeRouter()
        router.register_backend("cpu", CPUSimulatorBackend())
        router.register_backend("gpu", GPUSimulatorBackend())
        
        # Get optimal backend for workload
        decision = router.route(n_qubits=10, depth=100)
        result = decision.backend.execute(circuit)
    
    Routing logic:
    - Small circuits (n <= 12): CPU (low overhead)
    - Medium circuits (12 < n <= 20): GPU if available
    - Large circuits (n > 20): GPU required or error
    - Very deep circuits: Consider noise model
    """
    
    def __init__(self):
        self.backends: Dict[str, Backend] = {}
        self._lock = threading.Lock()
        self._routing_stats: Dict[str, int] = {}
    
    def register_backend(self, name: str, backend: Backend) -> None:
        """Register a backend."""
        self.backends[name] = backend
        self._routing_stats[name] = 0
    
    def auto_discover(self) -> None:
        """Auto-discover available backends."""
        # Always have CPU
        self.register_backend("cpu", CPUSimulatorBackend())
        
        # Check for GPU
        try:
            gpu = GPUSimulatorBackend()
            if gpu.get_capabilities().available:
                self.register_backend("gpu", gpu)
                logger.info("GPU backend discovered")
        except Exception as e:
            logger.debug(f"GPU backend not available: {e}")
    
    def route(
        self,
        n_qubits: int,
        depth: int,
        shots: int = 1,
        prefer_speed: bool = True,
    ) -> RoutingDecision:
        """
        Select optimal backend for workload.
        
        Args:
            n_qubits: Number of qubits
            depth: Circuit depth
            shots: Number of measurement shots
            prefer_speed: Prefer speed over accuracy
        
        Returns:
            RoutingDecision with selected backend
        """
        candidates: List[Tuple[str, Backend, float]] = []
        
        for name, backend in self.backends.items():
            caps = backend.get_capabilities()
            
            if not caps.available:
                continue
            
            # Check if backend can handle the workload
            if n_qubits > caps.max_qubits:
                continue
            if depth > caps.max_circuit_depth:
                continue
            
            # Estimate cost
            cost = caps.cost_estimate(n_qubits, depth) * shots
            
            # Adjust for current load
            cost *= (1 + caps.current_load)
            
            candidates.append((name, backend, cost))
        
        if not candidates:
            raise RuntimeError(
                f"No backend can handle {n_qubits} qubits, depth {depth}"
            )
        
        # Sort by cost
        candidates.sort(key=lambda x: x[2])
        
        best_name, best_backend, best_cost = candidates[0]
        
        with self._lock:
            self._routing_stats[best_name] += 1
        
        return RoutingDecision(
            backend=best_backend,
            reason=f"Lowest cost: {best_cost:.4f}s",
            estimated_cost=best_cost,
            alternatives=[(b, c) for _, b, c in candidates[1:]],
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        return dict(self._routing_stats)
    
    def execute(
        self,
        circuit: Any,
        n_qubits: int,
        depth: int,
        shots: int = 1,
    ) -> Any:
        """Route and execute a circuit."""
        decision = self.route(n_qubits, depth, shots)
        return decision.backend.execute(circuit, shots)


# Global router instance
_global_router: Optional[ComputeRouter] = None


def get_router() -> ComputeRouter:
    """Get or create global router."""
    global _global_router
    if _global_router is None:
        _global_router = ComputeRouter()
        _global_router.auto_discover()
    return _global_router


def route_circuit(n_qubits: int, depth: int, shots: int = 1) -> RoutingDecision:
    """Route a circuit using global router."""
    return get_router().route(n_qubits, depth, shots)
