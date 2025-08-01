"""
Enhanced ELCS - Dynamic Emergence Networks Core Implementation
=============================================================

This module implements the core Dynamic Emergence Network (DEN) architecture,
representing the paradigm shift from hierarchical discrete layers to continuous,
interpenetrating process topologies with scale-invariant properties.

Architecture Overview:
- Process-Relational Substrate: Dynamic process patterns with temporal persistence
- Scale-Invariant Properties: Fractal coherence mechanisms across scales
- Temporal Coherence: Process pattern persistence and transformation
- Network Topology: Dynamic reconfiguration based on interaction patterns

Author: Enhanced ELCS Development Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessScale(Enum):
    """Enumeration of process scales in the Enhanced ELCS framework"""
    QUANTUM = -1
    CHEMICAL = 0
    PROTOBIOLOGICAL = 1
    CELLULAR = 2
    MULTICELLULAR = 3
    SOCIAL = 4
    MENTAL = 5
    TECHNOLOGICAL = 6


@dataclass
class ProcessSignature:
    """Characteristic process patterns for a process entity"""
    pattern_frequencies: dict[str, float] = field(default_factory=dict)
    coupling_strengths: dict[str, float] = field(default_factory=dict)
    phase_relationships: dict[str, float] = field(default_factory=dict)
    energy_flow_rates: dict[str, float] = field(default_factory=dict)

    def similarity(self, other: ProcessSignature) -> float:
        """Calculate similarity between process signatures"""
        # Implement cosine similarity across all signature components
        all_keys = set(self.pattern_frequencies.keys()) | set(other.pattern_frequencies.keys())

        if not all_keys:
            return 1.0

        dot_product = sum(
            self.pattern_frequencies.get(key, 0.0) * other.pattern_frequencies.get(key, 0.0)
            for key in all_keys
        )

        norm_self = np.sqrt(sum(val**2 for val in self.pattern_frequencies.values()))
        norm_other = np.sqrt(sum(val**2 for val in other.pattern_frequencies.values()))

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return dot_product / (norm_self * norm_other)


@dataclass
class ProcessEntity:
    """Dynamic process pattern with temporal persistence and emergent properties"""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scale: ProcessScale = ProcessScale.CHEMICAL
    process_signature: ProcessSignature = field(default_factory=ProcessSignature)
    temporal_coherence: float = 0.5  # Pattern stability over time (0-1)
    interaction_affordances: set[str] = field(default_factory=set)
    emergence_potential: float = 0.3  # Capacity for next-scale transitions (0-1)

    # Process-relational properties
    scale_invariant_properties: dict[str, float] = field(default_factory=dict)
    temporal_dynamics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Network properties
    network_position: tuple[float, float] | None = None
    local_topology_strength: float = 0.0

    def __post_init__(self):
        """Initialize derived properties after object creation"""
        if not self.interaction_affordances:
            self.interaction_affordances = self._initialize_default_affordances()

        if not self.scale_invariant_properties:
            self.scale_invariant_properties = self._initialize_scale_invariant_properties()

    def _initialize_default_affordances(self) -> set[str]:
        """Initialize default interaction affordances based on scale"""
        affordance_map = {
            ProcessScale.QUANTUM: {"entangle", "decohere", "tunnel", "superpose"},
            ProcessScale.CHEMICAL: {"bind", "catalyze", "diffuse", "react"},
            ProcessScale.PROTOBIOLOGICAL: {"replicate", "metabolize", "compartmentalize"},
            ProcessScale.CELLULAR: {"divide", "differentiate", "communicate", "respond"},
            ProcessScale.MULTICELLULAR: {"cooperate", "compete", "specialize", "coordinate"},
            ProcessScale.SOCIAL: {"communicate", "collaborate", "learn", "organize"},
            ProcessScale.MENTAL: {"perceive", "reason", "imagine", "decide"},
            ProcessScale.TECHNOLOGICAL: {"process", "store", "transmit", "compute"}
        }
        return affordance_map.get(self.scale, {"interact", "transform"})

    def _initialize_scale_invariant_properties(self) -> dict[str, float]:
        """Initialize scale-invariant properties"""
        return {
            "fractal_dimension": np.random.uniform(1.5, 2.5),
            "power_law_exponent": np.random.uniform(-3.0, -1.0),
            "clustering_coefficient": np.random.uniform(0.1, 0.9),
            "small_world_ratio": np.random.uniform(0.5, 2.0)
        }

    def update_temporal_dynamics(self, property_name: str, value: float) -> None:
        """Update temporal dynamics tracking"""
        self.temporal_dynamics[property_name].append(value)

        # Maintain sliding window of recent values
        max_history = 100
        if len(self.temporal_dynamics[property_name]) > max_history:
            self.temporal_dynamics[property_name] = self.temporal_dynamics[property_name][-max_history:]

    def calculate_emergence_readiness(self) -> float:
        """Calculate readiness for emergence to next scale"""
        # Combine multiple factors for emergence readiness
        coherence_factor = self.temporal_coherence
        potential_factor = self.emergence_potential
        topology_factor = self.local_topology_strength

        # Weighted combination
        weights = [0.4, 0.4, 0.2]
        factors = [coherence_factor, potential_factor, topology_factor]

        return sum(w * f for w, f in zip(weights, factors, strict=True))


# Import DynamicEmergenceNetwork from den_core
from .den_core import DynamicEmergenceNetwork, EmergenceEvent, ScaleBridge

# Export all public classes and functions
__all__ = [
    'ProcessScale',
    'ProcessSignature',
    'ProcessEntity',
    'DynamicEmergenceNetwork',
    'EmergenceEvent',
    'ScaleBridge'
]
