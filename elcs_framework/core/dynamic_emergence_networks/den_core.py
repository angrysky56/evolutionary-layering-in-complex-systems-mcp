"""
Dynamic Emergence Network - Core Implementation
==============================================

This module contains the main DynamicEmergenceNetwork class that implements
the core architecture for process-relational substrate with scale-invariant
properties and temporal coherence mechanisms.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from . import ProcessEntity, ProcessScale

logger = logging.getLogger(__name__)


@dataclass
class EmergenceEvent:
    """Represents an emergence event detected in the network"""
    timestamp: float
    source_entities: list[str]
    emergent_entity: str
    emergence_type: str
    strength: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleBridge:
    """Represents connection between different scales"""
    lower_entity: str
    higher_entity: str
    emergence_strength: float
    causation_type: str  # "upward", "downward", "circular"
    stability: float = 0.5


class DynamicEmergenceNetwork:
    """
    Core implementation of Dynamic Emergence Networks (DENs)

    Implements continuous, interpenetrating process topologies with
    scale-invariant properties, replacing traditional hierarchical layers.
    """

    def __init__(self,
                 enable_temporal_tracking: bool = True,
                 emergence_threshold: float = 0.8,
                 max_entities: int = 10000):
        """
        Initialize Dynamic Emergence Network

        Args:
            enable_temporal_tracking: Track temporal dynamics
            emergence_threshold: Threshold for emergence detection
            max_entities: Maximum number of entities in network
        """
        # Core network structure
        self.network = nx.MultiDiGraph()  # Directed multi-graph for complex relationships
        self.process_entities: dict[str, ProcessEntity] = {}

        # Scale bridge management
        self.scale_bridges: dict[str, ScaleBridge] = {}
        self.scale_adjacency: dict[ProcessScale, set[ProcessScale]] = defaultdict(set)

        # Temporal state tracking
        self.temporal_state: dict[str, float] = {}
        self.enable_temporal_tracking = enable_temporal_tracking
        self.temporal_history: list[dict[str, Any]] = []

        # Emergence detection
        self.emergence_threshold = emergence_threshold
        self.emergence_events: list[EmergenceEvent] = []
        self.baseline_metrics: dict[str, float] = {}

        # Network constraints
        self.max_entities = max_entities

        # Performance metrics
        self.update_count = 0
        self.last_optimization = time.time()

        logger.info(f"Initialized DynamicEmergenceNetwork with {max_entities} max entities")

    def add_process_entity(self, entity: ProcessEntity) -> bool:
        """
        Add process entity to the network with dynamic topology integration

        Args:
            entity: ProcessEntity to add to network

        Returns:
            bool: Success status of addition
        """
        if len(self.process_entities) >= self.max_entities:
            logger.warning(f"Cannot add entity: maximum capacity ({self.max_entities}) reached")
            return False

        # Add node to network graph
        self.network.add_node(
            entity.entity_id,
            scale=entity.scale,
            process_signature=entity.process_signature,
            temporal_coherence=entity.temporal_coherence,
            emergence_potential=entity.emergence_potential,
            affordances=entity.interaction_affordances,
            scale_invariant_props=entity.scale_invariant_properties
        )

        # Store entity reference
        self.process_entities[entity.entity_id] = entity

        # Update scale adjacency
        self._update_scale_adjacency(entity.scale)

        # Connect to similar entities
        self._connect_to_similar_entities(entity)

        # Update network metrics
        self._update_network_metrics(entity.entity_id)

        logger.debug(f"Added process entity {entity.entity_id} at scale {entity.scale}")
        return True

    def create_scale_bridge(self,
                          lower_entity_id: str,
                          higher_entity_id: str,
                          emergence_strength: float,
                          causation_type: str = "upward") -> str:
        """
        Create inter-scale emergent relationship

        Args:
            lower_entity_id: ID of lower-scale entity
            higher_entity_id: ID of higher-scale entity
            emergence_strength: Strength of emergence relationship (0-1)
            causation_type: Type of causation ("upward", "downward", "circular")

        Returns:
            str: Bridge ID for reference
        """
        if lower_entity_id not in self.process_entities:
            raise ValueError(f"Lower entity {lower_entity_id} not found")
        if higher_entity_id not in self.process_entities:
            raise ValueError(f"Higher entity {higher_entity_id} not found")

        lower_entity = self.process_entities[lower_entity_id]
        higher_entity = self.process_entities[higher_entity_id]

        # Validate scale relationship
        if causation_type == "upward" and lower_entity.scale.value >= higher_entity.scale.value:
            raise ValueError("Upward causation requires lower scale -> higher scale")

        # Create scale bridge
        bridge = ScaleBridge(
            lower_entity=lower_entity_id,
            higher_entity=higher_entity_id,
            emergence_strength=emergence_strength,
            causation_type=causation_type,
            stability=min(lower_entity.temporal_coherence, higher_entity.temporal_coherence)
        )

        bridge_id = f"bridge_{len(self.scale_bridges)}"
        self.scale_bridges[bridge_id] = bridge

        # Add edge to network
        edge_attrs = {
            "type": "scale_bridge",
            "bridge_id": bridge_id,
            "strength": emergence_strength,
            "causation": causation_type
        }

        if causation_type in ["upward", "circular"]:
            self.network.add_edge(lower_entity_id, higher_entity_id, **edge_attrs)
        if causation_type in ["downward", "circular"]:
            self.network.add_edge(higher_entity_id, lower_entity_id, **edge_attrs)

        logger.debug(f"Created {causation_type} scale bridge: {lower_entity_id} -> {higher_entity_id}")
        return bridge_id

    def update_network_dynamics(self, timestep: int) -> None:
        """
        Update network dynamics for one timestep

        Args:
            timestep: Current simulation timestep
        """
        start_time = time.time()

        # Update entity temporal dynamics
        for entity_id, entity in self.process_entities.items():
            self._update_entity_dynamics(entity, timestep)

        # Update relational topology
        self._update_relational_topology(timestep)

        # Propagate scale bridge influences
        self._propagate_scale_influences(timestep)

        # Detect emergence events
        emergence_events = self._detect_emergence_events(timestep)
        self.emergence_events.extend(emergence_events)

        # Store temporal state if tracking enabled
        if self.enable_temporal_tracking:
            self._store_temporal_state(timestep)

        # Periodic optimization
        if time.time() - self.last_optimization > 10.0:  # Every 10 seconds
            self._optimize_network()
            self.last_optimization = time.time()

        self.update_count += 1

        update_time = time.time() - start_time
        if update_time > 0.1:  # Log if update takes >100ms
            logger.warning(f"Slow network update: {update_time:.3f}s for timestep {timestep}")

    def _update_scale_adjacency(self, scale: ProcessScale) -> None:
        """Update scale adjacency relationships"""
        # Adjacent scales can form bridges
        adjacent_scales = []
        current_value = scale.value

        for other_scale in ProcessScale:
            if abs(other_scale.value - current_value) <= 1:
                adjacent_scales.append(other_scale)

        for adj_scale in adjacent_scales:
            self.scale_adjacency[scale].add(adj_scale)
            self.scale_adjacency[adj_scale].add(scale)

    def _connect_to_similar_entities(self, entity: ProcessEntity) -> None:
        """Connect entity to similar entities based on process signatures"""
        similarity_threshold = 0.7

        for other_id, other_entity in self.process_entities.items():
            if other_id == entity.entity_id:
                continue

            # Calculate similarity
            similarity = entity.process_signature.similarity(other_entity.process_signature)

            if similarity > similarity_threshold:
                # Create connection with similarity-based weight
                self.network.add_edge(
                    entity.entity_id,
                    other_id,
                    type="similarity",
                    weight=similarity,
                    bidirectional=True
                )

    def _update_network_metrics(self, entity_id: str) -> None:
        """Update network metrics for entity"""
        if entity_id not in self.network:
            return

        entity = self.process_entities[entity_id]

        # Calculate local clustering coefficient
        neighbors = list(self.network.neighbors(entity_id))
        if len(neighbors) < 2:
            clustering = 0.0
        else:
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2

            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if self.network.has_edge(n1, n2):
                        triangles += 1

            clustering = triangles / possible_triangles if possible_triangles > 0 else 0.0

        # Update entity's scale-invariant properties
        entity.scale_invariant_properties["clustering_coefficient"] = clustering
        entity.local_topology_strength = clustering

        # Update network node attributes
        self.network.nodes[entity_id]["clustering"] = clustering

    def _update_entity_dynamics(self, entity: ProcessEntity, timestep: int) -> None:
        """Update individual entity dynamics"""
        # Simulate temporal coherence decay and renewal
        decay_rate = 0.01
        renewal_strength = 0.02

        # Natural decay
        entity.temporal_coherence *= (1 - decay_rate)

        # Renewal from network interactions
        neighbors = list(self.network.neighbors(entity.entity_id))
        if neighbors:
            avg_neighbor_coherence = np.mean([
                self.process_entities[n].temporal_coherence
                for n in neighbors if n in self.process_entities
            ])
            # cast numpy scalar to float
            entity.temporal_coherence += float(renewal_strength * avg_neighbor_coherence)

        # Clamp to [0, 1]
        entity.temporal_coherence = float(np.clip(entity.temporal_coherence, 0.0, 1.0))

        # Update emergence potential based on network position
        network = self.network
        # Use degree property with indexing to avoid type checker issues
        degree = len(list(network.neighbors(entity.entity_id)))
        max_possible_degree = max(len(network.nodes()) - 1, 1)
        normalized_degree = degree / max_possible_degree
        entity.emergence_potential = 0.3 * entity.temporal_coherence + 0.7 * normalized_degree
        # cast numpy scalar to float
        entity.emergence_potential = float(np.clip(entity.emergence_potential, 0.0, 1.0))

        # Track temporal dynamics
        entity.update_temporal_dynamics("coherence", entity.temporal_coherence)
        entity.update_temporal_dynamics("emergence_potential", entity.emergence_potential)

    def _update_relational_topology(self, timestep: int) -> None:
        """Update network topology based on interaction patterns"""
        # Strengthen connections between frequently interacting entities
        edge_updates = []

        for edge in self.network.edges(data=True):
            source, target, attrs = edge

            if attrs.get("type") == "similarity":
                current_weight = attrs.get("weight", 0.5)

                # Strengthen based on continued similarity
                if source in self.process_entities and target in self.process_entities:
                    similarity = self.process_entities[source].process_signature.similarity(
                        self.process_entities[target].process_signature
                    )

                    # Exponential moving average update
                    alpha = 0.1
                    new_weight = alpha * similarity + (1 - alpha) * current_weight
                    edge_updates.append((source, target, new_weight))

        # Apply edge weight updates
        for source, target, new_weight in edge_updates:
            self.network[source][target][0]["weight"] = new_weight

    def _propagate_scale_influences(self, timestep: int) -> None:
        """Propagate influences through scale bridges"""
        for bridge_id, bridge in self.scale_bridges.items():
            lower_entity = self.process_entities[bridge.lower_entity]
            higher_entity = self.process_entities[bridge.higher_entity]

            influence_strength = bridge.emergence_strength * bridge.stability

            if bridge.causation_type in ["upward", "circular"]:
                # Lower scale influences higher scale
                influence = lower_entity.emergence_potential * influence_strength
                higher_entity.temporal_coherence += 0.01 * influence
                higher_entity.temporal_coherence = min(higher_entity.temporal_coherence, 1.0)

            if bridge.causation_type in ["downward", "circular"]:
                # Higher scale constrains lower scale
                constraint = higher_entity.temporal_coherence * influence_strength
                lower_entity.emergence_potential *= (1 - 0.01 * constraint)
                lower_entity.emergence_potential = max(lower_entity.emergence_potential, 0.0)

    def _detect_emergence_events(self, timestep: int) -> list[EmergenceEvent]:
        """Detect emergence events in the network"""
        events = []

        # Check for entities ready for emergence
        for entity_id, entity in self.process_entities.items():
            emergence_readiness = entity.calculate_emergence_readiness()

            if emergence_readiness > self.emergence_threshold:
                # Look for entities to merge/transform with
                neighbors = list(self.network.neighbors(entity_id))
                ready_neighbors = [
                    n for n in neighbors
                    if n in self.process_entities and
                    self.process_entities[n].calculate_emergence_readiness() > self.emergence_threshold
                ]

                if ready_neighbors:
                    # Create emergence event
                    event = EmergenceEvent(
                        timestamp=time.time(),
                        source_entities=[entity_id] + ready_neighbors[:3],  # Limit to 4 total
                        emergent_entity=f"emergent_{timestep}_{len(events)}",
                        emergence_type="collective_emergence",
                        strength=emergence_readiness,
                        properties={
                            "source_scales": [entity.scale for entity in
                                            [self.process_entities[eid] for eid in [entity_id] + ready_neighbors[:3]]],
                            "avg_coherence": np.mean([
                                self.process_entities[eid].temporal_coherence
                                for eid in [entity_id] + ready_neighbors[:3]
                            ])
                        }
                    )
                    events.append(event)

        return events

    def _store_temporal_state(self, timestep: int) -> None:
        """Store current network state for temporal tracking"""
        state = {
            "timestep": timestep,
            "timestamp": time.time(),
            "num_entities": len(self.process_entities),
            "num_bridges": len(self.scale_bridges),
            "avg_coherence": np.mean([e.temporal_coherence for e in self.process_entities.values()]),
            "avg_emergence_potential": np.mean([e.emergence_potential for e in self.process_entities.values()]),
            "network_density": nx.density(self.network) if self.network.number_of_nodes() > 1 else 0.0
        }

        self.temporal_history.append(state)

        # Maintain sliding window
        max_history = 1000
        if len(self.temporal_history) > max_history:
            self.temporal_history = self.temporal_history[-max_history:]

    def _optimize_network(self) -> None:
        """Optimize network structure for performance"""
        # Remove weak connections
        weak_edges = []
        # Initialize isolated to avoid unbound variable in logger
        isolated = []
        for edge in self.network.edges(data=True):
            source, target, attrs = edge
            if attrs.get("type") == "similarity" and attrs.get("weight", 0) < 0.3:
                weak_edges.append((source, target))

        for source, target in weak_edges:
            self.network.remove_edge(source, target)

        # Remove isolated nodes (except if they're the only node)
        if len(self.process_entities) > 1:
            isolated = list(nx.isolates(self.network))
            for node_id in isolated:
                if node_id in self.process_entities:
                    del self.process_entities[node_id]
                self.network.remove_node(node_id)

        logger.debug(f"Network optimization: removed {len(weak_edges)} weak edges, {len(isolated)} isolated nodes")

    def get_network_statistics(self) -> dict[str, Any]:
        """Get comprehensive network statistics"""
        if not self.process_entities:
            return {"error": "No entities in network"}

        stats = {
            "entities": {
                "total": len(self.process_entities),
                "by_scale": {scale.name: sum(1 for e in self.process_entities.values() if e.scale == scale)
                           for scale in ProcessScale}
            },
            "network": {
                "nodes": self.network.number_of_nodes(),
                "edges": self.network.number_of_edges(),
                "density": nx.density(self.network) if self.network.number_of_nodes() > 1 else 0.0,
                "connected_components": nx.number_weakly_connected_components(self.network)
            },
            "bridges": {
                "total": len(self.scale_bridges),
                "by_type": {
                    causation_type: sum(1 for b in self.scale_bridges.values()
                                      if b.causation_type == causation_type)
                    for causation_type in ["upward", "downward", "circular"]
                }
            },
            "dynamics": {
                "avg_coherence": np.mean([e.temporal_coherence for e in self.process_entities.values()]),
                "avg_emergence_potential": np.mean([e.emergence_potential for e in self.process_entities.values()]),
                "total_emergence_events": len(self.emergence_events)
            },
            "performance": {
                "update_count": self.update_count,
                "entities_per_update": len(self.process_entities) / max(self.update_count, 1)
            }
        }

        return stats

    def visualize_network(self,
                         layout: str = "spring",
                         save_path: str | None = None) -> Any:
        """
        Create network visualization (returns matplotlib figure)

        Args:
            layout: Layout algorithm ("spring", "circular", "hierarchical")
            save_path: Optional path to save visualization

        Returns:
            matplotlib figure object
        """
        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib required for visualization")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        if not self.process_entities:
            ax.text(0.5, 0.5, "No entities to visualize", ha='center', va='center')
            return fig

        # Create layout
        if layout == "spring":
            pos = nx.spring_layout(self.network)
        elif layout == "circular":
            pos = nx.circular_layout(self.network)
        else:  # hierarchical
            pos = nx.shell_layout(self.network)

        # Color nodes by scale
        scale_colors = {
            ProcessScale.QUANTUM: '#FF6B6B',
            ProcessScale.CHEMICAL: '#4ECDC4',
            ProcessScale.PROTOBIOLOGICAL: '#45B7D1',
            ProcessScale.CELLULAR: '#96CEB4',
            ProcessScale.MULTICELLULAR: '#FFEAA7',
            ProcessScale.SOCIAL: '#DDA0DD',
            ProcessScale.MENTAL: '#FFB347',
            ProcessScale.TECHNOLOGICAL: '#98D8C8'
        }

        node_colors = [scale_colors.get(self.process_entities[node].scale, '#CCCCCC')
                      for node in self.network.nodes()]

        # Draw network
        nx.draw(self.network, pos, ax=ax,
                node_color=node_colors,
                node_size=300,
                with_labels=False,
                edge_color='gray',
                alpha=0.7)

        # Add legend
        legend_elements = [patches.Patch(color=color, label=scale.name)
                          for scale, color in scale_colors.items()
                          if any(e.scale == scale for e in self.process_entities.values())]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title("Dynamic Emergence Network Visualization")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network visualization saved to {save_path}")

        return fig
