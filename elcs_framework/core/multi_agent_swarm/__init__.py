"""
Enhanced ELCS - Multi-Agent Swarm Intelligence Implementation
============================================================

This module implements collective intelligence mechanisms based on 2024-2025 research
showing 90.2% performance improvement over single agents through emergent specialization,
distributed cognition, and self-modifying architectures.

Key Features:
- Emergent Specialization: Agents dynamically assume roles based on local interactions
- Distributed Cognition: Cognitive processes distributed across agent networks
- Self-Modifying Architectures: Agents modify their own and others' architectures
- Collective Decision Making: Swarm-based problem solving with emergence detection

Based on:
- Anthropic's multi-agent research system (2024)
- "Emergent collective intelligence from massive-agent cooperation" (Chen et al., 2023)
- Current LLM-based multi-agent systems with unprecedented flexibility

Author: Enhanced ELCS Development Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import networkx as nx
import numpy as np

from ..dynamic_emergence_networks import DynamicEmergenceNetwork, ProcessEntity, ProcessScale, ProcessSignature

logger = logging.getLogger(__name__)


class SpecializationRole(Enum):
    """Emergent specialization roles for agents"""
    EXPLORER = "explorer"          # Explore new areas/possibilities
    EXPLOITER = "exploiter"        # Optimize known solutions
    COORDINATOR = "coordinator"    # Coordinate group activities
    SPECIALIST = "specialist"      # Deep expertise in specific domain
    GENERALIST = "generalist"      # Broad capabilities across domains
    INNOVATOR = "innovator"        # Generate novel solutions
    VALIDATOR = "validator"        # Validate and verify solutions
    COMMUNICATOR = "communicator"  # Information sharing and translation


@dataclass
class AgentCapabilities:
    """Represents an agent's current capabilities and performance metrics"""
    processing_power: float = 0.5
    memory_capacity: float = 0.5
    learning_rate: float = 0.1
    communication_efficiency: float = 0.5
    specialization_focus: dict[str, float] = field(default_factory=dict)
    performance_history: list[float] = field(default_factory=list)

    def update_performance(self, performance_score: float) -> None:
        """Update performance history with sliding window"""
        self.performance_history.append(performance_score)
        if len(self.performance_history) > 100:  # Keep last 100 scores
            self.performance_history = self.performance_history[-100:]

    def get_avg_performance(self) -> float:
        """Get average performance over recent history"""
        if not self.performance_history:
            return 0.5
        return float(np.mean(self.performance_history[-20:]))  # Last 20 scores


@runtime_checkable
class CollectiveIntelligenceAgent(Protocol):
    """Protocol defining interface for agents participating in collective intelligence"""

    agent_id: str
    capabilities: AgentCapabilities
    current_role: SpecializationRole | None
    local_memory: dict[str, Any]

    def perceive_environment(self, environment_state: dict[str, Any]) -> dict[str, float]:
        """Gather information from environment"""
        ...

    def communicate_with_peers(self, peer_agents: Sequence[CollectiveIntelligenceAgent]) -> dict[str, Any]:
        """Exchange information with other agents"""
        ...

    def make_local_decision(self,
                           local_info: dict[str, float],
                           peer_info: dict[str, Any]) -> dict[str, Any]:
        """Make decision based on local and peer information"""
        ...

    def execute_action(self, decision: dict[str, Any]) -> dict[str, float]:
        """Execute decided action and return results"""
        ...

    def modify_self_architecture(self, performance_feedback: float) -> None:
        """Self-modify based on performance feedback"""
        ...

    def get_specialization_preference(self) -> dict[SpecializationRole, float]:
        """Return preference scores for each specialization role"""
        ...


class SwarmAgent(CollectiveIntelligenceAgent):
    """
    Implementation of CollectiveIntelligenceAgent for swarm intelligence

    Features emergent specialization, distributed cognition, and self-modification
    """

    def __init__(self,
                 agent_id: str | None = None,
                 initial_capabilities: AgentCapabilities | None = None,
                 den: DynamicEmergenceNetwork | None = None):
        """
        Initialize SwarmAgent

        Args:
            agent_id: Unique identifier for agent
            initial_capabilities: Starting capabilities
            den: Dynamic Emergence Network for integration
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.capabilities = initial_capabilities or AgentCapabilities()
        self.current_role: SpecializationRole | None = None
        self.local_memory: dict[str, Any] = {}
        self.den = den

        # Interaction tracking
        self.interaction_history: list[dict[str, Any]] = []
        self.peer_relationships: dict[str, float] = {}  # agent_id -> trust/collaboration score

        # Specialization tracking
        self.role_experience: dict[SpecializationRole, float] = dict.fromkeys(SpecializationRole, 0.0)
        self.role_performance: dict[SpecializationRole, list[float]] = {role: [] for role in SpecializationRole}

        # Self-modification tracking
        self.modification_history: list[dict[str, Any]] = []
        self.learning_trajectory: list[float] = []

        logger.debug(f"Initialized SwarmAgent {self.agent_id}")

    def perceive_environment(self, environment_state: dict[str, Any]) -> dict[str, float]:
        """Gather information from environment with role-specific filtering"""
        perception_filters = {
            SpecializationRole.EXPLORER: ["novelty", "uncertainty", "unexplored_areas"],
            SpecializationRole.EXPLOITER: ["optimization_opportunities", "efficiency_metrics"],
            SpecializationRole.COORDINATOR: ["group_dynamics", "task_distribution", "bottlenecks"],
            SpecializationRole.SPECIALIST: ["domain_specific_signals", "technical_details"],
            SpecializationRole.GENERALIST: ["broad_patterns", "cross_domain_connections"],
            SpecializationRole.INNOVATOR: ["creative_potential", "combination_opportunities"],
            SpecializationRole.VALIDATOR: ["quality_indicators", "consistency_metrics"],
            SpecializationRole.COMMUNICATOR: ["information_gaps", "communication_needs"]
        }

        # Base perception (all agents see basic environment)
        perceived = {
            "complexity": environment_state.get("complexity", 0.5),
            "resources": environment_state.get("resources", 0.5),
            "threats": environment_state.get("threats", 0.0),
            "opportunities": environment_state.get("opportunities", 0.5)
        }

        # Role-specific enhanced perception
        if self.current_role:
            role_filters = perception_filters.get(self.current_role, [])
            for filter_key in role_filters:
                if filter_key in environment_state:
                    perceived[filter_key] = environment_state[filter_key]

        # Apply capability-based perception enhancement
        perception_strength = self.capabilities.processing_power
        for key, value in perceived.items():
            # Add noise reduction based on processing power
            noise = np.random.normal(0, 0.1 * (1 - perception_strength))
            perceived[key] = np.clip(value + noise, 0.0, 1.0)

        return perceived

    def communicate_with_peers(self, peer_agents: Sequence[CollectiveIntelligenceAgent]) -> dict[str, Any]:
        """Exchange information with other agents using trust-based filtering"""
        shared_info = {}
        received_info = {}

        for peer in peer_agents:
            if peer.agent_id == self.agent_id:
                continue

            # Calculate trust level with this peer
            trust_level = self.peer_relationships.get(peer.agent_id, 0.5)

            # Share information based on trust and role compatibility
            info_to_share = self._select_information_to_share(peer, trust_level)
            if info_to_share:
                shared_info[peer.agent_id] = info_to_share

            # Receive information from peer
            peer_info = peer.communicate_with_peers([self])
            if self.agent_id in peer_info:
                received_info[peer.agent_id] = peer_info[self.agent_id]
                # Update trust based on information quality
                self._update_peer_trust(peer.agent_id, peer_info[self.agent_id])

        # Record interaction
        interaction_record = {
            "timestamp": time.time(),
            "peers_contacted": list(shared_info.keys()),
            "info_exchanged": len(shared_info),
            "trust_updates": len(received_info)
        }
        self.interaction_history.append(interaction_record)

        return received_info

    def make_local_decision(self,
                           local_info: dict[str, float],
                           peer_info: dict[str, Any]) -> dict[str, Any]:
        """Make decision based on local and peer information"""
        decision_context = {
            "local_assessment": local_info,
            "peer_insights": peer_info,
            "agent_state": {
                "role": self.current_role.value if self.current_role else "unassigned",
                "performance": self.capabilities.get_avg_performance(),
                "memory_utilization": len(self.local_memory) / max(self.capabilities.memory_capacity * 100, 1)
            }
        }

        # Role-specific decision making
        if self.current_role == SpecializationRole.EXPLORER:
            decision = self._make_exploration_decision(decision_context)
        elif self.current_role == SpecializationRole.EXPLOITER:
            decision = self._make_exploitation_decision(decision_context)
        elif self.current_role == SpecializationRole.COORDINATOR:
            decision = self._make_coordination_decision(decision_context)
        else:
            decision = self._make_general_decision(decision_context)

        # Add decision confidence and reasoning
        decision["confidence"] = self._calculate_decision_confidence(decision_context)
        decision["reasoning"] = self._generate_decision_reasoning(decision_context, decision)

        return decision

    def execute_action(self, decision: dict[str, Any]) -> dict[str, float]:
        """Execute decided action and return performance results"""
        action_type = decision.get("action_type", "observe")
        action_parameters = decision.get("parameters", {})

        # Simulate action execution with role-specific performance
        base_performance = 0.5

        # Role-based performance modifiers
        role_multipliers = {
            SpecializationRole.EXPLORER: 1.2 if action_type == "explore" else 0.8,
            SpecializationRole.EXPLOITER: 1.3 if action_type == "optimize" else 0.9,
            SpecializationRole.COORDINATOR: 1.4 if action_type == "coordinate" else 0.8,
            SpecializationRole.SPECIALIST: 1.5 if action_type == "specialize" else 0.7,
            SpecializationRole.INNOVATOR: 1.3 if action_type == "innovate" else 0.9,
        }

        role_multiplier = role_multipliers.get(self.current_role, 1.0) if self.current_role else 1.0
        capability_multiplier = (self.capabilities.processing_power +
                               self.capabilities.memory_capacity) / 2

        performance_score = base_performance * role_multiplier * capability_multiplier
        performance_score = np.clip(performance_score + np.random.normal(0, 0.1), 0.0, 1.0)

        # Update performance history
        self.capabilities.update_performance(performance_score)

        # Store action in memory
        action_record = {
            "timestamp": time.time(),
            "action_type": action_type,
            "parameters": action_parameters,
            "performance": performance_score,
            "role": self.current_role
        }

        memory_key = f"action_{len(self.local_memory)}"
        if len(self.local_memory) < self.capabilities.memory_capacity * 100:
            self.local_memory[memory_key] = action_record

        return {
            "performance": performance_score,
            "efficiency": performance_score * capability_multiplier,
            "role_fit": role_multiplier,
            "action_success": performance_score > 0.6
        }

    def modify_self_architecture(self, performance_feedback: float) -> None:
        """Self-modify based on performance feedback"""
        modification_threshold = 0.1  # Minimum performance change to trigger modification

        current_performance = self.capabilities.get_avg_performance()
        performance_delta = performance_feedback - current_performance

        if abs(performance_delta) > modification_threshold:
            modifications = {}

            if performance_delta > 0:  # Positive feedback - enhance successful patterns
                if self.current_role:
                    # Strengthen specialization in current role
                    role_experience = self.role_experience.get(self.current_role, 0.0)
                    self.role_experience[self.current_role] = min(role_experience + 0.1, 1.0)
                    modifications["role_specialization"] = 0.1

                # Enhance successful capabilities
                if performance_feedback > 0.8:
                    self.capabilities.processing_power = min(self.capabilities.processing_power + 0.05, 1.0)
                    modifications["processing_enhancement"] = 0.05

            else:  # Negative feedback - adapt architecture
                # Consider role change if consistently poor performance
                if len(self.capabilities.performance_history) >= 10:
                    recent_avg = np.mean(self.capabilities.performance_history[-10:])
                    if recent_avg < 0.4:
                        # Suggest role reassignment
                        modifications["role_change_suggested"] = True

                # Diversify capabilities
                self.capabilities.learning_rate = min(self.capabilities.learning_rate + 0.02, 0.5)
                modifications["learning_rate_increase"] = 0.02

            # Record modification
            modification_record = {
                "timestamp": time.time(),
                "performance_feedback": performance_feedback,
                "performance_delta": performance_delta,
                "modifications": modifications
            }
            self.modification_history.append(modification_record)

            logger.debug(f"Agent {self.agent_id} self-modified: {modifications}")

    def get_specialization_preference(self) -> dict[SpecializationRole, float]:
        """Return preference scores for each specialization role"""
        preferences = {}

        for role in SpecializationRole:
            # Base preference from experience
            experience_score = self.role_experience.get(role, 0.0)

            # Performance history in this role
            role_performances = self.role_performance.get(role, [])
            performance_score = np.mean(role_performances) if role_performances else 0.5

            # Capability alignment
            capability_alignment = self._calculate_role_capability_alignment(role)

            # Combined preference
            preference = (0.4 * experience_score +
                         0.4 * performance_score +
                         0.2 * capability_alignment)

            preferences[role] = preference

        return preferences

    def _select_information_to_share(self, peer: CollectiveIntelligenceAgent, trust_level: float) -> dict[str, Any]:
        """Select information to share with peer based on trust and relevance"""
        shareable_info = {}

        # Share based on trust level
        if trust_level > 0.7:  # High trust - share detailed information
            recent_actions = [record for record in self.local_memory.values()
                            if isinstance(record, dict) and
                            record.get("timestamp", 0) > time.time() - 300]  # Last 5 minutes
            shareable_info["recent_actions"] = recent_actions[-3:]  # Last 3 actions

        if trust_level > 0.5:  # Medium trust - share performance insights
            shareable_info["avg_performance"] = self.capabilities.get_avg_performance()
            shareable_info["current_role"] = self.current_role.value if self.current_role else None

        if trust_level > 0.3:  # Low trust - share basic status
            shareable_info["online_status"] = True
            shareable_info["capabilities_summary"] = {
                "processing": self.capabilities.processing_power > 0.7,
                "memory": self.capabilities.memory_capacity > 0.7
            }

        return shareable_info

    def _update_peer_trust(self, peer_id: str, peer_info: dict[str, Any]) -> None:
        """Update trust level with peer based on information quality"""
        current_trust = self.peer_relationships.get(peer_id, 0.5)

        # Evaluate information quality
        info_quality = 0.5
        if "recent_actions" in peer_info:
            info_quality += 0.2  # Detailed information increases trust
        if "avg_performance" in peer_info:
            info_quality += 0.1
        if "capabilities_summary" in peer_info:
            info_quality += 0.1

        # Update trust with exponential moving average
        alpha = 0.1
        new_trust = alpha * info_quality + (1 - alpha) * current_trust
        self.peer_relationships[peer_id] = np.clip(new_trust, 0.0, 1.0)

    def _make_exploration_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Make exploration-focused decision"""
        local_info = context["local_assessment"]

        # Prioritize high uncertainty and novelty
        exploration_score = local_info.get("uncertainty", 0.5) + local_info.get("novelty", 0.5)

        if exploration_score > 1.0:
            return {
                "action_type": "explore",
                "parameters": {
                    "target": "high_uncertainty_area",
                    "intensity": min(exploration_score, 1.0)
                }
            }
        else:
            return {
                "action_type": "observe",
                "parameters": {"mode": "wide_scan"}
            }

    def _make_exploitation_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Make exploitation-focused decision"""
        local_info = context["local_assessment"]

        # Look for optimization opportunities
        if local_info.get("efficiency_metrics", 0.5) < 0.8:
            return {
                "action_type": "optimize",
                "parameters": {
                    "target": "efficiency_improvement",
                    "method": "incremental"
                }
            }
        else:
            return {
                "action_type": "maintain",
                "parameters": {"current_performance": True}
            }

    def _make_coordination_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Make coordination-focused decision"""
        peer_info = context["peer_insights"]

        # Analyze group needs
        if len(peer_info) > 0:
            return {
                "action_type": "coordinate",
                "parameters": {
                    "coordination_type": "task_distribution",
                    "peer_count": len(peer_info)
                }
            }
        else:
            return {
                "action_type": "seek_peers",
                "parameters": {"broadcast_capability": True}
            }

    def _make_general_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """Make general-purpose decision"""
        local_info = context["local_assessment"]

        # Simple decision based on local information
        if local_info.get("opportunities", 0.5) > 0.7:
            return {
                "action_type": "engage",
                "parameters": {"opportunity_focus": True}
            }
        elif local_info.get("threats", 0.0) > 0.5:
            return {
                "action_type": "defensive",
                "parameters": {"threat_mitigation": True}
            }
        else:
            return {
                "action_type": "observe",
                "parameters": {"mode": "general_awareness"}
            }

    def _calculate_decision_confidence(self, context: dict[str, Any]) -> float:
        """Calculate confidence in decision based on available information"""
        local_info_quality = len(context["local_assessment"]) / 10.0  # Normalize by expected info
        peer_info_quality = len(context["peer_insights"]) / 5.0      # Normalize by expected peers

        role_confidence = 0.8 if self.current_role else 0.4
        performance_confidence = self.capabilities.get_avg_performance()

        total_confidence = np.mean([local_info_quality, peer_info_quality,
                                  role_confidence, performance_confidence])
        return float(np.clip(total_confidence, 0.0, 1.0))

    def _generate_decision_reasoning(self, context: dict[str, Any], decision: dict[str, Any]) -> str:
        """Generate human-readable reasoning for decision"""
        action_type = decision.get("action_type", "unknown")
        role_name = self.current_role.value if self.current_role else "unassigned"

        return f"Agent {self.agent_id} ({role_name}) decided to {action_type} based on local assessment and peer input"

    def _calculate_role_capability_alignment(self, role: SpecializationRole) -> float:
        """Calculate how well current capabilities align with role requirements"""
        role_requirements = {
            SpecializationRole.EXPLORER: {"processing_power": 0.7, "memory_capacity": 0.5},
            SpecializationRole.EXPLOITER: {"processing_power": 0.8, "memory_capacity": 0.6},
            SpecializationRole.COORDINATOR: {"communication_efficiency": 0.8, "memory_capacity": 0.7},
            SpecializationRole.SPECIALIST: {"processing_power": 0.9, "learning_rate": 0.3},
            SpecializationRole.GENERALIST: {"processing_power": 0.6, "memory_capacity": 0.6},
            SpecializationRole.INNOVATOR: {"processing_power": 0.7, "learning_rate": 0.4},
            SpecializationRole.VALIDATOR: {"processing_power": 0.8, "memory_capacity": 0.8},
            SpecializationRole.COMMUNICATOR: {"communication_efficiency": 0.9, "memory_capacity": 0.5}
        }

        requirements = role_requirements.get(role, {})
        alignment_scores = []

        for capability, required_level in requirements.items():
            current_level = getattr(self.capabilities, capability, 0.5)
            alignment = 1.0 - abs(current_level - required_level)
            alignment_scores.append(alignment)

        return float(np.mean(alignment_scores)) if alignment_scores else 0.5



class CollectiveDecisionType(Enum):
    """Types of collective decision-making processes"""
    CONSENSUS = "consensus"
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_VOTING = "weighted_voting"
    EMERGENCE_BASED = "emergence_based"
    EXPERT_DELEGATION = "expert_delegation"


@dataclass
class EmergentBehavior:
    """Detected emergent behavior in the swarm"""
    behavior_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    behavior_type: str = "unknown"
    participating_agents: set[str] = field(default_factory=set)
    emergence_strength: float = 0.0
    pattern_signature: dict[str, float] = field(default_factory=dict)
    detection_timestamp: float = field(default_factory=time.time)
    stability_score: float = 0.0

    def __post_init__(self):
        """Initialize derived properties"""
        if not self.pattern_signature:
            self.pattern_signature = self._calculate_default_signature()

    def _calculate_default_signature(self) -> dict[str, float]:
        """Calculate default pattern signature based on behavior type"""
        signatures = {
            "clustering": {"spatial_coherence": 0.8, "interaction_density": 0.7},
            "specialization": {"role_diversity": 0.9, "performance_variance": 0.6},
            "coordination": {"synchronization": 0.8, "information_flow": 0.9},
            "collective_learning": {"knowledge_transfer": 0.7, "adaptation_rate": 0.8}
        }
        return signatures.get(self.behavior_type, {"generic_emergence": 0.5})


class EmergenceBehaviorDetector:
    """
    Detects emergent behaviors in swarm through pattern analysis

    Uses multiple detection algorithms to identify when collective behaviors
    emerge that are not explicitly programmed into individual agents.
    """

    def __init__(self,
                 detection_threshold: float = 0.6,
                 stability_window: int = 10):
        """
        Initialize emergence detector

        Args:
            detection_threshold: Minimum emergence strength to trigger detection
            stability_window: Number of cycles to verify behavior stability
        """
        self.detection_threshold = detection_threshold
        self.stability_window = stability_window
        self.detected_behaviors: list[EmergentBehavior] = []
        self.behavior_history: dict[str, list[float]] = defaultdict(list)

        logger.debug("Initialized EmergenceBehaviorDetector")

    def detect_emergence(self,
                        agents: list[SwarmAgent],
                        interaction_data: dict[str, Any]) -> list[EmergentBehavior]:
        """
        Detect emergent behaviors in current swarm state

        Args:
            agents: list of swarm agents to analyze
            interaction_data: Current interaction patterns and metrics

        Returns:
            list of detected emergent behaviors
        """
        newly_detected = []

        # Detect clustering emergence
        clustering_behavior = self._detect_clustering(agents, interaction_data)
        if clustering_behavior:
            newly_detected.append(clustering_behavior)

        # Detect role specialization emergence
        specialization_behavior = self._detect_specialization(agents)
        if specialization_behavior:
            newly_detected.append(specialization_behavior)

        # Detect coordination emergence
        coordination_behavior = self._detect_coordination(agents, interaction_data)
        if coordination_behavior:
            newly_detected.append(coordination_behavior)

        # Detect collective learning emergence
        learning_behavior = self._detect_collective_learning(agents)
        if learning_behavior:
            newly_detected.append(learning_behavior)

        # Validate stability of detected behaviors
        stable_behaviors = self._validate_behavior_stability(newly_detected)

        # Update behavior history
        self.detected_behaviors.extend(stable_behaviors)

        return stable_behaviors

    def _detect_clustering(self,
                          agents: list[SwarmAgent],
                          interaction_data: dict[str, Any]) -> EmergentBehavior | None:
        """Detect spatial or functional clustering of agents"""
        if len(agents) < 3:
            return None

        # Calculate interaction density matrix
        interaction_matrix = self._build_interaction_matrix(agents, interaction_data)

        # Identify clusters using simple threshold-based approach
        clusters = []
        processed_agents = set()

        for i, agent in enumerate(agents):
            if agent.agent_id in processed_agents:
                continue

            cluster = {agent.agent_id}
            for j, other_agent in enumerate(agents):
                if i != j and interaction_matrix[i][j] > 0.7:
                    cluster.add(other_agent.agent_id)

            if len(cluster) >= 3:  # Minimum cluster size
                clusters.append(cluster)
                processed_agents.update(cluster)

        if clusters:
            # Calculate emergence strength based on clustering quality
            total_agents = len(agents)
            clustered_agents = sum(len(cluster) for cluster in clusters)
            emergence_strength = clustered_agents / total_agents

            if emergence_strength > self.detection_threshold:
                return EmergentBehavior(
                    behavior_type="clustering",
                    participating_agents=set().union(*clusters),
                    emergence_strength=emergence_strength,
                    pattern_signature={
                        "cluster_count": len(clusters),
                        "largest_cluster_size": max(len(c) for c in clusters),
                        "clustering_coefficient": emergence_strength
                    }
                )

        return None

    def _detect_specialization(self, agents: list[SwarmAgent]) -> EmergentBehavior | None:
        """Detect emergent role specialization patterns"""
        if len(agents) < 3:
            return None

        # Count agents by role
        role_counts = defaultdict(int)
        specialized_agents = set()

        for agent in agents:
            if agent.current_role:
                role_counts[agent.current_role] += 1
                specialized_agents.add(agent.agent_id)

        if len(role_counts) >= 2:  # At least 2 different roles
            # Calculate specialization emergence strength
            specialization_ratio = len(specialized_agents) / len(agents)
            role_diversity = len(role_counts) / len(SpecializationRole)

            emergence_strength = (specialization_ratio + role_diversity) / 2

            if emergence_strength > self.detection_threshold:
                behavior = EmergentBehavior(
                    behavior_type="specialization",
                    participating_agents=specialized_agents,
                    emergence_strength=emergence_strength,
                    pattern_signature={
                        "role_diversity": role_diversity,
                        "specialization_ratio": specialization_ratio,
                        "unique_roles": len(role_counts)
                    }
                )
                return behavior

        return None

    def _detect_coordination(self,
                           agents: list[SwarmAgent],
                           interaction_data: dict[str, Any]) -> EmergentBehavior | None:
        """Detect coordinated behavior patterns"""
        if len(agents) < 3:
            return None

        # Analyze decision synchronization
        recent_decisions = []
        for agent in agents:
            if agent.local_memory:
                recent_actions = [record for record in agent.local_memory.values()
                                if isinstance(record, dict) and
                                record.get("timestamp", 0) > time.time() - 60]  # Last minute
                recent_decisions.extend(recent_actions)

        if len(recent_decisions) < 3:
            return None

        # Calculate temporal coordination
        decision_times = [d.get("timestamp", 0) for d in recent_decisions]
        decision_times.sort()

        # Check for synchronized decision making (decisions within small time windows)
        sync_windows = []
        window_size = 10.0  # 10 seconds

        i = 0
        while i < len(decision_times):
            window_start = decision_times[i]
            window_decisions = [decision_times[i]]

            j = i + 1
            while j < len(decision_times) and decision_times[j] - window_start <= window_size:
                window_decisions.append(decision_times[j])
                j += 1

            if len(window_decisions) >= 3:  # At least 3 coordinated decisions
                sync_windows.append(window_decisions)

            i = j if j > i + 1 else i + 1

        if sync_windows:
            # Calculate coordination strength
            total_decisions = len(recent_decisions)
            coordinated_decisions = sum(len(window) for window in sync_windows)
            coordination_ratio = coordinated_decisions / total_decisions

            if coordination_ratio > self.detection_threshold:
                participating_agents = {d.get("agent_id", "") for d in recent_decisions
                                         if d.get("agent_id")}

                return EmergentBehavior(
                    behavior_type="coordination",
                    participating_agents=participating_agents,
                    emergence_strength=coordination_ratio,
                    pattern_signature={
                        "sync_window_count": len(sync_windows),
                        "coordination_ratio": coordination_ratio,
                        "temporal_coherence": 1.0 - (max(decision_times) - min(decision_times)) / 3600
                    }
                )

        return None

    def _detect_collective_learning(self, agents: list[SwarmAgent]) -> EmergentBehavior | None:
        """Detect collective learning and knowledge transfer patterns"""
        if len(agents) < 3:
            return None

        # Analyze performance correlation and knowledge transfer
        performance_trends = []
        learning_indicators = []

        for agent in agents:
            if len(agent.capabilities.performance_history) >= 5:
                recent_performance = agent.capabilities.performance_history[-5:]
                # Simple trend calculation
                trend = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
                performance_trends.append(trend)

                # Learning indicator based on modification history
                recent_modifications = len([m for m in agent.modification_history
                                          if m.get("timestamp", 0) > time.time() - 300])
                learning_indicators.append(recent_modifications)

        if len(performance_trends) >= 3:
            # Calculate collective learning metrics
            positive_trends = sum(1 for trend in performance_trends if trend > 0)
            learning_synchrony = len([indicator for indicator in learning_indicators if indicator > 0]) / len(agents)

            emergence_strength = (positive_trends / len(agents) + learning_synchrony) / 2

            if emergence_strength > self.detection_threshold:
                return EmergentBehavior(
                    behavior_type="collective_learning",
                    participating_agents={agent.agent_id for agent in agents},
                    emergence_strength=emergence_strength,
                    pattern_signature={
                        "positive_trend_ratio": positive_trends / len(agents),
                        "learning_synchrony": learning_synchrony,
                        "avg_performance_trend": float(np.mean(performance_trends))
                    }
                )

        return None

    def _build_interaction_matrix(self,
                                 agents: list[SwarmAgent],
                                 interaction_data: dict[str, Any]) -> list[list[float]]:
        """Build interaction strength matrix between agents"""
        n_agents = len(agents)
        matrix = [[0.0 for _ in range(n_agents)] for _ in range(n_agents)]

        for i, agent in enumerate(agents):
            for j, other_agent in enumerate(agents):
                if i != j:
                    # Calculate interaction strength based on trust and communication history
                    trust_level = agent.peer_relationships.get(other_agent.agent_id, 0.0)
                    interaction_count = len([record for record in agent.interaction_history
                                           if other_agent.agent_id in record.get("peers_contacted", [])])

                    # Normalize interaction count
                    max_interactions = max(1, len(agent.interaction_history))
                    interaction_strength = interaction_count / max_interactions

                    # Combined interaction strength
                    matrix[i][j] = (trust_level + interaction_strength) / 2

        return matrix

    def _validate_behavior_stability(self,
                                   behaviors: list[EmergentBehavior]) -> list[EmergentBehavior]:
        """Validate that detected behaviors are stable over time"""
        stable_behaviors = []

        for behavior in behaviors:
            behavior_key = f"{behavior.behavior_type}_{len(behavior.participating_agents)}"
            self.behavior_history[behavior_key].append(behavior.emergence_strength)

            # Keep sliding window of history
            if len(self.behavior_history[behavior_key]) > self.stability_window:
                self.behavior_history[behavior_key] = self.behavior_history[behavior_key][-self.stability_window:]

            # Check stability - behavior should be consistently detected
            # For very strong behaviors (>= 2x threshold), allow immediate detection
            # For others, require stability across multiple detections
            history_length = len(self.behavior_history[behavior_key])
            min_history_required = min(3, self.stability_window)

            if behavior.emergence_strength >= (self.detection_threshold * 2.0):
                # Very strong behaviors can be detected immediately
                behavior.stability_score = float(behavior.emergence_strength)
                stable_behaviors.append(behavior)
            elif history_length >= min_history_required:
                recent_detections = self.behavior_history[behavior_key][-3:]
                stability_score = np.mean(recent_detections) if recent_detections else 0.0

                if stability_score > self.detection_threshold * 0.8:  # 80% of threshold for stability
                    behavior.stability_score = float(stability_score)
                    stable_behaviors.append(behavior)

        return stable_behaviors
class CollectiveDecisionMaker:
    """
    Implements various collective decision-making algorithms for swarm intelligence

    Supports consensus building, voting mechanisms, and emergence-based decisions
    based on current swarm state and agent capabilities.
    """

    def __init__(self,
                 default_decision_type: CollectiveDecisionType = CollectiveDecisionType.CONSENSUS):
        """
        Initialize collective decision maker

        Args:
            default_decision_type: Default decision-making process to use
        """
        self.default_decision_type = default_decision_type
        self.decision_history: list[dict[str, Any]] = []

        logger.debug("Initialized CollectiveDecisionMaker")

    def make_collective_decision(self,
                               agents: list[SwarmAgent],
                               decision_context: dict[str, Any],
                               decision_type: CollectiveDecisionType | None = None) -> dict[str, Any]:
        """
        Make a collective decision using specified or default algorithm

        Args:
            agents: list of agents participating in decision
            decision_context: Context information for the decision
            decision_type: Specific decision-making algorithm to use

        Returns:
            dictionary containing collective decision and metadata
        """
        decision_type = decision_type or self.default_decision_type

        if decision_type == CollectiveDecisionType.CONSENSUS:
            result = self._consensus_decision(agents, decision_context)
        elif decision_type == CollectiveDecisionType.MAJORITY_VOTING:
            result = self._majority_voting_decision(agents, decision_context)
        elif decision_type == CollectiveDecisionType.WEIGHTED_VOTING:
            result = self._weighted_voting_decision(agents, decision_context)
        elif decision_type == CollectiveDecisionType.EMERGENCE_BASED:
            result = self._emergence_based_decision(agents, decision_context)
        elif decision_type == CollectiveDecisionType.EXPERT_DELEGATION:
            result = self._expert_delegation_decision(agents, decision_context)
        else:
            result = self._consensus_decision(agents, decision_context)

        # Record decision in history
        decision_record = {
            "timestamp": time.time(),
            "decision_type": decision_type.value,
            "participating_agents": [agent.agent_id for agent in agents],
            "context": decision_context,
            "result": result
        }
        self.decision_history.append(decision_record)

        return result

    def _consensus_decision(self,
                          agents: list[SwarmAgent],
                          context: dict[str, Any]) -> dict[str, Any]:
        """Build consensus through iterative information sharing"""
        if not agents:
            return {"decision": "no_action", "confidence": 0.0, "consensus_level": 0.0}

        # Gather initial positions from all agents
        agent_positions = {}
        for agent in agents:
            agent_decision = agent.make_local_decision(
                context.get("environment_state", {}),
                context.get("peer_info", {})
            )
            agent_positions[agent.agent_id] = agent_decision

        # Iterative consensus building
        max_iterations = 5
        consensus_threshold = 0.8
        iteration = 0

        for iteration in range(max_iterations):
            # Share positions and allow agents to update
            shared_positions = {}
            for agent in agents:
                # Share position based on trust relationships
                shareable_position = {
                    "action_type": agent_positions[agent.agent_id].get("action_type"),
                    "confidence": agent_positions[agent.agent_id].get("confidence", 0.5)
                }
                shared_positions[agent.agent_id] = shareable_position

            # Calculate current consensus level
            action_types = [pos.get("action_type") for pos in agent_positions.values()]
            most_common_action = max(set(action_types), key=action_types.count)
            consensus_level = action_types.count(most_common_action) / len(action_types)

            if consensus_level >= consensus_threshold:
                break

            # Update positions based on shared information
            updated_positions = {}
            for agent in agents:
                # Agent considers peer positions and updates their own
                peer_positions = {aid: pos for aid, pos in shared_positions.items()
                                if aid != agent.agent_id}

                # Simple consensus mechanism - agents move toward majority if confidence is low
                current_confidence = agent_positions[agent.agent_id].get("confidence", 0.5)
                if current_confidence < 0.7:
                    # Consider switching to majority position
                    peer_actions = [pos.get("action_type") for pos in peer_positions.values()]
                    if peer_actions:
                        majority_action = max(set(peer_actions), key=peer_actions.count)
                        majority_count = peer_actions.count(majority_action)

                        if majority_count / len(peer_actions) > 0.6:
                            # Switch to majority position with adjusted confidence
                            updated_positions[agent.agent_id] = {
                                "action_type": majority_action,
                                "confidence": current_confidence * 0.8,  # Reduced confidence
                                "reasoning": f"Consensus adaptation to majority ({majority_action})"
                            }
                        else:
                            updated_positions[agent.agent_id] = agent_positions[agent.agent_id]
                    else:
                        updated_positions[agent.agent_id] = agent_positions[agent.agent_id]
                else:
                    # High confidence agents maintain their position
                    updated_positions[agent.agent_id] = agent_positions[agent.agent_id]

            agent_positions = updated_positions

        # Final consensus calculation
        final_action_types = [pos.get("action_type") for pos in agent_positions.values()]
        final_action = max(set(final_action_types), key=final_action_types.count)
        final_consensus_level = final_action_types.count(final_action) / len(final_action_types)

        # Calculate average confidence
        confidences = [pos.get("confidence", 0.5) for pos in agent_positions.values()]
        avg_confidence = np.mean(confidences)

        return {
            "decision": final_action,
            "confidence": avg_confidence,
            "consensus_level": final_consensus_level,
            "iterations": iteration + 1,
            "participating_agents": len(agents),
            "decision_method": "consensus_building"
        }

    def _majority_voting_decision(self,
                                agents: list[SwarmAgent],
                                context: dict[str, Any]) -> dict[str, Any]:
        """Simple majority voting mechanism"""
        if not agents:
            return {"decision": "no_action", "confidence": 0.0, "vote_ratio": 0.0}

        # Collect votes from all agents
        votes = {}
        confidences = []

        for agent in agents:
            agent_decision = agent.make_local_decision(
                context.get("environment_state", {}),
                context.get("peer_info", {})
            )

            vote = agent_decision.get("action_type", "no_action")
            confidence = agent_decision.get("confidence", 0.5)

            votes[vote] = votes.get(vote, 0) + 1
            confidences.append(confidence)

        # Determine majority decision
        majority_decision = max(votes.keys(), key=lambda k: votes[k])
        majority_count = votes[majority_decision]
        vote_ratio = majority_count / len(agents)

        return {
            "decision": majority_decision,
            "confidence": np.mean(confidences),
            "vote_ratio": vote_ratio,
            "vote_distribution": votes,
            "decision_method": "majority_voting"
        }

    def _weighted_voting_decision(self,
                                agents: list[SwarmAgent],
                                context: dict[str, Any]) -> dict[str, Any]:
        """Weighted voting based on agent performance and expertise"""
        if not agents:
            return {"decision": "no_action", "confidence": 0.0, "weighted_score": 0.0}

        # Collect weighted votes
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        confidences = []

        for agent in agents:
            agent_decision = agent.make_local_decision(
                context.get("environment_state", {}),
                context.get("peer_info", {})
            )

            vote = agent_decision.get("action_type", "no_action")
            confidence = agent_decision.get("confidence", 0.5)

            # Calculate agent weight based on performance and role
            performance_weight = agent.capabilities.get_avg_performance()
            role_weight = 1.2 if agent.current_role else 1.0
            expertise_weight = confidence

            agent_weight = (performance_weight + role_weight + expertise_weight) / 3

            weighted_votes[vote] += agent_weight
            total_weight += agent_weight
            confidences.append(confidence)

        # Normalize weights and find winner
        if total_weight > 0:
            normalized_votes = {vote: weight/total_weight for vote, weight in weighted_votes.items()}
            winning_decision = max(normalized_votes.keys(), key=lambda k: normalized_votes[k])
            winning_score = normalized_votes[winning_decision]
        else:
            winning_decision = "no_action"
            winning_score = 0.0
            normalized_votes = {}

        return {
            "decision": winning_decision,
            "confidence": np.mean(confidences) if confidences else 0.0,
            "weighted_score": winning_score,
            "vote_distribution": dict(normalized_votes),
            "decision_method": "weighted_voting"
        }

    def _emergence_based_decision(self,
                                agents: list[SwarmAgent],
                                context: dict[str, Any]) -> dict[str, Any]:
        """Decision based on detected emergent behaviors"""
        if not agents:
            return {"decision": "no_action", "confidence": 0.0, "emergence_strength": 0.0}

        # Detect current emergent behaviors
        detector = EmergenceBehaviorDetector()
        emergent_behaviors = detector.detect_emergence(agents, context)

        if not emergent_behaviors:
            # Fallback to consensus if no emergence detected
            return self._consensus_decision(agents, context)

        # Select decision based on strongest emergent behavior
        strongest_behavior = max(emergent_behaviors, key=lambda b: b.emergence_strength)

        # Map emergent behavior to decision
        behavior_decisions = {
            "clustering": "coordinate",
            "specialization": "specialize",
            "coordination": "collaborate",
            "collective_learning": "adapt"
        }

        decision = behavior_decisions.get(strongest_behavior.behavior_type, "coordinate")

        # Calculate confidence based on emergence strength and stability
        confidence = (strongest_behavior.emergence_strength +
                     strongest_behavior.stability_score) / 2

        return {
            "decision": decision,
            "confidence": confidence,
            "emergence_strength": strongest_behavior.emergence_strength,
            "guiding_behavior": strongest_behavior.behavior_type,
            "participating_agents": len(strongest_behavior.participating_agents),
            "decision_method": "emergence_based"
        }

    def _expert_delegation_decision(self,
                                  agents: list[SwarmAgent],
                                  context: dict[str, Any]) -> dict[str, Any]:
        """Delegate decision to most qualified expert agent"""
        if not agents:
            return {"decision": "no_action", "confidence": 0.0, "expert_score": 0.0}

        # Identify expert based on context requirements
        decision_context_type = context.get("decision_type", "general")

        # Score agents based on expertise for this context
        expert_scores = {}
        for agent in agents:
            # Base score from performance
            performance_score = agent.capabilities.get_avg_performance()

            # Role relevance score
            role_relevance = self._calculate_role_relevance(agent.current_role, decision_context_type)

            # Experience score
            experience_score = len(agent.interaction_history) / 100.0  # Normalize
            experience_score = min(experience_score, 1.0)

            # Combined expert score
            expert_score = (performance_score * 0.5 +
                          role_relevance * 0.3 +
                          experience_score * 0.2)

            expert_scores[agent.agent_id] = expert_score

        # Select expert agent
        expert_agent_id = max(expert_scores.keys(), key=lambda k: expert_scores[k])
        expert_agent = next(agent for agent in agents if agent.agent_id == expert_agent_id)
        expert_score = expert_scores[expert_agent_id]

        # Get expert's decision
        expert_decision = expert_agent.make_local_decision(
            context.get("environment_state", {}),
            context.get("peer_info", {})
        )

        return {
            "decision": expert_decision.get("action_type", "no_action"),
            "confidence": expert_decision.get("confidence", 0.5),
            "expert_score": expert_score,
            "expert_agent": expert_agent_id,
            "expert_role": expert_agent.current_role.value if expert_agent.current_role else "unassigned",
            "decision_method": "expert_delegation"
        }

    def _calculate_role_relevance(self,
                                role: SpecializationRole | None,
                                context_type: str) -> float:
        """Calculate how relevant an agent's role is to the decision context"""
        if not role:
            return 0.3  # Generalist score

        relevance_map = {
            "exploration": {
                SpecializationRole.EXPLORER: 1.0,
                SpecializationRole.INNOVATOR: 0.8,
                SpecializationRole.GENERALIST: 0.6
            },
            "optimization": {
                SpecializationRole.EXPLOITER: 1.0,
                SpecializationRole.SPECIALIST: 0.8,
                SpecializationRole.VALIDATOR: 0.7
            },
            "coordination": {
                SpecializationRole.COORDINATOR: 1.0,
                SpecializationRole.COMMUNICATOR: 0.9,
                SpecializationRole.GENERALIST: 0.6
            },
            "general": {
                SpecializationRole.GENERALIST: 1.0,
                SpecializationRole.COORDINATOR: 0.7
            }
        }

        context_relevance = relevance_map.get(context_type, {})
        return context_relevance.get(role, 0.5)


class SwarmCommunicationProtocol:
    """
    Manages efficient communication protocols for large-scale swarm operations

    Implements hierarchical, broadcast, and selective communication patterns
    to scale beyond direct peer-to-peer communication.
    """

    def __init__(self,
                 max_direct_connections: int = 10,
                 broadcast_threshold: int = 50):
        """
        Initialize communication protocol

        Args:
            max_direct_connections: Maximum direct peer connections per agent
            broadcast_threshold: Swarm size threshold for switching to broadcast mode
        """
        self.max_direct_connections = max_direct_connections
        self.broadcast_threshold = broadcast_threshold
        self.message_history: list[dict[str, Any]] = []
        self.communication_topology: nx.Graph = nx.Graph()

        logger.debug("Initialized SwarmCommunicationProtocol")

    def facilitate_communication(self,
                                agents: list[SwarmAgent],
                                communication_context: dict[str, Any]) -> dict[str, Any]:
        """
        Facilitate communication between agents using appropriate protocol

        Args:
            agents: list of agents to facilitate communication for
            communication_context: Context and parameters for communication

        Returns:
            Communication results and statistics
        """
        if len(agents) <= self.broadcast_threshold:
            return self._direct_communication(agents, communication_context)
        else:
            return self._hierarchical_communication(agents, communication_context)

    def _direct_communication(self,
                            agents: list[SwarmAgent],
                            context: dict[str, Any]) -> dict[str, Any]:
        """Direct peer-to-peer communication for smaller swarms"""
        communication_results = {}
        total_messages = 0
        successful_exchanges = 0

        for agent in agents:
            # Select communication partners based on trust and relevance
            potential_partners = [other for other in agents if other.agent_id != agent.agent_id]

            # Limit connections to prevent communication overload
            max_partners = min(self.max_direct_connections, len(potential_partners))

            # Select top partners based on trust scores
            partners = sorted(potential_partners,
                            key=lambda p: agent.peer_relationships.get(p.agent_id, 0.0),
                            reverse=True)[:max_partners]

            # Facilitate communication
            if partners:
                peer_info = agent.communicate_with_peers(partners)
                communication_results[agent.agent_id] = {
                    "partners_contacted": len(partners),
                    "information_received": len(peer_info),
                    "communication_success": len(peer_info) > 0
                }

                total_messages += len(partners)
                if len(peer_info) > 0:
                    successful_exchanges += 1

        return {
            "communication_type": "direct",
            "total_messages": total_messages,
            "successful_exchanges": successful_exchanges,
            "exchange_rate": successful_exchanges / len(agents) if agents else 0.0,
            "agent_results": communication_results
        }

    def _hierarchical_communication(self,
                                  agents: list[SwarmAgent],
                                  context: dict[str, Any]) -> dict[str, Any]:
        """Hierarchical communication for large swarms"""
        # Organize agents into communication groups based on roles and performance
        groups = self._organize_communication_groups(agents)

        # Select group representatives (coordinators)
        representatives = self._select_group_representatives(groups)

        # Facilitate intra-group communication
        intra_group_results = {}
        for group_id, group_agents in groups.items():
            if len(group_agents) > 1:
                group_result = self._direct_communication(group_agents, context)
                intra_group_results[group_id] = group_result

        # Facilitate inter-group communication through representatives
        if len(representatives) > 1:
            inter_group_result = self._direct_communication(representatives, context)
        else:
            inter_group_result = {"communication_type": "none", "total_messages": 0}

        # Propagate representative insights back to groups
        propagation_results = self._propagate_representative_insights(
            groups, representatives, context
        )

        # Aggregate results
        total_messages = sum(result.get("total_messages", 0)
                           for result in intra_group_results.values())
        total_messages += inter_group_result.get("total_messages", 0)

        total_exchanges = sum(result.get("successful_exchanges", 0)
                            for result in intra_group_results.values())
        total_exchanges += inter_group_result.get("successful_exchanges", 0)

        return {
            "communication_type": "hierarchical",
            "total_messages": total_messages,
            "successful_exchanges": total_exchanges,
            "groups_formed": len(groups),
            "representatives": len(representatives),
            "intra_group_results": intra_group_results,
            "inter_group_result": inter_group_result,
            "propagation_results": propagation_results
        }

    def _organize_communication_groups(self, agents: list[SwarmAgent]) -> dict[str, list[SwarmAgent]]:
        """Organize agents into communication groups"""
        groups = defaultdict(list)

        # Group by role first
        role_groups = defaultdict(list)
        unassigned_agents = []

        for agent in agents:
            if agent.current_role:
                role_groups[agent.current_role.value].append(agent)
            else:
                unassigned_agents.append(agent)

        # Create balanced groups (max 8 agents per group)
        max_group_size = 8
        group_counter = 0

        for role, role_agents in role_groups.items():
            if len(role_agents) <= max_group_size:
                groups[f"role_{role}"] = role_agents
            else:
                # Split large role groups
                for i in range(0, len(role_agents), max_group_size):
                    group_agents = role_agents[i:i+max_group_size]
                    groups[f"role_{role}_{group_counter}"] = group_agents
                    group_counter += 1

        # Distribute unassigned agents
        if unassigned_agents:
            for i in range(0, len(unassigned_agents), max_group_size):
                group_agents = unassigned_agents[i:i+max_group_size]
                groups[f"unassigned_{group_counter}"] = group_agents
                group_counter += 1

        return dict(groups)

    def _select_group_representatives(self, groups: dict[str, list[SwarmAgent]]) -> list[SwarmAgent]:
        """Select representative agents from each group"""
        representatives = []

        for group_id, group_agents in groups.items():
            if not group_agents:
                continue

            # Select representative based on performance and communication skills
            best_representative = None
            best_score = -1.0

            for agent in group_agents:
                # Score based on performance, communication efficiency, and role
                performance_score = agent.capabilities.get_avg_performance()
                comm_score = agent.capabilities.communication_efficiency
                role_bonus = 0.2 if agent.current_role == SpecializationRole.COORDINATOR else 0.0

                total_score = performance_score * 0.4 + comm_score * 0.4 + role_bonus

                if total_score > best_score:
                    best_score = total_score
                    best_representative = agent

            if best_representative:
                representatives.append(best_representative)

        return representatives

    def _propagate_representative_insights(self,
                                         groups: dict[str, list[SwarmAgent]],
                                         representatives: list[SwarmAgent],
                                         context: dict[str, Any]) -> dict[str, Any]:
        """Propagate insights from representatives back to their groups"""
        propagation_results = {}

        # Collect insights from representatives
        representative_insights = {}
        for rep in representatives:
            # Get recent high-value information from representative
            recent_info = {}
            if rep.local_memory:
                recent_actions = [record for record in rep.local_memory.values()
                                if isinstance(record, dict) and
                                record.get("timestamp", 0) > time.time() - 180]  # Last 3 minutes
                if recent_actions:
                    recent_info["recent_insights"] = recent_actions[-3:]  # Last 3 insights

            representative_insights[rep.agent_id] = recent_info

        # Propagate to group members
        for group_id, group_agents in groups.items():
            group_representative = None
            for agent in group_agents:
                if agent in representatives:
                    group_representative = agent
                    break

            if group_representative and group_representative.agent_id in representative_insights:
                insights_to_share = representative_insights[group_representative.agent_id]

                # Share insights with group members
                propagated_count = 0
                for agent in group_agents:
                    if agent.agent_id != group_representative.agent_id:
                        # Store shared insight in agent's memory
                        insight_key = f"representative_insight_{int(time.time())}"
                        if len(agent.local_memory) < agent.capabilities.memory_capacity * 100:
                            agent.local_memory[insight_key] = {
                                "source": "representative",
                                "representative_id": group_representative.agent_id,
                                "insights": insights_to_share,
                                "timestamp": time.time()
                            }
                            propagated_count += 1

                propagation_results[group_id] = {
                    "representative": group_representative.agent_id,
                    "insights_shared": len(insights_to_share),
                    "members_reached": propagated_count
                }

        return propagation_results


class SwarmIntelligence:
    """
    Main orchestrator for collective intelligence in multi-agent swarms

    Implements the core Enhanced ELCS swarm intelligence capabilities:
    - Emergent specialization through dynamic role assignment
    - Distributed cognition across agent networks
    - Self-modifying architectures with collective learning
    - Integration with Dynamic Emergence Networks for cross-scale emergence

    Based on 2024-2025 research showing 90.2% performance improvement through
    collective intelligence mechanisms over single-agent systems.
    """

    def __init__(self,
                 den: DynamicEmergenceNetwork | None = None,
                 max_agents: int = 100,
                 emergence_detection_interval: float = 30.0,
                 decision_timeout: float = 10.0):
        """
        Initialize SwarmIntelligence orchestrator

        Args:
            den: Dynamic Emergence Network for cross-scale integration
            max_agents: Maximum number of agents in the swarm
            emergence_detection_interval: Seconds between emergence detection cycles
            decision_timeout: Maximum time for collective decision making
        """
        self.den = den
        self.max_agents = max_agents
        self.emergence_detection_interval = emergence_detection_interval
        self.decision_timeout = decision_timeout

        # Core components
        self.agents: dict[str, SwarmAgent] = {}
        self.emergence_detector = EmergenceBehaviorDetector()
        self.decision_maker = CollectiveDecisionMaker()
        self.communication_protocol = SwarmCommunicationProtocol()

        # Swarm state tracking
        self.current_emergent_behaviors: list[EmergentBehavior] = []
        self.swarm_performance_history: list[float] = []
        self.role_assignment_history: list[dict[str, str | None]] = []

        # Integration with DEN
        self.process_entities: dict[str, ProcessEntity] = {}
        self.cross_scale_interactions: list[dict[str, Any]] = []

        # Control flow
        self.is_active = False
        self.last_emergence_detection = 0.0
        self.swarm_metrics: dict[str, float] = {}

        logger.info(f"Initialized SwarmIntelligence with max {max_agents} agents")

    def add_agent(self,
                  agent: SwarmAgent | None = None,
                  initial_capabilities: AgentCapabilities | None = None) -> str:
        """
        Add agent to the swarm or create new agent

        Args:
            agent: Existing SwarmAgent to add, or None to create new
            initial_capabilities: Capabilities for new agent if agent is None

        Returns:
            Agent ID of added/created agent

        Raises:
            ValueError: If swarm is at maximum capacity
        """
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Swarm at maximum capacity ({self.max_agents} agents)")

        if agent is None:
            agent = SwarmAgent(
                initial_capabilities=initial_capabilities or AgentCapabilities(),
                den=self.den
            )

        self.agents[agent.agent_id] = agent

        # Create corresponding ProcessEntity in DEN if available
        if self.den is not None:
            process_entity = ProcessEntity(
                entity_id=f"agent_{agent.agent_id}",
                scale=ProcessScale.SOCIAL,  # Agents operate at social scale
                emergence_potential=0.7  # High potential for emergence
            )
            self.process_entities[agent.agent_id] = process_entity
            # Integrate with DEN
            self.den.add_process_entity(process_entity)

        logger.debug(f"Added agent {agent.agent_id} to swarm (total: {len(self.agents)})")
        return agent.agent_id

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from swarm

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_id in self.agents:
            del self.agents[agent_id]

            # Remove from DEN tracking
            if agent_id in self.process_entities:
                del self.process_entities[agent_id]

            logger.debug(f"Removed agent {agent_id} from swarm (remaining: {len(self.agents)})")
            return True

        return False

    def start_swarm_intelligence(self) -> None:
        """Start the swarm intelligence orchestration process"""
        if self.is_active:
            logger.warning("SwarmIntelligence already active")
            return

        if not self.agents:
            logger.warning("No agents in swarm - cannot start intelligence process")
            return

        self.is_active = True
        logger.info(f"Started SwarmIntelligence with {len(self.agents)} agents")

    def stop_swarm_intelligence(self) -> None:
        """Stop the swarm intelligence orchestration process"""
        self.is_active = False
        logger.info("Stopped SwarmIntelligence")

    def execute_swarm_cycle(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute one complete swarm intelligence cycle

        Args:
            environment_state: Current environment state for agent perception

        Returns:
            Swarm cycle results including decisions, emergent behaviors, and metrics
        """
        if not self.is_active:
            return {"error": "SwarmIntelligence not active", "cycle_completed": False}

        if not self.agents:
            return {"error": "No agents in swarm", "cycle_completed": False}

        cycle_start_time = time.time()

        # Phase 1: Agent Perception and Local Processing
        perception_results = self._execute_perception_phase(environment_state)

        # Phase 2: Inter-Agent Communication
        communication_results = self._execute_communication_phase(environment_state)

        # Phase 3: Role Assignment and Specialization
        specialization_results = self._execute_specialization_phase()

        # Phase 4: Collective Decision Making
        decision_results = self._execute_decision_phase(environment_state)

        # Phase 5: Emergence Detection and Response
        emergence_results = self._execute_emergence_phase(environment_state)

        # Phase 6: Action Execution and Performance Tracking
        execution_results = self._execute_action_phase(decision_results)

        # Phase 7: Self-Modification and Learning
        learning_results = self._execute_learning_phase(execution_results)

        # Calculate cycle metrics
        cycle_time = time.time() - cycle_start_time
        cycle_metrics = self._calculate_cycle_metrics(
            perception_results, communication_results, specialization_results,
            decision_results, emergence_results, execution_results, learning_results
        )

        # Update swarm performance history
        overall_performance = cycle_metrics.get("overall_performance", 0.5)
        self.swarm_performance_history.append(overall_performance)
        if len(self.swarm_performance_history) > 1000:  # Keep last 1000 cycles
            self.swarm_performance_history = self.swarm_performance_history[-1000:]

        return {
            "cycle_completed": True,
            "cycle_time": cycle_time,
            "phase_results": {
                "perception": perception_results,
                "communication": communication_results,
                "specialization": specialization_results,
                "decision": decision_results,
                "emergence": emergence_results,
                "execution": execution_results,
                "learning": learning_results
            },
            "cycle_metrics": cycle_metrics,
            "swarm_state": self._get_swarm_state_summary()
        }

    def _execute_perception_phase(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        """Execute agent perception phase"""
        perception_results = {}
        total_perception_quality = 0.0

        for agent_id, agent in self.agents.items():
            try:
                perceived_state = agent.perceive_environment(environment_state)
                perception_quality = len(perceived_state) / 10.0  # Normalize by expected signals
                perception_quality = min(perception_quality, 1.0)

                perception_results[agent_id] = {
                    "perceived_signals": len(perceived_state),
                    "perception_quality": perception_quality,
                    "role_specific_focus": agent.current_role.value if agent.current_role else None
                }

                total_perception_quality += perception_quality

            except Exception as e:
                logger.error(f"Perception error for agent {agent_id}: {e}")
                perception_results[agent_id] = {
                    "error": str(e),
                    "perceived_signals": 0,
                    "perception_quality": 0.0
                }

        avg_perception_quality = total_perception_quality / len(self.agents) if self.agents else 0.0

        return {
            "agent_results": perception_results,
            "average_perception_quality": avg_perception_quality,
            "total_agents_active": len([r for r in perception_results.values() if "error" not in r])
        }

    def _execute_communication_phase(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        """Execute inter-agent communication phase"""
        communication_context = {
            "environment_state": environment_state,
            "swarm_size": len(self.agents),
            "current_behaviors": [b.behavior_type for b in self.current_emergent_behaviors]
        }

        try:
            agent_list = list(self.agents.values())
            communication_results = self.communication_protocol.facilitate_communication(
                agent_list, communication_context
            )

            return communication_results

        except Exception as e:
            logger.error(f"Communication phase error: {e}")
            return {
                "error": str(e),
                "communication_type": "failed",
                "total_messages": 0,
                "successful_exchanges": 0
            }

    def _execute_specialization_phase(self) -> dict[str, Any]:
        """Execute role assignment and specialization phase"""
        role_assignments = {}
        role_changes = 0

        for agent_id, agent in self.agents.items():
            try:
                # Get agent's specialization preferences
                preferences = agent.get_specialization_preference()

                # Determine optimal role based on preferences and swarm needs
                optimal_role = self._determine_optimal_role(agent, preferences)

                # Assign role if different from current
                previous_role = agent.current_role
                if optimal_role != previous_role:
                    agent.current_role = optimal_role
                    role_changes += 1

                    # Update role experience
                    if optimal_role:
                        agent.role_experience[optimal_role] += 0.1

                role_assignments[agent_id] = {
                    "previous_role": previous_role.value if previous_role else None,
                    "assigned_role": optimal_role.value if optimal_role else None,
                    "role_changed": optimal_role != previous_role,
                    "specialization_confidence": max(preferences.values()) if preferences else 0.0
                }

            except Exception as e:
                logger.error(f"Specialization error for agent {agent_id}: {e}")
                role_assignments[agent_id] = {"error": str(e)}

        # Calculate specialization metrics
        role_distribution = defaultdict(int)
        for agent in self.agents.values():
            if agent.current_role:
                role_distribution[agent.current_role.value] += 1

        specialization_diversity = len(role_distribution) / len(SpecializationRole) if SpecializationRole else 0.0
        specialization_balance = 1.0 - (max(role_distribution.values()) / len(self.agents)) if self.agents and role_distribution else 0.0

        # Store role assignment history
        current_assignment = {agent_id: agent.current_role.value if agent.current_role else None
                            for agent_id, agent in self.agents.items()}
        self.role_assignment_history.append(current_assignment)
        if len(self.role_assignment_history) > 100:
            self.role_assignment_history = self.role_assignment_history[-100:]

        return {
            "role_assignments": role_assignments,
            "role_changes": role_changes,
            "role_distribution": dict(role_distribution),
            "specialization_diversity": specialization_diversity,
            "specialization_balance": specialization_balance
        }

    def _execute_decision_phase(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        """Execute collective decision-making phase"""
        decision_context = {
            "environment_state": environment_state,
            "swarm_state": self._get_swarm_state_summary(),
            "emergent_behaviors": self.current_emergent_behaviors,
            "decision_type": self._determine_decision_type()
        }

        try:
            agent_list = list(self.agents.values())

            # Select decision-making algorithm based on context
            decision_type = self._select_decision_algorithm(decision_context)

            # Execute collective decision
            collective_decision = self.decision_maker.make_collective_decision(
                agent_list, decision_context, decision_type
            )

            return {
                "collective_decision": collective_decision,
                "decision_algorithm": decision_type.value,
                "participating_agents": len(agent_list),
                "decision_context": decision_context
            }

        except Exception as e:
            logger.error(f"Decision phase error: {e}")
            return {
                "error": str(e),
                "collective_decision": {"decision": "no_action", "confidence": 0.0},
                "decision_algorithm": "error_fallback"
            }

    def _execute_emergence_phase(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        """Execute emergence detection and response phase"""
        current_time = time.time()

        # Check if it's time for emergence detection
        if current_time - self.last_emergence_detection < self.emergence_detection_interval:
            return {
                "emergence_detection_skipped": True,
                "time_until_next_detection": self.emergence_detection_interval - (current_time - self.last_emergence_detection),
                "current_behaviors": len(self.current_emergent_behaviors)
            }

        try:
            # Build interaction data for emergence detection
            interaction_data = self._build_interaction_data(environment_state)

            # Detect emergent behaviors
            agent_list = list(self.agents.values())
            detected_behaviors = self.emergence_detector.detect_emergence(agent_list, interaction_data)

            # Update current emergent behaviors
            self.current_emergent_behaviors = detected_behaviors

            # Integrate with Dynamic Emergence Network if available
            den_integration_results = self._integrate_emergence_with_den(detected_behaviors)

            self.last_emergence_detection = current_time

            return {
                "emergence_detection_completed": True,
                "behaviors_detected": len(detected_behaviors),
                "detected_behaviors": [
                    {
                        "type": behavior.behavior_type,
                        "strength": behavior.emergence_strength,
                        "stability": behavior.stability_score,
                        "participants": len(behavior.participating_agents)
                    }
                    for behavior in detected_behaviors
                ],
                "den_integration": den_integration_results,
                "detection_timestamp": current_time
            }

        except Exception as e:
            logger.error(f"Emergence phase error: {e}")
            return {
                "error": str(e),
                "emergence_detection_completed": False,
                "behaviors_detected": 0
            }

    def _execute_action_phase(self, decision_results: dict[str, Any]) -> dict[str, Any]:
        """Execute actions based on collective decisions"""
        collective_decision = decision_results.get("collective_decision", {})
        action_to_execute = collective_decision.get("decision", "no_action")
        decision_confidence = collective_decision.get("confidence", 0.0)

        execution_results = {}
        total_performance = 0.0
        successful_executions = 0

        for agent_id, agent in self.agents.items():
            try:
                # Create action decision based on collective decision and agent specialization
                agent_action_decision = self._adapt_collective_decision_for_agent(
                    agent, collective_decision
                )

                # Execute action
                action_results = agent.execute_action(agent_action_decision)

                execution_results[agent_id] = {
                    "action_executed": agent_action_decision.get("action_type", "no_action"),
                    "performance": action_results.get("performance", 0.0),
                    "efficiency": action_results.get("efficiency", 0.0),
                    "role_fit": action_results.get("role_fit", 1.0),
                    "success": action_results.get("action_success", False)
                }

                performance = action_results.get("performance", 0.0)
                total_performance += performance

                if action_results.get("action_success", False):
                    successful_executions += 1

            except Exception as e:
                logger.error(f"Action execution error for agent {agent_id}: {e}")
                execution_results[agent_id] = {
                    "error": str(e),
                    "action_executed": "error",
                    "performance": 0.0,
                    "success": False
                }

        avg_performance = total_performance / len(self.agents) if self.agents else 0.0
        success_rate = successful_executions / len(self.agents) if self.agents else 0.0

        return {
            "collective_action": action_to_execute,
            "decision_confidence": decision_confidence,
            "agent_results": execution_results,
            "average_performance": avg_performance,
            "success_rate": success_rate,
            "total_agents": len(self.agents)
        }

    def _execute_learning_phase(self, execution_results: dict[str, Any]) -> dict[str, Any]:
        """Execute self-modification and learning phase"""
        avg_performance = execution_results.get("average_performance", 0.5)
        success_rate = execution_results.get("success_rate", 0.5)

        modification_count = 0
        learning_metrics = {}

        for agent_id, agent in self.agents.items():
            try:
                # Get agent's specific performance
                agent_result = execution_results.get("agent_results", {}).get(agent_id, {})
                agent_performance = agent_result.get("performance", 0.5)

                # Trigger self-modification based on performance
                agent.modify_self_architecture(agent_performance)

                # Update role performance history
                if agent.current_role:
                    agent.role_performance[agent.current_role].append(agent_performance)
                    # Keep sliding window
                    if len(agent.role_performance[agent.current_role]) > 50:
                        agent.role_performance[agent.current_role] = agent.role_performance[agent.current_role][-50:]

                # Check if agent was modified
                recent_modifications = [m for m in agent.modification_history
                                     if m.get("timestamp", 0) > time.time() - 60]
                if recent_modifications:
                    modification_count += 1

            except Exception as e:
                logger.error(f"Learning phase error for agent {agent_id}: {e}")

        # Calculate swarm-level learning metrics
        performance_trend = self._calculate_performance_trend()
        adaptation_rate = modification_count / len(self.agents) if self.agents else 0.0

        learning_metrics = {
            "agents_modified": modification_count,
            "adaptation_rate": adaptation_rate,
            "performance_trend": performance_trend,
            "swarm_avg_performance": avg_performance,
            "collective_success_rate": success_rate
        }

        return learning_metrics

    def _determine_optimal_role(self,
                              agent: SwarmAgent,
                              preferences: dict[SpecializationRole, float]) -> SpecializationRole | None:
        """Determine optimal role for agent based on preferences and swarm needs"""
        if not preferences:
            return None

        # Calculate swarm role needs
        current_role_counts = defaultdict(int)
        for a in self.agents.values():
            if a.current_role:
                current_role_counts[a.current_role] += 1

        # Target distribution (balanced with slight preference for coordinators and generalists)
        target_ratios = {
            SpecializationRole.COORDINATOR: 0.15,  # 15% coordinators
            SpecializationRole.COMMUNICATOR: 0.10,  # 10% communicators
            SpecializationRole.GENERALIST: 0.20,   # 20% generalists
            SpecializationRole.SPECIALIST: 0.15,   # 15% specialists
            SpecializationRole.EXPLORER: 0.15,     # 15% explorers
            SpecializationRole.EXPLOITER: 0.15,    # 15% exploiters
            SpecializationRole.INNOVATOR: 0.05,    # 5% innovators
            SpecializationRole.VALIDATOR: 0.05     # 5% validators
        }

        # Calculate role scores combining preference and need
        role_scores = {}
        for role, preference_score in preferences.items():
            current_count = current_role_counts[role]
            target_count = target_ratios.get(role, 0.1) * len(self.agents)

            # Need score - higher if we have fewer than target
            need_score = max(0.0, (target_count - current_count) / max(target_count, 1.0))
            need_score = min(need_score, 1.0)

            # Combined score
            combined_score = 0.6 * preference_score + 0.4 * need_score
            role_scores[role] = combined_score

        # Select role with highest combined score
        if role_scores:
            optimal_role = max(role_scores.keys(), key=lambda r: role_scores[r])
            # Only assign if score is above threshold
            if role_scores[optimal_role] > 0.3:
                return optimal_role

        return None

    def _determine_decision_type(self) -> str:
        """Determine appropriate decision type based on current context"""
        # Simple heuristics for decision type selection
        if len(self.agents) < 5:
            return "consensus"
        elif any(behavior.behavior_type == "coordination" for behavior in self.current_emergent_behaviors):
            return "emergence"
        elif len([agent for agent in self.agents.values() if agent.current_role]) > len(self.agents) * 0.7:
            return "expert_delegation"
        else:
            return "weighted_voting"

    def _select_decision_algorithm(self, context: dict[str, Any]) -> CollectiveDecisionType:
        """Select appropriate decision algorithm based on context"""
        decision_type_str = context.get("decision_type", "consensus")

        algorithm_map = {
            "consensus": CollectiveDecisionType.CONSENSUS,
            "majority": CollectiveDecisionType.MAJORITY_VOTING,
            "weighted": CollectiveDecisionType.WEIGHTED_VOTING,
            "emergence": CollectiveDecisionType.EMERGENCE_BASED,
            "expert_delegation": CollectiveDecisionType.EXPERT_DELEGATION
        }

        return algorithm_map.get(decision_type_str, CollectiveDecisionType.CONSENSUS)

    def _build_interaction_data(self, environment_state: dict[str, Any]) -> dict[str, Any]:
        """Build interaction data for emergence detection"""
        interaction_data = {
            "environment_state": environment_state,
            "agent_count": len(self.agents),
            "role_distribution": {},
            "communication_patterns": {},
            "performance_metrics": {}
        }

        # Role distribution
        role_counts = defaultdict(int)
        for agent in self.agents.values():
            if agent.current_role:
                role_counts[agent.current_role.value] += 1
        interaction_data["role_distribution"] = dict(role_counts)

        # Communication patterns
        total_interactions = 0
        total_trust = 0.0
        for agent in self.agents.values():
            total_interactions += len(agent.interaction_history)
            total_trust += sum(agent.peer_relationships.values())

        interaction_data["communication_patterns"] = {
            "total_interactions": total_interactions,
            "average_trust": total_trust / max(len(self.agents), 1),
            "communication_density": total_interactions / max(len(self.agents) ** 2, 1)
        }

        # Performance metrics
        performances = [agent.capabilities.get_avg_performance() for agent in self.agents.values()]
        interaction_data["performance_metrics"] = {
            "average_performance": np.mean(performances) if performances else 0.5,
            "performance_variance": np.var(performances) if performances else 0.0,
            "performance_trend": self._calculate_performance_trend()
        }

        return interaction_data

    def _integrate_emergence_with_den(self, behaviors: list[EmergentBehavior]) -> dict[str, Any]:
        """Integrate detected emergence with Dynamic Emergence Network"""
        if self.den is None:
            return {"integration_available": False}

        integration_results = {
            "integration_available": True,
            "behaviors_integrated": 0,
            "scale_transitions_detected": 0,
            "cross_scale_patterns": []
        }

        for behavior in behaviors:
            try:
                # Check for potential scale transitions
                if behavior.emergence_strength > 0.8 and behavior.stability_score > 0.7:
                    # Strong, stable emergence might indicate scale transition
                    integration_results["scale_transitions_detected"] += 1

                    # Record cross-scale interaction
                    cross_scale_interaction = {
                        "behavior_type": behavior.behavior_type,
                        "source_agents": list(behavior.participating_agents),
                        "emergence_strength": behavior.emergence_strength,
                        "potential_scale": ProcessScale.TECHNOLOGICAL.value  # Swarm → Tech scale
                    }
                    self.cross_scale_interactions.append(cross_scale_interaction)
                    integration_results["cross_scale_patterns"].append(cross_scale_interaction)

                integration_results["behaviors_integrated"] += 1

            except Exception as e:
                logger.error(f"DEN integration error for behavior {behavior.behavior_id}: {e}")

        return integration_results

    def _adapt_collective_decision_for_agent(self,
                                           agent: SwarmAgent,
                                           collective_decision: dict[str, Any]) -> dict[str, Any]:
        """Adapt collective decision for individual agent based on role and capabilities"""
        base_decision = collective_decision.get("decision", "no_action")
        base_confidence = collective_decision.get("confidence", 0.5)

        # Role-specific adaptations
        if agent.current_role == SpecializationRole.COORDINATOR:
            # Coordinators focus on group management
            if base_decision in ["collaborate", "coordinate"]:
                adapted_decision = "coordinate"
                confidence_boost = 0.1
            else:
                adapted_decision = base_decision
                confidence_boost = 0.0

        elif agent.current_role == SpecializationRole.EXPLORER:
            # Explorers emphasize exploration actions
            if base_decision in ["explore", "investigate"]:
                adapted_decision = "explore"
                confidence_boost = 0.15
            else:
                adapted_decision = base_decision
                confidence_boost = 0.0

        elif agent.current_role == SpecializationRole.SPECIALIST:
            # Specialists focus on optimization and specialization
            if base_decision in ["optimize", "specialize"]:
                adapted_decision = "specialize"
                confidence_boost = 0.12
            else:
                adapted_decision = base_decision
                confidence_boost = 0.0

        else:
            # Default adaptation for other roles
            adapted_decision = base_decision
            confidence_boost = 0.0

        # Capability-based adjustments
        capability_multiplier = (agent.capabilities.processing_power +
                               agent.capabilities.memory_capacity) / 2

        adapted_confidence = min(base_confidence + confidence_boost, 1.0) * capability_multiplier

        return {
            "action_type": adapted_decision,
            "confidence": adapted_confidence,
            "parameters": {
                "collective_origin": True,
                "base_decision": base_decision,
                "role_adaptation": agent.current_role.value if agent.current_role else None
            }
        }

    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend for the swarm"""
        if len(self.swarm_performance_history) < 10:
            return 0.0

        recent_performances = self.swarm_performance_history[-10:]
        if len(recent_performances) < 2:
            return 0.0

        # Simple linear trend calculation
        x = list(range(len(recent_performances)))
        y = recent_performances

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
        sum_x2 = sum(xi ** 2 for xi in x)

        # Linear regression slope
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _calculate_cycle_metrics(self, *phase_results) -> dict[str, float]:
        """Calculate comprehensive metrics for the swarm cycle"""
        perception_results, communication_results, specialization_results, \
        decision_results, emergence_results, execution_results, learning_results = phase_results

        # Perception metrics
        perception_quality = perception_results.get("average_perception_quality", 0.5)
        active_agents_ratio = perception_results.get("total_agents_active", 0) / max(len(self.agents), 1)

        # Communication metrics
        communication_efficiency = communication_results.get("exchange_rate", 0.0)

        # Specialization metrics
        specialization_diversity = specialization_results.get("specialization_diversity", 0.0)
        specialization_balance = specialization_results.get("specialization_balance", 0.0)

        # Decision metrics
        decision_confidence = decision_results.get("collective_decision", {}).get("confidence", 0.5)

        # Emergence metrics
        emergence_strength = 0.0
        if emergence_results.get("behaviors_detected", 0) > 0:
            behaviors = emergence_results.get("detected_behaviors", [])
            if behaviors:
                emergence_strength = float(np.mean([b.get("strength", 0.0) for b in behaviors]))

        # Execution metrics
        execution_performance = execution_results.get("average_performance", 0.5)
        execution_success_rate = execution_results.get("success_rate", 0.5)

        # Learning metrics
        adaptation_rate = learning_results.get("adaptation_rate", 0.0)
        performance_trend = learning_results.get("performance_trend", 0.0)

        # Overall performance calculation
        performance_components = [
            float(perception_quality) * 0.10,
            float(active_agents_ratio) * 0.05,
            float(communication_efficiency) * 0.15,
            float(specialization_diversity) * 0.10,
            float(specialization_balance) * 0.05,
            float(decision_confidence) * 0.15,
            float(emergence_strength) * 0.10,
            float(execution_performance) * 0.20,
            float(execution_success_rate) * 0.10
        ]
        overall_performance = float(np.mean(performance_components))

        return {
            "overall_performance": overall_performance,
            "perception_quality": perception_quality,
            "active_agents_ratio": active_agents_ratio,
            "communication_efficiency": communication_efficiency,
            "specialization_diversity": specialization_diversity,
            "specialization_balance": specialization_balance,
            "decision_confidence": decision_confidence,
            "emergence_strength": emergence_strength,
            "execution_performance": execution_performance,
            "execution_success_rate": execution_success_rate,
            "adaptation_rate": adaptation_rate,
            "performance_trend": performance_trend
        }

    def _get_swarm_state_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of current swarm state"""
        # Agent statistics
        agent_count = len(self.agents)
        role_distribution = defaultdict(int)
        avg_performance = 0.0
        total_experience = 0.0

        for agent in self.agents.values():
            if agent.current_role:
                role_distribution[agent.current_role.value] += 1
            avg_performance += agent.capabilities.get_avg_performance()
            total_experience += len(agent.interaction_history)

        if agent_count > 0:
            avg_performance /= agent_count
            avg_experience = total_experience / agent_count
        else:
            avg_experience = 0.0

        # Emergence statistics
        active_behaviors = len(self.current_emergent_behaviors)
        behavior_types = [b.behavior_type for b in self.current_emergent_behaviors]

        # Performance history statistics
        recent_performance_trend = self._calculate_performance_trend()

        return {
            "agent_count": agent_count,
            "role_distribution": dict(role_distribution),
            "average_performance": avg_performance,
            "average_experience": avg_experience,
            "active_emergent_behaviors": active_behaviors,
            "behavior_types": behavior_types,
            "performance_trend": recent_performance_trend,
            "swarm_active": self.is_active,
            "total_cycles": len(self.swarm_performance_history)
        }

    def get_swarm_analytics(self) -> dict[str, Any]:
        """Get comprehensive analytics about swarm performance and behavior"""
        return {
            "swarm_state": self._get_swarm_state_summary(),
            "current_metrics": self.swarm_metrics,
            "performance_history": self.swarm_performance_history[-50:],  # Last 50 cycles
            "role_assignment_history": self.role_assignment_history[-10:],  # Last 10 assignments
            "current_emergent_behaviors": [
                {
                    "id": behavior.behavior_id,
                    "type": behavior.behavior_type,
                    "strength": behavior.emergence_strength,
                    "stability": behavior.stability_score,
                    "participants": len(behavior.participating_agents),
                    "age": time.time() - behavior.detection_timestamp
                }
                for behavior in self.current_emergent_behaviors
            ],
            "cross_scale_interactions": self.cross_scale_interactions[-20:],  # Last 20 interactions
            "agent_summaries": {
                agent_id: {
                    "role": agent.current_role.value if agent.current_role else None,
                    "performance": agent.capabilities.get_avg_performance(),
                    "interactions": len(agent.interaction_history),
                    "modifications": len(agent.modification_history),
                    "trust_relationships": len(agent.peer_relationships)
                }
                for agent_id, agent in self.agents.items()
            }
        }
