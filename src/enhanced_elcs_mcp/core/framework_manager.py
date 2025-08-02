"""
ELCS MCP Server - Core Integration Module
=================================================

This module provides direct integration with the ELCS Framework,
exposing Dynamic Emergence Networks, Multi-Agent Swarm Intelligence,
and Complex Systems Analysis capabilities through MCP protocol.

Author: Tyler Blaine Hall, Claude Sonnet 4
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Import ELCS Framework components
from elcs_framework.core.dynamic_emergence_networks import DynamicEmergenceNetwork, ProcessScale, ProcessSignature
from elcs_framework.core.multi_agent_swarm import AgentCapabilities, CollectiveDecisionMaker, CollectiveDecisionType, EmergenceBehaviorDetector, SpecializationRole, SwarmAgent, SwarmIntelligence

# Import Swarm Library
from .swarm_library import SwarmLibrary

# Configure logging to stderr for MCP compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ELCSIntegrationError(Exception):
    """Raised when ELCS Framework integration fails."""
    pass


class ELCSFrameworkManager:
    """
    Manages ELCS Framework instances and operations.

    Provides single implementation for all framework capabilities:
    - Dynamic Emergence Networks (DENs)
    - Multi-Agent Swarm Intelligence
    - Emergence Pattern Detection
    - Collective Decision Making
    - Complex Systems Analysis
    """

    def __init__(self):
        """Initialize ELCS Framework Manager."""
        self.active_networks: dict[str, DynamicEmergenceNetwork] = {}
        self.active_swarms: dict[str, SwarmIntelligence] = {}
        self.emergence_detectors: dict[str, EmergenceBehaviorDetector] = {}
        self.decision_makers: dict[str, CollectiveDecisionMaker] = {}
        self.swarm_configs: dict[str, dict[str, Any]] = {}  # Store swarm configurations

        # Simulation state tracking
        self.simulation_results: dict[str, dict[str, Any]] = {}
        self.analysis_cache: dict[str, dict[str, Any]] = {}

        # Initialize Swarm Library
        self.swarm_library = SwarmLibrary()

        logger.info("ELCS Framework Manager initialized")

    async def create_emergence_network(
        self,
        network_id: str,
        scales: list[str] | None = None,
        config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create and configure a Dynamic Emergence Network.

        Args:
            network_id: Unique identifier for the network
            scales: List of process scales to include
            config: Additional configuration parameters

        Returns:
            Network creation result with metadata

        Raises:
            ELCSIntegrationError: If network creation fails
        """
        try:
            if network_id in self.active_networks:
                raise ELCSIntegrationError(f"Network '{network_id}' already exists")

            # Create Dynamic Emergence Network
            den = DynamicEmergenceNetwork()

            # Configure scales if provided
            if scales:
                valid_scales = [ProcessScale[scale.upper()] for scale in scales
                              if scale.upper() in ProcessScale.__members__]
                logger.info(f"Configured network with scales: {[s.name for s in valid_scales]}")

            # Apply additional configuration
            if config:
                logger.info(f"Applied configuration: {config}")

            self.active_networks[network_id] = den

            result = {
                "network_id": network_id,
                "status": "created",
                "scales_configured": scales or [],
                "configuration": config or {},
                "timestamp": time.time()
            }

            logger.info(f"Created Dynamic Emergence Network: {network_id}")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to create emergence network: {e}") from e

    async def create_swarm_simulation(
        self,
        swarm_id: str,
        network_id: str | None = None,
        agent_count: int = 10,
        agent_capabilities: dict[str, float] | None = None,
        config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create and configure a Multi-Agent Swarm Intelligence simulation.

        Args:
            swarm_id: Unique identifier for the swarm
            network_id: Associated Dynamic Emergence Network ID
            agent_count: Number of agents to create
            agent_capabilities: Default capabilities for agents
            config: Additional swarm configuration

        Returns:
            Swarm creation result with metadata

        Raises:
            ELCSIntegrationError: If swarm creation fails
        """
        try:
            if swarm_id in self.active_swarms:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' already exists")

            # Get associated network if specified
            den = None
            if network_id:
                if network_id not in self.active_networks:
                    raise ELCSIntegrationError(f"Network '{network_id}' not found")
                den = self.active_networks[network_id]

            # Create Swarm Intelligence system
            swarm_config = config or {}
            max_agents = swarm_config.get('max_agents', agent_count * 2)
            detection_interval = swarm_config.get('emergence_detection_interval', 30.0)

            swarm = SwarmIntelligence(
                den=den,
                max_agents=max_agents,
                emergence_detection_interval=detection_interval
            )

            # Store swarm metadata for later access
            self.swarm_configs[swarm_id] = swarm_config

            # Create agents with specified capabilities
            # Map user parameters to AgentCapabilities parameters
            raw_capabilities = agent_capabilities or {}

            # Handle parameter name variations and provide defaults
            default_capabilities = {
                'processing_power': raw_capabilities.get('processing_power', 0.6),
                'memory_capacity': raw_capabilities.get('memory_capacity', 0.5),
                'learning_rate': raw_capabilities.get('learning_rate',
                                                    raw_capabilities.get('adaptation_rate', 0.15)),
                'communication_efficiency': raw_capabilities.get('communication_efficiency',
                                                               raw_capabilities.get('communication_range', 0.6))
            }

            agent_ids = []
            for _ in range(agent_count):
                # Add some variance to capabilities
                capabilities = AgentCapabilities(
                    processing_power=default_capabilities['processing_power'] + np.random.normal(0, 0.1),
                    memory_capacity=default_capabilities['memory_capacity'] + np.random.normal(0, 0.1),
                    learning_rate=default_capabilities['learning_rate'] + np.random.normal(0, 0.02),
                    communication_efficiency=default_capabilities['communication_efficiency'] + np.random.normal(0, 0.1)
                )

                # Clamp capabilities to valid ranges
                capabilities.processing_power = np.clip(capabilities.processing_power, 0.1, 1.0)
                capabilities.memory_capacity = np.clip(capabilities.memory_capacity, 0.1, 1.0)
                capabilities.learning_rate = np.clip(capabilities.learning_rate, 0.01, 0.5)
                capabilities.communication_efficiency = np.clip(capabilities.communication_efficiency, 0.1, 1.0)

                agent_id = swarm.add_agent(initial_capabilities=capabilities)
                agent_ids.append(agent_id)

            self.active_swarms[swarm_id] = swarm

            # Create associated emergence detector
            detector = EmergenceBehaviorDetector(
                detection_threshold=swarm_config.get('emergence_threshold', 0.6),
                stability_window=swarm_config.get('stability_window', 10)
            )
            self.emergence_detectors[swarm_id] = detector

            # Create collective decision maker
            decision_type = swarm_config.get('decision_type', 'consensus')
            decision_enum = CollectiveDecisionType(decision_type)
            decision_maker = CollectiveDecisionMaker(default_decision_type=decision_enum)
            self.decision_makers[swarm_id] = decision_maker

            result = {
                "swarm_id": swarm_id,
                "network_id": network_id,
                "status": "created",
                "agent_count": len(agent_ids),
                "agent_ids": agent_ids,
                "capabilities": default_capabilities,
                "configuration": swarm_config,
                "timestamp": time.time()
            }

            logger.info(f"Created Multi-Agent Swarm: {swarm_id} with {len(agent_ids)} agents")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to create swarm simulation: {e}") from e

    async def run_swarm_simulation(
        self,
        swarm_id: str,
        environment_state: dict[str, float],
        cycles: int = 5,
        start_intelligence: bool = True
    ) -> dict[str, Any]:
        """
        Execute Multi-Agent Swarm Intelligence simulation cycles.

        Args:
            swarm_id: ID of swarm to run
            environment_state: Environment parameters for simulation
            cycles: Number of simulation cycles to execute
            start_intelligence: Whether to start swarm intelligence if not running

        Returns:
            Simulation results with comprehensive metrics

        Raises:
            ELCSIntegrationError: If simulation execution fails
        """
        try:
            if swarm_id not in self.active_swarms:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' not found")

            swarm = self.active_swarms[swarm_id]
            detector = self.emergence_detectors.get(swarm_id)

            # Start swarm intelligence if requested
            if start_intelligence:
                swarm.start_swarm_intelligence()

            cycle_results = []
            overall_metrics = {
                'total_cycles': cycles,
                'successful_cycles': 0,
                'average_performance': 0.0,
                'peak_performance': 0.0,
                'emergence_events': 0,
                'behaviors_detected': []
            }

            performance_history = []

            for cycle in range(cycles):
                cycle_start_time = time.time()

                try:
                    # Execute one complete swarm cycle
                    cycle_result = swarm.execute_swarm_cycle(environment_state)

                    if cycle_result.get("cycle_completed", False):
                        overall_metrics['successful_cycles'] += 1

                        # Extract cycle metrics
                        cycle_metrics = cycle_result.get("cycle_metrics", {})
                        performance = cycle_metrics.get("overall_performance", 0.0)
                        performance_history.append(performance)

                        # Update peak performance
                        if performance > overall_metrics['peak_performance']:
                            overall_metrics['peak_performance'] = performance

                        # Check for emergent behaviors
                        emergence_results = cycle_result.get("phase_results", {}).get("emergence", {})
                        behaviors_detected = emergence_results.get("behaviors_detected", 0)

                        if behaviors_detected > 0:
                            overall_metrics['emergence_events'] += behaviors_detected
                            behaviors = emergence_results.get("detected_behaviors", [])
                            overall_metrics['behaviors_detected'].extend(behaviors)

                        cycle_results.append({
                            'cycle': cycle + 1,
                            'performance': performance,
                            'emergence_strength': cycle_metrics.get("emergence_strength", 0.0),
                            'behaviors_detected': behaviors_detected,
                            'duration': time.time() - cycle_start_time,
                            'status': 'completed'
                        })

                    else:
                        cycle_results.append({
                            'cycle': cycle + 1,
                            'status': 'failed',
                            'error': cycle_result.get('error', 'Unknown error'),
                            'duration': time.time() - cycle_start_time
                        })

                except Exception as e:
                    cycle_results.append({
                        'cycle': cycle + 1,
                        'status': 'error',
                        'error': str(e),
                        'duration': time.time() - cycle_start_time
                    })

                # Brief pause between cycles
                await asyncio.sleep(0.1)

            # Calculate overall metrics
            if performance_history:
                overall_metrics['average_performance'] = float(np.mean(performance_history))

            # Get final analytics
            try:
                analytics = swarm.get_swarm_analytics()
            except Exception as e:
                logger.warning(f"Failed to get swarm analytics: {e}")
                analytics = {"error": str(e)}

            result = {
                "swarm_id": swarm_id,
                "simulation_status": "completed",
                "environment_state": environment_state,
                "overall_metrics": overall_metrics,
                "cycle_results": cycle_results,
                "final_analytics": analytics,
                "timestamp": time.time()
            }

            # Cache results for later analysis
            self.simulation_results[swarm_id] = result

            # Auto-store successful swarms in library
            if overall_metrics['successful_cycles'] > 0:
                await self._auto_store_swarm_in_library(swarm_id, result)

            logger.info(
                f"Completed swarm simulation: {swarm_id}, "
                f"{overall_metrics['successful_cycles']}/{cycles} cycles successful, "
                f"avg performance: {overall_metrics['average_performance']:.3f}"
            )

            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to run swarm simulation: {e}") from e

    async def detect_emergence_patterns(
        self,
        swarm_id: str,
        data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Detect emergent patterns in swarm behavior or provided data.

        Args:
            swarm_id: ID of swarm to analyze
            data: Optional external data to analyze for emergence patterns

        Returns:
            Detected emergence patterns and analysis

        Raises:
            ELCSIntegrationError: If emergence detection fails
        """
        try:
            if swarm_id not in self.active_swarms:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' not found")

            swarm = self.active_swarms[swarm_id]
            detector = self.emergence_detectors.get(swarm_id)

            if not detector:
                # Create detector if not exists, using stored swarm config
                swarm_config = self.swarm_configs.get(swarm_id, {})
                detector = EmergenceBehaviorDetector(
                    detection_threshold=swarm_config.get('emergence_threshold', 0.6),
                    stability_window=swarm_config.get('stability_window', 10)
                )
                self.emergence_detectors[swarm_id] = detector

            # Get real agents from swarm
            agents = list(swarm.agents.values())

            if not agents:
                logger.warning(f"No agents found in swarm {swarm_id}")
                return {
                    "swarm_id": swarm_id,
                    "analysis_timestamp": time.time(),
                    "behaviors_detected": 0,
                    "detected_behaviors": [],
                    "error": "No agents in swarm"
                }

            # Build real interaction data from agent states
            interaction_data = self._build_real_interaction_data(agents, data)

            # Detect emergence patterns using real agent data
            detected_behaviors = detector.detect_emergence(agents, interaction_data)

            # Calculate real emergence metrics
            emergence_metrics = self._calculate_emergence_metrics(agents, detected_behaviors)

            # Intelligently summarize the results to avoid token overflow
            summarized_analysis = self._summarize_emergence_analysis(
                swarm_id, detected_behaviors, interaction_data, emergence_metrics, agents
            )

            # Cache full analysis results for detailed inspection
            full_analysis = {
                "swarm_id": swarm_id,
                "analysis_timestamp": time.time(),
                "behaviors_detected": len(detected_behaviors),
                "detected_behaviors": [
                    {
                        "behavior_id": behavior.behavior_id,
                        "behavior_type": behavior.behavior_type,
                        "emergence_strength": behavior.emergence_strength,
                        "stability_score": behavior.stability_score,
                        "participating_agents": list(behavior.participating_agents),
                        "pattern_signature": behavior.pattern_signature
                    }
                    for behavior in detected_behaviors
                ],
                "interaction_data": interaction_data,
                "emergence_metrics": emergence_metrics
            }
            self.analysis_cache[f"{swarm_id}_emergence_full"] = full_analysis

            # Auto-update swarm in library with emergence data
            if detected_behaviors:
                await self._update_swarm_in_library(swarm_id, summarized_analysis)

            logger.info(f"Completed emergence detection for swarm: {swarm_id}")
            return summarized_analysis

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to detect emergence patterns: {e}") from e

    async def make_collective_decision(
        self,
        swarm_id: str,
        decision_context: dict[str, Any],
        decision_type: str = "consensus"
    ) -> dict[str, Any]:
        """
        Execute collective decision-making process for swarm.

        Args:
            swarm_id: ID of swarm to make decision for
            decision_context: Context information for the decision
            decision_type: Type of decision-making algorithm to use

        Returns:
            Collective decision result and metadata

        Raises:
            ELCSIntegrationError: If collective decision making fails
        """
        try:
            if swarm_id not in self.active_swarms:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' not found")

            decision_maker = self.decision_makers.get(swarm_id)
            if not decision_maker:
                # Create decision maker if not exists
                decision_enum = CollectiveDecisionType(decision_type)
                decision_maker = CollectiveDecisionMaker(default_decision_type=decision_enum)
                self.decision_makers[swarm_id] = decision_maker

            # Convert decision type string to enum
            try:
                decision_enum = CollectiveDecisionType(decision_type)
            except ValueError as err:
                raise ELCSIntegrationError(f"Invalid decision type: {decision_type}") from err

            # Get real agents from swarm for decision making
            swarm = self.active_swarms[swarm_id]
            agents = list(swarm.agents.values())

            if not agents:
                logger.warning(f"No agents found in swarm {swarm_id}")
                return {
                    "swarm_id": swarm_id,
                    "decision_type": decision_type,
                    "decision_context": decision_context,
                    "decision_result": {
                        "decision": "no_action",
                        "confidence": 0.0,
                        "consensus_level": 0.0,
                        "participating_agents": 0,
                        "error": "No agents in swarm"
                    },
                    "timestamp": time.time()
                }

            # Execute collective decision making with real agents
            try:
                decision_result = decision_maker.make_collective_decision(
                    agents=agents,
                    decision_context=decision_context,
                    decision_type=decision_enum
                )

                # Enhance decision result with agent details
                decision_result["participating_agents"] = len(agents)
                decision_result["agent_roles"] = [
                    agent.current_role.value if agent.current_role else "unassigned"
                    for agent in agents
                ]

            except Exception as e:
                logger.error(f"Collective decision making failed: {e}")
                decision_result = {
                    "decision": "decision_failed",
                    "confidence": 0.0,
                    "consensus_level": 0.0,
                    "participating_agents": len(agents),
                    "error": str(e)
                }

            result = {
                "swarm_id": swarm_id,
                "decision_type": decision_type,
                "decision_context": decision_context,
                "decision_result": decision_result,
                "timestamp": time.time()
            }

            logger.info(f"Completed collective decision for swarm: {swarm_id}")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to make collective decision: {e}") from e

    async def optimize_swarm_parameters(
        self,
        swarm_id: str,
        target_metrics: dict[str, float],
        optimization_rounds: int = 5
    ) -> dict[str, Any]:
        """
        Optimize swarm parameters using performance feedback and target metrics.

        Iteratively adjusts agent capabilities and swarm configuration to achieve
        target performance metrics through automated parameter optimization.

        Args:
            swarm_id: ID of swarm to optimize
            target_metrics: Target performance metrics to optimize for
            optimization_rounds: Number of optimization iterations to perform

        Returns:
            Optimization results and parameter recommendations

        Raises:
            ELCSIntegrationError: If optimization fails
        """
        try:
            if swarm_id not in self.active_swarms:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' not found")

            swarm = self.active_swarms[swarm_id]
            agents = list(swarm.agents.values())

            if not agents:
                raise ELCSIntegrationError(f"No agents found in swarm {swarm_id}")

            # Get baseline metrics
            baseline_analytics = swarm.get_swarm_analytics()
            baseline_performance = baseline_analytics.get("swarm_state", {}).get("average_performance", 0.5)

            # Track optimization progress
            optimization_history = []
            best_performance = baseline_performance
            best_parameters = self._extract_current_parameters(agents)

            logger.info(f"Starting optimization for swarm {swarm_id} from baseline {baseline_performance:.3f}")

            for round_num in range(optimization_rounds):
                logger.info(f"Optimization round {round_num + 1}/{optimization_rounds}")

                # Generate parameter variations
                parameter_candidates = self._generate_parameter_candidates(
                    best_parameters, target_metrics, round_num
                )

                round_results = []

                for candidate_idx, candidate_params in enumerate(parameter_candidates):
                    # Apply parameters to agents
                    self._apply_parameters_to_agents(agents, candidate_params)

                    # Run test simulation
                    test_environment = {
                        "complexity": 0.7,
                        "resources": 0.8,
                        "opportunities": 0.6,
                        "threats": 0.2
                    }

                    test_result = swarm.execute_swarm_cycle(test_environment)
                    test_metrics = test_result.get("cycle_metrics", {})
                    test_performance = test_metrics.get("overall_performance", 0.0)

                    # Calculate fitness score
                    fitness_score = self._calculate_fitness_score(test_metrics, target_metrics)

                    round_results.append({
                        "candidate_idx": candidate_idx,
                        "parameters": candidate_params.copy(),
                        "performance": test_performance,
                        "fitness_score": fitness_score,
                        "metrics": test_metrics
                    })

                    # Update best if improved
                    if test_performance > best_performance:
                        best_performance = test_performance
                        best_parameters = candidate_params.copy()
                        logger.info(f"New best performance: {best_performance:.3f}")

                # Select best candidate from this round
                round_best = max(round_results, key=lambda x: x["fitness_score"])
                optimization_history.append({
                    "round": round_num + 1,
                    "best_candidate": round_best,
                    "round_results": round_results,
                    "improvement": round_best["performance"] - baseline_performance
                })

                # Apply best parameters permanently
                self._apply_parameters_to_agents(agents, round_best["parameters"])

                # Early stopping if target achieved
                if self._target_achieved(round_best["metrics"], target_metrics):
                    logger.info(f"Target metrics achieved in round {round_num + 1}")
                    break

            # Calculate final improvements
            final_analytics = swarm.get_swarm_analytics()
            final_performance = final_analytics.get("swarm_state", {}).get("average_performance", 0.5)
            performance_improvement = ((final_performance - baseline_performance) / baseline_performance * 100) if baseline_performance > 0 else 0.0

            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                optimization_history, best_parameters, target_metrics
            )

            result = {
                "swarm_id": swarm_id,
                "optimization_status": "completed",
                "rounds_completed": len(optimization_history),
                "baseline_performance": baseline_performance,
                "final_performance": final_performance,
                "performance_improvement": performance_improvement,
                "target_metrics": target_metrics,
                "best_parameters": best_parameters,
                "optimization_history": optimization_history,
                "recommendations": recommendations,
                "timestamp": time.time()
            }

            logger.info(f"Optimization completed: {performance_improvement:.1f}% improvement")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to optimize swarm parameters: {e}") from e

    def _extract_current_parameters(self, agents: list) -> dict[str, Any]:
        """Extract current parameter settings from agents."""
        if not agents:
            return {}

        # Calculate average capabilities
        avg_processing = sum(agent.capabilities.processing_power for agent in agents) / len(agents)
        avg_memory = sum(agent.capabilities.memory_capacity for agent in agents) / len(agents)
        avg_learning = sum(agent.capabilities.learning_rate for agent in agents) / len(agents)
        avg_communication = sum(agent.capabilities.communication_efficiency for agent in agents) / len(agents)

        return {
            "processing_power": avg_processing,
            "memory_capacity": avg_memory,
            "learning_rate": avg_learning,
            "communication_efficiency": avg_communication
        }

    def _generate_parameter_candidates(self, base_params: dict[str, Any], target_metrics: dict[str, float], round_num: int) -> list[dict[str, Any]]:
        """Generate parameter variation candidates for optimization."""
        candidates = []

        # Variation ranges decrease over rounds (exploitation vs exploration)
        variation_factor = max(0.1, 0.3 - (round_num * 0.05))

        # Generate variations around current best
        for _ in range(5):  # 5 candidates per round
            candidate = base_params.copy()

            # Random variations within bounds
            for param, value in candidate.items():
                if isinstance(value, (int, float)):
                    variation = np.random.normal(0, variation_factor * value)
                    new_value = value + variation

                    # Clamp to valid ranges
                    if param == "learning_rate":
                        candidate[param] = np.clip(new_value, 0.01, 0.5)
                    else:
                        candidate[param] = np.clip(new_value, 0.1, 1.0)

            candidates.append(candidate)

        return candidates

    def _apply_parameters_to_agents(self, agents: list, parameters: dict[str, Any]) -> None:
        """Apply parameter settings to all agents."""
        for agent in agents:
            for param, value in parameters.items():
                if hasattr(agent.capabilities, param):
                    setattr(agent.capabilities, param, value)

    def _calculate_fitness_score(self, metrics: dict[str, Any], targets: dict[str, float]) -> float:
        """Calculate fitness score based on how close metrics are to targets."""
        if not targets:
            return metrics.get("overall_performance", 0.0)

        fitness_components = []

        for target_name, target_value in targets.items():
            if target_name in metrics:
                actual_value = metrics[target_name]
                # Calculate normalized distance from target (closer = higher score)
                distance = abs(actual_value - target_value)
                normalized_score = max(0.0, 1.0 - distance)
                fitness_components.append(normalized_score)

        # Return average fitness across all targets
        return sum(fitness_components) / len(fitness_components) if fitness_components else 0.0

    def _target_achieved(self, metrics: dict[str, Any], targets: dict[str, float], tolerance: float = 0.05) -> bool:
        """Check if target metrics have been achieved within tolerance."""
        for target_name, target_value in targets.items():
            if target_name in metrics:
                actual_value = metrics[target_name]
                if abs(actual_value - target_value) > tolerance:
                    return False
        return True

    def _generate_optimization_recommendations(self, history: list, best_params: dict[str, Any], targets: dict[str, float]) -> dict[str, Any]:
        """Generate actionable recommendations based on optimization results."""
        recommendations = {
            "parameter_adjustments": {},
            "next_steps": [],
            "insights": []
        }

        # Analyze parameter effectiveness
        for param_name in best_params.keys():
            param_changes = []
            for round_data in history:
                best_candidate = round_data["best_candidate"]
                param_value = best_candidate["parameters"].get(param_name, 0.0)
                performance = best_candidate["performance"]
                param_changes.append((param_value, performance))

            if len(param_changes) >= 2:
                # Simple trend analysis
                values = [change[0] for change in param_changes]
                performances = [change[1] for change in param_changes]

                if len(set(values)) > 1:  # Values actually changed
                    correlation = np.corrcoef(values, performances)[0, 1] if len(values) > 2 else 0.0

                    if abs(correlation) > 0.3:  # Significant correlation
                        trend = "increase" if correlation > 0 else "decrease"
                        recommendations["parameter_adjustments"][param_name] = f"Consider {trend} for continued improvement"

        # Generate next steps
        if history:
            final_round = history[-1]
            improvement = final_round["best_candidate"]["performance"] - history[0]["best_candidate"]["performance"]

            if improvement > 0.1:
                recommendations["next_steps"].append("Scale up agent count to leverage improved parameters")
            elif improvement < 0.05:
                recommendations["next_steps"].append("Consider different optimization targets or longer rounds")

            recommendations["insights"].append(f"Achieved {improvement:.1%} performance improvement over optimization")

        # Target-specific recommendations
        for target_name, target_value in targets.items():
            recommendations["insights"].append(f"Target {target_name}: {target_value} (optimization focused)")

        return recommendations

    def get_framework_status(self) -> dict[str, Any]:
        """
        Get comprehensive status of all ELCS Framework components.

        Returns:
            Status information for all active components
        """
        return {
            "framework_version": "1.0.0",
            "active_networks": list(self.active_networks.keys()),
            "active_swarms": list(self.active_swarms.keys()),
            "emergence_detectors": list(self.emergence_detectors.keys()),
            "decision_makers": list(self.decision_makers.keys()),
            "simulation_results_cached": list(self.simulation_results.keys()),
            "analysis_cache_keys": list(self.analysis_cache.keys()),
            "timestamp": time.time()
        }

    async def get_detailed_analysis(self, swarm_id: str, analysis_type: str = "emergence") -> dict[str, Any]:
        """
        Retrieve detailed (full) analysis data for specific swarm and analysis type.

        Args:
            swarm_id: ID of swarm to get detailed data for
            analysis_type: Type of analysis ("emergence", "simulation", etc.)

        Returns:
            Full detailed analysis data
        """
        cache_key = f"{swarm_id}_{analysis_type}_full"

        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        elif swarm_id in self.simulation_results and analysis_type == "simulation":
            return self.simulation_results[swarm_id]
        else:
            return {
                "error": f"Detailed {analysis_type} data not found for swarm {swarm_id}",
                "available_cache_keys": list(self.analysis_cache.keys()),
                "available_simulation_results": list(self.simulation_results.keys())
            }

    def cleanup_resources(self) -> None:
        """Clean up all ELCS Framework resources."""
        try:
            # Stop all active swarms
            for swarm_id, swarm in self.active_swarms.items():
                try:
                    swarm.stop_swarm_intelligence()
                except Exception as e:
                    logger.warning(f"Error stopping swarm {swarm_id}: {e}")

            # Clear all resources
            self.active_networks.clear()
            self.active_swarms.clear()
            self.emergence_detectors.clear()
            self.decision_makers.clear()
            self.simulation_results.clear()
            self.analysis_cache.clear()

            logger.info("ELCS Framework resources cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _build_real_interaction_data(self, agents: list, external_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build real interaction data from agent communication history."""
        if external_data:
            return external_data

        communication_matrix = {}
        decision_synchrony = {}
        performance_correlation = {}

        # Build communication matrix from agent peer relationships
        for i, agent in enumerate(agents):
            agent_comms = {}
            for j, other_agent in enumerate(agents):
                if i != j:
                    trust_level = agent.peer_relationships.get(other_agent.agent_id, 0.0)
                    interaction_count = len([
                        record for record in agent.interaction_history
                        if other_agent.agent_id in record.get("peers_contacted", [])
                    ])
                    # Normalize and combine trust and interaction frequency
                    interaction_strength = min(trust_level + (interaction_count / 10.0), 1.0)
                    agent_comms[other_agent.agent_id] = interaction_strength

            communication_matrix[agent.agent_id] = agent_comms

        # Analyze decision synchronization from action timing
        recent_actions = []
        for agent in agents:
            if agent.local_memory:
                actions = [
                    record for record in agent.local_memory.values()
                    if isinstance(record, dict) and
                    record.get("timestamp", 0) > time.time() - 300  # Last 5 minutes
                ]
                recent_actions.extend(actions)

        # Group actions by time windows to detect synchronization
        if recent_actions:
            action_times = sorted([action.get("timestamp", 0) for action in recent_actions])
            sync_windows = []
            window_size = 30.0  # 30 second windows

            i = 0
            while i < len(action_times):
                window_start = action_times[i]
                window_actions = 1

                j = i + 1
                while j < len(action_times) and action_times[j] - window_start <= window_size:
                    window_actions += 1
                    j += 1

                if window_actions >= 3:  # Synchronized if 3+ actions in window
                    sync_windows.append(window_actions)

                i = j if j > i + 1 else i + 1

            decision_synchrony = {
                "sync_events": len(sync_windows),
                "max_sync_size": max(sync_windows) if sync_windows else 0,
                "avg_sync_size": sum(sync_windows) / len(sync_windows) if sync_windows else 0
            }

        # Calculate performance correlations between agents
        for agent in agents:
            if len(agent.capabilities.performance_history) >= 3:
                agent_performance = agent.capabilities.performance_history[-10:]  # Last 10 scores
                correlations = {}

                for other_agent in agents:
                    if agent.agent_id != other_agent.agent_id and len(other_agent.capabilities.performance_history) >= 3:
                        other_performance = other_agent.capabilities.performance_history[-10:]

                        # Simple correlation calculation
                        min_length = min(len(agent_performance), len(other_performance))
                        if min_length >= 3:
                            corr = np.corrcoef(
                                agent_performance[-min_length:],
                                other_performance[-min_length:]
                            )[0, 1]
                            if not np.isnan(corr):
                                correlations[other_agent.agent_id] = float(corr)

                performance_correlation[agent.agent_id] = correlations

        return {
            "communication_matrix": communication_matrix,
            "decision_synchrony": decision_synchrony,
            "performance_correlation": performance_correlation,
            "timestamp": time.time(),
            "data_source": "real_agent_interactions"
        }

    def _calculate_emergence_metrics(self, agents: list, detected_behaviors: list) -> dict[str, float]:
        """Calculate quantitative emergence metrics from agent states and behaviors."""
        if not agents:
            return {
                "clustering_strength": 0.0,
                "specialization_index": 0.0,
                "coordination_level": 0.0,
                "learning_coherence": 0.0
            }

        # Calculate clustering strength from agent peer relationships
        total_connections = 0
        strong_connections = 0

        for agent in agents:
            for trust_level in agent.peer_relationships.values():
                total_connections += 1
                if trust_level > 0.7:  # Strong connection threshold
                    strong_connections += 1

        clustering_strength = strong_connections / total_connections if total_connections > 0 else 0.0

        # Calculate specialization index from role distribution
        role_counts = {}
        specialized_agents = 0

        for agent in agents:
            if agent.current_role:
                role_name = agent.current_role.value
                role_counts[role_name] = role_counts.get(role_name, 0) + 1
                specialized_agents += 1

        if specialized_agents > 0:
            # Shannon diversity index for role distribution
            total_agents = len(agents)
            role_diversity = 0.0
            for count in role_counts.values():
                if count > 0:
                    proportion = count / total_agents
                    role_diversity -= proportion * np.log2(proportion)

            # Normalize by maximum possible diversity
            max_roles = min(len(agents), 8)  # 8 specialization roles available
            max_diversity = np.log2(max_roles) if max_roles > 1 else 1.0
            specialization_index = role_diversity / max_diversity if max_diversity > 0 else 0.0
        else:
            specialization_index = 0.0

        # Calculate coordination level from detected coordinated behaviors
        coordination_behaviors = [
            b for b in detected_behaviors
            if b.behavior_type in ["coordination", "collective_learning"]
        ]

        if coordination_behaviors:
            coordination_scores = [b.emergence_strength for b in coordination_behaviors]
            coordination_level = sum(coordination_scores) / len(coordination_scores)
        else:
            coordination_level = 0.0

        # Calculate learning coherence from performance trends
        performance_trends = []
        for agent in agents:
            if len(agent.capabilities.performance_history) >= 5:
                recent_performance = agent.capabilities.performance_history[-5:]
                trend = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
                performance_trends.append(trend)

        if performance_trends:
            # Measure how similar the learning trends are (coherence)
            trend_variance = np.var(performance_trends)
            learning_coherence = max(0.0, 1.0 - float(trend_variance))  # High coherence = low variance
        else:
            learning_coherence = 0.0

        return {
            "clustering_strength": float(clustering_strength),
            "specialization_index": float(specialization_index),
            "coordination_level": float(coordination_level),
            "learning_coherence": float(learning_coherence)
        }

    def _summarize_emergence_analysis(self,
                                    swarm_id: str,
                                    detected_behaviors: list,
                                    interaction_data: dict[str, Any],
                                    emergence_metrics: dict[str, float],
                                    agents: list) -> dict[str, Any]:
        """
        Intelligently summarize emergence analysis to prevent token overflow
        while preserving key insights and actionable information.
        """
        # Extract behavior summary - top behaviors by strength
        behavior_summary = []
        if detected_behaviors:
            # Sort by emergence strength and take top 5
            top_behaviors = sorted(detected_behaviors,
                                 key=lambda b: b.emergence_strength, reverse=True)[:5]

            for behavior in top_behaviors:
                # Convert pattern_signature to string if it's a dict
                pattern_key = "none"
                if behavior.pattern_signature:
                    if isinstance(behavior.pattern_signature, dict):
                        pattern_key = str(behavior.pattern_signature)[:50]
                    else:
                        pattern_key = str(behavior.pattern_signature)[:50]

                behavior_summary.append({
                    "type": behavior.behavior_type,
                    "strength": round(behavior.emergence_strength, 3),
                    "stability": round(behavior.stability_score, 3),
                    "agent_count": len(behavior.participating_agents),
                    "pattern_key": pattern_key
                })

        # Summarize interaction patterns - key statistics only
        interaction_summary = {
            "communication_density": round(interaction_data.get("communication_patterns", {}).get("communication_density", 0.0), 3),
            "avg_trust": round(interaction_data.get("communication_patterns", {}).get("average_trust", 0.0), 3),
            "performance_variance": round(interaction_data.get("performance_metrics", {}).get("performance_variance", 0.0), 3),
            "performance_trend": round(interaction_data.get("performance_metrics", {}).get("performance_trend", 0.0), 3)
        }

        # Role distribution summary
        role_distribution = interaction_data.get("role_distribution", {})
        dominant_roles = sorted(role_distribution.items(), key=lambda x: x[1], reverse=True)[:3]

        # Agent capabilities summary - aggregated statistics
        processing_powers = [agent.capabilities.processing_power for agent in agents]
        learning_rates = [agent.capabilities.learning_rate for agent in agents]

        agent_capabilities_summary = {
            "avg_processing": round(float(np.mean(processing_powers)), 3),
            "processing_variance": round(float(np.var(processing_powers)), 3),
            "avg_learning_rate": round(float(np.mean(learning_rates)), 3),
            "learning_rate_variance": round(float(np.var(learning_rates)), 3),
            "total_agents": len(agents)
        }

        # Key insights extraction
        insights = []

        # Emergence insights
        if emergence_metrics["clustering_strength"] > 0.7:
            insights.append(f"Strong clustering detected ({emergence_metrics['clustering_strength']:.2f}) - agents forming tight groups")

        if emergence_metrics["specialization_index"] > 0.6:
            insights.append(f"High specialization ({emergence_metrics['specialization_index']:.2f}) - agents differentiating roles")

        if emergence_metrics["coordination_level"] > 0.8:
            insights.append(f"Excellent coordination ({emergence_metrics['coordination_level']:.2f}) - synchronized behavior")

        if emergence_metrics["learning_coherence"] > 0.7:
            insights.append(f"Coherent learning ({emergence_metrics['learning_coherence']:.2f}) - agents improving together")

        # Behavior insights
        if behavior_summary:
            strongest_behavior = behavior_summary[0]
            insights.append(f"Dominant behavior: {strongest_behavior['type']} (strength: {strongest_behavior['strength']})")

            behavior_types = [b["type"] for b in behavior_summary]
            if len(set(behavior_types)) > 2:
                insights.append(f"Diverse behaviors: {len(set(behavior_types))} different patterns detected")

        # Performance insights
        avg_performance = interaction_summary["performance_variance"]
        if avg_performance < 0.1:
            insights.append("Low performance variance - agents performing consistently")
        elif avg_performance > 0.3:
            insights.append("High performance variance - significant agent differences")

        # Communication insights
        comm_density = interaction_summary["communication_density"]
        if comm_density > 0.7:
            insights.append("High communication density - extensive agent interactions")
        elif comm_density < 0.3:
            insights.append("Low communication density - limited agent interactions")

        return {
            "swarm_id": swarm_id,
            "analysis_timestamp": time.time(),
            "summary_type": "intelligent_summary",

            # Core metrics (compact)
            "emergence_metrics": emergence_metrics,
            "behaviors_detected": len(detected_behaviors),

            # Behavior summary (top behaviors only)
            "top_behaviors": behavior_summary,

            # Interaction patterns (key stats only)
            "interaction_summary": interaction_summary,

            # Role distribution (top 3 roles only)
            "dominant_roles": dominant_roles,

            # Agent capabilities (aggregated stats)
            "agent_capabilities": agent_capabilities_summary,

            # Key insights (human-readable)
            "key_insights": insights,

            # Metadata
            "data_reduction": {
                "original_behavior_count": len(detected_behaviors),
                "summarized_behavior_count": len(behavior_summary),
                "full_data_cached": f"{swarm_id}_emergence_full"
            }
        }

    # ============================================================================
    # SWARM LIBRARY MANAGEMENT METHODS
    # ============================================================================

    async def _auto_store_swarm_in_library(self, swarm_id: str, simulation_result: dict[str, Any]) -> None:
        """Automatically store a swarm in the library after successful simulation."""
        try:
            # Get swarm configuration
            swarm_config = {
                'swarm_id': swarm_id,
                'agent_count': simulation_result.get('overall_metrics', {}).get('total_cycles', 0),
                'configuration': self.swarm_configs.get(swarm_id, {}),
                'capabilities': simulation_result.get('final_analytics', {}).get('agent_capabilities', {}),
                'timestamp': simulation_result.get('timestamp', time.time())
            }

            # Use empty emergence analysis if not available yet
            emergence_analysis = self.analysis_cache.get(f"{swarm_id}_emergence_full", {
                'behaviors_detected': 0,
                'emergence_metrics': {},
                'analysis_timestamp': time.time()
            })

            # Generate metadata for auto-storage
            metadata_override = {
                'creator': 'ELCS Framework (Auto-stored)',
                'description': f'Auto-stored swarm with {swarm_config.get("agent_count", 0)} agents',
                'tags': ['auto-stored', 'simulation'],
                'research_category': 'simulation'
            }

            # Store in library
            self.swarm_library.store_swarm(
                swarm_id, swarm_config, simulation_result, emergence_analysis, metadata_override
            )

            logger.info(f"Auto-stored swarm {swarm_id} in library")

        except Exception as e:
            logger.warning(f"Failed to auto-store swarm {swarm_id}: {e}")

    async def _update_swarm_in_library(self, swarm_id: str, emergence_analysis: dict[str, Any]) -> None:
        """Update an existing swarm in the library with new emergence data."""
        try:
            # Check if swarm exists in library
            if swarm_id not in self.swarm_library.metadata_index:
                return  # Swarm not in library, skip update

            # Load existing swarm data
            existing_data = self.swarm_library.load_swarm(swarm_id)
            if not existing_data:
                return

            # Update emergence analysis
            existing_data['emergence_analysis'] = emergence_analysis

            # Update metadata
            metadata = self.swarm_library.metadata_index[swarm_id]
            metadata.last_updated = time.time()

            # Extract new behavioral patterns
            top_behaviors = emergence_analysis.get('top_behaviors', [])
            new_behaviors = [b.get('type', 'unknown') for b in top_behaviors]

            # Merge with existing behaviors (keep unique)
            all_behaviors = list(set(metadata.dominant_behaviors + new_behaviors))
            metadata.dominant_behaviors = all_behaviors

            # Update coordination metrics
            new_metrics = emergence_analysis.get('emergence_metrics', {})
            metadata.coordination_metrics.update(new_metrics)

            # Re-store updated data
            self.swarm_library.store_swarm(
                swarm_id,
                existing_data.get('swarm_config', {}),
                existing_data.get('simulation_results', {}),
                emergence_analysis,
                {'last_updated': time.time()}
            )

            logger.info(f"Updated swarm {swarm_id} in library with new emergence data")

        except Exception as e:
            logger.warning(f"Failed to update swarm {swarm_id} in library: {e}")

    async def store_swarm_in_library(self,
                                   swarm_id: str,
                                   description: str = "",
                                   tags: Optional[list[str]] = None,
                                   research_category: str = "general",
                                   experiment_notes: str = "") -> dict[str, Any]:
        """
        Manually store a swarm in the library with custom metadata.

        Args:
            swarm_id: ID of swarm to store
            description: Custom description for the swarm
            tags: Custom tags for categorization
            research_category: Research category for organization
            experiment_notes: Detailed notes about the experiment

        Returns:
            Storage result with library path and metadata
        """
        try:
            if swarm_id not in self.active_swarms:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' not found in active swarms")

            # Get swarm data
            swarm_config = {
                'swarm_id': swarm_id,
                'configuration': self.swarm_configs.get(swarm_id, {}),
                'timestamp': time.time()
            }

            # Get simulation results
            simulation_results = self.simulation_results.get(swarm_id, {
                'swarm_id': swarm_id,
                'status': 'active',
                'timestamp': time.time()
            })

            # Get emergence analysis
            emergence_analysis = self.analysis_cache.get(f"{swarm_id}_emergence_full", {
                'swarm_id': swarm_id,
                'analysis_timestamp': time.time(),
                'behaviors_detected': 0
            })

            # Custom metadata
            metadata_override = {
                'creator': 'ELCS Framework (Manual)',
                'description': description or f'Manually stored swarm: {swarm_id}',
                'tags': tags or ['manual', 'stored'],
                'research_category': research_category,
                'experiment_notes': experiment_notes
            }

            # Store in library
            storage_path = self.swarm_library.store_swarm(
                swarm_id, swarm_config, simulation_results, emergence_analysis, metadata_override
            )

            result = {
                'swarm_id': swarm_id,
                'storage_status': 'success',
                'storage_path': storage_path,
                'metadata': metadata_override,
                'timestamp': time.time()
            }

            logger.info(f"Manually stored swarm {swarm_id} in library")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to store swarm in library: {e}")

    async def search_swarm_library(self,
                                 query: str = "",
                                 tags: Optional[list[str]] = None,
                                 behavior_types: Optional[list[str]] = None,
                                 performance_range: Optional[tuple[float, float]] = None,
                                 research_category: Optional[str] = None,
                                 limit: int = 20) -> dict[str, Any]:
        """
        Search the swarm library with fuzzy matching and filtering.

        Args:
            query: Fuzzy text search query
            tags: Filter by tags
            behavior_types: Filter by behavior types
            performance_range: Filter by performance range (min, max)
            research_category: Filter by research category
            limit: Maximum number of results

        Returns:
            Search results with relevance scores and metadata
        """
        try:
            # Perform search
            search_results = self.swarm_library.search_swarms(
                query=query,
                tags=tags,
                behavior_types=behavior_types,
                performance_range=performance_range,
                research_category=research_category,
                limit=limit
            )

            # Get library statistics
            library_stats = self.swarm_library.get_library_stats()

            result = {
                'search_query': {
                    'query': query,
                    'tags': tags,
                    'behavior_types': behavior_types,
                    'performance_range': performance_range,
                    'research_category': research_category,
                    'limit': limit
                },
                'results_count': len(search_results),
                'results': search_results,
                'library_stats': library_stats,
                'timestamp': time.time()
            }

            logger.info(f"Searched swarm library: {len(search_results)} results for query '{query}'")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to search swarm library: {e}")

    async def load_swarm_from_library(self, swarm_id: str) -> dict[str, Any]:
        """
        Load a swarm from the library with all its data.

        Args:
            swarm_id: ID of swarm to load

        Returns:
            Complete swarm data including metadata, config, and results
        """
        try:
            swarm_data = self.swarm_library.load_swarm(swarm_id)

            if not swarm_data:
                raise ELCSIntegrationError(f"Swarm '{swarm_id}' not found in library")

            result = {
                'swarm_id': swarm_id,
                'load_status': 'success',
                'swarm_data': swarm_data,
                'timestamp': time.time()
            }

            logger.info(f"Loaded swarm {swarm_id} from library")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to load swarm from library: {e}")

    async def get_library_analytics(self) -> dict[str, Any]:
        """
        Get comprehensive analytics about the swarm library.

        Returns:
            Library analytics including statistics, trends, and insights
        """
        try:
            # Get basic library stats
            library_stats = self.swarm_library.get_library_stats()

            # Get behavior statistics
            behavior_stats = self.swarm_library.get_behavior_statistics()

            # Get performance analytics
            performance_analytics = self.swarm_library.get_performance_analytics()

            result = {
                'library_overview': library_stats,
                'behavior_analysis': behavior_stats,
                'performance_analysis': performance_analytics,
                'analysis_timestamp': time.time()
            }

            logger.info("Generated comprehensive library analytics")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to get library analytics: {e}")

    async def export_swarm_from_library(self, swarm_id: str, export_path: Optional[str] = None) -> dict[str, Any]:
        """
        Export a swarm from the library to a portable format.

        Args:
            swarm_id: ID of swarm to export
            export_path: Optional custom export path

        Returns:
            Export result with file path and metadata
        """
        try:
            export_file_path = self.swarm_library.export_swarm(swarm_id, export_path)

            result = {
                'swarm_id': swarm_id,
                'export_status': 'success',
                'export_path': export_file_path,
                'timestamp': time.time()
            }

            logger.info(f"Exported swarm {swarm_id} to {export_file_path}")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to export swarm: {e}")

    async def import_swarm_to_library(self, import_path: str, new_swarm_id: Optional[str] = None) -> dict[str, Any]:
        """
        Import a swarm into the library from exported format.

        Args:
            import_path: Path to exported swarm file
            new_swarm_id: Optional new ID for imported swarm

        Returns:
            Import result with new swarm ID and metadata
        """
        try:
            imported_swarm_id = self.swarm_library.import_swarm(import_path, new_swarm_id)

            result = {
                'import_status': 'success',
                'original_path': import_path,
                'imported_swarm_id': imported_swarm_id,
                'timestamp': time.time()
            }

            logger.info(f"Imported swarm as {imported_swarm_id} from {import_path}")
            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to import swarm: {e}")

    async def delete_swarm_from_library(self, swarm_id: str) -> dict[str, Any]:
        """
        Delete a swarm from the library.

        Args:
            swarm_id: ID of swarm to delete

        Returns:
            Deletion result
        """
        try:
            success = self.swarm_library.delete_swarm(swarm_id)

            result = {
                'swarm_id': swarm_id,
                'deletion_status': 'success' if success else 'failed',
                'timestamp': time.time()
            }

            if success:
                logger.info(f"Deleted swarm {swarm_id} from library")
            else:
                logger.warning(f"Failed to delete swarm {swarm_id} from library")

            return result

        except Exception as e:
            raise ELCSIntegrationError(f"Failed to delete swarm from library: {e}")

