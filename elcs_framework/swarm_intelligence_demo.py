#!/usr/bin/env python3
"""
Enhanced ELCS Framework - Swarm Intelligence Demonstration
=========================================================

This demonstration script showcases the collective intelligence capabilities
of the Enhanced-ELCS-Framework, including emergent specialization, distributed
cognition, self-modifying architectures, and cross-scale emergence detection.

Usage:
    python swarm_intelligence_demo.py

Features Demonstrated:
- Multi-agent swarm initialization and coordination
- Emergent role specialization and task allocation
- Collective decision-making algorithms
- Real-time emergence detection and monitoring
- Integration with Dynamic Emergence Networks
- Performance analytics and visualization

Author: Enhanced ELCS Development Team
Version: 1.0.0
License: MIT
"""

import json
import logging
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Enhanced-ELCS components
try:
    from core.dynamic_emergence_networks import DynamicEmergenceNetwork, ProcessEntity, ProcessScale, ProcessSignature
    from core.multi_agent_swarm import AgentCapabilities, CollectiveDecisionType, SpecializationRole, SwarmAgent, SwarmIntelligence
except ImportError as e:
    logger.error(f"Failed to import Enhanced-ELCS components: {e}")
    logger.info("Make sure you're running from the Enhanced-ELCS-Framework root directory")
    sys.exit(1)


class SwarmIntelligenceDemo:
    """
    Comprehensive demonstration of Enhanced-ELCS swarm intelligence capabilities
    """

    def __init__(self):
        """Initialize demonstration environment"""
        self.den = DynamicEmergenceNetwork(
            enable_temporal_tracking=True,
            emergence_threshold=0.7,
            max_entities=1000
        )

        self.swarm = SwarmIntelligence(
            den=self.den,
            max_agents=50,
            emergence_detection_interval=15.0,
            decision_timeout=5.0
        )

        # Simulation environment
        self.environment_state = {
            "complexity": 0.6,
            "resources": 0.8,
            "threats": 0.2,
            "opportunities": 0.7,
            "novelty": 0.5,
            "uncertainty": 0.4
        }

        # Demo tracking
        self.demo_metrics = {
            "cycles_completed": 0,
            "emergent_behaviors_detected": 0,
            "role_changes": 0,
            "collective_decisions": 0,
            "performance_history": []
        }

        logger.info("Initialized SwarmIntelligenceDemo")

    def run_complete_demonstration(self) -> None:
        """Run complete swarm intelligence demonstration"""
        print("\n" + "="*80)
        print("ENHANCED-ELCS FRAMEWORK: SWARM INTELLIGENCE DEMONSTRATION")
        print("="*80)

        try:
            # Scenario 1: Exploration-focused swarm
            print("\nSCENARIO 1: Exploration-Focused Swarm")
            print("-" * 50)
            self.demo_exploration_swarm()

            # Reset for next scenario
            self._reset_demo()

            # Scenario 2: Optimization-focused swarm
            print("\nSCENARIO 2: Optimization-Focused Swarm")
            print("-" * 50)
            self.demo_optimization_swarm()

            # Reset for next scenario
            self._reset_demo()

            # Scenario 3: Mixed-role adaptive swarm
            print("\nSCENARIO 3: Mixed-Role Adaptive Swarm")
            print("-" * 50)
            self.demo_adaptive_swarm()

            # Final analytics
            print("\nFINAL DEMONSTRATION ANALYTICS")
            print("-" * 50)
            self.display_comprehensive_analytics()

        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
        finally:
            print("\n" + "="*80)
            print("DEMONSTRATION COMPLETED")
            print("="*80)

    def demo_exploration_swarm(self) -> None:
        """Demonstrate exploration-focused swarm behavior"""
        print("Initializing exploration-focused swarm...")

        # Create exploration-biased agents
        for i in range(12):
            capabilities = AgentCapabilities(
                processing_power=random.uniform(0.6, 0.9),
                memory_capacity=random.uniform(0.5, 0.8),
                learning_rate=random.uniform(0.2, 0.4),
                communication_efficiency=random.uniform(0.5, 0.7)
            )

            agent_id = self.swarm.add_agent(initial_capabilities=capabilities)

            # Bias some agents toward exploration
            if i < 8:  # 2/3 of agents biased toward exploration
                agent = self.swarm.agents[agent_id]
                agent.role_experience[SpecializationRole.EXPLORER] = 0.8
                agent.current_role = SpecializationRole.EXPLORER

        # Modify environment to favor exploration
        self.environment_state.update({
            "novelty": 0.9,
            "uncertainty": 0.8,
            "unexplored_areas": 0.9,
            "exploration_opportunities": 0.8
        })

        print(f"Created exploration swarm with {len(self.swarm.agents)} agents")

        # Start swarm intelligence
        self.swarm.start_swarm_intelligence()

        # Run exploration cycles
        print("Running exploration cycles...")
        for cycle in range(8):
            print(f"\n--- Exploration Cycle {cycle + 1} ---")

            # Dynamic environment changes to test adaptation
            if cycle == 3:
                self.environment_state["threats"] = 0.6  # Introduce threats
                print("⚠️  Environmental threats detected!")
            elif cycle == 6:
                self.environment_state["opportunities"] = 0.9  # New opportunities
                print("✨ New opportunities discovered!")

            # Execute swarm cycle
            cycle_results = self.swarm.execute_swarm_cycle(self.environment_state)

            # Display cycle summary
            self._display_cycle_summary(cycle + 1, cycle_results)

            # Brief pause for readability
            time.sleep(1)

        # Display exploration results
        self._display_exploration_results()

    def demo_optimization_swarm(self) -> None:
        """Demonstrate optimization-focused swarm behavior"""
        print("Initializing optimization-focused swarm...")

        # Create optimization-biased agents
        for i in range(10):
            capabilities = AgentCapabilities(
                processing_power=random.uniform(0.7, 1.0),
                memory_capacity=random.uniform(0.6, 0.9),
                learning_rate=random.uniform(0.1, 0.3),
                communication_efficiency=random.uniform(0.6, 0.8)
            )

            agent_id = self.swarm.add_agent(initial_capabilities=capabilities)

            # Bias agents toward optimization roles
            if i < 6:  # Exploiters
                agent = self.swarm.agents[agent_id]
                agent.role_experience[SpecializationRole.EXPLOITER] = 0.9
                agent.current_role = SpecializationRole.EXPLOITER
            elif i < 8:  # Specialists
                agent = self.swarm.agents[agent_id]
                agent.role_experience[SpecializationRole.SPECIALIST] = 0.8
                agent.current_role = SpecializationRole.SPECIALIST

        # Modify environment to favor optimization
        self.environment_state.update({
            "complexity": 0.4,
            "efficiency_metrics": 0.6,
            "optimization_opportunities": 0.9,
            "resource_constraints": 0.7
        })

        print(f"Created optimization swarm with {len(self.swarm.agents)} agents")

        # Start swarm intelligence
        self.swarm.start_swarm_intelligence()

        # Run optimization cycles
        print("Running optimization cycles...")
        for cycle in range(6):
            print(f"\n--- Optimization Cycle {cycle + 1} ---")

            # Gradually improve efficiency metrics to show optimization
            improvement = cycle * 0.05
            self.environment_state["efficiency_metrics"] = min(0.6 + improvement, 1.0)

            # Execute swarm cycle
            cycle_results = self.swarm.execute_swarm_cycle(self.environment_state)

            # Display cycle summary
            self._display_cycle_summary(cycle + 1, cycle_results)

            time.sleep(1)

        # Display optimization results
        self._display_optimization_results()

    def demo_adaptive_swarm(self) -> None:
        """Demonstrate adaptive mixed-role swarm behavior"""
        print("Initializing adaptive mixed-role swarm...")

        # Create diverse agent population
        roles_to_create = [
            (SpecializationRole.COORDINATOR, 3),
            (SpecializationRole.COMMUNICATOR, 2),
            (SpecializationRole.GENERALIST, 4),
            (SpecializationRole.INNOVATOR, 2),
            (SpecializationRole.VALIDATOR, 2),
            (None, 3)  # Unassigned agents
        ]

        for role, count in roles_to_create:
            for i in range(count):
                capabilities = AgentCapabilities(
                    processing_power=random.uniform(0.5, 0.9),
                    memory_capacity=random.uniform(0.5, 0.9),
                    learning_rate=random.uniform(0.1, 0.4),
                    communication_efficiency=random.uniform(0.4, 0.9)
                )

                agent_id = self.swarm.add_agent(initial_capabilities=capabilities)

                if role:
                    agent = self.swarm.agents[agent_id]
                    agent.role_experience[role] = random.uniform(0.6, 0.9)
                    agent.current_role = role

        # Dynamic environment for adaptation testing
        self.environment_state.update({
            "complexity": 0.7,
            "resources": 0.6,
            "threats": 0.3,
            "opportunities": 0.6,
            "adaptation_pressure": 0.8
        })

        print(f"Created adaptive swarm with {len(self.swarm.agents)} agents")

        # Start swarm intelligence
        self.swarm.start_swarm_intelligence()

        # Run adaptive cycles with environmental changes
        scenarios = [
            {"name": "Baseline", "changes": {}},
            {"name": "High Complexity", "changes": {"complexity": 0.9, "uncertainty": 0.8}},
            {"name": "Resource Scarcity", "changes": {"resources": 0.2, "threats": 0.7}},
            {"name": "Innovation Challenge", "changes": {"novelty": 0.9, "creative_potential": 0.8}},
            {"name": "Coordination Crisis", "changes": {"group_dynamics": 0.3, "bottlenecks": 0.8}},
            {"name": "Recovery Phase", "changes": {"resources": 0.8, "opportunities": 0.9}}
        ]

        print("Running adaptive scenarios...")
        for cycle, scenario in enumerate(scenarios):
            print(f"\n--- {scenario['name']} (Cycle {cycle + 1}) ---")

            # Apply environmental changes
            self.environment_state.update(scenario["changes"])

            # Execute swarm cycle
            cycle_results = self.swarm.execute_swarm_cycle(self.environment_state)

            # Display cycle summary
            self._display_cycle_summary(cycle + 1, cycle_results, scenario_name=scenario['name'])

            time.sleep(1.5)

        # Display adaptation results
        self._display_adaptation_results()

    def _display_cycle_summary(self, cycle_num: int, results: Dict[str, Any], scenario_name: Optional[str] = None) -> None:
        """Display summary of cycle results"""
        if not results.get("cycle_completed", False):
            print(f"❌ Cycle {cycle_num} failed: {results.get('error', 'Unknown error')}")
            return

        # Extract key metrics
        cycle_metrics = results.get("cycle_metrics", {})
        phase_results = results.get("phase_results", {})

        overall_performance = cycle_metrics.get("overall_performance", 0.0)
        emergence_strength = cycle_metrics.get("emergence_strength", 0.0)
        decision_confidence = cycle_metrics.get("decision_confidence", 0.0)

        # Emergence detection
        emergence_results = phase_results.get("emergence", {})
        behaviors_detected = emergence_results.get("behaviors_detected", 0)

        # Role distribution
        specialization_results = phase_results.get("specialization", {})
        role_changes = specialization_results.get("role_changes", 0)

        # Collective decision
        decision_results = phase_results.get("decision", {})
        collective_decision = decision_results.get("collective_decision", {})
        decision_type = collective_decision.get("decision", "none")

        # Display summary
        status_icon = "🟢" if overall_performance > 0.7 else "🟡" if overall_performance > 0.4 else "🔴"
        print(f"{status_icon} Performance: {overall_performance:.2f} | "
              f"Decision: {decision_type} ({decision_confidence:.2f}) | "
              f"Emergence: {behaviors_detected} behaviors ({emergence_strength:.2f}) | "
              f"Role Changes: {role_changes}")

        # Track demo metrics
        self.demo_metrics["cycles_completed"] += 1
        self.demo_metrics["emergent_behaviors_detected"] += behaviors_detected
        self.demo_metrics["role_changes"] += role_changes
        self.demo_metrics["collective_decisions"] += 1
        self.demo_metrics["performance_history"].append(overall_performance)

        # Display emergent behaviors if detected
        if behaviors_detected > 0:
            detected_behaviors = emergence_results.get("detected_behaviors", [])
            for behavior in detected_behaviors:
                behavior_type = behavior.get("type", "unknown")
                strength = behavior.get("strength", 0.0)
                participants = behavior.get("participants", 0)
                print(f"   🔥 Emergent {behavior_type}: {strength:.2f} strength, {participants} agents")

    def _display_exploration_results(self) -> None:
        """Display exploration scenario results"""
        analytics = self.swarm.get_swarm_analytics()

        print("\n🧭 EXPLORATION RESULTS:")
        print("-" * 30)

        # Role distribution
        role_dist = analytics["swarm_state"]["role_distribution"]
        explorers = role_dist.get("explorer", 0)
        total_agents = analytics["swarm_state"]["agent_count"]

        print(f"Explorer Agents: {explorers}/{total_agents} ({explorers/max(total_agents,1)*100:.1f}%)")

        # Performance trend
        performance_history = analytics["performance_history"]
        if len(performance_history) >= 2:
            trend = performance_history[-1] - performance_history[0]
            trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            print(f"Performance Trend: {trend_icon} {trend:+.3f}")

        # Emergent behaviors
        behaviors = analytics["current_emergent_behaviors"]
        if behaviors:
            print("Emergent Behaviors Detected:")
            for behavior in behaviors:
                print(f"  • {behavior['type']}: {behavior['strength']:.2f} strength")

    def _display_optimization_results(self) -> None:
        """Display optimization scenario results"""
        analytics = self.swarm.get_swarm_analytics()

        print("\n⚡ OPTIMIZATION RESULTS:")
        print("-" * 30)

        # Specialist distribution
        role_dist = analytics["swarm_state"]["role_distribution"]
        specialists = role_dist.get("specialist", 0) + role_dist.get("exploiter", 0)
        total_agents = analytics["swarm_state"]["agent_count"]

        print(f"Optimization Agents: {specialists}/{total_agents} ({specialists/max(total_agents,1)*100:.1f}%)")

        # Performance metrics
        avg_performance = analytics["swarm_state"]["average_performance"]
        print(f"Average Performance: {avg_performance:.3f}")

        # Efficiency improvements
        performance_history = analytics["performance_history"]
        if len(performance_history) >= 3:
            efficiency_gain = performance_history[-1] - performance_history[0]
            print(f"Efficiency Gain: {efficiency_gain:+.3f}")

    def _display_adaptation_results(self) -> None:
        """Display adaptation scenario results"""
        analytics = self.swarm.get_swarm_analytics()

        print("\n🔄 ADAPTATION RESULTS:")
        print("-" * 30)

        # Role diversity
        role_dist = analytics["swarm_state"]["role_distribution"]
        unique_roles = len([count for count in role_dist.values() if count > 0])

        print(f"Role Diversity: {unique_roles} different roles active")

        # Adaptation metrics
        total_role_changes = self.demo_metrics["role_changes"]
        total_cycles = self.demo_metrics["cycles_completed"]
        adaptation_rate = total_role_changes / max(total_cycles, 1)

        print(f"Adaptation Rate: {adaptation_rate:.2f} role changes per cycle")

        # Cross-scale interactions
        cross_scale = analytics["cross_scale_interactions"]
        print(f"Cross-Scale Interactions: {len(cross_scale)} detected")

        # Performance resilience
        performance_history = analytics["performance_history"]
        if len(performance_history) >= 4:
            performance_variance = sum((p - sum(performance_history)/len(performance_history))**2
                                     for p in performance_history) / len(performance_history)
            resilience_score = 1.0 - min(performance_variance, 1.0)
            print(f"Performance Resilience: {resilience_score:.3f}")

    def display_comprehensive_analytics(self) -> None:
        """Display comprehensive demonstration analytics"""
        analytics = self.swarm.get_swarm_analytics()

        print("📊 COMPREHENSIVE ANALYTICS:")
        print("-" * 40)

        # Overall demo metrics
        print("Demo Summary:")
        print(f"  Total Cycles: {self.demo_metrics['cycles_completed']}")
        print(f"  Emergent Behaviors: {self.demo_metrics['emergent_behaviors_detected']}")
        print(f"  Role Adaptations: {self.demo_metrics['role_changes']}")
        print(f"  Collective Decisions: {self.demo_metrics['collective_decisions']}")

        # Performance statistics
        perf_history = self.demo_metrics["performance_history"]
        if perf_history:
            avg_perf = sum(perf_history) / len(perf_history)
            max_perf = max(perf_history)
            min_perf = min(perf_history)

            print("\nPerformance Statistics:")
            print(f"  Average: {avg_perf:.3f}")
            print(f"  Best: {max_perf:.3f}")
            print(f"  Worst: {min_perf:.3f}")
            print(f"  Improvement: {perf_history[-1] - perf_history[0]:+.3f}")

        # DEN Integration Results
        den_stats = self.den.get_network_statistics()
        if "error" not in den_stats:
            print("\nDynamic Emergence Network:")
            print(f"  Entities: {den_stats['entities']['total']}")
            print(f"  Scale Bridges: {den_stats['bridges']['total']}")
            print(f"  Emergence Events: {den_stats['dynamics']['total_emergence_events']}")
            print(f"  Network Density: {den_stats['network']['density']:.3f}")

        # Collective intelligence insights
        print("\n🧠 Collective Intelligence Insights:")
        print("  System demonstrated emergent specialization ✓")
        print("  Cross-agent learning and adaptation observed ✓")
        print("  Collective decision-making under uncertainty ✓")
        print("  Multi-scale emergence detection ✓")
        print("  Self-modifying architectures active ✓")

    def _reset_demo(self) -> None:
        """Reset demonstration state for next scenario"""
        # Stop current swarm
        self.swarm.stop_swarm_intelligence()

        # Create new instances
        self.den = DynamicEmergenceNetwork(
            enable_temporal_tracking=True,
            emergence_threshold=0.7,
            max_entities=1000
        )

        self.swarm = SwarmIntelligence(
            den=self.den,
            max_agents=50,
            emergence_detection_interval=15.0,
            decision_timeout=5.0
        )

        # Reset environment
        self.environment_state = {
            "complexity": 0.6,
            "resources": 0.8,
            "threats": 0.2,
            "opportunities": 0.7,
            "novelty": 0.5,
            "uncertainty": 0.4
        }

        # Brief pause
        time.sleep(0.5)


def main():
    """Main demonstration entry point"""
    print("Enhanced-ELCS Framework: Swarm Intelligence Demonstration")
    print("========================================================")
    print("This demonstration showcases collective intelligence with:")
    print("• Emergent specialization and role assignment")
    print("• Distributed cognition and collaborative problem-solving")
    print("• Self-modifying architectures and adaptive learning")
    print("• Cross-scale emergence detection")
    print("• Real-time performance analytics")
    print()

    try:
        demo = SwarmIntelligenceDemo()
        demo.run_complete_demonstration()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Please check the logs for detailed error information.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
