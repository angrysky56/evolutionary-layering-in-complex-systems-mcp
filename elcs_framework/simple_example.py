#!/usr/bin/env python3
"""
Enhanced ELCS Framework - Simple Usage Example
==============================================

This script demonstrates basic usage of the Enhanced-ELCS-Framework
swarm intelligence system with minimal setup and clear examples.

Usage:
    python simple_example.py

Author: Enhanced ELCS Development Team
Version: 1.0.0
License: MIT
"""

import logging
import time

# Configure simple logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise for example

# Import Enhanced-ELCS components
from core.dynamic_emergence_networks import DynamicEmergenceNetwork
from core.multi_agent_swarm import AgentCapabilities, SpecializationRole, SwarmAgent, SwarmIntelligence


def simple_swarm_example():
    """Simple example showing basic swarm intelligence usage"""

    print("Enhanced-ELCS Framework: Simple Swarm Intelligence Example")
    print("=" * 60)

    # Step 1: Create Dynamic Emergence Network (optional but recommended)
    print("\n1. Creating Dynamic Emergence Network...")
    den = DynamicEmergenceNetwork()
    print("   ✓ Dynamic Emergence Network initialized")

    # Step 2: Create Swarm Intelligence orchestrator
    print("\n2. Creating Swarm Intelligence system...")
    swarm = SwarmIntelligence(
        den=den,
        max_agents=10,
        emergence_detection_interval=30.0
    )
    print("   ✓ Swarm Intelligence system initialized")

    # Step 3: Add agents to the swarm
    print("\n3. Adding agents to swarm...")
    agent_ids = []

    for i in range(6):
        # Create agent with random capabilities
        capabilities = AgentCapabilities(
            processing_power=0.6 + i * 0.05,
            memory_capacity=0.5 + i * 0.08,
            learning_rate=0.15,
            communication_efficiency=0.6
        )

        agent_id = swarm.add_agent(initial_capabilities=capabilities)
        agent_ids.append(agent_id)

    print(f"   ✓ Added {len(agent_ids)} agents to swarm")

    # Step 4: Start swarm intelligence
    print("\n4. Starting swarm intelligence...")
    swarm.start_swarm_intelligence()
    print("   ✓ Swarm intelligence active")

    # Step 5: Create simple environment for agents to interact with
    environment_state = {
        "complexity": 0.7,
        "resources": 0.8,
        "opportunities": 0.6,
        "threats": 0.2
    }

    # Step 6: Run swarm intelligence cycles
    print("\n5. Running swarm intelligence cycles...")

    for cycle in range(5):
        print(f"\n   --- Cycle {cycle + 1} ---")

        # Execute one complete swarm cycle
        results = swarm.execute_swarm_cycle(environment_state)

        if results.get("cycle_completed", False):
            # Extract key metrics
            metrics = results.get("cycle_metrics", {})
            performance = metrics.get("overall_performance", 0.0)
            emergence_strength = metrics.get("emergence_strength", 0.0)

            # Check for emergent behaviors
            emergence_results = results.get("phase_results", {}).get("emergence", {})
            behaviors_detected = emergence_results.get("behaviors_detected", 0)

            # Display simple summary
            performance_icon = "🟢" if performance > 0.6 else "🟡" if performance > 0.4 else "🔴"
            print(f"   {performance_icon} Performance: {performance:.2f}")

            if behaviors_detected > 0:
                print(f"   🔥 Emergent behaviors detected: {behaviors_detected}")
                behaviors = emergence_results.get("detected_behaviors", [])
                for behavior in behaviors:
                    behavior_type = behavior.get("type", "unknown")
                    strength = behavior.get("strength", 0.0)
                    print(f"      • {behavior_type}: {strength:.2f} strength")

            # Show role distribution
            specialization = results.get("phase_results", {}).get("specialization", {})
            role_dist = specialization.get("role_distribution", {})
            if role_dist:
                active_roles = [role for role, count in role_dist.items() if count > 0]
                print(f"   👥 Active roles: {', '.join(active_roles)}")
        else:
            print(f"   ❌ Cycle failed: {results.get('error', 'Unknown error')}")

        # Brief pause for readability
        time.sleep(1)

    # Step 7: Get final analytics
    print("\n6. Final Analytics:")
    print("   " + "-" * 25)

    analytics = swarm.get_swarm_analytics()
    swarm_state = analytics["swarm_state"]

    print(f"   Total Agents: {swarm_state['agent_count']}")
    print(f"   Average Performance: {swarm_state['average_performance']:.3f}")
    print(f"   Role Distribution: {swarm_state['role_distribution']}")
    print(f"   Emergent Behaviors: {len(analytics['current_emergent_behaviors'])}")

    # Step 8: Stop swarm intelligence
    print("\n7. Stopping swarm...")
    swarm.stop_swarm_intelligence()
    print("   ✓ Swarm intelligence stopped")

    print("\n" + "=" * 60)
    print("Simple swarm intelligence example completed!")
    print("=" * 60)


def agent_interaction_example():
    """Example showing direct agent interactions"""

    print("\n\nBonus: Direct Agent Interaction Example")
    print("=" * 45)

    # Create two agents directly
    print("\n1. Creating individual agents...")

    agent1 = SwarmAgent()
    agent1.current_role = SpecializationRole.EXPLORER

    agent2 = SwarmAgent()
    agent2.current_role = SpecializationRole.COORDINATOR

    print("   ✓ Created Explorer and Coordinator agents")

    # Simulate environment perception
    print("\n2. Agents perceive environment...")

    environment = {
        "complexity": 0.8,
        "novelty": 0.9,
        "threats": 0.1
    }

    perception1 = agent1.perceive_environment(environment)
    perception2 = agent2.perceive_environment(environment)

    print(f"   Explorer perceived: {len(perception1)} environmental signals")
    print(f"   Coordinator perceived: {len(perception2)} environmental signals")

    # Agents communicate
    print("\n3. Agents communicate...")

    comm_results1 = agent1.communicate_with_peers([agent2])
    comm_results2 = agent2.communicate_with_peers([agent1])

    print(f"   Communication exchanges: {len(comm_results1) + len(comm_results2)}")

    # Make decisions
    print("\n4. Agents make decisions...")

    decision1 = agent1.make_local_decision(perception1, comm_results1)
    decision2 = agent2.make_local_decision(perception2, comm_results2)

    print(f"   Explorer decides: {decision1.get('action_type', 'none')} "
          f"(confidence: {decision1.get('confidence', 0):.2f})")
    print(f"   Coordinator decides: {decision2.get('action_type', 'none')} "
          f"(confidence: {decision2.get('confidence', 0):.2f})")

    # Execute actions
    print("\n5. Agents execute actions...")

    result1 = agent1.execute_action(decision1)
    result2 = agent2.execute_action(decision2)

    print(f"   Explorer performance: {result1.get('performance', 0):.3f}")
    print(f"   Coordinator performance: {result2.get('performance', 0):.3f}")

    print("\n" + "=" * 45)
    print("Agent interaction example completed!")


if __name__ == "__main__":
    try:
        # Run simple swarm example
        simple_swarm_example()

        # Run bonus agent interaction example
        agent_interaction_example()

        print("\n🎉 All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install numpy networkx")
        raise
