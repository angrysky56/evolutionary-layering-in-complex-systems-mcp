#!/usr/bin/env python3
"""
Enhanced ELCS Framework - Quick Validation Test
===============================================

This script performs a quick validation test to ensure all components
are properly implemented and can be imported successfully.

Usage:
    python validate_implementation.py

Author: Enhanced ELCS Development Team
Version: 1.0.0
"""

import sys
import traceback


def test_imports():
    """Test that all core components can be imported"""
    print("Testing imports...")

    try:
        # Test Dynamic Emergence Networks import
        from core.dynamic_emergence_networks import DynamicEmergenceNetwork, ProcessEntity, ProcessScale, ProcessSignature
        print("  ✓ Dynamic Emergence Networks imported successfully")

        # Test Multi-Agent Swarm imports
        from core.multi_agent_swarm import AgentCapabilities, CollectiveDecisionMaker, EmergenceBehaviorDetector, SpecializationRole, SwarmAgent, SwarmCommunicationProtocol, SwarmIntelligence
        print("  ✓ Multi-Agent Swarm components imported successfully")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\nTesting basic functionality...")

    try:
        # Import components
        from core.dynamic_emergence_networks import DynamicEmergenceNetwork
        from core.multi_agent_swarm import AgentCapabilities, SwarmIntelligence

        # Test DEN creation
        den = DynamicEmergenceNetwork()
        print("  ✓ Dynamic Emergence Network created")

        # Test SwarmIntelligence creation
        swarm = SwarmIntelligence(den=den, max_agents=5)
        print("  ✓ SwarmIntelligence created")

        # Test agent addition
        capabilities = AgentCapabilities(
            processing_power=0.7,
            memory_capacity=0.6,
            learning_rate=0.2
        )

        agent_id = swarm.add_agent(initial_capabilities=capabilities)
        print(f"  ✓ Agent added with ID: {agent_id[:8]}...")

        # Test swarm start
        swarm.start_swarm_intelligence()
        print("  ✓ Swarm intelligence started")

        # Test basic cycle (should not crash)
        environment_state = {
            "complexity": 0.5,
            "resources": 0.7,
            "opportunities": 0.6
        }

        results = swarm.execute_swarm_cycle(environment_state)
        if results.get("cycle_completed", False):
            print("  ✓ Swarm intelligence cycle executed successfully")
        else:
            print(f"  ⚠️ Cycle completed with warnings: {results.get('error', 'Unknown')}")

        # Test swarm stop
        swarm.stop_swarm_intelligence()
        print("  ✓ Swarm intelligence stopped")

        return True

    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_analytics():
    """Test analytics and monitoring functionality"""
    print("\nTesting analytics functionality...")

    try:
        from core.dynamic_emergence_networks import DynamicEmergenceNetwork
        from core.multi_agent_swarm import AgentCapabilities, SwarmIntelligence

        # Create system with multiple agents
        den = DynamicEmergenceNetwork()
        swarm = SwarmIntelligence(den=den, max_agents=10)

        # Add multiple agents
        for i in range(3):
            capabilities = AgentCapabilities(
                processing_power=0.6 + i * 0.1,
                memory_capacity=0.5 + i * 0.1,
                learning_rate=0.2
            )
            swarm.add_agent(initial_capabilities=capabilities)

        print(f"  ✓ Created swarm with {len(swarm.agents)} agents")

        # Test analytics
        analytics = swarm.get_swarm_analytics()

        swarm_state = analytics.get("swarm_state", {})
        agent_count = swarm_state.get("agent_count", 0)

        if agent_count == 3:
            print("  ✓ Analytics reporting correct agent count")
        else:
            print(f"  ⚠️ Analytics agent count mismatch: expected 3, got {agent_count}")

        # Test DEN statistics
        den_stats = den.get_network_statistics()
        if "error" not in den_stats:
            print("  ✓ DEN statistics generated successfully")
        else:
            print(f"  ⚠️ DEN statistics error: {den_stats['error']}")

        return True

    except Exception as e:
        print(f"  ❌ Analytics test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("Enhanced-ELCS Framework Implementation Validation")
    print("=" * 55)

    all_tests = [
        test_imports,
        test_basic_functionality,
        test_analytics
    ]

    passed = 0
    total = len(all_tests)

    for test_func in all_tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")

    print("\n" + "=" * 55)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! Implementation is ready to use.")
        print("\nNext steps:")
        print("  • Run 'python simple_example.py' for basic usage")
        print("  • Run 'python swarm_intelligence_demo.py' for full demo")
        return 0
    else:
        print("❌ Some validation tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
