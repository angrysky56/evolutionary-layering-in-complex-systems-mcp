#!/usr/bin/env python3
"""
Debug script to test emergence detection with detailed output
"""
import sys
import asyncio

# Add the project root to the path
sys.path.insert(0, '/home/ty/Repositories/ai_workspace/evolutionary-layering-in-complex-systems-mcp')

from src.enhanced_elcs_mcp.core.framework_manager import ELCSFrameworkManager

async def main():
    print("Starting emergence detection debug test...")

    # Initialize framework
    framework = ELCSFrameworkManager()

    # Create a test swarm
    print("\n1. Creating test swarm...")
    result = await framework.create_swarm_simulation(
        swarm_id="debug_test_roles",
        agent_count=3,
        agent_capabilities={
            "processing_power": 0.7,
            "memory_capacity": 0.6,
            "learning_rate": 0.2,
            "communication_efficiency": 0.7
        },
        config={"emergence_threshold": 0.01}
    )
    print(f"Swarm created: {result}")

    # Run simulation to trigger role assignment
    print("\n2. Running simulation...")
    sim_result = await framework.run_swarm_simulation(
        swarm_id="debug_test_roles",
        environment_state={
            "complexity": 0.5,
            "opportunities": 0.5,
            "resources": 0.8,
            "threats": 0.2
        },
        cycles=1
    )
    print(f"Simulation completed. Role distribution: {sim_result.get('final_analytics', {}).get('swarm_state', {}).get('role_distribution', {})}")

    # Test emergence detection
    print("\n3. Testing emergence detection...")
    try:
        emergence_result = await framework.detect_emergence_patterns(swarm_id="debug_test_roles")
        print(f"Emergence detection result: {emergence_result}")
    except Exception as e:
        print(f"ERROR in emergence detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
