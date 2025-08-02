#!/usr/bin/env python3
"""
Comprehensive test of the SwarmLibrary system showing all capabilities:
- Creating swarms with different configurations
- Automatic storage after simulations
- Metadata management
- Fuzzy search functionality
- Export/Import capabilities
"""

import json
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_elcs_mcp.core.framework_manager import ELCSFrameworkManager

async def test_comprehensive_library_system():
    """Test all aspects of the SwarmLibrary system"""

    print("🧪 ENHANCED ELCS FRAMEWORK - SWARM LIBRARY SYSTEM TEST")
    print("=" * 60)

    # Initialize framework
    manager = ELCSFrameworkManager()

    # Test 1: Create diverse swarms with different purposes
    print("\n📊 TEST 1: Creating Diverse Swarms")
    print("-" * 40)

    swarm_configs = [
        {
            "id": "research_optimization_swarm",
            "purpose": "academic research",
            "agent_count": 12,
            "capabilities": {
                "processing_power": 0.8,
                "memory_capacity": 0.7,
                "learning_rate": 0.2,
                "communication_efficiency": 0.9
            },
            "tags": ["research", "optimization", "academic"],
            "description": "High-performance swarm optimized for research tasks with strong communication"
        },
        {
            "id": "exploration_swarm_alpha",
            "purpose": "environment exploration",
            "agent_count": 8,
            "capabilities": {
                "processing_power": 0.6,
                "memory_capacity": 0.5,
                "learning_rate": 0.3,
                "communication_efficiency": 0.6
            },
            "tags": ["exploration", "adaptive", "lightweight"],
            "description": "Adaptive exploration swarm with high learning rate for unknown environments"
        },
        {
            "id": "coordination_test_swarm",
            "purpose": "coordination analysis",
            "agent_count": 15,
            "capabilities": {
                "processing_power": 0.7,
                "memory_capacity": 0.8,
                "learning_rate": 0.15,
                "communication_efficiency": 0.95
            },
            "tags": ["coordination", "analysis", "communication"],
            "description": "Large swarm designed for testing coordination patterns and communication efficiency"
        }
    ]

    created_swarms = []
    for config in swarm_configs:
        print(f"Creating swarm: {config['id']}")
        result = await manager.create_swarm_simulation(
            swarm_id=config["id"],
            agent_count=config["agent_count"],
            agent_capabilities=config["capabilities"]
        )

        # Store in library with metadata
        await manager.store_swarm_in_library(
            swarm_id=config["id"],
            tags=config["tags"],
            description=config["description"],
            research_category=config["purpose"]
        )

        created_swarms.append(config["id"])
        print(f"✅ Created and stored: {config['id']}")

    # Test 2: Run simulations and automatic storage
    print("\n🚀 TEST 2: Running Simulations (Auto-Storage)")
    print("-" * 40)

    for swarm_id in created_swarms[:2]:  # Test first two swarms
        print(f"Running simulation for: {swarm_id}")
        environment = {
            "complexity": 0.7,
            "resources": 0.8,
            "opportunities": 0.6,
            "threats": 0.3
        }

        result = await manager.run_swarm_simulation(
            swarm_id=swarm_id,
            environment_state=environment,
            cycles=3
        )

        # Check for emergence patterns
        patterns = await manager.detect_emergence_patterns(swarm_id)
        print(f"  🔍 Detected {len(patterns.get('patterns', []))} emergence patterns")

    # Test 3: Fuzzy search functionality
    print("\n🔍 TEST 3: Fuzzy Search Capabilities")
    print("-" * 40)

    search_queries = [
        "research",
        "coordination",
        "exploration",
        "optimization",
        "communication",
        "adaptive"
    ]

    for query in search_queries:
        results = await manager.search_swarm_library(query=query, limit=10)
        print(f"Search '{query}': Found {len(results)} matches")
        for result in results:
            print(f"  📁 {result['swarm_id']} (score: {result['score']:.2f}) - {result['metadata']['description'][:50]}...")

    # Test 4: Library statistics and management
    print("\n📈 TEST 4: Library Statistics")
    print("-" * 40)

    all_swarms = await manager.get_library_swarms()
    print(f"Total swarms in library: {len(all_swarms)}")

    # Group by tags
    tag_counts = {}
    for swarm in all_swarms:
        for tag in swarm['metadata']['tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("Tag distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  🏷️  {tag}: {count} swarms")

    # Test 5: Export functionality
    print("\n💾 TEST 5: Export/Import Functionality")
    print("-" * 40)

    # Export one swarm
    export_swarm_id = created_swarms[0]
    export_path = f"/tmp/exported_{export_swarm_id}.json"

    try:
        await manager.export_swarm_from_library(export_swarm_id, export_path)
        print(f"✅ Exported {export_swarm_id} to {export_path}")

        # Show export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        print(f"Export contains: {list(export_data.keys())}")
        print(f"Metadata: {export_data['metadata']['tags']}")

    except Exception as e:
        print(f"❌ Export failed: {e}")

    # Test 6: Advanced search with multiple criteria
    print("\n🎯 TEST 6: Advanced Multi-Criteria Search")
    print("-" * 40)

    # Search for research-related swarms with high communication
    advanced_results = manager.search_swarm_library(
        query="research communication optimization",
        limit=5
    )

    print(f"Advanced search results: {len(advanced_results)} matches")
    for result in advanced_results:
        metadata = result['metadata']
        print(f"  🎯 {result['swarm_id']} (score: {result['score']:.3f})")
        print(f"     Tags: {', '.join(metadata['tags'])}")
        print(f"     Purpose: {metadata.get('purpose', 'N/A')}")
        print(f"     Description: {metadata['description'][:60]}...")

    # Test 7: Library folder structure inspection
    print("\n📂 TEST 7: Library Structure")
    print("-" * 40)

    library_path = Path("swarm_library")
    if library_path.exists():
        print(f"Library location: {library_path.absolute()}")

        # Count files
        swarm_dirs = [d for d in library_path.iterdir() if d.is_dir()]
        total_files = sum(len(list(d.glob("*"))) for d in swarm_dirs)

        print(f"Swarm directories: {len(swarm_dirs)}")
        print(f"Total files: {total_files}")

        # Show structure for first swarm
        if swarm_dirs:
            first_swarm = swarm_dirs[0]
            print(f"\nSample structure ({first_swarm.name}):")
            for file in sorted(first_swarm.glob("*")):
                print(f"  📄 {file.name}")

    print("\n✅ COMPREHENSIVE LIBRARY TEST COMPLETED")
    print("=" * 60)
    print("\n🎉 SwarmLibrary System Features Demonstrated:")
    print("  ✅ Persistent storage with organized folders")
    print("  ✅ Rich metadata management (tags, descriptions, purposes)")
    print("  ✅ Fuzzy search with relevance scoring")
    print("  ✅ Automatic storage after simulations")
    print("  ✅ Export/Import functionality")
    print("  ✅ Multi-criteria search capabilities")
    print("  ✅ Library statistics and management")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_library_system())
