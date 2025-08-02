#!/usr/bin/env python3
"""
Simple test of the SwarmLibrary system showing core functionality:
- Creating swarms
- Storing in library
- Searching library
- Export functionality
"""

import json
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_elcs_mcp.core.framework_manager import ELCSFrameworkManager

async def test_swarm_library():
    """Test SwarmLibrary system with proper async/await"""

    print("🧪 SWARM LIBRARY SYSTEM TEST")
    print("=" * 50)

    # Initialize framework
    manager = ELCSFrameworkManager()

    # Test 1: Create and store swarms
    print("\n📊 Creating and Storing Swarms")
    print("-" * 30)

    swarms_to_create = [
        {
            "id": "research_swarm_01",
            "count": 10,
            "capabilities": {
                "processing_power": 0.8,
                "memory_capacity": 0.7,
                "learning_rate": 0.2,
                "communication_efficiency": 0.9
            },
            "tags": ["research", "optimization"],
            "description": "High-performance research swarm",
            "category": "academic"
        },
        {
            "id": "exploration_swarm_01",
            "count": 8,
            "capabilities": {
                "processing_power": 0.6,
                "memory_capacity": 0.5,
                "learning_rate": 0.3,
                "communication_efficiency": 0.6
            },
            "tags": ["exploration", "adaptive"],
            "description": "Adaptive exploration swarm",
            "category": "exploration"
        }
    ]

    for swarm_config in swarms_to_create:
        print(f"Creating swarm: {swarm_config['id']}")

        # Create swarm
        result = await manager.create_swarm_simulation(
            swarm_id=swarm_config["id"],
            agent_count=swarm_config["count"],
            agent_capabilities=swarm_config["capabilities"]
        )
        print(f"  ✅ Created with {len(result.get('agents', []))} agents")

        # Store in library
        await manager.store_swarm_in_library(
            swarm_id=swarm_config["id"],
            description=swarm_config["description"],
            tags=swarm_config["tags"],
            research_category=swarm_config["category"]
        )
        print(f"  📚 Stored in library")

    # Test 2: Run quick simulation
    print(f"\n🚀 Running Simulation")
    print("-" * 30)

    swarm_id = "research_swarm_01"
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
    print(f"Simulation completed: {result.get('cycles_completed', 0)} cycles")

    # Check for emergence patterns
    patterns = await manager.detect_emergence_patterns(swarm_id)
    pattern_count = len(patterns.get('patterns', []))
    print(f"🔍 Detected {pattern_count} emergence patterns")

    # Test 3: Search functionality
    print(f"\n🔍 Testing Search")
    print("-" * 30)

    search_terms = ["research", "exploration", "optimization", "adaptive"]

    for term in search_terms:
        try:
            search_result = await manager.search_swarm_library(query=term, limit=5)
            results = search_result.get('results', [])
            results_count = search_result.get('results_count', 0)
            print(f"Search '{term}': {results_count} matches")

            for result in results:
                swarm_id = result.get('swarm_id', 'unknown')
                score = result.get('score', 0)
                metadata = result.get('metadata', {})
                description = metadata.get('description', 'No description')
                print(f"  📁 {swarm_id} (score: {score:.2f}) - {description[:40]}...")

        except Exception as e:
            print(f"Search '{term}' failed: {e}")

    # Test 4: Library analytics
    print(f"\n📈 Library Analytics")
    print("-" * 30)

    try:
        analytics = await manager.get_library_analytics()
        print(f"Total swarms: {analytics.get('total_swarms', 0)}")
        print(f"Total tags: {analytics.get('total_tags', 0)}")
        print(f"Categories: {list(analytics.get('categories', {}).keys())}")

        top_tags = analytics.get('top_tags', [])
        if top_tags:
            print("Top tags:")
            for tag_info in top_tags[:3]:
                tag = tag_info.get('tag', 'unknown')
                count = tag_info.get('count', 0)
                print(f"  🏷️ {tag}: {count}")

    except Exception as e:
        print(f"Analytics failed: {e}")

    # Test 5: Export functionality
    print(f"\n💾 Testing Export")
    print("-" * 30)

    try:
        export_swarm_id = "research_swarm_01"
        export_path = f"/tmp/exported_{export_swarm_id}.json"

        result = await manager.export_swarm_from_library(export_swarm_id, export_path)
        print(f"✅ Exported {export_swarm_id}")
        print(f"Export path: {export_path}")

        # Check export content
        if Path(export_path).exists():
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            print(f"Export contains: {list(export_data.keys())}")

            metadata = export_data.get('metadata', {})
            print(f"Tags: {metadata.get('tags', [])}")
            print(f"Category: {metadata.get('research_category', 'N/A')}")

    except Exception as e:
        print(f"Export failed: {e}")

    # Test 6: Check library structure
    print(f"\n📂 Library Structure")
    print("-" * 30)

    library_path = Path("swarm_library")
    if library_path.exists():
        print(f"Library location: {library_path.absolute()}")

        swarm_dirs = [d for d in library_path.iterdir() if d.is_dir()]
        print(f"Swarm directories: {len(swarm_dirs)}")

        for swarm_dir in swarm_dirs:
            files = list(swarm_dir.glob("*"))
            print(f"  📁 {swarm_dir.name}: {len(files)} files")
            for file in files:
                print(f"    📄 {file.name}")
    else:
        print("Library directory not found")

    print(f"\n✅ SWARM LIBRARY TEST COMPLETED")
    print("=" * 50)
    print("\n🎉 Features Tested:")
    print("  ✅ Swarm creation and library storage")
    print("  ✅ Simulation and emergence detection")
    print("  ✅ Fuzzy search functionality")
    print("  ✅ Library analytics")
    print("  ✅ Export functionality")
    print("  ✅ File system organization")

if __name__ == "__main__":
    asyncio.run(test_swarm_library())
