"""
ELCS Swarm Library - Persistent Storage and Search System
========================================================

This module provides comprehensive library functionality for storing,
organizing, and searching swarm intelligence data with metadata,
behavioral patterns, and fuzzy search capabilities.

Author: Tyler Blaine Hall, Claude Sonnet 4
License: MIT
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SwarmMetadata:
    """Comprehensive metadata for a swarm instance."""
    swarm_id: str
    creation_timestamp: float
    last_updated: float
    creator: str
    description: str
    tags: List[str]

    # Configuration
    agent_count: int
    emergence_threshold: float
    stability_window: int

    # Performance metrics
    peak_performance: float
    average_performance: float
    total_cycles: int
    successful_cycles: int

    # Behavioral patterns
    dominant_behaviors: List[str]
    specialization_patterns: Dict[str, int]
    coordination_metrics: Dict[str, float]

    # Research context
    research_category: str
    experiment_notes: str
    success_indicators: List[str]
    failure_patterns: List[str]

    # File locations
    data_folder: str
    config_file: str
    simulation_data: str
    emergence_data: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmMetadata':
        """Create from dictionary."""
        return cls(**data)


class SwarmLibrary:
    """
    Persistent library system for swarm intelligence data.

    Features:
    - Hierarchical folder organization
    - Rich metadata storage
    - Fuzzy search capabilities
    - Behavioral pattern indexing
    - Performance analytics
    - Export/import functionality
    """

    def __init__(self, library_root: str = "swarm_library"):
        """Initialize swarm library system."""
        self.library_root = Path(library_root)
        self.metadata_index: Dict[str, SwarmMetadata] = {}
        self.behavior_index: Dict[str, List[str]] = {}  # behavior_type -> [swarm_ids]
        self.performance_index: Dict[str, List[Tuple[str, float]]] = {}  # category -> [(swarm_id, performance)]

        # Create library structure
        self._ensure_library_structure()

        # Load existing metadata index
        self._load_metadata_index()

        logger.info(f"SwarmLibrary initialized at {self.library_root}")

    def _ensure_library_structure(self):
        """Create necessary directory structure."""
        directories = [
            self.library_root,
            self.library_root / "swarms",
            self.library_root / "metadata",
            self.library_root / "exports",
            self.library_root / "indexes",
            self.library_root / "analytics"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_metadata_index(self):
        """Load the metadata index from disk."""
        index_file = self.library_root / "indexes" / "metadata_index.json"

        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)

                for swarm_id, metadata_dict in index_data.items():
                    self.metadata_index[swarm_id] = SwarmMetadata.from_dict(metadata_dict)

                # Rebuild behavioral and performance indexes
                self._rebuild_indexes()

                logger.info(f"Loaded {len(self.metadata_index)} swarms from metadata index")

            except Exception as e:
                logger.error(f"Failed to load metadata index: {e}")
                self.metadata_index = {}

    def _save_metadata_index(self):
        """Save the metadata index to disk."""
        index_file = self.library_root / "indexes" / "metadata_index.json"

        try:
            index_data = {
                swarm_id: metadata.to_dict()
                for swarm_id, metadata in self.metadata_index.items()
            }

            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metadata index: {e}")

    def _rebuild_indexes(self):
        """Rebuild behavioral and performance indexes from metadata."""
        self.behavior_index.clear()
        self.performance_index.clear()

        for swarm_id, metadata in self.metadata_index.items():
            # Build behavior index
            for behavior in metadata.dominant_behaviors:
                if behavior not in self.behavior_index:
                    self.behavior_index[behavior] = []
                self.behavior_index[behavior].append(swarm_id)

            # Build performance index
            category = metadata.research_category
            if category not in self.performance_index:
                self.performance_index[category] = []
            self.performance_index[category].append((swarm_id, metadata.peak_performance))

    def store_swarm(self,
                   swarm_id: str,
                   swarm_data: Dict[str, Any],
                   simulation_results: Dict[str, Any],
                   emergence_analysis: Dict[str, Any],
                   metadata_override: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a complete swarm with all its data and metadata.

        Args:
            swarm_id: Unique identifier for the swarm
            swarm_data: Complete swarm configuration and state
            simulation_results: All simulation results and metrics
            emergence_analysis: Emergence detection and behavioral data
            metadata_override: Optional metadata overrides

        Returns:
            Storage path for the swarm
        """
        # Create swarm folder
        swarm_folder = self.library_root / "swarms" / swarm_id
        swarm_folder.mkdir(parents=True, exist_ok=True)

        # Generate comprehensive metadata
        metadata = self._generate_metadata(
            swarm_id, swarm_data, simulation_results, emergence_analysis, metadata_override
        )

        # Store data files
        data_files = {
            "swarm_config.json": swarm_data,
            "simulation_results.json": simulation_results,
            "emergence_analysis.json": emergence_analysis,
            "metadata.json": metadata.to_dict()
        }

        file_paths = {}
        for filename, data in data_files.items():
            file_path = swarm_folder / filename
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            file_paths[filename] = str(file_path)

        # Store binary data if present
        if 'agent_states' in swarm_data:
            binary_path = swarm_folder / "agent_states.pkl"
            with open(binary_path, 'wb') as f:
                pickle.dump(swarm_data['agent_states'], f)
            file_paths["agent_states.pkl"] = str(binary_path)

        # Update metadata with file paths
        metadata.data_folder = str(swarm_folder)
        metadata.config_file = file_paths["swarm_config.json"]
        metadata.simulation_data = file_paths["simulation_results.json"]
        metadata.emergence_data = file_paths["emergence_analysis.json"]

        # Add to indexes
        self.metadata_index[swarm_id] = metadata
        self._update_indexes(swarm_id, metadata)

        # Save updated index
        self._save_metadata_index()

        logger.info(f"Stored swarm {swarm_id} in library at {swarm_folder}")
        return str(swarm_folder)

    def _generate_metadata(self,
                          swarm_id: str,
                          swarm_data: Dict[str, Any],
                          simulation_results: Dict[str, Any],
                          emergence_analysis: Dict[str, Any],
                          metadata_override: Optional[Dict[str, Any]] = None) -> SwarmMetadata:
        """Generate comprehensive metadata from swarm data."""

        # Extract basic configuration
        config = swarm_data.get('configuration', {})
        agent_count = swarm_data.get('agent_count', 0)

        # Extract performance metrics
        overall_metrics = simulation_results.get('overall_metrics', {})
        peak_performance = overall_metrics.get('peak_performance', 0.0)
        average_performance = overall_metrics.get('average_performance', 0.0)
        total_cycles = overall_metrics.get('total_cycles', 0)
        successful_cycles = overall_metrics.get('successful_cycles', 0)

        # Extract behavioral patterns
        behaviors_detected = emergence_analysis.get('top_behaviors', [])
        dominant_behaviors = [b.get('type', 'unknown') for b in behaviors_detected]

        # Extract specialization patterns
        analytics = simulation_results.get('final_analytics', {})
        swarm_state = analytics.get('swarm_state', {})
        role_distribution = swarm_state.get('role_distribution', {})

        # Extract coordination metrics
        emergence_metrics = emergence_analysis.get('emergence_metrics', {})

        # Default values with overrides
        defaults = {
            'creation_timestamp': time.time(),
            'last_updated': time.time(),
            'creator': 'ELCS Framework',
            'description': f'Multi-agent swarm with {agent_count} agents',
            'tags': ['generated', 'simulation'],
            'research_category': 'general',
            'experiment_notes': 'Auto-generated from simulation',
            'success_indicators': [],
            'failure_patterns': []
        }

        if metadata_override:
            defaults.update(metadata_override)

        return SwarmMetadata(
            swarm_id=swarm_id,
            creation_timestamp=defaults['creation_timestamp'],
            last_updated=defaults['last_updated'],
            creator=defaults['creator'],
            description=defaults['description'],
            tags=defaults['tags'],

            agent_count=agent_count,
            emergence_threshold=config.get('emergence_threshold', 0.6),
            stability_window=config.get('stability_window', 10),

            peak_performance=peak_performance,
            average_performance=average_performance,
            total_cycles=total_cycles,
            successful_cycles=successful_cycles,

            dominant_behaviors=dominant_behaviors,
            specialization_patterns=role_distribution,
            coordination_metrics=emergence_metrics,

            research_category=defaults['research_category'],
            experiment_notes=defaults['experiment_notes'],
            success_indicators=defaults['success_indicators'],
            failure_patterns=defaults['failure_patterns'],

            data_folder="",  # Will be set after storage
            config_file="",
            simulation_data="",
            emergence_data=""
        )

    def _update_indexes(self, swarm_id: str, metadata: SwarmMetadata):
        """Update behavioral and performance indexes."""
        # Update behavior index
        for behavior in metadata.dominant_behaviors:
            if behavior not in self.behavior_index:
                self.behavior_index[behavior] = []
            if swarm_id not in self.behavior_index[behavior]:
                self.behavior_index[behavior].append(swarm_id)

        # Update performance index
        category = metadata.research_category
        if category not in self.performance_index:
            self.performance_index[category] = []

        # Remove existing entry and add updated one
        self.performance_index[category] = [
            (sid, perf) for sid, perf in self.performance_index[category]
            if sid != swarm_id
        ]
        self.performance_index[category].append((swarm_id, metadata.peak_performance))

    def load_swarm(self, swarm_id: str) -> Optional[Dict[str, Any]]:
        """Load complete swarm data from library."""
        if swarm_id not in self.metadata_index:
            return None

        metadata = self.metadata_index[swarm_id]
        swarm_folder = Path(metadata.data_folder)

        if not swarm_folder.exists():
            logger.error(f"Swarm folder not found: {swarm_folder}")
            return None

        try:
            # Load JSON data
            swarm_data = {}
            json_files = ['swarm_config.json', 'simulation_results.json', 'emergence_analysis.json']

            for filename in json_files:
                file_path = swarm_folder / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        key = filename.replace('.json', '')
                        swarm_data[key] = json.load(f)

            # Load binary data if present
            binary_path = swarm_folder / "agent_states.pkl"
            if binary_path.exists():
                with open(binary_path, 'rb') as f:
                    swarm_data['agent_states'] = pickle.load(f)

            swarm_data['metadata'] = metadata.to_dict()
            return swarm_data

        except Exception as e:
            logger.error(f"Failed to load swarm {swarm_id}: {e}")
            return None

    def search_swarms(self,
                     query: str = "",
                     tags: Optional[List[str]] = None,
                     behavior_types: Optional[List[str]] = None,
                     performance_range: Optional[Tuple[float, float]] = None,
                     research_category: Optional[str] = None,
                     limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fuzzy search for swarms based on various criteria.

        Args:
            query: Fuzzy text search in descriptions, notes, etc.
            tags: Filter by tags
            behavior_types: Filter by dominant behaviors
            performance_range: Filter by performance range (min, max)
            research_category: Filter by research category
            limit: Maximum number of results

        Returns:
            List of matching swarm metadata with relevance scores
        """
        results = []

        for swarm_id, metadata in self.metadata_index.items():
            relevance_score = 0.0

            # Text search
            if query:
                searchable_text = f"{metadata.description} {metadata.experiment_notes} {' '.join(metadata.tags)}"
                relevance_score += self._fuzzy_text_match(query.lower(), searchable_text.lower())

            # Tag filtering
            if tags:
                tag_matches = len(set(tags) & set(metadata.tags))
                if tag_matches > 0:
                    relevance_score += tag_matches / len(tags) * 0.3
                else:
                    continue  # Skip if no tag matches when tags specified

            # Behavior filtering
            if behavior_types:
                behavior_matches = len(set(behavior_types) & set(metadata.dominant_behaviors))
                if behavior_matches > 0:
                    relevance_score += behavior_matches / len(behavior_types) * 0.4
                else:
                    continue  # Skip if no behavior matches when behaviors specified

            # Performance filtering
            if performance_range:
                min_perf, max_perf = performance_range
                if min_perf <= metadata.peak_performance <= max_perf:
                    relevance_score += 0.2
                else:
                    continue  # Skip if outside performance range

            # Category filtering
            if research_category and metadata.research_category != research_category:
                continue

            # Add to results if any relevance
            if relevance_score > 0 or not any([query, tags, behavior_types, performance_range, research_category]):
                result = {
                    'swarm_id': swarm_id,
                    'score': relevance_score,
                    'metadata': metadata.to_dict()
                }
                results.append(result)

        # Sort by relevance and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def _fuzzy_text_match(self, query: str, text: str) -> float:
        """Simple fuzzy text matching with word overlap."""
        query_words = set(query.split())
        text_words = set(text.split())

        if not query_words:
            return 0.0

        # Calculate word overlap
        overlap = len(query_words & text_words)

        # Bonus for substring matches
        substring_bonus = 0.0
        for word in query_words:
            if word in text:
                substring_bonus += 0.1

        return (overlap / len(query_words)) + substring_bonus

    def get_behavior_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about behaviors across all swarms."""
        behavior_stats = {}

        for behavior, swarm_ids in self.behavior_index.items():
            performances = []
            agent_counts = []

            for swarm_id in swarm_ids:
                metadata = self.metadata_index[swarm_id]
                performances.append(metadata.peak_performance)
                agent_counts.append(metadata.agent_count)

            behavior_stats[behavior] = {
                'swarm_count': len(swarm_ids),
                'avg_performance': float(np.mean(performances)) if performances else 0.0,
                'best_performance': float(np.max(performances)) if performances else 0.0,
                'avg_agent_count': float(np.mean(agent_counts)) if agent_counts else 0.0,
                'example_swarms': swarm_ids[:3]  # Top 3 examples
            }

        return behavior_stats

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics across categories."""
        analytics = {}

        for category, swarm_performances in self.performance_index.items():
            performances = [perf for _, perf in swarm_performances]

            if performances:
                analytics[category] = {
                    'swarm_count': len(swarm_performances),
                    'avg_performance': float(np.mean(performances)),
                    'best_performance': float(np.max(performances)),
                    'worst_performance': float(np.min(performances)),
                    'performance_std': float(np.std(performances)),
                    'top_performers': sorted(swarm_performances, key=lambda x: x[1], reverse=True)[:3]
                }

        return analytics

    def export_swarm(self, swarm_id: str, export_path: Optional[str] = None) -> str:
        """Export a swarm to a portable format."""
        if swarm_id not in self.metadata_index:
            raise ValueError(f"Swarm {swarm_id} not found in library")

        if not export_path:
            export_file = self.library_root / "exports" / f"{swarm_id}_{int(time.time())}.json"
        else:
            export_file = Path(export_path)

        # Load complete swarm data
        swarm_data = self.load_swarm(swarm_id)
        if not swarm_data:
            raise ValueError(f"Failed to load swarm {swarm_id}")

        # Create export package
        export_package = {
            'export_version': '1.0',
            'export_timestamp': time.time(),
            'swarm_id': swarm_id,
            'swarm_data': swarm_data
        }

        # Save export
        with open(export_file, 'w') as f:
            json.dump(export_package, f, indent=2, default=str)

        logger.info(f"Exported swarm {swarm_id} to {export_file}")
        return str(export_file)

    def import_swarm(self, import_path: str, new_swarm_id: Optional[str] = None) -> str:
        """Import a swarm from exported format."""
        import_file = Path(import_path)

        if not import_file.exists():
            raise ValueError(f"Import file not found: {import_path}")

        try:
            with open(import_file, 'r') as f:
                export_package = json.load(f)

            original_swarm_id = export_package['swarm_id']
            swarm_data_package = export_package['swarm_data']

            # Use new ID or generate one if conflict
            target_swarm_id = new_swarm_id or original_swarm_id
            if target_swarm_id in self.metadata_index:
                target_swarm_id = f"{target_swarm_id}_{int(time.time())}"

            # Extract components
            swarm_config = swarm_data_package.get('swarm_config', {})
            simulation_results = swarm_data_package.get('simulation_results', {})
            emergence_analysis = swarm_data_package.get('emergence_analysis', {})

            # Update metadata for import
            metadata_override = {
                'creation_timestamp': time.time(),
                'tags': ['imported'] + swarm_data_package.get('metadata', {}).get('tags', [])
            }

            # Store in library
            self.store_swarm(
                target_swarm_id, swarm_config, simulation_results,
                emergence_analysis, metadata_override
            )

            logger.info(f"Imported swarm as {target_swarm_id} from {import_path}")
            return target_swarm_id

        except Exception as e:
            raise ValueError(f"Failed to import swarm: {e}")

    def delete_swarm(self, swarm_id: str) -> bool:
        """Delete a swarm from the library."""
        if swarm_id not in self.metadata_index:
            return False

        metadata = self.metadata_index[swarm_id]
        swarm_folder = Path(metadata.data_folder)

        try:
            # Remove files
            if swarm_folder.exists():
                import shutil
                shutil.rmtree(swarm_folder)

            # Remove from indexes
            del self.metadata_index[swarm_id]

            # Update behavior index
            for behavior, swarm_ids in self.behavior_index.items():
                if swarm_id in swarm_ids:
                    swarm_ids.remove(swarm_id)

            # Update performance index
            for category, swarm_performances in self.performance_index.items():
                self.performance_index[category] = [
                    (sid, perf) for sid, perf in swarm_performances if sid != swarm_id
                ]

            # Save updated index
            self._save_metadata_index()

            logger.info(f"Deleted swarm {swarm_id} from library")
            return True

        except Exception as e:
            logger.error(f"Failed to delete swarm {swarm_id}: {e}")
            return False

    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""
        total_swarms = len(self.metadata_index)

        if total_swarms == 0:
            return {'total_swarms': 0, 'message': 'Library is empty'}

        # Aggregate statistics
        all_performances = [m.peak_performance for m in self.metadata_index.values()]
        all_agent_counts = [m.agent_count for m in self.metadata_index.values()]
        all_categories = [m.research_category for m in self.metadata_index.values()]
        all_behaviors = []
        for m in self.metadata_index.values():
            all_behaviors.extend(m.dominant_behaviors)

        # Calculate storage usage
        total_size = 0
        for metadata in self.metadata_index.values():
            folder_path = Path(metadata.data_folder)
            if folder_path.exists():
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

        return {
            'total_swarms': total_swarms,
            'performance_stats': {
                'avg_performance': float(np.mean(all_performances)),
                'best_performance': float(np.max(all_performances)),
                'performance_range': [float(np.min(all_performances)), float(np.max(all_performances))]
            },
            'agent_stats': {
                'avg_agent_count': float(np.mean(all_agent_counts)),
                'total_agents_simulated': sum(all_agent_counts),
                'agent_count_range': [min(all_agent_counts), max(all_agent_counts)]
            },
            'category_distribution': {cat: all_categories.count(cat) for cat in set(all_categories)},
            'behavior_distribution': {beh: all_behaviors.count(beh) for beh in set(all_behaviors)},
            'storage_stats': {
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'avg_size_per_swarm_kb': round(total_size / total_swarms / 1024, 2) if total_swarms > 0 else 0
            },
            'creation_timeline': {
                'oldest': min(m.creation_timestamp for m in self.metadata_index.values()),
                'newest': max(m.creation_timestamp for m in self.metadata_index.values())
            }
        }
