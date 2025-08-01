"""
ELCS Framework Core Components
==============================

Core implementations for Dynamic Emergence Networks and Multi-Agent Swarm Intelligence.
"""

# Re-export main components for easier imports
from .dynamic_emergence_networks import (
    DynamicEmergenceNetwork,
    ProcessScale,
    ProcessSignature,
    ProcessEntity,
    EmergenceEvent,
    ScaleBridge
)

from .multi_agent_swarm import (
    SwarmIntelligence,
    SwarmAgent,
    AgentCapabilities,
    SpecializationRole,
    EmergenceBehaviorDetector,
    CollectiveDecisionMaker,
    CollectiveDecisionType,
    SwarmCommunicationProtocol
)

__all__ = [
    # Dynamic Emergence Networks
    'DynamicEmergenceNetwork',
    'ProcessScale',
    'ProcessSignature', 
    'ProcessEntity',
    'EmergenceEvent',
    'ScaleBridge',
    # Multi-Agent Swarm
    'SwarmIntelligence',
    'SwarmAgent',
    'AgentCapabilities',
    'SpecializationRole',
    'EmergenceBehaviorDetector',
    'CollectiveDecisionMaker',
    'CollectiveDecisionType',
    'SwarmCommunicationProtocol'
]
