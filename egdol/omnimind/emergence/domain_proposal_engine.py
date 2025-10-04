"""
Domain Proposal Engine
Autonomous domain expansion system that allows personalities to collaboratively identify gaps
and propose entirely new domains beyond the current strategic/historical/legal/mystical quadrants.
"""

import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import sqlite3
from collections import defaultdict, deque
import statistics
import numpy as np

from ..conversational.personality_framework import Personality, PersonalityType
from ..civilization.multi_agent_system import CivilizationAgent, AgentMessage, MessageType


class DomainType(Enum):
    """Types of domains that can be proposed."""
    ECONOMIC = "economic"
    LINGUISTIC = "linguistic"
    ECOLOGICAL = "ecological"
    METAPHYSICAL = "metaphysical"
    TECHNOLOGICAL = "technological"
    SOCIAL = "social"
    PSYCHOLOGICAL = "psychological"
    AESTHETIC = "aesthetic"
    ETHICAL = "ethical"
    POLITICAL = "political"
    SCIENTIFIC = "scientific"
    ARTISTIC = "artistic"
    PHILOSOPHICAL = "philosophical"
    CULTURAL = "cultural"
    SPIRITUAL = "spiritual"


class ProposalStatus(Enum):
    """Status of domain proposals."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    DEPRECATED = "deprecated"


class DomainComplexity(Enum):
    """Complexity levels for domain implementation."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


@dataclass
class DomainProposal:
    """A proposal for a new domain."""
    proposal_id: str
    domain_name: str
    domain_type: DomainType
    description: str
    rationale: str
    complexity: DomainComplexity
    proposer_agent_id: str
    supporting_agents: List[str] = field(default_factory=list)
    opposing_agents: List[str] = field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    implementation_plan: Dict[str, Any] = field(default_factory=dict)
    testing_requirements: List[str] = field(default_factory=list)
    integration_dependencies: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    votes: Dict[str, str] = field(default_factory=dict)  # agent_id -> vote (support/oppose)
    discussion_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "domain_name": self.domain_name,
            "domain_type": self.domain_type.value,
            "description": self.description,
            "rationale": self.rationale,
            "complexity": self.complexity.value,
            "proposer_agent_id": self.proposer_agent_id,
            "supporting_agents": self.supporting_agents,
            "opposing_agents": self.opposing_agents,
            "status": self.status.value,
            "implementation_plan": self.implementation_plan,
            "testing_requirements": self.testing_requirements,
            "integration_dependencies": self.integration_dependencies,
            "success_metrics": self.success_metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "votes": self.votes,
            "discussion_log": self.discussion_log
        }


@dataclass
class DomainImplementation:
    """Implementation of an approved domain."""
    implementation_id: str
    proposal_id: str
    domain_name: str
    domain_type: DomainType
    implementation_status: str
    code_modules: List[str] = field(default_factory=list)
    test_suites: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_tested: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "implementation_id": self.implementation_id,
            "proposal_id": self.proposal_id,
            "domain_name": self.domain_name,
            "domain_type": self.domain_type.value,
            "implementation_status": self.implementation_status,
            "code_modules": self.code_modules,
            "test_suites": self.test_suites,
            "integration_points": self.integration_points,
            "performance_metrics": self.performance_metrics,
            "error_logs": self.error_logs,
            "created_at": self.created_at.isoformat(),
            "last_tested": self.last_tested.isoformat() if self.last_tested else None
        }


class DomainProposalEngine:
    """Engine for autonomous domain expansion."""
    
    def __init__(self, db_path: str = "domain_proposals.db"):
        self.db_path = db_path
        self.proposals: Dict[str, DomainProposal] = {}
        self.implementations: Dict[str, DomainImplementation] = {}
        self.agents: Dict[str, CivilizationAgent] = {}
        self.domain_patterns: Dict[str, List[str]] = defaultdict(list)
        self.collaboration_networks: Dict[str, Set[str]] = defaultdict(set)
        self._init_database()
    
    def _init_database(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create proposals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_proposals (
                proposal_id TEXT PRIMARY KEY,
                domain_name TEXT NOT NULL,
                domain_type TEXT NOT NULL,
                description TEXT NOT NULL,
                rationale TEXT NOT NULL,
                complexity TEXT NOT NULL,
                proposer_agent_id TEXT NOT NULL,
                supporting_agents TEXT,
                opposing_agents TEXT,
                status TEXT NOT NULL,
                implementation_plan TEXT,
                testing_requirements TEXT,
                integration_dependencies TEXT,
                success_metrics TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                votes TEXT,
                discussion_log TEXT
            )
        ''')
        
        # Create implementations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_implementations (
                implementation_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                domain_name TEXT NOT NULL,
                domain_type TEXT NOT NULL,
                implementation_status TEXT NOT NULL,
                code_modules TEXT,
                test_suites TEXT,
                integration_points TEXT,
                performance_metrics TEXT,
                error_logs TEXT,
                created_at TEXT NOT NULL,
                last_tested TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_agents(self, agents: Dict[str, CivilizationAgent]):
        """Register agents for domain proposal collaboration."""
        self.agents.update(agents)
        
        # Build collaboration networks based on personality types
        for agent_id, agent in agents.items():
            personality_type = agent.personality.personality_type
            for other_agent_id, other_agent in agents.items():
                if agent_id != other_agent_id:
                    other_personality_type = other_agent.personality.personality_type
                    # Create collaboration networks based on complementary personalities
                    if self._are_complementary_personalities(personality_type, other_personality_type):
                        self.collaboration_networks[agent_id].add(other_agent_id)
    
    def _are_complementary_personalities(self, type1: PersonalityType, type2: PersonalityType) -> bool:
        """Check if two personality types are complementary for domain proposals."""
        complementary_pairs = [
            (PersonalityType.STRATEGOS, PersonalityType.ARCHIVIST),
            (PersonalityType.LAWMAKER, PersonalityType.ORACLE),
            (PersonalityType.STRATEGOS, PersonalityType.LAWMAKER),
            (PersonalityType.ARCHIVIST, PersonalityType.ORACLE)
        ]
        return (type1, type2) in complementary_pairs or (type2, type1) in complementary_pairs
    
    def propose_domain(self, 
                      domain_name: str,
                      domain_type: DomainType,
                      description: str,
                      rationale: str,
                      complexity: DomainComplexity,
                      proposer_agent_id: str) -> str:
        """Propose a new domain."""
        proposal_id = str(uuid.uuid4())
        
        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(domain_type, complexity)
        
        # Generate testing requirements
        testing_requirements = self._generate_testing_requirements(domain_type, complexity)
        
        # Generate integration dependencies
        integration_dependencies = self._generate_integration_dependencies(domain_type)
        
        # Generate success metrics
        success_metrics = self._generate_success_metrics(domain_type)
        
        proposal = DomainProposal(
            proposal_id=proposal_id,
            domain_name=domain_name,
            domain_type=domain_type,
            description=description,
            rationale=rationale,
            complexity=complexity,
            proposer_agent_id=proposer_agent_id,
            implementation_plan=implementation_plan,
            testing_requirements=testing_requirements,
            integration_dependencies=integration_dependencies,
            success_metrics=success_metrics
        )
        
        self.proposals[proposal_id] = proposal
        self._save_proposal(proposal)
        
        # Notify collaboration network
        self._notify_collaboration_network(proposal)
        
        return proposal_id
    
    def _generate_implementation_plan(self, domain_type: DomainType, complexity: DomainComplexity) -> Dict[str, Any]:
        """Generate implementation plan for domain."""
        base_modules = {
            "reasoning_engine": f"{domain_type.value}_reasoning.py",
            "personality_adaptations": f"{domain_type.value}_personalities.py",
            "integration_layer": f"{domain_type.value}_integration.py",
            "testing_framework": f"test_{domain_type.value}.py"
        }
        
        complexity_multipliers = {
            DomainComplexity.BASIC: 1,
            DomainComplexity.INTERMEDIATE: 2,
            DomainComplexity.ADVANCED: 3,
            DomainComplexity.EXPERT: 4,
            DomainComplexity.MASTER: 5
        }
        
        multiplier = complexity_multipliers[complexity]
        
        return {
            "base_modules": base_modules,
            "estimated_complexity": multiplier,
            "development_phases": [
                f"Phase {i+1}: {module}" 
                for i, module in enumerate(base_modules.keys())
            ],
            "integration_points": [
                "conversational_layer",
                "civilization_system",
                "emergence_control"
            ],
            "estimated_development_time": f"{multiplier * 2} days"
        }
    
    def _generate_testing_requirements(self, domain_type: DomainType, complexity: DomainComplexity) -> List[str]:
        """Generate testing requirements for domain."""
        base_tests = [
            f"Test {domain_type.value} reasoning accuracy",
            f"Test {domain_type.value} personality integration",
            f"Test {domain_type.value} cross-domain compatibility",
            f"Test {domain_type.value} performance benchmarks"
        ]
        
        complexity_tests = {
            DomainComplexity.BASIC: [],
            DomainComplexity.INTERMEDIATE: ["Test edge case handling"],
            DomainComplexity.ADVANCED: ["Test edge case handling", "Test scalability"],
            DomainComplexity.EXPERT: ["Test edge case handling", "Test scalability", "Test fault tolerance"],
            DomainComplexity.MASTER: ["Test edge case handling", "Test scalability", "Test fault tolerance", "Test emergent behavior"]
        }
        
        return base_tests + complexity_tests[complexity]
    
    def _generate_integration_dependencies(self, domain_type: DomainType) -> List[str]:
        """Generate integration dependencies for domain."""
        return [
            "conversational.personality_framework",
            "civilization.multi_agent_system",
            "emergence.domain_proposal_engine",
            "conversational.reasoning_engine"
        ]
    
    def _generate_success_metrics(self, domain_type: DomainType) -> List[str]:
        """Generate success metrics for domain."""
        return [
            f"{domain_type.value}_reasoning_accuracy > 0.85",
            f"{domain_type.value}_integration_success_rate > 0.95",
            f"{domain_type.value}_performance_latency < 100ms",
            f"{domain_type.value}_user_satisfaction > 0.8"
        ]
    
    def _notify_collaboration_network(self, proposal: DomainProposal):
        """Notify collaboration network about new proposal."""
        proposer_id = proposal.proposer_agent_id
        collaborators = self.collaboration_networks.get(proposer_id, set())
        
        for collaborator_id in collaborators:
            if collaborator_id in self.agents:
                # Send notification message
                message = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=proposer_id,
                    receiver_id=collaborator_id,
                    message_type=MessageType.COLLABORATION,
                    content=f"New domain proposal: {proposal.domain_name} - {proposal.description}",
                    priority=5
                )
                # In a real implementation, this would be sent through the message bus
                print(f"📢 Notification to {collaborator_id}: {message.content}")
    
    def vote_on_proposal(self, proposal_id: str, agent_id: str, vote: str) -> bool:
        """Vote on a domain proposal."""
        if proposal_id not in self.proposals:
            return False
        
        if vote not in ["support", "oppose"]:
            return False
        
        proposal = self.proposals[proposal_id]
        proposal.votes[agent_id] = vote
        proposal.updated_at = datetime.now()
        
        # Update supporting/opposing agents lists
        if vote == "support":
            if agent_id not in proposal.supporting_agents:
                proposal.supporting_agents.append(agent_id)
            if agent_id in proposal.opposing_agents:
                proposal.opposing_agents.remove(agent_id)
        else:
            if agent_id not in proposal.opposing_agents:
                proposal.opposing_agents.append(agent_id)
            if agent_id in proposal.supporting_agents:
                proposal.supporting_agents.remove(agent_id)
        
        # Check if proposal should be approved
        self._check_proposal_approval(proposal)
        
        self._save_proposal(proposal)
        return True
    
    def _check_proposal_approval(self, proposal: DomainProposal):
        """Check if proposal should be approved based on votes."""
        total_votes = len(proposal.votes)
        if total_votes < 2:  # Need at least 2 votes
            return
        
        support_count = len(proposal.supporting_agents)
        oppose_count = len(proposal.opposing_agents)
        
        # Approval criteria: 75% support or unanimous support
        support_ratio = support_count / total_votes if total_votes > 0 else 0
        
        if support_ratio >= 0.75 or (support_count > 0 and oppose_count == 0):
            proposal.status = ProposalStatus.APPROVED
            self._initiate_implementation(proposal)
        elif oppose_count > support_count:
            proposal.status = ProposalStatus.REJECTED
    
    def _initiate_implementation(self, proposal: DomainProposal):
        """Initiate implementation of approved proposal."""
        implementation_id = str(uuid.uuid4())
        
        implementation = DomainImplementation(
            implementation_id=implementation_id,
            proposal_id=proposal.proposal_id,
            domain_name=proposal.domain_name,
            domain_type=proposal.domain_type,
            implementation_status="pending"
        )
        
        self.implementations[implementation_id] = implementation
        self._save_implementation(implementation)
        
        # Generate scaffolding code
        self._generate_domain_scaffolding(proposal, implementation)
    
    def _generate_domain_scaffolding(self, proposal: DomainProposal, implementation: DomainImplementation):
        """Generate scaffolding code for domain implementation."""
        domain_name = proposal.domain_name.lower().replace(" ", "_")
        domain_type = proposal.domain_type.value
        
        # Generate reasoning module
        reasoning_code = f'''
"""
{proposal.domain_name} Domain Reasoning Module
Auto-generated scaffolding for {proposal.description}
"""

import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class {domain_name.title()}ReasoningType(Enum):
    """Types of {domain_name} reasoning."""
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"

@dataclass
class {domain_name.title()}Context:
    """Context for {domain_name} reasoning."""
    domain_specific_data: Dict[str, Any]
    reasoning_type: {domain_name.title()}ReasoningType
    complexity_level: float
    timestamp: datetime

class {domain_name.title()}ReasoningEngine:
    """Reasoning engine for {domain_name} domain."""
    
    def __init__(self):
        self.domain_type = "{domain_type}"
        self.complexity = "{proposal.complexity.value}"
    
    def process_{domain_name}_query(self, query: str, context: {domain_name.title()}Context) -> Dict[str, Any]:
        """Process {domain_name} domain query."""
        # TODO: Implement domain-specific reasoning logic
        return {{
            "success": True,
            "response": f"Processing {domain_name} query: {{query}}",
            "reasoning_trace": {{}},
            "confidence": 0.8,
            "domain_insights": []
        }}
    
    def analyze_{domain_name}_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze {domain_name} patterns."""
        # TODO: Implement pattern analysis
        return []
    
    def generate_{domain_name}_recommendations(self, context: {domain_name.title()}Context) -> List[str]:
        """Generate {domain_name} recommendations."""
        # TODO: Implement recommendation generation
        return []
'''
        
        # Generate test module
        test_code = f'''
"""
Tests for {proposal.domain_name} Domain
Auto-generated test suite
"""

import unittest
from {domain_name}_reasoning import {domain_name.title()}ReasoningEngine, {domain_name.title()}Context, {domain_name.title()}ReasoningType

class Test{domain_name.title()}Domain(unittest.TestCase):
    """Test {domain_name} domain functionality."""
    
    def setUp(self):
        self.reasoning_engine = {domain_name.title()}ReasoningEngine()
    
    def test_{domain_name}_query_processing(self):
        """Test {domain_name} query processing."""
        context = {domain_name.title()}Context(
            domain_specific_data={{"test": "data"}},
            reasoning_type={domain_name.title()}ReasoningType.ANALYSIS,
            complexity_level=0.5,
            timestamp=datetime.now()
        )
        
        result = self.reasoning_engine.process_{domain_name}_query("test query", context)
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
    
    def test_{domain_name}_pattern_analysis(self):
        """Test {domain_name} pattern analysis."""
        data = {{"pattern": "test"}}
        patterns = self.reasoning_engine.analyze_{domain_name}_patterns(data)
        self.assertIsInstance(patterns, list)
    
    def test_{domain_name}_recommendations(self):
        """Test {domain_name} recommendation generation."""
        context = {domain_name.title()}Context(
            domain_specific_data={{"test": "data"}},
            reasoning_type={domain_name.title()}ReasoningType.ANALYSIS,
            complexity_level=0.5,
            timestamp=datetime.now()
        )
        
        recommendations = self.reasoning_engine.generate_{domain_name}_recommendations(context)
        self.assertIsInstance(recommendations, list)

if __name__ == "__main__":
    unittest.main()
'''
        
        # Store generated code
        implementation.code_modules = [
            f"{domain_name}_reasoning.py",
            f"test_{domain_name}.py"
        ]
        implementation.test_suites = [f"test_{domain_name}.py"]
        implementation.implementation_status = "scaffolding_generated"
        
        # Save code to files (in a real implementation)
        print(f"🔧 Generated scaffolding for {proposal.domain_name}")
        print(f"   - Reasoning module: {domain_name}_reasoning.py")
        print(f"   - Test suite: test_{domain_name}.py")
        
        self._save_implementation(implementation)
    
    def get_proposals(self, status: Optional[ProposalStatus] = None) -> List[DomainProposal]:
        """Get domain proposals, optionally filtered by status."""
        if status is None:
            return list(self.proposals.values())
        return [p for p in self.proposals.values() if p.status == status]
    
    def get_implementations(self) -> List[DomainImplementation]:
        """Get domain implementations."""
        return list(self.implementations.values())
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about domain proposals and implementations."""
        total_proposals = len(self.proposals)
        approved_proposals = len([p for p in self.proposals.values() if p.status == ProposalStatus.APPROVED])
        implemented_domains = len(self.implementations)
        
        domain_type_counts = defaultdict(int)
        for proposal in self.proposals.values():
            domain_type_counts[proposal.domain_type.value] += 1
        
        return {
            "total_proposals": total_proposals,
            "approved_proposals": approved_proposals,
            "implemented_domains": implemented_domains,
            "approval_rate": approved_proposals / total_proposals if total_proposals > 0 else 0,
            "domain_type_distribution": dict(domain_type_counts),
            "active_collaboration_networks": len(self.collaboration_networks)
        }
    
    def _save_proposal(self, proposal: DomainProposal):
        """Save proposal to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO domain_proposals 
            (proposal_id, domain_name, domain_type, description, rationale, complexity,
             proposer_agent_id, supporting_agents, opposing_agents, status,
             implementation_plan, testing_requirements, integration_dependencies,
             success_metrics, created_at, updated_at, votes, discussion_log)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            proposal.proposal_id, proposal.domain_name, proposal.domain_type.value,
            proposal.description, proposal.rationale, proposal.complexity.value,
            proposal.proposer_agent_id, json.dumps(proposal.supporting_agents),
            json.dumps(proposal.opposing_agents), proposal.status.value,
            json.dumps(proposal.implementation_plan), json.dumps(proposal.testing_requirements),
            json.dumps(proposal.integration_dependencies), json.dumps(proposal.success_metrics),
            proposal.created_at.isoformat(), proposal.updated_at.isoformat(),
            json.dumps(proposal.votes), json.dumps(proposal.discussion_log)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_implementation(self, implementation: DomainImplementation):
        """Save implementation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO domain_implementations 
            (implementation_id, proposal_id, domain_name, domain_type, implementation_status,
             code_modules, test_suites, integration_points, performance_metrics,
             error_logs, created_at, last_tested)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            implementation.implementation_id, implementation.proposal_id,
            implementation.domain_name, implementation.domain_type.value,
            implementation.implementation_status, json.dumps(implementation.code_modules),
            json.dumps(implementation.test_suites), json.dumps(implementation.integration_points),
            json.dumps(implementation.performance_metrics), json.dumps(implementation.error_logs),
            implementation.created_at.isoformat(),
            implementation.last_tested.isoformat() if implementation.last_tested else None
        ))
        
        conn.commit()
        conn.close()
