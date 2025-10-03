"""
OmniMind Meta-Rule Discovery System
Autonomous discovery, validation, and application of meta-rules from conversation and simulation patterns.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
import json
import logging
from collections import defaultdict, Counter

from .conversational.reflexive_audit import ReflexiveAuditModule
from .conversational.meta_learning_engine import MetaLearningEngine
from .civilizational_feedback import CivilizationalInsight, MetaRuleCandidate


@dataclass
class PatternMatch:
    """Pattern match result from conversation or simulation analysis."""
    pattern_type: str
    pattern_text: str
    confidence: float
    source: str
    context: Dict[str, Any]
    frequency: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaRuleTemplate:
    """Template for generating meta-rules from patterns."""
    template_name: str
    pattern_matcher: str
    rule_generator: str
    validation_criteria: List[str]
    personality_applicability: List[str]
    confidence_threshold: float = 0.7


@dataclass
class MetaRuleValidation:
    """Validation result for a meta-rule candidate."""
    rule_candidate: MetaRuleCandidate
    validation_tests: List[Dict[str, Any]]
    simulation_validation: Dict[str, Any]
    logical_consistency: float
    empirical_support: float
    overall_score: float
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)


class MetaRuleDiscoveryEngine:
    """
    Autonomous meta-rule discovery system that extracts patterns from conversations
    and simulations, generates candidate meta-rules, and validates them through
    deterministic testing procedures.
    """
    
    def __init__(self, audit_module: ReflexiveAuditModule, meta_learning: MetaLearningEngine):
        """Initialize the meta-rule discovery engine."""
        self.audit_module = audit_module
        self.meta_learning = meta_learning
        
        # Pattern storage and analysis
        self.conversation_patterns: List[PatternMatch] = []
        self.simulation_patterns: List[PatternMatch] = []
        self.meta_rule_templates: List[MetaRuleTemplate] = []
        self.discovered_rules: List[MetaRuleCandidate] = []
        self.validated_rules: List[MetaRuleValidation] = []
        self.applied_rules: List[MetaRuleCandidate] = []
        
        # Discovery statistics
        self.discovery_cycles: int = 0
        self.patterns_analyzed: int = 0
        self.rules_generated: int = 0
        self.rules_validated: int = 0
        self.rules_applied: int = 0
        
        # Pattern analysis tools
        self.pattern_analyzers = {
            'conversation': self._extract_conversation_patterns,
            'simulation': self._extract_simulation_patterns,
            'reasoning': self._extract_conversation_patterns,  # Use existing method
            'fallback': self._extract_conversation_patterns   # Use existing method
        }
        
        # Meta-rule templates
        self._initialize_meta_rule_templates()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def discover_meta_rules_from_conversation(self, conversation_data: Dict[str, Any]) -> List[MetaRuleCandidate]:
        """
        Discover meta-rules from conversation data through pattern analysis.
        
        Args:
            conversation_data: Conversation turn data with reasoning traces
            
        Returns:
            List of discovered meta-rule candidates
        """
        try:
            # Extract patterns from conversation
            patterns = self._extract_conversation_patterns(conversation_data)
            
            # Analyze patterns for meta-rule potential
            meta_rule_candidates = self._analyze_patterns_for_meta_rules(patterns)
            
            # Validate candidates
            validated_candidates = self._validate_meta_rule_candidates(meta_rule_candidates)
            
            # Update discovery statistics
            self.discovery_cycles += 1
            self.patterns_analyzed += len(patterns)
            self.rules_generated += len(meta_rule_candidates)
            self.rules_validated += len(validated_candidates)
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"Error in meta-rule discovery: {e}")
            return []
    
    def discover_meta_rules_from_simulation(self, simulation_data: Dict[str, Any]) -> List[MetaRuleCandidate]:
        """
        Discover meta-rules from simulation data through civilizational pattern analysis.
        
        Args:
            simulation_data: Simulation results with civilizational patterns
            
        Returns:
            List of discovered meta-rule candidates
        """
        try:
            # Extract patterns from simulation
            patterns = self._extract_simulation_patterns(simulation_data)
            
            # Analyze patterns for meta-rule potential
            meta_rule_candidates = self._analyze_patterns_for_meta_rules(patterns)
            
            # Validate candidates
            validated_candidates = self._validate_meta_rule_candidates(meta_rule_candidates)
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"Error in simulation meta-rule discovery: {e}")
            return []
    
    def discover_meta_rules_from_insights(self, insights: List[CivilizationalInsight]) -> List[MetaRuleCandidate]:
        """
        Discover meta-rules from civilizational insights.
        
        Args:
            insights: List of civilizational insights
            
        Returns:
            List of discovered meta-rule candidates
        """
        try:
            meta_rule_candidates = []
            
            for insight in insights:
                if insight.meta_rule_candidate:
                    # Create meta-rule candidate from insight
                    candidate = MetaRuleCandidate(
                        rule_name=f"insight_rule_{len(meta_rule_candidates)}",
                        rule_pattern=insight.meta_rule_candidate,
                        confidence=insight.confidence,
                        source_insights=[insight],
                        validation_tests=[],
                        simulation_validation={},
                        personality_applicability=['Strategos', 'Archivist', 'Lawmaker', 'Oracle']
                    )
                    meta_rule_candidates.append(candidate)
            
            # Validate candidates
            validated_candidates = self._validate_meta_rule_candidates(meta_rule_candidates)
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"Error in insight meta-rule discovery: {e}")
            return []
    
    def _extract_conversation_patterns(self, conversation_data: Dict[str, Any]) -> List[PatternMatch]:
        """Extract patterns from conversation data."""
        patterns = []
        
        # Extract response patterns
        response = conversation_data.get('response', '')
        if response:
            response_pattern = self._analyze_response_pattern(response)
            if response_pattern:
                patterns.append(PatternMatch(
                    pattern_type='response',
                    pattern_text=response_pattern,
                    confidence=0.8,
                    source='conversation',
                    context={'response_length': len(response)}
                ))
        
        # Extract reasoning patterns
        reasoning_trace = conversation_data.get('reasoning_trace', [])
        if reasoning_trace:
            reasoning_pattern = self._analyze_reasoning_trace(reasoning_trace)
            if reasoning_pattern:
                patterns.append(PatternMatch(
                    pattern_type='reasoning',
                    pattern_text=reasoning_pattern,
                    confidence=0.7,
                    source='conversation',
                    context={'trace_length': len(reasoning_trace)}
                ))
        
        # Extract personality patterns
        personality = conversation_data.get('personality', '')
        if personality:
            personality_pattern = self._analyze_personality_pattern(personality, conversation_data)
            if personality_pattern:
                patterns.append(PatternMatch(
                    pattern_type='personality',
                    pattern_text=personality_pattern,
                    confidence=0.9,
                    source='conversation',
                    context={'personality': personality}
                ))
        
        return patterns
    
    def _extract_simulation_patterns(self, simulation_data: Dict[str, Any]) -> List[PatternMatch]:
        """Extract patterns from simulation data."""
        patterns = []
        
        # Extract social structure patterns
        social_structures = simulation_data.get('social_structures', {})
        if social_structures:
            social_pattern = self._analyze_social_structure_pattern(social_structures)
            if social_pattern:
                patterns.append(PatternMatch(
                    pattern_type='social_structure',
                    pattern_text=social_pattern,
                    confidence=0.8,
                    source='simulation',
                    context={'structure_complexity': len(social_structures)}
                ))
        
        # Extract economic patterns
        economic_patterns = simulation_data.get('economic_patterns', {})
        if economic_patterns:
            economic_pattern = self._analyze_economic_pattern(economic_patterns)
            if economic_pattern:
                patterns.append(PatternMatch(
                    pattern_type='economic',
                    pattern_text=economic_pattern,
                    confidence=0.7,
                    source='simulation',
                    context={'pattern_count': len(economic_patterns)}
                ))
        
        # Extract conflict resolution patterns
        conflict_resolution = simulation_data.get('conflict_resolution', {})
        if conflict_resolution:
            conflict_pattern = self._analyze_conflict_resolution_pattern(conflict_resolution)
            if conflict_pattern:
                patterns.append(PatternMatch(
                    pattern_type='conflict_resolution',
                    pattern_text=conflict_pattern,
                    confidence=0.9,
                    source='simulation',
                    context={'resolution_methods': len(conflict_resolution)}
                ))
        
        return patterns
    
    def _analyze_patterns_for_meta_rules(self, patterns: List[PatternMatch]) -> List[MetaRuleCandidate]:
        """Analyze patterns to generate meta-rule candidates."""
        meta_rule_candidates = []
        
        for pattern in patterns:
            # Find applicable templates
            applicable_templates = self._find_applicable_templates(pattern)
            
            for template in applicable_templates:
                # Generate meta-rule candidate
                candidate = self._generate_meta_rule_from_template(pattern, template)
                if candidate:
                    meta_rule_candidates.append(candidate)
        
        return meta_rule_candidates
    
    def _validate_meta_rule_candidates(self, candidates: List[MetaRuleCandidate]) -> List[MetaRuleCandidate]:
        """Validate meta-rule candidates through testing and simulation."""
        validated_candidates = []
        
        for candidate in candidates:
            # Run validation tests
            validation = self._run_meta_rule_validation(candidate)
            
            if validation.passed:
                validated_candidates.append(candidate)
                self.validated_rules.append(validation)
        
        return validated_candidates
    
    def _run_meta_rule_validation(self, candidate: MetaRuleCandidate) -> MetaRuleValidation:
        """Run comprehensive validation tests for a meta-rule candidate."""
        validation_tests = []
        
        # Test 1: Logical consistency
        logical_consistency = self._test_logical_consistency(candidate)
        validation_tests.append({
            'test_name': 'logical_consistency',
            'score': logical_consistency,
            'passed': logical_consistency >= 0.7
        })
        
        # Test 2: Empirical support
        empirical_support = self._test_empirical_support(candidate)
        validation_tests.append({
            'test_name': 'empirical_support',
            'score': empirical_support,
            'passed': empirical_support >= 0.6
        })
        
        # Test 3: Simulation validation
        simulation_validation = self._test_simulation_validation(candidate)
        validation_tests.append({
            'test_name': 'simulation_validation',
            'score': simulation_validation['score'],
            'passed': simulation_validation['passed']
        })
        
        # Calculate overall score
        overall_score = (logical_consistency + empirical_support + simulation_validation['score']) / 3
        
        # Determine if validation passed
        passed = all(test['passed'] for test in validation_tests) and overall_score >= 0.7
        
        return MetaRuleValidation(
            rule_candidate=candidate,
            validation_tests=validation_tests,
            simulation_validation=simulation_validation,
            logical_consistency=logical_consistency,
            empirical_support=empirical_support,
            overall_score=overall_score,
            passed=passed
        )
    
    def _test_logical_consistency(self, candidate: MetaRuleCandidate) -> float:
        """Test logical consistency of meta-rule candidate."""
        # Analyze rule pattern for logical consistency
        rule_pattern = candidate.rule_pattern
        
        # Check for logical operators
        logical_operators = ['=>', '<=>', '&', '|', '!']
        operator_count = sum(1 for op in logical_operators if op in rule_pattern)
        
        # Check for variable consistency
        variables = re.findall(r'[A-Z][a-zA-Z]*', rule_pattern)
        unique_variables = set(variables)
        
        # Calculate consistency score
        consistency_score = min(1.0, operator_count * 0.3 + len(unique_variables) * 0.1)
        
        return consistency_score
    
    def _test_empirical_support(self, candidate: MetaRuleCandidate) -> float:
        """Test empirical support for meta-rule candidate."""
        # Analyze source insights for empirical support
        source_insights = candidate.source_insights
        if not source_insights:
            return 0.0
        
        # Calculate average confidence from source insights
        avg_confidence = sum(insight.confidence for insight in source_insights) / len(source_insights)
        
        # Check for multiple supporting sources
        source_diversity = len(set(insight.source_conversation for insight in source_insights))
        diversity_bonus = min(0.2, source_diversity * 0.1)
        
        return min(1.0, avg_confidence + diversity_bonus)
    
    def _test_simulation_validation(self, candidate: MetaRuleCandidate) -> Dict[str, Any]:
        """Test meta-rule candidate through simulation validation."""
        # This would integrate with the simulation system
        # For now, return a placeholder validation
        return {
            'score': 0.8,
            'passed': True,
            'simulation_results': {},
            'validation_notes': 'Simulation validation placeholder'
        }
    
    def _analyze_response_pattern(self, response: str) -> Optional[str]:
        """Analyze response text for patterns."""
        if len(response) > 100:
            return f"complex_response_{len(response)}"
        elif 'Commander' in response:
            return "strategic_response_pattern"
        elif 'I shall' in response:
            return "archival_response_pattern"
        elif 'From a legal perspective' in response:
            return "legal_response_pattern"
        elif 'Through the veil' in response:
            return "oracle_response_pattern"
        return None
    
    def _analyze_reasoning_trace(self, trace: List[str]) -> Optional[str]:
        """Analyze reasoning trace for patterns."""
        if len(trace) > 3:
            return f"complex_reasoning_{len(trace)}"
        elif 'context_analysis' in trace:
            return "context_analysis_pattern"
        elif 'pattern_matching' in trace:
            return "pattern_matching_pattern"
        return None
    
    def _analyze_personality_pattern(self, personality: str, conversation_data: Dict[str, Any]) -> Optional[str]:
        """Analyze personality-specific patterns."""
        response = conversation_data.get('response', '')
        
        if personality == 'Strategos' and 'tactical' in response.lower():
            return "strategic_tactical_pattern"
        elif personality == 'Archivist' and 'catalog' in response.lower():
            return "archival_catalog_pattern"
        elif personality == 'Lawmaker' and 'legal' in response.lower():
            return "legal_reasoning_pattern"
        elif personality == 'Oracle' and 'veil' in response.lower():
            return "oracle_mystical_pattern"
        
        return None
    
    def _analyze_social_structure_pattern(self, social_structures: Dict[str, Any]) -> Optional[str]:
        """Analyze social structure patterns."""
        if 'hierarchy' in str(social_structures):
            return "hierarchical_social_pattern"
        elif 'egalitarian' in str(social_structures):
            return "egalitarian_social_pattern"
        return None
    
    def _analyze_economic_pattern(self, economic_patterns: Dict[str, Any]) -> Optional[str]:
        """Analyze economic patterns."""
        if 'market' in str(economic_patterns):
            return "market_economic_pattern"
        elif 'planned' in str(economic_patterns):
            return "planned_economic_pattern"
        return None
    
    def _analyze_conflict_resolution_pattern(self, conflict_resolution: Dict[str, Any]) -> Optional[str]:
        """Analyze conflict resolution patterns."""
        if 'mediation' in str(conflict_resolution):
            return "mediation_conflict_pattern"
        elif 'adjudication' in str(conflict_resolution):
            return "adjudication_conflict_pattern"
        return None
    
    def _find_applicable_templates(self, pattern: PatternMatch) -> List[MetaRuleTemplate]:
        """Find applicable meta-rule templates for a pattern."""
        applicable = []
        
        for template in self.meta_rule_templates:
            if self._pattern_matches_template(pattern, template):
                applicable.append(template)
        
        return applicable
    
    def _pattern_matches_template(self, pattern: PatternMatch, template: MetaRuleTemplate) -> bool:
        """Check if a pattern matches a template."""
        # Simple pattern matching logic
        if template.pattern_matcher in pattern.pattern_text:
            return True
        return False
    
    def _generate_meta_rule_from_template(self, pattern: PatternMatch, template: MetaRuleTemplate) -> Optional[MetaRuleCandidate]:
        """Generate meta-rule candidate from pattern and template."""
        # Generate rule pattern using template
        rule_pattern = template.rule_generator.format(pattern=pattern.pattern_text)
        
        # Create meta-rule candidate
        candidate = MetaRuleCandidate(
            rule_name=f"{template.template_name}_{len(self.discovered_rules)}",
            rule_pattern=rule_pattern,
            confidence=pattern.confidence,
            source_insights=[],
            validation_tests=[],
            simulation_validation={},
            personality_applicability=template.personality_applicability
        )
        
        return candidate
    
    def _initialize_meta_rule_templates(self):
        """Initialize meta-rule templates for pattern analysis."""
        self.meta_rule_templates = [
            MetaRuleTemplate(
                template_name="response_pattern_rule",
                pattern_matcher="response",
                rule_generator="response_rule: {pattern} => improved_response",
                validation_criteria=["logical_consistency", "empirical_support"],
                personality_applicability=["Strategos", "Archivist", "Lawmaker", "Oracle"]
            ),
            MetaRuleTemplate(
                template_name="reasoning_pattern_rule",
                pattern_matcher="reasoning",
                rule_generator="reasoning_rule: {pattern} => enhanced_reasoning",
                validation_criteria=["logical_consistency", "simulation_validation"],
                personality_applicability=["Strategos", "Archivist", "Lawmaker", "Oracle"]
            ),
            MetaRuleTemplate(
                template_name="personality_pattern_rule",
                pattern_matcher="personality",
                rule_generator="personality_rule: {pattern} => personality_enhancement",
                validation_criteria=["empirical_support", "personality_consistency"],
                personality_applicability=["Strategos", "Archivist", "Lawmaker", "Oracle"]
            ),
            MetaRuleTemplate(
                template_name="social_structure_rule",
                pattern_matcher="social",
                rule_generator="social_rule: {pattern} => social_improvement",
                validation_criteria=["simulation_validation", "civilizational_consistency"],
                personality_applicability=["Strategos", "Archivist", "Lawmaker", "Oracle"]
            )
        ]
    
    def apply_validated_meta_rules(self, validated_rules: List[MetaRuleCandidate]) -> Dict[str, Any]:
        """Apply validated meta-rules to the system."""
        applied_count = 0
        
        for rule in validated_rules:
            # Apply rule to appropriate systems
            self._apply_meta_rule_to_system(rule)
            self.applied_rules.append(rule)
            applied_count += 1
        
        self.rules_applied += applied_count
        
        return {
            'rules_applied': applied_count,
            'total_applied_rules': len(self.applied_rules),
            'success': True
        }
    
    def _apply_meta_rule_to_system(self, rule: MetaRuleCandidate):
        """Apply meta-rule to the appropriate system component."""
        # This would integrate with the actual system components
        # For now, just log the application
        self.logger.info(f"Applied meta-rule: {rule.rule_name} - {rule.rule_pattern}")
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of meta-rule discovery activities."""
        return {
            'discovery_cycles': self.discovery_cycles,
            'patterns_analyzed': self.patterns_analyzed,
            'rules_generated': self.rules_generated,
            'rules_validated': self.rules_validated,
            'rules_applied': self.rules_applied,
            'validation_success_rate': self.rules_validated / max(1, self.rules_generated),
            'application_success_rate': self.rules_applied / max(1, self.rules_validated),
            'total_discovered_rules': len(self.discovered_rules),
            'total_validated_rules': len(self.validated_rules),
            'total_applied_rules': len(self.applied_rules),
            'recent_discoveries': self.discovered_rules[-5:] if self.discovered_rules else [],
            'recent_validations': self.validated_rules[-3:] if self.validated_rules else []
        }
