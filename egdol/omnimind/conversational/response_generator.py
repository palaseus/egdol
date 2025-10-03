"""
Response Generator for Conversational Interface
Generates natural language responses based on personality and reasoning.
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from .personality_framework import Personality
from .reasoning_engine import ReasoningTrace, CivilizationalInsight


class ResponseStyle(Enum):
    """Response style types."""
    FORMAL = auto()
    CONVERSATIONAL = auto()
    TECHNICAL = auto()
    MYSTICAL = auto()
    AUTHORITATIVE = auto()
    SCHOLARLY = auto()


@dataclass
class ResponseTemplate:
    """Template for generating responses."""
    template: str
    style: ResponseStyle
    personality_type: str
    context_requirements: List[str]
    variables: List[str]


class ResponseGenerator:
    """Generates natural language responses based on personality and reasoning."""
    
    def __init__(self):
        self.response_templates = self._initialize_templates()
        self.personality_phrases = self._initialize_personality_phrases()
        self.reasoning_connectors = self._initialize_reasoning_connectors()
    
    def _initialize_templates(self) -> Dict[str, List[ResponseTemplate]]:
        """Initialize response templates for different scenarios."""
        templates = {
            'civilizational_analysis': [
                ResponseTemplate(
                    template="Based on my analysis of {domain} patterns, I observe {insight}. The underlying dynamics suggest {conclusion}.",
                    style=ResponseStyle.SCHOLARLY,
                    personality_type="Archivist",
                    context_requirements=["domain", "insight", "conclusion"],
                    variables=["domain", "insight", "conclusion"]
                ),
                ResponseTemplate(
                    template="From a strategic perspective, the {domain} situation reveals {insight}. This suggests {conclusion} for future planning.",
                    style=ResponseStyle.AUTHORITATIVE,
                    personality_type="Strategos",
                    context_requirements=["domain", "insight", "conclusion"],
                    variables=["domain", "insight", "conclusion"]
                )
            ],
            'strategic_analysis': [
                ResponseTemplate(
                    template="My tactical assessment indicates {assessment}. The key factors are {factors}, leading to {recommendation}.",
                    style=ResponseStyle.AUTHORITATIVE,
                    personality_type="Strategos",
                    context_requirements=["assessment", "factors", "recommendation"],
                    variables=["assessment", "factors", "recommendation"]
                )
            ],
            'meta_rule_discovery': [
                ResponseTemplate(
                    template="Through systematic analysis, I have identified the meta-rule: {rule}. This principle governs {scope} and suggests {implication}.",
                    style=ResponseStyle.FORMAL,
                    personality_type="Lawmaker",
                    context_requirements=["rule", "scope", "implication"],
                    variables=["rule", "scope", "implication"]
                )
            ],
            'universe_comparison': [
                ResponseTemplate(
                    template="Across the cosmic tapestry, I perceive {pattern}. This universal truth reveals {insight} about {domain}.",
                    style=ResponseStyle.MYSTICAL,
                    personality_type="Oracle",
                    context_requirements=["pattern", "insight", "domain"],
                    variables=["pattern", "insight", "domain"]
                )
            ],
            'general_response': [
                ResponseTemplate(
                    template="{greeting} {response_content}",
                    style=ResponseStyle.CONVERSATIONAL,
                    personality_type="General",
                    context_requirements=["greeting", "response_content"],
                    variables=["greeting", "response_content"]
                )
            ]
        }
        return templates
    
    def _initialize_personality_phrases(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize personality-specific phrases."""
        return {
            'Strategos': {
                'greetings': [
                    "Commander,", "Strategically speaking,", "From a tactical standpoint,",
                    "In military terms,", "From the battlefield of knowledge,"
                ],
                'transitions': [
                    "Furthermore,", "Additionally,", "Moreover,", "In addition,",
                    "From a strategic perspective,", "Tactically speaking,"
                ],
                'conclusions': [
                    "This is the optimal course of action.",
                    "The strategic advantage lies in this approach.",
                    "Victory requires this understanding.",
                    "This is the path to success."
                ]
            },
            'Archivist': {
                'greetings': [
                    "In the annals of history,", "From the archives of knowledge,",
                    "Through the lens of time,", "In the chronicles of civilization,",
                    "From the depths of historical wisdom,"
                ],
                'transitions': [
                    "Furthermore,", "Additionally,", "Moreover,", "In addition,",
                    "From a historical perspective,", "Through the ages,"
                ],
                'conclusions': [
                    "This wisdom has stood the test of time.",
                    "History teaches us this truth.",
                    "This is the accumulated wisdom of ages.",
                    "The past illuminates this path."
                ]
            },
            'Lawmaker': {
                'greetings': [
                    "According to the principles of governance,", "From a legal perspective,",
                    "In accordance with the rule of law,", "From the foundation of order,",
                    "Through the lens of justice,"
                ],
                'transitions': [
                    "Furthermore,", "Additionally,", "Moreover,", "In addition,",
                    "From a legal perspective,", "According to the principles,"
                ],
                'conclusions': [
                    "This is the fundamental principle.",
                    "Justice demands this understanding.",
                    "This is the foundation of order.",
                    "The law reveals this truth."
                ]
            },
            'Oracle': {
                'greetings': [
                    "In the cosmic dance of reality,", "From the depths of universal wisdom,",
                    "Through the veil of existence,", "In the tapestry of the multiverse,",
                    "From the perspective of the infinite,"
                ],
                'transitions': [
                    "Furthermore,", "Additionally,", "Moreover,", "In addition,",
                    "From a cosmic perspective,", "In the grand scheme,"
                ],
                'conclusions': [
                    "This is the universal truth.",
                    "The cosmos reveals this wisdom.",
                    "This is the pattern of reality.",
                    "The universe speaks this truth."
                ]
            }
        }
    
    def _initialize_reasoning_connectors(self) -> List[str]:
        """Initialize connectors for reasoning traces."""
        return [
            "Based on my analysis,",
            "Through careful consideration,",
            "After examining the evidence,",
            "From my reasoning process,",
            "Through systematic analysis,",
            "Based on the data,",
            "From my understanding,"
        ]
    
    def generate_response(self, 
                         personality: Personality,
                         reasoning_trace: Optional[ReasoningTrace] = None,
                         civilizational_insights: List[CivilizationalInsight] = None,
                         context: Dict[str, Any] = None) -> str:
        """Generate a response based on personality and reasoning."""
        if not context:
            context = {}
        
        # Determine response type
        response_type = self._determine_response_type(reasoning_trace, context)
        
        # Get appropriate template
        template = self._select_template(response_type, personality.name)
        
        # Generate content based on reasoning
        content = self._generate_content_from_reasoning(
            personality, reasoning_trace, civilizational_insights, context
        )
        
        # Apply personality styling
        styled_response = self._apply_personality_styling(
            personality, content, template
        )
        
        return styled_response
    
    def _determine_response_type(self, reasoning_trace: Optional[ReasoningTrace], context: Dict[str, Any]) -> str:
        """Determine the type of response to generate."""
        if not reasoning_trace:
            return 'general_response'
        
        reasoning_type = reasoning_trace.reasoning_type
        
        if reasoning_type == 'civilizational_query':
            return 'civilizational_analysis'
        elif reasoning_type == 'strategic_analysis':
            return 'strategic_analysis'
        elif reasoning_type == 'meta_rule_discovery':
            return 'meta_rule_discovery'
        elif reasoning_type == 'universe_comparison':
            return 'universe_comparison'
        else:
            return 'general_response'
    
    def _select_template(self, response_type: str, personality_name: str) -> ResponseTemplate:
        """Select appropriate template for response type and personality."""
        templates = self.response_templates.get(response_type, self.response_templates['general_response'])
        
        # Filter by personality if possible
        personality_templates = [t for t in templates if t.personality_type == personality_name]
        if personality_templates:
            return random.choice(personality_templates)
        
        # Fall back to any template of the right type
        return random.choice(templates)
    
    def _generate_content_from_reasoning(self, 
                                       personality: Personality,
                                       reasoning_trace: Optional[ReasoningTrace],
                                       civilizational_insights: List[CivilizationalInsight],
                                       context: Dict[str, Any]) -> Dict[str, str]:
        """Generate content based on reasoning trace and insights."""
        content = {}
        
        if not reasoning_trace:
            content['response_content'] = "I'm here to help with your inquiry."
            return content
        
        # Extract insights from reasoning trace
        if reasoning_trace.output_data:
            if 'patterns' in reasoning_trace.output_data:
                patterns = reasoning_trace.output_data['patterns']
                content['insight'] = f"discovered {len(patterns)} key patterns"
            elif 'simulation_result' in reasoning_trace.output_data:
                simulation = reasoning_trace.output_data['simulation_result']
                content['assessment'] = "the strategic scenario shows multiple possible outcomes"
                content['factors'] = "resource allocation and tactical positioning"
                content['recommendation'] = "careful planning and adaptive strategy"
            elif 'meta_rules' in reasoning_trace.output_data:
                meta_rules = reasoning_trace.output_data['meta_rules']
                content['rule'] = f"a fundamental principle governing {len(meta_rules)} domains"
                content['scope'] = "systematic behavior and governance"
                content['implication'] = "enhanced understanding of underlying dynamics"
            elif 'cross_universe_analysis' in reasoning_trace.output_data:
                analysis = reasoning_trace.output_data['cross_universe_analysis']
                content['pattern'] = "universal patterns across multiple realities"
                content['insight'] = "fundamental truths about existence"
                content['domain'] = "cosmic and universal phenomena"
        
        # Add civilizational insights
        if civilizational_insights:
            latest_insight = civilizational_insights[-1]
            content['insight'] = latest_insight.content
            content['conclusion'] = "this reveals deeper patterns in civilizational dynamics"
        
        # Add personality-specific content
        if personality.name == 'Strategos':
            content['assessment'] = content.get('assessment', 'the strategic landscape is complex')
            content['factors'] = content.get('factors', 'tactical positioning and resource allocation')
            content['recommendation'] = content.get('recommendation', 'adaptive strategy and careful planning')
        elif personality.name == 'Archivist':
            content['insight'] = content.get('insight', 'historical patterns reveal important trends')
            content['conclusion'] = content.get('conclusion', 'this wisdom has stood the test of time')
        elif personality.name == 'Lawmaker':
            content['rule'] = content.get('rule', 'a fundamental principle of governance')
            content['scope'] = content.get('scope', 'systematic behavior and order')
            content['implication'] = content.get('implication', 'enhanced understanding of underlying dynamics')
        elif personality.name == 'Oracle':
            content['pattern'] = content.get('pattern', 'universal patterns in the cosmic tapestry')
            content['insight'] = content.get('insight', 'fundamental truths about reality')
            content['domain'] = content.get('domain', 'cosmic and universal phenomena')
        
        return content
    
    def _apply_personality_styling(self, 
                                 personality: Personality,
                                 content: Dict[str, str],
                                 template: ResponseTemplate) -> str:
        """Apply personality-specific styling to response."""
        # Get personality phrases
        phrases = self.personality_phrases.get(personality.name, self.personality_phrases['Strategos'])
        
        # Add greeting
        greeting = random.choice(phrases['greetings'])
        
        # Generate response using template
        response_parts = []
        
        # Add greeting
        response_parts.append(greeting)
        
        # Add main content
        if template.template:
            try:
                formatted_content = template.template.format(**content)
                response_parts.append(formatted_content)
            except KeyError as e:
                # Fallback if template variables missing
                response_parts.append(content.get('response_content', 'I have analyzed your request.'))
        else:
            response_parts.append(content.get('response_content', 'I have analyzed your request.'))
        
        # Add conclusion
        if personality.name in phrases['conclusions']:
            conclusion = random.choice(phrases['conclusions'])
            response_parts.append(conclusion)
        
        # Join parts
        response = ' '.join(response_parts)
        
        # Apply personality-specific formatting
        if personality.response_style.get('tone') == 'authoritative':
            response = response.replace('I', 'I, as your strategic advisor,')
        elif personality.response_style.get('tone') == 'scholarly':
            response = response.replace('I', 'I, from my scholarly perspective,')
        elif personality.response_style.get('tone') == 'mystical':
            response = response.replace('I', 'I, in my cosmic awareness,')
        
        return response
    
    def generate_reasoning_explanation(self, reasoning_trace: ReasoningTrace) -> str:
        """Generate explanation of reasoning process."""
        explanation_parts = []
        
        # Add reasoning connector
        connector = random.choice(self.reasoning_connectors)
        explanation_parts.append(connector)
        
        # Add processing steps
        if reasoning_trace.processing_steps:
            explanation_parts.append("I performed the following analysis:")
            for i, step in enumerate(reasoning_trace.processing_steps, 1):
                explanation_parts.append(f"{i}. {step}")
        
        # Add confidence level
        if reasoning_trace.confidence > 0.8:
            explanation_parts.append("I have high confidence in this analysis.")
        elif reasoning_trace.confidence > 0.6:
            explanation_parts.append("I have moderate confidence in this analysis.")
        else:
            explanation_parts.append("This analysis is preliminary and may require further investigation.")
        
        # Add meta insights
        if reasoning_trace.meta_insights:
            explanation_parts.append("Key insights discovered:")
            for insight in reasoning_trace.meta_insights:
                explanation_parts.append(f"- {insight}")
        
        return ' '.join(explanation_parts)
    
    def generate_personality_switch_response(self, old_personality: str, new_personality: str) -> str:
        """Generate response for personality switch."""
        switch_phrases = {
            'Strategos': "Commander, I am now your strategic advisor.",
            'Archivist': "From the archives of knowledge, I now speak as your scholarly guide.",
            'Lawmaker': "In accordance with the principles of governance, I now serve as your legal advisor.",
            'Oracle': "In the cosmic dance of reality, I now channel universal wisdom."
        }
        
        return switch_phrases.get(new_personality, f"I have switched from {old_personality} to {new_personality}.")
    
    def generate_error_response(self, error: str, personality: Personality) -> str:
        """Generate error response with personality styling."""
        error_phrases = {
            'Strategos': f"Commander, I encountered a tactical error: {error}. I recommend alternative approaches.",
            'Archivist': f"From the archives, I must report an error in my analysis: {error}. Let me consult other sources.",
            'Lawmaker': f"According to the principles of governance, I must report an error: {error}. Let me review the legal framework.",
            'Oracle': f"In the cosmic tapestry, I perceive an error in my analysis: {error}. Let me consult the universal patterns."
        }
        
        return error_phrases.get(personality.name, f"I encountered an error: {error}. Let me try a different approach.")
