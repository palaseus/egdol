"""
Intent Parser for Conversational Interface
Analyzes user input to determine intent and context.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class IntentType(Enum):
    """Types of user intents."""
    QUESTION = auto()
    COMMAND = auto()
    STATEMENT = auto()
    REQUEST = auto()
    CLARIFICATION = auto()
    PERSONALITY_SWITCH = auto()
    REFLECTION = auto()
    CIVILIZATIONAL_QUERY = auto()
    STRATEGIC_ANALYSIS = auto()
    META_RULE_DISCOVERY = auto()
    UNIVERSE_COMPARISON = auto()


@dataclass
class Intent:
    """Represents a parsed user intent."""
    intent_type: IntentType
    confidence: float
    entities: List[str]
    context_clues: List[str]
    personality_hint: Optional[str] = None
    complexity_level: float = 0.5
    domain: Optional[str] = None
    urgency: float = 0.5
    emotional_tone: str = "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary."""
        return {
            'intent_type': self.intent_type.name,
            'confidence': self.confidence,
            'entities': self.entities,
            'context_clues': self.context_clues,
            'personality_hint': self.personality_hint,
            'complexity_level': self.complexity_level,
            'domain': self.domain,
            'urgency': self.urgency,
            'emotional_tone': self.emotional_tone
        }


class IntentParser:
    """Parses user input to determine intent and context."""
    
    def __init__(self):
        # Intent patterns
        self.question_patterns = [
            r'\b(what|who|when|where|why|how|which|can|could|would|should|is|are|do|does|did)\b',
            r'\?',
            r'\b(explain|describe|tell me|show me)\b'
        ]
        
        self.command_patterns = [
            r'\b(calculate|compute|solve|generate|create|build|make)\b',
            r'\b(analyze|examine|investigate|explore)\b',
            r'\b(switch|change|become|act as)\b'
        ]
        
        self.statement_patterns = [
            r'\b(I think|I believe|I feel|I know|I understand)\b',
            r'\b(this is|that is|it is)\b',
            r'^[A-Z][^.!?]*[.!]$'  # Capitalized statements
        ]
        
        self.personality_switch_patterns = [
            r'\b(switch to|become|act as|channel)\b',
            r'\b(strategos|archivist|lawmaker|oracle)\b'
        ]
        
        self.civilizational_patterns = [
            r'\b(civilization|culture|society|population|governance)\b',
            r'\b(evolution|development|growth|decline)\b',
            r'\b(patterns|trends|emergence|complexity)\b'
        ]
        
        self.strategic_patterns = [
            r'\b(strategy|tactics|war|conflict|battle)\b',
            r'\b(planning|coordination|alliance|diplomacy)\b',
            r'\b(resources|territory|power|influence)\b'
        ]
        
        self.meta_rule_patterns = [
            r'\b(meta|rule|principle|law|governance)\b',
            r'\b(universal|fundamental|underlying)\b',
            r'\b(discover|find|identify|detect)\b'
        ]
        
        self.universe_patterns = [
            r'\b(universe|reality|dimension|world)\b',
            r'\b(compare|contrast|different|alternative)\b',
            r'\b(parallel|multiverse|cosmic)\b'
        ]
        
        # Emotional tone patterns
        self.emotional_patterns = {
            'excited': [r'\b(excited|thrilled|amazing|wonderful|fantastic)\b'],
            'concerned': [r'\b(worried|concerned|anxious|nervous|troubled)\b'],
            'curious': [r'\b(curious|interested|fascinated|intrigued)\b'],
            'frustrated': [r'\b(frustrated|annoyed|irritated|angry)\b'],
            'confident': [r'\b(confident|sure|certain|positive)\b'],
            'uncertain': [r'\b(uncertain|unsure|confused|puzzled)\b']
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'high': [r'\b(complex|sophisticated|advanced|intricate|nuanced)\b'],
            'medium': [r'\b(moderate|somewhat|partially|relatively)\b'],
            'low': [r'\b(simple|basic|straightforward|easy)\b']
        }
    
    def parse(self, user_input: str) -> Intent:
        """Parse user input to determine intent."""
        input_lower = user_input.lower()
        
        # Determine intent type
        intent_type = self._determine_intent_type(input_lower)
        
        # Extract entities
        entities = self._extract_entities(user_input)
        
        # Extract context clues
        context_clues = self._extract_context_clues(input_lower)
        
        # Determine personality hint
        personality_hint = self._determine_personality_hint(input_lower)
        
        # Calculate complexity level
        complexity_level = self._calculate_complexity(input_lower)
        
        # Determine domain
        domain = self._determine_domain(input_lower)
        
        # Calculate urgency
        urgency = self._calculate_urgency(input_lower)
        
        # Determine emotional tone
        emotional_tone = self._determine_emotional_tone(input_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(intent_type, input_lower)
        
        return Intent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            context_clues=context_clues,
            personality_hint=personality_hint,
            complexity_level=complexity_level,
            domain=domain,
            urgency=urgency,
            emotional_tone=emotional_tone
        )
    
    def _determine_intent_type(self, input_lower: str) -> IntentType:
        """Determine the primary intent type."""
        # Check for personality switch
        if any(re.search(pattern, input_lower) for pattern in self.personality_switch_patterns):
            return IntentType.PERSONALITY_SWITCH
        
        # Check for civilizational query
        if any(re.search(pattern, input_lower) for pattern in self.civilizational_patterns):
            return IntentType.CIVILIZATIONAL_QUERY
        
        # Check for strategic analysis
        if any(re.search(pattern, input_lower) for pattern in self.strategic_patterns):
            return IntentType.STRATEGIC_ANALYSIS
        
        # Check for meta rule discovery
        if any(re.search(pattern, input_lower) for pattern in self.meta_rule_patterns):
            return IntentType.META_RULE_DISCOVERY
        
        # Check for universe comparison
        if any(re.search(pattern, input_lower) for pattern in self.universe_patterns):
            return IntentType.UNIVERSE_COMPARISON
        
        # Check for reflection
        if any(word in input_lower for word in ['reflect', 'think', 'consider', 'ponder']):
            return IntentType.REFLECTION
        
        # Check for questions
        if any(re.search(pattern, input_lower) for pattern in self.question_patterns):
            return IntentType.QUESTION
        
        # Check for commands
        if any(re.search(pattern, input_lower) for pattern in self.command_patterns):
            return IntentType.COMMAND
        
        # Check for statements
        if any(re.search(pattern, input_lower) for pattern in self.statement_patterns):
            return IntentType.STATEMENT
        
        # Default to request
        return IntentType.REQUEST
    
    def _extract_entities(self, user_input: str) -> List[str]:
        """Extract named entities from input."""
        entities = []
        
        # Simple entity extraction (can be enhanced with NER)
        words = user_input.split()
        
        # Capitalized words (potential proper nouns)
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entities.append(word)
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', user_input)
        entities.extend(numbers)
        
        # Quoted strings
        quoted = re.findall(r'"([^"]*)"', user_input)
        entities.extend(quoted)
        
        return entities
    
    def _extract_context_clues(self, input_lower: str) -> List[str]:
        """Extract context clues from input."""
        clues = []
        
        # Domain indicators
        if any(word in input_lower for word in ['math', 'calculate', 'number']):
            clues.append('mathematical')
        if any(word in input_lower for word in ['logic', 'reasoning', 'argument']):
            clues.append('logical')
        if any(word in input_lower for word in ['code', 'programming', 'software']):
            clues.append('technical')
        if any(word in input_lower for word in ['philosophy', 'ethics', 'morality']):
            clues.append('philosophical')
        if any(word in input_lower for word in ['history', 'past', 'ancient']):
            clues.append('historical')
        
        return clues
    
    def _determine_personality_hint(self, input_lower: str) -> Optional[str]:
        """Determine if input hints at a specific personality."""
        if any(word in input_lower for word in ['strategy', 'war', 'tactics', 'military']):
            return 'strategos'
        if any(word in input_lower for word in ['history', 'archive', 'knowledge', 'past']):
            return 'archivist'
        if any(word in input_lower for word in ['law', 'rule', 'governance', 'meta']):
            return 'lawmaker'
        if any(word in input_lower for word in ['universe', 'cosmic', 'reality', 'compare']):
            return 'oracle'
        
        return None
    
    def _calculate_complexity(self, input_lower: str) -> float:
        """Calculate complexity level of input."""
        complexity = 0.5  # Default medium complexity
        
        # High complexity indicators
        if any(re.search(pattern, input_lower) for pattern in self.complexity_indicators['high']):
            complexity = 0.8
        elif any(re.search(pattern, input_lower) for pattern in self.complexity_indicators['low']):
            complexity = 0.2
        
        # Adjust based on sentence length and structure
        sentence_count = len(re.findall(r'[.!?]+', input_lower))
        word_count = len(input_lower.split())
        
        if sentence_count > 2 or word_count > 20:
            complexity = min(1.0, complexity + 0.2)
        elif sentence_count == 1 and word_count < 5:
            complexity = max(0.1, complexity - 0.2)
        
        return complexity
    
    def _determine_domain(self, input_lower: str) -> Optional[str]:
        """Determine the domain of the input."""
        if any(word in input_lower for word in ['math', 'calculate', 'number', 'equation']):
            return 'mathematical'
        if any(word in input_lower for word in ['logic', 'reasoning', 'argument', 'proof']):
            return 'logical'
        if any(word in input_lower for word in ['code', 'programming', 'software', 'algorithm']):
            return 'technical'
        if any(word in input_lower for word in ['philosophy', 'ethics', 'morality', 'meaning']):
            return 'philosophical'
        if any(word in input_lower for word in ['history', 'past', 'ancient', 'tradition']):
            return 'historical'
        if any(word in input_lower for word in ['science', 'research', 'experiment', 'hypothesis']):
            return 'scientific'
        
        return None
    
    def _calculate_urgency(self, input_lower: str) -> float:
        """Calculate urgency level of input."""
        urgency = 0.5  # Default medium urgency
        
        # High urgency indicators
        if any(word in input_lower for word in ['urgent', 'immediate', 'critical', 'emergency']):
            urgency = 0.9
        elif any(word in input_lower for word in ['asap', 'quickly', 'fast', 'now']):
            urgency = 0.8
        
        # Low urgency indicators
        if any(word in input_lower for word in ['whenever', 'sometime', 'eventually', 'later']):
            urgency = 0.2
        
        return urgency
    
    def _determine_emotional_tone(self, input_lower: str) -> str:
        """Determine emotional tone of input."""
        for tone, patterns in self.emotional_patterns.items():
            if any(re.search(pattern, input_lower) for pattern in patterns):
                return tone
        
        return 'neutral'
    
    def _calculate_confidence(self, intent_type: IntentType, input_lower: str) -> float:
        """Calculate confidence in intent classification."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on pattern matches
        if intent_type == IntentType.QUESTION:
            if any(re.search(pattern, input_lower) for pattern in self.question_patterns):
                confidence = 0.9
        elif intent_type == IntentType.COMMAND:
            if any(re.search(pattern, input_lower) for pattern in self.command_patterns):
                confidence = 0.9
        elif intent_type == IntentType.PERSONALITY_SWITCH:
            if any(re.search(pattern, input_lower) for pattern in self.personality_switch_patterns):
                confidence = 0.9
        
        return confidence
