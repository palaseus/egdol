"""
Pattern Analyzer for OmniMind Meta-Learning
Analyzes conversation patterns to identify potential new skills.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict


class PatternAnalyzer:
    """Analyzes conversation patterns to identify skill opportunities."""
    
    def __init__(self):
        self.pattern_history: List[Dict[str, Any]] = []
        self.skill_keywords = {
            'convert': 'conversion',
            'analyze': 'analysis',
            'process': 'data_processing',
            'calculate': 'calculation',
            'generate': 'generation',
            'format': 'formatting',
            'parse': 'parsing',
            'validate': 'validation',
            'transform': 'transformation',
            'extract': 'extraction',
            'filter': 'filtering',
            'sort': 'sorting',
            'search': 'searching',
            'find': 'finding',
            'match': 'matching'
        }
        
    def analyze_conversation(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze conversation history for patterns."""
        patterns = []
        
        # Extract user inputs
        user_inputs = [msg['content'] for msg in conversation_history if msg.get('type') == 'user']
        
        if not user_inputs:
            return patterns
            
        # Analyze different types of patterns
        patterns.extend(self._analyze_repeated_phrases(user_inputs))
        patterns.extend(self._analyze_skill_keywords(user_inputs))
        patterns.extend(self._analyze_question_patterns(user_inputs))
        patterns.extend(self._analyze_command_patterns(user_inputs))
        patterns.extend(self._analyze_sequence_patterns(user_inputs))
        
        # Store patterns
        for pattern in patterns:
            pattern['analyzed_at'] = time.time()
            self.pattern_history.append(pattern)
            
        return patterns
        
    def _analyze_repeated_phrases(self, user_inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze repeated phrases in user inputs."""
        patterns = []
        
        # Count word sequences
        phrase_counts = Counter()
        for input_text in user_inputs:
            words = input_text.lower().split()
            # Extract 2-4 word phrases
            for length in range(2, min(5, len(words) + 1)):
                for i in range(len(words) - length + 1):
                    phrase = ' '.join(words[i:i+length])
                    phrase_counts[phrase] += 1
                    
        # Find frequently repeated phrases
        for phrase, count in phrase_counts.items():
            if count >= 3:  # Threshold for pattern recognition
                patterns.append({
                    'type': 'repeated_phrase',
                    'pattern': phrase,
                    'frequency': count,
                    'confidence': min(count / 10.0, 1.0),
                    'suggestion': f"Skill for handling: {phrase}"
                })
                
        return patterns
        
    def _analyze_skill_keywords(self, user_inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze skill-related keywords."""
        patterns = []
        
        keyword_counts = Counter()
        for input_text in user_inputs:
            words = input_text.lower().split()
            for word in words:
                if word in self.skill_keywords:
                    keyword_counts[word] += 1
                    
        for keyword, count in keyword_counts.items():
            if count >= 2:  # Threshold for keyword recognition
                patterns.append({
                    'type': 'skill_keyword',
                    'pattern': keyword,
                    'frequency': count,
                    'confidence': min(count / 5.0, 1.0),
                    'skill_type': self.skill_keywords[keyword],
                    'suggestion': f"Create {self.skill_keywords[keyword]} skill"
                })
                
        return patterns
        
    def _analyze_question_patterns(self, user_inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze question patterns."""
        patterns = []
        
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        question_counts = Counter()
        
        for input_text in user_inputs:
            if '?' in input_text:
                words = input_text.lower().split()
                for word in words:
                    if word in question_words:
                        question_counts[word] += 1
                        
        for question_word, count in question_counts.items():
            if count >= 2:
                patterns.append({
                    'type': 'question_pattern',
                    'pattern': question_word,
                    'frequency': count,
                    'confidence': min(count / 5.0, 1.0),
                    'suggestion': f"Skill for answering {question_word} questions"
                })
                
        return patterns
        
    def _analyze_command_patterns(self, user_inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze command patterns."""
        patterns = []
        
        command_words = ['create', 'make', 'build', 'generate', 'write', 'show', 'display']
        command_counts = Counter()
        
        for input_text in user_inputs:
            words = input_text.lower().split()
            for word in words:
                if word in command_words:
                    command_counts[word] += 1
                    
        for command_word, count in command_counts.items():
            if count >= 2:
                patterns.append({
                    'type': 'command_pattern',
                    'pattern': command_word,
                    'frequency': count,
                    'confidence': min(count / 5.0, 1.0),
                    'suggestion': f"Skill for {command_word} operations"
                })
                
        return patterns
        
    def _analyze_sequence_patterns(self, user_inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze sequential patterns."""
        patterns = []
        
        if len(user_inputs) < 3:
            return patterns
            
        # Look for sequential operations
        sequences = []
        for i in range(len(user_inputs) - 2):
            sequence = user_inputs[i:i+3]
            sequences.append(' -> '.join(sequence))
            
        sequence_counts = Counter(sequences)
        
        for sequence, count in sequence_counts.items():
            if count >= 2:
                patterns.append({
                    'type': 'sequence_pattern',
                    'pattern': sequence,
                    'frequency': count,
                    'confidence': min(count / 5.0, 1.0),
                    'suggestion': f"Skill for sequence: {sequence}"
                })
                
        return patterns
        
    def get_pattern_suggestions(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get skill suggestions based on patterns."""
        suggestions = []
        
        # Group patterns by type
        by_type = defaultdict(list)
        for pattern in patterns:
            by_type[pattern['type']].append(pattern)
            
        # Generate suggestions for each type
        for pattern_type, type_patterns in by_type.items():
            if pattern_type == 'repeated_phrase':
                suggestions.extend(self._suggest_phrase_skills(type_patterns))
            elif pattern_type == 'skill_keyword':
                suggestions.extend(self._suggest_keyword_skills(type_patterns))
            elif pattern_type == 'question_pattern':
                suggestions.extend(self._suggest_question_skills(type_patterns))
            elif pattern_type == 'command_pattern':
                suggestions.extend(self._suggest_command_skills(type_patterns))
            elif pattern_type == 'sequence_pattern':
                suggestions.extend(self._suggest_sequence_skills(type_patterns))
                
        return suggestions
        
    def _suggest_phrase_skills(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest skills for repeated phrases."""
        suggestions = []
        
        for pattern in patterns:
            phrase = pattern['pattern']
            frequency = pattern['frequency']
            
            suggestions.append({
                'type': 'phrase_skill',
                'name': f"{phrase.replace(' ', '_')}_handler",
                'description': f"Handles requests related to: {phrase}",
                'confidence': pattern['confidence'],
                'priority': frequency,
                'implementation': f"Process '{phrase}' requests"
            })
            
        return suggestions
        
    def _suggest_keyword_skills(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest skills for keyword patterns."""
        suggestions = []
        
        for pattern in patterns:
            keyword = pattern['pattern']
            skill_type = pattern['skill_type']
            frequency = pattern['frequency']
            
            suggestions.append({
                'type': 'keyword_skill',
                'name': f"{skill_type}_skill",
                'description': f"Handles {skill_type} operations",
                'confidence': pattern['confidence'],
                'priority': frequency,
                'implementation': f"Process {keyword} operations"
            })
            
        return suggestions
        
    def _suggest_question_skills(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest skills for question patterns."""
        suggestions = []
        
        for pattern in patterns:
            question_word = pattern['pattern']
            frequency = pattern['frequency']
            
            suggestions.append({
                'type': 'question_skill',
                'name': f"{question_word}_answerer",
                'description': f"Answers {question_word} questions",
                'confidence': pattern['confidence'],
                'priority': frequency,
                'implementation': f"Handle {question_word} questions"
            })
            
        return suggestions
        
    def _suggest_command_skills(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest skills for command patterns."""
        suggestions = []
        
        for pattern in patterns:
            command_word = pattern['pattern']
            frequency = pattern['frequency']
            
            suggestions.append({
                'type': 'command_skill',
                'name': f"{command_word}_executor",
                'description': f"Executes {command_word} commands",
                'confidence': pattern['confidence'],
                'priority': frequency,
                'implementation': f"Execute {command_word} operations"
            })
            
        return suggestions
        
    def _suggest_sequence_skills(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest skills for sequence patterns."""
        suggestions = []
        
        for pattern in patterns:
            sequence = pattern['pattern']
            frequency = pattern['frequency']
            
            suggestions.append({
                'type': 'sequence_skill',
                'name': f"sequence_{hash(sequence) % 10000}",
                'description': f"Handles sequence: {sequence}",
                'confidence': pattern['confidence'],
                'priority': frequency,
                'implementation': f"Process sequence: {sequence}"
            })
            
        return suggestions
        
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of pattern analysis."""
        if not self.pattern_history:
            return {'total_patterns': 0, 'pattern_types': {}}
            
        # Count patterns by type
        type_counts = Counter()
        for pattern in self.pattern_history:
            type_counts[pattern['type']] += 1
            
        # Calculate average confidence
        avg_confidence = sum(p['confidence'] for p in self.pattern_history) / len(self.pattern_history)
        
        return {
            'total_patterns': len(self.pattern_history),
            'pattern_types': dict(type_counts),
            'average_confidence': avg_confidence,
            'recent_patterns': len([p for p in self.pattern_history 
                                  if p.get('analyzed_at', 0) > time.time() - 3600])
        }
        
    def clear_history(self):
        """Clear pattern analysis history."""
        self.pattern_history.clear()
