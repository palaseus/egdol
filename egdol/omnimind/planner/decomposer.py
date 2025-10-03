"""
Goal Decomposer for OmniMind
Decomposes complex goals into discrete sub-tasks.
"""

import re
import time
from typing import Dict, Any, List, Optional
from .planner import Task, TaskPriority


class GoalDecomposer:
    """Decomposes complex goals into discrete sub-tasks."""
    
    def __init__(self):
        self.decomposition_patterns = {
            'analyze': self._decompose_analysis_goal,
            'summarize': self._decompose_summary_goal,
            'compare': self._decompose_comparison_goal,
            'create': self._decompose_creation_goal,
            'improve': self._decompose_improvement_goal,
            'security': self._decompose_security_goal,
            'performance': self._decompose_performance_goal,
            'documentation': self._decompose_documentation_goal
        }
        
    def decompose_goal(self, goal_description: str, context: Dict[str, Any] = None) -> List[Task]:
        """Decompose a goal into tasks."""
        context = context or {}
        tasks = []
        
        # Extract key phrases from goal description
        key_phrases = self._extract_key_phrases(goal_description)
        
        # Determine decomposition strategy
        strategy = self._determine_strategy(goal_description, key_phrases)
        
        # Decompose based on strategy
        if strategy == 'analysis':
            tasks = self._decompose_analysis_goal(goal_description, context)
        elif strategy == 'summary':
            tasks = self._decompose_summary_goal(goal_description, context)
        elif strategy == 'comparison':
            tasks = self._decompose_comparison_goal(goal_description, context)
        elif strategy == 'creation':
            tasks = self._decompose_creation_goal(goal_description, context)
        elif strategy == 'improvement':
            tasks = self._decompose_improvement_goal(goal_description, context)
        elif strategy == 'security':
            tasks = self._decompose_security_goal(goal_description, context)
        elif strategy == 'performance':
            tasks = self._decompose_performance_goal(goal_description, context)
        elif strategy == 'documentation':
            tasks = self._decompose_documentation_goal(goal_description, context)
        else:
            # Default decomposition
            tasks = self._decompose_default_goal(goal_description, context)
            
        return tasks
        
    def _extract_key_phrases(self, goal_description: str) -> List[str]:
        """Extract key phrases from goal description."""
        # Convert to lowercase for analysis
        text = goal_description.lower()
        
        # Extract action words
        action_words = []
        for word in ['analyze', 'summarize', 'compare', 'create', 'improve', 'review', 'check', 'test', 'optimize']:
            if word in text:
                action_words.append(word)
                
        # Extract domain words
        domain_words = []
        for word in ['security', 'performance', 'code', 'file', 'documentation', 'bug', 'error', 'issue']:
            if word in text:
                domain_words.append(word)
                
        # Extract object words
        object_words = []
        for word in ['file', 'code', 'system', 'application', 'database', 'network', 'user', 'data']:
            if word in text:
                object_words.append(word)
                
        return action_words + domain_words + object_words
        
    def _determine_strategy(self, goal_description: str, key_phrases: List[str]) -> str:
        """Determine decomposition strategy based on goal description."""
        text = goal_description.lower()
        
        # Check for specific patterns
        if 'analyze' in text and 'security' in text:
            return 'security'
        elif 'analyze' in text and 'performance' in text:
            return 'performance'
        elif 'analyze' in text:
            return 'analysis'
        elif 'summarize' in text:
            return 'summary'
        elif 'compare' in text:
            return 'comparison'
        elif 'create' in text or 'generate' in text:
            return 'creation'
        elif 'improve' in text or 'optimize' in text:
            return 'improvement'
        elif 'document' in text or 'documentation' in text:
            return 'documentation'
        else:
            return 'default'
            
    def _decompose_analysis_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose analysis goals."""
        tasks = []
        
        # Extract target from context or description
        target = context.get('target', 'the specified item')
        
        tasks.append(Task(
            id=f"analyze_{int(time.time())}",
            name="Analyze Target",
            description=f"Analyze {target} for key characteristics and patterns",
            skill_required="analysis",
            parameters={'target': target, 'analysis_type': 'general'},
            dependencies=[],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"identify_issues_{int(time.time())}",
            name="Identify Issues",
            description=f"Identify potential issues in {target}",
            skill_required="analysis",
            parameters={'target': target, 'analysis_type': 'issues'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.NORMAL
        ))
        
        tasks.append(Task(
            id=f"generate_report_{int(time.time())}",
            name="Generate Analysis Report",
            description=f"Generate comprehensive analysis report for {target}",
            skill_required="reporting",
            parameters={'target': target, 'report_type': 'analysis'},
            dependencies=[tasks[0].id, tasks[1].id],
            priority=TaskPriority.NORMAL
        ))
        
        return tasks
        
    def _decompose_summary_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose summary goals."""
        tasks = []
        
        target = context.get('target', 'the specified content')
        
        tasks.append(Task(
            id=f"extract_key_points_{int(time.time())}",
            name="Extract Key Points",
            description=f"Extract key points from {target}",
            skill_required="summarization",
            parameters={'target': target, 'extraction_type': 'key_points'},
            dependencies=[],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"create_summary_{int(time.time())}",
            name="Create Summary",
            description=f"Create concise summary of {target}",
            skill_required="summarization",
            parameters={'target': target, 'summary_type': 'concise'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.NORMAL
        ))
        
        return tasks
        
    def _decompose_comparison_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose comparison goals."""
        tasks = []
        
        items = context.get('items', ['item1', 'item2'])
        
        for i, item in enumerate(items):
            tasks.append(Task(
                id=f"analyze_item_{i}_{int(time.time())}",
                name=f"Analyze {item}",
                description=f"Analyze {item} for comparison",
                skill_required="analysis",
                parameters={'target': item, 'analysis_type': 'comparison'},
                dependencies=[],
                priority=TaskPriority.NORMAL
            ))
            
        tasks.append(Task(
            id=f"compare_items_{int(time.time())}",
            name="Compare Items",
            description=f"Compare {', '.join(items)}",
            skill_required="comparison",
            parameters={'items': items, 'comparison_type': 'detailed'},
            dependencies=[task.id for task in tasks[:-1]],
            priority=TaskPriority.HIGH
        ))
        
        return tasks
        
    def _decompose_creation_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose creation goals."""
        tasks = []
        
        target_type = context.get('target_type', 'content')
        
        tasks.append(Task(
            id=f"plan_creation_{int(time.time())}",
            name="Plan Creation",
            description=f"Plan the creation of {target_type}",
            skill_required="planning",
            parameters={'target_type': target_type, 'planning_type': 'creation'},
            dependencies=[],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"create_content_{int(time.time())}",
            name="Create Content",
            description=f"Create the {target_type}",
            skill_required="creation",
            parameters={'target_type': target_type, 'creation_type': 'content'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"review_creation_{int(time.time())}",
            name="Review Creation",
            description=f"Review the created {target_type}",
            skill_required="review",
            parameters={'target_type': target_type, 'review_type': 'quality'},
            dependencies=[tasks[1].id],
            priority=TaskPriority.NORMAL
        ))
        
        return tasks
        
    def _decompose_improvement_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose improvement goals."""
        tasks = []
        
        target = context.get('target', 'the specified item')
        
        tasks.append(Task(
            id=f"assess_current_{int(time.time())}",
            name="Assess Current State",
            description=f"Assess current state of {target}",
            skill_required="assessment",
            parameters={'target': target, 'assessment_type': 'current_state'},
            dependencies=[],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"identify_improvements_{int(time.time())}",
            name="Identify Improvements",
            description=f"Identify potential improvements for {target}",
            skill_required="analysis",
            parameters={'target': target, 'analysis_type': 'improvements'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"implement_improvements_{int(time.time())}",
            name="Implement Improvements",
            description=f"Implement identified improvements",
            skill_required="implementation",
            parameters={'target': target, 'implementation_type': 'improvements'},
            dependencies=[tasks[1].id],
            priority=TaskPriority.HIGH
        ))
        
        return tasks
        
    def _decompose_security_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose security analysis goals."""
        tasks = []
        
        target = context.get('target', 'the specified system')
        
        tasks.append(Task(
            id=f"security_scan_{int(time.time())}",
            name="Security Scan",
            description=f"Perform security scan of {target}",
            skill_required="security",
            parameters={'target': target, 'scan_type': 'comprehensive'},
            dependencies=[],
            priority=TaskPriority.CRITICAL
        ))
        
        tasks.append(Task(
            id=f"vulnerability_analysis_{int(time.time())}",
            name="Vulnerability Analysis",
            description=f"Analyze vulnerabilities in {target}",
            skill_required="security",
            parameters={'target': target, 'analysis_type': 'vulnerabilities'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.CRITICAL
        ))
        
        tasks.append(Task(
            id=f"security_recommendations_{int(time.time())}",
            name="Security Recommendations",
            description=f"Generate security recommendations for {target}",
            skill_required="security",
            parameters={'target': target, 'recommendation_type': 'security'},
            dependencies=[tasks[1].id],
            priority=TaskPriority.HIGH
        ))
        
        return tasks
        
    def _decompose_performance_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose performance analysis goals."""
        tasks = []
        
        target = context.get('target', 'the specified system')
        
        tasks.append(Task(
            id=f"performance_benchmark_{int(time.time())}",
            name="Performance Benchmark",
            description=f"Benchmark performance of {target}",
            skill_required="performance",
            parameters={'target': target, 'benchmark_type': 'comprehensive'},
            dependencies=[],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"bottleneck_analysis_{int(time.time())}",
            name="Bottleneck Analysis",
            description=f"Analyze performance bottlenecks in {target}",
            skill_required="performance",
            parameters={'target': target, 'analysis_type': 'bottlenecks'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"optimization_recommendations_{int(time.time())}",
            name="Optimization Recommendations",
            description=f"Generate optimization recommendations for {target}",
            skill_required="performance",
            parameters={'target': target, 'recommendation_type': 'optimization'},
            dependencies=[tasks[1].id],
            priority=TaskPriority.NORMAL
        ))
        
        return tasks
        
    def _decompose_documentation_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose documentation goals."""
        tasks = []
        
        target = context.get('target', 'the specified system')
        
        tasks.append(Task(
            id=f"analyze_documentation_needs_{int(time.time())}",
            name="Analyze Documentation Needs",
            description=f"Analyze documentation needs for {target}",
            skill_required="documentation",
            parameters={'target': target, 'analysis_type': 'documentation_needs'},
            dependencies=[],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"create_documentation_{int(time.time())}",
            name="Create Documentation",
            description=f"Create documentation for {target}",
            skill_required="documentation",
            parameters={'target': target, 'documentation_type': 'comprehensive'},
            dependencies=[tasks[0].id],
            priority=TaskPriority.HIGH
        ))
        
        tasks.append(Task(
            id=f"review_documentation_{int(time.time())}",
            name="Review Documentation",
            description=f"Review and validate documentation for {target}",
            skill_required="documentation",
            parameters={'target': target, 'review_type': 'quality'},
            dependencies=[tasks[1].id],
            priority=TaskPriority.NORMAL
        ))
        
        return tasks
        
    def _decompose_default_goal(self, goal_description: str, context: Dict[str, Any]) -> List[Task]:
        """Default goal decomposition."""
        tasks = []
        
        target = context.get('target', 'the specified item')
        
        tasks.append(Task(
            id=f"process_{int(time.time())}",
            name="Process Goal",
            description=f"Process the goal: {goal_description}",
            skill_required="general",
            parameters={'goal': goal_description, 'target': target},
            dependencies=[],
            priority=TaskPriority.NORMAL
        ))
        
        return tasks
