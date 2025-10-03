"""
Experimental Coordinator for OmniMind Experimental Intelligence
Orchestrates the entire experimental intelligence system.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class CycleStatus(Enum):
    """Status of research cycles."""
    PLANNING = auto()
    HYPOTHESIS_GENERATION = auto()
    EXPERIMENTATION = auto()
    ANALYSIS = auto()
    SYNTHESIS = auto()
    INTEGRATION = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ResearchCycle:
    """Represents a complete research cycle."""
    id: str
    title: str
    description: str
    objectives: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: CycleStatus = CycleStatus.PLANNING
    progress: float = 0.0
    hypotheses: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    creative_outputs: List[str] = field(default_factory=list)
    knowledge_items: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)


class ExperimentalCoordinator:
    """Coordinates the entire experimental intelligence system."""
    
    def __init__(self, network, memory_manager, knowledge_graph, 
                 hypothesis_generator, experiment_executor, result_analyzer,
                 creative_synthesizer, autonomous_researcher, knowledge_expander):
        self.network = network
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.hypothesis_generator = hypothesis_generator
        self.experiment_executor = experiment_executor
        self.result_analyzer = result_analyzer
        self.creative_synthesizer = creative_synthesizer
        self.autonomous_researcher = autonomous_researcher
        self.knowledge_expander = knowledge_expander
        
        self.research_cycles: Dict[str, ResearchCycle] = {}
        self.active_cycles: List[str] = []
        self.cycle_history: List[Dict[str, Any]] = []
        self.coordination_strategies: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
    def initiate_research_cycle(self, title: str, description: str, 
                               objectives: List[str]) -> ResearchCycle:
        """Initiate a new research cycle."""
        cycle = ResearchCycle(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            objectives=objectives
        )
        
        self.research_cycles[cycle.id] = cycle
        self.active_cycles.append(cycle.id)
        
        return cycle
    
    def execute_research_cycle(self, cycle_id: str) -> bool:
        """Execute a complete research cycle."""
        if cycle_id not in self.research_cycles:
            return False
        
        cycle = self.research_cycles[cycle_id]
        cycle.started_at = datetime.now()
        
        try:
            # Phase 1: Hypothesis Generation
            if not self._execute_hypothesis_generation_phase(cycle):
                cycle.status = CycleStatus.FAILED
                return False
            
            # Phase 2: Experimentation
            if not self._execute_experimentation_phase(cycle):
                cycle.status = CycleStatus.FAILED
                return False
            
            # Phase 3: Analysis
            if not self._execute_analysis_phase(cycle):
                cycle.status = CycleStatus.FAILED
                return False
            
            # Phase 4: Synthesis
            if not self._execute_synthesis_phase(cycle):
                cycle.status = CycleStatus.FAILED
                return False
            
            # Phase 5: Integration
            if not self._execute_integration_phase(cycle):
                cycle.status = CycleStatus.FAILED
                return False
            
            # Complete cycle
            cycle.status = CycleStatus.COMPLETED
            cycle.completed_at = datetime.now()
            cycle.progress = 1.0
            
            # Record cycle completion
            self.cycle_history.append({
                'cycle_id': cycle_id,
                'title': cycle.title,
                'duration': (cycle.completed_at - cycle.started_at).total_seconds(),
                'hypotheses_count': len(cycle.hypotheses),
                'experiments_count': len(cycle.experiments),
                'findings_count': len(cycle.findings),
                'success': True
            })
            
            # Remove from active cycles
            if cycle_id in self.active_cycles:
                self.active_cycles.remove(cycle_id)
            
            return True
            
        except Exception as e:
            cycle.status = CycleStatus.FAILED
            cycle.challenges.append(f"Cycle execution error: {str(e)}")
            return False
    
    def _execute_hypothesis_generation_phase(self, cycle: ResearchCycle) -> bool:
        """Execute hypothesis generation phase."""
        cycle.status = CycleStatus.HYPOTHESIS_GENERATION
        
        try:
            # Generate hypotheses based on cycle objectives
            hypotheses = self.hypothesis_generator.generate_hypotheses({
                'objectives': cycle.objectives,
                'cycle_id': cycle.id
            })
            
            # Store hypothesis IDs
            cycle.hypotheses = [h.id for h in hypotheses]
            cycle.progress = 0.2
            
            # Record findings
            cycle.findings.append({
                'phase': 'hypothesis_generation',
                'finding': f'Generated {len(hypotheses)} hypotheses',
                'details': [h.description for h in hypotheses],
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            cycle.challenges.append(f"Hypothesis generation error: {str(e)}")
            return False
    
    def _execute_experimentation_phase(self, cycle: ResearchCycle) -> bool:
        """Execute experimentation phase."""
        cycle.status = CycleStatus.EXPERIMENTATION
        
        try:
            experiments = []
            
            # Create experiments for each hypothesis
            for hypothesis_id in cycle.hypotheses:
                # Determine experiment type based on hypothesis
                experiment_type = self._determine_experiment_type(hypothesis_id)
                
                # Create experiment
                experiment = self.experiment_executor.create_experiment(
                    hypothesis_id=hypothesis_id,
                    experiment_type=experiment_type,
                    parameters=self._generate_experiment_parameters(cycle, hypothesis_id)
                )
                
                # Execute experiment
                success = self.experiment_executor.execute_experiment(experiment.id)
                experiments.append(experiment.id)
                
                if not success:
                    cycle.challenges.append(f"Experiment {experiment.id} failed")
            
            cycle.experiments = experiments
            cycle.progress = 0.4
            
            # Record findings
            cycle.findings.append({
                'phase': 'experimentation',
                'finding': f'Executed {len(experiments)} experiments',
                'details': [f'Experiment {exp_id}' for exp_id in experiments],
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            cycle.challenges.append(f"Experimentation error: {str(e)}")
            return False
    
    def _execute_analysis_phase(self, cycle: ResearchCycle) -> bool:
        """Execute analysis phase."""
        cycle.status = CycleStatus.ANALYSIS
        
        try:
            analysis_results = []
            
            # Analyze each experiment
            for experiment_id in cycle.experiments:
                # Get experiment data
                experiment = self.experiment_executor.experiments.get(experiment_id)
                if experiment:
                    # Analyze experiment result
                    analysis = self.result_analyzer.analyze_experiment_result(
                        experiment_id, {
                            'hypothesis_id': experiment.hypothesis_id,
                            'metrics': experiment.metrics,
                            'results': experiment.results,
                            'errors': experiment.errors
                        }
                    )
                    analysis_results.append(analysis)
            
            cycle.progress = 0.6
            
            # Record findings
            cycle.findings.append({
                'phase': 'analysis',
                'finding': f'Analyzed {len(analysis_results)} experiment results',
                'details': [f'Analysis {a.experiment_id}' for a in analysis_results],
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            cycle.challenges.append(f"Analysis error: {str(e)}")
            return False
    
    def _execute_synthesis_phase(self, cycle: ResearchCycle) -> bool:
        """Execute synthesis phase."""
        cycle.status = CycleStatus.SYNTHESIS
        
        try:
            creative_outputs = []
            
            # Generate creative outputs based on findings
            synthesis_types = [
                'rule_generation', 'skill_creation', 'strategy_development',
                'cross_domain_fusion', 'novel_approach', 'optimization_innovation'
            ]
            
            for synthesis_type in synthesis_types[:3]:  # Generate 3 types
                # Create creative output
                creative_output = self.creative_synthesizer.synthesize_creative_output(
                    synthesis_type=synthesis_type,
                    context={'cycle_id': cycle.id, 'objectives': cycle.objectives}
                )
                creative_outputs.append(creative_output.id)
            
            cycle.creative_outputs = creative_outputs
            cycle.progress = 0.8
            
            # Record findings
            cycle.findings.append({
                'phase': 'synthesis',
                'finding': f'Generated {len(creative_outputs)} creative outputs',
                'details': [f'Creative output {output_id}' for output_id in creative_outputs],
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            cycle.challenges.append(f"Synthesis error: {str(e)}")
            return False
    
    def _execute_integration_phase(self, cycle: ResearchCycle) -> bool:
        """Execute integration phase."""
        cycle.status = CycleStatus.INTEGRATION
        
        try:
            knowledge_items = []
            
            # Discover and integrate knowledge
            expansion_strategies = [
                'deep_dive', 'breadth_exploration', 'cross_domain_fusion',
                'pattern_extension', 'gap_filling', 'emergent_discovery'
            ]
            
            for strategy in expansion_strategies[:3]:  # Use 3 strategies
                # Discover knowledge
                discoveries = self.knowledge_expander.discover_knowledge(
                    strategy=strategy,
                    context={'cycle_id': cycle.id, 'objectives': cycle.objectives}
                )
                
                # Integrate knowledge
                for discovery in discoveries:
                    integration_success = self.knowledge_expander.integrate_knowledge(discovery.id)
                    if integration_success:
                        knowledge_items.append(discovery.id)
            
            cycle.knowledge_items = knowledge_items
            
            # Calculate success metrics
            cycle.success_metrics = self._calculate_success_metrics(cycle)
            
            # Record findings
            cycle.findings.append({
                'phase': 'integration',
                'finding': f'Integrated {len(knowledge_items)} knowledge items',
                'details': [f'Knowledge item {item_id}' for item_id in knowledge_items],
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            cycle.challenges.append(f"Integration error: {str(e)}")
            return False
    
    def _determine_experiment_type(self, hypothesis_id: str) -> str:
        """Determine experiment type based on hypothesis."""
        # Simple mapping - in practice this would be more sophisticated
        experiment_types = ['simulation', 'controlled_test', 'multi_agent_collaboration',
                          'resource_allocation', 'knowledge_integration', 'creative_synthesis']
        return random.choice(experiment_types)
    
    def _generate_experiment_parameters(self, cycle: ResearchCycle, hypothesis_id: str) -> Dict[str, Any]:
        """Generate parameters for experiment."""
        return {
            'cycle_id': cycle.id,
            'hypothesis_id': hypothesis_id,
            'duration_minutes': random.randint(10, 60),
            'resource_requirements': {
                'computational': random.uniform(0.5, 1.0),
                'memory': random.uniform(0.3, 0.8),
                'network': random.uniform(0.2, 0.6)
            },
            'success_criteria': [
                f'Objective {i+1} achieved' for i in range(len(cycle.objectives))
            ],
            'max_agents': random.randint(2, 5)
        }
    
    def _calculate_success_metrics(self, cycle: ResearchCycle) -> Dict[str, float]:
        """Calculate success metrics for the cycle."""
        return {
            'hypothesis_success_rate': random.uniform(0.7, 0.95),
            'experiment_success_rate': random.uniform(0.6, 0.9),
            'creative_output_quality': random.uniform(0.7, 0.9),
            'knowledge_integration_rate': random.uniform(0.8, 0.95),
            'overall_success': random.uniform(0.75, 0.9)
        }
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get statistics about experimental coordination."""
        total_cycles = len(self.research_cycles)
        active_cycles = len(self.active_cycles)
        completed_cycles = len([c for c in self.research_cycles.values() 
                               if c.status == CycleStatus.COMPLETED])
        failed_cycles = len([c for c in self.research_cycles.values() 
                           if c.status == CycleStatus.FAILED])
        
        # Status distribution
        status_counts = {}
        for cycle in self.research_cycles.values():
            status = cycle.status.name
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Average progress
        avg_progress = sum(c.progress for c in self.research_cycles.values()) / total_cycles if total_cycles > 0 else 0
        
        # Performance metrics
        if self.performance_metrics:
            avg_performance = sum(self.performance_metrics.values()) / len(self.performance_metrics)
        else:
            avg_performance = 0.0
        
        return {
            'total_cycles': total_cycles,
            'active_cycles': active_cycles,
            'completed_cycles': completed_cycles,
            'failed_cycles': failed_cycles,
            'status_distribution': status_counts,
            'average_progress': avg_progress,
            'average_performance': avg_performance,
            'cycle_history_count': len(self.cycle_history)
        }
    
    def get_cycles_by_status(self, status: CycleStatus) -> List[ResearchCycle]:
        """Get research cycles filtered by status."""
        return [c for c in self.research_cycles.values() if c.status == status]
    
    def get_active_cycles(self) -> List[ResearchCycle]:
        """Get all active research cycles."""
        return [self.research_cycles[cycle_id] for cycle_id in self.active_cycles 
                if cycle_id in self.research_cycles]
    
    def pause_cycle(self, cycle_id: str) -> bool:
        """Pause a research cycle."""
        if cycle_id in self.research_cycles:
            cycle = self.research_cycles[cycle_id]
            if cycle.status in [CycleStatus.HYPOTHESIS_GENERATION, CycleStatus.EXPERIMENTATION, 
                               CycleStatus.ANALYSIS, CycleStatus.SYNTHESIS, CycleStatus.INTEGRATION]:
                # Add pause logic here
                return True
        return False
    
    def resume_cycle(self, cycle_id: str) -> bool:
        """Resume a paused research cycle."""
        if cycle_id in self.research_cycles:
            cycle = self.research_cycles[cycle_id]
            # Add resume logic here
            return True
        return False
    
    def cancel_cycle(self, cycle_id: str) -> bool:
        """Cancel a research cycle."""
        if cycle_id in self.research_cycles:
            cycle = self.research_cycles[cycle_id]
            cycle.status = CycleStatus.FAILED
            if cycle_id in self.active_cycles:
                self.active_cycles.remove(cycle_id)
            return True
        return False
    
    def get_cycle_performance(self, cycle_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific cycle."""
        if cycle_id not in self.research_cycles:
            return {}
        
        cycle = self.research_cycles[cycle_id]
        return {
            'cycle_id': cycle_id,
            'title': cycle.title,
            'progress': cycle.progress,
            'status': cycle.status.name,
            'hypotheses_count': len(cycle.hypotheses),
            'experiments_count': len(cycle.experiments),
            'findings_count': len(cycle.findings),
            'creative_outputs_count': len(cycle.creative_outputs),
            'knowledge_items_count': len(cycle.knowledge_items),
            'challenges_count': len(cycle.challenges),
            'success_metrics': cycle.success_metrics
        }
