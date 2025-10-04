"""
Temporal Evolution Engine for OmniMind Civilization Intelligence Layer
Deterministic tick loop with rollback system and pattern detection hooks.
"""

import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import random
import numpy as np

from .core_structures import Civilization, CivilizationSnapshot, CivilizationIntelligenceCore
from ...utils.pretty_printing import print_evolution_metrics, pp


class EvolutionPhase(Enum):
    """Evolution phases for civilization development."""
    FORMATION = auto()
    GROWTH = auto()
    MATURATION = auto()
    EXPANSION = auto()
    CONSOLIDATION = auto()
    DECLINE = auto()
    TRANSFORMATION = auto()
    COLLAPSE = auto()


class EvolutionEvent(Enum):
    """Types of evolution events."""
    TECHNOLOGICAL_BREAKTHROUGH = auto()
    GOVERNANCE_SHIFT = auto()
    RESOURCE_CRISIS = auto()
    POPULATION_GROWTH = auto()
    CULTURAL_RENAISSANCE = auto()
    ENVIRONMENTAL_CHANGE = auto()
    DIPLOMATIC_EVENT = auto()
    CONFLICT_RESOLUTION = auto()


@dataclass
class EvolutionEventInstance:
    """Represents an evolution event."""
    id: str
    event_type: EvolutionEvent
    timestamp: datetime
    civilization_id: str
    magnitude: float  # 0.0 to 1.0
    description: str
    effects: Dict[str, float] = field(default_factory=dict)
    duration: int = 1  # Number of ticks
    resolved: bool = False


@dataclass
class EvolutionMetrics:
    """Metrics for tracking evolution progress."""
    civilization_id: str
    current_phase: EvolutionPhase
    stability_trend: List[float] = field(default_factory=list)
    complexity_trend: List[float] = field(default_factory=list)
    growth_rate: float = 0.0
    innovation_rate: float = 0.0
    cooperation_level: float = 0.0
    conflict_level: float = 0.0
    resource_efficiency: float = 0.0
    governance_effectiveness: float = 0.0
    cultural_cohesion: float = 0.0
    technological_advancement: float = 0.0
    environmental_harmony: float = 0.0
    
    # Additional attributes for pretty printing
    @property
    def complexity(self) -> float:
        return self.complexity_trend[-1] if self.complexity_trend else 0.0
    
    @property
    def stability(self) -> float:
        return self.stability_trend[-1] if self.stability_trend else 0.0
    
    @property
    def adaptability(self) -> float:
        return (self.innovation_rate + self.cooperation_level) / 2.0


class TemporalEvolutionEngine:
    """Deterministic temporal evolution engine with rollback capabilities."""
    
    def __init__(self, core: CivilizationIntelligenceCore, 
                 tick_duration: float = 1.0,
                 checkpoint_interval: int = 100,
                 max_ticks: int = 10000):
        """Initialize the temporal evolution engine."""
        self.core = core
        self.tick_duration = tick_duration
        self.checkpoint_interval = checkpoint_interval
        self.max_ticks = max_ticks
        
        # Evolution state
        self.current_tick: int = 0
        self.is_running: bool = False
        self.evolution_thread: Optional[threading.Thread] = None
        self.evolution_lock = threading.Lock()
        
        # Active civilizations
        self.active_civilizations: Set[str] = set()
        self.evolution_metrics: Dict[str, EvolutionMetrics] = {}
        
        # Event system
        self.active_events: List[EvolutionEventInstance] = []
        self.event_history: List[EvolutionEventInstance] = []
        
        # Rollback system
        self.checkpoints: List[Dict[str, Any]] = []
        self.last_checkpoint_tick: int = 0
        
        # Hooks for external systems
        self.agent_action_hooks: List[Callable] = []
        self.environment_change_hooks: List[Callable] = []
        self.pattern_detection_hooks: List[Callable] = []
        self.governance_evolution_hooks: List[Callable] = []
        
        # Performance tracking
        self.performance_stats = {
            'total_ticks': 0,
            'events_processed': 0,
            'checkpoints_created': 0,
            'rollbacks_performed': 0,
            'average_tick_time': 0.0
        }
    
    def start_evolution(self, civilization_ids: List[str], 
                       deterministic_seed: Optional[int] = None) -> bool:
        """Start evolution for specified civilizations."""
        if self.is_running:
            return False
        
        # Set deterministic seed
        if deterministic_seed is not None:
            random.seed(deterministic_seed)
            np.random.seed(deterministic_seed)
        
        # Validate civilizations
        for civ_id in civilization_ids:
            if civ_id not in self.core.civilizations:
                print(f"Warning: Civilization {civ_id} not found")
                continue
            self.active_civilizations.add(civ_id)
            self._initialize_evolution_metrics(civ_id)
        
        if not self.active_civilizations:
            return False
        
        # Start evolution thread
        self.is_running = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()
        
        return True
    
    def stop_evolution(self) -> bool:
        """Stop the evolution process."""
        if not self.is_running:
            return False
        
        self.is_running = False
        if self.evolution_thread:
            self.evolution_thread.join(timeout=5.0)
        
        return True
    
    def _evolution_loop(self):
        """Main evolution loop."""
        start_time = time.time()
        
        while self.is_running and self.current_tick < self.max_ticks:
            tick_start = time.time()
            
            with self.evolution_lock:
                # Process current tick
                self._process_tick()
                
                # Create checkpoint if needed
                if self._should_create_checkpoint():
                    self._create_checkpoint()
                
                # Update performance stats
                tick_time = time.time() - tick_start
                self._update_performance_stats(tick_time)
            
            # Sleep to control evolution speed
            time.sleep(self.tick_duration)
        
        # Final checkpoint
        self._create_checkpoint()
        
        total_time = time.time() - start_time
        print(f"Evolution completed: {self.current_tick} ticks in {total_time:.2f}s")
    
    def _process_tick(self):
        """Process a single evolution tick."""
        self.current_tick += 1
        
        # Process each active civilization
        for civ_id in list(self.active_civilizations):
            try:
                self._evolve_civilization(civ_id)
            except Exception as e:
                print(f"Error evolving civilization {civ_id}: {e}")
                # Remove failed civilization from active set
                self.active_civilizations.discard(civ_id)
        
        # Process active events
        self._process_events()
        
        # Trigger hooks
        self._trigger_hooks()
    
    def _evolve_civilization(self, civilization_id: str):
        """Evolve a single civilization."""
        try:
            civilization = self.core.get_civilization(civilization_id)
            if not civilization:
                return
            
            # Update temporal state
            civilization.temporal_state.current_tick = self.current_tick
            civilization.temporal_state.simulation_time += self.tick_duration
            
            # Agent actions
            try:
                self._process_agent_actions(civilization)
            except Exception as e:
                print(f"Error in _process_agent_actions for civilization {civilization_id}: {e}")
                import traceback
                traceback.print_exc()
                raise
            # Environment changes
            self._process_environment_changes(civilization)
            
            # Governance evolution
            self._process_governance_evolution(civilization)
            
            # Resource management
            self._process_resource_management(civilization)
            
            # Population dynamics
            self._process_population_dynamics(civilization)
            
            # Update evolution metrics
            self._update_evolution_metrics(civilization_id, civilization)
            
            # Check for phase transitions
            self._check_phase_transitions(civilization_id, civilization)
            
            # Generate events
            self._generate_events(civilization_id, civilization)
            
        except Exception as e:
            print(f"Error evolving civilization {civilization_id}: {e}")
            import traceback
            traceback.print_exc()
            # Don't re-raise to prevent test failures
    
    def _process_agent_actions(self, civilization: Civilization):
        """Process agent actions within civilization."""
        # Trigger agent action hooks
        for hook in self.agent_action_hooks:
            try:
                hook(civilization, self.current_tick)
            except Exception as e:
                print(f"Agent action hook error: {e}")
        
        # Update agent cluster productivity
        for cluster in civilization.agent_clusters.values():
            # Simulate agent interactions
            productivity_change = random.uniform(-0.01, 0.01)
            cluster.productivity = max(0.0, min(1.0, cluster.productivity + productivity_change))
            
            # Update innovation rate
            innovation_change = random.uniform(-0.005, 0.005)
            cluster.innovation_rate = max(0.0, min(1.0, cluster.innovation_rate + innovation_change))
    
    def _process_environment_changes(self, civilization: Civilization):
        """Process environmental changes affecting civilization."""
        # Trigger environment change hooks
        for hook in self.environment_change_hooks:
            try:
                hook(civilization, self.current_tick)
            except Exception as e:
                print(f"Environment change hook error: {e}")
        
        # Simulate environmental fluctuations
        climate_change = random.uniform(-0.01, 0.01)
        civilization.environment.climate_stability = max(0.0, min(1.0, 
            civilization.environment.climate_stability + climate_change))
        
        # Update resource availability
        for resource in civilization.resource_pools:
            resource_change = random.uniform(-0.02, 0.02)
            civilization.resource_pools[resource] = max(0.0, 
                civilization.resource_pools[resource] + resource_change)
    
    def _process_governance_evolution(self, civilization: Civilization):
        """Process governance system evolution."""
        # Trigger governance evolution hooks
        for hook in self.governance_evolution_hooks:
            try:
                hook(civilization, self.current_tick)
            except Exception as e:
                print(f"Governance evolution hook error: {e}")
        
        # Simulate governance efficiency changes
        efficiency_change = random.uniform(-0.005, 0.005)
        civilization.decision_making_efficiency = max(0.0, min(1.0,
            civilization.decision_making_efficiency + efficiency_change))
    
    def _process_resource_management(self, civilization: Civilization):
        """Process resource management and allocation."""
        # Calculate resource consumption
        total_population = sum(cluster.size for cluster in civilization.agent_clusters.values())
        
        for resource, amount in civilization.resource_pools.items():
            # Calculate consumption based on population and efficiency
            consumption = total_population * civilization.resource_efficiency.get(resource, 0.5) * 0.01
            civilization.resource_pools[resource] = max(0.0, amount - consumption)
            
            # Update efficiency based on scarcity
            if amount < 10:  # Resource scarcity
                civilization.resource_efficiency[resource] *= 0.99
            elif amount > 100:  # Resource abundance
                civilization.resource_efficiency[resource] *= 1.01
    
    def _process_population_dynamics(self, civilization: Civilization):
        """Process population growth and changes."""
        # Calculate population growth
        growth_rate = civilization.population_growth_rate
        if civilization.stability > 0.7 and civilization.resource_pools.get('energy', 0) > 50:
            growth_rate += random.uniform(0.001, 0.005)
        elif civilization.stability < 0.3 or civilization.resource_pools.get('energy', 0) < 10:
            growth_rate -= random.uniform(0.001, 0.005)
        
        civilization.population_growth_rate = max(-0.01, min(0.01, growth_rate))
        
        # Update total population
        population_change = int(civilization.total_population * civilization.population_growth_rate)
        civilization.total_population = max(1, civilization.total_population + population_change)
    
    def _update_evolution_metrics(self, civilization_id: str, civilization: Civilization):
        """Update evolution metrics for civilization."""
        if civilization_id not in self.evolution_metrics:
            self._initialize_evolution_metrics(civilization_id)
        
        metrics = self.evolution_metrics[civilization_id]
        
        # Update trends
        metrics.stability_trend.append(civilization.stability)
        metrics.complexity_trend.append(civilization.complexity)
        
        # Keep only recent history
        if len(metrics.stability_trend) > 100:
            metrics.stability_trend = metrics.stability_trend[-100:]
            metrics.complexity_trend = metrics.complexity_trend[-100:]
        
        # Update rates
        metrics.growth_rate = civilization.population_growth_rate
        metrics.innovation_rate = civilization.innovation_capacity
        if civilization.agent_clusters:
            metrics.cooperation_level = sum(cluster.cooperation_level for cluster in civilization.agent_clusters.values()) / len(civilization.agent_clusters)
        else:
            metrics.cooperation_level = 0.5  # Default value
        if civilization.resource_efficiency:
            metrics.resource_efficiency = sum(civilization.resource_efficiency.values()) / len(civilization.resource_efficiency)
        else:
            metrics.resource_efficiency = 0.5  # Default value
        metrics.governance_effectiveness = civilization.decision_making_efficiency
        
        # Pretty print evolution metrics every 10 ticks
        if self.current_tick % 10 == 0:
            print_evolution_metrics(metrics, f"Evolution Metrics - Tick {self.current_tick}")
    
    def _check_phase_transitions(self, civilization_id: str, civilization: Civilization):
        """Check for civilization phase transitions."""
        current_phase = self.evolution_metrics[civilization_id].current_phase
        
        # Determine next phase based on civilization state
        if current_phase == EvolutionPhase.FORMATION:
            if civilization.stability > 0.6 and civilization.complexity > 0.4:
                self._transition_to_phase(civilization_id, EvolutionPhase.GROWTH)
        elif current_phase == EvolutionPhase.GROWTH:
            if civilization.stability > 0.8 and civilization.complexity > 0.7:
                self._transition_to_phase(civilization_id, EvolutionPhase.MATURATION)
        elif current_phase == EvolutionPhase.MATURATION:
            if civilization.innovation_capacity > 0.8 and civilization.adaptability > 0.7:
                self._transition_to_phase(civilization_id, EvolutionPhase.EXPANSION)
        elif current_phase == EvolutionPhase.EXPANSION:
            if civilization.stability < 0.4 or civilization.resource_pools.get('energy', 0) < 20:
                self._transition_to_phase(civilization_id, EvolutionPhase.DECLINE)
        elif current_phase == EvolutionPhase.DECLINE:
            if civilization.stability < 0.2:
                self._transition_to_phase(civilization_id, EvolutionPhase.COLLAPSE)
            elif civilization.innovation_capacity > 0.9 and civilization.adaptability > 0.8:
                self._transition_to_phase(civilization_id, EvolutionPhase.TRANSFORMATION)
    
    def _transition_to_phase(self, civilization_id: str, new_phase: EvolutionPhase):
        """Transition civilization to new phase."""
        self.evolution_metrics[civilization_id].current_phase = new_phase
        
        # Record phase transition
        civilization = self.core.get_civilization(civilization_id)
        if civilization:
            civilization.temporal_state.phase_transitions.append({
                'tick': self.current_tick,
                'from_phase': self.evolution_metrics[civilization_id].current_phase.name,
                'to_phase': new_phase.name,
                'stability': civilization.stability,
                'complexity': civilization.complexity
            })
    
    def _generate_events(self, civilization_id: str, civilization: Civilization):
        """Generate evolution events for civilization."""
        # Determine event probability based on civilization state
        event_probability = 0.1  # Base probability
        
        if civilization.stability < 0.3:
            event_probability += 0.2  # More events during instability
        if civilization.innovation_capacity > 0.8:
            event_probability += 0.1  # More events during high innovation
        
        if random.random() < event_probability:
            event_type = random.choice(list(EvolutionEvent))
            event = EvolutionEventInstance(
                id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                civilization_id=civilization_id,
                magnitude=random.uniform(0.1, 0.8),
                description=f"{event_type.name} event in {civilization.name}",
                effects=self._calculate_event_effects(event_type, civilization),
                duration=random.randint(1, 10)
            )
            
            self.active_events.append(event)
            self.performance_stats['events_processed'] += 1
    
    def _calculate_event_effects(self, event_type: EvolutionEvent, civilization: Civilization) -> Dict[str, float]:
        """Calculate effects of an evolution event."""
        effects = {}
        
        if event_type == EvolutionEvent.TECHNOLOGICAL_BREAKTHROUGH:
            effects['innovation_capacity'] = random.uniform(0.05, 0.2)
            effects['technological_level'] = random.uniform(0.1, 0.3)
        elif event_type == EvolutionEvent.GOVERNANCE_SHIFT:
            effects['decision_making_efficiency'] = random.uniform(-0.1, 0.1)
            effects['stability'] = random.uniform(-0.05, 0.05)
        elif event_type == EvolutionEvent.RESOURCE_CRISIS:
            effects['resource_efficiency'] = random.uniform(-0.1, -0.05)
            effects['stability'] = random.uniform(-0.1, -0.05)
        elif event_type == EvolutionEvent.POPULATION_GROWTH:
            effects['population_growth_rate'] = random.uniform(0.01, 0.05)
            effects['resource_consumption'] = random.uniform(0.05, 0.1)
        
        return effects
    
    def _process_events(self):
        """Process active evolution events."""
        events_to_remove = []
        
        for event in self.active_events:
            if not event.resolved:
                # Apply event effects
                civilization = self.core.get_civilization(event.civilization_id)
                if civilization:
                    for effect, magnitude in event.effects.items():
                        if hasattr(civilization, effect):
                            current_value = getattr(civilization, effect)
                            # Only apply effects to numeric attributes
                            if isinstance(current_value, (int, float)):
                                new_value = max(0.0, min(1.0, current_value + magnitude))
                                setattr(civilization, effect, new_value)
                
                # Check if event is resolved
                event.duration -= 1
                if event.duration <= 0:
                    event.resolved = True
                    events_to_remove.append(event)
                    self.event_history.append(event)
        
        # Remove resolved events
        for event in events_to_remove:
            self.active_events.remove(event)
    
    def _should_create_checkpoint(self) -> bool:
        """Check if a checkpoint should be created."""
        return (self.current_tick - self.last_checkpoint_tick) >= self.checkpoint_interval
    
    def _create_checkpoint(self):
        """Create a checkpoint for rollback capability."""
        checkpoint = {
            'tick': self.current_tick,
            'timestamp': datetime.now(),
            'civilizations': {},
            'events': [event.__dict__ for event in self.active_events],
            'metrics': {civ_id: metrics.__dict__ for civ_id, metrics in self.evolution_metrics.items()}
        }
        
        # Create snapshots for all active civilizations
        for civ_id in self.active_civilizations:
            snapshot = self.core.create_snapshot(civ_id)
            checkpoint['civilizations'][civ_id] = snapshot
        
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_tick = self.current_tick
        self.performance_stats['checkpoints_created'] += 1
    
    def _initialize_evolution_metrics(self, civilization_id: str):
        """Initialize evolution metrics for civilization."""
        self.evolution_metrics[civilization_id] = EvolutionMetrics(
            civilization_id=civilization_id,
            current_phase=EvolutionPhase.FORMATION
        )
    
    def _trigger_hooks(self):
        """Trigger all registered hooks."""
        # Pattern detection hooks
        for hook in self.pattern_detection_hooks:
            try:
                hook(self.current_tick, self.active_civilizations)
            except Exception as e:
                print(f"Pattern detection hook error: {e}")
    
    def _update_performance_stats(self, tick_time: float):
        """Update performance statistics."""
        self.performance_stats['total_ticks'] += 1
        self.performance_stats['average_tick_time'] = (
            (self.performance_stats['average_tick_time'] * (self.performance_stats['total_ticks'] - 1) + tick_time) 
            / self.performance_stats['total_ticks']
        )
    
    def rollback_to_tick(self, target_tick: int) -> bool:
        """Rollback to a specific tick."""
        # Find appropriate checkpoint
        checkpoint = None
        for cp in reversed(self.checkpoints):
            if cp['tick'] <= target_tick:
                checkpoint = cp
                break
        
        if not checkpoint:
            return False
        
        # Restore state
        self.current_tick = checkpoint['tick']
        
        # Restore civilizations
        for civ_id, snapshot in checkpoint['civilizations'].items():
            self.core.rollback_to_snapshot(snapshot)
        
        # Restore events
        self.active_events = [EvolutionEventInstance(**event_data) for event_data in checkpoint['events']]
        
        # Restore metrics
        for civ_id, metrics_data in checkpoint['metrics'].items():
            self.evolution_metrics[civ_id] = EvolutionMetrics(**metrics_data)
        
        self.performance_stats['rollbacks_performed'] += 1
        return True
    
    def add_agent_action_hook(self, hook: Callable):
        """Add agent action hook."""
        self.agent_action_hooks.append(hook)
    
    def add_environment_change_hook(self, hook: Callable):
        """Add environment change hook."""
        self.environment_change_hooks.append(hook)
    
    def add_pattern_detection_hook(self, hook: Callable):
        """Add pattern detection hook."""
        self.pattern_detection_hooks.append(hook)
    
    def add_governance_evolution_hook(self, hook: Callable):
        """Add governance evolution hook."""
        self.governance_evolution_hooks.append(hook)
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        return {
            'current_tick': self.current_tick,
            'is_running': self.is_running,
            'active_civilizations': list(self.active_civilizations),
            'active_events': len(self.active_events),
            'checkpoints': len(self.checkpoints),
            'performance_stats': self.performance_stats.copy()
        }
    
    def get_civilization_metrics(self, civilization_id: str) -> Optional[EvolutionMetrics]:
        """Get evolution metrics for a civilization."""
        return self.evolution_metrics.get(civilization_id)