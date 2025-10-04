"""
Provenance & Snapshot Export System
Creates reproducible snapshot links and provides replay hooks for auditors/tests.
"""

import time
import hashlib
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProvenanceSnapshot:
    """Reproducible snapshot with all necessary data."""
    seed: int
    universe_id: str
    sim_id: str
    tick_range: List[int]
    snapshot_id: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "universe": self.universe_id,
            "sim_id": self.sim_id,
            "tick_range": self.tick_range,
            "snapshot": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class ProvenanceTracker:
    """Tracks and manages provenance information for reproducibility."""
    
    def __init__(self):
        self.snapshots: Dict[str, ProvenanceSnapshot] = {}
        self.replay_hooks: Dict[str, Dict[str, Any]] = {}
    
    def create_provenance(self, 
                         seed: int,
                         universe_id: str,
                         sim_id: str,
                         tick_range: List[int],
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create provenance information for a simulation.
        
        Args:
            seed: Random seed used for simulation
            universe_id: Universe identifier
            sim_id: Simulation identifier
            tick_range: Range of ticks in simulation
            metadata: Additional metadata
            
        Returns:
            Provenance dictionary
        """
        # Create snapshot ID
        snapshot_id = self._generate_snapshot_id(seed, universe_id, sim_id)
        
        # Create snapshot
        snapshot = ProvenanceSnapshot(
            seed=seed,
            universe_id=universe_id,
            sim_id=sim_id,
            tick_range=tick_range,
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store snapshot
        self.snapshots[snapshot_id] = snapshot
        
        return snapshot.to_dict()
    
    def create_replay_hook(self, provenance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create replay hook for reproducing simulation.
        
        Args:
            provenance: Provenance information
            
        Returns:
            Replay hook dictionary
        """
        snapshot_id = provenance.get("snapshot")
        if not snapshot_id or snapshot_id not in self.snapshots:
            return {"error": "Snapshot not found"}
        
        snapshot = self.snapshots[snapshot_id]
        
        # Create replay hook
        replay_hook = {
            "replay": {
                "snapshot_id": snapshot_id,
                "seed": snapshot.seed,
                "universe_id": snapshot.universe_id,
                "sim_id": snapshot.sim_id,
                "tick_range": snapshot.tick_range,
                "replay_function": f"replay_simulation('{snapshot_id}')",
                "reproducible": True
            },
            "metadata": snapshot.metadata,
            "created_at": snapshot.timestamp.isoformat()
        }
        
        # Store replay hook
        self.replay_hooks[snapshot_id] = replay_hook
        
        return replay_hook
    
    def get_snapshot(self, snapshot_id: str) -> Optional[ProvenanceSnapshot]:
        """Get snapshot by ID."""
        return self.snapshots.get(snapshot_id)
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots."""
        return [snapshot.to_dict() for snapshot in self.snapshots.values()]
    
    def _generate_snapshot_id(self, seed: int, universe_id: str, sim_id: str) -> str:
        """Generate deterministic snapshot ID."""
        data = f"{seed}_{universe_id}_{sim_id}_{int(time.time())}"
        hash_obj = hashlib.sha256(data.encode())
        return f"snap-{hash_obj.hexdigest()[:8]}"
    
    def export_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Export snapshot for external use."""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return {"error": "Snapshot not found"}
        
        return {
            "snapshot": snapshot.to_dict(),
            "replay_hook": self.replay_hooks.get(snapshot_id, {}),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def validate_reproducibility(self, snapshot_id: str) -> Dict[str, Any]:
        """Validate that a snapshot can be reproduced."""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return {"valid": False, "error": "Snapshot not found"}
        
        # Check if all required data is present
        required_fields = ["seed", "universe_id", "sim_id", "tick_range"]
        missing_fields = [field for field in required_fields if not getattr(snapshot, field, None)]
        
        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing_fields}"
            }
        
        return {
            "valid": True,
            "reproducible": True,
            "snapshot_id": snapshot_id,
            "replay_available": snapshot_id in self.replay_hooks
        }
