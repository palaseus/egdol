"""
Conversation Memory Manager for OmniMind
Handles conversation history and persistent knowledge storage.
"""

import json
import time
import os
from typing import Dict, Any, List, Optional
from ..memory import MemoryStore


class ConversationMemory:
    """Manages conversation memory and persistent knowledge."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize memory store
        self.memory_store = MemoryStore(os.path.join(data_dir, "conversation_memory.db"))
        
        # Session data
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.current_session = None
        
    def store_input(self, user_input: str, session_id: str):
        """Store user input in memory."""
        self.memory_store.store(
            content=user_input,
            item_type='user_input',
            source=session_id,
            confidence=1.0,
            metadata={'timestamp': time.time()}
        )
        
        # Store in session
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        self.sessions[session_id].append({
            'type': 'user_input',
            'content': user_input,
            'timestamp': time.time()
        })
        
    def store_response(self, response: str, session_id: str):
        """Store assistant response in memory."""
        self.memory_store.store(
            content=response,
            item_type='assistant_response',
            source=session_id,
            confidence=0.9,
            metadata={'timestamp': time.time()}
        )
        
        # Store in session
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        self.sessions[session_id].append({
            'type': 'assistant_response',
            'content': response,
            'timestamp': time.time()
        })
        
    def store_fact(self, fact: str, session_id: str) -> int:
        """Store a fact in memory."""
        memory_id = self.memory_store.store(
            content=fact,
            item_type='fact',
            source=session_id,
            confidence=0.8,
            metadata={'timestamp': time.time(), 'explicit_fact': True}
        )
        
        return memory_id
        
    def store_rule(self, rule: str, session_id: str) -> int:
        """Store a rule in memory."""
        memory_id = self.memory_store.store(
            content=rule,
            item_type='rule',
            source=session_id,
            confidence=0.8,
            metadata={'timestamp': time.time(), 'explicit_rule': True}
        )
        
        return memory_id
        
    def get_recent_memories(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories for a session."""
        if session_id not in self.sessions:
            return []
            
        return self.sessions[session_id][-limit:]
        
    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Get all stored facts."""
        facts = self.memory_store.search(item_type='fact')
        return [{'content': fact.content, 'id': fact.id, 'confidence': fact.confidence} for fact in facts]
        
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all stored rules."""
        rules = self.memory_store.search(item_type='rule')
        return [{'content': rule.content, 'id': rule.id, 'confidence': rule.confidence} for rule in rules]
        
    def search_memories(self, query: str, session_id: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search memories by content."""
        memories = self.memory_store.search(limit=limit)
        
        # Filter by session if specified
        if session_id:
            memories = [m for m in memories if m.source == session_id]
            
        # Filter by query
        if query:
            memories = [m for m in memories if query.lower() in str(m.content).lower()]
            
        return [{'content': m.content, 'type': m.item_type, 'timestamp': m.timestamp} for m in memories]
        
    def forget_pattern(self, pattern: str, session_id: str = None) -> int:
        """Forget memories matching a pattern."""
        if session_id:
            return self.memory_store.forget(pattern=pattern, source=session_id)
        else:
            return self.memory_store.forget(pattern=pattern)
            
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session."""
        if session_id not in self.sessions:
            return {'session_id': session_id, 'messages': 0, 'duration': 0}
            
        session = self.sessions[session_id]
        if not session:
            return {'session_id': session_id, 'messages': 0, 'duration': 0}
            
        start_time = min(msg['timestamp'] for msg in session)
        end_time = max(msg['timestamp'] for msg in session)
        duration = end_time - start_time
        
        user_inputs = len([msg for msg in session if msg['type'] == 'user_input'])
        responses = len([msg for msg in session if msg['type'] == 'assistant_response'])
        
        return {
            'session_id': session_id,
            'messages': len(session),
            'user_inputs': user_inputs,
            'responses': responses,
            'duration': duration,
            'start_time': start_time,
            'end_time': end_time
        }
        
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions."""
        return [self.get_session_summary(session_id) for session_id in self.sessions.keys()]
        
    def get_context_for_session(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get conversation context for a session."""
        if session_id not in self.sessions:
            return []
            
        return self.sessions[session_id][-limit:]
        
    def export_session(self, session_id: str, file_path: str) -> bool:
        """Export a session to a file."""
        try:
            if session_id not in self.sessions:
                return False
                
            session_data = {
                'session_id': session_id,
                'messages': self.sessions[session_id],
                'summary': self.get_session_summary(session_id),
                'export_timestamp': time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            return True
        except Exception:
            return False
            
    def import_session(self, file_path: str) -> bool:
        """Import a session from a file."""
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
                
            session_id = session_data['session_id']
            self.sessions[session_id] = session_data['messages']
            return True
        except Exception:
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        memory_stats = self.memory_store.get_stats()
        
        return {
            'total_sessions': len(self.sessions),
            'total_memories': memory_stats['total_items'],
            'by_type': memory_stats['by_type'],
            'recent_memories': memory_stats['recent_items']
        }
        
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old sessions."""
        cutoff_time = time.time() - (days_old * 86400)
        
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session:
                last_activity = max(msg['timestamp'] for msg in session)
                if last_activity < cutoff_time:
                    sessions_to_remove.append(session_id)
                    
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            
        return len(sessions_to_remove)
        
    def get_memory_health(self) -> Dict[str, Any]:
        """Get memory health metrics."""
        stats = self.get_stats()
        
        # Calculate health score
        total_memories = stats['total_memories']
        if total_memories == 0:
            return {'status': 'empty', 'score': 0.0}
            
        # Check for recent activity
        recent_memories = stats['recent_memories']
        activity_score = min(recent_memories / 10.0, 1.0)
        
        # Check for session diversity
        session_count = stats['total_sessions']
        diversity_score = min(session_count / 5.0, 1.0)
        
        # Overall health score
        health_score = (activity_score + diversity_score) / 2
        
        if health_score > 0.8:
            status = 'excellent'
        elif health_score > 0.6:
            status = 'good'
        elif health_score > 0.4:
            status = 'fair'
        else:
            status = 'poor'
            
        return {
            'status': status,
            'score': health_score,
            'activity_score': activity_score,
            'diversity_score': diversity_score,
            'total_memories': total_memories,
            'total_sessions': session_count
        }
