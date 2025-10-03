"""
Persistent Memory Store for Egdol.
Provides SQLite-based storage for facts, rules, and context.
"""

import sqlite3
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from ..parser import Term, Variable, Constant, Rule, Fact


class MemoryItem:
    """Represents a memory item with metadata."""
    
    def __init__(self, id_: int, content: Any, item_type: str, 
                 timestamp: float, source: str, confidence: float = 1.0,
                 metadata: Dict[str, Any] = None):
        self.id = id_
        self.content = content
        self.item_type = item_type  # 'fact', 'rule', 'context', 'watcher'
        self.timestamp = timestamp
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'item_type': self.item_type,
            'timestamp': self.timestamp,
            'source': self.source,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        return cls(
            id_=data['id'],
            content=data['content'],
            item_type=data['item_type'],
            timestamp=data['timestamp'],
            source=data['source'],
            confidence=data['confidence'],
            metadata=data.get('metadata', {})
        )


class MemoryStore:
    """Persistent memory store using SQLite."""
    
    def __init__(self, db_path: str = "egdol_memory.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for fast retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON memory(item_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memory(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON memory(source)
            """)
            
            # Create versioning table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memory(id)
                )
            """)
            
            conn.commit()
            
    def store(self, content: Any, item_type: str, source: str, 
              confidence: float = 1.0, metadata: Dict[str, Any] = None) -> int:
        """Store a memory item."""
        timestamp = time.time()
        content_json = json.dumps(content, default=str)
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory (content, item_type, timestamp, source, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (content_json, item_type, timestamp, source, confidence, metadata_json))
            
            memory_id = cursor.lastrowid
            
            # Store version
            cursor.execute("""
                INSERT INTO versions (memory_id, version, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (memory_id, 1, content_json, timestamp))
            
            conn.commit()
            return memory_id
            
    def retrieve(self, memory_id: int) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content, item_type, timestamp, source, confidence, metadata
                FROM memory WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                return MemoryItem(
                    id_=row[0],
                    content=json.loads(row[1]),
                    item_type=row[2],
                    timestamp=row[3],
                    source=row[4],
                    confidence=row[5],
                    metadata=json.loads(row[6])
                )
            return None
            
    def search(self, item_type: str = None, source: str = None, 
               min_confidence: float = 0.0, limit: int = 100) -> List[MemoryItem]:
        """Search memory items with filters."""
        query = "SELECT id, content, item_type, timestamp, source, confidence, metadata FROM memory WHERE 1=1"
        params = []
        
        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)
            
        if source:
            query += " AND source = ?"
            params.append(source)
            
        if min_confidence > 0:
            query += " AND confidence >= ?"
            params.append(min_confidence)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            items = []
            for row in cursor.fetchall():
                items.append(MemoryItem(
                    id_=row[0],
                    content=json.loads(row[1]),
                    item_type=row[2],
                    timestamp=row[3],
                    source=row[4],
                    confidence=row[5],
                    metadata=json.loads(row[6])
                ))
            return items
            
    def update(self, memory_id: int, content: Any = None, 
               confidence: float = None, metadata: Dict[str, Any] = None) -> bool:
        """Update a memory item."""
        updates = []
        params = []
        
        if content is not None:
            updates.append("content = ?")
            params.append(json.dumps(content, default=str))
            
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
            
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
            
        if not updates:
            return False
            
        params.append(memory_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE memory SET {', '.join(updates)} WHERE id = ?
            """, params)
            
            # Store new version
            if content is not None:
                cursor.execute("""
                    SELECT MAX(version) FROM versions WHERE memory_id = ?
                """, (memory_id,))
                max_version = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    INSERT INTO versions (memory_id, version, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (memory_id, max_version + 1, json.dumps(content, default=str), time.time()))
            
            conn.commit()
            return cursor.rowcount > 0
            
    def delete(self, memory_id: int) -> bool:
        """Delete a memory item."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
            cursor.execute("DELETE FROM versions WHERE memory_id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
            
    def forget(self, pattern: str = None, item_type: str = None, 
               source: str = None, older_than: float = None) -> int:
        """Forget memory items matching criteria."""
        query = "DELETE FROM memory WHERE 1=1"
        params = []
        
        if pattern:
            query += " AND content LIKE ?"
            params.append(f"%{pattern}%")
            
        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)
            
        if source:
            query += " AND source = ?"
            params.append(source)
            
        if older_than:
            query += " AND timestamp < ?"
            params.append(older_than)
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            
            # Also delete versions
            cursor.execute("""
                DELETE FROM versions WHERE memory_id IN (
                    SELECT id FROM memory WHERE 1=0
                )
            """)
            
            conn.commit()
            return deleted_count
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total items
            cursor.execute("SELECT COUNT(*) FROM memory")
            total_items = cursor.fetchone()[0]
            
            # Items by type
            cursor.execute("""
                SELECT item_type, COUNT(*) FROM memory GROUP BY item_type
            """)
            by_type = dict(cursor.fetchall())
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM memory WHERE timestamp > ?
            """, (time.time() - 86400,))  # Last 24 hours
            recent_items = cursor.fetchone()[0]
            
            return {
                'total_items': total_items,
                'by_type': by_type,
                'recent_items': recent_items,
                'database_path': self.db_path
            }
            
    def export_memories(self, file_path: str) -> bool:
        """Export all memories to a JSON file."""
        try:
            memories = self.search(limit=10000)  # Get all memories
            data = {
                'export_timestamp': time.time(),
                'total_memories': len(memories),
                'memories': [item.to_dict() for item in memories]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
            
    def import_memories(self, file_path: str) -> int:
        """Import memories from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            imported_count = 0
            for memory_data in data.get('memories', []):
                # Create new memory item
                item = MemoryItem.from_dict(memory_data)
                self.store(
                    content=item.content,
                    item_type=item.item_type,
                    source=item.source,
                    confidence=item.confidence,
                    metadata=item.metadata
                )
                imported_count += 1
                
            return imported_count
        except Exception:
            return 0
