"""
Knowledge Graph Core for OmniMind
Implements bidirectional nodes and edges for semantic reasoning.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    ENTITY = auto()
    CONCEPT = auto()
    RELATIONSHIP = auto()
    ATTRIBUTE = auto()
    RULE = auto()


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""
    IS_A = auto()
    HAS_A = auto()
    PART_OF = auto()
    RELATED_TO = auto()
    CAUSES = auto()
    IMPLIES = auto()
    SIMILAR_TO = auto()
    OPPOSITE_OF = auto()


@dataclass
class Node:
    """A node in the knowledge graph."""
    id: str
    name: str
    node_type: NodeType
    properties: Dict[str, Any]
    confidence: float = 1.0
    created_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()
            
    def update(self, properties: Dict[str, Any] = None, confidence: float = None):
        """Update node properties."""
        if properties:
            self.properties.update(properties)
        if confidence is not None:
            self.confidence = confidence
        self.updated_at = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'node_type': self.node_type.name,
            'properties': self.properties,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            node_type=NodeType[data['node_type']],
            properties=data['properties'],
            confidence=data['confidence'],
            created_at=data['created_at'],
            updated_at=data['updated_at']
        )


@dataclass
class Edge:
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any]
    confidence: float = 1.0
    created_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()
            
    def update(self, properties: Dict[str, Any] = None, confidence: float = None):
        """Update edge properties."""
        if properties:
            self.properties.update(properties)
        if confidence is not None:
            self.confidence = confidence
        self.updated_at = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type.name,
            'properties': self.properties,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create edge from dictionary."""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            edge_type=EdgeType[data['edge_type']],
            properties=data['properties'],
            confidence=data['confidence'],
            created_at=data['created_at'],
            updated_at=data['updated_at']
        )


class KnowledgeGraph:
    """Core knowledge graph implementation."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.node_index: Dict[str, Set[str]] = {}  # name -> node_ids
        self.edge_index: Dict[str, Set[str]] = {}  # (source_id, target_id) -> edge_ids
        
    def add_node(self, name: str, node_type: NodeType, 
                 properties: Dict[str, Any] = None, confidence: float = 1.0) -> str:
        """Add a node to the graph."""
        node_id = str(uuid.uuid4())
        
        node = Node(
            id=node_id,
            name=name,
            node_type=node_type,
            properties=properties or {},
            confidence=confidence
        )
        
        self.nodes[node_id] = node
        
        # Update index
        if name not in self.node_index:
            self.node_index[name] = set()
        self.node_index[name].add(node_id)
        
        return node_id
        
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType,
                 properties: Dict[str, Any] = None, confidence: float = 1.0) -> str:
        """Add an edge to the graph."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target node not found")
            
        edge_id = str(uuid.uuid4())
        
        edge = Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=properties or {},
            confidence=confidence
        )
        
        self.edges[edge_id] = edge
        
        # Update index
        edge_key = (source_id, target_id)
        if edge_key not in self.edge_index:
            self.edge_index[edge_key] = set()
        self.edge_index[edge_key].add(edge_id)
        
        return edge_id
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
        
    def get_node_by_name(self, name: str) -> List[Node]:
        """Get nodes by name."""
        node_ids = self.node_index.get(name, set())
        return [self.nodes[node_id] for node_id in node_ids]
        
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
        
    def get_edges(self, source_id: str = None, target_id: str = None, 
                  edge_type: EdgeType = None) -> List[Edge]:
        """Get edges with optional filters."""
        edges = list(self.edges.values())
        
        if source_id:
            edges = [e for e in edges if e.source_id == source_id]
        if target_id:
            edges = [e for e in edges if e.target_id == target_id]
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
            
        return edges
        
    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> List[Node]:
        """Get neighboring nodes."""
        neighbors = []
        
        # Outgoing edges
        outgoing_edges = self.get_edges(source_id=node_id, edge_type=edge_type)
        for edge in outgoing_edges:
            neighbor = self.get_node(edge.target_id)
            if neighbor:
                neighbors.append(neighbor)
                
        # Incoming edges
        incoming_edges = self.get_edges(target_id=node_id, edge_type=edge_type)
        for edge in incoming_edges:
            neighbor = self.get_node(edge.source_id)
            if neighbor:
                neighbors.append(neighbor)
                
        return neighbors
        
    def find_path(self, source_id: str, target_id: str, 
                  max_depth: int = 5) -> List[List[str]]:
        """Find paths between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
            
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth > max_depth:
                return
                
            if current_id == target_id:
                paths.append(path + [current_id])
                return
                
            if current_id in visited:
                return
                
            visited.add(current_id)
            
            # Get neighbors
            neighbors = self.get_neighbors(current_id)
            for neighbor in neighbors:
                dfs(neighbor.id, path + [current_id], depth + 1)
                
            visited.remove(current_id)
            
        dfs(source_id, [], 0)
        return paths
        
    def get_subgraph(self, node_ids: List[str]) -> 'KnowledgeGraph':
        """Get a subgraph containing only the specified nodes."""
        subgraph = KnowledgeGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subgraph.nodes[node_id] = node
                subgraph.node_index[node.name] = {node_id}
                
        # Add edges between nodes in subgraph
        for edge in self.edges.values():
            if (edge.source_id in node_ids and 
                edge.target_id in node_ids):
                subgraph.edges[edge.id] = edge
                edge_key = (edge.source_id, edge.target_id)
                if edge_key not in subgraph.edge_index:
                    subgraph.edge_index[edge_key] = set()
                subgraph.edge_index[edge_key].add(edge.id)
                
        return subgraph
        
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        
        # Remove all edges connected to this node
        edges_to_remove = []
        for edge in self.edges.values():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge.id)
                
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
            
        # Remove from index
        if node.name in self.node_index:
            self.node_index[node.name].discard(node_id)
            if not self.node_index[node.name]:
                del self.node_index[node.name]
                
        # Remove node
        del self.nodes[node_id]
        return True
        
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        if edge_id not in self.edges:
            return False
            
        edge = self.edges[edge_id]
        
        # Remove from index
        edge_key = (edge.source_id, edge.target_id)
        if edge_key in self.edge_index:
            self.edge_index[edge_key].discard(edge_id)
            if not self.edge_index[edge_key]:
                del self.edge_index[edge_key]
                
        # Remove edge
        del self.edges[edge_id]
        return True
        
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_types = {}
        edge_types = {}
        
        for node in self.nodes.values():
            node_type = node.node_type.name
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        for edge in self.edges.values():
            edge_type = edge.edge_type.name
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'avg_confidence': sum(n.confidence for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0
        }
        
    def export_graph(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'exported_at': time.time()
        }
        
    def import_graph(self, data: Dict[str, Any]) -> bool:
        """Import graph from dictionary."""
        try:
            # Clear existing graph
            self.nodes.clear()
            self.edges.clear()
            self.node_index.clear()
            self.edge_index.clear()
            
            # Import nodes
            for node_data in data.get('nodes', []):
                node = Node.from_dict(node_data)
                self.nodes[node.id] = node
                
                if node.name not in self.node_index:
                    self.node_index[node.name] = set()
                self.node_index[node.name].add(node.id)
                
            # Import edges
            for edge_data in data.get('edges', []):
                edge = Edge.from_dict(edge_data)
                self.edges[edge.id] = edge
                
                edge_key = (edge.source_id, edge.target_id)
                if edge_key not in self.edge_index:
                    self.edge_index[edge_key] = set()
                self.edge_index[edge_key].add(edge.id)
                
            return True
            
        except Exception:
            return False
