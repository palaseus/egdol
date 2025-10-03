"""
Graph Integrator for OmniMind Knowledge Graph
Integrates knowledge graph with OmniMind's NLU and DSL translator.
"""

from typing import Dict, Any, List, Optional, Tuple
from .graph import KnowledgeGraph, Node, Edge, NodeType, EdgeType
from .inference import GraphInference


class GraphIntegrator:
    """Integrates knowledge graph with OmniMind systems."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.inference = GraphInference(graph)
        
    def add_fact_from_dsl(self, dsl_fact: str) -> bool:
        """Add a fact from DSL to the knowledge graph."""
        try:
            # Parse DSL fact (simplified parsing)
            if ' is ' in dsl_fact.lower():
                parts = dsl_fact.split(' is ')
                if len(parts) == 2:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    
                    # Create nodes
                    subject_id = self._get_or_create_node(subject, NodeType.ENTITY)
                    predicate_id = self._get_or_create_node(predicate, NodeType.CONCEPT)
                    
                    # Create edge
                    self.graph.add_edge(subject_id, predicate_id, EdgeType.IS_A)
                    return True
                    
            elif ' has ' in dsl_fact.lower():
                parts = dsl_fact.split(' has ')
                if len(parts) == 2:
                    subject = parts[0].strip()
                    attribute = parts[1].strip()
                    
                    # Create nodes
                    subject_id = self._get_or_create_node(subject, NodeType.ENTITY)
                    attribute_id = self._get_or_create_node(attribute, NodeType.ATTRIBUTE)
                    
                    # Create edge
                    self.graph.add_edge(subject_id, attribute_id, EdgeType.HAS_A)
                    return True
                    
            elif ' implies ' in dsl_fact.lower():
                parts = dsl_fact.split(' implies ')
                if len(parts) == 2:
                    premise = parts[0].strip()
                    conclusion = parts[1].strip()
                    
                    # Create nodes
                    premise_id = self._get_or_create_node(premise, NodeType.CONCEPT)
                    conclusion_id = self._get_or_create_node(conclusion, NodeType.CONCEPT)
                    
                    # Create edge
                    self.graph.add_edge(premise_id, conclusion_id, EdgeType.IMPLIES)
                    return True
                    
        except Exception:
            return False
            
        return False
        
    def add_rule_from_dsl(self, dsl_rule: str) -> bool:
        """Add a rule from DSL to the knowledge graph."""
        try:
            # Parse DSL rule (simplified parsing)
            if ' if ' in dsl_rule.lower() and ' then ' in dsl_rule.lower():
                parts = dsl_rule.split(' if ')
                if len(parts) == 2:
                    conclusion = parts[0].strip()
                    condition_parts = parts[1].split(' then ')
                    if len(condition_parts) == 2:
                        condition = condition_parts[0].strip()
                        conclusion = condition_parts[1].strip()
                        
                        # Create nodes
                        condition_id = self._get_or_create_node(condition, NodeType.CONCEPT)
                        conclusion_id = self._get_or_create_node(conclusion, NodeType.CONCEPT)
                        
                        # Create edge
                        self.graph.add_edge(condition_id, conclusion_id, EdgeType.IMPLIES)
                        return True
                        
        except Exception:
            return False
            
        return False
        
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge graph with natural language."""
        results = []
        
        # Simple query patterns
        if query.lower().startswith('what is '):
            entity = query[8:].strip()
            results.extend(self._query_what_is(entity))
            
        elif query.lower().startswith('who is '):
            entity = query[7:].strip()
            results.extend(self._query_who_is(entity))
            
        elif query.lower().startswith('what does '):
            entity = query[10:].strip()
            results.extend(self._query_what_does(entity))
            
        elif ' related to ' in query.lower():
            parts = query.split(' related to ')
            if len(parts) == 2:
                entity1 = parts[0].strip()
                entity2 = parts[1].strip()
                results.extend(self._query_related_to(entity1, entity2))
                
        return results
        
    def _get_or_create_node(self, name: str, node_type: NodeType) -> str:
        """Get existing node or create new one."""
        # Check if node exists
        existing_nodes = self.graph.get_node_by_name(name)
        for node in existing_nodes:
            if node.node_type == node_type:
                return node.id
                
        # Create new node
        return self.graph.add_node(name, node_type)
        
    def _query_what_is(self, entity: str) -> List[Dict[str, Any]]:
        """Query what an entity is."""
        results = []
        
        # Find entity nodes
        entity_nodes = self.graph.get_node_by_name(entity)
        
        for entity_node in entity_nodes:
            # Get IS_A relationships
            is_a_edges = self.graph.get_edges(source_id=entity_node.id, edge_type=EdgeType.IS_A)
            
            for edge in is_a_edges:
                target_node = self.graph.get_node(edge.target_id)
                if target_node:
                    results.append({
                        'type': 'is_a_relationship',
                        'entity': entity,
                        'concept': target_node.name,
                        'confidence': edge.confidence,
                        'explanation': f"{entity} is a {target_node.name}"
                    })
                    
        return results
        
    def _query_who_is(self, entity: str) -> List[Dict[str, Any]]:
        """Query who an entity is."""
        return self._query_what_is(entity)  # Same logic
        
    def _query_what_does(self, entity: str) -> List[Dict[str, Any]]:
        """Query what an entity does."""
        results = []
        
        # Find entity nodes
        entity_nodes = self.graph.get_node_by_name(entity)
        
        for entity_node in entity_nodes:
            # Get all relationships
            outgoing_edges = self.graph.get_edges(source_id=entity_node.id)
            
            for edge in outgoing_edges:
                target_node = self.graph.get_node(edge.target_id)
                if target_node:
                    results.append({
                        'type': 'relationship',
                        'entity': entity,
                        'target': target_node.name,
                        'relationship': edge.edge_type.name,
                        'confidence': edge.confidence,
                        'explanation': f"{entity} {edge.edge_type.name} {target_node.name}"
                    })
                    
        return results
        
    def _query_related_to(self, entity1: str, entity2: str) -> List[Dict[str, Any]]:
        """Query how two entities are related."""
        results = []
        
        # Find entity nodes
        entity1_nodes = self.graph.get_node_by_name(entity1)
        entity2_nodes = self.graph.get_node_by_name(entity2)
        
        for entity1_node in entity1_nodes:
            for entity2_node in entity2_nodes:
                # Find paths between entities
                paths = self.graph.find_path(entity1_node.id, entity2_node.id, max_depth=3)
                
                for path in paths:
                    path_nodes = [self.graph.get_node(node_id) for node_id in path]
                    path_names = [node.name for node in path_nodes if node]
                    
                    results.append({
                        'type': 'path_relationship',
                        'entity1': entity1,
                        'entity2': entity2,
                        'path': path_names,
                        'path_length': len(path) - 1,
                        'explanation': ' → '.join(path_names)
                    })
                    
        return results
        
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary of the knowledge graph."""
        stats = self.graph.get_stats()
        
        # Get node type distribution
        node_types = {}
        for node in self.graph.nodes.values():
            node_type = node.node_type.name
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        # Get edge type distribution
        edge_types = {}
        for edge in self.graph.edges.values():
            edge_type = edge.edge_type.name
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
        # Get average confidence
        avg_confidence = sum(node.confidence for node in self.graph.nodes.values()) / len(self.graph.nodes) if self.graph.nodes else 0
        
        return {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'average_confidence': avg_confidence,
            'density': len(self.graph.edges) / (len(self.graph.nodes) * (len(self.graph.nodes) - 1)) if len(self.graph.nodes) > 1 else 0
        }
        
    def export_for_omnimind(self) -> Dict[str, Any]:
        """Export graph data for OmniMind integration."""
        return {
            'nodes': [node.to_dict() for node in self.graph.nodes.values()],
            'edges': [edge.to_dict() for edge in self.graph.edges.values()],
            'summary': self.get_graph_summary(),
            'exported_at': self._get_timestamp()
        }
        
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')
