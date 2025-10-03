"""
Graph Inference Engine for OmniMind Knowledge Graph
Provides multi-hop reasoning and inference capabilities.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Set
from .graph import KnowledgeGraph, Node, Edge, NodeType, EdgeType


class GraphInference:
    """Provides inference capabilities for the knowledge graph."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        
    def infer_relationships(self, source_node: str, target_node: str, 
                          max_depth: int = 3) -> List[Dict[str, Any]]:
        """Infer relationships between two nodes."""
        if source_node not in self.graph.nodes or target_node not in self.graph.nodes:
            return []
            
        # Find all paths between nodes
        paths = self.graph.find_path(source_node, target_node, max_depth)
        
        inferences = []
        for path in paths:
            inference = self._analyze_path(path)
            if inference:
                inferences.append(inference)
                
        return inferences
        
    def _analyze_path(self, path: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze a path to create an inference."""
        if len(path) < 2:
            return None
            
        # Get nodes and edges in the path
        nodes = [self.graph.get_node(node_id) for node_id in path]
        edges = []
        
        for i in range(len(path) - 1):
            edge_list = self.graph.get_edges(source_id=path[i], target_id=path[i+1])
            if edge_list:
                edges.append(edge_list[0])  # Take first edge
                
        if not edges:
            return None
            
        # Create inference
        inference = {
            'path': path,
            'nodes': [node.name for node in nodes],
            'edges': [edge.edge_type.name for edge in edges],
            'confidence': self._calculate_path_confidence(edges),
            'explanation': self._generate_explanation(nodes, edges)
        }
        
        return inference
        
    def _calculate_path_confidence(self, edges: List[Edge]) -> float:
        """Calculate confidence for a path."""
        if not edges:
            return 0.0
            
        # Average confidence of all edges
        total_confidence = sum(edge.confidence for edge in edges)
        return total_confidence / len(edges)
        
    def _generate_explanation(self, nodes: List[Node], edges: List[Edge]) -> str:
        """Generate natural language explanation of the path."""
        if len(nodes) < 2 or len(edges) < 1:
            return "No valid path found"
            
        explanation_parts = []
        
        for i, edge in enumerate(edges):
            source_node = nodes[i]
            target_node = nodes[i + 1]
            
            if edge.edge_type == EdgeType.IS_A:
                explanation_parts.append(f"{source_node.name} is a {target_node.name}")
            elif edge.edge_type == EdgeType.HAS_A:
                explanation_parts.append(f"{source_node.name} has a {target_node.name}")
            elif edge.edge_type == EdgeType.PART_OF:
                explanation_parts.append(f"{source_node.name} is part of {target_node.name}")
            elif edge.edge_type == EdgeType.IMPLIES:
                explanation_parts.append(f"{source_node.name} implies {target_node.name}")
            elif edge.edge_type == EdgeType.CAUSES:
                explanation_parts.append(f"{source_node.name} causes {target_node.name}")
            else:
                explanation_parts.append(f"{source_node.name} is related to {target_node.name}")
                
        return " → ".join(explanation_parts)
        
    def find_implications(self, node_id: str, edge_type: EdgeType = EdgeType.IMPLIES) -> List[Dict[str, Any]]:
        """Find all implications of a node."""
        implications = []
        
        # Get direct implications
        direct_edges = self.graph.get_edges(source_id=node_id, edge_type=edge_type)
        for edge in direct_edges:
            target_node = self.graph.get_node(edge.target_id)
            if target_node:
                implications.append({
                    'type': 'direct',
                    'target': target_node.name,
                    'confidence': edge.confidence,
                    'explanation': f"Direct implication: {self.graph.get_node(node_id).name} implies {target_node.name}"
                })
                
        # Get transitive implications
        transitive_implications = self._find_transitive_implications(node_id, edge_type)
        implications.extend(transitive_implications)
        
        return implications
        
    def _find_transitive_implications(self, node_id: str, edge_type: EdgeType, 
                                    visited: Set[str] = None) -> List[Dict[str, Any]]:
        """Find transitive implications."""
        if visited is None:
            visited = set()
            
        if node_id in visited:
            return []
            
        visited.add(node_id)
        implications = []
        
        # Get direct edges
        direct_edges = self.graph.get_edges(source_id=node_id, edge_type=edge_type)
        
        for edge in direct_edges:
            target_node = self.graph.get_node(edge.target_id)
            if target_node:
                # Add direct implication
                implications.append({
                    'type': 'transitive',
                    'target': target_node.name,
                    'confidence': edge.confidence,
                    'explanation': f"Transitive implication through {target_node.name}"
                })
                
                # Recursively find implications of target
                recursive_implications = self._find_transitive_implications(
                    edge.target_id, edge_type, visited.copy()
                )
                implications.extend(recursive_implications)
                
        return implications
        
    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Find potential contradictions in the graph."""
        contradictions = []
        
        # Look for opposite relationships
        opposite_edges = self.graph.get_edges(edge_type=EdgeType.OPPOSITE_OF)
        
        for edge in opposite_edges:
            source_node = self.graph.get_node(edge.source_id)
            target_node = self.graph.get_node(edge.target_id)
            
            if source_node and target_node:
                # Check if both nodes have the same relationships
                source_edges = self.graph.get_edges(source_id=edge.source_id)
                target_edges = self.graph.get_edges(source_id=edge.target_id)
                
                for source_edge in source_edges:
                    for target_edge in target_edges:
                        if (source_edge.target_id == target_edge.target_id and 
                            source_edge.edge_type == target_edge.edge_type):
                            contradictions.append({
                                'type': 'opposite_relationship',
                                'source': source_node.name,
                                'target': target_node.name,
                                'conflict': f"{source_node.name} and {target_node.name} both have same relationship",
                                'confidence': min(source_edge.confidence, target_edge.confidence)
                            })
                            
        return contradictions
        
    def find_missing_links(self, node_id: str) -> List[Dict[str, Any]]:
        """Find potentially missing links for a node."""
        missing_links = []
        
        node = self.graph.get_node(node_id)
        if not node:
            return missing_links
            
        # Get neighbors
        neighbors = self.graph.get_neighbors(node_id)
        
        # Look for transitive relationships that might be missing
        for neighbor in neighbors:
            neighbor_edges = self.graph.get_edges(source_id=neighbor.id)
            
            for edge in neighbor_edges:
                target_node = self.graph.get_node(edge.target_id)
                if target_node and target_node.id != node_id:
                    # Check if direct relationship exists
                    direct_edges = self.graph.get_edges(source_id=node_id, target_id=target_node.id)
                    if not direct_edges:
                        missing_links.append({
                            'type': 'missing_direct_link',
                            'source': node.name,
                            'target': target_node.name,
                            'suggested_edge_type': edge.edge_type.name,
                            'confidence': edge.confidence * 0.8,  # Lower confidence for suggestion
                            'explanation': f"Missing direct link: {node.name} → {target_node.name}"
                        })
                        
        return missing_links
        
    def get_node_centrality(self, node_id: str) -> Dict[str, Any]:
        """Calculate centrality measures for a node."""
        if node_id not in self.graph.nodes:
            return {}
            
        # Degree centrality (number of connections)
        neighbors = self.graph.get_neighbors(node_id)
        degree_centrality = len(neighbors)
        
        # Betweenness centrality (how often node appears in shortest paths)
        betweenness = self._calculate_betweenness(node_id)
        
        # Closeness centrality (average distance to all other nodes)
        closeness = self._calculate_closeness(node_id)
        
        return {
            'node_id': node_id,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness,
            'closeness_centrality': closeness,
            'neighbor_count': degree_centrality
        }
        
    def _calculate_betweenness(self, node_id: str) -> float:
        """Calculate betweenness centrality for a node."""
        # Simplified betweenness calculation
        total_paths = 0
        paths_through_node = 0
        
        # Get all pairs of nodes
        all_nodes = list(self.graph.nodes.keys())
        
        for source in all_nodes:
            for target in all_nodes:
                if source != target and source != node_id and target != node_id:
                    paths = self.graph.find_path(source, target, max_depth=3)
                    total_paths += len(paths)
                    
                    for path in paths:
                        if node_id in path[1:-1]:  # Node is in the middle of the path
                            paths_through_node += 1
                            
        return paths_through_node / total_paths if total_paths > 0 else 0.0
        
    def _calculate_closeness(self, node_id: str) -> float:
        """Calculate closeness centrality for a node."""
        # Simplified closeness calculation
        total_distance = 0
        reachable_nodes = 0
        
        for other_node_id in self.graph.nodes:
            if other_node_id != node_id:
                paths = self.graph.find_path(node_id, other_node_id, max_depth=3)
                if paths:
                    min_distance = min(len(path) - 1 for path in paths)
                    total_distance += min_distance
                    reachable_nodes += 1
                    
        return reachable_nodes / total_distance if total_distance > 0 else 0.0
        
    def suggest_inferences(self, node_id: str) -> List[Dict[str, Any]]:
        """Suggest potential inferences for a node."""
        suggestions = []
        
        node = self.graph.get_node(node_id)
        if not node:
            return suggestions
            
        # Get all relationships
        outgoing_edges = self.graph.get_edges(source_id=node_id)
        incoming_edges = self.graph.get_edges(target_id=node_id)
        
        # Suggest transitive relationships
        for edge in outgoing_edges:
            target_node = self.graph.get_node(edge.target_id)
            if target_node:
                # Get relationships of target node
                target_edges = self.graph.get_edges(source_id=edge.target_id)
                
                for target_edge in target_edges:
                    final_node = self.graph.get_node(target_edge.target_id)
                    if final_node and final_node.id != node_id:
                        # Check if direct relationship exists
                        direct_edges = self.graph.get_edges(source_id=node_id, target_id=final_node.id)
                        if not direct_edges:
                            suggestions.append({
                                'type': 'transitive_inference',
                                'source': node.name,
                                'intermediate': target_node.name,
                                'target': final_node.name,
                                'suggested_edge_type': target_edge.edge_type.name,
                                'confidence': edge.confidence * target_edge.confidence * 0.8,
                                'explanation': f"If {node.name} → {target_node.name} and {target_node.name} → {final_node.name}, then {node.name} → {final_node.name}"
                            })
                            
        return suggestions
