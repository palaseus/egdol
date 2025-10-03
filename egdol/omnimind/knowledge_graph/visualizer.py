"""
Graph Visualizer for OmniMind Knowledge Graph
Provides visualization capabilities for the knowledge graph.
"""

import json
from typing import Dict, Any, List, Optional
from .graph import KnowledgeGraph, Node, Edge, NodeType, EdgeType


class GraphVisualizer:
    """Provides visualization capabilities for the knowledge graph."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        
    def export_to_d3(self) -> Dict[str, Any]:
        """Export graph data in D3.js format."""
        nodes = []
        links = []
        
        # Convert nodes
        for node in self.graph.nodes.values():
            d3_node = {
                'id': node.id,
                'name': node.name,
                'type': node.node_type.name,
                'properties': node.properties,
                'confidence': node.confidence,
                'group': self._get_node_group(node)
            }
            nodes.append(d3_node)
            
        # Convert edges
        for edge in self.graph.edges.values():
            d3_link = {
                'source': edge.source_id,
                'target': edge.target_id,
                'type': edge.edge_type.name,
                'properties': edge.properties,
                'confidence': edge.confidence,
                'strength': edge.confidence
            }
            links.append(d3_link)
            
        return {
            'nodes': nodes,
            'links': links,
            'metadata': {
                'total_nodes': len(nodes),
                'total_links': len(links),
                'exported_at': self._get_timestamp()
            }
        }
        
    def export_to_cytoscape(self) -> Dict[str, Any]:
        """Export graph data in Cytoscape.js format."""
        elements = []
        
        # Convert nodes
        for node in self.graph.nodes.values():
            cytoscape_node = {
                'data': {
                    'id': node.id,
                    'label': node.name,
                    'type': node.node_type.name,
                    'properties': node.properties,
                    'confidence': node.confidence
                },
                'classes': self._get_node_classes(node)
            }
            elements.append(cytoscape_node)
            
        # Convert edges
        for edge in self.graph.edges.values():
            cytoscape_edge = {
                'data': {
                    'id': edge.id,
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'label': edge.edge_type.name,
                    'type': edge.edge_type.name,
                    'confidence': edge.confidence
                },
                'classes': self._get_edge_classes(edge)
            }
            elements.append(cytoscape_edge)
            
        return {
            'elements': elements,
            'metadata': {
                'total_elements': len(elements),
                'exported_at': self._get_timestamp()
            }
        }
        
    def export_to_graphml(self) -> str:
        """Export graph data in GraphML format."""
        graphml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        
        # Define attributes
        graphml += '  <key id="type" for="node" attr.name="type" attr.type="string"/>\n'
        graphml += '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>\n'
        graphml += '  <key id="properties" for="node" attr.name="properties" attr.type="string"/>\n'
        graphml += '  <key id="edge_type" for="edge" attr.name="type" attr.type="string"/>\n'
        graphml += '  <key id="edge_confidence" for="edge" attr.name="confidence" attr.type="double"/>\n'
        
        # Start graph
        graphml += '  <graph id="knowledge_graph" edgedefault="directed">\n'
        
        # Add nodes
        for node in self.graph.nodes.values():
            graphml += f'    <node id="{node.id}">\n'
            graphml += f'      <data key="type">{node.node_type.name}</data>\n'
            graphml += f'      <data key="confidence">{node.confidence}</data>\n'
            graphml += f'      <data key="properties">{json.dumps(node.properties)}</data>\n'
            graphml += f'    </node>\n'
            
        # Add edges
        for edge in self.graph.edges.values():
            graphml += f'    <edge id="{edge.id}" source="{edge.source_id}" target="{edge.target_id}">\n'
            graphml += f'      <data key="type">{edge.edge_type.name}</data>\n'
            graphml += f'      <data key="confidence">{edge.confidence}</data>\n'
            graphml += f'    </edge>\n'
            
        # Close graph
        graphml += '  </graph>\n'
        graphml += '</graphml>\n'
        
        return graphml
        
    def export_to_dot(self) -> str:
        """Export graph data in DOT format for Graphviz."""
        dot = 'digraph KnowledgeGraph {\n'
        dot += '  rankdir=LR;\n'
        dot += '  node [shape=ellipse, style=filled];\n'
        dot += '  edge [fontsize=10];\n\n'
        
        # Add nodes
        for node in self.graph.nodes.values():
            color = self._get_node_color(node)
            label = f'"{node.name}"'
            dot += f'  {node.id} [label={label}, fillcolor="{color}"];\n'
            
        # Add edges
        for edge in self.graph.edges.values():
            source_node = self.graph.get_node(edge.source_id)
            target_node = self.graph.get_node(edge.target_id)
            
            if source_node and target_node:
                label = edge.edge_type.name.replace('_', ' ')
                dot += f'  {edge.source_id} -> {edge.target_id} [label="{label}"];\n'
                
        dot += '}\n'
        return dot
        
    def _get_node_group(self, node: Node) -> str:
        """Get group for D3 visualization."""
        if node.node_type == NodeType.ENTITY:
            return 'entity'
        elif node.node_type == NodeType.CONCEPT:
            return 'concept'
        elif node.node_type == NodeType.RELATIONSHIP:
            return 'relationship'
        elif node.node_type == NodeType.ATTRIBUTE:
            return 'attribute'
        elif node.node_type == NodeType.RULE:
            return 'rule'
        else:
            return 'other'
            
    def _get_node_classes(self, node: Node) -> List[str]:
        """Get CSS classes for Cytoscape visualization."""
        classes = [node.node_type.name.lower()]
        
        if node.confidence > 0.8:
            classes.append('high-confidence')
        elif node.confidence > 0.5:
            classes.append('medium-confidence')
        else:
            classes.append('low-confidence')
            
        return classes
        
    def _get_edge_classes(self, edge: Edge) -> List[str]:
        """Get CSS classes for Cytoscape visualization."""
        classes = [edge.edge_type.name.lower().replace('_', '-')]
        
        if edge.confidence > 0.8:
            classes.append('high-confidence')
        elif edge.confidence > 0.5:
            classes.append('medium-confidence')
        else:
            classes.append('low-confidence')
            
        return classes
        
    def _get_node_color(self, node: Node) -> str:
        """Get color for node visualization."""
        if node.node_type == NodeType.ENTITY:
            return '#FF6B6B'  # Red
        elif node.node_type == NodeType.CONCEPT:
            return '#4ECDC4'  # Teal
        elif node.node_type == NodeType.RELATIONSHIP:
            return '#45B7D1'  # Blue
        elif node.node_type == NodeType.ATTRIBUTE:
            return '#96CEB4'  # Green
        elif node.node_type == NodeType.RULE:
            return '#FFEAA7'  # Yellow
        else:
            return '#DDA0DD'  # Plum
            
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')
        
    def generate_html_visualization(self, output_file: str = None) -> str:
        """Generate HTML visualization of the graph."""
        html = '''
<!DOCTYPE html>
<html>
<head>
    <title>OmniMind Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .node {
            stroke: #fff;
            stroke-width: 2px;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .node-label {
            font-size: 12px;
            text-anchor: middle;
        }
        .link-label {
            font-size: 10px;
            text-anchor: middle;
        }
    </style>
</head>
<body>
    <div id="graph-container"></div>
    <script>
        // Graph data will be inserted here
        const graphData = ''' + json.dumps(self.export_to_d3()) + ''';
        
        const width = 800;
        const height = 600;
        
        const svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
            
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
            
        const link = svg.append("g")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link");
            
        const node = svg.append("g")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", 8)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
                
        const nodeLabels = svg.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.name);
            
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
                
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
                
            nodeLabels
                .attr("x", d => d.x)
                .attr("y", d => d.y + 4);
        });
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    </script>
</body>
</html>
        '''
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
                
        return html
