"""
Semantic Indexer for Egdol Memory.
Provides fast retrieval of concepts and relationships.
"""

import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from .store import MemoryStore, MemoryItem


class SemanticIndexer:
    """Indexes memory for semantic retrieval."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.concept_index: Dict[str, Set[int]] = defaultdict(set)
        self.relationship_index: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.predicate_index: Dict[str, Set[int]] = defaultdict(set)
        self._build_indexes()
        
    def _build_indexes(self):
        """Build semantic indexes from existing memories."""
        memories = self.memory_store.search(limit=10000)
        for memory in memories:
            self._index_memory(memory)
            
    def _index_memory(self, memory: MemoryItem):
        """Index a single memory item."""
        content = memory.content
        
        # Extract concepts from content
        concepts = self._extract_concepts(content)
        for concept in concepts:
            self.concept_index[concept].add(memory.id)
            
        # Extract relationships
        relationships = self._extract_relationships(content)
        for rel in relationships:
            self.relationship_index[rel].add(memory.id)
            
        # Extract predicates
        predicates = self._extract_predicates(content)
        for pred in predicates:
            self.predicate_index[pred].add(memory.id)
            
    def _extract_concepts(self, content: Any) -> Set[str]:
        """Extract concepts from content."""
        concepts = set()
        
        if isinstance(content, dict):
            # Handle serialized terms
            if content.get('type') == 'Term':
                concepts.add(content['name'])
                for arg in content.get('args', []):
                    concepts.update(self._extract_concepts(arg))
            elif content.get('type') == 'Constant':
                concepts.add(content['value'])
            elif content.get('type') == 'Variable':
                concepts.add(content['name'])
        elif isinstance(content, str):
            # Extract words from text
            words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
            concepts.update(words)
            
        return concepts
        
    def _extract_relationships(self, content: Any) -> Set[Tuple[str, str]]:
        """Extract relationships from content."""
        relationships = set()
        
        if isinstance(content, dict):
            if content.get('type') == 'Term':
                name = content['name']
                args = content.get('args', [])
                
                # Extract relationships between arguments
                for i, arg1 in enumerate(args):
                    for j, arg2 in enumerate(args[i+1:], i+1):
                        if isinstance(arg1, dict) and isinstance(arg2, dict):
                            if arg1.get('type') == 'Constant' and arg2.get('type') == 'Constant':
                                relationships.add((arg1['value'], arg2['value']))
                                relationships.add((arg2['value'], arg1['value']))
                                
        return relationships
        
    def _extract_predicates(self, content: Any) -> Set[str]:
        """Extract predicates from content."""
        predicates = set()
        
        if isinstance(content, dict):
            if content.get('type') == 'Term':
                predicates.add(content['name'])
            elif content.get('type') == 'Rule':
                # Extract predicates from rule head and body
                head = content.get('head', {})
                if head.get('type') == 'Term':
                    predicates.add(head['name'])
                    
                for term in content.get('body', []):
                    if term.get('type') == 'Term':
                        predicates.add(term['name'])
                        
        return predicates
        
    def search_by_concept(self, concept: str) -> List[MemoryItem]:
        """Search memories by concept."""
        memory_ids = self.concept_index.get(concept.lower(), set())
        memories = []
        for memory_id in memory_ids:
            memory = self.memory_store.retrieve(memory_id)
            if memory:
                memories.append(memory)
        return memories
        
    def search_by_relationship(self, subject: str, object_: str) -> List[MemoryItem]:
        """Search memories by relationship."""
        memory_ids = self.relationship_index.get((subject.lower(), object_.lower()), set())
        memories = []
        for memory_id in memory_ids:
            memory = self.memory_store.retrieve(memory_id)
            if memory:
                memories.append(memory)
        return memories
        
    def search_by_predicate(self, predicate: str) -> List[MemoryItem]:
        """Search memories by predicate."""
        memory_ids = self.predicate_index.get(predicate.lower(), set())
        memories = []
        for memory_id in memory_ids:
            memory = self.memory_store.retrieve(memory_id)
            if memory:
                memories.append(memory)
        return memories
        
    def find_related_concepts(self, concept: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Find concepts related to the given concept."""
        concept = concept.lower()
        related = defaultdict(int)
        
        # Find memories containing this concept
        concept_memories = self.search_by_concept(concept)
        
        for memory in concept_memories:
            # Extract other concepts from these memories
            other_concepts = self._extract_concepts(memory.content)
            for other_concept in other_concepts:
                if other_concept != concept:
                    related[other_concept] += 1
                    
        # Sort by frequency and return top results
        sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
        return sorted_related[:limit]
        
    def find_similar_memories(self, memory: MemoryItem, limit: int = 5) -> List[Tuple[MemoryItem, float]]:
        """Find memories similar to the given memory."""
        # Extract concepts from the memory
        concepts = self._extract_concepts(memory.content)
        
        # Find memories that share concepts
        candidate_memories = set()
        for concept in concepts:
            concept_memories = self.search_by_concept(concept)
            for mem in concept_memories:
                if mem.id != memory.id:
                    candidate_memories.add(mem)
                    
        # Calculate similarity scores
        similarities = []
        for candidate in candidate_memories:
            candidate_concepts = self._extract_concepts(candidate.content)
            
            # Jaccard similarity
            intersection = len(concepts.intersection(candidate_concepts))
            union = len(concepts.union(candidate_concepts))
            similarity = intersection / union if union > 0 else 0
            
            similarities.append((candidate, similarity))
            
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
        
    def get_concept_network(self, max_concepts: int = 50) -> Dict[str, List[str]]:
        """Get a concept network for visualization."""
        # Get all concepts
        all_concepts = set()
        for concept_set in self.concept_index.values():
            all_concepts.update(concept_set)
            
        # Limit to most frequent concepts
        concept_counts = {concept: len(memory_ids) for concept, memory_ids in self.concept_index.items()}
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
        
        # Build network
        network = {}
        for concept, _ in top_concepts:
            related = self.find_related_concepts(concept, limit=10)
            network[concept] = [rel_concept for rel_concept, _ in related]
            
        return network
        
    def update_index(self, memory: MemoryItem):
        """Update indexes when a new memory is added."""
        self._index_memory(memory)
        
    def remove_from_index(self, memory_id: int):
        """Remove memory from indexes."""
        # Remove from all indexes
        for concept, memory_ids in self.concept_index.items():
            memory_ids.discard(memory_id)
            
        for relationship, memory_ids in self.relationship_index.items():
            memory_ids.discard(memory_id)
            
        for predicate, memory_ids in self.predicate_index.items():
            memory_ids.discard(memory_id)
            
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_concepts': len(self.concept_index),
            'total_relationships': len(self.relationship_index),
            'total_predicates': len(self.predicate_index),
            'concept_counts': {concept: len(memory_ids) for concept, memory_ids in self.concept_index.items()},
            'predicate_counts': {predicate: len(memory_ids) for predicate, memory_ids in self.predicate_index.items()}
        }
