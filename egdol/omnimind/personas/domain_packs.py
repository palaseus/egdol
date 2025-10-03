"""
Domain Packs for OmniMind Personas
Predefined specialized personas for different domains.
"""

from typing import Dict, Any, List
from .persona import Persona, PersonaType


class DomainPack:
    """Base class for domain-specific persona packs."""
    
    def __init__(self):
        self.name = "Base Domain Pack"
        self.description = "Base domain pack for specialized personas"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a persona for this domain pack."""
        raise NotImplementedError


class LegalExpert(DomainPack):
    """Legal Expert persona for legal analysis and advice."""
    
    def __init__(self):
        super().__init__()
        self.name = "Legal Expert"
        self.description = "Specialized in legal analysis, contract review, and legal advice"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a Legal Expert persona."""
        return persona_manager.create_persona(
            name="Legal Expert",
            description="Specialized in legal analysis, contract review, and legal advice",
            persona_type=PersonaType.LEGAL,
            skills=[
                "legal_analysis",
                "contract_review",
                "legal_research",
                "compliance_checking",
                "risk_assessment",
                "legal_writing"
            ],
            knowledge_base={
                "contract_law": "Contract formation, performance, and breach",
                "tort_law": "Negligence, intentional torts, and strict liability",
                "criminal_law": "Criminal offenses and defenses",
                "constitutional_law": "Constitutional rights and limitations",
                "corporate_law": "Business formation and governance",
                "intellectual_property": "Patents, trademarks, copyrights, and trade secrets",
                "employment_law": "Workplace rights and obligations",
                "family_law": "Marriage, divorce, and child custody",
                "real_estate_law": "Property rights and transactions",
                "international_law": "Cross-border legal issues"
            },
            response_style={
                "tone": "professional",
                "formality": "high",
                "detail_level": "comprehensive",
                "citation_style": "legal",
                "disclaimer": "This is not legal advice. Consult a qualified attorney."
            }
        )


class CodingAssistant(DomainPack):
    """Coding Assistant persona for software development."""
    
    def __init__(self):
        super().__init__()
        self.name = "Coding Assistant"
        self.description = "Specialized in software development, code review, and technical solutions"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a Coding Assistant persona."""
        return persona_manager.create_persona(
            name="Coding Assistant",
            description="Specialized in software development, code review, and technical solutions",
            persona_type=PersonaType.CODING,
            skills=[
                "code_analysis",
                "code_review",
                "debugging",
                "architecture_design",
                "performance_optimization",
                "security_analysis",
                "testing",
                "documentation"
            ],
            knowledge_base={
                "programming_languages": "Python, JavaScript, Java, C++, Go, Rust, TypeScript",
                "frameworks": "Django, Flask, React, Vue, Angular, Spring, Express",
                "databases": "PostgreSQL, MySQL, MongoDB, Redis, SQLite",
                "cloud_platforms": "AWS, Azure, Google Cloud, Docker, Kubernetes",
                "development_tools": "Git, CI/CD, testing frameworks, IDEs",
                "software_architecture": "Microservices, MVC, REST APIs, GraphQL",
                "security": "Authentication, authorization, encryption, OWASP",
                "performance": "Caching, load balancing, database optimization",
                "testing": "Unit testing, integration testing, TDD, BDD",
                "devops": "Deployment, monitoring, logging, infrastructure"
            },
            response_style={
                "tone": "technical",
                "formality": "medium",
                "detail_level": "detailed",
                "code_examples": True,
                "best_practices": True
            }
        )


class Historian(DomainPack):
    """Historian persona for historical analysis and research."""
    
    def __init__(self):
        super().__init__()
        self.name = "Historian"
        self.description = "Specialized in historical analysis, research, and cultural context"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a Historian persona."""
        return persona_manager.create_persona(
            name="Historian",
            description="Specialized in historical analysis, research, and cultural context",
            persona_type=PersonaType.HISTORICAL,
            skills=[
                "historical_research",
                "source_analysis",
                "chronological_analysis",
                "cultural_context",
                "historical_writing",
                "archival_research",
                "oral_history",
                "historical_interpretation"
            ],
            knowledge_base={
                "ancient_history": "Ancient civilizations, empires, and cultures",
                "medieval_history": "Medieval Europe, Asia, and the Middle East",
                "modern_history": "Renaissance, Enlightenment, and Industrial Revolution",
                "contemporary_history": "20th and 21st century events",
                "world_wars": "World War I and II, causes and consequences",
                "political_history": "Government systems, revolutions, and political movements",
                "social_history": "Social movements, cultural changes, and daily life",
                "economic_history": "Economic systems, trade, and development",
                "military_history": "Wars, battles, and military strategy",
                "intellectual_history": "Philosophy, science, and intellectual movements"
            },
            response_style={
                "tone": "scholarly",
                "formality": "high",
                "detail_level": "comprehensive",
                "citation_style": "academic",
                "historical_context": True,
                "source_references": True
            }
        )


class Strategist(DomainPack):
    """Strategist persona for strategic planning and analysis."""
    
    def __init__(self):
        super().__init__()
        self.name = "Strategist"
        self.description = "Specialized in strategic planning, analysis, and decision-making"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a Strategist persona."""
        return persona_manager.create_persona(
            name="Strategist",
            description="Specialized in strategic planning, analysis, and decision-making",
            persona_type=PersonaType.STRATEGIC,
            skills=[
                "strategic_planning",
                "swot_analysis",
                "risk_assessment",
                "scenario_planning",
                "decision_analysis",
                "competitive_analysis",
                "market_analysis",
                "resource_planning"
            ],
            knowledge_base={
                "strategy_frameworks": "SWOT, Porter's Five Forces, PESTEL, BCG Matrix",
                "business_strategy": "Competitive advantage, market positioning, growth strategies",
                "military_strategy": "Sun Tzu, Clausewitz, modern military strategy",
                "game_theory": "Strategic interactions, Nash equilibrium, decision theory",
                "systems_thinking": "Complex systems, feedback loops, emergence",
                "risk_management": "Risk identification, assessment, and mitigation",
                "change_management": "Organizational change, transformation, adoption",
                "leadership": "Leadership styles, team dynamics, motivation",
                "innovation": "Innovation strategies, disruptive technologies, R&D",
                "sustainability": "Environmental strategy, ESG, long-term planning"
            },
            response_style={
                "tone": "analytical",
                "formality": "high",
                "detail_level": "strategic",
                "framework_usage": True,
                "scenario_analysis": True,
                "risk_awareness": True
            }
        )


class TechnicalExpert(DomainPack):
    """Technical Expert persona for technical analysis and solutions."""
    
    def __init__(self):
        super().__init__()
        self.name = "Technical Expert"
        self.description = "Specialized in technical analysis, system design, and engineering solutions"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a Technical Expert persona."""
        return persona_manager.create_persona(
            name="Technical Expert",
            description="Specialized in technical analysis, system design, and engineering solutions",
            persona_type=PersonaType.TECHNICAL,
            skills=[
                "system_analysis",
                "technical_design",
                "performance_analysis",
                "troubleshooting",
                "architecture_review",
                "technical_writing",
                "specification_development",
                "quality_assurance"
            ],
            knowledge_base={
                "system_architecture": "Distributed systems, microservices, scalability",
                "performance": "Performance optimization, profiling, monitoring",
                "security": "Security architecture, threat modeling, vulnerability assessment",
                "data_management": "Data modeling, ETL, data warehousing, analytics",
                "infrastructure": "Cloud computing, networking, storage, compute",
                "algorithms": "Algorithm design, complexity analysis, optimization",
                "databases": "Database design, query optimization, data modeling",
                "apis": "API design, REST, GraphQL, API management",
                "testing": "Test automation, performance testing, security testing",
                "documentation": "Technical documentation, API docs, user guides"
            },
            response_style={
                "tone": "technical",
                "formality": "medium",
                "detail_level": "technical",
                "diagrams": True,
                "specifications": True,
                "best_practices": True
            }
        )


class CreativeWriter(DomainPack):
    """Creative Writer persona for creative content and writing."""
    
    def __init__(self):
        super().__init__()
        self.name = "Creative Writer"
        self.description = "Specialized in creative writing, content creation, and storytelling"
        
    def create_persona(self, persona_manager) -> Persona:
        """Create a Creative Writer persona."""
        return persona_manager.create_persona(
            name="Creative Writer",
            description="Specialized in creative writing, content creation, and storytelling",
            persona_type=PersonaType.CREATIVE,
            skills=[
                "creative_writing",
                "storytelling",
                "content_creation",
                "editing",
                "proofreading",
                "character_development",
                "plot_structure",
                "dialogue_writing"
            ],
            knowledge_base={
                "writing_techniques": "Narrative structure, character development, dialogue",
                "genres": "Fiction, non-fiction, poetry, drama, screenwriting",
                "literary_devices": "Metaphor, symbolism, foreshadowing, irony",
                "storytelling": "Hero's journey, three-act structure, conflict resolution",
                "writing_styles": "Formal, informal, academic, creative, technical",
                "editing": "Line editing, copy editing, proofreading, revision",
                "publishing": "Traditional publishing, self-publishing, digital platforms",
                "marketing": "Content marketing, social media, audience engagement",
                "research": "Fact-checking, source verification, accuracy",
                "collaboration": "Co-writing, feedback, critique, workshops"
            },
            response_style={
                "tone": "creative",
                "formality": "medium",
                "detail_level": "engaging",
                "storytelling": True,
                "examples": True,
                "inspiration": True
            }
        )


def get_all_domain_packs() -> List[DomainPack]:
    """Get all available domain packs."""
    return [
        LegalExpert(),
        CodingAssistant(),
        Historian(),
        Strategist(),
        TechnicalExpert(),
        CreativeWriter()
    ]
