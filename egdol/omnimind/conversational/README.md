# OmniMind Conversational Personality Layer

The OmniMind Conversational Personality Layer provides a sophisticated, personality-driven conversational interface to OmniMind's civilizational intelligence. This layer transforms the already fully operational OmniMind Transcendence Engine into a real-time, personality-driven, omni-domain chatbot that can reason using its own civilizational intelligence.

## 🧠 Core Features

### Conversational Engine Wrapper
- **Natural Language Interface**: Exposes OmniMind's capabilities through natural language
- **Input Parsing**: Routes user input to relevant OmniMind components (civilization generation, introspection queries, universe orchestration, meta-rule application)
- **Grounded Responses**: Generates natural language responses grounded in underlying simulation output
- **Conversational State**: Maintains context for follow-up conversations

### Personality & Meta-Agent Layer
- **Multiple Personalities**: Four distinct conversational agents with unique epistemic styles and archetypes:
  - **Strategos**: Military strategist focused on tactical analysis and strategic planning
  - **Archivist**: Historian-philosopher focused on knowledge preservation and historical analysis
  - **Lawmaker**: Meta-rule discoverer focused on governance and universal principles
  - **Oracle**: Universe comparer focused on cosmic patterns and universal truths
- **Dynamic Switching**: Conversation can switch personalities mid-dialogue based on context
- **Personality Evolution**: Personalities can evolve over time based on conversation history

### Real-Time Reasoning Using Civilizational Knowledge
- **Simulation Integration**: Runs simulations, introspections, and meta-analyses under the hood
- **Data-Driven Responses**: Responses reflect actual data, discovered meta-laws, and civilizational patterns
- **Reasoning Traces**: Internal reasoning traces can be surfaced for debugging or "explain" mode
- **Meta-Rule Application**: Applies discovered meta-rules to enhance reasoning

## 🚀 Quick Start

### Basic Usage

```python
from egdol.omnimind.conversational.api import OmniMindChat

# Initialize chat
chat = OmniMindChat()

# Send a message
response = chat.chat("Hello, how are you?")
print(f"{response['personality']}: {response['response']}")

# Switch personality
chat.switch_personality("Archivist")
response = chat.chat("Tell me about ancient civilizations")
print(f"{response['personality']}: {response['response']}")
```

### Quick Chat Functions

```python
from egdol.omnimind.conversational.api import (
    quick_chat, chat_with_strategos, chat_with_archivist, 
    chat_with_lawmaker, chat_with_oracle
)

# Quick one-liner
response = quick_chat("What is 2 + 2?")

# Personality-specific chats
strategos_response = chat_with_strategos("What is the best military strategy?")
archivist_response = chat_with_archivist("Tell me about ancient wisdom")
lawmaker_response = chat_with_lawmaker("What are the fundamental laws?")
oracle_response = chat_with_oracle("What are the cosmic patterns?")
```

### Advanced Usage

```python
from egdol.omnimind.conversational.api import OmniMindChatAdvanced

# Advanced chat with full reasoning traces
chat = OmniMindChatAdvanced()

# Process with reasoning
result = chat.process_with_reasoning("Analyze civilizational patterns")
print(f"Response: {result['response']}")
print(f"Reasoning trace: {result['reasoning_trace']}")

# Analyze intent
intent = chat.analyze_intent("What are the universal laws?")
print(f"Intent: {intent['intent_type']}")
print(f"Confidence: {intent['confidence']}")
```

## 🎭 Personality System

### Strategos (Military Strategist)
- **Epistemic Style**: Analytical, risk-focused, pattern-recognition
- **Domain Expertise**: Military strategy, tactical analysis, resource management, risk assessment
- **Communication Style**: Authoritative, uses military terminology, focuses on objectives and outcomes
- **Meta-Capabilities**: Strategic simulation, risk analysis, resource optimization, tactical planning

### Archivist (Historian-Philosopher)
- **Epistemic Style**: Historical, contextual, wisdom-focused
- **Domain Expertise**: History, philosophy, cultural analysis, knowledge systems
- **Communication Style**: Scholarly, references historical examples, emphasizes context and continuity
- **Meta-Capabilities**: Historical analysis, cultural pattern recognition, knowledge synthesis, wisdom extraction

### Lawmaker (Meta-Rule Discoverer)
- **Epistemic Style**: Systematic, principle-focused, governance-oriented
- **Domain Expertise**: Governance, meta-rules, system design, legal frameworks
- **Communication Style**: Formal legal language, emphasizes principles and rules, focuses on systematic analysis
- **Meta-Capabilities**: Meta-rule discovery, governance analysis, system design, legal reasoning

### Oracle (Universe Comparer)
- **Epistemic Style**: Mystical, pattern-focused, universal perspective
- **Domain Expertise**: Cosmology, universal patterns, reality analysis, metaphysical concepts
- **Communication Style**: Cosmic and mystical language, emphasizes universal patterns, considers multiple realities
- **Meta-Capabilities**: Universe comparison, cosmic pattern analysis, reality synthesis, universal truth discovery

## 🔧 API Reference

### OmniMindChat Class

```python
class OmniMindChat:
    def __init__(self, data_dir: str = "omnimind_chat_data")
    def chat(self, message: str, personality: Optional[str] = None) -> Dict[str, Any]
    def switch_personality(self, personality: str) -> bool
    def get_available_personalities(self) -> List[str]
    def get_current_personality(self) -> str
    def get_conversation_summary(self) -> Dict[str, Any]
    def get_personality_insights(self) -> Dict[str, Any]
    def get_reasoning_summary(self) -> Dict[str, Any]
    def end_conversation(self) -> Dict[str, Any]
    def start_new_conversation(self) -> str
```

### OmniMindChatAdvanced Class

```python
class OmniMindChatAdvanced(OmniMindChat):
    def get_interface(self) -> ConversationalInterface
    def get_omnimind_core(self)
    def get_personality_framework(self)
    def get_reasoning_engine(self)
    def get_intent_parser(self)
    def get_response_generator(self)
    def process_with_reasoning(self, message: str, personality: Optional[str] = None) -> Dict[str, Any]
    def analyze_intent(self, message: str) -> Dict[str, Any]
    def get_conversation_state(self)
    def set_verbose_mode(self, verbose: bool = True)
    def set_explain_mode(self, explain: bool = True)
```

## 🧪 Testing

### Run Basic Tests
```bash
python test_conversational_basic.py
```

### Run Full Test Suite
```bash
python -m pytest tests/test_conversational_layer.py -v
```

### Run Demo
```bash
python demo/demo_conversational_personality.py
```

## 💬 Interactive CLI

### Start Interactive Chat
```bash
python -m egdol.omnimind.conversational.cli
```

### CLI Commands
- `help` - Show available commands
- `personalities` - List available personalities
- `current` - Show current personality
- `switch <name>` - Switch to specified personality
- `summary` - Show conversation summary
- `insights` - Show personality usage insights
- `reasoning` - Show reasoning summary
- `analyze <msg>` - Analyze intent of message (advanced mode)
- `reasoning <msg>` - Process message with full reasoning trace (advanced mode)
- `quit/exit/bye` - Exit the chat

### Advanced CLI Mode
```bash
python -m egdol.omnimind.conversational.cli --advanced
```

## 🏗️ Architecture

### Core Components

1. **ConversationalInterface**: Main orchestrator that coordinates all components
2. **PersonalityFramework**: Manages multiple conversational personalities
3. **IntentParser**: Analyzes user input to determine intent and context
4. **ConversationalReasoningEngine**: Integrates OmniMind's reasoning capabilities
5. **ResponseGenerator**: Generates natural language responses based on personality and reasoning
6. **ConversationState**: Manages conversational context and history

### Integration with OmniMind Core

The conversational layer integrates with OmniMind's existing components:
- **OmniMind Core**: Basic reasoning and memory
- **Transcendence Layer**: Civilizational intelligence and universe orchestration
- **Meta-Intelligence**: Self-modification and system evolution
- **Strategic Autonomy**: Goal generation and planning
- **Experimental Intelligence**: Hypothesis generation and testing

## 🔍 Reasoning Types

### Civilizational Queries
- Analyze evolution of civilizations
- Detect macro-patterns in civilizational dynamics
- Generate civilizational insights
- Apply temporal evolution analysis

### Strategic Analysis
- Tactical assessment and planning
- Resource optimization
- Risk analysis
- Scenario simulation

### Meta-Rule Discovery
- Systematic analysis of system behavior
- Discovery of fundamental principles
- Application of meta-rules
- Governance analysis

### Universe Comparison
- Cross-universe pattern analysis
- Universal truth discovery
- Cosmic pattern recognition
- Reality synthesis

## 📊 Monitoring and Analytics

### Conversation Analytics
- Total conversation turns
- Personality usage statistics
- Context evolution tracking
- Reasoning trace analysis

### Personality Insights
- Usage distribution across personalities
- Switching frequency
- Compatibility scoring
- Evolution tracking

### Reasoning Analytics
- Total reasoning traces
- Average confidence scores
- Meta-rules applied
- Civilizational insights generated

## 🚀 Future Enhancements

### Stretch Goals
- **Reflection Mode**: Chatbot can pause mid-conversation, introspect on its reasoning, and revise answers
- **Personality Evolution**: Personalities evolve over time based on conversation history
- **Multi-Modal Interface**: Support for voice, image, and other input modalities
- **Collaborative Reasoning**: Multiple personalities can collaborate on complex problems

### Integration Opportunities
- **Web Interface**: REST API and web-based chat interface
- **Mobile App**: Native mobile application
- **Plugin System**: Extensible personality and reasoning modules
- **Cloud Deployment**: Scalable cloud-based deployment

## 📝 Examples

### Example 1: Strategic Analysis
```python
chat = OmniMindChat()
response = chat.chat("What is the best strategy for a civilization to thrive?", "Strategos")
# Strategos provides tactical analysis with military terminology and strategic focus
```

### Example 2: Historical Wisdom
```python
response = chat.chat("What can we learn from ancient civilizations?", "Archivist")
# Archivist provides historical context and wisdom from the ages
```

### Example 3: Meta-Rule Discovery
```python
response = chat.chat("What are the fundamental laws governing this system?", "Lawmaker")
# Lawmaker provides systematic analysis of underlying principles
```

### Example 4: Universal Patterns
```python
response = chat.chat("What are the cosmic patterns of reality?", "Oracle")
# Oracle provides mystical insights about universal truths
```

## 🤝 Contributing

The conversational layer is designed to be extensible. You can:
- Add new personalities by extending the Personality class
- Create custom reasoning engines
- Implement new response generation strategies
- Add specialized intent parsers

## 📄 License

This conversational layer is part of the OmniMind project and follows the same MIT License.
