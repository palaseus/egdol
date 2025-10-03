egdol — a local rule-based reasoning system
============================================

Overview
--------
egdol is a research-oriented rule-based reasoning system implemented in Python
with zero external dependencies. It provides a foundation for experimenting with
logic programming, multi-agent systems, and autonomous reasoning architectures.

The system consists of two main components:
- **Core Engine**: A Prolog-style rule and fact engine for local reasoning
- **OmniMind**: A multi-agent reasoning system with autonomous capabilities

Core Engine
-----------
The core engine provides basic rule-based reasoning with:
- `egdol/lexer.py` — tokenization of the mini-DSL
- `egdol/parser.py` — AST and parser for facts, rules, and queries  
- `egdol/rules_engine.py` — storage, indexing, and constraint support
- `egdol/interpreter.py` — unification and depth-first proof search
- `egdol/main.py` — REPL and session runner

Quick start
-----------
Start the REPL:

```bash
python3 -m egdol.main
```

Examples:
- Add a fact: `fact: human(socrates).`
- Add a rule: `rule: mortal(X) => human(X).`
- Query: `? mortal(socrates).`

REPL commands:
- `:load <file>` — load facts/rules from a file
- `:facts` — list loaded facts
- `:rules` — list loaded rules
- `:trace on|off` — enable/disable tracing
- `:help` — show help

OmniMind System
---------------
The OmniMind system extends the core engine with:

**Multi-Agent Network Layer** (`egdol/omnimind/network/`)
- Agent communication and coordination
- Task delegation and collaborative reasoning
- Network monitoring and learning

**Strategic Autonomy** (`egdol/omnimind/strategic/`)
- Autonomous goal generation and planning
- Scenario simulation and risk assessment
- Policy evolution and optimization

**Experimental Intelligence** (`egdol/omnimind/experimental/`)
- Hypothesis generation and testing
- Creative synthesis and knowledge expansion
- Autonomous research capabilities

**Meta-Intelligence** (`egdol/omnimind/meta/`)
- Architecture invention and self-modification
- Skill and policy innovation
- System evolution and evaluation

**Self-Creation System** (`egdol/omnimind/progeny/`)
- Autonomous agent generation
- Sandbox testing and evaluation
- Safe integration and rollback mechanisms

Testing
-------
Run the test suite:

```bash
# Core engine tests
python3 -m pytest tests/test_*.py -v

# OmniMind system tests  
python3 -m pytest tests/test_omnimind_*.py -v
python3 -m pytest tests/test_network_*.py -v
python3 -m pytest tests/test_strategic_*.py -v
python3 -m pytest tests/test_experimental_*.py -v
python3 -m pytest tests/test_meta_*.py -v
python3 -m pytest tests/test_self_creation_*.py -v
```

Architecture
------------
The system is designed with modularity and extensibility in mind:

- **Core Engine**: Pure rule-based reasoning without external dependencies
- **Network Layer**: Multi-agent communication and coordination
- **Strategic Layer**: High-level planning and goal management
- **Experimental Layer**: Research and knowledge discovery
- **Meta Layer**: Self-modification and system evolution
- **Progeny Layer**: Autonomous agent creation and testing

All components operate fully offline and deterministically, making the system
suitable for research, education, and local experimentation.

Profiling and Analysis
----------------------
The system includes built-in profiling for performance analysis:

- `:profile summary` — performance overview
- `:profile top N` — identify bottlenecks
- `:profile export-json <file>` — structured data export
- `:profile analyze` — automated analysis

Contributing
------------
This is a research and educational project. Contributions should focus on:
- Extending reasoning capabilities
- Improving multi-agent coordination
- Enhancing autonomous behaviors
- Adding new experimental modalities

Please include comprehensive tests for any new functionality.

License
-------
MIT License.