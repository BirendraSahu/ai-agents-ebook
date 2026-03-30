# Knowledge Graphs for Agentic Reasoning - Demo App

A comprehensive Streamlit application demonstrating Knowledge Graph concepts, ontology, multi-hop reasoning, and GraphRAG using a local in-memory graph database.

## Features

- **Local Graph Database**: Simple in-memory Knowledge Graph (no external setup required)
- **Ontology Management**: Define classes, properties, and relationships
- **Triple Management**: Add, query, and visualize S-P-O triples
- **Multi-Hop Reasoning**: Traverse multiple relationships to find answers
- **Natural Language Queries**: Convert natural language to graph queries using OpenAI
- **GraphRAG Hybrid**: Combine Knowledge Graph with RAG for enhanced reasoning
- **SPARQL Generation**: Generate SPARQL-like queries from natural language
- **Hands-on Labs**: Interactive exercises for building and querying KGs

## Installation

```bash
pip install -r requirements_kg_demo.txt
```

## Usage

```bash
streamlit run knowledge_graph_demo.py
```

## Graph Database

The app uses a **Simple In-Memory Graph Database** that:
- Stores triples as (Subject, Predicate, Object)
- Supports pattern matching queries
- Enables multi-hop reasoning
- Converts to NetworkX for visualization
- No external database setup required!

## Concepts Demonstrated

### 1. Knowledge Graph Fundamentals
- Triple structure (S-P-O)
- Ontology as schema
- Comparison: KG vs RAG vs Metadata Graph

### 2. Building Knowledge Graphs
- Manual triple insertion
- Sample corporate project management ontology
- Entity and relationship management

### 3. Querying Knowledge Graphs
- Natural language to graph query conversion
- SPARQL query generation
- Pattern matching queries

### 4. Multi-Hop Reasoning
- Traversing multiple relationships
- Path finding algorithms
- Complex query resolution

### 5. GraphRAG Hybrid
- Combining KG structural retrieval with RAG semantic retrieval
- Enhanced reasoning capabilities
- Context assembly from multiple sources

### 6. Hands-on Labs
- Lab 1: Build the Graph
- Lab 2: Agent Reasoning
- Lab 3: Virtual Mapping Simulation

## Sample Use Cases

### Corporate Project Management
- Employee skills and project requirements
- Management hierarchies
- Project dependencies
- Program and portfolio relationships

### Example Queries
- "What skills are needed for the project Alice manages?"
- "Who should lead Project Y if it requires Python and Cloud Architecture?"
- "Find all dependencies of Payment Service"
- "What projects are part of the Digital Transformation Program?"

## Key Takeaways

1. **KGs provide structure**: While RAG provides semantic similarity, KGs provide exact relationships
2. **Multi-hop reasoning**: KGs excel at traversing multiple relationships
3. **Deterministic logic**: KGs provide verifiable, exact facts (critical for regulated industries)
4. **GraphRAG is powerful**: Combining KG structure with RAG semantics provides the best of both worlds
5. **Ontology is governance**: Well-defined ontologies prevent inconsistent world-views

## Architecture

- **Graph Storage**: In-memory triple store
- **Query Engine**: Pattern matching + multi-hop traversal
- **LLM Integration**: OpenAI for natural language understanding and query generation
- **Visualization**: NetworkX for graph structure representation

## Future Enhancements

- [ ] Integration with actual GraphDB (Neo4j, GraphDB)
- [ ] SPARQL endpoint simulation
- [ ] RDFLib full integration
- [ ] Graph visualization with plotly/networkx
- [ ] Virtual mapping to real SQL databases
- [ ] Ontology editor UI
- [ ] Inference engine for transitive/inverse relationships
