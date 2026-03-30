# Agentic AI Systems: From Prompts to Production
> **Companion code repository** — *Birendra Kumar Sahu*

## About

This repository contains all **interactive Streamlit demo applications** from the book *Agentic AI Systems: From Prompts to Production*. Each demo is a standalone, fully runnable application that makes a core chapter concept tangible through hands-on experimentation — not just reading about it.

Every demo maps directly to a chapter. Run the app, experiment, then read — or read first, then explore. Either order works.

---

## Repository Structure

```
ai-agents-ebook/
│
├── README.md                               ← You are here
├── requirements-all.txt                    ← Install all dependencies at once
├── LICENSE
│
├── 01-prompt-engineering/
│   ├── prompt_engineering_demo.py          ← Chapter 2
│   ├── requirements.txt
│   └── README.md
│
├── 02-rag/
│   ├── rag_demo_app.py                     ← Chapter 4
│   ├── requirements.txt
│   └── README.md
│
├── 03-memory-architectures/
│   ├── memory_architectures_demo.py        ← Chapter 5
│   ├── requirements.txt
│   └── README.md
│
└── 04-knowledge-graph/
│   ├── knowledge_graph_demo.py             ← Chapter 6
│   ├── requirements.txt
│   └── README.md
│
├── 05-context-engineering/
│   ├── context_engineering_demo.py         ← Chapter 7
│   ├── requirements.txt
│   └── README.md
│

```

---

## The Five Demos

| Folder | Demo App | Chapter | What You Learn |
|--------|----------|---------|----------------|
| `01-prompt-engineering` | Prompt Engineering | Ch 2 | Bad→Good prompt ladder, 4 real finance scenarios, data grounding vs hallucination |
| `02-rag` | RAG Application | Ch 4 | Chunking, LanceDB vector search, Naive/Agentic/CRAG/HyDE patterns, evaluation |
| `03-memory-architectures` | Memory Architectures | Ch 5 | Episodic, semantic, procedural memory, decay scoring, Mem0 integration |
| `04-knowledge-graph` | Knowledge Graphs | Ch 6 | Triples, ontology, multi-hop reasoning, GraphRAG, SPARQL queries |
| `05-context-engineering` | Context Engineering | Ch 7 | Token budget assembly, memory pruning, re-ranking, context poisoning |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/BirendraSahu/ai-agents-ebook.git
cd ai-agents-ebook
```

### 2. Install dependencies

```bash
# All demos at once
pip install -r requirements-all.txt

# Or just one demo
cd 01-prompt-engineering && pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

> You can also enter the key directly in the sidebar of any demo app.

### 4. Run a demo

```bash
streamlit run 01-prompt-engineering/prompt_engineering_demo.py
```

All apps open at `http://localhost:8501`.

---

## Running All Demos

```bash
# Prompt Engineering — Chapter 2
streamlit run 01-prompt-engineering/prompt_engineering_demo.py

# RAG — Chapter 4
streamlit run 02-rag/rag_demo_app.py

# Memory Architectures — Chapter 5
streamlit run 03-memory-architectures/memory_architectures_demo.py

# Knowledge Graphs — Chapter 6
streamlit run 04-knowledge-graph/knowledge_graph_demo.py

# Context Engineering — Chapter 7
streamlit run 05-context-engineering/context_engineering_demo.py


```

---

## Requirements

All demos use Python 3.11+. Full dependency list:

```
streamlit>=1.32.0
openai>=1.12.0
pandas>=2.0.0
numpy>=1.24.0
lancedb>=0.5.0
pyarrow>=14.0.0
networkx>=3.2
mem0ai>=0.0.20
rdflib>=7.0.0
tiktoken>=0.5.0
plotly>=5.18.0
```

---

## About the Book

**Agentic AI Systems: From Prompts to Production** is a practitioner's guide to building production-grade agentic AI — systems that work reliably under load with real data, not toy demos.

**What the book covers:**

- How LLMs work: training pipeline, tokens, embeddings, hallucination
- Prompt engineering from Worst to Good with four enterprise scenarios
- RAG: Naive, Agentic, CRAG, and HyDE patterns with production pipelines
- Memory: working, episodic, semantic, and procedural architectures
- Knowledge Graphs, multi-hop reasoning, and GraphRAG hybrids
- Context Engineering: the four-pillar assembly pipeline
- LangChain and LangGraph with five complete production use cases
- Reasoning patterns: ReAct, Plan-Execute-Reflect, Tree-of-Thoughts
- MCP and multi-agent orchestration
- Real production failure stories with root causes and exact fixes

---

## About the Author

**Birendra Kumar Sahu** is a Distinguished Engineer and systems architect with 25+ years of experience building enterprise-scale data platforms and agentic AI systems. He holds 5 U.S. patents in data systems and machine learning algorithms, and has published IEEE research on AI-powered payment routing systems.

- LinkedIn: [linkedin.com/in/birendrasahu](https://linkedin.com/in/birendrasahu)
- GitHub: [github.com/BirendraSahu](https://github.com/BirendraSahu)

---

## Contributing

Found a bug or want to improve a demo? Pull requests are welcome.

1. Fork the repo
2. Create a branch: `git checkout -b fix/your-description`
3. Commit: `git commit -m "fix: describe the change"`
4. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*If this repo helped you, please give it a ⭐ — it helps others find it.*

