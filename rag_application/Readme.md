# RAG Demo Application - Comprehensive RAG Tutorial

A comprehensive Streamlit application that demonstrates all concepts from **Module 2: Retrieval-Augmented Generation (RAG)**. This interactive tutorial covers grounding LLMs, semantic search, embeddings, vector databases, tool use, and the multi-modal lakehouse pattern.

## Features

### 📚 Complete Module Coverage

- **Module 2.1**: The Necessity of Grounding & RAG Architecture Patterns
  - Why vanilla LLMs fail
  - RAG architecture patterns (Naive, Agentic, Corrective, HyDE)
  - Chunking strategies (Fixed-size, Recursive, Semantic)
  - Interactive demos for each concept

- **Module 2.2**: Semantic Search, Embeddings, and Vector Databases
  - Semantic search vs keyword search
  - How embeddings work
  - Understanding embedding dimensions
  - Vector database concepts
  - Search mechanisms (Sparse, Dense, Hybrid)

- **Module 2.3**: Designing for Tool Use - The Agentic Contract
  - Semantic tool definitions
  - ReAct pattern (Reason + Act)
  - Best practices for RAG tool implementation
  - RAG evaluation metrics (RAGAS framework)

- **Module 2.4**: The Multi-Modal Lakehouse Pattern
  - Synchronization gap problem
  - Multi-modal lakehouse solution
  - Using LanceDB as a multi-modal store
  - Workflow integration
  - Best practices

### 🎮 Interactive Demos

- **Vanilla LLM vs RAG**: Compare responses with and without grounding
- **Chunking Strategies**: Try fixed-size, recursive, and semantic chunking
- **Semantic Search**: Compare keyword vs semantic search results
- **Embedding Generation**: Generate and visualize embeddings
- **Vector Database Search**: Search LanceDB with semantic queries
- **ReAct Pattern**: Simulate agent reasoning and tool use
- **RAG Playground**: Upload documents and query with full RAG pipeline

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) LanceDB for vector storage

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_rag_demo.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run rag_demo_app.py
   ```

3. **Configure in sidebar:**
   - Enter your OpenAI API key
   - Set LanceDB path (default: `./lancedb_rag_demo`)
   - Select models (embeddings and LLM)

## Usage

### Getting Started

1. **Open the app** in your browser (usually `http://localhost:8501`)

2. **Configure settings** in the sidebar:
   - Enter OpenAI API key (required for most features)
   - Set LanceDB path for vector storage
   - Choose embedding and LLM models

3. **Navigate through modules:**
   - Use the tabs to explore different RAG concepts
   - Each module has interactive demos
   - Try the examples and experiment with your own data

### Module Navigation

- **📚 Overview**: Introduction to RAG and course structure
- **🔧 Module 2.1**: Grounding & RAG Patterns
- **🔍 Module 2.2**: Semantic Search & Embeddings
- **🛠️ Module 2.3**: Tool Use & Agents
- **🏗️ Module 2.4**: Multi-Modal Lakehouse
- **🎮 Interactive Playground**: Full RAG pipeline demo

### Interactive Playground

The playground allows you to:

1. **Upload Documents:**
   - Upload text files (.txt, .md)
   - Or enter text manually
   - Documents are automatically chunked

2. **Query with RAG:**
   - Enter natural language questions
   - System retrieves relevant chunks
   - LLM generates grounded answers with citations

3. **Experiment:**
   - Try different chunking strategies
   - Adjust top-K retrieval
   - Compare RAG vs vanilla LLM responses

## Key Concepts Demonstrated

### 1. Why Grounding Matters

- **Static Knowledge Problem**: LLMs are frozen in time
- **Hallucinations**: LLMs make up answers when they don't know
- **No Citations**: Vanilla LLMs can't prove their answers
- **Data Privacy**: Can't easily train on sensitive data

### 2. RAG Architecture Patterns

- **Naive RAG**: Simple linear pipeline
- **Agentic RAG**: Agent decides when/how to search
- **Corrective RAG**: Self-grading quality checks
- **HyDE**: Hypothetical document embeddings

### 3. Chunking Strategies

- **Fixed-Size**: Split every N characters/tokens
- **Recursive**: Split at natural boundaries (paragraphs, sentences)
- **Semantic**: Use embeddings to find meaning boundaries

### 4. Semantic Search

- **Keyword Search**: Exact character matching
- **Semantic Search**: Understanding intent and meaning
- **Embeddings**: Converting text to vectors
- **Vector Databases**: Fast similarity search

### 5. Tool Use & Agents

- **Semantic Tool Definitions**: Clear, purpose-driven tool descriptions
- **ReAct Pattern**: Reasoning and acting in language models
- **Structured Output**: JSON/Pydantic schemas
- **Graceful Failures**: Error handling and retry logic

### 6. Multi-Modal Lakehouse

- **Synchronization Gap**: Embeddings vs source data
- **Unified Storage**: Vectors + raw data + metadata in one place
- **LanceDB**: Multi-modal columnar format
- **Hybrid Retrieval**: Semantic search + metadata filtering

## Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **OpenAI**: Embeddings and LLM API
- **LanceDB**: Vector database for local storage
- **PyArrow**: Columnar data format
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

### Models Used

- **Embeddings**: `text-embedding-3-small` (default), `text-embedding-3-large`, `text-embedding-ada-002`
- **LLM**: `gpt-4o-mini` (default), `gpt-4o`, `gpt-3.5-turbo`

### Data Storage

- **LanceDB**: Local vector database (default path: `./lancedb_rag_demo`)
- **Format**: Apache Arrow columnar format
- **Indexing**: Disk-based indexes (IVF-PQ) for fast search

## Examples

### Example 1: Compare Vanilla LLM vs RAG

1. Go to **Module 2.1 → Why Grounding?**
2. Enter a question about recent events: "What was NVIDIA's stock price on January 15, 2025?"
3. Click "Ask Vanilla LLM" - see potentially outdated/incorrect answer
4. Click "Ask with RAG" - see how RAG would ground the answer in retrieved documents

### Example 2: Try Different Chunking Strategies

1. Go to **Module 2.1 → Chunking Strategies**
2. Enter sample text
3. Try **Fixed-Size Chunking** with different chunk sizes and overlaps
4. Try **Recursive Chunking** and see how it preserves sentence boundaries
5. Compare results

### Example 3: Semantic Search Demo

1. Go to **Module 2.2 → Semantic Search**
2. Enter a search query: "company earnings"
3. Compare **Keyword Search** (exact matches) vs **Semantic Search** (meaning-based)
4. See how semantic search finds related concepts even with different words

### Example 4: Full RAG Pipeline

1. Go to **Interactive Playground**
2. Upload documents or enter text manually
3. Click "Add Documents to Knowledge Base"
4. Enter a question about your documents
5. Click "Query Knowledge Base"
6. See retrieved chunks and RAG-generated answer with citations

## Best Practices

### For Chunking

- Use **recursive chunking** for structured documents
- Use **semantic chunking** for complex, unstructured text
- Add **contextual metadata** to chunks (title, category, etc.)
- Consider **overlap** between chunks to preserve context

### For Embeddings

- Choose **dimension size** based on your needs:
  - Small (300-512): Fast, good for simple use cases
  - Large (1536+): Better accuracy, more expensive
- Use **consistent models** for embeddings and queries
- Consider **fine-tuning** for domain-specific data

### For Vector Databases

- Use **metadata filtering** to combine semantic search with traditional filters
- Implement **hybrid search** (sparse + dense) for best results
- Monitor **retrieval metrics** (Recall@K, MRR)
- Keep **embeddings in sync** with source data

### For RAG Implementation

- **Ground answers** in retrieved context
- **Cite sources** for transparency
- **Handle failures** gracefully
- **Evaluate quality** using RAGAS metrics
- **Version control** your knowledge base

## Troubleshooting

### OpenAI API Errors

- **Invalid API Key**: Check that your API key is correct and has credits
- **Rate Limits**: You may be hitting rate limits - try a different model or wait
- **Model Not Found**: Ensure you're using a valid model name

### LanceDB Errors

- **Path Issues**: Ensure the LanceDB path is writable
- **Table Not Found**: Initialize the table first using the demo buttons
- **Schema Mismatch**: Delete the existing table and recreate it

### Embedding Generation Fails

- **API Key**: Ensure OpenAI API key is set
- **Text Too Long**: Some models have token limits - chunk large text first
- **Network Issues**: Check your internet connection

## Resources

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)

## License

This demo application is for educational purposes.

## Contributing

Feel free to extend this demo with:
- Additional RAG patterns
- More chunking strategies
- Evaluation metrics visualization
- Multi-modal support (images, audio)
- Integration with other vector databases
