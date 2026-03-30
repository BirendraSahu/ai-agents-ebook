"""
RAG Demo Application - Comprehensive RAG Tutorial and Interactive Demo
Covers all topics from Module 2: Retrieval-Augmented Generation
"""

import streamlit as st
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# OpenAI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# LanceDB imports
try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Demo - Retrieval-Augmented Generation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Helper Functions
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None
    if 'lancedb_path' not in st.session_state:
        st.session_state.lancedb_path = None
    if 'lancedb_db' not in st.session_state:
        st.session_state.lancedb_db = None
    if 'lancedb_table' not in st.session_state:
        st.session_state.lancedb_table = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'embeddings_model' not in st.session_state:
        st.session_state.embeddings_model = "text-embedding-3-small"
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "gpt-4o-mini"

def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for text using OpenAI."""
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def chunk_text_fixed_size(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Fixed-size chunking strategy."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap: keep last N words
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + [word]
            current_size = sum(len(w) + 1 for w in current_chunk)
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_text_recursive(text: str, max_size: int = 512, separators: List[str] = None) -> List[str]:
    """Recursive character splitting strategy."""
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ', '']
    
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    for separator in separators:
        if separator in text:
            splits = text.split(separator)
            for i, split in enumerate(splits):
                if i < len(splits) - 1:
                    split += separator
                if len(split) <= max_size:
                    chunks.append(split)
                else:
                    # Recursively split further
                    sub_chunks = chunk_text_recursive(split, max_size, separators[1:])
                    chunks.extend(sub_chunks)
            break
    
    return chunks if chunks else [text[:max_size]]

# ============================================================================
# Main App
# ============================================================================

def main():
    initialize_session_state()
    
    st.title("🤖 RAG Demo - Retrieval-Augmented Generation")
    st.markdown("**Comprehensive tutorial and interactive demo covering all RAG concepts**")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable all features"
        )
        
        if openai_key:
            try:
                st.session_state.openai_client = OpenAI(api_key=openai_key)
                st.success("✅ OpenAI connected")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.openai_client = None
        else:
            st.session_state.openai_client = None
            st.info("💡 Enter OpenAI API key to enable features")
        
        st.divider()
        
        # LanceDB Configuration
        st.subheader("LanceDB Setup")
        lancedb_path = st.text_input(
            "LanceDB Path",
            value="./lancedb_rag_demo",
            help="Local path for LanceDB storage"
        )
        
        if lancedb_path:
            st.session_state.lancedb_path = lancedb_path
            try:
                if LANCEDB_AVAILABLE:
                    # Create directory if it doesn't exist
                    os.makedirs(lancedb_path, exist_ok=True)
                    db = lancedb.connect(lancedb_path)
                    st.session_state.lancedb_db = db
                    st.success("✅ LanceDB connected")
                else:
                    st.warning("⚠️ Install LanceDB: `pip install lancedb`")
                    st.session_state.lancedb_db = None
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.lancedb_db = None
        else:
            st.session_state.lancedb_db = None
        
        st.divider()
        
        # Model Selection
        st.subheader("Model Selection")
        embeddings_model = st.selectbox(
            "Embeddings Model",
            options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0,
            help="OpenAI embeddings model"
        )
        st.session_state.embeddings_model = embeddings_model
        
        llm_model = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="OpenAI LLM model"
        )
        st.session_state.llm_model = llm_model
        
        st.divider()
        
        # Status
        st.subheader("Status")
        if st.session_state.openai_client:
            st.success("✅ OpenAI Ready")
        else:
            st.warning("⚠️ OpenAI Not Configured")
        
        if st.session_state.lancedb_db:
            st.success("✅ LanceDB Ready")
        else:
            st.warning("⚠️ LanceDB Not Configured")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📚 Overview",
        "🔧 Module 2.1: Grounding & RAG Patterns",
        "🔍 Module 2.2: Semantic Search & Embeddings",
        "🛠️ Module 2.3: Tool Use & Agents",
        "🏗️ Module 2.4: Multi-Modal Lakehouse",
        "🎮 Interactive Playground"
    ])
    
    # ========================================================================
    # Tab 1: Overview
    # ========================================================================
    
    with tab1:
        st.header("📚 RAG Overview")
        
        st.markdown("""
        ## What is Retrieval-Augmented Generation (RAG)?
        
        RAG is a technique that enhances Large Language Models (LLMs) by:
        - **Retrieving** relevant information from external knowledge bases
        - **Augmenting** the LLM's prompt with this retrieved context
        - **Generating** more accurate, grounded, and up-to-date responses
        
        ### Why RAG Matters
        
        **Vanilla LLMs have limitations:**
        - ❌ Static knowledge (trained on historical data)
        - ❌ Cannot access private/real-time data
        - ❌ Prone to hallucinations
        - ❌ No citations or source attribution
        - ❌ Expensive to retrain on new data
        
        **RAG solves these problems:**
        - ✅ Access to up-to-date information
        - ✅ Grounded in your own documents/data
        - ✅ Citations and source attribution
        - ✅ No need to retrain models
        - ✅ Cost-effective and scalable
        
        ### Course Structure
        
        This demo covers all modules:
        
        1. **Module 2.1**: The Necessity of Grounding & RAG Architecture Patterns
        2. **Module 2.2**: Semantic Search, Embeddings, and Vector Databases
        3. **Module 2.3**: Designing for Tool Use - The Agentic Contract
        4. **Module 2.4**: The Multi-Modal Lakehouse Pattern
        
        ### Getting Started
        
        1. Enter your OpenAI API key in the sidebar
        2. Configure LanceDB path (default: `./lancedb_rag_demo`)
        3. Navigate through the modules to learn concepts
        4. Use the Interactive Playground to experiment
        """)
        
        st.divider()
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modules", "4")
        with col2:
            st.metric("Concepts", "15+")
        with col3:
            st.metric("Interactive Demos", "10+")
        with col4:
            st.metric("Code Examples", "20+")
    
    # ========================================================================
    # Tab 2: Module 2.1 - Grounding & RAG Patterns
    # ========================================================================
    
    with tab2:
        st.header("🔧 Module 2.1: The Necessity of Grounding & RAG Patterns")
        
        # Sub-tabs for different sections
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "Why Grounding?",
            "RAG Architecture Patterns",
            "Chunking Strategies"
        ])
        
        with sub_tab1:
            st.subheader("1. The Necessity of Grounding: Why Vanilla LLMs Fail")
            
            st.markdown("""
            ### Key Problems with Vanilla LLMs
            
            **1. Static Parametric Knowledge**
            - LLMs are "frozen" in time
            - Cannot answer questions about events after training cutoff
            - Example: "What was our Q4 revenue?" → LLM doesn't know
            
            **2. Hallucinations**
            - When LLM lacks specific information, it generates plausible but incorrect answers
            - No way to verify accuracy
            
            **3. Lack of Citations**
            - Cannot "prove" answers by pointing to specific documents
            - No source attribution
            
            **4. Data Privacy**
            - Cannot easily train on sensitive, frequently changing corporate data
            - Massive costs and privacy risks
            """)
            
            st.divider()
            
            # Interactive Demo: Vanilla LLM vs RAG
            st.subheader("🎮 Interactive Demo: Vanilla LLM vs RAG")
            
            if not st.session_state.openai_client:
                st.warning("⚠️ Please enter OpenAI API key in sidebar to run this demo")
            else:
                # Hardcoded demo data for NVIDIA stock
                demo_documents = [
                    {
                        "text": "NVIDIA Corporation (NVDA) stock price on January 15, 2025 was $145.32 per share. The stock opened at $144.50 and closed at $145.32, representing a gain of 2.1% for the day. Trading volume was 45.2 million shares.",
                        "source": "Financial Report - January 2025"
                    },
                    {
                        "text": "NVIDIA's stock performance in January 2025 has been strong, with the stock reaching new highs. On January 15, 2025, NVDA closed at $145.32, up from $142.10 the previous day. The company's AI chip sales continue to drive investor confidence.",
                        "source": "Market Analysis - Tech Stocks"
                    },
                    {
                        "text": "As of January 15, 2025, NVIDIA (NVDA) stock price stood at $145.32. The company announced strong quarterly earnings, with revenue growth of 35% year-over-year. Analysts maintain a bullish outlook on the stock.",
                        "source": "Earnings Report Q1 2025"
                    },
                    {
                        "text": "NVIDIA stock reached $145.32 on January 15, 2025, marking a significant milestone. The stock has gained over 40% in the past 6 months, driven by strong demand for AI accelerators and data center solutions.",
                        "source": "Stock Market Summary"
                    }
                ]
                
                demo_query = st.text_input(
                    "Ask a question about recent events or private data:",
                    value="What was the stock price of NVIDIA on January 15, 2025?",
                    key="vanilla_query"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ❌ Vanilla LLM Response")
                    if st.button("Ask Vanilla LLM", key="vanilla_btn"):
                        if demo_query:
                            try:
                                response = st.session_state.openai_client.chat.completions.create(
                                    model=st.session_state.llm_model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": demo_query}
                                    ],
                                    temperature=0.7
                                )
                                vanilla_answer = response.choices[0].message.content
                                st.info(vanilla_answer)
                                st.caption("⚠️ Note: This answer may be outdated or incorrect - no grounding!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                with col2:
                    st.markdown("### ✅ RAG Response (with grounding)")
                    if st.button("Ask with RAG", key="rag_btn"):
                        if demo_query:
                            try:
                                # Step 1: Generate embeddings for demo documents (if not already done)
                                if 'demo_embeddings' not in st.session_state:
                                    with st.spinner("Generating embeddings for demo documents..."):
                                        demo_embeddings = []
                                        for doc in demo_documents:
                                            emb = get_embedding(
                                                doc["text"],
                                                st.session_state.openai_client,
                                                st.session_state.embeddings_model
                                            )
                                            if emb:
                                                demo_embeddings.append({
                                                    "text": doc["text"],
                                                    "source": doc["source"],
                                                    "embedding": emb
                                                })
                                        st.session_state.demo_embeddings = demo_embeddings
                                
                                # Step 2: Embed the query
                                query_embedding = get_embedding(
                                    demo_query,
                                    st.session_state.openai_client,
                                    st.session_state.embeddings_model
                                )
                                
                                if query_embedding and st.session_state.demo_embeddings:
                                    # Step 3: Find most similar documents (cosine similarity)
                                    similarities = []
                                    for doc in st.session_state.demo_embeddings:
                                        # Cosine similarity
                                        dot_product = sum(a * b for a, b in zip(query_embedding, doc["embedding"]))
                                        norm_a = sum(a * a for a in query_embedding) ** 0.5
                                        norm_b = sum(b * b for b in doc["embedding"]) ** 0.5
                                        similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
                                        similarities.append((doc, similarity))
                                    
                                    # Sort by similarity and get top 2
                                    similarities.sort(key=lambda x: x[1], reverse=True)
                                    top_docs = similarities[:2]
                                    
                                    # Step 4: Build context from retrieved documents
                                    context_parts = []
                                    with st.expander("📄 Retrieved Context (Top 2 documents)", expanded=False):
                                        for i, (doc, sim) in enumerate(top_docs, 1):
                                            st.markdown(f"**Document {i}** (similarity: {sim:.3f})")
                                            st.text(f"Source: {doc['source']}")
                                            st.text(doc['text'])
                                            st.divider()
                                            context_parts.append(f"[Source: {doc['source']}]\n{doc['text']}")
                                    
                                    full_context = "\n\n".join(context_parts)
                                    
                                    # Step 5: Generate answer with RAG
                                    rag_prompt = f"""
You are a helpful assistant. Answer the user's question based ONLY on the provided context.

Context:
{full_context}

User Question: {demo_query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite which source you used for your answer
- Be concise and accurate
- Include the exact stock price if mentioned in the context
                                    """
                                    
                                    response = st.session_state.openai_client.chat.completions.create(
                                        model=st.session_state.llm_model,
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                                            {"role": "user", "content": rag_prompt}
                                        ],
                                        temperature=0.3
                                    )
                                    
                                    rag_answer = response.choices[0].message.content
                                    st.success(rag_answer)
                                    st.caption("✅ Answer is grounded in retrieved documents with citations")
                                else:
                                    st.error("Failed to generate embeddings")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                st.exception(e)
        
        with sub_tab2:
            st.subheader("2. RAG Architecture Patterns")
            
            st.markdown("""
            ### Pattern 1: Naive RAG (Simple)
            
            **Linear Pipeline:**
            ```
            User Query → Embedding → Vector Search → Augment Prompt → Generate
            ```
            
            **Pros:**
            - Simple to implement
            - Fast and efficient
            - Good for straightforward Q&A
            
            **Cons:**
            - No query refinement
            - Single-pass retrieval
            - No quality checks
            """)
            
            st.divider()
            
            st.markdown("""
            ### Pattern 2: Agentic RAG
            
            **Agent-Based Flow:**
            ```
            User Query → Agent (LLM) → Decide: Search? → Multi-turn Search → Refine Query → Generate
            ```
            
            **Features:**
            - Agent uses LLM to decide if/how to search
            - Can perform multi-turn searches
            - Refines query if first result is poor
            - Can combine multiple tools
            
            **Use Cases:**
            - Complex queries requiring multiple steps
            - Dynamic query refinement
            - Multi-source information gathering
            """)
            
            st.divider()
            
            st.markdown("""
            ### Pattern 3: Corrective RAG (CRAG)
            
            **Self-Grading Pipeline:**
            ```
            User Query → Retrieve → Grade Quality → If Poor: Re-retrieve → Augment → Generate
            ```
            
            **Features:**
            - Includes "self-grading" step
            - Evaluates quality of retrieved documents
            - Re-retrieves if quality is poor
            - Ensures high-quality context
            
            **Benefits:**
            - Reduces hallucinations
            - Improves answer quality
            - Self-correcting mechanism
            """)
            
            st.divider()
            
            st.markdown("""
            ### Pattern 4: HyDE (Hypothetical Document Embeddings)
            
            **Two-Step Process:**
            ```
            User Query → Generate "Fake" Ideal Answer → Embed Fake Answer → 
            Search with Fake Answer → Retrieve Real Documents → Generate Final Answer
            ```
            
            **How it Works:**
            1. LLM generates a "fake" ideal answer to the query
            2. Use fake answer's embedding to search for real documents
            3. Retrieve documents similar to the ideal answer
            4. Generate final answer from retrieved documents
            
            **Advantages:**
            - Better semantic matching
            - Finds documents even with different terminology
            - Improves retrieval quality
            """)
            
            st.divider()
            
            # Architecture Comparison
            st.subheader("📊 Architecture Comparison")
            
            comparison_data = {
                "Pattern": ["Naive RAG", "Agentic RAG", "Corrective RAG", "HyDE"],
                "Complexity": ["Low", "High", "Medium", "Medium"],
                "Query Refinement": ["No", "Yes", "No", "Yes"],
                "Quality Checks": ["No", "No", "Yes", "No"],
                "Multi-turn": ["No", "Yes", "No", "No"],
                "Best For": [
                    "Simple Q&A",
                    "Complex queries",
                    "High accuracy needs",
                    "Better retrieval"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        with sub_tab3:
            st.subheader("3. Chunking Strategies & Embeddings")
            
            st.markdown("""
            ### Why Chunking Matters
            
            > **"How you break up data (chunking) is often more important than the model itself."**
            
            Chunking determines:
            - What context the LLM receives
            - How well semantic search works
            - Quality of retrieved information
            """)
            
            st.divider()
            
            # Chunking Strategy 1: Fixed-Size
            st.markdown("""
            ### Strategy 1: Fixed-Size Chunking
            
            **How it works:**
            - Splits text every N tokens or characters (e.g., 512 tokens)
            - Simple and predictable
            
            **Pros:**
            - Easy to implement
            - Consistent chunk sizes
            - Fast processing
            
            **Cons:**
            - Can cut off sentences mid-thought
            - May split related concepts
            - No semantic awareness
            """)
            
            # Interactive Demo: Fixed-Size Chunking
            if st.session_state.openai_client:
                st.markdown("#### 🎮 Try Fixed-Size Chunking")
                sample_text = st.text_area(
                    "Enter text to chunk:",
                    value="This is a sample document. It contains multiple sentences. Each sentence has meaning. We want to split this into chunks. Fixed-size chunking will split at character boundaries.",
                    height=100,
                    key="fixed_chunk_text"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    chunk_size = st.slider("Chunk Size (characters)", 50, 200, 100, key="fixed_chunk_size")
                with col2:
                    overlap = st.slider("Overlap (words)", 0, 20, 5, key="fixed_overlap")
                
                if st.button("Chunk Text (Fixed-Size)", key="fixed_chunk_btn"):
                    chunks = chunk_text_fixed_size(sample_text, chunk_size, overlap)
                    st.success(f"Created {len(chunks)} chunks")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
                            st.text(chunk)
            
            st.divider()
            
            # Chunking Strategy 2: Recursive
            st.markdown("""
            ### Strategy 2: Recursive Character Splitting
            
            **How it works:**
            - Tries to split at natural boundaries (paragraphs, newlines, sentences)
            - Recursively splits if chunk is still too large
            - Respects document structure
            
            **Pros:**
            - Preserves sentence/paragraph boundaries
            - More semantic coherence
            - Better for structured documents
            
            **Cons:**
            - More complex implementation
            - Variable chunk sizes
            - May still split related content
            """)
            
            # Interactive Demo: Recursive Chunking
            if st.session_state.openai_client:
                st.markdown("#### 🎮 Try Recursive Chunking")
                recursive_text = st.text_area(
                    "Enter text to chunk:",
                    value="Paragraph 1: This is the first paragraph. It has multiple sentences.\n\nParagraph 2: This is the second paragraph. It also has sentences.\n\nParagraph 3: Final paragraph with more content.",
                    height=100,
                    key="recursive_chunk_text"
                )
                
                max_size = st.slider("Max Chunk Size (characters)", 50, 200, 100, key="recursive_max_size")
                
                if st.button("Chunk Text (Recursive)", key="recursive_chunk_btn"):
                    chunks = chunk_text_recursive(recursive_text, max_size)
                    st.success(f"Created {len(chunks)} chunks")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
                            st.text(chunk)
            
            st.divider()
            
            # Chunking Strategy 3: Semantic
            st.markdown("""
            ### Strategy 3: Semantic Chunking
            
            **How it works:**
            - Uses LLM or embedding model to find "meaning boundaries"
            - Splits where topics change
            - Most sophisticated approach
            
            **Pros:**
            - Preserves semantic coherence
            - Best for complex documents
            - Topic-aware splitting
            
            **Cons:**
            - Requires embedding model
            - More expensive (API calls)
            - Slower processing
            """)
            
            st.divider()
            
            # Contextual Embeddings
            st.markdown("""
            ### Contextual Embeddings
            
            **Adding Global Metadata:**
            - Include document title, summary, or category with every chunk
            - Helps retriever understand broader context
            - Improves search quality
            
            **Example:**
            ```
            Chunk: "The revenue increased by 20%..."
            With Context: "[Financial Report Q4 2024] The revenue increased by 20%..."
            ```
            
            This helps the retriever know this chunk is from a financial report.
            """)
    
    # ========================================================================
    # Tab 3: Module 2.2 - Semantic Search & Embeddings
    # ========================================================================
    
    with tab3:
        st.header("🔍 Module 2.2: Semantic Search, Embeddings, and Vector Databases")
        
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "Semantic Search",
            "Embeddings",
            "Vector Databases",
            "Search Mechanisms"
        ])
        
        with sub_tab1:
            st.subheader("1. What is Semantic Search?")
            
            st.markdown("""
            ### Traditional Keyword Search vs Semantic Search
            
            **Traditional Search (Keyword):**
            - Looks for exact character matches
            - Example: Searching "dog" won't find "puppy"
            - Limited to exact matches
            
            **Semantic Search:**
            - Focuses on understanding intent and contextual meaning
            - Asks: "Is this document about cats?" not "Does this contain 'cat'?"
            - Handles synonyms, natural language, and context
            """)
            
            st.divider()
            
            # Interactive Demo: Keyword vs Semantic
            if st.session_state.openai_client and st.session_state.lancedb_db:
                st.subheader("🎮 Interactive Demo: Keyword vs Semantic Search")
                
                # Sample documents
                sample_docs = [
                    "The company's revenue increased significantly this quarter.",
                    "Our financial performance showed strong growth in Q4.",
                    "The dog ran in the park.",
                    "The canine played outside."
                ]
                
                query = st.text_input("Enter search query:", value="company earnings", key="semantic_query")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔤 Keyword Search Results")
                    if st.button("Search (Keyword)", key="keyword_search_btn"):
                        # Simple keyword matching
                        results = []
                        query_lower = query.lower()
                        for doc in sample_docs:
                            if query_lower in doc.lower():
                                results.append(doc)
                        
                        if results:
                            for i, result in enumerate(results, 1):
                                st.write(f"{i}. {result}")
                        else:
                            st.info("No exact matches found")
                
                with col2:
                    st.markdown("### 🧠 Semantic Search Results")
                    if st.button("Search (Semantic)", key="semantic_search_btn"):
                        try:
                            # Get query embedding
                            query_embedding = get_embedding(query, st.session_state.openai_client, st.session_state.embeddings_model)
                            
                            if query_embedding:
                                # Get document embeddings
                                doc_embeddings = []
                                for doc in sample_docs:
                                    emb = get_embedding(doc, st.session_state.openai_client, st.session_state.embeddings_model)
                                    if emb:
                                        doc_embeddings.append((doc, emb))
                                
                                # Calculate similarities (cosine similarity)
                                similarities = []
                                for doc, emb in doc_embeddings:
                                    # Cosine similarity
                                    dot_product = sum(a * b for a, b in zip(query_embedding, emb))
                                    norm_a = sum(a * a for a in query_embedding) ** 0.5
                                    norm_b = sum(b * b for b in emb) ** 0.5
                                    similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
                                    similarities.append((doc, similarity))
                                
                                # Sort by similarity
                                similarities.sort(key=lambda x: x[1], reverse=True)
                                
                                for i, (doc, sim) in enumerate(similarities, 1):
                                    st.write(f"{i}. {doc} (similarity: {sim:.3f})")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with sub_tab2:
            st.subheader("2. The Engine: How Embeddings Work")
            
            st.markdown("""
            ### What are Embeddings?
            
            Embeddings convert unstructured data (text, images, audio) into vectors (lists of numbers).
            
            **Key Concepts:**
            - **Mapping Meaning**: ML models place similar concepts close together in embedding space
            - **Example**: "king" is closer to "queen" than to "apple" in the vector space
            - **Contextual Awareness**: Modern models understand context (e.g., "bank" in "river bank" vs "investment bank")
            """)
            
            st.divider()
            
            # Embedding Dimensions
            st.markdown("""
            ### 3. Understanding Embedding Dimensions
            
            **Small Dimensions (300-512):**
            - ✅ Faster searches
            - ✅ Lower memory usage
            - ✅ Lower storage costs
            - ❌ May lose fine-grained nuance
            
            **Large Dimensions (1536-3072):**
            - ✅ Captures detailed relationships
            - ✅ Better for complex datasets
            - ❌ Higher computational cost
            - ❌ Slower query latency
            - ❌ Higher storage costs
            
            **Trade-off**: Doubling dimensions typically doubles memory and compute.
            """)
            
            # Interactive Demo: Embedding Visualization
            if st.session_state.openai_client:
                st.subheader("🎮 Interactive Demo: Generate Embeddings")
                
                text_input = st.text_area(
                    "Enter text to embed:",
                    value="The quick brown fox jumps over the lazy dog",
                    height=100,
                    key="embedding_text"
                )
                
                if st.button("Generate Embedding", key="generate_embedding_btn"):
                    try:
                        embedding = get_embedding(text_input, st.session_state.openai_client, st.session_state.embeddings_model)
                        if embedding:
                            st.success(f"✅ Generated embedding with {len(embedding)} dimensions")
                            
                            # Show first 10 dimensions
                            with st.expander("View Embedding Vector (first 20 dimensions)"):
                                st.code(str(embedding[:20]))
                            
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Dimensions", len(embedding))
                            with col2:
                                st.metric("Min Value", f"{min(embedding):.4f}")
                            with col3:
                                st.metric("Max Value", f"{max(embedding):.4f}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with sub_tab3:
            st.subheader("4. What is a Vector Database (VectorDB)?")
            
            st.markdown("""
            ### Vector Database vs Traditional Database
            
            **Traditional Database (SQL):**
            - Great for: "price < 100"
            - Structured queries
            - Exact matches
            
            **Vector Database:**
            - Purpose-built for high-dimensional vectors
            - Finds "nearest neighbors" to query vector
            - Semantic similarity search
            """)
            
            st.markdown("""
            ### Key Features
            
            **Fast Retrieval:**
            - Uses algorithms like HNSW (Hierarchical Navigable Small World)
            - Navigates vector space without checking every item
            - Sub-second search on billions of vectors
            
            **Metadata Filtering:**
            - Combines semantic search with traditional filters
            - Example: "Find documents about 'climate change' AND 'published in 2024'"
            """)
            
            st.divider()
            
            # LanceDB Demo
            if st.session_state.lancedb_db:
                st.subheader("🎮 Interactive Demo: LanceDB Vector Search")
                
                # Create or load table
                table_name = "rag_demo_table"
                
                if st.button("Initialize Demo Table", key="init_table_btn"):
                    try:
                        # Sample documents
                        demo_docs = [
                            {"text": "Python is a programming language", "category": "programming"},
                            {"text": "Machine learning uses algorithms", "category": "AI"},
                            {"text": "Databases store structured data", "category": "database"},
                            {"text": "Vector databases enable semantic search", "category": "database"}
                        ]
                        
                        if st.session_state.openai_client:
                            # Generate embeddings
                            data = []
                            for doc in demo_docs:
                                embedding = get_embedding(
                                    doc["text"],
                                    st.session_state.openai_client,
                                    st.session_state.embeddings_model
                                )
                                if embedding:
                                    data.append({
                                        "id": str(uuid.uuid4()),
                                        "text": doc["text"],
                                        "category": doc["category"],
                                        "vector": embedding
                                    })
                            
                            if data:
                                # Create table
                                schema = pa.schema([
                                    pa.field("id", pa.string()),
                                    pa.field("text", pa.string()),
                                    pa.field("category", pa.string()),
                                    pa.field("vector", pa.list_(pa.float32(), len(data[0]["vector"])))
                                ])
                                
                                table = pa.Table.from_pylist(data, schema=schema)
                                st.session_state.lancedb_db.create_table(table_name, table, mode="overwrite")
                                st.success(f"✅ Created table '{table_name}' with {len(data)} documents")
                        else:
                            st.warning("⚠️ OpenAI API key required to generate embeddings")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                # Search
                search_query = st.text_input("Search query:", value="programming languages", key="vectordb_query")
                
                if st.button("Search Vector Database", key="vectordb_search_btn"):
                    try:
                        if st.session_state.openai_client:
                            # Get query embedding
                            query_embedding = get_embedding(
                                search_query,
                                st.session_state.openai_client,
                                st.session_state.embeddings_model
                            )
                            
                            if query_embedding:
                                # Search
                                table = st.session_state.lancedb_db.open_table(table_name)
                                results = table.search(query_embedding).limit(3).to_pandas()
                                
                                st.success(f"✅ Found {len(results)} results")
                                for idx, row in results.iterrows():
                                    with st.expander(f"Result {idx + 1} (distance: {row.get('_distance', 'N/A'):.4f})"):
                                        st.write(f"**Text:** {row['text']}")
                                        st.write(f"**Category:** {row['category']}")
                        else:
                            st.warning("⚠️ OpenAI API key required")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with sub_tab4:
            st.subheader("5. Search Mechanisms: Sparse vs Dense vs Hybrid")
            
            comparison_data = {
                "Feature": ["Method", "Strengths", "Weaknesses"],
                "Sparse (BM25)": [
                    "Exact keyword matching",
                    "Fast, handles jargon/acronyms perfectly",
                    "Fails on synonyms or context"
                ],
                "Dense (Embeddings)": [
                    "Semantic meaning/intent",
                    "Finds 'car' when you search 'vehicle'",
                    "Can be 'too fuzzy' and miss exact IDs"
                ],
                "Hybrid": [
                    "Combines Sparse + Dense",
                    "Best of both worlds",
                    "More complex implementation"
                ]
            }
            
            df_search = pd.DataFrame(comparison_data)
            st.dataframe(df_search, use_container_width=True, hide_index=True)
            
            st.markdown("""
            ### Hybrid Search
            
            **Industry Standard:**
            - Combines scores from both Sparse and Dense retrieval
            - Uses Reciprocal Rank Fusion (RRF) to merge results
            - Provides best of both worlds
            """)
    
    # ========================================================================
    # Tab 4: Module 2.3 - Tool Use & Agents
    # ========================================================================
    
    with tab4:
        st.header("🛠️ Module 2.3: Designing for Tool Use - The Agentic Contract")
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "Tool Definitions",
            "ReAct Pattern",
            "Best Practices"
        ])
        
        with sub_tab1:
            st.subheader("A. Semantic Tool Definitions")
            
            st.markdown("""
            ### The Tool as Interface Contract
            
            The LLM doesn't see the code; it only sees the tool's name and description.
            
            **❌ Bad Definition:**
            ```python
            def search(q): ...
            ```
            - Too generic
            - No context about when to use
            - Unclear purpose
            
            **✅ Good Definition:**
            ```python
            def financial_report_search(query: str):
                \"\"\"
                Searches the SEC 10-K database for financial risks.
                Use this only when asked about a company's fiscal health.
                \"\"\"
            ```
            - Clear purpose
            - Specific use case
            - Semantic description
            """)
            
            st.divider()
            
            # Interactive Demo: Tool Definition
            st.subheader("🎮 Interactive Demo: Tool Definition Quality")
            
            tool_examples = {
                "Bad": {
                    "name": "search",
                    "description": "Searches for stuff"
                },
                "Good": {
                    "name": "financial_report_search",
                    "description": "Searches the SEC 10-K database for financial risks. Use this only when asked about a company's fiscal health, revenue, or financial performance."
                }
            }
            
            selected_example = st.radio("Select example:", ["Bad", "Good"], key="tool_example")
            example = tool_examples[selected_example]
            
            st.code(f"""
Tool Name: {example['name']}
Description: {example['description']}
            """)
            
            if st.session_state.openai_client:
                test_query = st.text_input(
                    "Test query:",
                    value="What was NVIDIA's revenue in 2024?",
                    key="tool_test_query"
                )
                
                if st.button("Would LLM use this tool?", key="tool_test_btn"):
                    try:
                        prompt = f"""
You are an AI assistant with access to tools. Given this tool definition:

Tool: {example['name']}
Description: {example['description']}

User Query: {test_query}

Should you use this tool? Why or why not? Respond with:
1. Yes/No
2. Brief reasoning
                        """
                        
                        response = st.session_state.openai_client.chat.completions.create(
                            model=st.session_state.llm_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that decides when to use tools."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3
                        )
                        
                        answer = response.choices[0].message.content
                        st.info(answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with sub_tab2:
            st.subheader("B. The ReAct Pattern (Reason + Act)")
            
            st.markdown("""
            ### ReAct: Reasoning and Acting in Language Models
            
            Instead of linear "Retrieve → Generate," design your agent to iterate:
            
            **Pattern:**
            1. **Thought**: Agent reasons about what it needs
            2. **Action**: Calls appropriate tool
            3. **Observation**: Receives tool result
            4. **Final Response**: Generates answer based on observations
            """)
            
            st.divider()
            
            # ReAct Example
            st.markdown("""
            ### Example: ReAct in Action
            
            **User Query:** "What was NVIDIA's 2024 revenue?"
            
            **Agent Process:**
            
            1. **Thought**: "I need the 2024 revenue for NVIDIA. I should check the financial tool."
            
            2. **Action**: 
               ```python
               financial_report_search(query="NVIDIA 2024 revenue")
               ```
            
            3. **Observation**: 
               ```
               Tool returned: 2024 Revenue was $60.9B
               ```
            
            4. **Final Response**: 
               "NVIDIA's 2024 revenue was $60.9 billion."
            """)
            
            st.divider()
            
            # Interactive ReAct Demo
            if st.session_state.openai_client:
                st.subheader("🎮 Interactive Demo: ReAct Pattern")
                
                react_query = st.text_input(
                    "Enter a query that requires tool use:",
                    value="What was the weather in San Francisco yesterday?",
                    key="react_query"
                )
                
                if st.button("Run ReAct Agent", key="react_btn"):
                    try:
                        # Simulate ReAct pattern
                        st.markdown("### Agent Reasoning Process")
                        
                        # Step 1: Thought
                        thought_prompt = f"""
You are an AI agent. A user asked: "{react_query}"

What do you need to do to answer this? What tools might you need?
Respond with your reasoning.
                        """
                        
                        thought_response = st.session_state.openai_client.chat.completions.create(
                            model=st.session_state.llm_model,
                            messages=[
                                {"role": "system", "content": "You are a reasoning AI agent."},
                                {"role": "user", "content": thought_prompt}
                            ],
                            temperature=0.3
                        )
                        
                        thought = thought_response.choices[0].message.content
                        
                        with st.expander("🤔 Thought", expanded=True):
                            st.write(thought)
                        
                        # Step 2: Action (simulated)
                        st.markdown("### ⚡ Action")
                        st.info("Agent would call appropriate tool here (e.g., weather_api, financial_search)")
                        
                        # Step 3: Observation (simulated)
                        st.markdown("### 👁️ Observation")
                        st.info("Tool would return results here")
                        
                        # Step 4: Final Response
                        final_prompt = f"""
Based on this reasoning: "{thought}"

And assuming the tool returned relevant information, provide a final answer to: "{react_query}"
                        """
                        
                        final_response = st.session_state.openai_client.chat.completions.create(
                            model=st.session_state.llm_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=0.3
                        )
                        
                        final_answer = final_response.choices[0].message.content
                        
                        with st.expander("💬 Final Response", expanded=True):
                            st.success(final_answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with sub_tab3:
            st.subheader("3. Best Practices for RAG Tool Implementation")
            
            st.markdown("""
            ### 1. Structured Output (JSON/Pydantic)
            
            Force the agent to call tools using strict schemas.
            - Prevents "hallucinated parameters"
            - Prevents API crashes
            - Ensures type safety
            """)
            
            st.code("""
# Example: Structured tool call
{
    "tool_name": "financial_report_search",
    "parameters": {
        "query": "NVIDIA 2024 revenue",
        "company": "NVIDIA",
        "year": 2024
    }
}
            """)
            
            st.divider()
            
            st.markdown("""
            ### 2. Tool Sandboxing
            
            Never let an agent execute raw Python or SQL in production without:
            - Restricted execution environment
            - Human-in-the-loop (HITL) approval
            - Input validation
            - Output sanitization
            """)
            
            st.divider()
            
            st.markdown("""
            ### 3. Graceful Failure Handling
            
            If a tool returns an error:
            - Agent should "Reflect" on the error
            - Try a different tool or search query
            - Don't just crash
            - Provide helpful error messages
            """)
            
            st.divider()
            
            st.markdown("""
            ### 4. Retrieval as a Modular Tool
            
            Treat your entire RAG pipeline as a single tool:
            - Tool name: `knowledge_base_search`
            - Allows agent to decide NOT to search if user just says "Hello"
            - Modular and reusable
            """)
            
            st.divider()
            
            st.markdown("""
            ### 5. RAG Evaluation Metrics (RAGAS Framework)
            
            **Faithfulness:**
            - Does the answer stay true to retrieved documents?
            - No hallucinations
            
            **Answer Relevance:**
            - Does the response directly address the user's intent?
            
            **Context Precision:**
            - Were the most relevant documents ranked at the top?
            """)
    
    # ========================================================================
    # Tab 5: Module 2.4 - Multi-Modal Lakehouse
    # ========================================================================
    
    with tab5:
        st.header("🏗️ Module 2.4: The Multi-Modal Lakehouse Pattern")
        
        st.markdown("""
        ### The Synchronization Gap Problem
        
        **Naive RAG Issue:**
        - Embeddings live in vector store
        - Original data lives in data lake/lakehouse
        - Risk of data getting "out of sync"
        - Requires "glue code" to join them
        """)
        
        st.divider()
        
        st.markdown("""
        ### The Multi-Modal Lakehouse Solution
        
        **How it Works:**
        - Multi-modal store (like LanceDB) stores:
          - Vectors (embeddings)
          - Raw data (text, images, JSON)
          - Metadata (from lakehouse)
        - All in the same columnar files
        - Single source of truth
        """)
        
        st.divider()
        
        st.markdown("""
        ### Using LanceDB as a Multi-Modal Store
        
        **Key Features:**
        - Built on Lance columnar format
        - Designed for AI workloads
        - Fast random access to large blobs
        - Schema-first design (Pydantic/Apache Arrow)
        - Disk-based indexing (IVF-PQ)
        - Can search billions of vectors without massive RAM
        """)
        
        st.divider()
        
        # Schema Example
        st.subheader("📋 Schema Example")
        
        st.code("""
# Single row contains:
{
    "vector": [0.1, 0.2, ..., 0.9],  # 1536-dim embedding
    "content": "Raw text or reference to image blob",
    "metadata": {
        "user_id": "12345",
        "timestamp": "2024-01-15",
        "security_clearance": "public"
    }
}
        """)
        
        st.divider()
        
        # Workflow
        st.subheader("🔄 Workflow: Integrating Lakehouse Metadata & Embeddings")
        
        st.markdown("""
        **Pipeline:**
        
        1. **Ingestion from Lakehouse:**
           - Pull snapshot/stream from lakehouse
           - Example: Customer support tickets table
        
        2. **Multimodal Vectorization:**
           - Text → vectors (standard model)
           - Images/Charts → described by Vision LLM or embedded via CLIP
        
        3. **Storage in LanceDB:**
           - Embeddings + original data + metadata
           - Written to .lance file (S3 or local)
        
        4. **Hybrid Retrieval:**
           - Semantic Search: "Find tickets similar to this issue"
           - Metadata Filter: WHERE status = 'open' AND priority = 'high'
        """)
        
        st.divider()
        
        # Comparison Table
        st.subheader("📊 Technical Comparison")
        
        comparison_lakehouse = {
            "Feature": [
                "Data Storage",
                "Consistency",
                "Compute/Storage",
                "Complex Joins"
            ],
            "Specialized Vector DB (Pinecone)": [
                "Embeddings + Pointers to data",
                "High risk of 'Out of Sync' data",
                "Often coupled; expensive to scale",
                "Requires external orchestration"
            ],
            "Multi-Modal Lakehouse (LanceDB)": [
                "Unified: Embeddings + Raw Data + Metadata",
                "ACID-compliant: Single source of truth",
                "Decoupled: Store on S3, compute on-demand",
                "Native SQL joins with metadata"
            ]
        }
        
        df_lakehouse = pd.DataFrame(comparison_lakehouse)
        st.dataframe(df_lakehouse, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Best Practices
        st.subheader("✅ Best Practices for Implementation")
        
        st.markdown("""
        1. **Columnar Evolution:**
           - Don't be afraid to add columns as agent evolves
           - LanceDB allows adding new columns without rewriting entire table
        
        2. **Version Control:**
           - Leverage built-in table versioning
           - "Time travel" to see why agent gave specific answer yesterday vs today
        
        3. **Local-to-Cloud:**
           - Start with embedded LanceDB (like SQLite) for local dev
           - Point same code at S3 for production scale
        """)
    
    # ========================================================================
    # Tab 6: Interactive Playground
    # ========================================================================
    
    with tab6:
        st.header("🎮 Interactive RAG Playground")
        
        st.markdown("""
        Experiment with RAG concepts in this interactive playground.
        Upload documents, generate embeddings, and query with RAG.
        """)
        
        # Debug info
        debug_info = st.checkbox("Show Debug Info", key="playground_debug", value=False)
        if debug_info:
            st.write("**Debug Information:**")
            st.write(f"- OpenAI client: {st.session_state.openai_client is not None}")
            st.write(f"- LanceDB DB: {st.session_state.lancedb_db is not None}")
            st.write(f"- LanceDB path: {st.session_state.get('lancedb_path', 'Not set')}")
            st.write(f"- LANCEDB_AVAILABLE: {LANCEDB_AVAILABLE}")
        
        if not st.session_state.openai_client:
            st.warning("⚠️ Please enter OpenAI API key in sidebar to use the playground")
        elif st.session_state.lancedb_db is None:
            st.warning("⚠️ Please configure LanceDB path in sidebar and ensure it's connected")
            st.info(f"💡 Current path: {st.session_state.get('lancedb_path', 'Not set')}")
            st.info("💡 Make sure LanceDB is installed: `pip install lancedb`")
            st.info("💡 Try refreshing the page after setting the path")
        else:
            # Document Upload
            st.subheader("📄 Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Upload text files",
                type=["txt", "md"],
                accept_multiple_files=True,
                help="Upload documents to create a knowledge base"
            )
            
            manual_text = st.text_area(
                "Or enter text manually:",
                height=200,
                help="Enter text to add to knowledge base"
            )
            
            if st.button("Add Documents to Knowledge Base", key="add_docs_btn"):
                try:
                    documents = []
                    
                    # Process uploaded files
                    if uploaded_files:
                        for file in uploaded_files:
                            text = file.read().decode('utf-8')
                            documents.append({
                                "filename": file.name,
                                "text": text,
                                "source": "uploaded_file"
                            })
                    
                    # Process manual text
                    if manual_text:
                        documents.append({
                            "filename": "manual_input.txt",
                            "text": manual_text,
                            "source": "manual"
                        })
                    
                    if documents:
                        # Chunk documents
                        all_chunks = []
                        for doc in documents:
                            chunks = chunk_text_recursive(doc["text"], max_size=512)
                            for i, chunk in enumerate(chunks):
                                all_chunks.append({
                                    "id": str(uuid.uuid4()),
                                    "chunk_index": i,
                                    "text": chunk,
                                    "filename": doc["filename"],
                                    "source": doc["source"]
                                })
                        
                        # Generate embeddings
                        data = []
                        progress_bar = st.progress(0)
                        total_chunks = len(all_chunks)
                        
                        for idx, chunk in enumerate(all_chunks):
                            embedding = get_embedding(
                                chunk["text"],
                                st.session_state.openai_client,
                                st.session_state.embeddings_model
                            )
                            
                            if embedding:
                                data.append({
                                    "id": chunk["id"],
                                    "text": chunk["text"],
                                    "filename": chunk["filename"],
                                    "source": chunk["source"],
                                    "chunk_index": chunk["chunk_index"],
                                    "vector": embedding
                                })
                            
                            progress_bar.progress((idx + 1) / total_chunks)
                        
                        progress_bar.empty()
                        
                        if data:
                            # Store in LanceDB
                            schema = pa.schema([
                                pa.field("id", pa.string()),
                                pa.field("text", pa.string()),
                                pa.field("filename", pa.string()),
                                pa.field("source", pa.string()),
                                pa.field("chunk_index", pa.int32()),
                                pa.field("vector", pa.list_(pa.float32(), len(data[0]["vector"])))
                            ])
                            
                            table = pa.Table.from_pylist(data, schema=schema)
                            st.session_state.lancedb_db.create_table("playground_kb", table, mode="overwrite")
                            
                            st.success(f"✅ Added {len(data)} chunks to knowledge base")
                            st.session_state.documents = data
                        else:
                            st.error("Failed to generate embeddings")
                    else:
                        st.warning("No documents to add")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.divider()
            
            # RAG Query
            st.subheader("🔍 Query with RAG")
            
            rag_query = st.text_input(
                "Enter your question:",
                placeholder="Ask a question about the uploaded documents",
                key="playground_query"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Top K results", 1, 10, 3, key="playground_top_k")
            with col2:
                use_rag = st.checkbox("Use RAG (augment prompt)", value=True, key="playground_rag")
            
            if st.button("Query Knowledge Base", key="playground_query_btn"):
                if rag_query:
                    try:
                        # Step 1: Retrieve
                        query_embedding = get_embedding(
                            rag_query,
                            st.session_state.openai_client,
                            st.session_state.embeddings_model
                        )
                        
                        if query_embedding:
                            table = st.session_state.lancedb_db.open_table("playground_kb")
                            results = table.search(query_embedding).limit(top_k).to_pandas()
                            
                            st.success(f"✅ Retrieved {len(results)} relevant chunks")
                            
                            # Display retrieved chunks
                            with st.expander("📄 Retrieved Context", expanded=True):
                                context_parts = []
                                for idx, row in results.iterrows():
                                    st.markdown(f"**Chunk {idx + 1}** (from {row['filename']})")
                                    st.text(row['text'])
                                    st.divider()
                                    context_parts.append(row['text'])
                                
                                full_context = "\n\n".join(context_parts)
                            
                            # Step 2: Generate with RAG
                            if use_rag:
                                st.markdown("### 🤖 RAG-Generated Answer")
                                
                                rag_prompt = f"""
You are a helpful assistant. Answer the user's question based ONLY on the provided context.

Context:
{full_context}

User Question: {rag_query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite which document/chunk you used for your answer
- Be concise and accurate
                                """
                                
                                response = st.session_state.openai_client.chat.completions.create(
                                    model=st.session_state.llm_model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                                        {"role": "user", "content": rag_prompt}
                                    ],
                                    temperature=0.3
                                )
                                
                                answer = response.choices[0].message.content
                                st.success(answer)
                            else:
                                st.info("💡 Enable 'Use RAG' to generate answer with context")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a query")
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📚 Resources
    - [LanceDB Documentation](https://lancedb.github.io/lancedb/)
    - [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
    - [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
    """)

if __name__ == "__main__":
    main()
