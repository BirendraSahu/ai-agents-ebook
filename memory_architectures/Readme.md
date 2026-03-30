# Memory Architectures Demo

Streamlit app that demonstrates **multi-layer memory** for persistent agents: working (short-term) context, long-term typed memories (episodic, semantic, procedural), optional **Mem0** vector search, and supporting patterns (recursive summaries, decay/pruning, token-aware injection, provenance logs). It uses the OpenAI API for extraction, embeddings, chat, and summarization.

**Script:** `memory_architectures_demo.py`

## Prerequisites

- Python 3.10+ recommended
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to:
  - `gpt-4o-mini` (chat, JSON extraction, summaries)
  - `text-embedding-3-small` (embeddings for local relevance scoring)
- **Mem0** is optional: install `mem0ai` and enable the sidebar toggle to use Chroma-backed Mem0 alongside the local JSON store.

## Install

From the directory that contains this README:

```bash
pip install -r requirements_memory_architectures_demo.txt
```

Declared dependencies: `streamlit`, `openai`, `numpy`, `pandas`, `mem0ai`.

## Run

```bash
streamlit run memory_architectures_demo.py
```

Default URL: [http://localhost:8501](http://localhost:8501).

## Configuration (sidebar)

| Control | Purpose |
|--------|---------|
| **User ID** | Namespace for stored memories (default `demo_user_001`). |
| **OpenAI API Key** | Required for LLM, embeddings, and hierarchical summaries; paste in the UI. |
| **Recall Top-K** | How many local memories to score and return for retrieval (1–15). |
| **Working Memory Turns** | Max chat turns kept in session working memory (4–20). |
| **Pruning Threshold** | Health threshold for decay/pruning (see Maintenance tab). |
| **Enable Mem0 (if installed)** | When on and Mem0 initializes, writes and searches also go through Mem0 (Chroma under `.mem0_store`). |

The footer shows whether OpenAI loaded, whether the Mem0 package is importable, and whether Mem0 is actually active.

## On-disk artifacts

| Path | Role |
|------|------|
| `.memory_agent_demo_store.json` | Local long-term store (append/update/delete `MemoryItem` records). |
| `.mem0_store/` | Created when Mem0 uses local Chroma (if enabled and working). |

## Tabs

| Tab | What it does |
|-----|----------------|
| **Architecture** | Explains working vs episodic vs semantic vs procedural memory; metrics and table of all stored memories for the current user. |
| **Health Coach Agent** | Main interactive loop: send messages or run a **Scripted Session 1→5** demo (peanut allergy + noise + procedural + recipe). Extracts memories via `gpt-4o-mini` (or heuristic fallback), recalls with embedding similarity + importance/recency/access, updates tiered summaries, answers as a health coach. Includes semantic cache (normalized exact key), **Recall Preview** (local + optional Mem0 JSON). |
| **Maintenance** | **Decay + pruning** by a composite “health” score; **episodic → semantic** consolidation via one LLM-derived rule; **Refresh** tier-1/2/3 recursive summaries from working history. |
| **Token Optimization** | Demo-only pipeline: de-duplicate lines, **lost-in-the-middle** style reordering, **selective injection** under a token budget (~250 tokens in the example). |
| **Enterprise Logs** | Contrasts volatile **working history** with append-only **provenance** entries (query, cache hit, recalled IDs); export as JSON. |

## Key behaviors (implementation)

- **Memory extraction:** `extract_candidate_memories()` returns JSON from the model, or keyword heuristics if the client is missing or the call fails.
- **Local recall:** `retrieve_relevant_memories()` embeds the query and each memory, then scores with cosine similarity, importance, recency, and access frequency.
- **Summaries:** `update_recursive_summaries()` builds session → epoch → “master” profile text for injection into replies.
- **Token estimate:** `rough_tokens()` uses length / 4 (approximate, not `tiktoken`).

## Troubleshooting

- **No API features** — Enter an OpenAI key in the sidebar; without it, some paths use heuristics or short fallbacks only.
- **Mem0 inactive** — Check footer flags; ensure `mem0ai` is installed, toggle is on, and Chroma/OpenAI config succeeds (see `Mem0Client` in the script).
- **Empty recall** — Add memories first (send messages or run the scripted demo); confirm **User ID** matches the data you expect.

## Security note

This demo persists data in a local JSON file and optional local vector store. Do not use real PHI or production secrets in shared environments.
