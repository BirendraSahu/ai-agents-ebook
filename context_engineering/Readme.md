# Context Engineering Demo

Interactive [Streamlit](https://streamlit.io/) app that demonstrates **context engineering**: assembling, ranking, pruning, and compressing information before it reaches an LLM. The demo uses a travel-assistant scenario (hotel and flight preferences) and the OpenAI API.

**Script:** `context_engineering_demo.py`

## Prerequisites

- Python 3.10+ recommended
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o-mini`

## Install

From the directory that contains this README (the `src` folder in this repo):

```bash
pip install -r requirements_context_engineering.txt
```

Dependencies: `streamlit`, `openai`, `pandas`.

## Run

```bash
streamlit run context_engineering_demo.py
```

The app opens in your browser (default [http://localhost:8501](http://localhost:8501)).

## Configuration

1. In the sidebar, paste your API key or rely on the `OPENAI_API_KEY` environment variable (the key field is pre-filled from the env var when set).
2. Click **Connect to OpenAI**.
3. For demos that need sample data, click **Initialize Sample Memories** (or use tabs that auto-initialize memories when empty).

Optional: **Clear Cache** resets the in-session semantic cache.

## What the app covers

| Tab | Purpose |
|-----|---------|
| **Concepts Overview** | Static reference: prompt vs context engineering, context composition, window management, ranking/pruning, and context poisoning (no API calls). |
| **Context Assembly Pipeline** | End-to-end pipeline: retrieve memories → LLM-based ranking → category/score pruning → optional summarization. Shows metrics and assembled context. |
| **Hotel Booking Demo** | Compares answers **with** assembled (hotel-focused) context vs **without** context to show personalization and pruning (flight memories dropped for hotel queries). |
| **Context Management** | Semantic cache (exact hash + simple substring match), context compression vs original token estimate, and a two-stage “broad retrieval + re-rank” walkthrough. |
| **Advanced Techniques** | Token budget split calculator, LLM-based “poisoning” check on arbitrary context text, and a small JSON knowledge-graph illustration. |

## Context assembly pipeline (code)

The main flow is implemented in `assemble_context()`:

1. **Retrieval** — Start from all in-memory “user memories.”
2. **Ranking** — `rank_context_with_openai()` asks `gpt-4o-mini` (JSON mode) for relevance scores and reasons; optional.
3. **Pruning** — `prune_context()` keeps items matching the query category or above a relevance threshold; optional.
4. **Cap** — Truncate to `max_items`.
5. **Summarization** — Optional `summarize_context_with_openai()` over the final items.

Responses to the user in the booking demo use `query_with_context()`, which injects assembled preference lines into the travel-assistant prompt.

## Models and costs

- Default chat model: **`gpt-4o-mini`** for ranking, summarization, final answers, security check, etc.
- Each tab that calls the API incurs usage; ranking and multi-step flows multiply calls.

## Implementation notes

- **Token estimate:** `estimate_tokens()` uses a rough heuristic (~4 characters per token), not the official tokenizer.
- **Semantic cache:** `check_semantic_cache()` uses MD5 of the normalized query plus substring overlap against cached keys—not embeddings.
- **Sample data:** Memories are synthetic (hotel/flight preferences) for demonstration only.

## Troubleshooting

- **“Please connect to OpenAI in the sidebar”** — Connect with a valid key first.
- **Ranking or JSON errors** — The app falls back to unranked slices where implemented; check API errors in the UI warnings.
