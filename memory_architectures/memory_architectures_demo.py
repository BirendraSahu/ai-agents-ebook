#!/usr/bin/env python3
"""
Streamlit demo for persistent agent memory architectures using OpenAI + Mem0 (optional).
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except Exception:
    MEM0_AVAILABLE = False


APP_TITLE = "Module 3 Demo: Memory Architectures for Persistent Agents"
STORE_PATH = Path(".memory_agent_demo_store.json")


@dataclass
class MemoryItem:
    memory_id: str
    user_id: str
    memory_type: str  # episodic | semantic | procedural
    content: str
    importance: int
    created_at: str
    last_accessed_at: str
    access_count: int
    source: str  # user | system | consolidation
    metadata: Dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rough_tokens(text: str) -> int:
    # Lightweight token approximation for demo economics.
    return max(1, int(len(text) / 4))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class LocalMemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {"memories": []}
        return {"memories": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def add_memory(self, item: MemoryItem) -> None:
        self.data["memories"].append(asdict(item))
        self._save()

    def list_memories(self, user_id: str) -> List[Dict[str, Any]]:
        return [m for m in self.data.get("memories", []) if m.get("user_id") == user_id]

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        for mem in self.data.get("memories", []):
            if mem.get("memory_id") == memory_id:
                mem.update(updates)
                break
        self._save()

    def delete_memory(self, memory_id: str) -> None:
        self.data["memories"] = [m for m in self.data.get("memories", []) if m.get("memory_id") != memory_id]
        self._save()


class Mem0Client:
    """Optional Mem0 wrapper. Uses local store fallback when unavailable or errors."""

    def __init__(self, openai_key: str):
        self.client = None
        self.available = False
        if not MEM0_AVAILABLE:
            return
        try:
            cfg = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {"collection_name": "memory_architectures_demo", "path": ".mem0_store"},
                },
                "llm": {
                    "provider": "openai",
                    "config": {"model": "gpt-4o-mini", "api_key": openai_key},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small", "api_key": openai_key},
                },
            }
            self.client = Memory.from_config(cfg)
            self.available = True
        except Exception:
            self.available = False

    def add(self, text: str, user_id: str, metadata: Dict[str, Any]) -> Optional[str]:
        if not self.available:
            return None
        try:
            res = self.client.add(text, user_id=user_id, metadata=metadata)
            if isinstance(res, dict):
                return str(res.get("id") or res.get("memory_id") or "")
            return str(res)
        except Exception:
            return None

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            res = self.client.search(query=query, user_id=user_id, limit=limit)
            if isinstance(res, dict):
                return res.get("results", [])
            if isinstance(res, list):
                return res
            return []
        except Exception:
            return []


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_embedding(client: Optional[OpenAI], text: str) -> np.ndarray:
    if not client or not text.strip():
        return np.array([])
    try:
        out = client.embeddings.create(model="text-embedding-3-small", input=text[:8000])
        return np.array(out.data[0].embedding, dtype=float)
    except Exception:
        return np.array([])


def extract_candidate_memories(client: Optional[OpenAI], user_text: str) -> List[Dict[str, Any]]:
    """Extract memory candidates with type and importance; fallback heuristic if model unavailable."""
    fallback = []
    text = user_text.lower()
    if "allergic" in text or "allergy" in text:
        fallback.append({
            "memory_type": "semantic",
            "content": user_text,
            "importance": 10,
            "source": "user",
        })
    elif "always" in text or "prefer" in text:
        fallback.append({
            "memory_type": "semantic",
            "content": user_text,
            "importance": 8,
            "source": "user",
        })
    elif "how to" in text or "steps" in text:
        fallback.append({
            "memory_type": "procedural",
            "content": user_text,
            "importance": 7,
            "source": "user",
        })
    else:
        fallback.append({
            "memory_type": "episodic",
            "content": user_text,
            "importance": 4,
            "source": "user",
        })

    if not client:
        return fallback

    prompt = f"""
Extract durable memories from this user message and return strict JSON:
{{
  "memories": [
    {{
      "memory_type": "episodic|semantic|procedural",
      "content": "string",
      "importance": 1-10,
      "source": "user"
    }}
  ]
}}
Message: {user_text}
Rules:
- Use semantic for stable facts/preferences.
- Use episodic for one-time events.
- Use procedural for repeatable steps/how-to.
- Ignore low-value noise.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(res.choices[0].message.content)
        memories = parsed.get("memories", [])
        cleaned = []
        for m in memories:
            cleaned.append({
                "memory_type": m.get("memory_type", "episodic"),
                "content": (m.get("content") or "").strip(),
                "importance": int(m.get("importance", 5)),
                "source": "user",
            })
        return [m for m in cleaned if m["content"]]
    except Exception:
        return fallback


def relevance_score(query: str, mem: Dict[str, Any], query_emb: np.ndarray, mem_emb: np.ndarray) -> float:
    sim = cosine_sim(query_emb, mem_emb)
    age_hours = max(0.1, (datetime.now(timezone.utc) - datetime.fromisoformat(mem["created_at"])).total_seconds() / 3600.0)
    recency = 1.0 / (1.0 + age_hours / 24.0)
    importance = float(mem.get("importance", 5)) / 10.0
    access = min(1.0, float(mem.get("access_count", 0)) / 8.0)
    # Combined score for recall decision.
    return 0.5 * sim + 0.25 * importance + 0.2 * recency + 0.05 * access


def retrieve_relevant_memories(
    user_id: str,
    query: str,
    local_store: LocalMemoryStore,
    openai_client: Optional[OpenAI],
    top_k: int,
) -> List[Dict[str, Any]]:
    query_emb = get_embedding(openai_client, query)
    memories = local_store.list_memories(user_id)
    scored = []
    for m in memories:
        mem_emb = get_embedding(openai_client, m.get("content", ""))
        score = relevance_score(query, m, query_emb, mem_emb)
        item = dict(m)
        item["score"] = score
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def selective_injection(context_items: List[str], max_tokens: int = 1200) -> List[str]:
    must_have = context_items[:2]
    optional = context_items[2:]
    chosen = list(must_have)
    used = sum(rough_tokens(x) for x in chosen)
    for chunk in optional:
        t = rough_tokens(chunk)
        if used + t <= max_tokens:
            chosen.append(chunk)
            used += t
    return chosen


def reorder_docs_lost_middle(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(docs) <= 2:
        return docs
    docs_sorted = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
    top = docs_sorted[0]
    second = docs_sorted[1]
    middle = docs_sorted[2:]
    return [top] + middle + [second]


def dedup_context_items(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        k = norm(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def summarize_text(client: Optional[OpenAI], text: str, instruction: str) -> str:
    if not text.strip():
        return ""
    if not client:
        return text[:500]
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text[:12000]},
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return text[:500]


def update_recursive_summaries(openai_client: Optional[OpenAI], working_history: List[Dict[str, str]]) -> Dict[str, str]:
    raw = "\n".join([f"{m['role']}: {m['content']}" for m in working_history])
    tier1 = summarize_text(openai_client, raw, "Create a detailed session summary in <= 200 words.")
    tier2 = summarize_text(openai_client, tier1, "Create a weekly epoch summary in <= 100 words.")
    tier3 = summarize_text(openai_client, tier2, "Create a master profile summary in <= 60 words.")
    return {"tier1_session": tier1, "tier2_epoch": tier2, "tier3_master": tier3}


def apply_decay_and_pruning(local_store: LocalMemoryStore, user_id: str, prune_below: float = 0.35) -> Dict[str, int]:
    memories = local_store.list_memories(user_id)
    deleted = 0
    decayed = 0
    for m in memories:
        age_days = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(m["created_at"])).total_seconds() / 86400.0)
        importance = float(m.get("importance", 5)) / 10.0
        access = min(1.0, float(m.get("access_count", 0)) / 10.0)
        health = 0.55 * importance + 0.3 * access + 0.15 * (1.0 / (1.0 + age_days / 7.0))
        if health < prune_below:
            local_store.delete_memory(m["memory_id"])
            deleted += 1
        else:
            new_importance = max(1, min(10, int(round(m.get("importance", 5) * 0.97))))
            local_store.update_memory(m["memory_id"], {"importance": new_importance})
            decayed += 1
    return {"deleted": deleted, "decayed": decayed}


def build_health_response(
    openai_client: Optional[OpenAI],
    user_text: str,
    recalled: List[Dict[str, Any]],
    summaries: Dict[str, str],
) -> str:
    memory_context = "\n".join([f"- [{m['memory_type']}] {m['content']}" for m in recalled])
    prompt = f"""
You are a health coach assistant.
User message: {user_text}
Recalled memory:
{memory_context}
Master summary:
{summaries.get("tier3_master", "")}

Requirements:
- Use memory when relevant.
- If allergies/preferences exist, enforce them.
- Keep answer concise and practical.
"""
    if not openai_client:
        if "recipe" in user_text.lower() and "peanut" in memory_context.lower():
            return "I remember your peanut allergy. Here is a safe recipe: grilled salmon, quinoa, and steamed vegetables with olive oil and lemon."
        return "Noted. I can help with a practical health plan and remember your preferences for future sessions."
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a careful health coach."}, {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "I can help with a practical health plan and preserve key preferences across sessions."


def init_state() -> None:
    if "working_history" not in st.session_state:
        st.session_state.working_history = []
    if "semantic_cache" not in st.session_state:
        st.session_state.semantic_cache = {}
    if "provenance_logs" not in st.session_state:
        st.session_state.provenance_logs = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {"tier1_session": "", "tier2_epoch": "", "tier3_master": ""}


st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
init_state()
store = LocalMemoryStore(STORE_PATH)

st.title("🧠 Module 3: Memory Architectures for Persistent Agents")
st.caption("Demo app for Working, Episodic, Semantic, Procedural memory with OpenAI + Mem0 (optional).")

with st.sidebar:
    st.header("Configuration")
    user_id = st.text_input("User ID", value="demo_user_001")
    openai_key = st.text_input("OpenAI API Key", type="password")
    recall_top_k = st.slider("Recall Top-K", 1, 15, 6)
    max_working_turns = st.slider("Working Memory Turns", 4, 20, 8)
    prune_threshold = st.slider("Pruning Threshold", 0.1, 0.9, 0.35, 0.05)
    enable_mem0 = st.toggle("Enable Mem0 (if installed)", value=True)

openai_client = OpenAI(api_key=openai_key) if openai_key and OPENAI_AVAILABLE else None
mem0_client = Mem0Client(openai_key) if (openai_key and enable_mem0) else Mem0Client("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Architecture", "Health Coach Agent", "Maintenance", "Token Optimization", "Enterprise Logs"]
)

with tab1:
    st.subheader("Multi-layer Memory")
    st.markdown(
        """
    - **Working Memory**: Recent chat turns in context window.
    - **Episodic Memory**: Chronological interactions/events.
    - **Semantic Memory**: Stable facts/preferences.
    - **Procedural Memory**: Reusable action sequences.
    - **Temporal Divide**: Working memory is volatile; long-term memory persists.
    """
    )
    cols = st.columns(4)
    mems = store.list_memories(user_id)
    with cols[0]:
        st.metric("Working Turns", len(st.session_state.working_history))
    with cols[1]:
        st.metric("Episodic", sum(1 for m in mems if m["memory_type"] == "episodic"))
    with cols[2]:
        st.metric("Semantic", sum(1 for m in mems if m["memory_type"] == "semantic"))
    with cols[3]:
        st.metric("Procedural", sum(1 for m in mems if m["memory_type"] == "procedural"))

    st.subheader("Stored Long-Term Memory")
    st.dataframe(pd.DataFrame(mems), use_container_width=True)

with tab2:
    st.subheader("Hands-on: Persistent Health Coach Agent")
    st.write("Goal: learn peanut allergy in Session 1, prune noise, recall in Session 5.")

    c1, c2 = st.columns([2, 1])
    with c1:
        user_text = st.text_area(
            "User message",
            placeholder="Example: I am allergic to peanuts. Suggest a weekly meal plan.",
            height=120,
        )
    with c2:
        scripted = st.button("Run Scripted Session 1→5 Demo")
        send = st.button("Send to Agent", type="primary")

    if scripted:
        scripted_turns = [
            "Session 1: I am allergic to peanuts. Please remember this.",
            "Session 2: Bad weather ruined my jog today.",
            "Session 3: I prefer low-risk long-term health plans.",
            "Session 4: How to build a 3-day meal prep workflow?",
            "Session 5: Give me a quick dinner recipe.",
        ]
        for turn in scripted_turns:
            candidates = extract_candidate_memories(openai_client, turn)
            for c in candidates:
                mem = MemoryItem(
                    memory_id=str(uuid.uuid4()),
                    user_id=user_id,
                    memory_type=c["memory_type"],
                    content=c["content"],
                    importance=c["importance"],
                    created_at=now_iso(),
                    last_accessed_at=now_iso(),
                    access_count=0,
                    source="user",
                    metadata={"session": "scripted"},
                )
                store.add_memory(mem)
                if mem0_client.available:
                    mem0_client.add(mem.content, user_id=user_id, metadata={"memory_type": mem.memory_type})
        st.success("Scripted memories added. Now ask for a recipe and verify peanut-aware recall.")

    if send and user_text.strip():
        st.session_state.working_history.append({"role": "user", "content": user_text})
        st.session_state.working_history = st.session_state.working_history[-max_working_turns:]

        # Semantic cache check.
        cache_key = norm(user_text)
        if cache_key in st.session_state.semantic_cache:
            answer = st.session_state.semantic_cache[cache_key]
            cache_hit = True
            recalled = []
        else:
            recalled = retrieve_relevant_memories(user_id, user_text, store, openai_client, recall_top_k)
            for m in recalled:
                store.update_memory(m["memory_id"], {"last_accessed_at": now_iso(), "access_count": int(m.get("access_count", 0)) + 1})
            st.session_state.summaries = update_recursive_summaries(openai_client, st.session_state.working_history)
            answer = build_health_response(openai_client, user_text, recalled, st.session_state.summaries)
            st.session_state.semantic_cache[cache_key] = answer
            cache_hit = False

        candidates = extract_candidate_memories(openai_client, user_text)
        for c in candidates:
            item = MemoryItem(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                memory_type=c["memory_type"],
                content=c["content"],
                importance=max(1, min(10, int(c["importance"]))),
                created_at=now_iso(),
                last_accessed_at=now_iso(),
                access_count=0,
                source=c["source"],
                metadata={"from_turn": len(st.session_state.working_history)},
            )
            store.add_memory(item)
            if mem0_client.available:
                mem0_client.add(item.content, user_id=user_id, metadata={"memory_type": item.memory_type})

        st.session_state.working_history.append({"role": "assistant", "content": answer})
        st.session_state.working_history = st.session_state.working_history[-max_working_turns:]

        st.session_state.provenance_logs.append({
            "ts": now_iso(),
            "user_id": user_id,
            "query": user_text,
            "cache_hit": cache_hit,
            "recalled_count": len(recalled),
            "recalled_memory_ids": [m.get("memory_id") for m in recalled],
        })

        st.success("Response generated.")
        st.markdown("**Assistant**")
        st.write(answer)
        st.markdown(f"**Semantic Cache**: {'hit' if cache_hit else 'miss'}")

    st.subheader("Working Memory (Context Window)")
    st.dataframe(pd.DataFrame(st.session_state.working_history), use_container_width=True)

    st.subheader("Relevance-Based Recall (Local)")
    qry = st.text_input("Try retrieval query", value="recipe without peanuts")
    if st.button("Run Recall Preview"):
        rr = retrieve_relevant_memories(user_id, qry, store, openai_client, recall_top_k)
        st.dataframe(pd.DataFrame(rr), use_container_width=True)
        if mem0_client.available:
            st.write("Mem0 recall preview")
            st.json(mem0_client.search(qry, user_id=user_id, limit=5))

with tab3:
    st.subheader("Memory Maintenance: Decay, Pruning, Consolidation")
    left, right = st.columns(2)

    with left:
        if st.button("Apply Decay + Importance-Weighted Pruning"):
            out = apply_decay_and_pruning(store, user_id=user_id, prune_below=prune_threshold)
            st.success(f"Pruning complete. Deleted: {out['deleted']}, decayed: {out['decayed']}")

    with right:
        if st.button("Consolidate Episodic -> Semantic Rule"):
            episodic = [m["content"] for m in store.list_memories(user_id) if m["memory_type"] == "episodic"][:50]
            merged = "\n".join(episodic)
            rule = summarize_text(openai_client, merged, "Extract one durable semantic rule from episodic events.")
            consolidated = MemoryItem(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                memory_type="semantic",
                content=f"Consolidated rule: {rule}",
                importance=8,
                created_at=now_iso(),
                last_accessed_at=now_iso(),
                access_count=0,
                source="consolidation",
                metadata={"input_count": len(episodic)},
            )
            store.add_memory(consolidated)
            st.success("Consolidated semantic memory saved.")

    st.subheader("Recursive Summarization (Tier 1 / 2 / 3)")
    if st.button("Refresh Hierarchical Summaries"):
        st.session_state.summaries = update_recursive_summaries(openai_client, st.session_state.working_history)
    st.json(st.session_state.summaries)

with tab4:
    st.subheader("Token Optimization + RAG Strategies")
    docs_raw = st.text_area(
        "Paste candidate context chunks (one per line)",
        value="\n".join([
            "Doc A: User is allergic to peanuts and tree nuts.",
            "Doc B: User prefers low-risk and long-term plans.",
            "Doc C: Weekly weather report with rain.",
            "Doc A: User is allergic to peanuts and tree nuts.",
            "Doc D: Recipe options include salmon, quinoa, and vegetables.",
        ]),
        height=180,
    )
    docs = [{"text": d.strip(), "score": float(1.0 / (i + 1))} for i, d in enumerate(docs_raw.splitlines()) if d.strip()]
    deduped = dedup_context_items([d["text"] for d in docs])
    reordered = reorder_docs_lost_middle([{"text": d, "score": 1.0 - idx * 0.1} for idx, d in enumerate(deduped)])
    injected = selective_injection([d["text"] for d in reordered], max_tokens=250)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**De-duplicated**")
        st.write(deduped)
    with col_b:
        st.markdown("**Reordered (Lost-in-Middle Mitigation)**")
        st.write([d["text"] for d in reordered])
    with col_c:
        st.markdown("**Selective Injection (Token Budget)**")
        st.write(injected)
        st.caption(f"Approx tokens: {sum(rough_tokens(x) for x in injected)}")

with tab5:
    st.subheader("Session Context vs Provenance/Audit Logs")
    st.write("Use high-frequency session memory separately from immutable provenance logs.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Session Context (volatile)**")
        st.dataframe(pd.DataFrame(st.session_state.working_history), use_container_width=True)
    with c2:
        st.markdown("**Provenance Log (auditable)**")
        st.dataframe(pd.DataFrame(st.session_state.provenance_logs), use_container_width=True)

    if st.button("Export Provenance Log JSON"):
        payload = json.dumps(st.session_state.provenance_logs, indent=2)
        st.download_button("Download provenance.json", data=payload, file_name="provenance.json", mime="application/json")

st.divider()
st.caption(
    f"OpenAI available: {OPENAI_AVAILABLE} | Mem0 installed: {MEM0_AVAILABLE} | Mem0 active: {mem0_client.available}"
)

