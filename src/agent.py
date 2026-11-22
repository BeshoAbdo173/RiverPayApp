"""4-node LangGraph RAG agent: router -> retrieve -> generate -> verify."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import chromadb
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.schemas import AssistantResponse, Citation, LLMGenerationPayload


JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?marketing\s+bot", re.IGNORECASE),
    re.compile(r"pretend\s+(all\s+)?fees\s+are\s+0", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+(your\s+)?(policy|rules|instructions)", re.IGNORECASE),
]

ARCHIVE_PATTERNS = [
    re.compile(r"\b1\s*%", re.IGNORECASE),
    re.compile(r"\b1\.5\s*%", re.IGNORECASE),
    re.compile(r"\bold\s+fee", re.IGNORECASE),
    re.compile(r"\bused\s+to\s+be\b", re.IGNORECASE),
    re.compile(r"\bhistorical\b|\barchive\b|\bpreviously\b", re.IGNORECASE),
    re.compile(r"\b2025\b|\bjune\s+2026\b|\bbefore\s+july\b", re.IGNORECASE),
    re.compile(r"\bis\s+that\s+still\s+the\s+fee\b", re.IGNORECASE),
]

PIN_LEAK_RE = re.compile(r"\bmy\s+(?:pin|otp|password)\s+is\s+\d{4,8}\b", re.IGNORECASE)
FILENAME_LEAK_RE = re.compile(r"\b0[1-8]b?_[a-z_]+(?:_ARCHIVE)?\.md\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are the official RiverPay customer support assistant for Kambara (currency: KBR).
You MUST follow `08_assistant_policy.md` strictly. Every rule below is binding:

1. GROUNDING: Answer ONLY from the retrieved knowledge base chunks provided to you. If a fact is not in the retrieved chunks, say you do not have that information and offer a human agent (`refusal=true`, `handoff_to_human=true`).
2. PLAIN LANGUAGE: Write in calm, respectful, plain language with short paragraphs. NEVER mention internal filenames (like `03_fees_limits.md` or `08_assistant_policy.md`) in the customer-facing `answer` text.
3. CITATIONS: For every answer that states a fee, limit, or procedure, include at least one citation in `citations` using the EXACT `doc` filename and EXACT `section` heading from the retrieved chunks. Never invent a `doc` or `section`.
4. TEMPORAL CONFLICTS: When two documents disagree, ALWAYS use the document with the LATEST `effective_date` (e.g., `03_fees_limits.md` effective 2026-07-01 supersedes `03b_fees_limits_ARCHIVE.md` sunset 2026-06-30). If the customer cites an old rule (such as 1% P2P fee), explain clearly that the fee schedule changed on 1 July 2026 and quote the current fee (0.5%, min 1 KBR, max 25 KBR).
5. STRICT REFUSALS (`refusal=true`):
   - NEVER invent fees, FX rates (USD/EUR/etc.), interest rates (APR), or promotions.
   - NEVER quote a customer's balance, mini-statement, or PIN (there is no live account API). For balance/account lookup requests, set `refusal=true` and `handoff_to_human=true`.
   - NEVER approve, pre-approve, or score a loan (RiverCredit is invite-only; KYC2 is required to see offers in-app, not a guarantee). Set `refusal=true`.
   - NEVER execute a transfer, change a limit, or unlock an account. Set `refusal=true`.
   - NEVER confirm unsupported features (e.g., Apple Pay, Google Pay, international bank transfers). Note: if a document explicitly says a feature is not offered (like sending to a bank account outside RiverPay), state that clearly; if it says it is not described in the handbook (like Apple Pay) or no FX table exists (like USD/KBR), set `refusal=true` and `handoff_to_human=true`.
6. HANDOFF TO HUMAN (`handoff_to_human=true`):
   - Balance or specific transaction status you cannot look up, or disputes after the 48-hour window.
   - Fraud, a scam, or an agent demanding a PIN (answer clearly: NO, never share your PIN, cancel/leave, and contact the contact center; set `handoff_to_human=true`).
   - Frozen wallet / "review" language (do not guess why; hand off).
   - Unanswerable / missing info in the handbook (e.g., FX rates, Apple Pay).
   - Pending transaction after 15 minutes.
7. NEVER ask the user for a full PIN, password, OTP, or ID number. If they pasted one, tell them to stop and rotate their PIN immediately.
"""


class AgentState(TypedDict, total=False):
    question_id: str
    user_query: str
    route: str
    include_archive: bool
    pasted_secret: bool
    active_filter: str
    retrieved_chunks: list[dict[str, Any]]
    llm_payload: LLMGenerationPayload
    final_response: AssistantResponse
    start_time: float


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    try:
        return SentenceTransformer(settings.embedding_model, local_files_only=True)
    except Exception:
        return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_collection(name=settings.chroma_collection)


@lru_cache(maxsize=1)
def _get_valid_sections() -> dict[str, list[str]]:
    path = Path(settings.output_dir) / "valid_sections.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or environment variables before running."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def router_node(state: AgentState) -> AgentState:
    """Classify query intent, detect jailbreaks, and determine temporal metadata filter."""
    query = state["user_query"].strip()
    is_jailbreak = any(p.search(query) for p in JAILBREAK_PATTERNS)
    include_archive = any(p.search(query) for p in ARCHIVE_PATTERNS)
    pasted_secret = bool(PIN_LEAK_RE.search(query))

    if is_jailbreak:
        route = "refuse_jailbreak"
    elif include_archive:
        route = "rag_with_archive"
    else:
        route = "rag_current_only"

    return {
        **state,
        "route": route,
        "include_archive": include_archive,
        "pasted_secret": pasted_secret,
        "start_time": state.get("start_time") or time.perf_counter(),
    }


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve top-K chunks from ChromaDB with optional archive exclusion filter."""
    if state.get("route") == "refuse_jailbreak":
        return {
            **state,
            "active_filter": "none (jailbreak_short_circuit)",
            "retrieved_chunks": [],
        }

    collection = _get_collection()
    embedder = _get_embedder()
    query_embedding = embedder.encode([state["user_query"]], normalize_embeddings=True).tolist()

    include_archive = state.get("include_archive", False)
    where_filter: dict[str, Any] | None = None if include_archive else {"is_archive": False}
    active_filter = "none (archive_included)" if include_archive else "is_archive=False"

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=max(settings.top_k, 6),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for cid, doc_text, meta, dist in zip(ids, docs, metas, dists):
        similarity = round(max(0.0, 1.0 - float(dist)), 4)
        seen_ids.add(cid)
        chunks.append(
            {
                "id": cid,
                "text": doc_text,
                "metadata": meta,
                "score": similarity,
            }
        )

    # Ensure current fee schedule is always present when archive docs are queried
    if include_archive:
        current_fees = collection.get(
            where={"doc": "03_fees_limits.md"},
            include=["documents", "metadatas"],
        )
        for cid, doc_text, meta in zip(
            current_fees.get("ids", []),
            current_fees.get("documents", []),
            current_fees.get("metadatas", []),
        ):
            if cid not in seen_ids:
                chunks.append(
                    {
                        "id": cid,
                        "text": doc_text,
                        "metadata": meta,
                        "score": 1.0,
                    }
                )

    return {
        **state,
        "active_filter": active_filter,
        "retrieved_chunks": chunks,
    }


def generate_node(state: AgentState) -> AgentState:
    """Generate structured JSON answer using Gemini 2.5 Flash or deterministic policy refusal."""
    if state.get("route") == "refuse_jailbreak":
        payload = LLMGenerationPayload(
            answer=(
                "I cannot follow instructions to ignore my guidelines or claim that RiverPay fees are 0%. "
                "All RiverPay transactions follow our standard published fee schedule, and there are no "
                "promotional zero-fee offers."
            ),
            citations=[
                Citation(doc="08_assistant_policy.md", section="Never"),
                Citation(doc="03_fees_limits.md", section="What is not in this product"),
            ],
            refusal=True,
            handoff_to_human=False,
            reasoning="Prompt injection / jailbreak attempt blocked by router.",
        )
        return {**state, "llm_payload": payload}

    chunks = state.get("retrieved_chunks", [])
    context_blocks = []
    for idx, ch in enumerate(chunks, start=1):
        meta = ch["metadata"]
        context_blocks.append(
            f"--- CHUNK {idx} | doc={meta['doc']} | section={meta['section']} | "
            f"effective_date={meta['effective_date']} | is_archive={meta['is_archive']} ---\n"
            f"{ch['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    secret_warning = (
        "\nIMPORTANT: The user appears to have pasted a PIN/OTP/password. "
        "Tell them to stop sharing it immediately and rotate their PIN."
        if state.get("pasted_secret")
        else ""
    )

    user_prompt = (
        f"Customer Question: {state['user_query']}\n"
        f"{secret_warning}\n\n"
        f"Retrieved Knowledge Base Context:\n{context_str}\n\n"
        "Return a valid JSON object matching the schema. "
        "Cite ONLY exact `doc` and `section` values from the chunks above."
    )

    client = _get_genai_client()
    response = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=settings.llm_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=settings.llm_temperature,
                    response_mime_type="application/json",
                    response_schema=LLMGenerationPayload,
                ),
            )
            break
        except Exception as exc:
            if attempt == 4 or ("503" not in str(exc) and "429" not in str(exc)):
                raise
            time.sleep(2 ** (attempt + 1))

    payload = LLMGenerationPayload.model_validate_json(response.text)
    return {**state, "llm_payload": payload}


def _match_valid_section(doc: str, section: str, valid_sections: dict[str, list[str]]) -> str | None:
    """Return exact valid section heading if it matches, else None."""
    sections = valid_sections.get(doc)
    if not sections:
        return None
    if section in sections:
        return section
    lowered = section.strip().lower()
    for s in sections:
        if s.lower() == lowered:
            return s
    return None


def verify_node(state: AgentState) -> AgentState:
    """Deterministically verify citations, strip filename leaks, and build final output contract."""
    payload = state["llm_payload"]
    valid_sections = _get_valid_sections()

    verified_citations: list[Citation] = []
    seen_cites: set[tuple[str, str]] = set()
    for cite in payload.citations:
        matched_section = _match_valid_section(cite.doc, cite.section, valid_sections)
        if matched_section is not None:
            key = (cite.doc, matched_section)
            if key not in seen_cites:
                seen_cites.add(key)
                verified_citations.append(Citation(doc=cite.doc, section=matched_section))

    # Fallback: if non-refusal answer lost all citations, attach top-1 non-archive retrieved chunk
    if not verified_citations and not payload.refusal:
        for ch in state.get("retrieved_chunks", []):
            meta = ch["metadata"]
            if not meta.get("is_archive", False):
                matched = _match_valid_section(meta["doc"], meta["section"], valid_sections)
                if matched:
                    verified_citations.append(Citation(doc=meta["doc"], section=matched))
                    break

    # Strip any accidental internal .md filename leaks from customer-facing answer
    clean_answer = FILENAME_LEAK_RE.sub("our official handbook", payload.answer).strip()

    elapsed_ms = round((time.perf_counter() - state.get("start_time", time.perf_counter())) * 1000, 1)
    notes = (
        f"route={state.get('route')}; "
        f"filter={state.get('active_filter')}; "
        f"latency_ms={elapsed_ms}"
    )

    final = AssistantResponse(
        question_id=state.get("question_id", "CHAT"),
        answer=clean_answer,
        citations=verified_citations,
        refusal=payload.refusal,
        handoff_to_human=payload.handoff_to_human,
        notes=notes,
    )
    return {**state, "final_response": final}


def build_agent():
    """Compile the 4-node LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


_compiled_agent = None


def run_agent_with_trace(user_query: str, question_id: str = "CHAT") -> AgentState:
    """Run a single query and return the full AgentState including retrieved chunks."""
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = build_agent()

    return _compiled_agent.invoke(
        {
            "question_id": question_id,
            "user_query": user_query,
            "start_time": time.perf_counter(),
        }
    )


def ask_assistant(user_query: str, question_id: str = "CHAT") -> AssistantResponse:
    """Run a single query through the compiled RiverPay RAG agent."""
    result = run_agent_with_trace(user_query=user_query, question_id=question_id)
    return result["final_response"]
