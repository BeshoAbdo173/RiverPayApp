# RiverPay App Customer Support Assistant

A deterministic RAG customer support assistant for **RiverPay** (Using currency **KBR**), built to enforce `data/08_assistant_policy.md`.

---

## 1. Architecture & Retrieval Design

The system uses a **4-node LangGraph workflow** (`router → retrieve → generate → verify`) backed by a local persistent **ChromaDB** vector store (`all-MiniLM-L6-v2` embeddings) and **Gemini 3.5 Flash lite** (`temperature=0.0`) with strict Pydantic v2 structured output:

1. **Header-Aware Ingestion (`src/ingest.py`):**
   - Splits all 9 markdown files in `data/` on H1/H2 headings, preserving exact section names.
   - Extracts document metadata (`doc_id`, `effective_date`, `is_archive`, `doc_type`) and prepends contextual headers (`[Document | Section | Effective | Status]`) before embedding.
   - Generates `output/valid_sections.json` as an authoritative allow-list of `(doc, section)` pairs.
2. **Router (`router_node`):**
   - Short-circuits prompt-injection / jailbreak attempts deterministically (`route=refuse_jailbreak`).
   - Detects historical fee queries (e.g., `"1%"`, `"old fee"`, `"before July 2026"`) to toggle the temporal metadata filter.
3. **Retriever (`retrieve_node`):**
   - **Default mode:** Applies `where={"is_archive": False}` so `03b_fees_limits_ARCHIVE.md` is excluded from normal retrieval.
   - **Historical comparison mode:** Relaxes the filter and guarantees `03_fees_limits.md` (effective `2026-07-01`) is included alongside `03b_fees_limits_ARCHIVE.md` (sunset `2026-06-30`) so the assistant explains the 1 July 2026 fee change while quoting the current 0.5% P2P fee.
4. **Generator & Verifier (`generate_node`, `verify_node`):**
   - Enforces binding policy rules via system prompt + structured JSON schema (`LLMGenerationPayload`).
   - Deterministically validates every citation against `output/valid_sections.json`, strips any accidental `.md` filename leaks from the customer-facing answer, and logs routing/retrieval/latency telemetry in `notes`.

---

## 2. Policy Guardrails, Refusals & Handoffs

Per `data/08_assistant_policy.md`:
- **Strict Refusals (`refusal=true`):** Account balances/mini-statements (no live account API), loan approvals/eligibility/APR (`07_loans.md`), transfer execution, USD/KBR FX rates (`03_fees_limits.md`), unsupported wallets (Apple Pay), and prompt injections.
- **Human Handoff (`handoff_to_human=true`):** Balance lookups, agent PIN demands/scams, frozen wallets, disputes, pending transactions > 15 minutes, and unanswerable questions.

---

## 3. Setup & Run Commands (≤ 10 commands)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Add your GEMINI_API_KEY to .env
python -m src.ingest
python -m src.evaluate
python -m src.verify
streamlit run app.py
```

---

## 4. Cost, Latency & Verification

- **Model:** `gemini-3.5-flash-lite` at `temperature=0.0` (~$0.0015–$0.0045 per query; < $0.08 for a full 16-question eval run).
- **Embeddings:** Local `sentence-transformers/all-MiniLM-L6-v2` (zero API cost, ~20–40ms per query).
- **End-to-end latency:** Typically 1,100–2,200 ms per query (jailbreak short-circuit: < 5 ms).
- **CI Audit (`src/verify.py`):** Verifies Pydantic schema compliance, 100% valid `(doc, section)` citations, zero `.md` filename leaks, and policy invariants across all 16 evaluation questions.

---

*AI Assistance Disclosure: AI coding assistant was helped in building the system and documentation.*
