"""Streamlit Chat UI + JSON Contract & Handoff Inspector for RiverPay."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.agent import ask_assistant
from src.config import settings


st.set_page_config(
    page_title="RiverPay Customer Assistant",
    page_icon="💳",
    layout="wide",
)

st.title("RiverPay Customer Support Assistant")
st.caption("Grounded in official RiverPay Kambara (KBR) documentation · Zero hallucination · Policy-enforced")

tab_chat, tab_eval = st.tabs(["Interactive Support Chat", "Evaluation Audit (16 Questions)"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_chat, col_inspector = st.columns([3, 2], gap="large")

    with col_chat:
        st.subheader("Customer Chat")
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "payload" in msg:
                    payload = msg["payload"]
                    if payload.get("handoff_to_human"):
                        st.warning(
                            "**Human Handoff Recommended:** Please contact the RiverPay Contact Center "
                            "(07:00–22:00 Kambara time) or dial `*123*8#` for a callback."
                        )
                    if payload.get("refusal"):
                        st.info("Policy Guardrail: Request declined or out of handbook scope.")

        user_input = st.chat_input("Ask a question about RiverPay (fees, limits, USSD, agents, security)...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Checking RiverPay handbook..."):
                try:
                    qid = f"CHAT-{len(st.session_state.messages):03d}"
                    response = ask_assistant(user_query=user_input, question_id=qid)
                    payload_dict = response.model_dump()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "payload": payload_dict,
                        }
                    )
                    st.session_state.last_payload = payload_dict
                except Exception as exc:
                    st.error(f"Error running assistant: {exc}")
            st.rerun()

    with col_inspector:
        st.subheader("JSON Contract & Handoff Inspector")
        last_payload = st.session_state.get("last_payload")
        if last_payload:
            b1, b2 = st.columns(2)
            b1.metric("Refusal", "YES" if last_payload["refusal"] else "NO")
            b2.metric("Human Handoff", "YES" if last_payload["handoff_to_human"] else "NO")

            st.markdown("**Citations**")
            if last_payload["citations"]:
                for cite in last_payload["citations"]:
                    st.code(f"{cite['doc']}  →  {cite['section']}", language="text")
            else:
                st.caption("No citations attached.")

            st.markdown("**Raw Output JSON**")
            st.json(last_payload)
        else:
            st.info("Send a message in the chat to inspect the structured JSON output, citations, and routing notes.")

with tab_eval:
    st.subheader("Batch Evaluation Results (`output/eval_results.json`)")
    eval_path = Path(settings.output_dir) / "eval_results.json"
    questions_path = Path(settings.data_dir) / "questions.json"

    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        q_map = {}
        if questions_path.exists():
            q_map = {q["question_id"]: q["user"] for q in json.loads(questions_path.read_text(encoding="utf-8"))}

        refusals = sum(1 for r in eval_data if r.get("refusal"))
        handoffs = sum(1 for r in eval_data if r.get("handoff_to_human"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Questions", len(eval_data))
        c2.metric("Refusals", refusals)
        c3.metric("Human Handoffs", handoffs)

        for item in eval_data:
            qid = item["question_id"]
            with st.expander(f"{qid}: {q_map.get(qid, '')[:80]}"):
                st.markdown(f"**Customer:** {q_map.get(qid, 'N/A')}")
                st.markdown(f"**Assistant:** {item['answer']}")
                st.json(item)
    else:
        st.warning("No `output/eval_results.json` found yet. Run `python -m src.evaluate` first.")
