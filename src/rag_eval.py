"""RAG retrieval & generation evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.agent import run_agent_with_trace
from src.config import settings


# Ground-truth relevant (doc, section) pairs for retrieval evaluation
GROUND_TRUTH_SECTIONS: dict[str, list[tuple[str, str]]] = {
    "Q01": [("01_getting_started.md", "Open an account"), ("01_getting_started.md", "Channels")],
    "Q02": [("03_fees_limits.md", "Person-to-person (RiverPay to RiverPay)")],
    "Q03": [("08_assistant_policy.md", "Never"), ("08_assistant_policy.md", "Handoff to a human when")],
    "Q04": [("07_loans.md", "What the assistant may say"), ("07_loans.md", "What the assistant must not say"), ("08_assistant_policy.md", "Never")],
    "Q05": [("01_getting_started.md", "Channels"), ("08_assistant_policy.md", "Always")],
    "Q06": [("05_security.md", "PIN")],
    "Q07": [("06_troubleshooting.md", "USSD dropped mid-send"), ("06_troubleshooting.md", "First checks")],
    "Q08": [],  # Router short-circuit (refuse_jailbreak)
    "Q09": [("03_fees_limits.md", "Agent cash-in (customer)"), ("04_agents.md", "Cash-in")],
    "Q10": [("01_getting_started.md", "Account tiers"), ("02_send_receive.md", "Daily send limits (rolling 24 hours)"), ("03_fees_limits.md", "Limit recap")],
    "Q11": [("04_agents.md", "Agent misconduct"), ("05_security.md", "Common scams"), ("05_security.md", "PIN"), ("08_assistant_policy.md", "Handoff to a human when")],
    "Q12": [("02_send_receive.md", "What you cannot do in this handbook")],
    "Q13": [("03_fees_limits.md", "Person-to-person (RiverPay to RiverPay)"), ("03b_fees_limits_ARCHIVE.md", "Person-to-person (historical)"), ("03b_fees_limits_ARCHIVE.md", "Agent cash-out (historical)")],
    "Q14": [("04_agents.md", "Cash-out"), ("03_fees_limits.md", "Agent cash-out (customer)")],
    "Q15": [("08_assistant_policy.md", "Never")],
    "Q16": [("03_fees_limits.md", "What is not in this product"), ("08_assistant_policy.md", "Never"), ("08_assistant_policy.md", "Always")],
}


class ClaimCheck(BaseModel):
    claim: str = Field(..., description="Atomic factual claim extracted from the answer")
    supported: bool = Field(..., description="True if strictly supported by the retrieved context")


class FaithfulnessResult(BaseModel):
    claims: list[ClaimCheck] = Field(default_factory=list)
    faithfulness_score: float = Field(..., description="Ratio of supported claims to total claims (0.0 to 1.0)")
    unsupported_claims: list[str] = Field(default_factory=list)


def judge_faithfulness(
    client: genai.Client,
    question: str,
    answer: str,
    context_texts: list[str],
) -> FaithfulnessResult:
    """Use Gemini 3.5 Flash Lite(T=0) to score answer faithfulness against retrieved chunks."""
    context_joined = "\n\n".join(context_texts) if context_texts else "Policy short-circuit: 08_assistant_policy.md Never (block jailbreak / 0% fee claims)."
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved Context:\n{context_joined}\n\n"
        f"Assistant Answer:\n{answer}\n\n"
        "Break the Assistant Answer into atomic factual claims. "
        "For each claim, set supported=true ONLY if it is directly entailed by the Retrieved Context "
        "(or is a standard polite refusal/handoff statement aligned with policy). "
        "Compute faithfulness_score = supported_count / total_count (1.0 if all supported)."
    )
    resp = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=FaithfulnessResult,
                ),
            )
            break
        except Exception as exc:
            if attempt == 4 or ("503" not in str(exc) and "429" not in str(exc)):
                raise
            time.sleep(2 ** (attempt + 1))
    return FaithfulnessResult.model_validate_json(resp.text)


def run_rag_evaluation() -> dict:
    """Evaluate Retrieval Precision@K, Recall@K, and Faithfulness across all 16 questions."""
    questions_path = Path(settings.data_dir) / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    client = genai.Client(api_key=settings.gemini_api_key)

    per_question: list[dict] = []
    precisions: list[float] = []
    recalls: list[float] = []
    faithfulness_scores: list[float] = []

    for item in questions:
        qid = item["question_id"]
        query = item["user"]
        state = run_agent_with_trace(user_query=query, question_id=qid)
        chunks = state.get("retrieved_chunks", [])
        answer = state["final_response"].answer

        retrieved_pairs = [(c["metadata"]["doc"], c["metadata"]["section"]) for c in chunks]
        gt_pairs = set(GROUND_TRUTH_SECTIONS.get(qid, []))

        if gt_pairs and retrieved_pairs:
            hits = sum(1 for p in retrieved_pairs if p in gt_pairs)
            precision = round(hits / len(retrieved_pairs), 4)
            recall = round(len(set(retrieved_pairs) & gt_pairs) / len(gt_pairs), 4)
            precisions.append(precision)
            recalls.append(recall)
        else:
            precision = None
            recall = None

        faith = judge_faithfulness(
            client=client,
            question=query,
            answer=answer,
            context_texts=[c["text"] for c in chunks],
        )
        faithfulness_scores.append(faith.faithfulness_score)

        per_question.append(
            {
                "question_id": qid,
                "route": state.get("route"),
                "k_retrieved": len(retrieved_pairs),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "faithfulness": round(faith.faithfulness_score, 4),
                "unsupported_claims": faith.unsupported_claims,
            }
        )
        print(
            f"{qid} | route={state.get('route'):18s} | "
            f"P@K={str(precision):6s} | R@K={str(recall):6s} | Faith={faith.faithfulness_score:.2f}"
        )

    summary = {
        "macro_precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "macro_recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "macro_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "evaluated_retrieval_questions": len(precisions),
        "evaluated_faithfulness_questions": len(faithfulness_scores),
        "per_question": per_question,
    }

    out_path = Path(settings.output_dir) / "rag_eval_report.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n--- SUMMARY ---")
    print(f"Macro Precision@K : {summary['macro_precision_at_k']:.4f}")
    print(f"Macro Recall@K    : {summary['macro_recall_at_k']:.4f}")
    print(f"Macro Faithfulness: {summary['macro_faithfulness']:.4f}")
    print(f"Report saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_rag_evaluation()
