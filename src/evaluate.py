"""Batch evaluation runner: data/questions.json -> output/eval_results.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import ask_assistant
from src.config import settings


def run_evaluation() -> list[dict]:
    """Run all 16 evaluation questions and write output/eval_results.json."""
    questions_path = Path(settings.data_dir) / "questions.json"
    output_path = Path(settings.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    results: list[dict] = []

    for item in questions:
        qid = item["question_id"]
        user_text = item["user"]
        print(f"Evaluating {qid}: {user_text[:65]}...")
        response = ask_assistant(user_query=user_text, question_id=qid)
        results.append(response.model_dump())

    out_file = output_path / "eval_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results)} evaluation results to {out_file}.")
    return results


if __name__ == "__main__":
    run_evaluation()
