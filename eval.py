# eval.py
"""
Compares multi-agent pipeline vs single direct LLM call.
Uses LLM-as-judge to score factual density (0-10).
"""
import requests
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT = """Rate this research report on a scale of 0-10 for:
- Factual density (specific facts, numbers, names)
- Coverage (addresses multiple angles of the question)
- Clarity (well-organized, not repetitive)

Return ONLY a JSON object: {{"factual_density": N, "coverage": N, "clarity": N}}

Report:
{report}
"""

judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def score_report(report: str) -> dict:
    response = judge_llm.invoke(JUDGE_PROMPT.format(report=report))
    import json
    return json.loads(response.content)


def single_llm_baseline(query: str) -> str:
    """Direct single LLM call — no agents, no search."""
    response = judge_llm.invoke(
        f"Write a comprehensive research report on: {query}"
    )
    return response.content


def run_eval(query: str):
    print(f"\nQuery: {query}")
    print("=" * 60)

    # Baseline: single LLM
    print("\n[1/2] Running single-LLM baseline...")
    baseline_report = single_llm_baseline(query)
    baseline_scores = score_report(baseline_report)

    # Pipeline: multi-agent
    print("[2/2] Running multi-agent pipeline...")
    resp = requests.post(
        "http://localhost:8000/research",
        json={"query": query},
        timeout=180,
    )
    pipeline_report = resp.json()["final_report"]
    pipeline_scores = score_report(pipeline_report)

    # Results
    print("\n── Scores (0-10) ──────────────────────")
    print(f"{'Metric':<20} {'Baseline':>10} {'Pipeline':>10} {'Δ':>10}")
    print("-" * 52)

    total_baseline, total_pipeline = 0, 0
    for metric in ["factual_density", "coverage", "clarity"]:
        b = baseline_scores[metric]
        p = pipeline_scores[metric]
        delta = ((p - b) / b * 100) if b > 0 else 0
        total_baseline += b
        total_pipeline += p
        print(f"{metric:<20} {b:>10.1f} {p:>10.1f} {delta:>+9.1f}%")

    overall_delta = ((total_pipeline - total_baseline) / total_baseline * 100)
    print("-" * 52)
    print(f"{'OVERALL':<20} {total_baseline:>10.1f} {total_pipeline:>10.1f} {overall_delta:>+9.1f}%")
    print(f"\nResume bullet: improved report quality by {overall_delta:.1f}% vs single-LLM baseline")


if __name__ == "__main__":
    run_eval("What are the latest advances in Retrieval-Augmented Generation?")