# graph.py

# graph.py
from dotenv import load_dotenv
load_dotenv()  # ← must be BEFORE the agent imports below

from langgraph.graph import StateGraph, END
from state import ResearchState
from agents.planner import planner_node
from agents.searcher import web_search_node, vector_search_node, summarizer_node
from agents.critic import critic_node
from agents.writer import writer_node

def build_graph():
    graph = StateGraph(ResearchState)

    # ── Register nodes ──────────────────────────────────────────────────────
    graph.add_node("planner",       planner_node)
    graph.add_node("web_search",    web_search_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("summarizer",    summarizer_node)
    graph.add_node("critic",        critic_node)
    graph.add_node("writer",        writer_node)

    # ── Define flow ─────────────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # Planner → both search nodes (they run in the same step, results merge)
    graph.add_edge("planner",       "web_search")
    graph.add_edge("planner",       "vector_search")

    # Both search nodes feed into summarizer
    graph.add_edge("web_search",    "summarizer")
    graph.add_edge("vector_search", "summarizer")

    # Linear from here: summarizer → critic → writer → done
    graph.add_edge("summarizer",    "critic")
    graph.add_edge("critic",        "writer")
    graph.add_edge("writer",        END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "query": "What are the latest advances in Retrieval-Augmented Generation?",
        "sub_questions": [],
        "raw_results": [],
        "verified_facts": [],
        "final_report": "",
        "messages": []
    })

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])

    print("\n" + "=" * 60)
    print(f"Sub-questions ({len(result['sub_questions'])}):")
    for q in result["sub_questions"]:
        print(f"  - {q}")

    print(f"\nVerified facts ({len(result['verified_facts'])}):")
    for f in result["verified_facts"]:
        print(f"  • {f}")