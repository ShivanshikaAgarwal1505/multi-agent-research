# state.py
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    # The original user question
    query: str

    # Planner breaks the query into focused sub-questions
    sub_questions: list[str]

    # Search agents append raw text here (Annotated means auto-append, not overwrite)
    raw_results: Annotated[list[str], lambda a, b: a + b]

    # Critic filters raw_results down to clean facts
    verified_facts: list[str]

    # Writer produces the final report here
    final_report: str

    # LangGraph internal message tracking
    messages: Annotated[list, add_messages]