# agents/planner.py
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning expert. Given a research question,
break it down into 3-5 specific, focused sub-questions that together will
fully answer the main question.

Return ONLY a valid JSON array of strings — no explanation, no markdown, no extra text.
Example output: ["What is X?", "How does Y relate to X?", "What are recent developments in Z?"]
"""),
    ("human", "Research question: {query}")
])


def planner_node(state: ResearchState) -> dict:
    """Decomposes the user query into focused sub-questions."""
    chain = planner_prompt | llm
    response = chain.invoke({"query": state["query"]})

    # Strip markdown fences if the model wraps in ```json ... ```
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        sub_questions = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat each non-empty line as a sub-question
        sub_questions = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    print(f"\n[PLANNER] Generated {len(sub_questions)} sub-questions:")
    for q in sub_questions:
        print(f"  - {q}")

    return {"sub_questions": sub_questions}