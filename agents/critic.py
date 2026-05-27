# agents/critic.py
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a critical fact-checker. Your job is to:
1. Remove duplicate or near-duplicate information
2. Discard vague, contradictory, or unsupported claims
3. Keep only concrete, well-supported facts relevant to the query
4. Return ONLY a valid JSON array of fact strings — no explanation, no markdown fences.

Example output: ["Fact one.", "Fact two.", "Fact three."]
Be strict — quality over quantity."""),
    ("human", """Original query: {query}

Raw research results:
{raw_results}

Return ONLY the JSON array of verified facts.""")
])


def critic_node(state: ResearchState) -> dict:
    """Validates, deduplicates, and filters the collected research."""
    combined = "\n\n".join(state["raw_results"])

    chain = critic_prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "raw_results": combined[:8000]  # guard against token limit
    })

    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        verified_facts = json.loads(raw)
        # Ensure it's actually a list of strings
        if not isinstance(verified_facts, list):
            raise ValueError("Expected a list")
        verified_facts = [str(f) for f in verified_facts]
    except (json.JSONDecodeError, ValueError):
        # Fallback: grab any line starting with - or •
        verified_facts = [
            line.lstrip("-•* ").strip()
            for line in raw.splitlines()
            if line.strip() and line.strip()[0] in "-•*"
        ]

    print(f"\n[CRITIC] Verified {len(verified_facts)} facts from raw results")
    return {"verified_facts": verified_facts}