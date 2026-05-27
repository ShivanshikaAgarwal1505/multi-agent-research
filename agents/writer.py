# agents/writer.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState

# Use the large model here — prose quality is the deliverable
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research writer. Synthesize the provided verified facts \
into a well-structured research report with these sections:

## Executive Summary
2-3 sentences capturing the core answer.

## Key Findings
Organized by theme — not a flat list. Group related facts under sub-headings.

## Implications & Open Questions
What does this mean? What is still unknown or worth investigating?

Write clearly for a technical audience. Do not invent facts beyond the provided list. \
Use markdown formatting."""),
    ("human", """Research question: {query}

Verified facts to synthesize:
{verified_facts}

Write the full research report:""")
])


def writer_node(state: ResearchState) -> dict:
    """Synthesizes verified facts into the final research report."""
    facts_text = "\n".join(f"- {f}" for f in state["verified_facts"])

    chain = writer_prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "verified_facts": facts_text
    })

    print("\n[WRITER] Report generated.")
    return {"final_report": response.content}