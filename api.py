# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from graph import build_graph

app = FastAPI(title="Multi-Agent Research Synthesizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit runs on a dynamic port, so allow all
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile the graph once at startup — expensive to rebuild each request
graph = build_graph()


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    sub_questions: list[str]
    verified_facts: list[str]
    final_report: str


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # LangGraph is synchronous internally; run it in a thread
    # so FastAPI's async event loop doesn't block
    result = await asyncio.to_thread(
        graph.invoke,
        {
            "query": request.query,
            "sub_questions": [],
            "raw_results": [],
            "verified_facts": [],
            "final_report": "",
            "messages": [],
        },
    )

    return ResearchResponse(
        query=result["query"],
        sub_questions=result["sub_questions"],
        verified_facts=result["verified_facts"],
        final_report=result["final_report"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}