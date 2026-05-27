# Multi-Agent Research Synthesizer

A 4-agent LangGraph pipeline that takes a research question and produces a
structured report using hybrid retrieval (web search + vector DB).

## Architecture
```bash
Planner → Web Search ─┐

                      ├─► Summarizer → Critic → Writer

        Vector Search ┘
```

**Agents:**
- **Planner** — decomposes the query into focused sub-questions
- **Web Search** — retrieves live results via Tavily
- **Vector Search** — retrieves semantically similar stored documents via ChromaDB
- **Critic** — deduplicates and validates facts
- **Writer** — synthesizes a structured report

**Stack:** LangGraph · LangChain · Groq (Llama 3) · Tavily · ChromaDB · FastAPI · Streamlit

## Setup

1. Clone the repo
```bash
   git clone https://github.com/YOUR_USERNAME/multi-agent-research.git
   cd multi-agent-research
```

2. Create and activate a virtual environment
```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Set up environment variables
```bash
   cp .env.example .env
   # Edit .env and add your API keys
```

## Running

Open two terminals:

**Terminal 1 — API server**
```bash
uvicorn api:app --reload --port 8000
```

**Terminal 2 — Streamlit UI**
```bash
streamlit run frontend.py
```

Open `http://localhost:8501` in your browser.

## Project Structure
```bash
├── agents/
│   ├── planner.py        # Query decomposition
│   ├── searcher.py       # Web + vector search + summarizer
│   ├── critic.py         # Fact validation
│   └── writer.py         # Report synthesis
├── state.py              # Shared LangGraph state
├── graph.py              # Pipeline assembly
├── vector_store.py       # ChromaDB wrapper
├── api.py                # FastAPI backend
├── frontend.py           # Streamlit UI
├── eval.py               # Baseline comparison script
├── .env.example          # Environment variable template
└── requirements.txt
```

## Project Output
<img width="1748" height="721" alt="image" src="https://github.com/user-attachments/assets/f324e74f-ac1e-4918-988d-4942839d6b94" />
