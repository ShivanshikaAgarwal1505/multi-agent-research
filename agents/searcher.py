# agents/searcher.py
import hashlib
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from state import ResearchState
from vector_store import search_similar, add_documents, get_collection_size

web_search_tool = TavilySearch(max_results=4)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ── Node 1: Web Search ────────────────────────────────────────────────────────

def web_search_node(state: ResearchState) -> dict:
    """Searches the web for each sub-question using Tavily."""
    all_results = []

    for question in state["sub_questions"]:
        try:
            results = web_search_tool.invoke({"query": question})

            # TavilySearch returns a list of dicts with a "content" key
            if isinstance(results, list):
                for r in results:
                    content = r.get("content", "") if isinstance(r, dict) else str(r)
                    if content:
                        all_results.append(content)
                        # Persist to ChromaDB so future queries benefit
                        doc_id = hashlib.md5(content.encode()).hexdigest()
                        add_documents([content], [doc_id])
            else:
                # Some versions return a single string summary
                all_results.append(str(results))

        except Exception as e:
            print(f"  [WEB SEARCH] Warning: search failed for '{question}': {e}")

    print(f"\n[WEB SEARCH] Retrieved {len(all_results)} documents")
    return {"raw_results": all_results}


# ── Node 2: Vector Search ─────────────────────────────────────────────────────

def vector_search_node(state: ResearchState) -> dict:
    """Searches ChromaDB for semantically similar content from previous runs."""
    all_results = []

    if get_collection_size() == 0:
        print("\n[VECTOR SEARCH] Collection empty — skipping (web search covers this)")
        return {"raw_results": []}

    for question in state["sub_questions"]:
        similar_docs = search_similar(question, n_results=3)
        all_results.extend(similar_docs)

    print(f"\n[VECTOR SEARCH] Found {len(all_results)} similar documents")
    return {"raw_results": all_results}


# ── Node 3: Summarizer ────────────────────────────────────────────────────────

def summarizer_node(state: ResearchState) -> dict:
    """Condenses raw retrieved documents into key bullet points."""
    if not state["raw_results"]:
        print("\n[SUMMARIZER] No raw results to summarize")
        return {"raw_results": ["No relevant information was retrieved."]}

    summarize_template = """Summarize the following retrieved documents into key facts \
relevant to this research query: {query}

Documents:
{documents}

Return a bullet-point list of the most important facts only. Be concise."""

    batch_size = 5
    summarized = []

    for i in range(0, len(state["raw_results"]), batch_size):
        batch = state["raw_results"][i : i + batch_size]
        try:
            response = llm.invoke(
                summarize_template.format(
                    query=state["query"],
                    documents="\n---\n".join(batch)
                )
            )
            summarized.append(response.content)
        except Exception as e:
            print(f"  [SUMMARIZER] Warning: batch {i} failed: {e}")

    print(f"\n[SUMMARIZER] Condensed into {len(summarized)} summary chunks")
    return {"raw_results": summarized}