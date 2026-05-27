import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# SentenceTransformer embedding model — runs locally, no API key needed
EMBED_MODEL = "all-MiniLM-L6-v2"

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# PersistentClient saves the DB to disk so it survives restarts
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="research_docs",
    embedding_function=embed_fn
)


def add_documents(texts: list[str], ids: list[str]) -> None:
    """
    Store text chunks in the vector DB.
    ChromaDB auto-converts text → 384-dim vectors using the embedding model.
    """
    # Filter out empty strings to avoid ChromaDB errors
    pairs = [(t, i) for t, i in zip(texts, ids) if t.strip()]
    if not pairs:
        return
    texts_clean, ids_clean = zip(*pairs)
    collection.add(documents=list(texts_clean), ids=list(ids_clean))


def search_similar(query: str, n_results: int = 5) -> list[str]:
    """
    Find the n most semantically similar documents to the query.
    Works on MEANING, not keyword matching — so 'how do vectors work'
    finds docs about 'embedding similarity' even without shared words.
    """
    # Guard: don't query an empty collection
    count = collection.count()
    if count == 0:
        return []

    actual_n = min(n_results, count)
    results = collection.query(
        query_texts=[query],
        n_results=actual_n
    )
    return results["documents"][0]


def get_collection_size() -> int:
    """Returns how many documents are currently stored."""
    return collection.count()


# ----- Quick sanity test -----
if __name__ == "__main__":
    print("Testing ChromaDB setup...")

    add_documents(
        texts=[
            "RAG systems improve factual accuracy of LLMs by retrieving relevant context",
            "Vector databases store embeddings for fast semantic similarity search",
            "LangGraph allows building multi-agent pipelines as directed graphs",
            "Groq provides ultra-fast LLM inference using LPU hardware",
        ],
        ids=["doc1", "doc2", "doc3", "doc4"]
    )

    print(f"Documents in DB: {get_collection_size()}")

    query = "how do embeddings work?"
    print(f"\nQuery: '{query}'")
    print("Most similar docs:")
    for doc in search_similar(query, n_results=2):
        print(f"  → {doc}")