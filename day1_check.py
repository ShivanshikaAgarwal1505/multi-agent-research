"""
Run this to verify your Day 1 setup is complete.
All three checks should pass before starting Day 2.
"""
import sys

def check_env():
    print("[ ] Checking environment variables...")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not groq_key:
        print("  ✗ GROQ_API_KEY missing in .env")
        return False
    if not tavily_key:
        print("  ✗ TAVILY_API_KEY missing in .env")
        return False
    
    print("  ✓ Both API keys found")
    return True


def check_groq():
    print("[ ] Checking Groq connection...")
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)  # ← updated
    response = llm.invoke("Reply with exactly: OK")
    
    if "OK" in response.content:
        print("  ✓ Groq API working")
        return True
    else:
        print(f"  ✗ Unexpected response: {response.content}")
        return False

def check_chromadb():
    print("[ ] Checking ChromaDB...")
    from vector_store import add_documents, search_similar, get_collection_size
    
    add_documents(["test document about neural networks"], ["check_test_1"])
    results = search_similar("machine learning", n_results=1)
    
    if results:
        print(f"  ✓ ChromaDB working ({get_collection_size()} docs in collection)")
        return True
    else:
        print("  ✗ ChromaDB search returned nothing")
        return False


def check_tavily():
    print("[ ] Checking Tavily search...")
    from langchain_tavily import TavilySearch      # ← updated import
    from dotenv import load_dotenv
    load_dotenv()
    
    tool = TavilySearch(max_results=1)             # ← updated class
    results = tool.invoke({"query": "LangGraph multi-agent"})
    
    if results and len(results) > 0:
        print(f"  ✓ Tavily working (got {len(results)} result)")
        return True
    else:
        print("  ✗ Tavily returned no results")
        return False

if __name__ == "__main__":
    print("=" * 45)
    print("       Day 1 Setup Verification")
    print("=" * 45)
    
    checks = [check_env, check_groq, check_chromadb, check_tavily]
    results = []
    
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append(False)
    
    print("\n" + "=" * 45)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} checks passed — ready for Day 2!")
    else:
        print(f"✗ {total - passed}/{total} checks failed — fix above before continuing")
        sys.exit(1)