from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch          # updated import
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",                  # ← updated model name
    temperature=0
)

tools = [TavilySearch(max_results=3)]              # ← updated class

agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [("user", "What are the latest breakthroughs in RAG systems in 2024?")]
})

print("\n=== FINAL ANSWER ===")
print(result["messages"][-1].content)

print("\n=== AGENT TRACE ===")
for msg in result["messages"]:
    print(f"[{msg.__class__.__name__}]: {str(msg.content)[:200]}")