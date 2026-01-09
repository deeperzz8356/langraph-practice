from typing import TypedDict, List
from mistralai import Mistral
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Mistral client with API key from .env
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("Set MISTRAL_API_KEY in your .env")

client = Mistral(api_key=api_key)

class AgentState(TypedDict):
    messages: List[dict]

def process(state: AgentState) -> AgentState:
    # Send messages to Mistral API
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=state["messages"]
    )
    ai_response = response.choices[0].message.content
    print(f"\nAI: {ai_response}")
    
    # Add assistant response to messages and return updated state
    state["messages"].append({
        "role": "assistant",
        "content": ai_response
    })
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Chat loop
print("Start chatting (type 'exit' to quit)")
user_input = input("You: ")
messages = []

while user_input.lower() != "exit":
    messages.append({
        "role": "user",
        "content": user_input
    })
    # Get updated state from agent
    result = agent.invoke({"messages": messages})
    messages = result["messages"]
    
    user_input = input("You: ")