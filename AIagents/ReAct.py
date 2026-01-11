from typing import Sequence, TypedDict, List,Dict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage,AIMessage,HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os

load_dotenv()
# Initialize Mistral client with API key from .env
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int):
    """This is addition function"""
    return a+b
tools=[add]
#connect with mistral api in env file
api_key=os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("Set MISTRAL_API_KEY in your .env")

model=ChatMistralAI(model='mistral-large-latest', api_key=api_key).bind_tools(tools)
def model_call(state:AgentState)->AgentState:
    system_prompt=SystemMessage(content="You are my AI asistant,please help me" )
    response=model.invoke(
        [system_prompt]+state['messages']
    )
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages=state['messages']
    last_message=messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph=StateGraph(AgentState)    
graph.add_node("model_call",model_call)
graph.add_edge(START,"model_call")
tool_node=ToolNode(tools=tools)
graph.add_node("tool_node",tool_node)

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue":"tool_node",
        "end":END
    }
)
graph.add_edge("tool_node","model_call")
agent=graph.compile()

def print_stream(stream):
    for s in stream:
        if 'messages' in s:
            messages = s['messages']
            last_message = messages[-1]
            
            if isinstance(last_message, HumanMessage):
                print(f"\n🧑 Human: {last_message.content}")
            elif isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    print(f"\n🤖 AI calling tool: {last_message.tool_calls[0]['name']}")
                    print(f"   Args: {last_message.tool_calls[0]['args']}")
                else:
                    print(f"\n🤖 AI: {last_message.content}")
            elif isinstance(last_message, ToolMessage):
                print(f"\n🔧 Tool Result: {last_message.content}")
        print("-" * 50)

inputs={"messages":[HumanMessage(content="what is 123 + 786? and what is your age?")]}
print("\n" + "="*50)
print("Starting ReAct Agent")
print("="*50)
print_stream(agent.stream(inputs,stream_mode="values"))