from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import dotenv

dotenv.load_dotenv()

LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

def multiply(x: int, y: int) -> int:
    """Multiplies two integers."""
    return x * y

llm_with_tools = LLM.bind_tools([multiply])

def tool_calling_llm(state: MessagesState) -> MessagesState:
    return {"messages": llm_with_tools.invoke(state["messages"])}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3.")})
for message in messages["messages"]:
    print(message)

# response = LLM.invoke(messages)
# print(response.content)
