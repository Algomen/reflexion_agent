from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition#
from langgraph.checkpoint.memory import MemorySaver

import dotenv

dotenv.load_dotenv()

LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

# Define tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm_with_tools = LLM.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(
    content="You are a helpful assistant that can perform arithmetic operations. "
            "You can add, multiply, and divide numbers. "
            "Use the tools provided to perform calculations."
)

# Define the assistant function that uses the LLM with tools
def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Create the state graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Use MemorySaver to save the state of the graph
memory = MemorySaver()
graph = builder.compile(checkpointer = memory)
config = {"configurable": {"thread_id": "1"}}

# Example usage
messages = graph.invoke({"messages": HumanMessage(content="Add 3 and 4.")}, config)
for message in messages["messages"]:
    print(message)

# messages = graph.invoke({"messages": HumanMessage(content="Now multiply that by 2.")}, config)