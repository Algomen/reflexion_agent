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
llm_with_tools = LLM.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a helpful assistant that can perform arithmetic operations. "
            "You can add, multiply, and divide numbers. "
            "Use the tools provided to perform calculations."
)

# Define the assistant function that uses the LLM with tools
def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# no-op node that should be interrupted on
def human_feedback(state: MessagesState):
    pass
    # user_input = input("Tell me how you want to update the state: ")
    # state["messages"].append(HumanMessage(content=user_input))
    # return state

# Create the state graph
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "human_feedback")

# Use MemorySaver to save the state of the graph
memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer = memory)



# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "5"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
    
# # Get user input
user_input = input("Tell me how you want to update the state: ")

# # We now update the state as if we are the human_feedback node
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()