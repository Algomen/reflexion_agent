from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
import dotenv

dotenv.load_dotenv()

# messages = [
#     HumanMessage(content="Give me the 3 best ideas for a new AI product."),
# ]
# LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=os.environ["GOOGLE_API_KEY"])
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

def multiply(x: int, y: int) -> int:
    return x * y

llm_with_tools = LLM.bind_tools([multiply])

def tool_calling_llm(state: MessagesState) -> MessagesState:
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph = StateGraph(MessagesState)
graph.add_node("tool_calling_llm", tool_calling_llm)
graph.add_edge(START, "tool_calling_llm")
graph.add_edge("tool_calling_llm", END)
image = graph.compile()

messages = image.invoke({"messages": HumanMessage(content="Multiply 2 and 3.")})
for message in messages["messages"]:
    print(message)

# response = LLM.invoke(messages)
# print(response.content)
