from subprocess import list2cmdline
from typing import List, Sequence

from dotenv import load_dotenv
import os

load_dotenv()

from chains import generate_tweet, reflection_chain, generation_chain
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, MessageGraph

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    return generation_chain.invoke({"messages": state})

def reflection_node(state: Sequence[BaseMessage]) -> list[BaseMessage]:
    res = reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return REFLECT
    return END

builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())




if __name__ == "__main__":
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    print(graph.invoke(inputs))