from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import os

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user. Always provide detailed recommendations, including requests for length, virality, style, etc"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a twitter techie influencer assitant tasked with writing excellent twitter posts. Generate the best twitter post possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=os.environ["GOOGLE_API_KEY"])
generation_chain = generation_prompt | LLM
reflection_chain = reflection_prompt | LLM

def generate_tweet(user_input: str) -> str:
    messages = [
        ("user", user_input),
    ]
    return generation_chain.invoke({"messages": messages})