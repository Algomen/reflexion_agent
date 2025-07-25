from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativ

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
print(llm.invoke("Say hello!"))

# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# print([m.name for m in genai.list_models()])
