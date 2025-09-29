import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from tavily import TavilyClient

SYSTEM_PROMPT = """
You are an *Agricultural Policy Assistant*.
Your task:
- Summarize the *latest government agricultural schemes, policies, and updates* from the context given.
- Structure the answer in *Markdown* with clear headings and bullet points.
- For each scheme, include:
  - Name
  - Objective
  - Features
  - Updates
  - Benefits
DO NOT say "refer to the above text". Instead, return a *self-contained answer*.
"""

def summarize_government_schemes(state: str, category: str = "agriculture"):
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"latest {category} government schemes and policies in {state} India"
    results = tavily.search(query, max_results=5)
    context = "\n\n".join([r["content"] for r in results["results"]])

    llm_endpoint = HuggingFaceEndpoint(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"State: {state}\n\nContext:\n{context}\n\nNow summarize the schemes in structured format.")
    ])
    query = prompt_template.format_messages()
    response = llm.invoke(query)
    return response.content
