"""
agriculture_news.py - Agricultural News & Policy Summarizer
"""

import os
import requests
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


def fetch_agriculture_summary(api_key, hf_token, state="Kerala", limit=10):
    """
    Fetch agriculture data from data.gov.in and summarize it using HuggingFace LLM.
    
    Args:
        api_key (str): API key for data.gov.in
        hf_token (str): HuggingFace API token
        state (str): Target state for summary (default: Kerala)
        limit (int): Number of records to fetch (default: 10)
    
    Returns:
        str: Structured summary of agriculture data
    """
   
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"  # Example: Mandi prices dataset
    params = {"api-key": api_key, "format": "json", "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    api_response = response.json()

    
    llm_endpoint = HuggingFaceEndpoint(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        huggingfacehub_api_token=hf_token
    )
    model = ChatHuggingFace(llm=llm_endpoint)

    
    system_prompt = """
    You are an agricultural policy assistant.

    Task:
    - Summarize the latest government agricultural policies, schemes, and mandi updates from the given API data.
    - Structure the answer in Markdown with clear headings and bullet points.
    - For each scheme/policy/market update, include:
      - Name
      - Objective
      - Key features
      - Latest updates
      - Benefits
    """

    
    query = f"""{system_prompt}

    Here is the API response data: {api_response}

    Now, summarize and organize this into agricultural schemes/policies/mandi updates for {state}.
    """

   
    response = model.invoke(query)

    return response.content


def get_agriculture_policy_and_market_summary(state="Kerala", limit=5):
    """
    High-level function to get agricultural policy and market summary.
    Loads API keys from environment variables or .env file.

    Env Vars:
        DATA_GOV_API_KEY
        HF_API_KEY
    """
    api_key = os.getenv("DATA_GOV_API_KEY")
    hf_token = os.getenv("HF_API_KEY")

    if not api_key or not hf_token:
        raise ValueError("❌ Missing API keys. Please set DATA_GOV_API_KEY and HF_API_KEY in environment or .env file.")

    return fetch_agriculture_summary(api_key, hf_token, state=state, limit=limit)

