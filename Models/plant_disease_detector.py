# -*- coding: utf-8 -*-
"""Plant_disease_detector.ipynb (Cleaned)"""
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",  
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# --- working_image_agent.py ---
import os, base64, requests, mimetypes
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# --- Gemini direct call config ---
API_KEY = GOOGLE_API_KEY
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

def encode_image_to_base64(image_path_or_url: str):
    if image_path_or_url.startswith("http"):
        r = requests.get(image_path_or_url, timeout=20)
        r.raise_for_status()
        data = r.content
        mime = r.headers.get("content-type", None)
    else:
        if not os.path.exists(image_path_or_url):
            raise FileNotFoundError(f"{image_path_or_url} not found")
        with open(image_path_or_url, "rb") as f:
            data = f.read()
        mime = mimetypes.guess_type(image_path_or_url)[0]
    if not mime:
        raise ValueError("Could not determine MIME type")
    return base64.b64encode(data).decode("utf-8"), mime

def analyze_image_with_gemini(prompt: str, image_url: str) -> str:
    try:
        b64, mime = encode_image_to_base64(image_url)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime, "data": b64}}
                    ]
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        cand = j.get("candidates", [])
        if not cand:
            return "No text candidate returned by Gemini."
        parts = cand[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                return p["text"]
        return "Unexpected Gemini response structure."
    except Exception as e:
        return f"Error calling Gemini: {e}"

def call_gemini(inputs: dict) -> str:
    return analyze_image_with_gemini(inputs["prompt"], inputs["image_url"])

image_chain_gemini = (
    RunnableParallel({
        "prompt": RunnableLambda(lambda x: x["user_query"]),
        "image_url": RunnableLambda(lambda x: x["image_url"])
    })
    | RunnableLambda(call_gemini)
)

def image_chain_gemini_fn(user_query: str, image_url: str) -> str:
    return image_chain_gemini.invoke({"user_query": user_query, "image_url": image_url})

class ImageInput(BaseModel):
    user_query: str = Field(..., description="instruction about the image")
    image_url: str   = Field(..., description="URL of the image")

image_tool = StructuredTool.from_function(
    func=image_chain_gemini_fn,
    name="ImageDescription",
    description="Analyze image from URL and return plain-text analysis.",
    args_schema=ImageInput,
    response_format="content",
)

SYSTEM_PROMPT = """
You are an expert plant pathologist.
You are given two pieces of information:
1. The raw analysis of the leaf image.
2. The user’s query.

You must return a complete, farmer-friendly report that includes:
- Detected disease name (clear heading).
- Cure or treatment methods.
- Preventive measures (precautions).
- Any other useful agricultural advice.
"""

def format_inputs(inputs: dict):
    analysis = inputs["analysis"]
    if isinstance(analysis, dict) and "content" in analysis:
        analysis = analysis["content"]

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {inputs['user_query']}\n\nImage analysis: {analysis}")
    ]

image_to_report_chain = (
    RunnableParallel({
        "analysis": image_tool,
        "user_query": RunnableLambda(lambda x: x["user_query"]),
    })
    | RunnableLambda(format_inputs)
    | llm
)



def run_pipeline(image_url: str, user_query: str):
    return image_to_report_chain.invoke({
        "user_query": user_query,
        "image_url": image_url
    })



# --- disease_report_chain.py ---

import json

disease_schema = {
    "title": "DiseaseReport",
    "type": "object",
    "properties": {
        "detected_disease": {"type": "string"},
        "cure_and_treatment_methods": {
            "type": "array",
            "items": {"type": "string"}
        },
        "preventive_measures": {
            "type": "array",
            "items": {"type": "string"}
        },
        "other_useful_agricultural_advice": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "detected_disease",
        "cure_and_treatment_methods",
        "preventive_measures",
        "other_useful_agricultural_advice"
    ]
}

structured_llm = llm.with_structured_output(disease_schema)

SYSTEM_PROMPT = """
You are an expert plant pathologist.
You will be given:
1. A raw analysis of a leaf image.
2. The user’s query.

Your task is to generate a structured JSON object strictly following this schema:

{
  "detected_disease": "string",
  "cure_and_treatment_methods": ["string", "string", "..."],
  "preventive_measures": ["string", "string", "..."],
  "other_useful_agricultural_advice": ["string", "string", "..."]
}

Rules:
- Output must be a valid JSON object, not wrapped in markdown or text.
- Do not include any extra fields or text outside the schema.
- Arrays must have at least 2–3 concise, clear bullet-style strings.
- "detected_disease" must always be specific.
"""

def format_inputs(inputs: dict):
    analysis = inputs["analysis"]
    if isinstance(analysis, dict) and "content" in analysis:
        analysis = analysis["content"]

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {inputs['user_query']}\n\nImage analysis: {analysis}")
    ]

image_to_report_chain = (
    RunnableParallel({
        "analysis": image_tool,
        "user_query": RunnableLambda(lambda x: x["user_query"]),
    })
    | RunnableLambda(format_inputs)
    | structured_llm
)

def run_report(user_query: str, image_url: str):
    result = image_to_report_chain.invoke({
        "user_query": user_query,
        "image_url": image_url
    })
    return result

# if __name__ == "__main__":
#     query = "Tell me about the disease in this leaf and how to treat it."
#     url = "https://extension.umn.edu/sites/extension.umn.edu/files/marssonia-leaf-spot-on-euonymus-grabowski.jpg"
#     output = run_report(query, url)
#     print(json.dumps(output, indent=2))

def run_pipeline(image_url: str, user_query: str = None):
    if not user_query:
        user_query = "Explain the detected disease, its cure, and preventive measures."
    return run_report(user_query, image_url)


