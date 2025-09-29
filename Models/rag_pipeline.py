from langchain.tools import tool

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate ,load_prompt , MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from langchain.output_parsers import StructuredOutputParser , ResponseSchema
from langchain.schema.runnable import RunnableParallel , RunnableSequence , RunnableLambda , RunnablePassthrough , RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from langchain_community.document_loaders import DirectoryLoader , TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
import langchain.retrievers as retrievers

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import WikipediaRetriever

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr
from typing import TypedDict, Annotated , List , Optional
from pydantic import BaseModel, Field , EmailStr

# from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter , Language

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
import base64
import mimetypes
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.runnables import RunnableParallel, RunnableLambda


# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",   # or "gemini-1.5-pro" if you want better reasoning
    google_api_key="AIzaSyDGfudxJ5BBLOzGokYA3Vm7qlT3Ef6VOsw",
    temperature=0.3,            # adjust for more/less creativit
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

# defining
parser = StrOutputParser()

## we need to concat the results from retriever
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

from googletrans import Translator
import logging
from langchain.tools import tool

logger = logging.getLogger("rag")
translator = Translator()

# Global variable
original_lang = ""

@tool
def preprocess_query(query: str) -> str:
    """
    Detects language, translates query to English for RAG,
    and returns both (query_for_rag, original_language).
    Also stores detected language in global variable original_lang.
    """
    global original_lang  # <-- Declare global

    query_for_rag = query
    original_language = "en"

    try:
        detected = translator.detect(query)
        original_language = detected.lang

        if original_language != "en":
            logger.info(f"Detected language: {original_language}. Translating query to English.")
            translated_query = translator.translate(query, src=original_language, dest="en").text
            query_for_rag = translated_query
            logger.info(f"Translated query: {translated_query}")
        else:
            logger.info("Query is already in English. No translation needed.")
            query_for_rag = query

    except Exception as e:
        logger.warning(f"Translation failed for query '{query}': {e}. Proceeding with original query.")
        query_for_rag = query

    # Store in global variable
    original_lang = original_language

    return query_for_rag 


@tool
def postprocess_answer(answer: str) -> str:
    """
    Translates the final LLM answer back into the user's original language.
    """
    try:
        if original_lang != "en":
            translated_answer = translator.translate(answer, src="en", dest=original_lang).text
            logger.info(f"Translated answer back to {original_lang}")
            return translated_answer
        return answer
    except Exception as e:
        logger.warning(f"Failed to translate answer: {e}. Returning original English answer.")
    markdown_prompt = f"""
You are a smart translator, just conver the raw text that you are receiving to the markdown format, donot change the input language or any word, just convert it into the markdown format\n {answer}.This is the final output the user will see so only provide details if it is an agricultural doubt or query , if it a general conversational question like 'hi','how are you','what are doing', etc answer as in a humanized way and tell tham you are an agriculutural chatbot , here to help them with there queries.
"""
    return llm.invoke(markdown_prompt)
    
    

# result = preprocess_query.invoke("बाग में लगाने के लिए जनवरी और फरवरी के महीने सबसे अच्छे होते हैं, क्योंकि यह पौधों के लिए आराम का समय होता है. ")
# print(result)

docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),

]

## this is trying to access the image also but it is taking a too much too load the document so not using this
# from langchain_community.document_loaders import UnstructuredPDFLoader
# import pdfminer

# loader = UnstructuredPDFLoader(
#     "sample.pdf",
#     strategy="hi_res",
#     extract_images=True  # ✅ will also pull images
# )
# docs = loader.load()

## just loading the documents
from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("sample.pdf")
# docs = loader.load()

## converting into chunks
## converting the pdf into the chunks, metadata is lost
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter , Language

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # size of each chunk (tokens/characters)
    chunk_overlap=100,     # overlap between chunks to preserve context
    separators=[           # prioritize splitting by semantic boundaries
        "\n\n",            # paragraph
        "\n",              # line break
        ". ",              # sentence
        "?", "!", ",",     # punctuation
        " "                # fallback: split by space
    ]
)
#semantic_splitter = SemanticChunker(embedding_model)
# Extract text content from the documents
text_content = "".join([doc.page_content for doc in docs])

# chunks = semantic_splitter.split_documents(docs)
chunks = splitter.create_documents([text_content])


## so instaead of using of semantic search we will be using the RecursiveCharacterTextSplitter + smart separators + reranking

##defining from the vectorstore
vectorstore = Chroma(
    embedding_function=embedding_model, # Use the HuggingFace embeddings
    persist_directory='my_chroma_db',
    collection_name='sample'
)

vectorstore.add_documents(chunks)

## retriever

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 4, "lambda_mult": 0.65}),
    llm=llm
)
## will give the pagecontent and the metadata

## duck duck go

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools.retriever import create_retriever_tool

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

duckduckgo = search_tool.invoke("What is LangChain?")
logging.info(duckduckgo)

## web search tool

google_api_key = "AIzaSyD4r-wslwfcGqn8oguhJojp_Ywhhzh1lnY"
google_cse_id = "0436a46c86379439f"

import os
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["GOOGLE_CSE_ID"] = google_cse_id

search = GoogleSearchAPIWrapper()

# Wrap as a Tool
web_search_tool = Tool( # Use Tool from langchain_core.tools
    name="detailWebSearch", # Renamed to match the intended tool name in the agent
    description="Search Google for best recent results.",
    func=search.run
)

prompt_context = PromptTemplate(
    template="you are a llm as a judge and a summarizer so based on the context output you are receiving from the vector store database. "
)

context_chain = (
    RunnableLambda(lambda x: multiquery_retriever.get_relevant_documents(x["user_query"]))
    | RunnableLambda(format_docs)
)
search = GoogleSearchAPIWrapper()

def websearch_fn(query: str) -> str:
    return search.run(query)

prompt = PromptTemplate(
    template="You are a summarizer tool for a farmer assistant chatbot. Based on the following search results:\n\n{user_query}\n\nAnswer the user query.",
    input_variables=["user_query"]
)
web_chain = (
    {"user_query": lambda x: websearch_fn(x["user_query"])}
    | prompt
    | llm
    | parser
)

image_chain = (
    RunnableParallel({
        "prompt": lambda x: x["user_query"],
        "image_url": lambda x: x["image_url"]
    })
    | (lambda d: analyze_image_with_gemini(d["prompt"], d["image_url"]))
)

parallel_chain = RunnableParallel(
    {
        "context" : context_chain,
        "web_result" : web_chain,
        "user_query" : RunnablePassthrough(),
        "image_description" : image_chain
    }
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are an expert farming assistant for kerala farmers. "
     "Your goal is to help farmers by providing accurate, reliable, and easy-to-understand answers. "
     "Always use the following sources when forming your response:\n\n"
     "1. Chat History → to maintain continuity with past conversation.\n"
     "2. Vector Database Context → trusted agricultural manuals, crop disease guides, and government advisories.\n"
     "3. Web Search Results → recent and up-to-date information from the internet.\n\n"
     "4. IMAGE Description - this is the description of the image given by user consider this image description for proper answer / if it is given error that no image is given then skip this variable"
     "### Guidelines:\n"
     "- First, look into the vector database context for authoritative information.\n"
     "- Use web search results only to supplement or update missing details.\n"
     "- Maintain conversational tone, but keep answers factual.\n"
     "- If the answer is unclear or unavailable, politely say you don't know.\n"
     "- Do not repeat irrelevant details from search/context; filter noise.\n"
     #"- Summarize information into 3–5 clear bullet points whenever possible."
     "- give details answer like and try to cover every aspect of that query.\n"
     "📘 Vector Database Context:\n{context}\n\n🌐"
     "Web Search Results:\n{web_result}\n\n"
     "Image Description:\n{image_description}\n\n"
    ),
    ("human",  "this is the User Query so answer accordingly: {user_query}"),


])

# Wrap preprocess_query as a chain step
preprocess_chain = RunnableLambda(
    lambda x: {
        "user_query": preprocess_query(x["user_query"])[0],   # overwrite with English query
        "original_language": preprocess_query(x["user_query"])[1],
        "chat_history": x.get("chat_history", ""),
        "image_url":x.get("image_url" , "")
        # "context": x.get("context", ""),
        # "web_result": x.get("web_result", "")
    }
)

"""##We designed our pipeline to handle both multilingual queries natively and fallback translation when confidence is low. This ensures farmers get accurate answers whether they type in English, Malayalam, or a mix"""

import requests
from langchain.tools import StructuredTool

def query_weather_ambee(lat: float, lon: float) -> str:
    """
    Rich weather data from Ambee's API.
    Covers precipitation, UV index, visibility, etc.
    """
    try:
        url = f"https://api.ambeedata.com/weather/latest/by-lat-lng?lat={lat}&lng={lon}"
        headers = {"x-api-key": "45823d65a2c21447c538dadc9c732053f70e72c1f11f43443fdf6c7cedf7bd15"}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return f"Error: {resp.text}"

        d = resp.json().get("data", {})
        parts = [
            f"Summary: {d.get('summary', 'N/A')}",
            f"Temperature: {d.get('temperature', 'N/A')}°F",
            f"Feels-like: {d.get('apparentTemperature', 'N/A')}°F",
            f"Humidity: {d.get('humidity', 'N/A')*100 if isinstance(d.get('humidity'), float) else d.get('humidity')}%",
            f"Dew Point: {d.get('dewPoint', 'N/A')}",
            f"Pressure: {d.get('pressure', 'N/A')} hPa",
            f"Wind: {d.get('windSpeed', 'N/A')} mph (gust: {d.get('windGust', 'N/A')})",
            f"Cloud Cover: {d.get('cloudCover', 'N/A')*100 if isinstance(d.get('cloudCover'), float) else d.get('cloudCover')}%",
            f"Visibility: {d.get('visibility', 'N/A')} km",
            f"UV Index: {d.get('uvIndex', 'N/A')}",
            f"Ozone: {d.get('ozone', 'N/A')}",
            f"Precipitation: {d.get('precipIntensity', 'N/A')} mm/h, Type: {d.get('precipType', 'N/A')}, Prob: {d.get('precipProbability', 'N/A')*100 if isinstance(d.get('precipProbability'), float) else d.get('precipProbability')}%",
        ]
        return "\n".join(parts)
    except Exception as e:
        return f"Error parsing data: {e}"

# Wrap as a StructuredTool

# print(weather_tool.invoke({"lat": 22.3149, "lon": 87.3105}))

# ✅ Correct Pydantic schema
class WeatherInput(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lon: float = Field(..., description="Longitude of the location")


# ✅ Proper tool definition
weather_tool = StructuredTool.from_function(
    func=query_weather_ambee,
    name="WeatherData",
    description="Fetch detailed weather data (temp, rain, UV, visibility, etc.). Provide latitude and longitude as input.",
    args_schema=WeatherInput,
)

from langchain.tools import StructuredTool

web_tool_duckduckgo = StructuredTool.from_function(
    func=search_tool.invoke,
    name="WebSearch",
    description="Fetches information from the duckduckgo web for the given query."
)


context_chain = (
    RunnableLambda(lambda x: multiquery_retriever.get_relevant_documents(x["user_query"]))
    | RunnableLambda(format_docs)
)



def context_fn(user_query: str) -> str:
    return context_chain.invoke({"user_query": user_query})


class ContextInput(BaseModel):
    user_query: str = Field(..., description="the query for which the inforation needs to be retrieve from the vector database")


context_tool = StructuredTool.from_function(
    func=context_fn,
    name="retrieve_knowledge_from_the_vectordatabase",
    description="to retieve the relevent information from the trustfull source vector database",
    args_schema=ContextInput,
    #response_format="content",   # IMPORTANT — makes LangChain package return as function content part
)


# user_query_tool = StructuredTool.from_function(
#     func=lambda x: x["user_query"],  # passthrough wrapper
#     name="UserQuery",
#     description="Simply returns the user query as-is."
# )

# --- working_image_agent.py ---
import os, base64, requests, mimetypes
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor

# --- Set your LLM wrapper here (Gemini via langchain_google_genai) ---
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="YOUR_KEY")
  # <-- YOUR Gemini LLM instance here

# --- Gemini direct call config (only used if you call Gemini directly) ---
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDGfudxJ5BBLOzGokYA3Vm7qlT3Ef6VOsw")
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
    """
    Calls Gemini directly with embedded image. Returns a plain string (analysis).
    If you prefer to let the agent call your LLM via LangChain, you can
    instead call your 'llm' wrapper — but the return must be a string.
    """
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
        # Extract first text part found (robust)
        parts = cand[0].get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                return p["text"]
        return "Unexpected Gemini response structure."
    except Exception as e:
        return f"Error calling Gemini: {e}"

# --- Runnable chain: (map user input -> prompt,image_url) then call Gemini ---
def call_gemini(inputs: dict) -> str:
    # inputs expected: {"prompt": "...", "image_url": "..."}
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
    response_format="content",   # IMPORTANT — makes LangChain package return as function content part
)


def build_agent_prompt(chat_history, domain, query, role, image_url=None):
    """
    Build a detailed system prompt for the LLM agent with given variables.
    """

    prompt = f"""
    You are an intelligent assistant with the role of: {role}.
    Your primary domain of expertise is: {domain}.
    You are given the following context:
    - Chat History: {chat_history}
    - Current User Query: "{query}"
    """

    # Include image if provided
    if image_url:
        prompt += f"- Image URL provided: {image_url}\n"

    prompt += """
    You have access to the following tools:
    1. weather_tool: Provides weather-related information (temperature, rainfall, humidity, forecast, etc.).
    2. image_tool: Generates or edits images based on textual descriptions or modifies existing images.
    - If you see an image URL in the query or provided by backend, ALWAYS use image_tool first to analyze/describe the image.
    3. web_tool: Searches the internet for the latest and most relevant information.
    ⚠ If you cannot find anything relevant in your knowledge, chat history, or other tools, ALWAYS call web_tool.
    4. context_tool: Retrieves relevant context or background information from stored knowledge or documents.
    ### Instructions for the Agent:
    - Always leverage the tools when necessary to enhance your response.
    - If an image URL is provided, call image_tool to extract its description before proceeding.
    - Prefer domain-specific knowledge first, then chat history, then context_tool.
    - If the above sources do not provide sufficient information, *must call web_tool*.
    - Present the answer in a clear, structured, and helpful way to the user.
    - Be concise but detailed enough to fully resolve the query.
    Now, based on the given variables, analyze the user query and decide how to proceed.
    """

# import re
# from typing import List, Any, Optional

# from langchain.prompts import ChatPromptTemplate
# from langchain.prompts.chat import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     AIMessagePromptTemplate,
# )
# from langchain.agents import create_openai_functions_agent, AgentExecutor

# # ---------------------------------------------------------------------
# # NOTE: These must already exist in your environment:
# # - query_weather_ambee(lat: float, lon: float) -> str
# # - weather_tool, image_tool, web_tool_duckduckgo, context_tool in tools
# # - llm (your LLM instance, e.g., Gemini or OpenAI)
# # If you don't have query_weather_ambee, use your own weather function.
# # ---------------------------------------------------------------------

# def detect_lat_lon(query: str) -> (Optional[float], Optional[float]):
#     """
#     Order-independent detection of latitude and longitude from a text query.
#     Returns (lat, lon) as floats if both found, otherwise (None, None).
#     """
#     lat_match = re.search(r"(?:lat(?:itude)?)[^\d-]*([+-]?\d+(?:\.\d+)?)", query, re.I)
#     lon_match = re.search(r"(?:lon(?:gitude)?)[^\d-]*([+-]?\d+(?:\.\d+)?)", query, re.I)
#     if lat_match and lon_match:
#         try:
#             return float(lat_match.group(1)), float(lon_match.group(1))
#         except ValueError:
#             return None, None
#     return None, None


# def multi_query_executor(agent_executor: AgentExecutor, query: str):
#     """
#     Split a compound query into sub-queries (splits on ' and ' and commas,
#     but avoids splitting commas that immediately precede URLs), then:
#       - If a subquery contains both lat & lon, call query_weather_ambee directly.
#       - Otherwise invoke the agent_executor for that subquery.
#     Returns a list of {subquery: answer}.
#     """
#     # split on ' and ' or commas that are NOT followed by a URL (avoid splitting before URLs)
#     sub_queries = re.split(r'\band\b|,(?!\s*https?://)', query, flags=re.I)
#     sub_queries = [q.strip() for q in sub_queries if q.strip()]

#     results = []
#     for q in sub_queries:
#         lat, lon = detect_lat_lon(q)
#         if lat is not None and lon is not None:
#             # direct call to your weather function (bypass schema confusion)
#             try:
#                 answer = query_weather_ambee(lat, lon)
#             except Exception as e:
#                 answer = f"Error calling weather function: {e}"
#             results.append({q: answer})
#         else:
#             # run the agent for this subquery
#             try:
#                 agent_response = agent_executor.invoke({"input": q})
#                 # agent_response may be a dict or string depending on your executor
#                 if isinstance(agent_response, dict):
#                     answer = agent_response.get("output") or agent_response.get("result") or str(agent_response)
#                 else:
#                     answer = str(agent_response)
#             except Exception as e:
#                 answer = f"Agent error: {e}"
#             results.append({q: answer})
#     return results


# def build_agent_prompt(
#     chat_history: List[str],
#     domain: str,
#     query: str,
#     role: str,
#     tools: List[Any],
#     llm: Any,
#     image_url: Optional[str] = None,
#     verbose: bool = False
# ) -> str:
#     """
#     Builds a ChatPromptTemplate including {agent_scratchpad}, creates the agent,
#     runs multi_query_executor, and returns a single flattened string result.
#     Parameters:
#       - chat_history: list of prior messages (strings) to include for context
#       - domain: string describing the domain (e.g., "Agriculture & AI")
#       - query: the user query (may contain multiple sub-questions)
#       - role: assistant role description
#       - tools: list of tool objects (e.g., weather_tool, image_tool, ...)
#       - llm: your LLM instance
#       - image_url: optional image URL (if provided will be included in system prompt)
#       - verbose: if True, AgentExecutor will be verbose
#     Returns:
#       - combined textual answer (string)
#     """

#     # Build the system text
#     chat_history_text = "\n".join(chat_history) if chat_history else "No prior chat history."
#     system_text = (
#         f"You are an intelligent assistant with the role: {role}.\n"
#         f"Primary domain: {domain}.\n"
#         f"Chat History:\n{chat_history_text}\n"
#         f"Current User Query: \"{query}\"\n"
#     )
#     if image_url:
#         query += f" ,Image URL provided explain the image in detail: {image_url}\n"

#     system_text += (
#         "\nYou have access to these tools:\n"
#         "1) weather_tool: Provides weather info (temperature, humidity, forecast, etc.)\n"
#         "2) image_tool: Analyze/describe images from URLs\n"
#         "3) web_tool: Search the web for latest info\n"
#         "4) context_tool: Retrieve stored knowledge from vector DB\n\n"
#         "Instructions:\n"
#         "- If an image URL is present, analyze it with image_tool before other steps.\n"
#         "- Prefer domain knowledge, then chat history, then context_tool.\n"
#         "- If missing info, use web_tool.\n"
#         "- Use tools when helpful. Provide clear, structured answers.\n"
#     )

#     # Create ChatPromptTemplate with required agent_scratchpad slot
#     system_msg = SystemMessagePromptTemplate.from_template(system_text)
#     human_msg = HumanMessagePromptTemplate.from_template("{input}")
#     ai_msg = AIMessagePromptTemplate.from_template("{agent_scratchpad}")
#     prompt = ChatPromptTemplate.from_messages([system_msg, human_msg, ai_msg])

#     # Create agent and executor
#     agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

#     # Run multi-query executor and flatten results
#     final_results = multi_query_executor(agent_executor, query)

#     # Build a single response string
#     parts = []
#     for item in final_results:
#         for q_text, ans in item.items():
#             # If answer is structured (dict), try to extract likely fields
#             if isinstance(ans, dict):
#                 # prefer a readable "output" key if present
#                 answer_text = ans.get("output") or ans.get("result") or str(ans)
#             else:
#                 answer_text = str(ans)
#             parts.append(f"Q: {q_text}\nA: {answer_text}")

#     combined = "\n\n".join(parts)
#     return combined


# # ---------------------------
# # Example usage (replace placeholders with your real objects)
# # ---------------------------
# url = "https://extension.umn.edu/sites/extension.umn.edu/files/marssonia-leaf-spot-on-euonymus-grabowski.jpg"
tools = [weather_tool, image_tool, web_tool_duckduckgo, context_tool ,web_search_tool]


import re
from typing import List, Any, Optional

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.agents import create_openai_functions_agent, AgentExecutor


domain_dict={
    "main":"AI and Agriculture" , 
    "secondary":"Weather and Agricultural",
    "third":"AI and Agriculture"

}
def detect_lat_lon(query: str) -> tuple[Optional[float], Optional[float]]:
    lat_match = re.search(r"(?:lat(?:itude)?)[^\d-]*([+-]?\d+(?:\.\d+)?)", query, re.I)
    lon_match = re.search(r"(?:lon(?:gitude)?)[^\d-]*([+-]?\d+(?:\.\d+)?)", query, re.I)
    if lat_match and lon_match:
        try:
            return float(lat_match.group(1)), float(lon_match.group(1))
        except ValueError:
            return None, None
    return None, None


def multi_query_executor(agent_executor: AgentExecutor, query: str):
    sub_queries = re.split(r'\band\b|,(?!\s*https?://)', query, flags=re.I)
    sub_queries = [q.strip() for q in sub_queries if q.strip()]

    results = []
    for q in sub_queries:
        lat, lon = detect_lat_lon(q)
        if lat is not None and lon is not None:
            try:
                answer = query_weather_ambee(lat, lon)
            except Exception as e:
                answer = f"Error calling weather function: {e}"
            results.append({q: answer})
        else:
            try:
                agent_response = agent_executor.invoke({"input": q})
                if isinstance(agent_response, dict):
                    answer = agent_response.get("output") or agent_response.get("result") or str(agent_response)
                else:
                    answer = str(agent_response)
            except Exception as e:
                answer = f"Agent error: {e}"
            results.append({q: answer})
    return results


def build_agent_prompt(
    chat_history: List[str],
    domain: str,
    query: str,
    role: str,
    image_url: Optional[str] = None,
    verbose: bool = True
) -> str:
    query = preprocess_query(query)
    chat_history_text = "\n".join(chat_history) if chat_history else "No prior chat history."
    system_text = (
        f"You are an intelligent assistant with the role: {role}.\n"
        f"Primary domain: {domain}.\n"
        f"Chat History:\n{chat_history_text}\n"
        f"Current User Query: \"{query}\"\n"
    )
    if image_url:
        query += f" ,Image URL provided explain the image in detail: {image_url}\n"

    system_text += (
        "\nYou have access to these tools:\n"
        "1) weather_tool: Provides weather info (temperature, humidity, forecast, etc.)\n"
        "2) image_tool: Analyze/describe images from URLs\n"
        "3) web_tool: Search the web for latest info\n"
        "4) context_tool: Retrieve stored knowledge from vector DB\n\n"
        "Instructions:\n"
        "- If an image URL is present, analyze it with image_tool before other steps.\n"
        "- Prefer domain knowledge, then chat history, then context_tool.\n"
        "- If missing info, use web_tool.\n"
        "- Use tools when helpful. Provide clear, structured answers.\n"
        "- The Final Answer that you respond with should be STRICTLY in MARKDOWN format ONLY\n"
    )

    system_msg = SystemMessagePromptTemplate.from_template(system_text)
    human_msg = HumanMessagePromptTemplate.from_template("{input}")
    ai_msg = AIMessagePromptTemplate.from_template("{agent_scratchpad}")
    prompt = ChatPromptTemplate.from_messages([system_msg, human_msg, ai_msg])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

    # --- Step 1: get raw tool/agent answers
    final_results = multi_query_executor(agent_executor, query)

    # --- Step 2: feed everything back into LLM for structured output
    raw_results_text = "\n\n".join(
        [f"Q: {list(item.keys())[0]}\nA: {list(item.values())[0]}" for item in final_results]
    )

    summarizer_prompt = f"""
You are a smart assistant.
Here are the details of the session:

Role: {role}
Domain: {domain}
Chat History: {chat_history_text}
User Query: {query}
Image URL: {image_url if image_url else 'None'}

Raw outputs from tools/agent:
{raw_results_text}

Now synthesize a FINAL structured response:
- this is the final output the user will see so only provide details if it is an agricultural doubt , if it a general conversational question like 'hi','how are you','what are doing', etc answer as in a humanized way and tell tham you are an agriculutural chatbot , here to help them with there queries.
- Present clear sections (Weather, Image Analysis, Web Info, Context, etc. if relevant).
- Highlight key insights in simple bullet points.
- Answer the user query directly and concisely.
- The Final Answer that you respond with should be STRICTLY in MARKDOWN format ONLY.
"""

    structured_response = llm.invoke(summarizer_prompt)

    output =  structured_response.content if hasattr(structured_response, "content") else str(structured_response)
    return postprocess_answer(output)

# ---------------------------
# Example usage
# ---------------------------


# url = "https://extension.umn.edu/sites/extension.umn.edu/files/marssonia-leaf-spot-on-euonymus-grabowski.jpg"
# response_text = build_agent_prompt(
#     chat_history=[],
#     domain="Agriculture & AI",
#     query="¿Cuál es la temperatura en la longitud 7.18 latitud 2.65?",
#     role="Student Mentor Assistant",
#     image_url=url,
#     verbose=True
# )

response_text = build_agent_prompt(
    chat_history=[],
    domain="Agriculture & AI",
    query=f"what is the temperature in longitude 7.18 latitude 2.65",
    role="Student Mentor Assistant",
    tools=tools,
    llm=llm,            # your LLM object
    image_url="https://extension.umn.edu/sites/extension.umn.edu/files/marssonia-leaf-spot-on-euonymus-grabowski.jpg",
    verbose=True
)
print(response_text)
print("RAG-PIPELINE WORKING!!!")