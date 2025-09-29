import os
import json
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    torch_dtype="auto"
)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)
llm = HuggingFacePipeline(pipeline=gen_pipeline)

DB_FILE = "farmer_feedback.json"
INDEX_FILE = "feedback.index"

if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        feedback_data = json.load(f)
else:
    feedback_data = []

dim = embeddings.client.get_sentence_embedding_dimension()
if os.path.exists(INDEX_FILE) and len(feedback_data) > 0:
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(dim)

def get_llm_feedback(question: str, answer: str, farmer_feedback: str) -> str:
    prompt = f"""
You are a feedback analyzer for agricultural Q&A.

Farmer Question: {question}
LLM Answer: {answer}
Farmer Feedback: {farmer_feedback}

Explain clearly:
- If feedback is LIKE → What worked well? What should be repeated?
- If feedback is DISLIKE → What was wrong/missing? How to improve?
"""
    return llm.invoke(prompt)

def store_feedback(question: str, answer: str, farmer_feedback: str, llm_feedback: str):
    entry = {
        "question": question,
        "answer": answer,
        "farmer_feedback": farmer_feedback,
        "llm_feedback": llm_feedback
    }
    feedback_data.append(entry)
    with open(DB_FILE, "w") as f:
        json.dump(feedback_data, f, indent=4)
    embedding = embeddings.embed_query(question)
    index.add(np.array([embedding], dtype=np.float32))
    faiss.write_index(index, INDEX_FILE)

def retrieve_similar_feedback(new_query: str, k: int = 3):
    if len(feedback_data) == 0:
        return []
    embedding = embeddings.embed_query(new_query)
    D, I = index.search(np.array([embedding], dtype=np.float32), k)
    return [feedback_data[idx] for idx in I[0] if idx < len(feedback_data)]

def process_query(question: str, answer: str, farmer_feedback: str):
    llm_feedback = get_llm_feedback(question, answer, farmer_feedback)
    store_feedback(question, answer, farmer_feedback, llm_feedback)
    return llm_feedback

def answer_with_feedback(new_query: str):
    similar_feedback = retrieve_similar_feedback(new_query)
    context = "\n\n".join(
        [f"Q: {item['question']}\nAns: {item['answer']}\nFeedback Summary: {item['llm_feedback']}"
         for item in similar_feedback]
    )
    prompt = f"""
Farmer's New Question: {new_query}

Here are past similar cases and their feedback:
{context}

Now provide the best possible answer considering these learnings.
"""
    return llm.invoke(prompt)
