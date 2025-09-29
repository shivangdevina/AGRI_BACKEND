from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from schemas.chats import ChatSession, ChatSessionCreate, Message, MessageCreate  , getChatSessions
import os
from core.database import supabase
from dotenv import load_dotenv
from  Models.Rag_ChatBot.ragChatPipeline import build_agent_prompt


router = APIRouter()

load_dotenv()

router = APIRouter(prefix="/chats", tags=["chats"])




# get all the chats_session name from chat_type and user_id
@router.get("/", response_model=List[ChatSession])
def list_chats(userReq:getChatSessions):
    query = supabase.table("chat_sessions").select("*").eq("userid", str(userReq.userid))
    if userReq.chat_type:
        query = query.eq("chat_type", userReq.chat_type)
    response = query.order("created_at", desc=True).execute()
    return response.data


# create a new chat session 
@router.post("/createSession", response_model=ChatSession)
def create_chat(session: ChatSessionCreate):
    response = (
        supabase.table("chat_sessions")
        .insert(
            {
                "userid": str(session.userid),
                "chat_type": session.chat_type,
                "title": session.title
            }
        )
        .execute()
    )
    return {
        "id": response.data[0]["id"],
        "chat_type": response.data[0]["chat_type"],
        "created_at": response.data[0]["created_at"],
        "title": response.data[0]["title"]
    }


# get a single chat session by chat_id
@router.get("/{chat_id}", response_model=ChatSession)
def get_chat(chat_id: UUID):
    response = (
        supabase.table("chat_sessions").select("*").eq("id", str(chat_id)).execute()
    )
    if not response.data:
        raise HTTPException(404, "Chat session not found")
    return response.data[0]


# add a new message to chat
@router.post("/addMessage", response_model=Message)
def add_message(message: MessageCreate):
    chat_history="";
    domain="Ai and Agriculture"
    if(message.domain=="secondary"):
        domain="Weather and Agriculture"
    
    Ai_model_response=build_agent_prompt(chat_history=chat_history ,domain=domain, user_query=message.user_query,image_url=message.imageUrls)

    response = (
            supabase.table("chat_messages")
            .insert(
                {
                    "session_id": str(message.session_id),
                    "user_query": message.user_query,
                    "ai_answer": Ai_model_response,
                    "imageurls": message.imageUrls,
                }
            )
            .execute()
        )
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to add message")
    message_data=response.data[0]

    return {
        "id": message_data["id"],
        "session_id": message_data["session_id"],
        "user_query": message_data["user_query"],
        "ai_answer": message_data["ai_answer"],
        "created_at": message_data["created_at"],
        "imageUrls": message_data["imageurls"]
    }


# get all messages from a chat session
@router.get("/messages/{session_id}")
def get_messages(session_id: UUID):
    response = (
        supabase.table("chat_messages")
        .select("*")
        .eq("session_id", str(session_id))
        .order("created_at", desc=False)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="No messages found for this session")

    # return all messages
    messages = []
    for msg in response.data:
        messages.append({
            "id": msg["id"],
            "user_query": msg["user_query"],
            "ai_answer": msg["ai_answer"],
            "created_at": msg["created_at"],
            "imageUrls": msg["imageurls"]
        })
    return messages



# delete chat session
@router.delete("/{chat_id}")
def delete_chat(chat_id: UUID):
    supabase.table("chat_sessions").delete().eq("id", str(chat_id)).execute()
    return {"status": "deleted"}