from pydantic import BaseModel , EmailStr
from uuid import UUID
from typing import Optional, List
from datetime import datetime


class ChatSessionCreate(BaseModel):
    title: Optional[str] = None
    chat_type: str  # "main", "secondary", "third"
    userid:UUID

class getChatSessions(BaseModel):
    userid: UUID
    chat_type: str


class ChatSession(BaseModel):
    id: UUID
    chat_type: str
    title: Optional[str]
    created_at: datetime


class MessageCreate(BaseModel):
    session_id: UUID  # chat id 
    
    user_query: str   # this is mandatary to pass
    role: Optional[str]=None # "user" or "ai"
    imageUrls: Optional[List[str]] = []
    content: Optional[str] = None
    other_content: Optional[str] = None
    ai_response: Optional[str] = None
    domain:Optional[str] = "main"
    


class Message(BaseModel):
    id: UUID    #message id               
    session_id: UUID
    user_query: str
    ai_answer: Optional[str]
    imageUrls: Optional[List[str]]
    created_at: datetime

