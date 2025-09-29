from pydantic import BaseModel
from uuid import UUID
from datetime import date
from typing import Optional
from datetime import datetime


class ActivityBase(BaseModel):
    title:str
    category: str
    date: date
    notes: str


class ActivityCreate(ActivityBase):
    userid:UUID

class ActivityOut(ActivityBase):
    id: UUID
    class Config:
          # allows returning ORM/dict objects
        orm_mode=True

    
    
