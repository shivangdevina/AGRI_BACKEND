from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional
from uuid import UUID


class IrrigationPlanBase(BaseModel):
    userid: UUID
    date: date
    time_slot: str
    duration_minutes: int
    amount_mm: Optional[float] = None
    method: Optional[str] = None
    notes: Optional[str] = None
    details: Optional[str] = None
    crop: Optional[str] = None
    status: str = "scheduled"
    class Config:
        json_encoders = {
            UUID: str,
            date: lambda v: v.isoformat(),
        }


class IrrigationPlanCreate(IrrigationPlanBase):
    pass


class IrrigationPlanOut(IrrigationPlanBase):
    id: UUID
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True  # allows returning ORM/dict objects
