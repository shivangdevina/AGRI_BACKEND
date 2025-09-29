from pydantic import BaseModel , Field 
from uuid import UUID   
from datetime import date, datetime
from typing import List, Optional





class PlannedTask(BaseModel):
    day: str = Field(..., description="Task date in YYYY-MM-DD format")
    task: str = Field(..., description="Name of the planned task")
    details: Optional[str] = Field(None, description="Optional details for the task")

class PastActivity(BaseModel):
    time: str = Field(..., description="Date/time of past activity")
    task: str = Field(..., description="Task performed")
    description: Optional[str] = Field(None, description="Additional info")

class TaskAlertRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    planned_tasks: List[PlannedTask] = Field(..., description="List of planned tasks")
    past_activities: Optional[List[PastActivity]] = Field(None, description="List of past activities")
class Token(BaseModel):
    refresh_token:str
    token_type:str

class TokenData(BaseModel):
    id:Optional[str] = None

class CropRequest(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    soilPh: float
    nitrogen: float
    phosphorus: float
    potassium: float
    soilType: str
    season: str


class CropResponse(BaseModel):
    name: str
    percent: int
    short_detail: str
    long_detail: str

class FertilizerRequest(BaseModel):
    temperature: int
    humidity: int
    rainfall: int
    soilPh: float
    nitrogen: int
    phosphorus: int
    potassium: int
    soilType: str
    season: str

class DetailedDescription(BaseModel):
    benefits: List[str]
    precautions: List[str]

class FertilizerRecommendation(BaseModel):
    fertilizer: str
    confidence: float
    short_description: str
    detailed_description: DetailedDescription

class FertilizerResponse(BaseModel):
    recommendations: List[FertilizerRecommendation]

class AgriNewsRequest(BaseModel):
    state: Optional[str] = Field("Kerala", description="State to fetch agricultural updates for")
    limit: Optional[int] = Field(5, description="Number of records to summarize")


class AnalyzeRequest(BaseModel):
    image_url: str
    crop: Optional[str] = ""
    description: Optional[str] = ""

class SchemeRequest(BaseModel):
    state: str
    category: str = "agriculture"


class PriceHistoryItem(BaseModel):
    date: str
    price: str

class ItemPriceResponse(BaseModel):
    item: str
    current_price: str
    price_history: List[PriceHistoryItem]
    insights: str