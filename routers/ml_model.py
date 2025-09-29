from fastapi import APIRouter, Query, HTTPException
from typing import Optional , List

from pydantic import BaseModel , Field

from datetime import date
from uuid import UUID
from core.database import supabase

from schemas.ml_modelSchemas import *


from Models.plant_disease_detector import run_pipeline
from Models.crop_recommendation import predict_crop_llm
from Models.Fertilizer_Recommender.fertilizer_recommender import recommend_fertilizers
from Models.WeatherInsights.weatherAlerts import get_task_alerts
from Models.Disaster_Alert.disasterAlert import get_disaster_alerts
from Models.News.news import get_agriculture_policy_and_market_summary
from Models.GovtScheme.govtSchemes import summarize_government_schemes
from M
from main import logger

router = APIRouter(prefix="/api_model", tags=["api_model"])




@router.post("/pest_detection_and_analyze")
async def analyze(req: AnalyzeRequest):
    # For now return dummy response
    response=run_pipeline(req.image_url, req.description)
    return response



@router.post("/crop_recommendations", response_model=List[CropResponse])
async def get_crop_recommendations(payload: CropRequest):
    raw = predict_crop_llm(payload.nitrogen, payload.phosphorus, payload.potassium,
                     payload.temperature, payload.humidity, payload.soilPh, payload.rainfall)

    transformed = []
    for item in raw["recommendations"]:
        transformed.append(CropResponse(
            name=item["crop"],                        # map crop → name
            percent=int(item.get("confidence", 0)),   # map confidence → percent
            short_detail=item.get("short_description", ""),
            long_detail=item.get("detailed_description", {}).get("irrigation", "")
        ))
    return transformed


@router.post("/fertilizer_recommendation", response_model=FertilizerResponse)
async def get_fertilizer_recommendation(request: FertilizerRequest):
    try:
        result=recommend_fertilizers(
            N=request.nitrogen,
            P=request.phosphorus,
            K=request.potassium,
            temperature=request.temperature,
            humidity=request.humidity,
            ph=request.soilPh,
            rainfall=request.rainfall,
            crop=request.soilType,
            verbose=False   
        )

        return result["structured"]

    except Exception as e:
        logger.error(f"Fertilizer recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    



@router.post("/task-alerts")
def task_alerts(request: TaskAlertRequest):
    try:
        alerts = get_task_alerts(
            latitude=request.latitude,
            longitude=request.longitude,
            planned_tasks=[task.dict() for task in request.planned_tasks],
            past_activities=[act.dict() for act in request.past_activities] if request.past_activities else None
        )
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
        


@router.post("/agri-news-summary")
def agri_news_summary(request: AgriNewsRequest):
    try:
        summary = get_agriculture_policy_and_market_summary(state=request.state, limit=request.limit)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        
    
@router.post("/summarize-schemes")
def get_schemes(request: SchemeRequest):
    try:
        summary = summarize_government_schemes(request.state, request.category)
        return {"state": request.state, "category": request.category, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/item-price", response_model=ItemPriceResponse)
def item_price(item_name: str = Query(..., description="Name of the item")):
    price_data = get_item_price(item_name)
    if not price_data:
        raise HTTPException(status_code=404, detail=f"No data found for item '{item_name}'")

    insights = analyze_with_gemini(price_data)

    response = ItemPriceResponse(
        item=price_data["item"],
        current_price=price_data["current_price"],
        price_history=price_data["price_history"],
        insights=insights
    )
    return response