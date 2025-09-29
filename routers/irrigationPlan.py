from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import date
from uuid import UUID
from core.database import supabase
from schemas.irrigationPlan import IrrigationPlanCreate , IrrigationPlanOut

router = APIRouter(prefix="/api/irrigation", tags=["Irrigation"])



@router.get("/")
async def get_irrigation_by_month(
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000, le=2100),
    userid: Optional[UUID] = None
):
    """
    Returns irrigation plans for a given month/year.
    Optionally filter by userid.
    """
    try:
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)

        query = (
            supabase.table("irrigation_plans")
            .select("*")
            .gte("date", str(start_date))
            .lt("date", str(end_date))
        )
        if userid:
            query = query.eq("userid", str(userid))

        response = query.order("date").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2️⃣ Fetch all irrigation plans (for sidebar)
@router.get("/list")
async def get_all_irrigation_plans(userid: Optional[UUID] = None):
    """
    Returns all irrigation plans (optionally filtered by userid).
    """
    try:
        query = supabase.table("irrigation_plans").select("*")
        if userid:
            query = query.eq("userid", str(userid))

        response = query.order("date").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3️⃣ Add new irrigation schedule

@router.post("/", response_model=IrrigationPlanOut)
async def add_irrigation_plan(plan: IrrigationPlanCreate):
   
    try:
        response = supabase.table("irrigation_plans").insert(plan.model_dump(mode="json")).execute()
        if response.data:
            # Supabase returns list of rows, we return first
            return response.data[0]
        raise HTTPException(status_code=400, detail="Failed to insert plan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))