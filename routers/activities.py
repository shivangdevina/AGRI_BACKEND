from fastapi import APIRouter  , HTTPException
from datetime import date
from core.database import supabase
from schemas.activities import ActivityBase , ActivityCreate

router=APIRouter(prefix="/api/activities" , tags=["activities"]  )


@router.post("/add"  )
def add_activity(plan :ActivityCreate):
    try:
        response=supabase.table("activities").insert(
            {
                "title":plan.title,
                "userid":str(plan.userid),
                "category":plan.category,
                "date":plan.date.isoformat(),
                "notes":plan.notes
            } 
        ).execute()
        if response.data:
            return response.data[0]
        raise HTTPException(status_code=400, detail="Failed to insert plan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


