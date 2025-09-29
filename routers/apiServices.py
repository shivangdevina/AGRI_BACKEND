from fastapi import FastAPI  , APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv
load_dotenv()
router=APIRouter(prefix="/apiServices" , tags=["apiServices"])
SOIL_HIVE_API_KEY = os.getenv("SOIL_HIVE_API_KEY")
SOIL_HIVE_ENDPOINT = "https://api.soilhive.com/v1/soil"  # Replace with actual endpoint
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"
@router.get("/autofill")
def autofill(lat: float = Query(...), lng: float = Query(...)):
    try:
        # -------- Soil Hive Call --------
        soil_resp = requests.get(
            SOIL_HIVE_ENDPOINT,
            params={"latitude": lat, "longitude": lng},
            headers={"Authorization": f"Bearer {SOIL_HIVE_API_KEY}"}, verify=False
        )
        if soil_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch soil data")
        soil_data = soil_resp.json()

        # -------- OpenWeatherMap Call --------
        weather_resp = requests.get(
            OPENWEATHER_ENDPOINT,
            params={"lat": lat, "lon": lng, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        )
        if weather_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch weather data")
        weather_data = weather_resp.json()

        # Prepare combined response
        autofill_data = {
            "N": soil_data.get("nitrogen", 0),
            "P": soil_data.get("phosphorus", 0),
            "K": soil_data.get("potassium", 0),
            "soilPh": soil_data.get("ph", 7.0),
            "soilType": soil_data.get("soil_type", "Loam"),
            "temperature": weather_data["main"].get("temp", 25),
            "humidity": weather_data["main"].get("humidity", 60),
            "rainfall": weather_data.get("rain", {}).get("1h", 0)  # Rain in mm for last 1h
        }

        return autofill_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching autofill data: {str(e)}")