"""
disaster_alert.py

Fetch disaster alerts (IMD Kerala warnings + USGS earthquakes) for a given location.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Kerala district to IMD ID mapping
KERALA_DISTRICT_MAP = {
    "Thiruvananthapuram": "635",
    "Kollam": "636",
    "Pathanamthitta": "637",
    "Alappuzha": "638",
    "Kottayam": "639",
    "Idukki": "640",
    "Ernakulam": "641",
    "Thrissur": "642",
    "Palakkad": "643",
    "Malappuram": "644",
    "Kozhikode": "645",
    "Wayanad": "646",
    "Kannur": "647",
    "Kasaragod": "648"
}

def fetch_usgs_earthquakes(lat: float, lon: float, radius_km: int = 200, min_magnitude: float = 4.5, days: int = 7) -> List[Dict]:
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": now.strftime("%Y-%m-%d"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        alerts = []
        for feat in data.get("features", []):
            props = feat["properties"]
            geom = feat["geometry"]
            alerts.append({
                "type": "earthquake",
                "severity": "high" if props.get("mag", 0) >= 5 else "moderate",
                "time": datetime.utcfromtimestamp(props["time"]/1000).isoformat(),
                "location": {"lat": geom["coordinates"][1], "lon": geom["coordinates"][0], "depth": geom["coordinates"][2]},
                "title": f"Earthquake {props.get('mag')}M - {props.get('place')}",
                "source_url": props.get("url")
            })
        return alerts
    except Exception:
        return []

def fetch_imd_warnings(district_id: str) -> List[Dict]:
    url = f"https://mausam.imd.gov.in/api/warnings_district_api.php?id={district_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        alerts = []
        for item in data.get("warning", []):
            alerts.append({
                "type": "storm",
                "severity": item.get("color", "info"),
                "time": item.get("issue_time"),
                "location": {"district_id": district_id},
                "title": item.get("event", "Weather Alert"),
                "description": item.get("warning"),
                "source_url": "https://mausam.imd.gov.in/"
            })
        return alerts
    except Exception:
        return []

def get_district_name_from_latlon(lat: float, lon: float) -> Optional[str]:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1}
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "DisasterAlertApp"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("address", {}).get("county") or data.get("address", {}).get("state_district")
    except Exception:
        return None

def get_district_id(district_name: str) -> Optional[str]:
    return KERALA_DISTRICT_MAP.get(district_name)

def get_disaster_alerts(lat: float, lon: float) -> List[Dict]:
    alerts = []
    district_name = get_district_name_from_latlon(lat, lon)
    district_id = get_district_id(district_name) if district_name else None
    if district_id:
        alerts.extend(fetch_imd_warnings(district_id))
    alerts.extend(fetch_usgs_earthquakes(lat, lon))
    return alerts
