"""
weather_llm.py

Smart Agricultural Assistant:
Check farmer task feasibility based on 7-day weather and past activities.
"""

import os
import json
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()
GEMINI_API_ENV = "GEMINI_API_KEY"

def initialize_gemini_from_env(env_var: str = GEMINI_API_ENV):
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Gemini API key not found in '{env_var}'")
    if genai is None:
        raise ImportError("Install 'google-generativeai' to use Gemini.")
    genai.configure(api_key=api_key)

def fetch_weather(latitude: float, longitude: float) -> Dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def extract_json(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    return raw_text[start:end + 1].strip() if start != -1 and end != -1 else "{}"

def get_task_alerts(
    latitude: float,
    longitude: float,
    planned_tasks: List[Dict],
    past_activities: Optional[List[Dict]] = None
) -> Dict:
    if genai is None:
        raise ImportError("Google Gemini LLM is not installed.")
    weather_data = fetch_weather(latitude, longitude)
    daily = weather_data.get("daily", {})

    weather_summary = "".join(
        f"- {daily['time'][i]}: Max {daily['temperature_2m_max'][i]}°C, "
        f"Min {daily['temperature_2m_min'][i]}°C, "
        f"Precipitation {daily['precipitation_sum'][i]} mm\n"
        for i in range(len(daily.get("time", [])))
    )

    tasks_summary = "".join(
        f"- Day: {task['day']}, Task: {task['task']}, Details: {task.get('details','N/A')}\n"
        for task in planned_tasks
    )

    history_summary = "".join(
        f"- Time: {activity['time']}, Task: {activity['task']}, Description: {activity.get('description','N/A')}\n"
        for activity in past_activities
    ) if past_activities else "No past activities recorded.\n"

    prompt = f"""You are an agricultural assistant.
Given 7-day weather, planned tasks, and past activities, return ONLY valid JSON in the format:
{{"tasks":[{{"day":"YYYY-MM-DD","task":"irrigation","feasible":true,"reason":"Short explanation"}}]}}
Weather Forecast:
{weather_summary}
Planned Tasks:
{tasks_summary}
Past Activities:
{history_summary}"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    raw_output = getattr(response, "text", None) or (
        getattr(response, "candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )

    cleaned_output = extract_json(raw_output or "")
    try:
        data = json.loads(cleaned_output)
        alerts = [task for task in data.get("tasks", []) if not task.get("feasible", True)]
        return {"alerts": alerts}
    except Exception as e:
        return {"error": str(e), "raw_output": raw_output}
