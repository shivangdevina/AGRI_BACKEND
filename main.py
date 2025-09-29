from fastapi import FastAPI, UploadFile, File , Form ,  HTTPException
from fastapi.responses import FileResponse , StreamingResponse
from gtts import gTTS
from langdetect import detect
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import io

# Load env vars
load_dotenv()



logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env")

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# print(supabase.table("users").select("*").execute())
# Logger
logger = logging.getLogger("my")

# FastAPI app
app = FastAPI(title="Chat API")
logger.info(" FastAPI server started")

# Middleware
origins = [
    "http://localhost:8080",  
    "http://localhost:5173",  
    "http://localhost:3000"  
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from routers import users, chats, auth , irrigationPlan , uploadRoute , ml_model , activities , apiServices


app.include_router(users.router)
app.include_router(chats.router)
logger.info("Test logger from auth.py")
app.include_router(auth.router)
app.include_router(irrigationPlan.router)
app.include_router(activities.router)
app.include_router(uploadRoute.router)
app.include_router(ml_model.router)
app.include_router(apiServices.router)



# Root route
@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.post("/speak")
def speak(text: str = Form(...)):
    # Detect the language of the input text
    lang = detect(text)
    lang = "en" if lang not in [
        "en","hi","bn","te","mr","ta","gu","kn","ml","pa","or","as","ur"
    ] else lang

    print("Detected language:", lang)

    
    mp3_fp = io.BytesIO()
    tts = gTTS(text=text, lang=lang)  
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

   
    return StreamingResponse(mp3_fp, media_type="audio/mpeg", headers={
        "Content-Disposition": f'inline; filename="speech.mp3"'
    })