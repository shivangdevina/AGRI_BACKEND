from fastapi import FastAPI ,APIRouter , HTTPException ,File , UploadFile
from uuid import UUID
from pydantic import BaseModel
from typing import Optional
from main import logger
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="dv2voz8ae",
    api_key="375113923188142",
    api_secret="KAnELsWxm_mcVytdOfGhKkL42hs"
)
router=APIRouter(prefix="/upload",tags=["upload"])

@router.post("/single")
async def upload_image(file: UploadFile = File(...), request_path: Optional[str] = None):
    folder_name = request_path or "Miscellaneous"
    result = cloudinary.uploader.upload(file.file, folder=folder_name)
    return {"uploaded_url": result["secure_url"]}


            
@router.post("/multiple")
async def upload_images(  files: list[UploadFile] = File(...) , request_path: Optional[str] = None):
    logger.info("**in the upload post route app\n")
    urls = []
    folder_name="Miscellaneous"
    if(request_path):
        folder_name=request_path

    for file in files:
       
        result = cloudinary.uploader.upload(
            file.file,
            folder=folder_name
        )
        urls.append(result["secure_url"])  
    logger.info("we are done*\n")
    return {"uploaded_urls": urls}