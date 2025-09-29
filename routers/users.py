from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from uuid import UUID



router = APIRouter(prefix="/users", tags=["users"])
