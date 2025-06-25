from fastapi import APIRouter, HTTPException, Depends
from app.models.user_model import UserCreate, UserLogin
from app.utils.security import hash_password, verify_password, create_access_token, get_current_user
from app.database import db
from bson import ObjectId

router = APIRouter()

@router.post("/register")
async def register(user: UserCreate):
    existing = await db["users"].find_one({"email": user.email})
    if existing:
        raise HTTPException(400, "Email already registered")
    await db["users"].insert_one({
        "email": user.email,
        "hashed_password": hash_password(user.password)
    })
    return {"message": "User registered"}

@router.post("/login")
async def login(user: UserLogin):
    db_user = await db["users"].find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": str(db_user["_id"])})
    return {"access_token": token}

@router.get("/me")
async def get_logged_user(user_id: str = Depends(get_current_user)):
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(404, "User not found")

    name = user["email"].split("@")[0].capitalize()
    avatar = name[0].upper()

    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": name,
        "avatar": avatar,
    }
