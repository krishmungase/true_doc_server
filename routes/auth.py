from fastapi import APIRouter, Depends
from controllers.auth_controller import register_user, authenticate_user, get_user_by_username
from models.user import UserCreate
from security import create_access_token
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()

# Register Route
@router.post("/register")
async def register(user: UserCreate):
    return await register_user(user)

# Login Route (JWT Token Generation)
@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return await authenticate_user(form_data.username, form_data.password)

# Get User Details (Protected Route)
@router.get("/{username}")
async def get_user(username: str):
    return await get_user_by_username(username)
