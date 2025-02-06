from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr

class User(Document):
    username: str = Indexed(str, unique=True) 
    email: EmailStr
    hashed_password: str

    class Settings:
        collection = "users"

# Schema for User Registration (Input Validation)
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

# Schema for Response (Excludes Password)
class UserResponse(BaseModel):
    username: str
    email: EmailStr
