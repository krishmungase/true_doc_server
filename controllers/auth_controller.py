from models.user import User, UserCreate
from security import hash_password, verify_password, create_access_token
from datetime import timedelta
from fastapi import HTTPException
from pymongo.errors import DuplicateKeyError

async def register_user(user_data: UserCreate):
    try:
        print("User data => ", user_data)

        existing_user = await User.find_one(User.username == user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")

        hashed_pw = hash_password(user_data.password)

        user = User(username=user_data.username, email=user_data.email, hashed_password=hashed_pw)
        await user.insert()

        return {"message": "User registered successfully"}

    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already exists")


async def authenticate_user(username: str, password: str):
    user = await User.find_one(User.username == username)

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token({"sub": user.username}, timedelta(minutes=30))

    return {
        "id": str(user.id),
    }


async def get_user_by_username(username: str):
    user = await User.find_one(User.username == username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "_id": str(user.id), 
        "username": user.username,
        "email": user.email
    }
