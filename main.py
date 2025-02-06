from fastapi import FastAPI
from routes.auth import router as user_router
from database import init_db
import asyncio

app = FastAPI()

# Register routes
app.include_router(user_router, prefix="/api/auth")

@app.on_event("startup")
async def startup_event():
    await init_db()  # Initialize MongoDB connection

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI with MongoDB"}
