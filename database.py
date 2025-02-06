import motor.motor_asyncio
from beanie import init_beanie
from models.user import User  # Importing the User model
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/mydatabase")

# Async function to initialize MongoDB connection
async def init_db():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db = client.get_default_database()
    await init_beanie(database=db, document_models=[User])
