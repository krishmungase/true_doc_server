from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.detection import router as document_router
from utils.database import MongoDB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Test database connection on startup
@app.on_event("startup")
async def startup_db_client():
    try:
        # Initialize MongoDB connection
        mongo = MongoDB()
        # Test the connection
        db = mongo.get_database()
        # Try to ping the database
        db.command('ping')
        logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"], 
)
app.include_router(document_router, prefix="/api/document")

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI with MongoDB"}

# Add a health check endpoint
@app.get("/health")
def health_check():
    try:
        # Test database connection
        mongo = MongoDB()
        db = mongo.get_database()
        db.command('ping')
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }
