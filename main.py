from fastapi import FastAPI
from routes.auth import router as user_router
from routes.detection import router as document_router
from database import init_db

app = FastAPI()

# Register routes
app.include_router(user_router, prefix="/api/auth")
app.include_router(document_router, prefix="/api/document")

@app.on_event("startup")
async def startup_event():
    await init_db()  

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI with MongoDB"}
