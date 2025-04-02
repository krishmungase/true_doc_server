from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.detection import router as document_router

app = FastAPI()

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
