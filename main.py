from fastapi import FastAPI
from routes.detection import router as document_router

app = FastAPI()

# Register routes
app.include_router(document_router, prefix="/api/document")


@app.get("/")
def root():
    return {"message": "Welcome to FastAPI with MongoDB"}
