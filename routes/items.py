# routers/items.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/items")
def get_items():
    return {"items": ["item1", "item2", "item3"]}

@router.post("/items")
def create_item(item: dict):
    return {"message": "Item created", "item": item}
