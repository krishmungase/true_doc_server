from fastapi import HTTPException, UploadFile
import shutil
import os
import cloudinary.uploader
from roboflow import Roboflow
import requests


def detect_objects(image_url: str, card_type: str):
    image_path = "temp_image.jpg"
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(image_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    else:
        raise Exception("Failed to download image from Cloudinary")
    
    rf = Roboflow(api_key="LFlCmvBFgaFn7jH9yIWs")
    project = rf.workspace().project("adhar_obj_detection")
    model = project.version(1).model 
    
    expected_fields = ['aadhar no', 'details', 'gov', 'logo', 'photo', 'qr']
    
    predictions = model.predict(image_path, confidence=40, overlap=30).json()
    detected_fields = [prediction["class"] for prediction in predictions["predictions"]]
    
    missing_fields = [field for field in expected_fields if field not in detected_fields]
    
    print(predictions)

    os.remove(image_path)
    
    if missing_fields:
        print(f"Invalid image. Missing fields: {', '.join(missing_fields)}")
        return False
    else:
        print("Valid image. All expected fields are present.")
        return True


async def process_document(card_type: str, file: UploadFile):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_result = cloudinary.uploader.upload(temp_file)
    cloudinary_url = upload_result["secure_url"]
    
    os.remove(temp_file)

    if not detect_objects(cloudinary_url, card_type):
        raise HTTPException(status_code=400, detail="Required objects not detected in the image.")  

    return {"message": "Success", "cloudinary_url": cloudinary_url}
