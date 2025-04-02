from fastapi import HTTPException, UploadFile
import shutil
import os
import cloudinary.uploader
from roboflow import Roboflow
import requests
from PIL import Image

def convert_to_jpg(temp_file):
    """Convert PNG to JPG if necessary."""
    img = Image.open(temp_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    jpg_path = temp_file.rsplit(".", 1)[0] + ".jpg"
    img.save(jpg_path, "JPEG")
    return jpg_path

def detect_objects(image_url: str, card_type: str):
    try:
        if card_type not in ["Aadhar", "PAN"]:
            raise HTTPException(status_code=400, detail="Unsupported card type")

        image_path = "temp_image.jpg"
        response = requests.get(image_url, stream=True)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from Cloudinary")

        with open(image_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        if card_type == "Aadhar":
            api_key = "LFlCmvBFgaFn7jH9yIWs"
            project_name = "adhar_obj_detection"
            expected_fields = ['aadhar no', 'details', 'gov', 'logo', 'photo', 'qr']
        else:  # PAN
            api_key = "l8euJldexdnlej6ptiMb"
            project_name = "pancard-mp1jt"
            expected_fields = ['details', 'goi', 'pan', 'photo', 'qr', 'silverLogo', 'symbol']

        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_name)

        model = project.version(1).model

        predictions = model.predict(image_path, confidence=10, overlap=30).json()
        
        if "predictions" not in predictions:
            raise HTTPException(status_code=500, detail="Failed to process image with Roboflow")

        detected_fields = list(set(prediction["class"] for prediction in predictions["predictions"])) 

        for prediction in predictions["predictions"]:
            print(f"Detected: {prediction['class']} with confidence {prediction['confidence']}")

        missing_fields = [field for field in expected_fields if field not in detected_fields]
        
        os.remove(image_path)

        return detected_fields, missing_fields

    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")  # Debugging log
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def process_document(card_type: str, file: UploadFile):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".png"):  # Fixed condition
        temp_file = convert_to_jpg(temp_file)

    upload_result = cloudinary.uploader.upload(temp_file)
    cloudinary_url = upload_result.get("secure_url")

    if not cloudinary_url:
        os.remove(temp_file)
        raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary")

    print("Uploaded Image URL:", cloudinary_url)

    os.remove(temp_file)

    detected_fields, missing_fields = detect_objects(cloudinary_url, card_type)

    if missing_fields:
        raise HTTPException(status_code=400, detail={
            "message": "Required objects not detected in the image.",
            "detected_fields": detected_fields,
            "missing_fields": missing_fields
        })  

    return {
        "message": "Success",
        "cloudinary_url": cloudinary_url,
        "detected_fields": detected_fields,
        "missing_fields": missing_fields
    }
