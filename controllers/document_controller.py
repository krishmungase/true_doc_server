from fastapi import HTTPException, UploadFile
import shutil
import os
import cloudinary.uploader
from roboflow import Roboflow
import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re

def convert_to_jpg(temp_file):
    """Convert PNG to JPG if necessary."""
    img = Image.open(temp_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    jpg_path = temp_file.rsplit(".", 1)[0] + ".jpg"
    img.save(jpg_path, "JPEG")
    return jpg_path

def extract_text_from_region(image_path, x, y, width, height):
    """Extract text from a specific region of the image."""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image")
            return ""
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract the region of interest with some padding
        padding = 20
        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + height + padding)
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + width + padding)
        
        roi = thresh[y_start:y_end, x_start:x_end]
        
        # Add white border around ROI
        roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        
        # Save the ROI for debugging
        debug_path = f"debug_roi_{x}_{y}.jpg"
        cv2.imwrite(debug_path, roi)
        print(f"Saved ROI to {debug_path}")
        
        # Extract text using pytesseract with different configurations
        # Try different PSM modes
        psm_modes = [6, 7, 8]  # Different page segmentation modes
        texts = []
        
        for psm in psm_modes:
            custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
            text = pytesseract.image_to_string(roi, config=custom_config)
            if text.strip():
                texts.append(text.strip())
        
        # Use the longest non-empty text
        final_text = max(texts, key=len) if texts else ""
        
        # Get the last line of the text
        lines = [line.strip() for line in final_text.split('\n') if line.strip()]
        if lines:
            final_text = lines[-1]  # Get the last line
        
        print(f"\nDebug - Text extraction:")
        print(f"Region coordinates: x={x}, y={y}, width={width}, height={height}")
        print(f"Extracted text: {final_text}")
        
        return final_text.strip()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def format_aadhar_text(details_text, aadhar_no):
    """Format the extracted text in the desired format."""
    try:
        # Split the details text into lines and remove empty lines
        lines = [line.strip() for line in details_text.split('\n') if line.strip()]
        
        print("\nDebug - Raw lines from details:")
        for i, line in enumerate(lines):
            print(f"Line {i}: {line}")
        
        # Initialize formatted text
        formatted_text = "=== Extracted text ===\n"
        
        # Initialize variables
        name = ""
        dob = ""
        gender = ""
        
        # Process lines
        for i, line in enumerate(lines):
            print(f"\nProcessing line {i}: {line}")
            
            # Get name from line 1 (index 1)
            if i == 1:
                # Remove the colon if present and clean the name
                name = line.split(':')[0].strip()
                # Remove any non-English characters
                name = ''.join(c for c in name if c.isalnum() or c.isspace())
                print(f"Debug - Found name: {name}")
            
            # Get DOB from line 2 (index 2)
            elif i == 2:
                print(f"Debug - Processing DOB line: {line}")
                # Try different DOB formats
                if 'DOB' in line:
                    # Extract everything after "DOB"
                    dob_parts = line.split('DOB', 1)
                    if len(dob_parts) > 1:
                        # Clean up DOB (keep only numbers and /)
                        dob = ''.join(c for c in dob_parts[1] if c.isdigit() or c == '/')
                        # Remove any extra spaces
                        dob = dob.strip()
                        print(f"Debug - Found DOB: {dob}")
                else:
                    # Try to find date pattern (DD/MM/YYYY)
                    date_pattern = r'\d{2}/\d{2}/\d{4}'
                    match = re.search(date_pattern, line)
                    if match:
                        dob = match.group(0)
                        print(f"Debug - Found DOB using pattern: {dob}")
            
            # Get gender from line 3 (index 3)
            elif i == 3:
                print(f"Debug - Processing gender line: {line}")
                if 'female' in line.lower():
                    gender = "Female"
                    print(f"Debug - Found gender: {gender}")
                elif 'male' in line.lower():
                    gender = "Male"
                    print(f"Debug - Found gender: {gender}")
        
        # Add the formatted information
        if name:
            formatted_text += f"Name - {name}\n"
        if dob:
            formatted_text += f"DOB - {dob}\n"
        if gender:
            formatted_text += f"Gender - {gender}\n"
        
        # Add Aadhar number
        formatted_text += f"aadhar no - {aadhar_no}"
        
        print("\nDebug - Final formatted text:")
        print(formatted_text)
        
        return formatted_text
    except Exception as e:
        print(f"Error formatting text: {str(e)}")
        return details_text

def format_pan_text(pan_text):
    """Format the extracted PAN number."""
    try:
        # Clean up PAN number (keep only alphanumeric characters)
        pan_number = ''.join(c for c in pan_text if c.isalnum())
        # Convert to uppercase
        pan_number = pan_number.upper()
        
        # Ensure PAN number is 10 characters
        if len(pan_number) > 10:
            pan_number = pan_number[:10]
        
        print("\nDebug - PAN Number extraction:")
        print(f"Raw PAN text: {pan_text}")
        print(f"Cleaned PAN number: {pan_number}")
        
        return f"PAN No: {pan_number}"
    except Exception as e:
        print(f"Error formatting PAN text: {str(e)}")
        return pan_text

def format_pan_details(extracted_text):
    """Format the extracted PAN card details."""
    try:
        formatted_text = "=== Extracted text ===\n"
        
        # Format each field
        if "name" in extracted_text:
            name = extracted_text["name"].strip()
            # Convert to uppercase
            name = name.upper()
            formatted_text += f"Name - {name}\n"
        
        if "pan number" in extracted_text:
            pan_number = extracted_text["pan number"].strip()
            # Convert to uppercase and ensure it's 10 characters
            pan_number = ''.join(c for c in pan_number if c.isalnum()).upper()
            if len(pan_number) > 10:
                pan_number = pan_number[:10]
            formatted_text += f"PAN No - {pan_number}\n"
        
        if "dob" in extracted_text:
            dob = extracted_text["dob"].strip()
            # Clean up DOB (keep only numbers and /)
            dob = ''.join(c for c in dob if c.isdigit() or c == '/')
            formatted_text += f"DOB - {dob}\n"
        
        if "father name" in extracted_text:
            father_name = extracted_text["father name"].strip()
            # Convert to uppercase
            father_name = father_name.upper()
            formatted_text += f"Father's Name - {father_name}\n"
        
        print("\nDebug - Formatted PAN details:")
        print(formatted_text)
        
        return formatted_text
    except Exception as e:
        print(f"Error formatting PAN details: {str(e)}")
        return ""

def detect_objects(image_url: str, card_type: str):
    try:
        if card_type not in ["Aadhar", "PAN", "Driving License"]:
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
        elif card_type == "PAN":
            api_key = "l8euJldexdnlej6ptiMb"
            project_name = "pan-detection-zj2sl"
            expected_fields = ["dob", "father name", "name", "pan number"]
        else:  # Driving License
            api_key = "K8xsKfZpKUg9anaKIw1I"
            project_name = "minorproject-5blne"
            expected_fields = ["Address", "Class", "DOB", "Exp date", "First name", 
                               "Issue date", "Last name", "License number", "Sex"]

        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_name)
        model = project.version(1).model

        predictions = model.predict(image_path, confidence=10, overlap=30).json()

        if "predictions" not in predictions:
            raise HTTPException(status_code=500, detail="Failed to process image with Roboflow")

        detected_fields = list(set(prediction["class"] for prediction in predictions["predictions"])) 

        # Extract text from detected fields
        extracted_text = {}
        formatted_text = ""
        if card_type == "Aadhar":
            print("\n=== Extracting Text from Aadhar Card ===")
            details_text = ""
            aadhar_no = ""
            
            for prediction in predictions["predictions"]:
                if prediction["class"] in ["aadhar no", "details"]:
                    x = int(prediction["x"] - prediction["width"]/2)
                    y = int(prediction["y"] - prediction["height"]/2)
                    width = int(prediction["width"])
                    height = int(prediction["height"])
                    
                    text = extract_text_from_region(image_path, x, y, width, height)
                    extracted_text[prediction["class"]] = text
                    
                    if prediction["class"] == "details":
                        details_text = text
                    elif prediction["class"] == "aadhar no":
                        aadhar_no = text
            
            # Format the extracted text
            formatted_text = format_aadhar_text(details_text, aadhar_no)
            print("\n" + formatted_text + "\n")
        elif card_type == "PAN":
            print("\n=== Extracting Text from PAN Card ===")
            
            for prediction in predictions["predictions"]:
                if prediction["class"] in ["name", "pan number", "dob", "father name"]:
                    x = int(prediction["x"] - prediction["width"]/2)
                    y = int(prediction["y"] - prediction["height"]/2)
                    width = int(prediction["width"])
                    height = int(prediction["height"])
                    
                    print(f"\nProcessing {prediction['class']} field:")
                    print(f"Confidence: {prediction['confidence']}")
                    print(f"Coordinates: x={x}, y={y}, width={width}, height={height}")
                    
                    text = extract_text_from_region(image_path, x, y, width, height)
                    extracted_text[prediction["class"]] = text
            
            # Format the extracted text
            formatted_text = format_pan_details(extracted_text)
            print("\n" + formatted_text + "\n")

        for prediction in predictions["predictions"]:
            print(f"Detected: {prediction['class']} with confidence {prediction['confidence']}")
    
        # Check for missing fields
        missing_fields = [field for field in expected_fields if field not in detected_fields]
        
        os.remove(image_path)

        return predictions, detected_fields, missing_fields, extracted_text, formatted_text

    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")  # Debugging log
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

async def process_document(card_type: str, file: UploadFile):
    print(f"\n=== Processing {card_type} Document ===")
    print(f"File name: {file.filename}")
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".png"):
        temp_file = convert_to_jpg(temp_file)
        print("Converted PNG to JPG format")

    upload_result = cloudinary.uploader.upload(temp_file)
    cloudinary_url = upload_result.get("secure_url")

    if not cloudinary_url:
        os.remove(temp_file)
        raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary")

    os.remove(temp_file)
    print("Image uploaded to Cloudinary successfully")

    predictions, detected_fields, missing_fields, extracted_text, formatted_text = detect_objects(cloudinary_url, card_type)

    if missing_fields:
        print(f"\nMissing fields detected: {missing_fields}")
        raise HTTPException(status_code=400, detail={
            "message": "Required objects not detected in the image.",
            "predictions": predictions,
            "detected_fields": detected_fields,
            "missing_fields": missing_fields
        })  

    print("\n=== Document Processing Complete ===")
    print(f"Detected fields: {detected_fields}")
    print("================================\n")

    return {
        "message": "Success",
        "cloudinary_url": cloudinary_url,
        "detected_fields": detected_fields,
        "missing_fields": missing_fields,
        "predictions": predictions,
        "extracted_text": extracted_text,
        "formatted_text": formatted_text
    }
