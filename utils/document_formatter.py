import re
from typing import Dict, Any

class DocumentFormatter:
    @staticmethod
    def format_aadhar_data(extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format Aadhar card data into structured format"""
        data = {
            "aadhar_number": "",
            "name": "",
            "dob": "",
            "gender": "",
            "address": ""
        }
        
        # Extract Aadhar number
        if "aadhar no" in extracted_text:
            aadhar_text = extracted_text["aadhar no"]["text"]
            aadhar_match = re.search(r'\d{4}\s?\d{4}\s?\d{4}', aadhar_text)
            if aadhar_match:
                data["aadhar_number"] = aadhar_match.group().replace(" ", "")
        
        # Extract other details
        if "details" in extracted_text:
            details_text = extracted_text["details"]["text"]
            
            # Extract DOB
            dob_match = re.search(r'\d{2}/\d{2}/\d{4}', details_text)
            if dob_match:
                data["dob"] = dob_match.group()
            
            # Extract Gender
            if any(gender in details_text.lower() for gender in ["female", "स्त्री", "महिला"]):
                data["gender"] = "Female"
            elif any(gender in details_text.lower() for gender in ["male", "पुरुष", "नर"]):
                data["gender"] = "Male"
            
            # Extract Name and Address
            lines = details_text.split('\n')
            for line in lines:
                if not any(keyword in line.lower() for keyword in ["dob", "date of birth", "gender", "sex"]):
                    if not data["name"]:
                        data["name"] = line.strip()
                    else:
                        data["address"] = line.strip()
        
        return data

    @staticmethod
    def format_pan_data(extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format PAN card data into structured format"""
        data = {
            "pan_number": "",
            "name": "",
            "father_name": "",
            "dob": ""
        }
        
        # Extract PAN number
        if "pan number" in extracted_text:
            pan_text = extracted_text["pan number"]["text"]
            pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', pan_text)
            if pan_match:
                data["pan_number"] = pan_match.group()
        
        # Extract other details
        if "details" in extracted_text:
            details_text = extracted_text["details"]["text"]
            
            # Extract DOB
            dob_match = re.search(r'\d{2}/\d{2}/\d{4}', details_text)
            if dob_match:
                data["dob"] = dob_match.group()
            
            # Extract Name and Father's Name
            lines = details_text.split('\n')
            for line in lines:
                if not any(keyword in line.lower() for keyword in ["dob", "date of birth"]):
                    if not data["name"]:
                        data["name"] = line.strip()
                    elif not data["father_name"]:
                        data["father_name"] = line.strip()
        
        return data

    @staticmethod
    def format_driving_license_data(extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format Driving License data into structured format"""
        data = {
            "license_number": "",
            "name": "",
            "dob": "",
            "address": "",
            "issue_date": "",
            "expiry_date": "",
            "class": "",
            "sex": ""
        }
        
        # Extract License Number
        if "License number" in extracted_text:
            data["license_number"] = extracted_text["License number"]["text"].strip()
        
        # Extract Name
        if "First name" in extracted_text and "Last name" in extracted_text:
            first_name = extracted_text["First name"]["text"].strip()
            last_name = extracted_text["Last name"]["text"].strip()
            data["name"] = f"{first_name} {last_name}"
        
        # Extract DOB
        if "DOB" in extracted_text:
            data["dob"] = extracted_text["DOB"]["text"].strip()
        
        # Extract Address
        if "Address" in extracted_text:
            data["address"] = extracted_text["Address"]["text"].strip()
        
        # Extract Issue Date
        if "Issue date" in extracted_text:
            data["issue_date"] = extracted_text["Issue date"]["text"].strip()
        
        # Extract Expiry Date
        if "Exp date" in extracted_text:
            data["expiry_date"] = extracted_text["Exp date"]["text"].strip()
        
        # Extract Class
        if "Class" in extracted_text:
            data["class"] = extracted_text["Class"]["text"].strip()
        
        # Extract Sex
        if "Sex" in extracted_text:
            data["sex"] = extracted_text["Sex"]["text"].strip()
        
        return data

    @classmethod
    def format_document_data(cls, card_type: str, extracted_text: Dict[str, Any]) -> Dict[str, str]:
        """Format extracted text based on document type"""
        if card_type == "Aadhar":
            return cls.format_aadhar_data(extracted_text)
        elif card_type == "PAN":
            return cls.format_pan_data(extracted_text)
        elif card_type == "Driving License":
            return cls.format_driving_license_data(extracted_text)
        else:
            return extracted_text 