import json
import os
import re
import time
from typing import Any, Dict

from dotenv import load_dotenv
from PIL import Image
from mistralai import Mistral

from schemas import (
    ExtractionResponse,
    DocumentInfo,
    PersonalInfo,
    AddressInfo
)

load_dotenv()

PIXTRAL_API_KEY = os.getenv("PIXTRAL_API_KEY")

if not PIXTRAL_API_KEY:
    raise ValueError("PIXTRAL_API_KEY environment variable not set")

client = Mistral(api_key=PIXTRAL_API_KEY)

VALIDATION_PROMPT = """
Analyze the uploaded image.

Determine whether the image contains a government-issued identity document.

Supported documents:
- Aadhaar Card
- PAN Card
- Passport
- Driving License
- Voter ID
- National ID Card
- Residence Permit
- Government Identity Card
- Student ID Card
- College ID Card

If the image is NOT a government-issued identity document,
return false.

Return ONLY valid JSON:

{
    "is_valid_document": true,
    "document_type": "Aadhaar Card",
    "confidence": 0.95
}
"""

EXTRACTION_PROMPT = """
You are an OCR and document understanding system.

Analyze the uploaded government document image and extract all visible information accurately.

IMPORTANT RULES:
1. Return ONLY valid JSON.
2. No explanations.
3. No markdown.
4. If a field is missing, return null.
5. Do not guess or infer missing values.
6. Preserve exact spelling from the document.
7. All output must be in ENGLISH only.
   - If any text in the document is in another language, translate it to English.
   - Do NOT keep original language text.

ADDRESS EXTRACTION RULE:
- Extract the FULL COMPLETE ADDRESS as it appears in the document.
- Combine all address-related lines into a single structured breakdown.
- Ensure the address is fully translated into English.
- Do NOT omit any part of the address if it is visible.
- Show City, State and Pin Code as well.

Return JSON in this exact format:

{
  "document_info": {
    "document_type": null,
    "card_number": null,
    "issuing_authority": null,
    "municipality": null
  },
  "personal_info": {
    "name": null,
    "father_name": null,
    "date_of_birth": null,
    "age": null,
    "gender": null,
    "ethnicity": null,
    "hair_color": null,
    "eye_color": null,
    "clothing": null
  },
  "address_info": {
    "full_address": null,
    "house_flat_no": null,
    "street_area": null,
    "city": null,
    "district": null,
    "state": null,
    "pincode": null,
    "country": null
  }
}
"""

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess_image(image_path: str) -> str:
    image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.thumbnail((1800, 1800))

    image.save(image_path, optimize=True, quality=90)

    return image_path


# -------------------------
# CLEAN MODEL OUTPUT
# -------------------------
def clean_json_response(response_text: str) -> str:
    if not response_text:
        return "{}"

    response_text = response_text.strip()

    response_text = re.sub(r"^```json", "", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r"^```", "", response_text)
    response_text = re.sub(r"```$", "", response_text)

    return response_text.strip()


# -------------------------
# NORMALIZATION
# -------------------------
def normalize_extraction_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    personal_info = data.get("personal_info")

    if isinstance(personal_info, dict):

        if isinstance(personal_info.get("name"), dict):
            name_obj = personal_info["name"]

            first_name = str(name_obj.get("first_name", "")).strip()
            surname = str(name_obj.get("surname", "")).strip()

            personal_info["name"] = " ".join(
                x for x in [first_name, surname] if x
            ) or None

        if isinstance(personal_info.get("father_name"), dict):
            father_obj = personal_info["father_name"]

            first_name = str(father_obj.get("first_name", "")).strip()
            surname = str(father_obj.get("surname", "")).strip()

            personal_info["father_name"] = " ".join(
                x for x in [first_name, surname] if x
            ) or None

    document_info = data.get("document_info")
    if isinstance(document_info, dict):
        for field in ["document_type", "card_number", "issuing_authority", "municipality"]:
            value = document_info.get(field)
            if isinstance(value, dict):
                document_info[field] = value.get("raw") or value.get("value") or None

    address_info = data.get("address_info")
    if isinstance(address_info, dict):
        for field in [
            "house_flat_no", "street_area", "city",
            "district", "state", "pincode", "country"
        ]:
            value = address_info.get(field)
            if isinstance(value, dict):
                address_info[field] = value.get("raw") or value.get("value") or None

    return data

# -------------------------
# PIXTRAL CALL
# -------------------------
def call_pixtral(image_path: str, prompt: str) -> str:
    print(f"[PIXTRAL] Uploading image: {image_path}")

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    uploaded_file = client.files.upload(
        file={
            "file_name": os.path.basename(image_path),
            "content": file_bytes
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(
        file_id=uploaded_file.id
    )

    response = client.chat.complete(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": signed_url.url
                    }
                ]
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    if isinstance(content, list):
        content = "".join(str(x) for x in content)

    return content

# -------------------------
# RETRY LOGIC
# -------------------------
def extract_with_retry(image_path: str, retries: int = 3) -> str:
    last_error = None

    for attempt in range(retries):
        try:
            print(f"[ATTEMPT {attempt + 1}] Processing image")
            return call_pixtral(image_path,EXTRACTION_PROMPT)

        except Exception as e:
            last_error = e
            print(f"[ERROR] Attempt {attempt + 1}/{retries} failed: {e}")

            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise last_error

# -------------------------
# EMPTY RESPONSE
# -------------------------
def build_empty_response(error_message: str = None, error_type: str = None) -> Dict[str, Any]:
    response = {
        "document_info": DocumentInfo().model_dump(),
        "personal_info": PersonalInfo().model_dump(),
        "address_info": AddressInfo().model_dump()
    }

    if error_message:
        response["error"] = error_message

    if error_type:
        response["error_type"] = error_type

    return response

# -------------------------
# MAIN PIPELINE
# -------------------------
def validate_document(image_path: str) -> dict:
    try:
        response = call_pixtral(
            image_path,
            VALIDATION_PROMPT
        )

        cleaned = clean_json_response(response)

        data = json.loads(cleaned)

        return {
            "is_valid_document": bool(
                data.get("is_valid_document", False)
            ),
            "document_type": data.get("document_type"),
            "confidence": float(
                data.get("confidence", 0)
            )
        }

    except Exception as e:
        print(f"[VALIDATION ERROR] {e}")

        return {
            "is_valid_document": False,
            "document_type": None,
            "confidence": 0
        }
    
def extract_information(image_path: str) -> Dict[str, Any]:
    try:
        print(f"[START] Processing: {image_path}")

        preprocess_image(image_path)

        validation = validate_document(image_path)

        print("[DOCUMENT VALIDATION]")
        print(validation)

        if (
            not validation["is_valid_document"]
            or validation["confidence"] < 0.80
        ):
            return {
                "success": False,
                "error": (
                    "Uploaded image is not a supported "
                    "government identity document."
                ),
                "document_detected": False,
                "validation": validation,
            }

        print("[PIXTRAL RESPONSE]")

        raw_response = extract_with_retry(image_path)
        cleaned_response = clean_json_response(raw_response)
        data = json.loads(cleaned_response)
        data = normalize_extraction_data(data)

        print("[NORMALIZED RESPONSE]")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            validated = ExtractionResponse(**data)

            print("[SUCCESS] Schema validation passed")

            return validated.model_dump()

        except Exception as validation_error:
            print(
                f"[WARNING] Schema validation failed: "
                f"{validation_error}"
            )

            return {
                "document_info": data.get(
                    "document_info",
                    DocumentInfo().model_dump()
                ),
                "personal_info": data.get(
                    "personal_info",
                    PersonalInfo().model_dump()
                ),
                "address_info": data.get(
                    "address_info",
                    AddressInfo().model_dump()
                ),
                "error": f"Partial extraction: {validation_error}",
            }

    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing failed: {e}"

        print(f"[ERROR] {error_msg}")

        return build_empty_response(
            error_message=error_msg,
            error_type="JSONDecodeError",
        )

    except Exception as e:
        error_msg = str(e)

        print(f"[ERROR] Extraction failed: {error_msg}")

        response = build_empty_response(
            error_message=error_msg,
            error_type=type(e).__name__,
        )

        response["document_info"]["document_type"] = "UNKNOWN"

        return response