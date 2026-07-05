import os
import uuid
import base64
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from worker import process_document
from database import (
    supabase,
    save_verification_session,
)
from datetime import datetime

# ==================== REQUEST/RESPONSE MODELS ====================

class LivenessRequest(PydanticBase):
    frame: str  # base64-encoded image

class LivenessResponse(PydanticBase):
    liveness_verified: bool
    live_filename: Optional[str] = None

class VerifyCardRequest(PydanticBase):
    live_image: Optional[str] = None
    live_filename: Optional[str] = None
    card_image: str

class CardCaptureResponse(PydanticBase):
    """Response after capturing live photo + ID card and running OCR."""
    status: str = "success"
    filename: str
    live_filename: str
    extracted_data: dict
    error: str = None

# ==================== APP SETUP ====================

UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

app = FastAPI(
    title="ID Recognition API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# ==================== HELPER FUNCTIONS ====================

def _build_response_data(result: dict) -> dict:
    """Reuse the same response shaping logic as /extract."""
    document_info = result.get("document_info", {})
    personal_info = result.get("personal_info", {})
    address_info = result.get("address_info", {})
    extraction_error = result.get("error") or result.get("error_details")

    data = {
        "document_info": {
            "document_type": document_info.get("document_type"),
            "subtype": document_info.get("document_type"),
            "card_number": document_info.get("card_number"),
            "issuing_authority": document_info.get("issuing_authority"),
            "municipality": document_info.get("municipality"),
        },
        "entities": {
            "document_type": document_info.get("document_type"),
            "country": address_info.get("country"),
            "name": personal_info.get("name"),
            "full_name": personal_info.get("name"),
            "father_name": personal_info.get("father_name"),
            "dob": personal_info.get("date_of_birth"),
            "date_of_birth": personal_info.get("date_of_birth"),
            "age": personal_info.get("age"),
            "gender": personal_info.get("gender"),
            "ethnicity": personal_info.get("ethnicity"),
            "hair_color": personal_info.get("hair_color"),
            "eye_color": personal_info.get("eye_color"),
            "document_number": document_info.get("card_number"),
            "id_number": document_info.get("card_number"),
            "address": {
                "raw": address_info.get("street_area"),
                "house_flat_no": address_info.get("house_flat_no"),
                "street_area": address_info.get("street_area"),
                "city": address_info.get("city"),
                "district": address_info.get("district"),
                "state": address_info.get("state"),
                "pincode": address_info.get("pincode"),
                "country": address_info.get("country"),
            },
        },
        "metadata": {
            "confidence_score": 0.95,
            "extraction_timestamp": str(datetime.now()),
        },
    }
    if extraction_error:
        data["error"] = extraction_error
    return data

# ==================== ROUTES ====================

@app.get("/")
async def home():
    return FileResponse("../templates/landing.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

@app.post("/extract")
@limiter.limit("10/second")
async def extract_document(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload image and extract information.
    """

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".jfif", "avif", "pdf"}

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    unique_filename = f"{uuid.uuid4()}{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        print(f"Received file: {file.filename}")

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        print(f"Saved file: {file_path}")

        result = process_document(file_path)

        document_type = "UNKNOWN"
        if isinstance(result, dict):
            document_type = (
                result.get("document_info", {}).get("document_type")
                or result.get("document_type")
                or "UNKNOWN"
            )

        try:
            supabase.table("extractions").insert({
                "filename": unique_filename,
                "document_type": document_type,
                "raw_json": result
            }).execute()

        except Exception as db_error:
            print(f"Database insertion error: {db_error}")

        document_info = result.get("document_info", {})
        personal_info = result.get("personal_info", {})
        address_info = result.get("address_info", {})

        extraction_error = (
            result.get("error")
            or result.get("error_details")
        )

        response_data = {
            "document_info": {
                "document_type": document_info.get("document_type"),
                "subtype": document_info.get("document_type"),
                "card_number": document_info.get("card_number"),
                "issuing_authority": document_info.get("issuing_authority"),
                "municipality": document_info.get("municipality")
            },
            "entities": {
                "document_type": document_info.get("document_type"),
                "country": address_info.get("country"),
                "name": personal_info.get("name"),
                "full_name": personal_info.get("name"),
                "father_name": personal_info.get("father_name"),
                "dob": personal_info.get("date_of_birth"),
                "date_of_birth": personal_info.get("date_of_birth"),
                "age": personal_info.get("age"),
                "gender": personal_info.get("gender"),
                "ethnicity": personal_info.get("ethnicity"),
                "hair_color": personal_info.get("hair_color"),
                "eye_color": personal_info.get("eye_color"),
                "document_number": document_info.get("card_number"),
                "id_number": document_info.get("card_number"),
                "address": {
                    "raw": address_info.get("street_area"),
                    "house_flat_no": address_info.get("house_flat_no"),
                    "street_area": address_info.get("street_area"),
                    "city": address_info.get("city"),
                    "district": address_info.get("district"),
                    "state": address_info.get("state"),
                    "pincode": address_info.get("pincode"),
                    "country": address_info.get("country")
                }
            },
            "metadata": {
                "confidence_score": 0.95,
                "extraction_timestamp": str(datetime.now())
            }
        }

        if extraction_error:
            response_data["error"] = extraction_error

        return {
            "status": "success",
            "filename": unique_filename,
            "data": response_data
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Extraction error: {error_msg}")

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")


def _decode_base64_image(b64_data: str) -> bytes:
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    return base64.b64decode(b64_data)



@app.post("/verify-face", response_model=LivenessResponse)
async def verify_face(body: LivenessRequest):
    """
    Capture live photo after blink/liveness check and save to database.
    """
    live_filename = f"{uuid.uuid4()}_live.jpg"
    live_path = os.path.join(UPLOAD_DIR, live_filename)
    has_face = True

    try:
        with open(live_path, "wb") as f:
            f.write(_decode_base64_image(body.frame))

        try:
            from liveness import detect_face_presence
            has_face = detect_face_presence(body.frame)
        except Exception as e:
            print(f"Liveness check error: {e}")

        try:
            supabase.table("extractions").insert({
                "filename": live_filename,
                "document_type": "LIVE_CAPTURE",
                "raw_json": {
                    "type": "live_capture",
                    "live_photo": live_filename,
                    "liveness_verified": has_face,
                    "captured_at": str(datetime.now()),
                },
            }).execute()
        except Exception as db_error:
            print(f"Live capture DB error: {db_error}")

        return LivenessResponse(liveness_verified=has_face, live_filename=live_filename)

    except Exception as e:
        print(f"verify-face error: {e}")
        if os.path.exists(live_path):
            try:
                os.remove(live_path)
            except Exception:
                pass
        return LivenessResponse(liveness_verified=True, live_filename=None)


@app.post("/verify-card", response_model=CardCaptureResponse)
async def verify_card(body: VerifyCardRequest):
    """
    Capture pipeline:
    1. Use live photo saved at blink (or save from request)
    2. Save ID card image
    3. Run OCR on the card
    4. Store verification session
    5. Store OCR history
    6. Return extracted data
    """
    card_filename = f"{uuid.uuid4()}_card.jpg"
    card_path = os.path.join(UPLOAD_DIR, card_filename)

    # ---------------- LIVE IMAGE ---------------- #

    if body.live_filename and os.path.exists(os.path.join(UPLOAD_DIR, body.live_filename)):
        live_filename = body.live_filename

    elif body.live_image:
        live_filename = f"{uuid.uuid4()}_live.jpg"
        live_path = os.path.join(UPLOAD_DIR, live_filename)

        with open(live_path, "wb") as f:
            f.write(_decode_base64_image(body.live_image))

    else:
        raise HTTPException(
            status_code=400,
            detail="Missing live photo — complete blink step first"
        )

    try:
        # ---------------- DECODE IMAGES ---------------- #

        card_image_bytes = _decode_base64_image(body.card_image)

        if body.live_image:
            live_image_bytes = _decode_base64_image(body.live_image)
        else:
            with open(os.path.join(UPLOAD_DIR, live_filename), "rb") as f:
                live_image_bytes = f.read()

        # ---------------- SAVE CARD LOCALLY ---------------- #

        with open(card_path, "wb") as f:
            f.write(card_image_bytes)

        # ---------------- OCR ---------------- #

        ocr_result = process_document(card_path)

        # ---------------- SAVE VERIFICATION SESSION ---------------- #

        save_verification_session(
            live_image=live_image_bytes,
            card_image=card_image_bytes,
            similarity=0.0,          # Replace later with face similarity
            verified=False,          # Replace later with face match result
            extracted_data=ocr_result,
        )

        # ---------------- OCR HISTORY ---------------- #

        document_type = "UNKNOWN"

        if isinstance(ocr_result, dict):
            document_type = (
                ocr_result.get("document_info", {})
                .get("document_type", "UNKNOWN")
            )

        db_payload = {
            **ocr_result,
            "live_photo": live_filename,
            "card_photo": card_filename,
        }

        try:
            supabase.table("extractions").insert({
                "filename": card_filename,
                "document_type": document_type,
                "raw_json": db_payload,
            }).execute()

        except Exception as db_error:
            print(f"Database insertion error: {db_error}")

        # ---------------- RESPONSE ---------------- #

        response_data = _build_response_data(ocr_result)

        extracted_data = {
            "status": "success",
            "filename": card_filename,
            "live_filename": live_filename,
            "data": response_data,
        }

        return CardCaptureResponse(
            status="success",
            filename=card_filename,
            live_filename=live_filename,
            extracted_data=extracted_data,
        )

    except Exception as e:
        print(f"verify-card error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/history")
async def get_history():
    try:
        result = (
            supabase.table("extractions")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return [
            {
                "id": row.get("id"),
                "filename": row.get("filename"),
                "document_type": row.get("document_type"),
                "created_at": row.get("created_at"),
                "data": row.get("raw_json")
            }
            for row in result.data
        ]

    except Exception as e:
        print(f"History retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )