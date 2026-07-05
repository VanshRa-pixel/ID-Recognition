import base64
import numpy as np
import cv2

# Lazy-loaded so startup doesn't fail if insightface isn't installed yet
_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def _b64_to_cv2(b64_str: str):
    """Decode a base64 image string (with or without data-URL prefix) to a BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _is_blurry(image_bgr, threshold=100.0):
    """
    Detect blur using Laplacian variance.
    Returns (is_blurry: bool, variance: float)
    Lower variance = more blur. Threshold ~100 for 720p.
    """
    if image_bgr is None:
        return True, 0.0
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def _enhance_image(image_bgr):
    """
    Apply preprocessing to improve face detection:
    - Contrast enhancement (CLAHE)
    - Gentle sharpening
    - Noise reduction
    """
    if image_bgr is None:
        return None
    
    # Convert to LAB for better contrast enhancement
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Gentle sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) / 1.5
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Light bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return enhanced


def get_embedding(image_bgr, min_confidence=0.9):
    """
    Return the L2-normalised face embedding for the largest detected face.
    
    Args:
        image_bgr: BGR image array
        min_confidence: Minimum detection confidence (0-1)
    
    Returns:
        (embedding: ndarray, confidence: float) or (None, 0.0) on failure
    """
    app = _get_face_app()
    
    # Preprocess the image
    enhanced = _enhance_image(image_bgr)
    if enhanced is None:
        return None, 0.0
    
    # Check for blur
    is_blur, blur_score = _is_blurry(enhanced, threshold=100.0)
    if is_blur:
        return None, 0.0  # Too blurry, reject
    
    faces = app.get(enhanced)
    if not faces:
        return None, 0.0
    
    # Pick the largest face by bounding-box area
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    
    # Check confidence score if available
    if hasattr(face, 'det_score') and face.det_score < min_confidence:
        return None, face.det_score
    
    emb = face.embedding
    norm = np.linalg.norm(emb)
    normalized_emb = emb / (norm + 1e-10)
    
    confidence = getattr(face, 'det_score', 1.0)
    return normalized_emb, confidence


def compare_faces(b64_live: str, b64_card: str, similarity_threshold=0.50) -> dict:
    """
    Compare a live webcam face with the face photo on an ID card.
    
    Returns:
        {
            "matched": bool,
            "similarity": float,
            "live_confidence": float,
            "card_confidence": float,
            "live_blur_score": float,
            "card_blur_score": float,
            "error": str (optional)
        }
    
    Threshold is 0.50 (relaxed) for low-quality card photos.
    """
    live_img = _b64_to_cv2(b64_live)
    card_img = _b64_to_cv2(b64_card)

    if live_img is None:
        return {
            "matched": False,
            "similarity": 0.0,
            "error": "Could not decode live image"
        }
    if card_img is None:
        return {
            "matched": False,
            "similarity": 0.0,
            "error": "Could not decode card image"
        }

    # Check blur on original images
    live_is_blur, live_blur_score = _is_blurry(live_img)
    card_is_blur, card_blur_score = _is_blurry(card_img)
    
    if live_is_blur:
        return {
            "matched": False,
            "similarity": 0.0,
            "live_blur_score": live_blur_score,
            "error": f"Live image too blurry (score: {live_blur_score:.2f}, need >100)"
        }
    
    if card_is_blur:
        return {
            "matched": False,
            "similarity": 0.0,
            "card_blur_score": card_blur_score,
            "error": f"Card image too blurry (score: {card_blur_score:.2f}, need >100)"
        }

    # Get embeddings with confidence
    live_emb, live_conf = get_embedding(live_img)
    card_emb, card_conf = get_embedding(card_img)

    if live_emb is None:
        return {
            "matched": False,
            "similarity": 0.0,
            "live_confidence": live_conf,
            "error": f"No face detected in live image (confidence: {live_conf:.2f})"
        }
    if card_emb is None:
        return {
            "matched": False,
            "similarity": 0.0,
            "card_confidence": card_conf,
            "error": f"No face detected in card image (confidence: {card_conf:.2f})"
        }

    # Cosine similarity (dot product of L2-normalised vectors)
    similarity = float(np.dot(live_emb, card_emb))
    
    return {
        "matched": similarity > similarity_threshold,
        "similarity": round(similarity, 4),
        "live_confidence": round(live_conf, 4),
        "card_confidence": round(card_conf, 4),
        "live_blur_score": round(live_blur_score, 2),
        "card_blur_score": round(card_blur_score, 2)
    }