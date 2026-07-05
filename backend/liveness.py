import base64
import numpy as np
import cv2

def _b64_to_cv2(b64_str: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_face_presence(b64_frame: str) -> bool:
    """
    Confirm at least one face is visible in the frame using MediaPipe FaceMesh.
    Used as a server-side sanity check after the client reports a blink.
    """
    import mediapipe as mp

    image = _b64_to_cv2(b64_frame)
    if image is None:
        return False

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)
        return results.multi_face_landmarks is not None
