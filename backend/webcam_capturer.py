import cv2
import base64
import numpy as np
from typing import Optional, Tuple


class WebcamCapturer:
    """
    Captures faces from webcam with quality checks and preprocessing.
    Optimized for 720p but works with any resolution.
    """
    
    def __init__(self, camera_id=0, resolution=(1280, 720), fps=30):
        """
        Args:
            camera_id: Camera device ID (usually 0 for default)
            resolution: Target resolution (width, height)
            fps: Target framerate
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.frame_count = 0
    
    def open(self):
        """Initialize camera and set optimal parameters."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Disable auto-focus for clearer images
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Increase exposure time for better light sensitivity
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 1)
        
        # Reduce auto-exposure to manual control
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=manual, 3=auto
        
        # Set white balance
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 32)
        
        print(f"Camera opened: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
    
    def get_frame_quality(self, frame: np.ndarray) -> Tuple[float, str]:
        """
        Assess frame quality.
        Returns (quality_score: 0-1, status: str)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur detection (Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 200.0)  # Normalized to 0-1
        
        # Brightness check
        brightness = np.mean(gray)
        brightness_score = 1.0 if 50 < brightness < 200 else 0.5
        
        # Overall quality
        quality = (blur_score * 0.7) + (brightness_score * 0.3)
        
        if laplacian_var < 100:
            status = "BLURRY"
        elif brightness < 50:
            status = "TOO_DARK"
        elif brightness > 200:
            status = "TOO_BRIGHT"
        else:
            status = "OK"
        
        return quality, status, laplacian_var, brightness
    
    def capture_best_frame(self, num_samples=10) -> Optional[np.ndarray]:
        """
        Capture multiple frames and return the clearest one.
        
        Args:
            num_samples: Number of frames to check
        
        Returns:
            Best frame or None if all are poor quality
        """
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        
        frames = []
        qualities = []
        
        for _ in range(num_samples):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            quality, status, blur, brightness = self.get_frame_quality(frame)
            frames.append(frame)
            qualities.append((quality, status, blur, brightness))
            
            print(f"Frame {len(frames)}: {status} (quality={quality:.2f}, blur={blur:.1f}, brightness={int(brightness)})")
        
        if not frames:
            return None
        
        # Return frame with highest quality
        best_idx = np.argmax([q[0] for q in qualities])
        best_quality, best_status, best_blur, best_brightness = qualities[best_idx]
        
        print(f"\nBest frame: {best_status} (quality={best_quality:.2f})")
        
        if best_quality < 0.5:
            print("⚠️  Warning: Low frame quality. Ensure good lighting and steady hand.")
        
        return frames[best_idx]
    
    def frame_to_b64(self, frame: np.ndarray) -> str:
        """Convert OpenCV BGR frame to base64 string."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return b64
    
    def close(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            print("Camera closed")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    # Capture with context manager (auto-closes camera)
    with WebcamCapturer(camera_id=0, resolution=(1280, 720), fps=30) as capturer:
        print("Capturing live face... Keep steady!")
        live_frame = capturer.capture_best_frame(num_samples=10)
        
        if live_frame is not None:
            live_b64 = capturer.frame_to_b64(live_frame)
            print(f"Captured live image: {len(live_b64)} bytes")
            
            # Save locally for testing
            cv2.imwrite("/tmp/live_face.jpg", live_frame)
            print("Saved to /tmp/live_face.jpg")
        else:
            print("Failed to capture clear frame")