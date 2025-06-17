import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Dict, Any, Optional
from models import FaceEmbedding

# Define constants for filtering
MIN_FACE_SIZE = 80 # Minimum width or height of a detected face in pixels
MAX_ASPECT_RATIO = 1.5 # Max width/height or height/width ratio (e.g., 1.5 means 1:1.5 or 1.5:1)

def preprocess_image_for_detection(image_bytes: bytes) -> np.ndarray:
    """Loads image bytes and converts to RGB numpy array for face_recognition."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None # Or raise an error
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img

def filter_face_locations(face_locations: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Filters face locations based on size and aspect ratio."""
    filtered_locations = []
    for top, right, bottom, left in face_locations:
        width = right - left
        height = bottom - top

        # Filter by minimum size
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        # Filter by aspect ratio
        if width > 0 and height > 0:
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > MAX_ASPECT_RATIO:
                continue
        
        filtered_locations.append((top, right, bottom, left))
    return filtered_locations


def extract_face_embeddings_from_image(
    rgb_image: np.ndarray, 
    known_face_locations: Optional[List[Tuple[int, int, int, int]]] = None
) -> List[np.ndarray]:
    """Extracts face embeddings from an RGB image after applying filters."""
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image, model="hog") # Can use "cnn" for more accuracy if GPU is available
    
    # Apply filtering
    filtered_face_locations = filter_face_locations(face_locations)

    if not filtered_face_locations:
        return []

    # Extract embeddings for filtered faces
    face_encodings = face_recognition.face_encodings(rgb_image, filtered_face_locations)
    return face_encodings

def get_face_locations_and_embeddings(
    rgb_image: np.ndarray
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """Detects, filters, and extracts embeddings from an RGB image."""
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    filtered_face_locations = filter_face_locations(face_locations)

    if not filtered_face_locations:
        return [], []
    
    face_encodings = face_recognition.face_encodings(rgb_image, filtered_face_locations)
    return filtered_face_locations, face_encodings 