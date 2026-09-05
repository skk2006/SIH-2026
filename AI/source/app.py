from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import base64
import os
import re
import logging
import numpy as np
import sys
if not hasattr(np, '_core'):
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = getattr(_core, 'multiarray', None)
    sys.modules['numpy._core.umath'] = getattr(_core, 'umath', None)
    sys.modules['numpy._core._multiarray_umath'] = getattr(_core, '_multiarray_umath', None)

import werkzeug
if not hasattr(werkzeug, '__version__'):
    werkzeug.__version__ = '3.0.0'

import mediapipe as mp
from keras_facenet import FaceNet
import pickle
from datetime import datetime
import time
from pymongo import MongoClient
from email.message import EmailMessage
import smtplib
import ssl
import threading
from queue import Queue
from twilio.rest import Client
import torch
from ultralytics import YOLO
import easyocr
import sys

# Add submodules to Python path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_PATH, "LOITERING_MODULE"))
sys.path.append(os.path.join(BASE_PATH, "night_detection"))
from loitering_detector import LoiteringDetector
from pipeline import DeepMultimodalThreatPipeline

task_queue = Queue()
vehicle_queue = Queue()


app = Flask(__name__)
app.secret_key = "your_secret_key"

# Email Configuration
EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'your_email@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your_email_password')
EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER', 'receiver_email@gmail.com')

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'your_account_sid_here')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'your_auth_token_here')    
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', 'your_twilio_phone_number')      
RECIPIENT_PHONE_NUMBER = os.environ.get('RECIPIENT_PHONE_NUMBER', 'your_recipient_phone_number') 

class LocalFallbackCollection:
    def __init__(self, filepath):
        self.filepath = filepath

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "rb") as f:
                    data = pickle.load(f)
                    return data if isinstance(data, list) else []
            except Exception as e:
                print(f"Error loading local db {self.filepath}: {e}")
                return []
        return []

    def _save(self, data):
        try:
            with open(self.filepath, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving local db {self.filepath}: {e}")

    def insert_one(self, document):
        data = self._load()
        try:
            from bson import ObjectId
            id_val = ObjectId()
        except Exception:
            import uuid
            id_val = str(uuid.uuid4())
        document = dict(document)
        if "_id" not in document:
            document["_id"] = id_val
        data.append(document)
        self._save(data)
        return document

    def find(self, filter=None):
        return self._load()

    def delete_many(self, filter=None):
        if not filter:
            self._save([])
            return 0
        self._save([])
        return 0


class SafeCollectionWrapper:
    """Safely attempts PyMongo collection operations; falls back immediately and gracefully to LocalFallbackCollection on any failure or timeout."""
    def __init__(self, get_coll_fn, filepath):
        self.get_coll_fn = get_coll_fn
        self.fallback = LocalFallbackCollection(filepath)

    def insert_one(self, document):
        try:
            coll = self.get_coll_fn()
            if coll is not None:
                return coll.insert_one(document)
        except Exception as e:
            print(f"Mongo insert failed ({e}), saving to local fallback: {self.fallback.filepath}")
        return self.fallback.insert_one(document)

    def find(self, filter=None):
        try:
            coll = self.get_coll_fn()
            if coll is not None:
                return list(coll.find(filter or {}))
        except Exception as e:
            print(f"Mongo find failed ({e}), loading from local fallback: {self.fallback.filepath}")
        return self.fallback.find(filter)

    def delete_many(self, filter=None):
        try:
            coll = self.get_coll_fn()
            if coll is not None:
                coll.delete_many(filter or {})
        except Exception as e:
            print(f"Mongo delete failed ({e}), clearing local fallback: {self.fallback.filepath}")
        return self.fallback.delete_many(filter)


# Connect to MongoDB with automatic fallback
mongo_client = None
mongo_db = None
try:
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
    mongo_client.server_info() # quick connection check
    mongo_db = mongo_client["face_recognition"]
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"MongoDB not available ({e}). Using local fallback databases.")
    mongo_client = None
    mongo_db = None

collection = SafeCollectionWrapper(
    lambda: mongo_db["suspects"] if mongo_db is not None else None,
    os.path.join(BASE_PATH, "suspects_db.pkl")
)
vehicles_collection = SafeCollectionWrapper(
    lambda: mongo_db["vehicles"] if mongo_db is not None else None,
    os.path.join(BASE_PATH, "vehicles_db.pkl")
)


# Initialize models
embedder = FaceNet()

# ── Thread safety locks for C++ deep learning runtimes ──
face_detection_lock = threading.Lock()
embedder_lock = threading.Lock()
stream_token_lock = threading.Lock()
active_stream_token = 0

# ── Global Camera Pool for instant mode switching ──
_camera_pool = {}          # {source_key: ThreadedCamera}
_camera_pool_refs = {}     # {source_key: int}  reference count
_camera_pool_lock = threading.Lock()

def acquire_camera(source, width=640, height=480, max_retries=3):
    """Get or create a ThreadedCamera from the global pool."""
    key = str(source)
    with _camera_pool_lock:
        if key in _camera_pool and _camera_pool[key].is_opened():
            _camera_pool_refs[key] = _camera_pool_refs.get(key, 0) + 1
            return _camera_pool[key]
        # Create new camera
        cam = ThreadedCamera(source, width=width, height=height, max_retries=max_retries)
        if cam.is_opened():
            _camera_pool[key] = cam
            _camera_pool_refs[key] = 1
        return cam

def release_camera(source):
    """Decrement reference count; only physically release when no users remain."""
    key = str(source)
    with _camera_pool_lock:
        if key in _camera_pool_refs:
            _camera_pool_refs[key] -= 1
            if _camera_pool_refs[key] <= 0:
                cam = _camera_pool.pop(key, None)
                _camera_pool_refs.pop(key, None)
                if cam is not None:
                    cam.release()

def flush_camera_pool():
    """Force-release all cameras in the pool (used by stop_feed)."""
    with _camera_pool_lock:
        for key, cam in list(_camera_pool.items()):
            try:
                cam.release()
            except Exception:
                pass
        _camera_pool.clear()
        _camera_pool_refs.clear()

def get_face_embedding(face_img):
    """Thread-safe FaceNet embedding inference."""
    with embedder_lock:
        return embedder.embeddings([face_img])[0]

def compute_face_embeddings(images):
    """Compute embeddings cleanly without printing Keras progress bars."""
    if not images:
        return []
    s = embedder.metadata.get('image_size', 160)
    resized = [cv2.resize(img, (s, s)) for img in images]
    X = np.float32([embedder._normalize(img) for img in resized])
    return embedder.model.predict(X, verbose=0)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} for AI models.")

vehicle_model = YOLO(os.path.join(BASE_PATH, "yolov8n.pt"))
vehicle_model.to(device)

weapon_model = YOLO(os.path.join(BASE_PATH, "weapon.pt"))
weapon_model.to(device)
try:
    _dummy = np.zeros((384, 384, 3), dtype=np.uint8)
    weapon_model(_dummy, device=device, verbose=False)
    vehicle_model(_dummy, device=device, verbose=False)
except Exception:
    pass
# Suppress EasyOCR verbosity
logging.getLogger("easyocr").setLevel(logging.ERROR)
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

print("Initializing Loitering Detector...")
loitering_detector = LoiteringDetector(
    yolo_path=os.path.join(BASE_PATH, "yolov8n.pt"),
    classifier_path=os.path.join(BASE_PATH, "LOITERING_MODULE", "loitering_model.pkl")
)

print("Initializing Fight Detection Pipeline...")
fight_pipeline = DeepMultimodalThreatPipeline()

current_head_count = 0

# Initialize YuNet for fast head counting
yunet_model_path = os.path.join(BASE_PATH, "face_detection_yunet_2023mar.onnx")
face_detector_yunet = None
if os.path.exists(yunet_model_path):
    print("Initializing YuNet for Head Counting...")
    face_detector_yunet = cv2.FaceDetectorYN.create(
        model=yunet_model_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
else:
    print("YuNet model not found at", yunet_model_path)


# --- Dedicated Night Detection Engine (Lazy Loaded) ---
night_detector_instance = None

def get_night_detector():
    """Retrieve or lazily initialize the dedicated night surveillance detector."""
    global night_detector_instance
    if night_detector_instance is None:
        try:
            from night_detection import NightDetector
            model_file = os.path.join(BASE_PATH, "night_detection", "night_model.pt")
            night_detector_instance = NightDetector(model_path=model_file)
            print("Dedicated Night Detection Engine initialized successfully.")
        except Exception as err:
            print(f"Failed to initialize NightDetector: {err}")
    return night_detector_instance


# --- Find the best available plate detector model across all training runs ---
def find_best_plate_model():
    """Search all training run directories for the best available plate detector weights."""
    candidates = [
        "anpr_model.pt",  # user's specific model
        "model/best.pt",  # manually placed model
        "runs/detect/license_plate_detector/weights/best.pt",
    ]
    # Also search numbered runs (license_plate_detector2, 3, 4, ...)
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")
    if os.path.isdir(runs_dir):
        for entry in sorted(os.listdir(runs_dir), reverse=True):
            # Prefer higher-numbered runs (trained later = better)
            candidate = os.path.join(runs_dir, entry, "weights", "best.pt")
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                candidates.insert(0, candidate)
    
    for path in candidates:
        abs_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            return abs_path
    return None

plate_model_path = find_best_plate_model()
if plate_model_path:
    print(f"Loading Custom Trained YOLO ALPR model: {plate_model_path}")
    plate_detector = YOLO(plate_model_path)
    plate_detector.to(device)
else:
    print("Custom Plate Detector not found. Using Haar Cascade fallback.")
    plate_detector = None
    
# Plate detection cascade (Fallback)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Characters allowed on license plates (include separators like - and .)
PLATE_ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
PLATE_ALLOWLIST_STRICT = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def clean_plate_text(raw_text):
    """Clean OCR output: keep only plate-valid characters, normalize separators."""
    text = raw_text.strip().upper()
    # Replace common OCR mis-reads for separators
    text = text.replace('|', '1').replace('\\', '1').replace('(', '').replace(')', '')
    text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    # Keep only valid plate characters
    cleaned = ''.join(c for c in text if c.isalnum() or c in '-.')
    # Remove leading/trailing separators
    cleaned = cleaned.strip('-.')
    return cleaned

def apply_ocr_corrections(text):
    """Apply post-processing corrections for known OCR confusions on license plate fonts.
    These are systematic errors where EasyOCR consistently misreads certain characters."""
    if not text:
        return text
    
    corrected = text
    
    # --- Fix merged characters ---
    # "51" often merges into "6" when characters are close together
    # If we see a plate starting with "6" followed by a letter, it's likely "51"
    # Vietnamese plates: ##X-###.## where ## is province code (2 digits), X is letter
    import re as _re
    
    # Pattern: plate starts with single digit + letter (like "6F" or "6E") 
    # but should be two digits + letter (like "51F", "51E")
    # Common province codes: 51 (Ho Chi Minh), 30 (Ha Noi), 43 (Da Nang), etc.
    match = _re.match(r'^(\d)([A-Z])', corrected)
    if match:
        single_digit = match.group(1)
        letter = match.group(2)
        rest = corrected[len(match.group(0)):]
        
        # Map of commonly merged digit pairs
        # When "5" and "1" are close, OCR reads "6"
        # When "3" and "1" are close, OCR reads "4" or "31"
        # When "6" and "1" are close, OCR reads "6" (absorbs the 1)
        merge_corrections = {
            '6': '51',  # 5+1 merged (most common for Vietnamese plates)
            '8': '51',  # 5+1 merged (alternative misread)
            '4': '41',  # 4+1 merged
            '9': '91',  # 9+1 merged (or could be actual 9)
        }
        
        if single_digit in merge_corrections:
            # Check if corrected version gives a known Vietnamese province code
            expanded = merge_corrections[single_digit]
            # Known Vietnamese province codes that start with these digits
            known_provinces = [
                '11','12','14','15','16','17','18','19','20','21','22','23','24','25','26','27',
                '29','30','31','32','33','34','35','36','37','38','39','40','41','43','47','48',
                '49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64',
                '65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80',
                '81','82','83','84','85','86','88','89','90','92','93','94','95','97','98','99'
            ]
            if expanded in known_provinces:
                corrected = expanded + letter + rest
    
    # --- Fix common single-character confusions ---
    # These happen at specific positions in the plate
    # E↔F confusion: In the letter position of a plate, "E" is often misread for "F" and vice versa
    # We can't fix this without context, but we can note it happens
    
    # O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6 confusions
    # For the LETTER position (3rd character in Vietnamese plates), prefer letters
    # For the NUMBER positions, prefer digits
    if len(corrected) >= 3:
        parts = list(corrected)
        # In Vietnamese plates, position index 2 (0-indexed) is always a letter
        letter_pos = None
        for i, c in enumerate(parts):
            if c.isalpha():
                letter_pos = i
                break
        
        if letter_pos is not None:
            # Characters before the letter position should be digits
            for i in range(letter_pos):
                if parts[i] == 'O': parts[i] = '0'
                elif parts[i] == 'I': parts[i] = '1'
                elif parts[i] == 'S': parts[i] = '5'
                elif parts[i] == 'B': parts[i] = '8'
                elif parts[i] == 'Z': parts[i] = '2'
                elif parts[i] == 'G': parts[i] = '6'
                elif parts[i] == 'T': parts[i] = '7'
                elif parts[i] == 'A': parts[i] = '4'
            
            # Characters after the letter position should be digits (with . and - allowed)
            for i in range(letter_pos + 1, len(parts)):
                if parts[i] == 'O': parts[i] = '0'
                elif parts[i] == 'I': parts[i] = '1'
                elif parts[i] == 'l': parts[i] = '1'
                elif parts[i] == 'S': parts[i] = '5'
                elif parts[i] == 'B': parts[i] = '8'
                elif parts[i] == 'Z': parts[i] = '2'
                elif parts[i] == 'G': parts[i] = '6'
                elif parts[i] == 'T': parts[i] = '7'
                elif parts[i] == 'A': parts[i] = '4'
                elif parts[i] == 'D': parts[i] = '0'
        
        corrected = ''.join(parts)
    
    return corrected

def format_plate_text(text):
    """Try to format the plate text into a standard Vietnamese plate format: ##X-###.##"""
    import re as _re
    
    # Remove all separators to get raw alphanumeric
    raw = text.replace('-', '').replace('.', '').replace(' ', '')
    
    if len(raw) < 7:
        return text  # Too short to format
    
    # Try to match Vietnamese plate pattern: 2 digits + 1 letter + 3-5 digits
    match = _re.match(r'^(\d{2})([A-Z]\d?)(\d{3,5})$', raw)
    if match:
        prefix = match.group(1)     # Province code (e.g., "51")
        series = match.group(2)     # Series letter (e.g., "F")
        numbers = match.group(3)    # Number portion
        
        # Format as ##X-###.## if number portion has 5+ digits
        if len(numbers) >= 5:
            return f"{prefix}{series}-{numbers[:3]}.{numbers[3:]}"
        elif len(numbers) >= 3:
            return f"{prefix}{series}-{numbers}"
    
    return text

def is_valid_plate(text):
    """Check if text looks like a valid license plate (supports formats with - and .)"""
    # Strip separators for length/content check
    core = text.replace('-', '').replace('.', '').replace(' ', '')
    if len(core) < 5 or len(core) > 12:
        return False
    letters = sum(1 for c in core if c.isalpha())
    numbers = sum(1 for c in core if c.isdigit())
    # A valid plate needs at least 1 letter and 2 numbers
    return letters >= 1 and numbers >= 2

def sort_ocr_results_reading_order(ocr_results):
    """Sort OCR results in reading order: top-to-bottom first, then left-to-right.
    This handles two-line plates (common in Asian countries)."""
    if not ocr_results:
        return ocr_results
    
    # Get vertical midpoints of each result
    midpoints = []
    for (bbox, text, prob) in ocr_results:
        y_mid = (bbox[0][1] + bbox[2][1]) / 2
        x_mid = (bbox[0][0] + bbox[2][0]) / 2
        midpoints.append((y_mid, x_mid))
    
    if len(midpoints) <= 1:
        return ocr_results
    
    # Calculate the height of the tallest bounding box to detect line breaks
    heights = [(bbox[2][1] - bbox[0][1]) for (bbox, _, _) in ocr_results]
    avg_height = sum(heights) / len(heights) if heights else 20
    
    # Group into lines: if two results are within half the avg char height, they're on the same line
    line_threshold = avg_height * 0.6
    
    # Create list of (y_mid, x_mid, index)
    indexed = [(midpoints[i][0], midpoints[i][1], i) for i in range(len(ocr_results))]
    # Sort by y first, then by x
    indexed.sort(key=lambda item: (item[0] // line_threshold, item[1]))
    
    return [ocr_results[idx] for (_, _, idx) in indexed]

def run_ocr_on_image(img, allowlist=PLATE_ALLOWLIST):
    """Run EasyOCR on a preprocessed image with optimized parameters."""
    try:
        ocr_results = ocr_reader.readtext(
            img,
            allowlist=allowlist,
            paragraph=False,
            min_size=5,
            text_threshold=0.4,
            low_text=0.3,
            width_ths=0.3,       # CRITICAL: prevent merging adjacent character boxes
            mag_ratio=1.0,       # Fast mode without internal upscale
            slope_ths=0.2,       # Allow slight rotation
        )
        return ocr_results if ocr_results else []
    except Exception:
        return []

def extract_text_from_ocr_results(ocr_results):
    """Extract and concatenate text from OCR results in reading order."""
    if not ocr_results:
        return "", 0
    
    ocr_results = sort_ocr_results_reading_order(ocr_results)
    
    parts = []
    total_conf = 0
    count = 0
    
    for (bbox, text, prob) in ocr_results:
        if prob > 0.15:
            cleaned = clean_plate_text(text)
            if cleaned:
                parts.append(cleaned)
                total_conf += prob
                count += 1
    
    if not parts:
        return "", 0
    
    avg_conf = total_conf / count if count > 0 else 0
    
    # Determine if this is a two-line plate by checking y-coordinates
    if len(ocr_results) >= 2:
        y_positions = [(bbox[0][1] + bbox[2][1]) / 2 for (bbox, _, _) in ocr_results]
        heights = [(bbox[2][1] - bbox[0][1]) for (bbox, _, _) in ocr_results]
        avg_h = sum(heights) / len(heights) if heights else 1
        
        # If there's a significant vertical gap between results, it's two lines
        if max(y_positions) - min(y_positions) > avg_h * 0.5 and len(parts) == 2:
            concatenated = parts[0] + '-' + parts[1]
        else:
            concatenated = ''.join(parts)
    else:
        concatenated = ''.join(parts)
    
    return concatenated, avg_conf

def ocr_plate_image(plate_img_color):
    """Run OCR on a plate image using multiple preprocessing strategies, 
    character separation, and post-processing corrections."""
    if plate_img_color is None or plate_img_color.size == 0:
        return "UNKNOWN"
    
    h, w = plate_img_color.shape[:2]
    if h < 5 or w < 10:
        return "UNKNOWN"
    
    # Tier 1: Try optimal scale and 2 best pre-processing methods
    scale = 300 / w if w > 0 else 2.0
    if scale < 0.8:
        scale = 1.0  # Avoid heavy downscaling
    
    scaled = cv2.resize(plate_img_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    
    preprocessed_tier1 = []
    
    # 1. CLAHE with bilateral filter
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bfilter)
    preprocessed_tier1.append(enhanced)
    
    # 2. Otsu binarization
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_tier1.append(otsu)
    
    all_candidates = []
    
    # Run Tier 1
    for img in preprocessed_tier1:
        results = run_ocr_on_image(img, allowlist=PLATE_ALLOWLIST)
        text, conf = extract_text_from_ocr_results(results)
        if text and len(text) >= 4:
            all_candidates.append((text, conf))
            # Immediate early exit if valid plate found with good confidence
            if conf > 0.65 and is_valid_plate(text):
                corrected = apply_ocr_corrections(text)
                formatted = format_plate_text(corrected)
                best = formatted
                while '--' in best: best = best.replace('--', '-')
                while '..' in best: best = best.replace('..', '.')
                return best.strip('-.')
    
    # Tier 2: If Tier 1 failed or had low confidence, try more methods
    preprocessed_tier2 = []
    
    # 3. Morphological opening to SEPARATE touching characters (key fix!)
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph_opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel_sep)
    preprocessed_tier2.append(morph_opened)
    
    # 4. Sharpened CLAHE
    sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen)
    preprocessed_tier2.append(sharpened)
    
    for img in preprocessed_tier2:
        results = run_ocr_on_image(img, allowlist=PLATE_ALLOWLIST)
        text, conf = extract_text_from_ocr_results(results)
        if text and len(text) >= 4:
            all_candidates.append((text, conf))
            
    # Also try strict mode (no separators) on Otsu
    results_strict = run_ocr_on_image(otsu, allowlist=PLATE_ALLOWLIST_STRICT)
    text_strict, conf_strict = extract_text_from_ocr_results(results_strict)
    if text_strict and len(text_strict) >= 4:
        all_candidates.append((text_strict, conf_strict))
    
    if not all_candidates:
        return "UNKNOWN"
    
    # --- Apply corrections to all candidates ---
    corrected_candidates = []
    for text, conf in all_candidates:
        corrected = apply_ocr_corrections(text)
        formatted = format_plate_text(corrected)
        corrected_candidates.append((formatted, conf))
        # Also keep the unformatted corrected version
        if corrected != formatted:
            corrected_candidates.append((corrected, conf * 0.9))
            
    # --- Score and pick the best candidate ---
    import re as _re
    scored = []
    for text, conf in corrected_candidates:
        score = conf
        raw = text.replace('-', '').replace('.', '')
        
        # Bonus for matching Vietnamese plate format
        if _re.match(r'^\d{2}[A-Z]\d?-\d{3}\.\d{2}$', text):
            score += 2.0  
        elif _re.match(r'^\d{2}[A-Z]', raw):
            score += 0.5  
        
        if is_valid_plate(text):
            score += 0.5
        
        if 7 <= len(text) <= 12:
            score += 0.3
        
        if text.startswith('0') or text.startswith('00'):
            score -= 0.5
        
        scored.append((text, score, conf))
    
    # Sort by score (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    
    # Final cleanup
    while '--' in best: best = best.replace('--', '-')
    while '..' in best: best = best.replace('..', '.')
    best = best.strip('-.')
    
    return best if len(best) >= 4 else "UNKNOWN"

def extract_plate_text(vehicle_crop):
    """Attempt to find a license plate in the vehicle crop and OCR it."""
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    
    plate_img = vehicle_crop
    plate_found = False
    
    # 1. Try Custom YOLO Model First (pick highest confidence detection)
    if plate_detector is not None:
        results = plate_detector(vehicle_crop, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            # Pick the detection with the highest confidence
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())
            
            # Add padding around the plate for better OCR (5% each side)
            h_crop, w_crop = vehicle_crop.shape[:2]
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.08)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w_crop, x2 + pad_x)
            y2 = min(h_crop, y2 + pad_y)
            
            plate_img = np.ascontiguousarray(vehicle_crop[y1:y2, x1:x2].copy())
            plate_found = True
            
    # 2. Try Haar Cascade if YOLO wasn't loaded or found nothing
    if not plate_found:
        plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
        if len(plates) > 0:
            plates = sorted(plates, key=lambda x: x[2]*x[3], reverse=True)
            px, py, pw, ph = plates[0]
            # More generous padding for Haar cascade
            px = max(0, px - 10)
            py = max(0, py - 10)
            pw = min(vehicle_crop.shape[1] - px, pw + 20)
            ph = min(vehicle_crop.shape[0] - py, ph + 20)
            plate_img = np.ascontiguousarray(vehicle_crop[py:py+ph, px:px+pw].copy())
            plate_found = True
            
    # 3. Fallback: crop the bottom half, avoiding extreme left/right edges
    if not plate_found:
        h, w = vehicle_crop.shape[:2]
        plate_img = vehicle_crop[int(h*0.5):h, int(w*0.1):int(w*0.9)]
        
    if plate_img.size == 0:
        return "UNKNOWN"

    return ocr_plate_image(plate_img)

# Directories
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "uploads")
DATASET_FOLDER = os.path.join(BASE_PATH, "dataset")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

DETECTED_VEHICLES_FOLDER = os.path.join(BASE_PATH, "detected_vehicles")
os.makedirs(DETECTED_VEHICLES_FOLDER, exist_ok=True)

# Load known faces embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    known_faces = pickle.load(f)

# Pre-normalize embeddings for faster matching
normalized_known_faces = {}
for person, embeddings in known_faces.items():
    if embeddings:
        embs = np.array(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        normalized_known_faces[person] = embs / norms


KNOWN_FACE_THRESHOLD = 0.88

def extract_face(img, detector=None):
    use_global = detector is None
    if use_global:
        detector = face_detector
    img_rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if use_global:
        with face_detection_lock:
            results = detector.process(img_rgb)
    else:
        results = detector.process(img_rgb)
    faces = []

    if results.detections:
        h, w, _ = img.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                   int(bboxC.width * w), int(bboxC.height * h))
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + width), min(h, y + height)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Use contiguous copy to prevent C++ heap corruption during OpenCV operations
            face = np.ascontiguousarray(img_rgb[y1:y2, x1:x2].copy())
            if face.shape[0] > 0 and face.shape[1] > 0:
                # Automatically align tilted / bent angle faces using eye landmarks
                try:
                    kps = detection.location_data.relative_keypoints
                    if len(kps) >= 2:
                        re_x, re_y = kps[0].x * w, kps[0].y * h
                        le_x, le_y = kps[1].x * w, kps[1].y * h
                        dx = le_x - re_x
                        dy = le_y - re_y
                        if abs(dx) > 1e-3:
                            angle = float(np.degrees(np.arctan2(dy, dx)))
                            if abs(angle) > 6.0:
                                fh, fw = face.shape[:2]
                                fc = (fw // 2, fh // 2)
                                M = cv2.getRotationMatrix2D(fc, angle, 1.0)
                                face = cv2.warpAffine(face, M, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                                face = np.ascontiguousarray(face)
                except Exception:
                    pass
                faces.append((face, (x1, y1, x2 - x1, y2 - y1)))
    return faces

def recognize_face(face_embedding, threshold=KNOWN_FACE_THRESHOLD):
    norm = np.linalg.norm(face_embedding)
    if norm == 0:
        return "Unknown"
    face_embedding = face_embedding / norm
    
    min_dist = float("inf")
    name = "Unknown"

    for person, norm_embs in normalized_known_faces.items():
        if len(norm_embs) == 0:
            continue
        # Vectorized distance computation
        diffs = norm_embs - face_embedding
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold and dists[min_idx] < min_dist:
            min_dist = dists[min_idx]
            name = person

    return name

alerts = []
last_detection_time = {}
fight_alerts = []
loitering_alerts = []

vehicle_alerts = []
last_vehicle_detection_time = {}

# Unknown person dwell tracking for auto-save after 3 seconds
unknown_first_seen = {}      # {global_id: first_seen_timestamp}
unknown_saved_to_db = set()  # set of global_ids already stored to DB

def extract_faces_from_video(name, video_path, num_images=500):
    if not os.path.exists(video_path):
        return "Error: Video file does not exist."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_images)

    output_dir = os.path.join(DATASET_FOLDER, name)
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_images = 0
    new_embeddings = []

    local_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    while saved_images < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        frame_count += 1
        faces = extract_face(frame, detector=local_detector)

        for face, _ in faces:
            face_resized = cv2.resize(face, (160, 160))
            embedding = get_face_embedding(face_resized)
            new_embeddings.append(embedding)

            image_path = os.path.join(output_dir, f"face_{saved_images}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            saved_images += 1

    cap.release()

    # Save embeddings to embeddings.pkl
    if new_embeddings:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}

        if name in known_faces:
            known_faces[name].extend(new_embeddings)
        else:
            known_faces[name] = new_embeddings

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        return f"Extracted {len(new_embeddings)} face embeddings for {name} and saved in {EMBEDDINGS_FILE}."
    
    return "No faces detected in the video."

def make_alert_call(name, timestamp):
    """Make a phone call alert when a suspect is detected"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create a TwiML response with text-to-speech
        twiml = f"""
        <Response>
            <Say>Alert! Suspect {name} has been detected at {timestamp}. Please check your email for more details.</Say>
            <Pause length="1"/>
            <Say>Repeating: Suspect {name} has been detected.</Say>
        </Response>
        """
        
        # Make the call
        call = client.calls.create(
            twiml=twiml,
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        
        print(f"Phone alert initiated for suspect: {name}, Call SID: {call.sid}")
        return True
    except Exception as e:
        print(f"Error making phone call: {e}")
        return False

def send_email_alert(name, timestamp, face_path):
    subject = f"Suspect Detected  : {name}"
    body = f"A suspect has been detected!!!!.\n\nName: {name}\nTime: {timestamp}"

    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach the detected face image
    try:
        with open(face_path, 'rb') as img:
            img_data = img.read()
            img_name = os.path.basename(face_path)
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)
    except Exception as e:
        print(f"Error attaching image: {e}")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            print(f"Email alert sent for suspect: {name}")
    except Exception as e:
        print(f"Email sending failed: {e}")

# Update in generate_frames() to send email alerts
def process_alerts():
    """Thread to process suspect alerts asynchronously.
    Supports both 3-tuple (legacy) and 4-tuple (with camera_id) tasks.
    """
    while True:
        task = task_queue.get()
        if task is None:
            break  # Stop the thread when None is added to the queue

        # Support both old (3-tuple) and new (4-tuple) formats
        if len(task) == 4:
            name, timestamp, face_path, camera_id = task
        else:
            name, timestamp, face_path = task
            camera_id = "Camera 0"

        camera_id = format_camera_id(camera_id)

        try:
            # Save suspect in MongoDB / local fallback
            with open(face_path, "rb") as img_file:
                image_data = img_file.read()

            suspect_data = {
                "suspect_name": name,
                "detected_image": image_data,
                "time": timestamp,
                "camera_id": camera_id
            }
            collection.insert_one(suspect_data)
            print(f"[DB] Saved '{name}' from {camera_id} at {timestamp}")

            # Only send email/call for known suspects (not Unknown-* persons)
            if not name.startswith("Unknown-"):
                send_email_alert(name, timestamp, face_path)
                make_alert_call(name, timestamp)

        except Exception as e:
            print(f"Error processing alert: {e}")

        task_queue.task_done()

# Start the background thread
alert_thread = threading.Thread(target=process_alerts, daemon=True)
alert_thread.start()

def process_vehicles():
    """Thread to process vehicle tracking asynchronously"""
    while True:
        task = vehicle_queue.get()
        if task is None:
            break
        
        number_plate, timestamp, vehicle_img_path = task
        try:
            with open(vehicle_img_path, "rb") as img_file:
                image_data = img_file.read()

            vehicle_data = {
                "number_plate": number_plate,
                "detected_image": image_data,
                "time": timestamp
            }
            vehicles_collection.insert_one(vehicle_data)
            print(f"Saved vehicle {number_plate} to DB.")
        except Exception as e:
            print(f"Error processing vehicle: {e}")
            
        vehicle_queue.task_done()

vehicle_thread = threading.Thread(target=process_vehicles, daemon=True)
vehicle_thread.start()

# ── Global Unknown ID Allocation (Multi-Embedding Cross-Camera Gallery) ──
global_unknown_counter = 0
global_unknown_gallery = {}   # {label: [norm_emb1, norm_emb2, ...]}
MAX_GALLERY_PER_PERSON = 20   # Max embeddings stored per unknown person
MAX_UNKNOWN_PERSONS = 250     # Max unknown persons tracked
UNKNOWN_MATCH_THRESHOLD = 0.50  # Cosine similarity threshold for cross-camera Re-ID

def get_global_unknown_id(face_embedding):
    """Assign a stable global ID to an unknown face across all cameras.
    
    Uses a multi-embedding gallery for robust cross-camera Re-ID:
    - Matches query against all representative embeddings per person
    - Uses cosine similarity threshold (0.50) tuned for cross-camera domain shifts
    - Adds new distinct angles (< 0.95 cos-sim to existing) to enrich gallery
    """
    global global_unknown_counter, global_unknown_gallery
    
    norm = np.linalg.norm(face_embedding)
    if norm == 0:
        return "Unknown"
    norm_embedding = face_embedding / norm
    
    best_label = None
    best_sim = -1.0
    
    for label, gallery in global_unknown_gallery.items():
        if not gallery:
            continue
        gallery_matrix = np.array(gallery)
        sims = np.dot(gallery_matrix, norm_embedding)
        max_sim = float(np.max(sims))
        
        if max_sim > best_sim:
            best_sim = max_sim
            best_label = label
    
    if best_label is not None and best_sim >= UNKNOWN_MATCH_THRESHOLD:
        # Add new embedding only if it adds novelty (< 0.95 similarity) to prevent gallery pollution
        if len(global_unknown_gallery[best_label]) < MAX_GALLERY_PER_PERSON:
            gallery_matrix = np.array(global_unknown_gallery[best_label])
            existing_sims = np.dot(gallery_matrix, norm_embedding)
            if np.max(existing_sims) < 0.95:
                global_unknown_gallery[best_label].append(norm_embedding)
        return best_label
    
    # No match found — create new global ID
    global_unknown_counter += 1
    new_label = f"Unknown-{global_unknown_counter}"
    global_unknown_gallery[new_label] = [norm_embedding]
    
    if len(global_unknown_gallery) > MAX_UNKNOWN_PERSONS:
        oldest_key = next(iter(global_unknown_gallery))
        del global_unknown_gallery[oldest_key]
    
    return new_label


def format_camera_id(source):
    """Normalize any camera index or representation to a clear 'Camera X' format."""
    if source is None:
        return "Camera 0"
    s = str(source).strip()
    if s.lower() in ("0", "cam-0", "cam 0", "cam0", "camera 0", "camera-0"):
        return "Camera 0"
    if s.lower() in ("1", "cam-1", "cam 1", "cam1", "camera 1", "camera-1"):
        return "Camera 1"
    if s.lower() in ("2", "cam-2", "cam 2", "cam2", "camera 2", "camera-2"):
        return "Camera 2"
    if s.isdigit():
        return f"Camera {s}"
    return f"Camera {s}"


class ThreadedCamera:
    """Thread-safe, non-blocking camera frame capture engine.
    Continuously acquires frames from hardware in a background daemon thread.
    read() returns immediately (0ms) with the newest available frame, eliminating
    USB buffer lag, frame latency, and multi-camera thread contention.

    Key improvements:
    - Uses MJPG codec to reduce USB bandwidth (critical for dual-camera setups)
    - Stale frame detection: returns False if the last captured frame is older than 2s
    """
    STALE_THRESHOLD = 2.0  # seconds before a frame is considered stale

    def __init__(self, source, width=640, height=480, max_retries=3):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_grab_time = 0
        self.open_success = False

        for attempt in range(max_retries):
            try:
                source_idx = int(source)
                # Try DirectShow with MJPG first (best for Windows USB cameras)
                cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # Request MJPG compressed format to drastically reduce USB bandwidth
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap = cap
                    break
                cap.release()
                time.sleep(0.15)
                # Fallback: default backend
                cap = cv2.VideoCapture(source_idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.cap = cap
                    break
                cap.release()
            except (ValueError, TypeError):
                # String source (RTSP, file path, etc.)
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    self.cap = cap
                    break
                cap.release()
            time.sleep(0.15)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Warm-up read
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame
                self.last_grab_time = time.time()

            self.open_success = True
            self.running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
        else:
            self.open_success = False

    def is_opened(self):
        return self.open_success and self.cap is not None and self.cap.isOpened()

    def _capture_worker(self):
        """Continuously grab frames from hardware in a dedicated thread."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.last_grab_time = time.time()
            else:
                time.sleep(0.01)
            time.sleep(0.004)

    def read(self):
        """Instant non-blocking frame retrieval with stale frame guard."""
        with self.lock:
            if self.latest_frame is not None:
                # Guard: if the camera froze or was unplugged, report failure
                if self.last_grab_time > 0 and (time.time() - self.last_grab_time) > self.STALE_THRESHOLD:
                    return False, None
                return True, self.latest_frame.copy()
        return False, None

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.6)
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        self.open_success = False


def open_capture(source):
    """Safely open camera capture using DirectShow on Windows (legacy fallback helper)."""
    cap = None
    try:
        source_idx = int(source)
        cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source_idx)
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(source)

    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def _handle_face_on_camera(name, face_embedding, face, camera_id, frame_counter, display_name=None):
    """Shared helper for both single and dual camera modes.
    Handles known suspect immediate save and unknown person 3-second dwell auto-save.
    Keyed per (person, camera_id) so every camera detects and logs independently.
    Returns the display label.
    """
    global unknown_first_seen, unknown_saved_to_db, last_detection_time, alerts

    if name != "Unknown":
        display_name = name
    elif display_name is None:
        display_name = get_global_unknown_id(face_embedding)

    current_time = time.time()

    # ── Known suspect: store immediately (debounced per suspect per camera) ──
    detect_key = (name, camera_id)
    if name != "Unknown" and (detect_key not in last_detection_time or current_time - last_detection_time[detect_key] >= 10):
        last_detection_time[detect_key] = current_time

        person_folder = os.path.join(DETECTED_FACES_FOLDER, name)
        os.makedirs(person_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        face_count = len(os.listdir(person_folder))
        face_path = os.path.join(person_folder, f"img_{face_count + 1}.jpg")
        cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        time_data_path = os.path.join(TIME_DATA_FOLDER, f"{name}.txt")
        with open(time_data_path, "a") as tf:
            tf.write(f"{timestamp}\n")

        alerts.append({"name": name, "time": timestamp, "camera_id": camera_id})
        task_queue.put((name, timestamp, face_path, camera_id))
        print(f"[Detect] Known suspect '{name}' detected on {camera_id} at {timestamp} -> queued for DB save")

    # ── Unknown person: track dwell time per camera, auto-save after 3 seconds ──
    if name == "Unknown":
        global_id = display_name  # e.g. "Unknown-5"
        unknown_key = (global_id, camera_id)

        # Record first-seen time on this specific camera
        if unknown_key not in unknown_first_seen:
            unknown_first_seen[unknown_key] = current_time

        dwell = current_time - unknown_first_seen[unknown_key]

        # Auto-save after 3 seconds of visibility on this camera
        if dwell >= 3.0 and unknown_key not in unknown_saved_to_db:
            unknown_saved_to_db.add(unknown_key)

            person_folder = os.path.join(DETECTED_FACES_FOLDER, global_id)
            os.makedirs(person_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
            face_count = len(os.listdir(person_folder))
            face_path = os.path.join(person_folder, f"img_{face_count + 1}.jpg")
            cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

            print(f"[AutoSave] Unknown '{global_id}' visible for {dwell:.1f}s on {camera_id} -> saving to DB")
            task_queue.put((global_id, timestamp, face_path, camera_id))

    return display_name


def generate_frames(camera_source=0):
    global alerts, last_detection_time, known_faces, last_vehicle_detection_time, current_head_count
    global unknown_first_seen, unknown_saved_to_db, active_stream_token

    with stream_token_lock:
        active_stream_token += 1
        my_token = active_stream_token

    camera_id = format_camera_id(camera_source)
    cam = acquire_camera(camera_source)
    local_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    if not cam.is_opened():
        print(f"[SingleCam] Camera ({camera_source}) failed to open")
        local_detector.close()
        release_camera(camera_source)
        return

    frame_counter = 0
    last_weapons = []
    last_vehicles = []
    last_threat_level = "NORMAL"
    last_threat_conf = 0.0
    
    # Loitering variables
    loitering_trajectories = {}
    loitering_ids = set()
    last_loiter_boxes = []

    # Face tracking cache for high-FPS face recognition without CPU bottleneck
    face_cache = {}
    next_track_id = 0
    consecutive_failures = 0  # Track consecutive read failures

    try:
        while True:
            # Check if a newer stream has taken over
            if my_token != active_stream_token:
                print(f"[SingleCam] Stream {my_token} superseded by {active_stream_token}, exiting cleanly.")
                break

            success, frame = cam.read()
            if not success or frame is None:
                consecutive_failures += 1
                # Only sleep briefly; never auto-stop - the user will stop manually
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            frame_counter += 1

            # Cap frame resolution to 640px width for fast real-time processing
            h_orig, w_orig = frame.shape[:2]
            if w_orig > 640:
                frame = cv2.resize(frame, (640, int(h_orig * 640 / w_orig)), interpolation=cv2.INTER_LINEAR)

            try:
                # Reload known faces dynamically (throttled to every 100 frames)
                if frame_counter % 100 == 1 and os.path.exists(EMBEDDINGS_FILE):
                    with open(EMBEDDINGS_FILE, "rb") as f:
                        known_faces = pickle.load(f)
                    
                    global normalized_known_faces
                    normalized_known_faces = {}
                    for person, embeddings in known_faces.items():
                        if embeddings:
                            embs = np.array(embeddings)
                            norms = np.linalg.norm(embs, axis=1, keepdims=True)
                            norms[norms == 0] = 1
                            normalized_known_faces[person] = embs / norms

                # Extract faces using MediaPipe with eye-alignment for bent angles
                faces = extract_face(frame, detector=local_detector)
                
                # Head counting using YuNet (run every 8 frames for performance)
                if frame_counter % 8 == 0:
                    if face_detector_yunet is not None:
                        try:
                            yunet_input = cv2.resize(frame, (320, 320))
                            _, faces_yunet = face_detector_yunet.detect(yunet_input)
                            current_head_count = 0 if faces_yunet is None else faces_yunet.shape[0]
                        except Exception as e:
                            current_head_count = len(faces)
                    else:
                        current_head_count = len(faces)
                elif current_head_count == 0 and len(faces) > 0:
                    current_head_count = len(faces)
                    
                cv2.putText(frame, f"Live Head Count: {current_head_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, camera_id, (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)

                # High-performance face recognition: track faces across frames and cache embeddings
                current_face_tracks = []
                for face, (x, y, width, height) in faces:
                    cx, cy = x + width // 2, y + height // 2
                    
                    best_match_id = None
                    best_dist = 60
                    for tid, tinfo in face_cache.items():
                        tcx, tcy = tinfo['center']
                        dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_match_id = tid

                    # Fast spatial tracking reuse for high FPS (refresh embedding every 25 frames)
                    if best_match_id is not None and (frame_counter - face_cache[best_match_id]['embed_frame'] < 25):
                        tinfo = face_cache[best_match_id]
                        name = tinfo['name']
                        display_name = tinfo['display_name']
                        color = tinfo['color']
                        face_embedding = tinfo['embedding']
                        _handle_face_on_camera(name, face_embedding, face, camera_id, frame_counter, display_name=display_name)
                        tinfo['center'] = (cx, cy)
                        tinfo['last_seen'] = frame_counter
                        current_face_tracks.append((x, y, width, height, display_name, color))
                    else:
                        face_resized = cv2.resize(face, (160, 160))
                        face_embedding = get_face_embedding(face_resized)
                        name = recognize_face(face_embedding)
                        if name != "Unknown":
                            display_name = name
                            color = (0, 255, 0)
                            _handle_face_on_camera(name, face_embedding, face, camera_id, frame_counter, display_name=name)
                        else:
                            display_name = get_global_unknown_id(face_embedding)
                            color = (0, 200, 255)
                            _handle_face_on_camera(name, face_embedding, face, camera_id, frame_counter, display_name=display_name)

                        tid = best_match_id if best_match_id is not None else next_track_id
                        if best_match_id is None:
                            next_track_id += 1
                        face_cache[tid] = {
                            'center': (cx, cy),
                            'embedding': face_embedding,
                            'name': name,
                            'display_name': display_name,
                            'color': color,
                            'embed_frame': frame_counter,
                            'last_seen': frame_counter
                        }
                        current_face_tracks.append((x, y, width, height, display_name, color))

                # Clean stale face tracks
                stale_tids = [tid for tid, tinfo in face_cache.items() if frame_counter - tinfo['last_seen'] > 30]
                for tid in stale_tids:
                    del face_cache[tid]

                # Draw faces
                for (x, y, width, height, display_name, color) in current_face_tracks:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Vehicle Detection every 25 frames (~1 sec) with optimized imgsz=384
                if frame_counter % 25 == 0:
                    try:
                        results = vehicle_model(frame, device=device, verbose=False, conf=0.55, imgsz=384)
                        last_vehicles = []
                        for result in results:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                if cls in [2, 3, 5, 7]:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    h_v, w_v = frame.shape[:2]
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w_v, x2), min(h_v, y2)
                                    if x2 <= x1 or y2 <= y1:
                                        continue
                                    vehicle_crop = np.ascontiguousarray(frame[y1:y2, x1:x2].copy())
                                    if vehicle_crop.size > 0:
                                        plate_text = extract_plate_text(vehicle_crop)
                                        if plate_text and plate_text != "UNKNOWN":
                                            last_vehicles.append((x1, y1, x2, y2, plate_text))
                                            current_time = time.time()
                                            if plate_text not in last_vehicle_detection_time or (current_time - last_vehicle_detection_time[plate_text] >= 15):
                                                last_vehicle_detection_time[plate_text] = current_time
                                                timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                                                vehicle_filename = f"{plate_text}_{timestamp.replace(' ', '_').replace('-', '')}.jpg"
                                                vehicle_path = os.path.join(DETECTED_VEHICLES_FOLDER, vehicle_filename)
                                                save_img = frame.copy()
                                                cv2.rectangle(save_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                                cv2.putText(save_img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                                cv2.imwrite(vehicle_path, save_img)
                                                vehicle_queue.put((plate_text, timestamp, vehicle_path))
                    except Exception as e_veh:
                        print(f"Vehicle detection error: {e_veh}")

                # Draw vehicles on every frame to prevent flickering
                for (vx1, vy1, vx2, vy2, vplate) in last_vehicles:
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.putText(frame, vplate, (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Weapon Detection every 10 frames with optimized imgsz=384
                if frame_counter % 10 == 0:
                    try:
                        results = weapon_model(frame, device=device, verbose=False, conf=0.45, iou=0.45, augment=False, imgsz=384)
                        last_weapons = []
                        for result in results:
                            for box in result.boxes:
                                conf = float(box.conf[0])
                                if conf > 0.25:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cls_name = weapon_model.names[int(box.cls[0])]
                                    last_weapons.append((x1, y1, x2, y2, cls_name, conf))
                    except Exception as e_wep:
                        print(f"Weapon detection error: {e_wep}")
                
                # Draw weapons on every frame to prevent flickering
                for (x1, y1, x2, y2, cls_name, conf) in last_weapons:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{cls_name.upper()} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Fight Detection: bypass completely when fewer than 2 people are detected
                if current_head_count >= 2:
                    if frame_counter % 3 == 0:
                        try:
                            _, threat_level, conf = fight_pipeline.process_frame(frame, imgsz=384)
                            last_threat_level = threat_level
                            last_threat_conf = conf
                            if threat_level in ["SUSPICIOUS", "PHYSICAL ALTERCATION", "CRITICAL"]:
                                timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                                alert_img_path = os.path.join(BASE_PATH, "static", "alerts", "fight_alert.jpg")
                                os.makedirs(os.path.dirname(alert_img_path), exist_ok=True)
                                cv2.imwrite(alert_img_path, frame)
                                image_url = f"/static/alerts/fight_alert.jpg?t={int(time.time())}"
                                fight_alerts.append({"type": threat_level, "time": timestamp, "confidence": float(conf), "image_url": image_url})
                        except Exception as e_fight:
                            print(f"Fight detection error: {e_fight}")
                else:
                    if fight_pipeline.buffer:
                        fight_pipeline.reset_state()
                    last_threat_level = "NORMAL"
                    last_threat_conf = 0.0

                # Loitering Detection (Track every 5 frames only if persons are present)
                if current_head_count > 0 and frame_counter % 5 == 0:
                    try:
                        l_results = loitering_detector.yolo.track(frame, persist=True, classes=[0], verbose=False, imgsz=384)
                        if l_results and l_results[0].boxes and l_results[0].boxes.id is not None:
                            boxes = l_results[0].boxes.xyxy.cpu().numpy()
                            ids = l_results[0].boxes.id.cpu().numpy().astype(int)
                            last_loiter_boxes = []
                            for box, person_id in zip(boxes, ids):
                                x1, y1, x2, y2 = map(int, box)
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                if person_id not in loitering_trajectories:
                                    loitering_trajectories[person_id] = []
                                loitering_trajectories[person_id].append((frame_counter, cx, cy))
                                
                                if len(loitering_trajectories[person_id]) > 150:
                                    loitering_trajectories[person_id].pop(0)
                                last_loiter_boxes.append((x1, y1, x2, y2, person_id))
                        else:
                            last_loiter_boxes = []

                        # Classify every 100 frames (~3.5 seconds)
                        if frame_counter % 100 == 0 and loitering_trajectories:
                            features = loitering_detector._extract_features(loitering_trajectories)
                            if features:
                                import pandas as pd
                                df = pd.DataFrame(features)
                                X = df[loitering_detector.feature_columns]
                                preds = loitering_detector.classifier.predict(X)
                                df["is_loitering"] = preds
                                loitering_ids = set(df[df["is_loitering"] == 1]["id"])
                                if loitering_ids:
                                    timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                                    loitering_alerts.append({"count": len(loitering_ids), "time": timestamp})
                    except Exception as e_loiter:
                        print(f"Loitering detection error: {e_loiter}")
                elif current_head_count == 0:
                    last_loiter_boxes = []
                    loitering_ids = set()

                # Draw loitering bounding boxes on every frame
                for (lx1, ly1, lx2, ly2, lpid) in last_loiter_boxes:
                    if lpid in loitering_ids:
                        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 165, 255), 2)
                        cv2.putText(frame, f"LOITERING {lpid}", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            except Exception as e_frame:
                print(f"[Frame processing error]: {e_frame}")

            # Fast JPEG encode with quality 72 for fluid streaming
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e_encode:
                print(f"[SingleCam] Encode error (non-fatal): {e_encode}")
                continue

    finally:
        local_detector.close()
        release_camera(camera_source)


def generate_dual_frames(cam_a=0, cam_b=1):
    """Dual-camera CCTV mode with asynchronous capture and cross-camera face Re-ID."""
    global alerts, last_detection_time, known_faces, last_vehicle_detection_time, current_head_count
    global unknown_first_seen, unknown_saved_to_db, active_stream_token

    with stream_token_lock:
        active_stream_token += 1
        my_token = active_stream_token

    camera_id_a = format_camera_id(cam_a)
    camera_id_b = format_camera_id(cam_b)

    cam_obj_a = acquire_camera(cam_a, max_retries=3)
    cam_obj_b = acquire_camera(cam_b, max_retries=3)

    if not cam_obj_a.is_opened() and not cam_obj_b.is_opened():
        print(f"[DualCam] Both cameras ({cam_a}, {cam_b}) failed to open")
        release_camera(cam_a)
        release_camera(cam_b)
        return

    if not cam_obj_a.is_opened():
        print(f"[DualCam] Camera A ({cam_a}) failed to open – falling back to single cam B ({cam_b})")
        release_camera(cam_a)
        release_camera(cam_b)
        yield from generate_frames(cam_b)
        return

    if not cam_obj_b.is_opened():
        print(f"[DualCam] Camera B ({cam_b}) failed to open – falling back to single cam A ({cam_a})")
        release_camera(cam_a)
        release_camera(cam_b)
        yield from generate_frames(cam_a)
        return

    print(f"[DualCam] Both cameras opened successfully: A={camera_id_a}, B={camera_id_b}")

    frame_counter = 0
    last_weapons_a = []
    last_weapons_b = []
    face_cache_a = {}
    face_cache_b = {}
    next_id_a = 0
    next_id_b = 0
    consecutive_failures = 0

    local_detector_a = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    local_detector_b = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    try:
        while True:
            if my_token != active_stream_token:
                print(f"[DualCam] Stream {my_token} superseded, exiting cleanly.")
                break

            ok_a, frame_a = cam_obj_a.read()
            ok_b, frame_b = cam_obj_b.read()

            if not ok_a and not ok_b:
                consecutive_failures += 1
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            frame_counter += 1

            if frame_counter % 100 == 1 and os.path.exists(EMBEDDINGS_FILE):
                try:
                    with open(EMBEDDINGS_FILE, "rb") as f:
                        known_faces = pickle.load(f)
                    global normalized_known_faces
                    normalized_known_faces = {}
                    for person, embeddings_list in known_faces.items():
                        if embeddings_list:
                            embs = np.array(embeddings_list)
                            norms = np.linalg.norm(embs, axis=1, keepdims=True)
                            norms[norms == 0] = 1
                            normalized_known_faces[person] = embs / norms
                except Exception as e:
                    print(f"Error reloading embeddings: {e}")

            active_identities_a = set()
            active_identities_b = set()
            head_count_a = 0
            head_count_b = 0

            # ── Process Camera A ──
            if ok_a and frame_a is not None:
                frame_a = cv2.flip(frame_a, 1)
                ha, wa = frame_a.shape[:2]
                if wa > 480:
                    frame_a = cv2.resize(frame_a, (480, int(ha * 480 / wa)), interpolation=cv2.INTER_LINEAR)

                faces_a = extract_face(frame_a, detector=local_detector_a)

                # YuNet head counting on Cam A (every 8 frames)
                if frame_counter % 8 == 0 and face_detector_yunet is not None:
                    try:
                        yunet_input = cv2.resize(frame_a, (320, 320))
                        _, faces_yunet = face_detector_yunet.detect(yunet_input)
                        head_count_a = 0 if faces_yunet is None else faces_yunet.shape[0]
                    except Exception:
                        head_count_a = len(faces_a)
                else:
                    head_count_a = len(faces_a)

                for face, (x, y, w, h) in faces_a:
                    cx, cy = x + w // 2, y + h // 2
                    best_tid = None
                    best_d = 60
                    for tid, tinfo in face_cache_a.items():
                        tcx, tcy = tinfo['center']
                        d = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                        if d < best_d:
                            best_d = d
                            best_tid = tid

                    # Fast spatial tracking reuse for high FPS (refresh embedding every 25 frames)
                    if best_tid is not None and (frame_counter - face_cache_a[best_tid]['embed_frame'] < 25):
                        tinfo = face_cache_a[best_tid]
                        display_name = tinfo['display_name']
                        color = tinfo['color']
                        emb = tinfo['embedding']
                        name = tinfo['name']
                        tinfo['center'] = (cx, cy)
                        tinfo['last_seen'] = frame_counter
                        _handle_face_on_camera(name, emb, face, camera_id_a, frame_counter, display_name=display_name)
                    else:
                        face_resized = cv2.resize(face, (160, 160))
                        emb = get_face_embedding(face_resized)
                        name = recognize_face(emb)
                        if name != "Unknown":
                            display_name = name
                            color = (0, 255, 0)
                            _handle_face_on_camera(name, emb, face, camera_id_a, frame_counter, display_name=name)
                        else:
                            display_name = get_global_unknown_id(emb)
                            color = (0, 200, 255)
                            _handle_face_on_camera(name, emb, face, camera_id_a, frame_counter, display_name=display_name)

                        tid = best_tid if best_tid is not None else next_id_a
                        if best_tid is None:
                            next_id_a += 1
                        face_cache_a[tid] = {
                            'center': (cx, cy),
                            'embedding': emb,
                            'name': name,
                            'display_name': display_name,
                            'color': color,
                            'embed_frame': frame_counter,
                            'last_seen': frame_counter
                        }

                    active_identities_a.add(display_name)
                    cv2.rectangle(frame_a, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame_a, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Clean stale tracks A
                stale_a = [tid for tid, tinfo in face_cache_a.items() if frame_counter - tinfo['last_seen'] > 30]
                for tid in stale_a:
                    del face_cache_a[tid]

                # Interleaved weapon detection (on frame % 12 == 0)
                if frame_counter % 12 == 0:
                    try:
                        w_res = weapon_model(frame_a, device=device, verbose=False, conf=0.45, iou=0.45, augment=False, imgsz=384)
                        last_weapons_a = []
                        for result in w_res:
                            for box in result.boxes:
                                c = float(box.conf[0])
                                if c > 0.25:
                                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                                    cn = weapon_model.names[int(box.cls[0])]
                                    last_weapons_a.append((bx1, by1, bx2, by2, cn, c))
                    except Exception:
                        pass

                for (bx1, by1, bx2, by2, cn, c) in last_weapons_a:
                    cv2.rectangle(frame_a, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(frame_a, f"{cn.upper()} {c:.2f}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame_a, f"CAM A ({camera_id_a})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)

            # ── Process Camera B ──
            if ok_b and frame_b is not None:
                frame_b = cv2.flip(frame_b, 1)
                hb, wb = frame_b.shape[:2]
                if wb > 480:
                    frame_b = cv2.resize(frame_b, (480, int(hb * 480 / wb)), interpolation=cv2.INTER_LINEAR)

                faces_b = extract_face(frame_b, detector=local_detector_b)

                # YuNet head counting on Cam B (every 8 frames)
                if frame_counter % 8 == 0 and face_detector_yunet is not None:
                    try:
                        yunet_input = cv2.resize(frame_b, (320, 320))
                        _, faces_yunet = face_detector_yunet.detect(yunet_input)
                        head_count_b = 0 if faces_yunet is None else faces_yunet.shape[0]
                    except Exception:
                        head_count_b = len(faces_b)
                else:
                    head_count_b = len(faces_b)

                for face, (x, y, w, h) in faces_b:
                    cx, cy = x + w // 2, y + h // 2
                    best_tid = None
                    best_d = 60
                    for tid, tinfo in face_cache_b.items():
                        tcx, tcy = tinfo['center']
                        d = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                        if d < best_d:
                            best_d = d
                            best_tid = tid

                    # Fast spatial tracking reuse for high FPS (refresh embedding every 25 frames)
                    if best_tid is not None and (frame_counter - face_cache_b[best_tid]['embed_frame'] < 25):
                        tinfo = face_cache_b[best_tid]
                        display_name = tinfo['display_name']
                        color = tinfo['color']
                        emb = tinfo['embedding']
                        name = tinfo['name']
                        tinfo['center'] = (cx, cy)
                        tinfo['last_seen'] = frame_counter
                        _handle_face_on_camera(name, emb, face, camera_id_b, frame_counter, display_name=display_name)
                    else:
                        face_resized = cv2.resize(face, (160, 160))
                        emb = get_face_embedding(face_resized)
                        name = recognize_face(emb)
                        if name != "Unknown":
                            display_name = name
                            color = (0, 255, 0)
                            _handle_face_on_camera(name, emb, face, camera_id_b, frame_counter, display_name=name)
                        else:
                            display_name = get_global_unknown_id(emb)
                            color = (0, 200, 255)
                            _handle_face_on_camera(name, emb, face, camera_id_b, frame_counter, display_name=display_name)

                        tid = best_tid if best_tid is not None else next_id_b
                        if best_tid is None:
                            next_id_b += 1
                        face_cache_b[tid] = {
                            'center': (cx, cy),
                            'embedding': emb,
                            'name': name,
                            'display_name': display_name,
                            'color': color,
                            'embed_frame': frame_counter,
                            'last_seen': frame_counter
                        }

                    active_identities_b.add(display_name)
                    cv2.rectangle(frame_b, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame_b, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Clean stale tracks B
                stale_b = [tid for tid, tinfo in face_cache_b.items() if frame_counter - tinfo['last_seen'] > 30]
                for tid in stale_b:
                    del face_cache_b[tid]

                # Interleaved weapon detection (on frame % 12 == 6)
                if frame_counter % 12 == 6:
                    try:
                        w_res = weapon_model(frame_b, device=device, verbose=False, conf=0.45, iou=0.45, augment=False, imgsz=384)
                        last_weapons_b = []
                        for result in w_res:
                            for box in result.boxes:
                                c = float(box.conf[0])
                                if c > 0.25:
                                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                                    cn = weapon_model.names[int(box.cls[0])]
                                    last_weapons_b.append((bx1, by1, bx2, by2, cn, c))
                    except Exception:
                        pass

                for (bx1, by1, bx2, by2, cn, c) in last_weapons_b:
                    cv2.rectangle(frame_b, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(frame_b, f"{cn.upper()} {c:.2f}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame_b, f"CAM B ({camera_id_b})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)

            # ── Deduplicated Cross-Camera Unique Head Count ──
            unique_active_persons = set(active_identities_a) | set(active_identities_b)
            yunet_extra_a = max(0, head_count_a - len(active_identities_a))
            yunet_extra_b = max(0, head_count_b - len(active_identities_b))
            current_head_count = max(len(unique_active_persons) + yunet_extra_a + yunet_extra_b, 0)

            # Combine frames side by side with uniform target height
            if ok_a and ok_b and frame_a is not None and frame_b is not None:
                h_a, w_a = frame_a.shape[:2]
                h_b, w_b = frame_b.shape[:2]
                target_h = 360
                if h_a != target_h:
                    scale_a = target_h / h_a
                    frame_a = cv2.resize(frame_a, (int(w_a * scale_a), target_h), interpolation=cv2.INTER_LINEAR)
                if h_b != target_h:
                    scale_b = target_h / h_b
                    frame_b = cv2.resize(frame_b, (int(w_b * scale_b), target_h), interpolation=cv2.INTER_LINEAR)
                combined = np.hstack([frame_a, frame_b])
            elif ok_a and frame_a is not None:
                combined = frame_a
            elif ok_b and frame_b is not None:
                combined = frame_b
            else:
                time.sleep(0.01)
                continue

            # Cyberpunk HUD Header with Cross-Camera Head Count
            count_text = f"CROSS-CAMERA UNIQUE HEAD COUNT: {current_head_count}"
            (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cx = combined.shape[1] // 2 - tw // 2
            cv2.rectangle(combined, (cx - 15, 6), (cx + tw + 15, th + 20), (10, 10, 10), -1)
            cv2.rectangle(combined, (cx - 15, 6), (cx + tw + 15, th + 20), (0, 210, 255), 1)
            cv2.putText(combined, count_text, (cx, th + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            try:
                ret, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e_encode:
                print(f"[DualCam] Encode error (non-fatal): {e_encode}")
                continue

    finally:
        local_detector_a.close()
        local_detector_b.close()
        release_camera(cam_a)
        release_camera(cam_b)

@app.route('/get_alerts')
def get_alerts():
    global alerts
    return jsonify(alerts)

@app.route('/get_fight_alerts')
def get_fight_alerts():
    global fight_alerts
    return jsonify(fight_alerts)

@app.route('/get_loitering_alerts')
def get_loitering_alerts():
    global loitering_alerts
    return jsonify(loitering_alerts)

@app.route('/get_headcount')
def get_headcount():
    global current_head_count
    return jsonify({'headcount': current_head_count})

@app.route('/suspects')
@app.route('/database')
@app.route('/Database')
@app.route('/Suspects')
def get_suspects():
    try:
        suspects = list(collection.find({}))
    except Exception as e:
        print(f"Error loading suspects from collection: {e}")
        fallback = LocalFallbackCollection(os.path.join(BASE_PATH, "suspects_db.pkl"))
        suspects = fallback.find({})

    formatted_suspects = []
    for suspect in suspects:
        try:
            item = dict(suspect)
            item['_id'] = str(item.get('_id', ''))
            item['camera_id'] = format_camera_id(item.get('camera_id', 'Camera 0'))
            img = item.get('detected_image')
            if img:
                if isinstance(img, (bytes, bytearray)):
                    item['image_b64'] = base64.b64encode(img).decode('utf-8')
                elif isinstance(img, str):
                    item['image_b64'] = img
            formatted_suspects.append(item)
        except Exception as err:
            print(f"Error formatting suspect document: {err}")
            formatted_suspects.append(suspect)

    return render_template('suspects.html', suspects=formatted_suspects)

@app.route('/clear_suspects', methods=['POST', 'GET'])
def clear_suspects():
    try:
        collection.delete_many({})
        global alerts, last_detection_time, unknown_first_seen, unknown_saved_to_db, global_unknown_gallery, global_unknown_counter
        alerts = []
        last_detection_time.clear()
        if 'recent_alerts' in globals():
            recent_alerts.clear()
        unknown_first_seen.clear()
        unknown_saved_to_db.clear()
        global_unknown_gallery.clear()
        global_unknown_counter = 0
        return jsonify({'status': 'success', 'message': 'Suspects database cleared successfully'})
    except Exception as e:
        print(f"Error clearing suspects: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear_vehicles', methods=['POST', 'GET'])
def clear_vehicles():
    try:
        vehicles_collection.delete_many({})
        return jsonify({'status': 'success', 'message': 'Vehicles database cleared successfully'})
    except Exception as e:
        print(f"Error clearing vehicles: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/vehicles')
@app.route('/Vehicles')
@app.route('/vehicle')
@app.route('/Vehicle')
def get_vehicles():
    try:
        vehicles = list(vehicles_collection.find({}))
    except Exception as e:
        print(f"Error loading vehicles from collection: {e}")
        fallback = LocalFallbackCollection(os.path.join(BASE_PATH, "vehicles_db.pkl"))
        vehicles = fallback.find({})

    formatted_vehicles = []
    for vehicle in vehicles:
        try:
            item = dict(vehicle)
            item['_id'] = str(item.get('_id', ''))
            img = item.get('detected_image')
            if img:
                if isinstance(img, (bytes, bytearray)):
                    item['image_b64'] = base64.b64encode(img).decode('utf-8')
                elif isinstance(img, str):
                    item['image_b64'] = img
            formatted_vehicles.append(item)
        except Exception as err:
            print(f"Error formatting vehicle document: {err}")
            formatted_vehicles.append(vehicle)

    return render_template('vehicles.html', vehicles=formatted_vehicles)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/live-mon')
def live_mon():
    global alerts, fight_alerts, loitering_alerts
    alerts = []  # Clear previous alerts
    fight_alerts = []
    loitering_alerts = []
    return render_template("livemon.html")

@app.route('/stop_feed')
def stop_feed():
    """Signal all running streams to exit and release cameras cleanly."""
    global active_stream_token
    with stream_token_lock:
        active_stream_token += 1
    # Allow running stream loop to detect superseded token and release hardware
    time.sleep(0.15)
    # Flush the global camera pool to physically release all USB hardware
    flush_camera_pool()
    return jsonify({'status': 'ok', 'token': active_stream_token})

@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'single')
    src = request.args.get('src', '0')
    try:
        src = int(src)
    except ValueError:
        pass  # Keep as string for paths/RTSP
    if mode == 'dual':
        cam_b = request.args.get('cam_b', '1')
        try:
            cam_b = int(cam_b)
        except ValueError:
            pass
        return Response(generate_dual_frames(src, cam_b), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(src), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/new-crim', methods=['GET', 'POST'])
def newcrim():
    if request.method == 'POST':
        name = request.form.get("name")
        file = request.files["video"]

        if file and name:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = extract_faces_from_video(name, filepath)
            flash(result)
            return redirect(url_for('newcrim'))

    return render_template('newcrim.html')

def analyze_video_file(video_path):
    """Analyze a video file with a high-speed, unified single-pass pipeline for registered faces,
    unknown faces, vehicle plates (ANPR), weapons, loitering, and fight detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25.0

    # Optimal sampling step: ~8 FPS provides excellent temporal detail while significantly reducing GPU load
    step = max(1, int(round(fps / 8)))

    registered_candidates = {}   # name -> dict (unique registered persons)
    unknown_face_tracks = []     # list of tracked unknown faces
    next_face_track_id = 1
    vehicles = []
    weapons = []
    max_head_count = 0
    unique_person_ids = set()

    seen_plates = set()          # avoid duplicate plates / vehicle labels
    seen_weapons = set()         # avoid spamming identical weapon detections
    vehicle_plate_map = {}       # track_id -> recognized plate string
    vehicle_ocr_attempts = {}    # track_id -> number of OCR attempts made
    trajectories = {}            # pid -> [(frame_count, cx, cy)] for loitering

    # Initialize fight detection state
    fight_pipeline.reset_state()
    threat_spans = []
    fight_highest_conf = 0.0
    fight_threat_level = "NORMAL"
    fight_people_involved = 0

    local_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)

    def compute_face_quality(crop):
        """Compute face quality score based on image sharpness and size."""
        if crop is None or crop.size == 0:
            return 0.0
        ch, cw = crop.shape[:2]
        area = ch * cw
        if area < 400:
            return float(area)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(variance * np.sqrt(area))
        except Exception:
            return float(area)

    # Reload known faces
    current_known = {}
    normalized_current = {}
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            current_known = pickle.load(f)
            
        for person, embeddings in current_known.items():
            if embeddings:
                embs = np.array(embeddings)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_current[person] = embs / norms

    # Free any cached VRAM before starting video processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    frame_count = 0
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % step != 0:
                frame_count += 1
                continue

            current_frame_idx = frame_count
            frame_count += 1

            # Resize frame to max 720p (1280px) for high efficiency
            h, w = frame.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]

            # --- 1. Unified YOLO Tracking: Person (0) + Vehicles (2, 3, 5, 7) in ONE Pass ---
            person_boxes = []
            vehicle_boxes = []
            try:
                yolo_res = vehicle_model.track(
                    frame,
                    classes=[0, 2, 3, 5, 7],
                    persist=True,
                    conf=0.35,
                    imgsz=640,
                    device=device,
                    verbose=False,
                    half=(device == 'cuda')
                )
                if yolo_res and yolo_res[0].boxes is not None:
                    boxes = yolo_res[0].boxes
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    xyxys = boxes.xyxy.cpu().numpy()
                    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls_ids)
                    confs = boxes.conf.cpu().numpy()

                    for idx, cls_id in enumerate(cls_ids):
                        x1, y1, x2, y2 = map(int, xyxys[idx])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        tid = track_ids[idx]
                        conf = float(confs[idx])

                        if cls_id == 0:  # Person
                            if tid is not None:
                                unique_person_ids.add(int(tid))
                                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                                if tid not in trajectories:
                                    trajectories[tid] = []
                                trajectories[tid].append((current_frame_idx, cx, cy))
                            person_boxes.append((x1, y1, x2, y2, tid, conf))
                        else:  # Vehicle (2=car, 3=motorcycle, 5=bus, 7=truck)
                            cls_name = vehicle_model.names.get(cls_id, "vehicle").upper()
                            vehicle_boxes.append((x1, y1, x2, y2, tid, cls_name, conf))
            except Exception as e:
                pass

            # --- 2. YuNet Fast Head Counting ---
            faces_yunet = None
            if face_detector_yunet is not None:
                try:
                    yunet_input = cv2.resize(frame, (320, 320))
                    _, faces_yunet = face_detector_yunet.detect(yunet_input)
                    if faces_yunet is not None:
                        count = faces_yunet.shape[0]
                        if count > max_head_count:
                            max_head_count = count
                except Exception:
                    pass

            # --- 3. Face Detection & Recognition (Detects even brief glimpses, shows each unique face ONCE) ---
            try:
                faces = extract_face(frame, detector=local_detector)
                
                # Fallback to YuNet detection if MediaPipe had no detections in this frame
                if not faces and face_detector_yunet is not None and faces_yunet is not None and len(faces_yunet) > 0:
                    orig_h, orig_w = frame.shape[:2]
                    for y_box in faces_yunet:
                        yx = int(y_box[0] * orig_w / 320.0)
                        yy = int(y_box[1] * orig_h / 320.0)
                        yw = int(y_box[2] * orig_w / 320.0)
                        yh = int(y_box[3] * orig_h / 320.0)
                        yx1, yy1 = max(0, yx), max(0, yy)
                        yx2, yy2 = min(orig_w, yx + yw), min(orig_h, yy + yh)
                        if (yx2 - yx1) >= 15 and (yy2 - yy1) >= 15:
                            y_crop = frame[yy1:yy2, yx1:yx2]
                            if y_crop.size > 0:
                                faces.append((cv2.cvtColor(y_crop, cv2.COLOR_BGR2RGB), (yx1, yy1, yx2 - yx1, yy2 - yy1)))

                valid_faces = []
                face_images = []
                for face, bbox in faces:
                    if face.shape[0] < 15 or face.shape[1] < 15:
                        continue
                    face_resized = cv2.resize(face, (160, 160))
                    valid_faces.append((face, face_resized, bbox))
                    face_images.append(face_resized)
                    
                if face_images:
                    embeddings_batch = compute_face_embeddings(face_images)
                    
                    for (orig_face_rgb, face_resized, bbox), face_embedding in zip(valid_faces, embeddings_batch):
                        norm = np.linalg.norm(face_embedding)
                        if norm > 0:
                            face_embedding = face_embedding / norm
                        
                        min_dist = float('inf')
                        best_name = 'Unknown'
                        
                        for person, norm_embs in normalized_current.items():
                            if len(norm_embs) == 0:
                                continue
                            diffs = norm_embs - face_embedding
                            dists = np.linalg.norm(diffs, axis=1)
                            min_idx = np.argmin(dists)
                            if dists[min_idx] < 0.70 and dists[min_idx] < min_dist:
                                min_dist = dists[min_idx]
                                best_name = person

                        # Pick the sharpest, highest-quality crop
                        face_bgr = cv2.cvtColor(orig_face_rgb, cv2.COLOR_RGB2BGR) if len(orig_face_rgb.shape) == 3 else orig_face_rgb
                        display_face = face_bgr if (face_bgr.shape[0] >= 60 and face_bgr.shape[1] >= 60) else cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                        quality = compute_face_quality(display_face)

                        if best_name != 'Unknown':
                            # Registered face: Keep ONLY ONCE, always updating to the best-quality image
                            if best_name not in registered_candidates:
                                registered_candidates[best_name] = {
                                    'name': best_name,
                                    'face_bgr': display_face,
                                    'score': quality,
                                    'frame_number': current_frame_idx
                                }
                            elif quality > registered_candidates[best_name]['score']:
                                registered_candidates[best_name]['face_bgr'] = display_face
                                registered_candidates[best_name]['score'] = quality
                                registered_candidates[best_name]['frame_number'] = current_frame_idx
                        else:
                            # Unknown face: Track spatially across consecutive frames and by FaceNet embedding
                            fx, fy, fw, fh = bbox
                            fcx, fcy = fx + fw / 2.0, fy + fh / 2.0
                            best_track = None
                            best_track_dist = float('inf')

                            # 1. Spatial proximity for recent tracks (within 1.5 seconds)
                            for trk in unknown_face_tracks:
                                frames_ago = current_frame_idx - trk['last_frame']
                                if frames_ago <= int(fps * 1.5):
                                    tx, ty, tw, th = trk['last_bbox']
                                    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
                                    s_dist = np.sqrt((fcx - tcx)**2 + (fcy - tcy)**2)
                                    max_d = max(fw, fh, tw, th)
                                    if s_dist < max_d * 2.0:
                                        best_track = trk
                                        break

                            # 2. If no spatial match, check FaceNet embedding similarity (< 0.75)
                            if best_track is None:
                                for trk in unknown_face_tracks:
                                    c_dist = float(np.linalg.norm(trk['centroid'] - face_embedding))
                                    if c_dist < 0.75 and c_dist < best_track_dist:
                                        best_track_dist = c_dist
                                        best_track = trk

                            if best_track is not None:
                                best_track['last_bbox'] = bbox
                                best_track['last_frame'] = current_frame_idx
                                best_track['embeddings'].append(face_embedding)
                                mean_emb = np.mean(best_track['embeddings'], axis=0)
                                best_track['centroid'] = mean_emb / (np.linalg.norm(mean_emb) or 1.0)
                                if quality > best_track['best_score']:
                                    best_track['face_bgr'] = display_face
                                    best_track['best_score'] = quality
                                    best_track['frame_number'] = current_frame_idx
                            else:
                                unknown_face_tracks.append({
                                    'track_id': next_face_track_id,
                                    'last_bbox': bbox,
                                    'last_frame': current_frame_idx,
                                    'embeddings': [face_embedding],
                                    'centroid': face_embedding,
                                    'face_bgr': display_face,
                                    'best_score': quality,
                                    'frame_number': current_frame_idx
                                })
                                next_face_track_id += 1
            except Exception as e:
                pass

            # --- 4. Vehicle & ANPR Detection (Optimized with Plate Tracking & Caching) ---
            try:
                direct_plates = []
                if plate_detector is not None and vehicle_boxes:
                    p_res = plate_detector(frame, conf=0.25, imgsz=640, device=device, verbose=False, half=(device == 'cuda'))
                    if p_res and len(p_res[0].boxes) > 0:
                        for pbox in p_res[0].boxes:
                            px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
                            pconf = float(pbox.conf[0])
                            direct_plates.append((px1, py1, px2, py2, pconf))

                for (x1, y1, x2, y2, tid, cls_name, vconf) in vehicle_boxes:
                    v_width = x2 - x1
                    v_height = y2 - y1
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue

                    # If this vehicle already has a recognized plate, skip OCR
                    cached_plate = vehicle_plate_map.get(tid) if tid is not None else None
                    if cached_plate and cached_plate != "UNKNOWN":
                        continue

                    plate_text = "UNKNOWN"
                    matched_plate_box = None

                    # 1. Match with any plate detected directly on the frame
                    for (px1, py1, px2, py2, pconf) in direct_plates:
                        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                        if x1 <= pcx <= x2 and y1 <= pcy <= y2:
                            matched_plate_box = (px1, py1, px2, py2)
                            p_crop = frame[max(0, py1-5):min(h, py2+5), max(0, px1-5):min(w, px2+5)]
                            if p_crop.size > 0:
                                plate_text = ocr_plate_image(p_crop)
                                if plate_text != "UNKNOWN":
                                    break

                    # 2. If not matched, attempt OCR only on clear, decently-sized vehicle crops (up to 2 tries)
                    attempts = vehicle_ocr_attempts.get(tid, 0) if tid is not None else 0
                    if plate_text == "UNKNOWN" and attempts < 2 and v_width >= 90 and v_height >= 60:
                        if tid is not None:
                            vehicle_ocr_attempts[tid] = attempts + 1
                        plate_text = extract_plate_text(vehicle_crop)

                    if tid is not None and plate_text != "UNKNOWN":
                        vehicle_plate_map[tid] = plate_text

                    display_label = plate_text if plate_text != "UNKNOWN" else f"{cls_name} DETECTED"
                    vehicle_key = plate_text if plate_text != "UNKNOWN" else (f"{cls_name}_{tid}" if tid is not None else f"{cls_name}_{x1//100}_{y1//100}")

                    if vehicle_key not in seen_plates:
                        seen_plates.add(vehicle_key)
                        annotated_frame = frame.copy()
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        if matched_plate_box:
                            mpx1, mpy1, mpx2, mpy2 = matched_plate_box
                            cv2.rectangle(annotated_frame, (mpx1, mpy1), (mpx2, mpy2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, display_label, (x1, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        v_display = annotated_frame[max(0, y1-15):min(h, y2+15), max(0, x1-15):min(w, x2+15)]
                        if v_display.size > 0:
                            _, vbuf = cv2.imencode('.jpg', v_display)
                            v_b64 = base64.b64encode(vbuf).decode('utf-8')
                            vehicles.append({
                                'plate': display_label,
                                'image_b64': v_b64,
                                'frame_number': current_frame_idx
                            })

                # Standalone plates that weren't inside vehicle bounding boxes
                for (px1, py1, px2, py2, pconf) in direct_plates:
                    p_crop = frame[max(0, py1-5):min(h, py2+5), max(0, px1-5):min(w, px2+5)]
                    if p_crop.size > 0:
                        p_text = ocr_plate_image(p_crop)
                        if p_text != "UNKNOWN" and p_text not in seen_plates:
                            seen_plates.add(p_text)
                            annotated_plate = frame[max(0, py1-20):min(h, py2+20), max(0, px1-20):min(w, px2+20)]
                            if annotated_plate.size > 0:
                                _, vbuf = cv2.imencode('.jpg', annotated_plate)
                                v_b64 = base64.b64encode(vbuf).decode('utf-8')
                                vehicles.append({
                                    'plate': p_text,
                                    'image_b64': v_b64,
                                    'frame_number': current_frame_idx
                                })
            except Exception as e:
                pass

            # --- 5. Weapon Detection ---
            try:
                # Run weapon detection every 2nd processed frame to significantly speed up without missing drawn weapons
                if (current_frame_idx // step) % 2 == 0:
                    w_results = weapon_model(frame, device=device, verbose=False, conf=0.30, iou=0.45, imgsz=640, half=(device == 'cuda'))
                else:
                    w_results = None
                if w_results and len(w_results[0].boxes) > 0:
                    for box in w_results[0].boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_name = weapon_model.names.get(int(box.cls[0]), "weapon")
                        
                        weapon_key = f"{cls_name}_{current_frame_idx // 15}"
                        if weapon_key not in seen_weapons:
                            seen_weapons.add(weapon_key)
                            
                            annotated_w = frame.copy()
                            cv2.rectangle(annotated_w, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(annotated_w, f"{cls_name.upper()} {conf:.2f}", (x1, max(25, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            weapon_display = annotated_w[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                            if weapon_display.size > 0:
                                _, wbuf = cv2.imencode('.jpg', weapon_display)
                                w_b64 = base64.b64encode(wbuf).decode('utf-8')
                                weapons.append({
                                    'type': cls_name.upper(),
                                    'image_b64': w_b64,
                                    'frame_number': current_frame_idx,
                                    'confidence': f"{conf:.2f}"
                                })
            except Exception as e:
                pass

            # --- 6. Fight Detection (Streamed on-the-fly, conditioned on multi-person presence) ---
            try:
                t_level, t_conf, t_people = fight_pipeline.process_stream_frame(
                    frame, people_count=len(person_boxes), imgsz=640
                )
                if t_level == "PHYSICAL ALTERCATION":
                    if t_conf > fight_highest_conf:
                        fight_highest_conf = t_conf
                        fight_threat_level = "PHYSICAL ALTERCATION"
                    fight_people_involved = max(fight_people_involved, t_people)

                    if not threat_spans or threat_spans[-1]["end_frame"] < current_frame_idx - 15:
                        threat_spans.append({
                            "max_confidence": float(t_conf),
                            "threat_level": "PHYSICAL ALTERCATION",
                            "start_frame": max(0, current_frame_idx - 30),
                            "end_frame": current_frame_idx
                        })
                    else:
                        threat_spans[-1]["end_frame"] = current_frame_idx
                        threat_spans[-1]["max_confidence"] = max(threat_spans[-1]["max_confidence"], float(t_conf))
                elif t_level == "SUSPICIOUS" and fight_threat_level != "PHYSICAL ALTERCATION":
                    if t_conf > fight_highest_conf:
                        fight_highest_conf = t_conf
                        fight_threat_level = "SUSPICIOUS"
            except Exception as e:
                pass

    cap.release()
    local_detector.close()

    # Package registered faces (each unique registered suspect appears EXACTLY ONCE with best crop)
    registered_faces = []
    for candidate in registered_candidates.values():
        _, buf = cv2.imencode('.jpg', candidate['face_bgr'])
        img_b64 = base64.b64encode(buf).decode('utf-8')
        registered_faces.append({
            'name': candidate['name'],
            'image_b64': img_b64,
            'frame_number': candidate['frame_number']
        })

    # Merge any unknown face tracks that have close embeddings (< 0.76)
    merged_unknown_tracks = []
    for trk in unknown_face_tracks:
        matched_m = None
        for m in merged_unknown_tracks:
            if float(np.linalg.norm(m['centroid'] - trk['centroid'])) < 0.76:
                matched_m = m
                break
        if matched_m is not None:
            if trk['best_score'] > matched_m['best_score']:
                matched_m['face_bgr'] = trk['face_bgr']
                matched_m['best_score'] = trk['best_score']
                matched_m['frame_number'] = trk['frame_number']
            matched_m['embeddings'].extend(trk['embeddings'])
            mean_e = np.mean(matched_m['embeddings'], axis=0)
            matched_m['centroid'] = mean_e / (np.linalg.norm(mean_e) or 1.0)
        else:
            merged_unknown_tracks.append(trk)

    # Package unknown faces (each unique unknown individual appears EXACTLY ONCE with best crop)
    unknown_faces = []
    for idx, trk in enumerate(merged_unknown_tracks):
        _, buf = cv2.imencode('.jpg', trk['face_bgr'])
        img_b64 = base64.b64encode(buf).decode('utf-8')
        label = f"Unknown Person #{idx + 1}" if len(merged_unknown_tracks) > 1 else "Unknown Person"
        unknown_faces.append({
            'name': label,
            'image_b64': img_b64,
            'frame_number': trk['frame_number']
        })

    # --- Compute Unique People / Head Count strictly based on unique faces ---
    total_unique_faces = len(registered_faces) + len(unknown_faces)
    if total_unique_faces > 0:
        unique_people_count = total_unique_faces
    else:
        unique_people_count = max(len(unique_person_ids), max_head_count)

    # --- Fight Analysis Packaging ---
    fight_analysis = {
        "overall_threat_level": fight_threat_level,
        "max_confidence": float(fight_highest_conf),
        "threat_spans": threat_spans,
        "people_involved": fight_people_involved
    }

    # --- Loitering Analysis (Instant Trajectory Classification, zero video re-read) ---
    loitering_res = None
    try:
        loitering_res = loitering_detector.predict_trajectories(trajectories, sample_step=step)
    except Exception as e:
        print(f"Loitering trajectory prediction error: {e}")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'registered_faces': registered_faces,
        'unknown_faces': unknown_faces,
        'vehicles': vehicles,
        'weapons': weapons,
        'fight_analysis': fight_analysis,
        'loitering': loitering_res,
        'max_head_count': unique_people_count,
        'unique_people_count': unique_people_count
    }


@app.route('/analyze-video', methods=['GET', 'POST'])
def analyze_video():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, 'analyze_' + file.filename)
            file.save(filepath)

            results = analyze_video_file(filepath)
            if results is None:
                flash('Error: Could not open the video file.')
                return redirect(url_for('analyze_video'))

            # Clean up uploaded file after processing
            try:
                os.remove(filepath)
            except:
                pass

            return render_template('analyze_video.html', results=results)
        else:
            flash('Please select a video file.')
            return redirect(url_for('analyze_video'))

    return render_template('analyze_video.html', results=None)


@app.route('/night-detection', methods=['GET', 'POST'])
def night_detection():
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename:
            filename = 'night_' + str(int(time.time())) + '_' + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            force_enhancement = request.form.get('force_enhancement') == 'true'
            detector = get_night_detector()
            if detector is None:
                flash('Error: Night Detection neural model could not be initialized.')
                return redirect(url_for('night_detection'))

            results = detector.analyze_video(filepath, force_enhancement=force_enhancement)

            # Clean up uploaded video file
            try:
                os.remove(filepath)
            except Exception:
                pass

            if results is None:
                flash('Error: Could not open or process the uploaded video file.')
                return redirect(url_for('night_detection'))

            return render_template('night_detection.html', results=results)
        else:
            flash('Please select a night surveillance video file.')
            return redirect(url_for('night_detection'))

    return render_template('night_detection.html', results=None)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)