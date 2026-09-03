import cv2
import torch
import numpy as np
import os
import sys
from collections import deque
from ultralytics import YOLO

# Add the FightDetection_Inference code to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGHT_INFERENCE_PATH = os.path.join(BASE_DIR, "FightDetection_Inference", "FightDetection_Inference", "code")
if FIGHT_INFERENCE_PATH not in sys.path:
    sys.path.append(FIGHT_INFERENCE_PATH)

from model import PoseTemporalClassifier, WINDOW_SIZE
from pose_features import build_frame_feature
from rule_layer import FightAlertAggregator, RuleLayerConfig

def choose_pair(people, previous_pair):
    if len(people) < 2:
        return None
    ids = list(people.keys())
    if previous_pair is not None:
        a, b = previous_pair
        if a in people and b in people:
            return (a, b)
    best_pair = None
    best_distance = float("inf")
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            p1 = people[ids[i]]
            p2 = people[ids[j]]
            b1, b2 = p1["bbox"], p2["bbox"]
            c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
            c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
            d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
            if d < best_distance:
                best_distance = d
                best_pair = (ids[i], ids[j])
    return best_pair


class DeepMultimodalThreatPipeline:
    def __init__(self, fight_threshold=None, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Temporal Classifier
        self.classifier = PoseTemporalClassifier().to(self.device)
        model_weights = os.path.join(BASE_DIR, "FightDetection_Inference", "FightDetection_Inference", "models", "best_model.pt")
        
        if os.path.exists(model_weights):
            state = torch.load(model_weights, map_location=self.device)
            self.classifier.load_state_dict(state)
            print(f"Loaded fight classifier weights from {model_weights}")
        else:
            print(f"Warning: Fight classifier weights not found at {model_weights}")
            
        self.classifier.eval()
        
        # Load calibrated threshold from model_config.json if not specified
        config_path = os.path.join(BASE_DIR, "FightDetection_Inference", "FightDetection_Inference", "models", "model_config.json")
        calibrated_threshold = 0.95
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                calibrated_threshold = float(cfg.get("threshold", 0.95))
            except Exception:
                pass
        
        self.fight_threshold = fight_threshold if fight_threshold is not None else calibrated_threshold
        print(f"Fight Detection threshold initialized to: {self.fight_threshold}")
        
        # Load YOLO Pose Model
        pose_model_path = os.path.join(BASE_DIR, "FightDetection_Inference", "FightDetection_Inference", "yolov8n-pose.pt")
        self.pose_model = YOLO(pose_model_path)
        if self.device == "cuda":
            self.pose_model.to(self.device)
        
        self.reset_state()

    def reset_state(self):
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.previous_people = {}
        self.previous_pair = None
        
        config = RuleLayerConfig(
            model_confidence_threshold=self.fight_threshold,
            consecutive_windows_required=3,
            min_relative_speed=0.025,
            max_distance=0.60
        )
        self.rules = FightAlertAggregator(config)

    def process_frame(self, frame, imgsz=640):
        height, width = frame.shape[:2]
        
        # Track pose (conf=0.35 matches trained model, imgsz=640 optimizes speed and VRAM)
        results = self.pose_model.track(
            source=frame,
            conf=0.35,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
            imgsz=imgsz
        )
        
        threat_level = "NORMAL"
        conf = 0.0
        
        result = results[0]
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            self.buffer.clear()
            self.previous_people = {}
            self.previous_pair = None
            return frame, threat_level, conf
            
        ids = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        
        people = {}
        for i, track_id in enumerate(ids):
            if keypoints[i].shape != (17, 2):
                continue
            people[int(track_id)] = {
                "track_id": int(track_id),
                "bbox": boxes[i].astype(np.float32),
                "keypoints": keypoints[i].astype(np.float32)
            }
            
        pair = choose_pair(people, self.previous_pair)
        if pair is None:
            self.buffer.clear()
            self.previous_people = people
            self.previous_pair = None
            return frame, threat_level, conf
            
        a_id, b_id = pair
        person_a = people[a_id]
        person_b = people[b_id]
        
        previous_a = self.previous_people.get(a_id)
        previous_b = self.previous_people.get(b_id)
        
        feature = build_frame_feature(person_a, person_b, previous_a, previous_b, width, height)
        self.buffer.append(feature)
        
        self.previous_people = people
        self.previous_pair = pair
        
        if len(self.buffer) == WINDOW_SIZE:
            window = np.asarray(self.buffer, dtype=np.float32)
            tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.classifier(tensor)
                probability = float(torch.sigmoid(logits).item())
                
            rule_result = self.rules.update(probability, window)
            confirmed = rule_result["confirmed"]
            
            if confirmed:
                threat_level = "PHYSICAL ALTERCATION"
                conf = probability
            elif probability >= self.fight_threshold and rule_result.get("proximity_ok", False):
                threat_level = "SUSPICIOUS"
                conf = probability
                
        return frame, threat_level, conf

    def process_stream_frame(self, frame, people_count=None, imgsz=640):
        """Streaming fight analysis optimized for real-time and single-pass video processing.
        Bypasses pose tracking if fewer than 2 people are in the frame."""
        if people_count is not None and people_count < 2:
            if self.buffer:
                self.buffer.clear()
                self.previous_people = {}
                self.previous_pair = None
            return "NORMAL", 0.0, 0

        _, threat_level, conf = self.process_frame(frame, imgsz=imgsz)
        return threat_level, conf, len(self.previous_people)

    def analyze_full_video_deep(self, all_frames, fps, imgsz=640):
        threat_spans = []
        self.reset_state()
        
        fight_occurred = False
        max_conf = 0.0
        people_involved = 0
        
        # Analyze frame by frame sequentially for accuracy
        for i, frame in enumerate(all_frames):
            _, threat_level, conf = self.process_frame(frame, imgsz=imgsz)
            if threat_level == "PHYSICAL ALTERCATION":
                fight_occurred = True
                if conf > max_conf:
                    max_conf = conf
                
                people_involved = max(people_involved, len(self.previous_people))
                
                # Expand span
                if not threat_spans or threat_spans[-1]["end_frame"] < i - 15:
                    threat_spans.append({
                        "max_confidence": float(conf),
                        "threat_level": "PHYSICAL ALTERCATION",
                        "start_frame": max(0, i - WINDOW_SIZE),
                        "end_frame": i
                    })
                else:
                    threat_spans[-1]["end_frame"] = i
                    threat_spans[-1]["max_confidence"] = max(threat_spans[-1]["max_confidence"], float(conf))

        return {
            "threat_spans": threat_spans,
            "people_involved": people_involved if fight_occurred else 0
        }
