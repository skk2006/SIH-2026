"""
Night Detection and Tracking Engine for Low-Light Surveillance.
Provides strict unique entity identification (people, vehicles, objects)
with multi-object tracking, low-light enhancement, and best-crop extraction.
"""

import os
import cv2
import base64
import numpy as np
import torch
from ultralytics import YOLO

from .night_enhancer import NightEnhancer


class NightDetector:
    """
    Dedicated night-time detection engine.
    Ensures that persons, vehicles, and objects detected across video footage
    are strictly deduplicated and presented as unique individual entries.
    """

    def __init__(self, model_path=None, conf_threshold=0.28, iou_threshold=0.45):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "night_model.pt")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NightDetector] Loading specialized night model on {self.device} from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enhancer = NightEnhancer(low_light_threshold=90.0)

        # Map classes to human-readable names and categories
        self.vehicle_classes = {'bike', 'bus', 'car', 'motor', 'other vehicle', 'truck'}
        self.person_classes = {'person'}

    @staticmethod
    def _compute_sharpness(img_bgr):
        """Estimate image sharpness using the variance of the Laplacian."""
        if img_bgr is None or img_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _extract_color_histogram(img_bgr):
        """Compute normalized HSV color histogram for re-ID / duplicate merging."""
        if img_bgr is None or img_bgr.size == 0:
            return None
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def analyze_video(self, video_path, sample_rate_fps=8, force_enhancement=False):
        """
        Process night video file and return strictly unique detected entities.
        
        Returns:
            dict containing:
                - unique_persons: list of unique persons (shown only once)
                - unique_vehicles: list of unique vehicles (shown only once)
                - unique_objects: list of unique objects (shown only once)
                - stats: overall detection and enhancement statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 25.0

        duration_sec = total_frames / orig_fps if total_frames > 0 else 0

        # Determine frame skip step to process at ~sample_rate_fps for high efficiency
        step = max(1, int(orig_fps / sample_rate_fps))

        # Persistent track registry: track_id -> TrackRecord
        # TrackRecord stores:
        # {
        #   'category': 'person' | 'vehicle' | 'object',
        #   'class_name': str,
        #   'first_frame': int,
        #   'last_frame': int,
        #   'first_sec': float,
        #   'last_sec': float,
        #   'max_conf': float,
        #   'best_crop': np.ndarray,
        #   'best_quality': float,
        #   'best_annotated_crop': np.ndarray,
        #   'total_detections': int,
        #   'hist': np.ndarray
        # }
        tracks = {}
        fallback_counter = 1000  # for detections that tracker didn't assign an ID to

        frame_index = 0
        processed_count = 0
        low_light_frames_count = 0
        total_orig_brightness = 0.0
        total_enhanced_brightness = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % step != 0:
                frame_index += 1
                continue

            processed_count += 1
            current_sec = frame_index / orig_fps

            # 1. Resize very high-res frames to 720p for fast reliable inference
            h, w = frame.shape[:2]
            if max(h, w) > 1280:
                scale = 1280.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]

            # 2. Adaptive Low-Light Enhancement
            enhanced_frame, is_low_light, lum_stats = self.enhancer.enhance(
                frame, force_enhance=force_enhancement
            )
            if is_low_light:
                low_light_frames_count += 1
            total_orig_brightness += lum_stats["orig_brightness"]
            total_enhanced_brightness += lum_stats["enhanced_brightness"]

            # 3. Multi-Object Tracking with specialized Night YOLO Model
            try:
                results = self.model.track(
                    enhanced_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            except Exception as e:
                # Fallback to standard inference if tracking encounters issue
                results = self.model(
                    enhanced_frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = self.model.names.get(cls_id, "object").lower()

                    # Determine track id
                    if box.id is not None:
                        track_id = int(box.id[0].item())
                    else:
                        # Spatial proximity fallback if no tracker ID
                        track_id = fallback_counter
                        fallback_counter += 1

                    # Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    if crop_w < 15 or crop_h < 15:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Determine high-level category
                    if class_name in self.person_classes:
                        category = "person"
                    elif class_name in self.vehicle_classes:
                        category = "vehicle"
                    else:
                        category = "object"

                    # Compute quality score for best thumbnail selection
                    # Prefers high confidence, sharp details (less blur), and good resolution
                    sharpness = self._compute_sharpness(crop)
                    area = crop_w * crop_h
                    quality = (conf * 0.55) + (min(sharpness / 400.0, 1.0) * 0.30) + (min(area / (140 * 140), 1.0) * 0.15)

                    # Create clean annotated crop for display
                    annotated_frame = enhanced_frame.copy()
                    color = (0, 255, 128) if category == "person" else (255, 128, 0) if category == "vehicle" else (0, 191, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{class_name.upper()} #{track_id} {conf:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label_text,
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
                    pad_y1, pad_y2 = max(0, y1 - 15), min(h, y2 + 15)
                    pad_x1, pad_x2 = max(0, x1 - 15), min(w, x2 + 15)
                    annotated_crop = annotated_frame[pad_y1:pad_y2, pad_x1:pad_x2]

                    if track_id not in tracks:
                        tracks[track_id] = {
                            "track_id": track_id,
                            "category": category,
                            "class_name": class_name,
                            "first_frame": frame_index,
                            "last_frame": frame_index,
                            "first_sec": current_sec,
                            "last_sec": current_sec,
                            "max_conf": conf,
                            "best_crop": crop,
                            "best_annotated_crop": annotated_crop if annotated_crop.size > 0 else crop,
                            "best_quality": quality,
                            "total_detections": 1,
                            "hist": self._extract_color_histogram(crop)
                        }
                    else:
                        rec = tracks[track_id]
                        rec["last_frame"] = frame_index
                        rec["last_sec"] = current_sec
                        rec["total_detections"] += 1
                        if conf > rec["max_conf"]:
                            rec["max_conf"] = conf

                        # Update best crop if higher visual quality or significantly higher confidence
                        if quality > rec["best_quality"]:
                            rec["best_quality"] = quality
                            rec["best_crop"] = crop
                            if annotated_crop.size > 0:
                                rec["best_annotated_crop"] = annotated_crop

            frame_index += 1

        cap.release()

        # 4. Secondary Deduplication & Track Merging
        # Merge fragmented tracks for the same entity if visual similarity is very high
        # and timestamps don't conflict, ensuring each entity is shown ONCE.
        merged_tracks = self._merge_similar_tracks(tracks)

        # 5. Format results into Unique Persons, Unique Vehicles, Unique Objects
        unique_persons = []
        unique_vehicles = []
        unique_objects = []

        # Sort tracks chronologically by first seen
        sorted_tracks = sorted(merged_tracks.values(), key=lambda t: t["first_sec"])

        p_counter = 1
        v_counter = 1
        o_counter = 1

        vehicle_type_counts = {}

        for tr in sorted_tracks:
            # Encode best annotated crop to base64
            best_img = tr.get("best_annotated_crop")
            if best_img is None or best_img.size == 0:
                best_img = tr.get("best_crop")

            if best_img is None or best_img.size == 0:
                continue

            _, buf = cv2.imencode(".jpg", best_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_b64 = base64.b64encode(buf).decode("utf-8")

            dur_sec = max(0.1, tr["last_sec"] - tr["first_sec"])
            first_m, first_s = divmod(int(tr["first_sec"]), 60)
            last_m, last_s = divmod(int(tr["last_sec"]), 60)
            time_range_str = f"{first_m:02d}:{first_s:02d} - {last_m:02d}:{last_s:02d}"

            item = {
                "track_id": tr["track_id"],
                "class_name": tr["class_name"].capitalize(),
                "confidence_pct": round(tr["max_conf"] * 100, 1),
                "first_frame": tr["first_frame"],
                "last_frame": tr["last_frame"],
                "time_range": time_range_str,
                "duration_sec": f"{dur_sec:.1f}s",
                "total_detections": tr["total_detections"],
                "image_b64": img_b64
            }

            if tr["category"] == "person":
                item["unique_id"] = f"PER-{p_counter:02d}"
                item["title"] = f"Person #{p_counter}"
                unique_persons.append(item)
                p_counter += 1
            elif tr["category"] == "vehicle":
                v_type = tr["class_name"].capitalize()
                item["unique_id"] = f"VEH-{v_counter:02d}"
                item["title"] = f"{v_type} #{v_counter}"
                item["vehicle_type"] = v_type
                unique_vehicles.append(item)
                v_counter += 1
                vehicle_type_counts[v_type] = vehicle_type_counts.get(v_type, 0) + 1
            else:
                item["unique_id"] = f"OBJ-{o_counter:02d}"
                item["title"] = f"{tr['class_name'].capitalize()} #{o_counter}"
                unique_objects.append(item)
                o_counter += 1

        avg_orig_lum = round(total_orig_brightness / max(1, processed_count), 1)
        avg_enh_lum = round(total_enhanced_brightness / max(1, processed_count), 1)
        low_light_pct = round((low_light_frames_count / max(1, processed_count)) * 100, 1)

        stats = {
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "duration_sec": round(duration_sec, 1),
            "fps": round(orig_fps, 1),
            "total_unique_persons": len(unique_persons),
            "total_unique_vehicles": len(unique_vehicles),
            "total_unique_objects": len(unique_objects),
            "vehicle_type_counts": vehicle_type_counts,
            "low_light_frames_pct": low_light_pct,
            "avg_original_brightness": avg_orig_lum,
            "avg_enhanced_brightness": avg_enh_lum,
            "night_vision_active": low_light_pct > 20.0 or force_enhancement
        }

        return {
            "unique_persons": unique_persons,
            "unique_vehicles": unique_vehicles,
            "unique_objects": unique_objects,
            "stats": stats
        }

    def _merge_similar_tracks(self, tracks):
        """
        Merge candidate tracks of the same category and class if their visual appearance
        matches closely and they occur closely in time, ensuring no duplicate entries
        for the same physical person or vehicle.
        """
        if len(tracks) <= 1:
            return tracks

        merged = {}
        processed_ids = set()

        # Separate by class for comparison
        track_list = list(tracks.values())

        for i in range(len(track_list)):
            tr1 = track_list[i]
            id1 = tr1["track_id"]
            if id1 in processed_ids:
                continue

            merged_record = tr1.copy()
            processed_ids.add(id1)

            for j in range(i + 1, len(track_list)):
                tr2 = track_list[j]
                id2 = tr2["track_id"]
                if id2 in processed_ids:
                    continue

                # Must be exact same class
                if tr1["class_name"] != tr2["class_name"]:
                    continue

                # Compare visual appearance if histograms available
                hist1 = tr1.get("hist")
                hist2 = tr2.get("hist")
                similarity = 0.0
                if hist1 is not None and hist2 is not None:
                    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

                # If appearance matches strongly (> 0.82) or time gaps are small with same class
                time_gap = abs(tr2["first_sec"] - merged_record["last_sec"])
                if (similarity > 0.84 and time_gap < 5.0) or (similarity > 0.90):
                    # Merge tr2 into merged_record
                    merged_record["last_frame"] = max(merged_record["last_frame"], tr2["last_frame"])
                    merged_record["last_sec"] = max(merged_record["last_sec"], tr2["last_sec"])
                    merged_record["max_conf"] = max(merged_record["max_conf"], tr2["max_conf"])
                    merged_record["total_detections"] += tr2["total_detections"]

                    # Keep the best visual crop between the two
                    if tr2["best_quality"] > merged_record["best_quality"]:
                        merged_record["best_quality"] = tr2["best_quality"]
                        merged_record["best_crop"] = tr2["best_crop"]
                        merged_record["best_annotated_crop"] = tr2["best_annotated_crop"]

                    processed_ids.add(id2)

            merged[merged_record["track_id"]] = merged_record

        return merged
