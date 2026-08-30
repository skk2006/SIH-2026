
# LOITERING DETECTION MODULE

Standalone loitering detection component for SENTINEL-AI.

## Files

- loitering_detector.py
- loitering_model.pkl
- yolov8n.pt
- requirements.txt

## Installation

Install the required packages:

pip install -r requirements.txt

## Usage

from loitering_detector import LoiteringDetector

detector = LoiteringDetector(
    yolo_path="yolov8n.pt",
    classifier_path="loitering_model.pkl"
)

result = detector.analyze(
    "input_video.mp4",
    "loitering_result.mp4"
)

print(result)

## Output

The detector returns:

{
    "loitering_ids": [...],
    "output_video": "loitering_result.mp4"
}

## Integration with SENTINEL-AI

This module is designed to remain independent.

The main SENTINEL-AI application can call:

result = detector.analyze(video_path)

and use:

result["loitering_ids"]

result["output_video"]

Other detection features can be implemented as separate modules.

## Detection Pipeline

Video
  ↓
YOLO person detection + tracking
  ↓
Trajectory extraction
  ↓
Movement feature extraction
  ↓
Random Forest classifier
  ↓
5-second minimum tracking-duration rule
  ↓
LOITERING / NORMAL

## Important

YOLO is used for person detection and tracking.

The Random Forest is the loitering classifier.

YOLO does not need to be retrained when using this module.
