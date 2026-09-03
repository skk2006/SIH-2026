FIGHT DETECTION SYSTEM
======================

This package contains the trained fight-detection model
and everything required to run inference on a laptop.

IMPORTANT:
This package is for INFERENCE / TESTING.

It does NOT contain the RWF-2000 training dataset.


------------------------------------------------------------
PROJECT PIPELINE
------------------------------------------------------------

Video
  |
  v
YOLOv8n-Pose
  |
  v
ByteTrack
  |
  v
17 human keypoints
  |
  v
Pose + motion features
  |
  v
GRU temporal classifier
  |
  v
Fight probability
  |
  v
Rule confirmation
  |
  +----> FIGHT CONFIRMED
  |
  +----> NO CONFIRMED FIGHT


------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------

Recommended:

Python 3.10 or Python 3.11

A GPU is recommended for faster inference, but the
current Python pipeline can also use CPU.


------------------------------------------------------------
INSTALLATION - WINDOWS
------------------------------------------------------------

1. Open Command Prompt or PowerShell.

2. Go to the FightDetection_Inference folder.

Example:

cd C:\FightDetection_Inference


3. Create a virtual environment:

python -m venv venv


4. Activate it:

venv\Scripts\activate


5. Install dependencies:

pip install -r requirements.txt


------------------------------------------------------------
RUN DETECTION
------------------------------------------------------------

Option 1 - easiest:

run.bat "test_videos\fight.mp4"


Option 2 - manually:

python code\inference.py ^
  --source "test_videos\fight.mp4" ^
  --weights "models\best_model.pt" ^
  --pose_model "yolov8n-pose.pt" ^
  --output "outputs\result.mp4"


------------------------------------------------------------
OUTPUT
------------------------------------------------------------

The annotated result will be:

outputs\result.mp4


The video contains the detection information generated
by the inference pipeline.


------------------------------------------------------------
IMPORTANT MODEL FILES
------------------------------------------------------------

models\best_model.pt

This is the trained GRU temporal fight classifier.


yolov8n-pose.pt

This is the YOLO pose model used to obtain human
keypoints.


models\model_config.json

Contains model configuration / threshold information.


------------------------------------------------------------
DO NOT RENAME THESE FILES
------------------------------------------------------------

Keep:

models\best_model.pt

models\model_config.json

yolov8n-pose.pt


------------------------------------------------------------
NOT INCLUDED
------------------------------------------------------------

The following are intentionally NOT included:

RWF-2000 dataset
features/
train.py
dataset_builder.py
latest_checkpoint.pt
fight_classifier.onnx
fight_classifier.onnx.data


These are not required for normal Python inference.


------------------------------------------------------------
TROUBLESHOOTING
------------------------------------------------------------

If Python is not recognized:

Install Python and make sure "Add Python to PATH"
was selected during installation.


If PyTorch installation fails:

Install the appropriate PyTorch build for the
friend's CPU/GPU system and then run:

pip install ultralytics opencv-python numpy


If inference.py reports a missing Python module:

Install the missing package with pip.


If the model does not detect fights correctly:

Do NOT retrain immediately.

First test several fight and non-fight videos and
record the results.


------------------------------------------------------------
VERSION
------------------------------------------------------------

This package contains the currently trained
Pose + Temporal + Rule Confirmation system.
