---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: objects
    struct:
    - name: bbox
      list:
        list: float32
        length: 4
    - name: category
      list:
        class_label:
          names:
            '0': license_plate
  splits:
  - name: train
    num_bytes: 167934117.152
    num_examples: 6176
  - name: validation
    num_bytes: 48452005.95
    num_examples: 1765
  - name: test
    num_bytes: 22564051
    num_examples: 882
  download_size: 236641716
  dataset_size: 238950174.102
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
license: cc-by-4.0
task_categories:
- object-detection
pretty_name: license plate detection dataset
size_categories:
- 1K<n<10K
tags:
- license-plate
- traffic
- vehicles
- alpr
- vision
- detection
---

# License Plate Detection Dataset

## Overview
This dataset is designed for **License Plate Detection** using modern object detection models such as **RT-DETR, DETR, YOLO**, etc.

It is a cleaned and simplified version of a Roboflow-exported COCO dataset, converted into **Hugging Face Datasets format** for easy training, evaluation, and reuse.

The dataset contains **only one object class: `license_plate`**, making it ideal for single-class detection tasks and real-time pipelines.

---

## Dataset Structure
The dataset is organized into three splits:

- `train`
- `validation`
- `test`

Each sample contains:
- An image
- One or more bounding boxes for license plates

Bounding boxes follow the **COCO format**: `(x_min, y_min, width, height)`

---

## Classes
This is a **single-class object detection dataset**.

| Class ID | Class Name      |
|--------|-----------------|
| 0      | license_plate   |

---

## 📥 Downloading the Dataset

This dataset can be downloaded and used directly with the Hugging Face `datasets` library.

```python
from datasets import load_dataset

dataset = load_dataset("justjuu/real-time-license-plate-detection-coco-hf")
```
The dataset will be loaded with the following splits:
- `train`
- `test`
- `valid`

### 🔧 Example Usage
```python
sample = dataset["train"][0]

image = sample["image"]
bboxes = sample["objects"]["bbox"]
labels = sample["objects"]["category"]
```
