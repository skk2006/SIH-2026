
import numpy as np

NUM_KEYPOINTS = 17
POSE_SIZE = 34

# Feature layout:
#
# Person A pose              34
# Person B pose              34
# Person A velocity          34
# Person B velocity          34
# Relative velocity          34
# Center distance             1
# Person A center             2
# Person B center             2
#
# TOTAL = 175


def normalize_keypoints(keypoints, bbox):

    x1, y1, x2, y2 = bbox

    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)

    kp = np.asarray(keypoints, dtype=np.float32).copy()

    kp[:, 0] = (kp[:, 0] - x1) / w
    kp[:, 1] = (kp[:, 1] - y1) / h

    return kp


def pose_vector(person):

    kp = normalize_keypoints(
        person["keypoints"],
        person["bbox"]
    )

    return kp.reshape(-1).astype(np.float32)


def center_normalized(bbox, frame_width, frame_height):

    x1, y1, x2, y2 = bbox

    cx = ((x1 + x2) / 2.0) / max(frame_width, 1)
    cy = ((y1 + y2) / 2.0) / max(frame_height, 1)

    return np.array([cx, cy], dtype=np.float32)


def build_frame_feature(
    person_a,
    person_b,
    previous_a,
    previous_b,
    frame_width,
    frame_height
):

    pose_a = pose_vector(person_a)
    pose_b = pose_vector(person_b)

    if previous_a is None or previous_b is None:

        velocity_a = np.zeros(POSE_SIZE, dtype=np.float32)
        velocity_b = np.zeros(POSE_SIZE, dtype=np.float32)

    else:

        previous_pose_a = pose_vector(previous_a)
        previous_pose_b = pose_vector(previous_b)

        velocity_a = pose_a - previous_pose_a
        velocity_b = pose_b - previous_pose_b

    relative_velocity = velocity_b - velocity_a

    center_a = center_normalized(
        person_a["bbox"],
        frame_width,
        frame_height
    )

    center_b = center_normalized(
        person_b["bbox"],
        frame_width,
        frame_height
    )

    center_distance = np.linalg.norm(
        center_a - center_b
    )

    feature = np.concatenate([
        pose_a,                    # 34
        pose_b,                    # 34
        velocity_a,                # 34
        velocity_b,                # 34
        relative_velocity,         # 34
        np.array([center_distance], dtype=np.float32), # 1
        center_a,                  # 2
        center_b                   # 2
    ])

    assert feature.shape == (175,), feature.shape

    return feature.astype(np.float32)


def relative_speed_from_feature(feature):

    # relative velocity occupies [136:170]
    rel_velocity = feature[136:170]

    return float(np.linalg.norm(rel_velocity))


def distance_from_feature(feature):

    return float(feature[170])
