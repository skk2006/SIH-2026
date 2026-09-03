
import argparse
import json
from collections import deque

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from model import (
    PoseTemporalClassifier,
    WINDOW_SIZE
)

from pose_features import (
    build_frame_feature
)

from rule_layer import (
    FightAlertAggregator,
    RuleLayerConfig
)


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

            b1 = p1["bbox"]
            b2 = p2["bbox"]

            c1 = (
                (b1[0] + b1[2]) / 2,
                (b1[1] + b1[3]) / 2
            )

            c2 = (
                (b2[0] + b2[2]) / 2,
                (b2[1] + b2[3]) / 2
            )

            d = (
                (c1[0] - c2[0]) ** 2 +
                (c1[1] - c2[1]) ** 2
            )

            if d < best_distance:

                best_distance = d
                best_pair = (ids[i], ids[j])

    return best_pair


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True
    )

    parser.add_argument(
        "--weights",
        required=True
    )

    parser.add_argument(
        "--pose_model",
        default="yolov8n-pose.pt"
    )

    parser.add_argument(
        "--output",
        default="annotated_output.mp4"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    classifier = PoseTemporalClassifier().to(
        device
    )

    state = torch.load(
        args.weights,
        map_location=device
    )

    classifier.load_state_dict(
        state
    )

    classifier.eval()

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = 0.75

    config_path = args.weights.replace(
        "best_model.pt",
        "model_config.json"
    )

    try:

        with open(config_path, "r") as f:
            config = json.load(f)

        threshold = float(
            config.get(
                "threshold",
                0.75
            )
        )

    except Exception:

        pass

    print(
        "Model threshold:",
        threshold
    )

    # --------------------------------------------------------
    # POSE MODEL
    # --------------------------------------------------------

    pose_model = YOLO(
        args.pose_model
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.source
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open source: {args.source}"
        )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 20.0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height)
    )

    cap.release()

    # --------------------------------------------------------
    # TEMPORAL STATE
    # --------------------------------------------------------

    buffer = deque(
        maxlen=WINDOW_SIZE
    )

    previous_people = {}

    previous_pair = None

    rules = FightAlertAggregator(
        RuleLayerConfig(
            model_confidence_threshold=threshold
        )
    )

    # --------------------------------------------------------
    # TRACK + POSE
    # --------------------------------------------------------

    results = pose_model.track(
        source=args.source,
        conf=args.conf,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False
    )

    for frame_idx, result in enumerate(results):

        frame = result.orig_img.copy()

        if (
            result.boxes is None
            or result.boxes.id is None
            or result.keypoints is None
        ):

            buffer.clear()
            previous_people = {}
            previous_pair = None

            writer.write(frame)

            continue

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

        pair = choose_pair(
            people,
            previous_pair
        )

        # Draw all tracked people.
        for track_id, person in people.items():

            x1, y1, x2, y2 = map(
                int,
                person["bbox"]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        if pair is None:

            buffer.clear()
            previous_people = people
            previous_pair = None

            writer.write(frame)

            continue

        a_id, b_id = pair

        person_a = people[a_id]
        person_b = people[b_id]

        previous_a = previous_people.get(
            a_id
        )

        previous_b = previous_people.get(
            b_id
        )

        feature = build_frame_feature(
            person_a,
            person_b,
            previous_a,
            previous_b,
            width,
            height
        )

        buffer.append(feature)

        previous_people = people
        previous_pair = pair

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        confirmed = False
        probability = 0.0

        if len(buffer) == WINDOW_SIZE:

            window = np.asarray(
                buffer,
                dtype=np.float32
            )

            tensor = torch.from_numpy(
                window
            ).unsqueeze(0).to(
                device
            )

            with torch.no_grad():

                logits = classifier(
                    tensor
                )

                probability = float(
                    torch.sigmoid(
                        logits
                    ).item()
                )

            rule_result = rules.update(
                probability,
                window
            )

            confirmed = rule_result[
                "confirmed"
            ]

            if confirmed:

                print(
                    f"CONFIRMED FIGHT | "
                    f"frame={frame_idx} | "
                    f"prob={probability:.3f} | "
                    f"distance={rule_result['distance']:.3f} | "
                    f"relative_speed={rule_result['relative_speed']:.4f}"
                )

        # ----------------------------------------------------
        # DRAW STATUS
        # ----------------------------------------------------

        status = (
            "FIGHT CONFIRMED"
            if confirmed
            else f"Fight probability: {probability:.2f}"
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

        writer.write(frame)

    writer.release()

    print()
    print(
        "Annotated output:",
        args.output
    )


if __name__ == "__main__":
    main()
