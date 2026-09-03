import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

# Compatibility alias for unpickling models saved under NumPy 2.x
if not hasattr(np, '_core'):
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = getattr(_core, 'multiarray', None)
    sys.modules['numpy._core.umath'] = getattr(_core, 'umath', None)
    sys.modules['numpy._core._multiarray_umath'] = getattr(_core, '_multiarray_umath', None)



class LoiteringDetector:

    def __init__(
        self,
        yolo_path="yolov8n.pt",
        classifier_path="loitering_model.pkl"
    ):

        self.yolo = YOLO(yolo_path)
        self.classifier = joblib.load(classifier_path)

        self.feature_columns = [
            "trajectory_length",
            "total_distance",
            "mean_speed",
            "max_speed",
            "std_speed",
            "x_range",
            "y_range",
            "displacement"
        ]


    def _extract_features(self, trajectories):

        features = []

        for person_id, points in trajectories.items():

            if len(points) < 2:
                continue

            points = sorted(points, key=lambda p: p[0])

            x = np.array([p[1] for p in points], dtype=float)
            y = np.array([p[2] for p in points], dtype=float)

            dx = np.diff(x)
            dy = np.diff(y)

            distance = np.sqrt(dx**2 + dy**2)

            features.append({
                "id": int(person_id),
                "trajectory_length": len(points),
                "total_distance": distance.sum(),
                "mean_speed": distance.mean(),
                "max_speed": distance.max(),
                "std_speed": distance.std(),
                "x_range": x.max() - x.min(),
                "y_range": y.max() - y.min(),
                "displacement": np.sqrt(
                    (x[-1] - x[0])**2 +
                    (y[-1] - y[0])**2
                )
            })

        return features


    def predict_trajectories(self, trajectories, sample_step=1):
        """Predict loitering on pre-collected trajectories without re-running video or YOLO."""
        if not trajectories:
            return {"loitering_ids": [], "total_persons_tracked": 0}

        # If sampled, interpolate trajectories to full frame rate so features match training
        processed_trajectories = {}
        for pid, points in trajectories.items():
            if len(points) < 2:
                continue
            sorted_pts = sorted(points, key=lambda p: p[0])
            if sample_step > 1:
                interp_pts = []
                for idx in range(len(sorted_pts) - 1):
                    f_start, x_start, y_start = sorted_pts[idx]
                    f_end, x_end, y_end = sorted_pts[idx + 1]
                    f_diff = f_end - f_start
                    if f_diff <= 0:
                        interp_pts.append((f_start, x_start, y_start))
                        continue
                    for f in range(f_start, f_end):
                        alpha = (f - f_start) / f_diff
                        interp_pts.append((f, x_start + alpha * (x_end - x_start), y_start + alpha * (y_end - y_start)))
                interp_pts.append(sorted_pts[-1])
                processed_trajectories[pid] = interp_pts
            else:
                processed_trajectories[pid] = sorted_pts

        feature_rows = self._extract_features(processed_trajectories)
        loitering_ids = set()

        if feature_rows:
            try:
                feature_df = pd.DataFrame(feature_rows)
                X = feature_df[self.feature_columns]
                predictions = self.classifier.predict(X)
                for row, prediction in zip(feature_rows, predictions):
                    if prediction == 1:
                        loitering_ids.add(int(row["id"]))
            except Exception as e:
                print(f"Loitering trajectory prediction error: {e}")

        return {
            "loitering_ids": list(loitering_ids),
            "total_persons_tracked": len(trajectories)
        }

    def analyze(
        self,
        video_path,
        output_path="loitering_result.mp4"
    ):

        # ----------------------------------------
        # YOLO TRACKING
        # ----------------------------------------

        results = self.yolo.track(
            source=video_path,
            persist=True,
            stream=True,
            classes=[0],
            verbose=False
        )

        trajectories = {}
        frame_data = []

        for frame_index, result in enumerate(results):

            frame = result.orig_img.copy()

            detections = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)

                for box, person_id in zip(boxes, ids):

                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if person_id not in trajectories:
                        trajectories[person_id] = []

                    trajectories[person_id].append(
                        (
                            frame_index,
                            center_x,
                            center_y
                        )
                    )

                    detections.append(
                        (
                            x1, y1, x2, y2,
                            int(person_id)
                        )
                    )

            frame_data.append(
                (frame, detections)
            )


        # ----------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------

        feature_rows = self._extract_features(
            trajectories
        )

        loitering_ids = set()

        if feature_rows:

            feature_df = pd.DataFrame(
                feature_rows
            )

            X = feature_df[
                self.feature_columns
            ]

            # DataFrame keeps feature names
            predictions = self.classifier.predict(X)

            for row, prediction in zip(
                feature_rows,
                predictions
            ):

                if prediction == 1:
                    loitering_ids.add(
                        row["id"]
                    )


        # ----------------------------------------
        # VIDEO INFORMATION
        # ----------------------------------------

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        cap.release()


        # ----------------------------------------
        # OUTPUT VIDEO
        # ----------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        # ----------------------------------------
        # DRAW RESULTS
        # ----------------------------------------

        for frame_index, (
            frame,
            detections
        ) in enumerate(frame_data):

            loitering_found = False

            for (
                x1, y1, x2, y2,
                person_id
            ) in detections:

                if person_id in loitering_ids:

                    loitering_found = True

                    color = (0, 0, 255)

                    label = (
                        f"ID {person_id} | "
                        f"LOITERING"
                    )

                    thickness = 3

                else:

                    color = (0, 255, 0)

                    label = (
                        f"ID {person_id} | "
                        f"NORMAL"
                    )

                    thickness = 2


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )


            # ------------------------------------
            # TOP STATUS BANNER
            # ------------------------------------

            if loitering_found:

                banner_color = (0, 0, 255)

                banner_text = (
                    "LOITERING DETECTED"
                )

            else:

                banner_color = (0, 140, 0)

                banner_text = "MONITORING"


            cv2.rectangle(
                frame,
                (0, 0),
                (width, 55),
                banner_color,
                -1
            )


            cv2.putText(
                frame,
                banner_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )


            # ------------------------------------
            # TIME
            # ------------------------------------

            current_time = (
                frame_index / fps
                if fps > 0
                else 0
            )

            cv2.putText(
                frame,
                f"Time: {current_time:.1f}s",
                (
                    width - 180,
                    height - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            out.write(frame)


        out.release()


        return {
            "loitering_ids":
                sorted(loitering_ids),

            "output_video":
                output_path
        }
