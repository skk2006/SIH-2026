
from dataclasses import dataclass

import numpy as np


@dataclass
class RuleLayerConfig:

    # GRU probability must stay above this.
    model_confidence_threshold: float = 0.75

    # People must be reasonably close.
    max_distance: float = 0.60

    # Relative keypoint motion must be sufficiently high.
    min_relative_speed: float = 0.025

    # Number of consecutive positive windows.
    consecutive_windows_required: int = 3


class FightAlertAggregator:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else RuleLayerConfig()
        )

        self.counter = 0

        self.alert_active = False

    def update(
        self,
        model_probability,
        feature_window
    ):

        # Distance is feature 170.
        distance = float(
            feature_window[-1, 170]
        )

        # Relative velocity = [136:170]
        relative_velocity = (
            feature_window[:, 136:170]
        )

        relative_speed = np.linalg.norm(
            relative_velocity,
            axis=1
        )

        mean_relative_speed = float(
            np.mean(relative_speed)
        )

        model_ok = (
            model_probability
            >= self.config.model_confidence_threshold
        )

        proximity_ok = (
            distance
            <= self.config.max_distance
        )

        motion_ok = (
            mean_relative_speed
            >= self.config.min_relative_speed
        )

        if (
            model_ok
            and proximity_ok
            and motion_ok
        ):

            self.counter += 1

        else:

            self.counter = 0
            self.alert_active = False

        if (
            self.counter
            >= self.config.consecutive_windows_required
        ):

            self.alert_active = True

        return {
            "confirmed": self.alert_active,
            "model_probability": model_probability,
            "distance": distance,
            "relative_speed": mean_relative_speed,
            "model_ok": model_ok,
            "proximity_ok": proximity_ok,
            "motion_ok": motion_ok,
            "counter": self.counter
        }
