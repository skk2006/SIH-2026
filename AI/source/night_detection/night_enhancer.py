"""
Night Vision Enhancement Pipeline for Surveillance Footage.
Optimized for low-light object detection of people, vehicles, and objects.
"""

import cv2
import numpy as np


class NightEnhancer:
    """
    Adaptive low-light enhancement module that analyzes frame luminance
    and applies adaptive CLAHE + gamma correction to uncover people and
    vehicles hidden in night-time shadows without oversaturating headlights.
    """

    def __init__(self, low_light_threshold=90.0, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.low_light_threshold = low_light_threshold
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def calculate_brightness(self, frame_bgr):
        """Calculate mean luminance (0-255) using perceptual weights."""
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0
        # Grayscale brightness using standard Rec. 601 luma
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def enhance(self, frame_bgr, force_enhance=False):
        """
        Enhance low-light frame if mean luminance is below threshold or if forced.
        Returns:
            enhanced_frame: np.ndarray (BGR)
            is_low_light: bool
            stats: dict with luminance information
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, False, {"orig_brightness": 0, "enhanced_brightness": 0}

        orig_brightness = self.calculate_brightness(frame_bgr)
        is_low_light = orig_brightness < self.low_light_threshold

        if not is_low_light and not force_enhance:
            return frame_bgr, False, {
                "orig_brightness": round(orig_brightness, 1),
                "enhanced_brightness": round(orig_brightness, 1),
                "enhancement_applied": "None (Well Lit)"
            }

        # Step 1: Convert BGR to LAB color space
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Step 2: Apply CLAHE on the Lightness channel
        enhanced_l = self.clahe.apply(l_channel)

        # Step 3: Adaptive Gamma Correction on Lightness channel
        mean_l = float(np.mean(l_channel)) / 255.0
        if mean_l > 0.03 and mean_l < 0.75:
            # Shift midtones upward for dark images (gamma < 1 brightens shadows)
            gamma = float(np.clip(mean_l ** 0.4, 0.45, 0.85))
            table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
            enhanced_l = cv2.LUT(enhanced_l, table)

        # Step 4: Re-merge and convert back to BGR
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Step 5: Subtle denoising for dark scenes to suppress camera sensor noise
        if orig_brightness < 60:
            enhanced_bgr = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=35, sigmaSpace=35)

        enhanced_brightness = self.calculate_brightness(enhanced_bgr)

        return enhanced_bgr, True, {
            "orig_brightness": round(orig_brightness, 1),
            "enhanced_brightness": round(enhanced_brightness, 1),
            "enhancement_applied": "Adaptive CLAHE + Gamma Correction"
        }
