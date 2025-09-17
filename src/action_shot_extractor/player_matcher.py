"""Player matching utilities for Action Shot Extractor.

This module provides functionality to identify target players in video frames
using either color-based detection or reference frame feature matching.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PlayerMatcher:
    """Handles player identification using color or reference frame matching."""

    def __init__(self, method: str = "color", **kwargs):
        """Initialize player matcher.

        Args:
            method: Either "color" or "reference" for detection method
            **kwargs: Additional arguments for specific methods
        """
        self.method = method
        self.matcher_data = {}

        if method == "color":
            self._init_color_matcher(**kwargs)
        elif method == "reference":
            self._init_reference_matcher(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _init_color_matcher(self, hex_color: str, tolerance: int = 30):
        """Initialize color-based matching.

        Args:
            hex_color: Target color in hex format (e.g., "#FF0000")
            tolerance: HSV hue tolerance for color matching
        """
        # Convert hex to BGR
        hex_color = hex_color.lstrip('#')
        bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

        # Convert to HSV for better color matching
        bgr_array = np.uint8([[bgr]])
        hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        target_hue = hsv[0][0][0]

        self.matcher_data = {
            "target_hue": target_hue,
            "tolerance": tolerance,
            "bgr": bgr
        }

        logger.info(f"Color matcher initialized: BGR{bgr}, HSV hue={target_hue}, tolerance={tolerance}")

    def _init_reference_matcher(self, reference_dir: str):
        """Initialize reference frame-based matching.

        Args:
            reference_dir: Directory containing front.jpg, side.jpg, back.jpg
        """
        ref_path = Path(reference_dir)

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

        # Load reference images
        reference_files = ["front.jpg", "side.jpg", "back.jpg"]
        reference_images = []
        reference_features = []

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)

        for ref_file in reference_files:
            ref_image_path = ref_path / ref_file

            if not ref_image_path.exists():
                logger.warning(f"Reference image not found: {ref_image_path}")
                continue

            # Load and process reference image
            ref_image = cv2.imread(str(ref_image_path))
            if ref_image is None:
                logger.warning(f"Could not load reference image: {ref_image_path}")
                continue

            # Convert to grayscale for feature detection
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

            # Detect features
            keypoints, descriptors = orb.detectAndCompute(ref_gray, None)

            if descriptors is not None:
                reference_images.append(ref_image)
                reference_features.append({
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "image": ref_image,
                    "name": ref_file.split('.')[0]
                })
                logger.info(f"Loaded {len(keypoints)} features from {ref_file}")
            else:
                logger.warning(f"No features detected in {ref_file}")

        if not reference_features:
            raise ValueError("No valid reference images found with detectable features")

        self.matcher_data = {
            "orb": orb,
            "reference_features": reference_features,
            "matcher": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        }

        logger.info(f"Reference matcher initialized with {len(reference_features)} reference images")

    def find_player_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Find target player in the given frame.

        Args:
            frame: Input video frame

        Returns:
            Dictionary containing detection results:
            - found: bool indicating if player was detected
            - confidence: confidence score (0.0 to 1.0)
            - bbox: bounding box (x, y, w, h) if found
            - method_data: method-specific additional data
        """
        if self.method == "color":
            return self._find_player_by_color(frame)
        elif self.method == "reference":
            return self._find_player_by_features(frame)
        else:
            return {"found": False, "confidence": 0.0, "bbox": None, "method_data": {}}

    def _find_player_by_color(self, frame: np.ndarray) -> Dict[str, Any]:
        """Find player using color-based detection."""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target_hue = self.matcher_data["target_hue"]
        tolerance = self.matcher_data["tolerance"]

        # Create color mask
        lower = np.array([max(0, target_hue - tolerance), 50, 50])
        upper = np.array([min(179, target_hue + tolerance), 255, 255])
        mask = cv2.inRange(hsv_frame, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"found": False, "confidence": 0.0, "bbox": None, "method_data": {"mask": mask}}

        # Find largest contour (assumed to be the player)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Calculate confidence based on area (simple heuristic)
        frame_area = frame.shape[0] * frame.shape[1]
        confidence = min(1.0, area / (frame_area * 0.05))  # Expect player to be ~5% of frame

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        return {
            "found": confidence > 0.1,  # Minimum confidence threshold
            "confidence": confidence,
            "bbox": (x, y, w, h),
            "method_data": {
                "mask": mask,
                "contour_area": area,
                "num_contours": len(contours)
            }
        }

    def _find_player_by_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Find player using ORB feature matching."""
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        orb = self.matcher_data["orb"]
        matcher = self.matcher_data["matcher"]
        reference_features = self.matcher_data["reference_features"]

        # Detect features in current frame
        frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

        if frame_descriptors is None:
            return {"found": False, "confidence": 0.0, "bbox": None,
                   "method_data": {"frame_features": 0}}

        best_match = None
        best_confidence = 0.0
        best_bbox = None
        match_details = []

        # Match against all reference images
        for ref_data in reference_features:
            ref_descriptors = ref_data["descriptors"]
            ref_keypoints = ref_data["keypoints"]

            # Match features
            matches = matcher.match(ref_descriptors, frame_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Filter good matches (distance threshold)
            good_matches = [m for m in matches if m.distance < 50]  # Tune this threshold

            if len(good_matches) < 10:  # Need at least 10 good matches
                match_details.append({
                    "reference": ref_data["name"],
                    "total_matches": len(matches),
                    "good_matches": len(good_matches),
                    "confidence": 0.0
                })
                continue

            # Calculate confidence based on match quality and quantity
            confidence = len(good_matches) / max(len(ref_keypoints), len(frame_keypoints))
            confidence = min(1.0, confidence * 2)  # Scale up confidence

            match_details.append({
                "reference": ref_data["name"],
                "total_matches": len(matches),
                "good_matches": len(good_matches),
                "confidence": confidence
            })

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = ref_data

                # Calculate bounding box from matched keypoints
                if len(good_matches) >= 4:  # Need at least 4 points for homography
                    try:
                        src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Find homography
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if M is not None:
                            # Transform reference image corners to frame coordinates
                            h, w = ref_data["image"].shape[:2]
                            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                            transformed_corners = cv2.perspectiveTransform(corners, M)

                            # Get bounding box
                            x_coords = transformed_corners[:, 0, 0]
                            y_coords = transformed_corners[:, 0, 1]
                            x, y = int(min(x_coords)), int(min(y_coords))
                            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                            best_bbox = (x, y, w, h)
                    except:
                        # Fallback: use centroid of matched points
                        matched_points = [frame_keypoints[m.trainIdx].pt for m in good_matches]
                        if matched_points:
                            x_coords = [p[0] for p in matched_points]
                            y_coords = [p[1] for p in matched_points]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)

                            # Add some padding
                            padding = 50
                            x = max(0, int(x_min - padding))
                            y = max(0, int(y_min - padding))
                            w = min(frame.shape[1] - x, int(x_max - x_min + 2 * padding))
                            h = min(frame.shape[0] - y, int(y_max - y_min + 2 * padding))
                            best_bbox = (x, y, w, h)

        return {
            "found": best_confidence > 0.3,  # Minimum confidence threshold
            "confidence": best_confidence,
            "bbox": best_bbox,
            "method_data": {
                "frame_features": len(frame_keypoints),
                "best_reference": best_match["name"] if best_match else None,
                "match_details": match_details
            }
        }

    def visualize_detection(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Visualize detection results on the frame.

        Args:
            frame: Original frame
            result: Detection result from find_player_in_frame

        Returns:
            Frame with detection visualization overlaid
        """
        vis_frame = frame.copy()

        if not result["found"]:
            # Draw "NOT FOUND" message
            cv2.putText(vis_frame, "Player not found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_frame

        # Draw bounding box if available
        if result["bbox"]:
            x, y, w, h = result["bbox"]
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw confidence
            conf_text = f"Player: {result['confidence']:.2f}"
            cv2.putText(vis_frame, conf_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Method-specific visualization
        if self.method == "color" and "mask" in result["method_data"]:
            # Show color mask in corner
            mask = result["method_data"]["mask"]
            mask_resized = cv2.resize(mask, (150, 100))
            mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

            h, w = vis_frame.shape[:2]
            vis_frame[10:110, w-160:w-10] = mask_colored
            cv2.putText(vis_frame, "Color Mask", (w-155, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif self.method == "reference":
            # Show match information
            method_data = result["method_data"]
            if method_data.get("best_reference"):
                ref_text = f"Best match: {method_data['best_reference']}"
                cv2.putText(vis_frame, ref_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            feature_text = f"Features: {method_data.get('frame_features', 0)}"
            cv2.putText(vis_frame, feature_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return vis_frame


def create_player_matcher(method: str, **kwargs) -> PlayerMatcher:
    """Factory function to create appropriate player matcher.

    Args:
        method: "color" or "reference"
        **kwargs: Method-specific arguments

    Returns:
        Configured PlayerMatcher instance
    """
    return PlayerMatcher(method=method, **kwargs)