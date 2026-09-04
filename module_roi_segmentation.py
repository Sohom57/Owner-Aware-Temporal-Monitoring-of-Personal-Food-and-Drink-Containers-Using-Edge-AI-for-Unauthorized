import cv2
import numpy as np


class ROISegmenter:
    """
    Preprocesses captured images to identify and segment regions of interest (ROI):
    - Object bounding boxes (food items)
    - Face bounding boxes
    Creates segmented ROI crops for focused analysis.
    """

    def __init__(self):
        self.object_rois = []
        self.face_rois = []

    def extract_object_roi(self, img, food_detections):
        """
        Extract and segment object regions of interest from detected food items.
        
        Args:
            img: Input image (BGR)
            food_detections: List of detected food items with boxes
            
        Returns:
            List of dict with ROI crops, boxes, and metadata
        """
        self.object_rois = []
        img_h, img_w = img.shape[:2]

        for food in food_detections:
            x, y, w, h = food["box"]
            category = food["category"]

            # Ensure box is within image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)

            # Extract ROI crop
            roi_crop = img[y1:y2, x1:x2].copy()

            # Store ROI metadata
            self.object_rois.append({
                "box": (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h)
                "bbox": (x1, y1, x2, y2),  # (x1, y1, x2, y2)
                "category": category,
                "crop": roi_crop,
                "area": (x2 - x1) * (y2 - y1)
            })

        return self.object_rois

    def extract_face_roi(self, img, face_bboxes):
        """
        Extract and segment face regions of interest.
        
        Args:
            img: Input image (BGR)
            face_bboxes: List of face bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            List of dict with face ROI crops and boxes
        """
        self.face_rois = []
        img_h, img_w = img.shape[:2]

        for i, (x1, y1, x2, y2) in enumerate(face_bboxes):
            # Ensure box is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Extract ROI crop
            roi_crop = img[y1:y2, x1:x2].copy()

            # Store ROI metadata
            self.face_rois.append({
                "bbox": (x1, y1, x2, y2),
                "crop": roi_crop,
                "area": (x2 - x1) * (y2 - y1),
                "face_id": i
            })

        return self.face_rois

    def check_hand_object_interaction(self, hand_landmarks, object_roi_box):
        """
        Check if any hand landmark is inside the object ROI.
        
        Args:
            hand_landmarks: List of (x, y) tuples for hand points
            object_roi_box: (x1, y1, x2, y2) bounding box
            
        Returns:
            bool: True if hand is inside object box
        """
        x1, y1, x2, y2 = object_roi_box

        for pt_x, pt_y in hand_landmarks:
            if x1 < pt_x < x2 and y1 < pt_y < y2:
                return True

        return False

    def check_face_object_overlap(self, face_roi_box, object_roi_box, overlap_threshold=0.3):
        """
        Check if face and object ROIs overlap significantly.
        Used to exclude false tampering alerts when owner's head is near the object.
        
        Args:
            face_roi_box: (x1, y1, x2, y2) face bounding box
            object_roi_box: (x1, y1, x2, y2) object bounding box
            overlap_threshold: Overlap ratio threshold (0.0 - 1.0)
            
        Returns:
            float: Overlap ratio (0.0 to 1.0)
        """
        fx1, fy1, fx2, fy2 = face_roi_box
        ox1, oy1, ox2, oy2 = object_roi_box

        # Calculate intersection
        inter_x1 = max(fx1, ox1)
        inter_y1 = max(fy1, oy1)
        inter_x2 = min(fx2, ox2)
        inter_y2 = min(fy2, oy2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        face_area = (fx2 - fx1) * (fy2 - fy1)

        overlap_ratio = intersection_area / face_area if face_area > 0 else 0.0

        return overlap_ratio

    def get_hand_distance_to_object(self, hand_landmarks, object_roi_box):
        """
        Calculate minimum distance from hand to object ROI.
        
        Args:
            hand_landmarks: List of (x, y) tuples
            object_roi_box: (x1, y1, x2, y2)
            
        Returns:
            float: Minimum distance in pixels (0 if inside box)
        """
        x1, y1, x2, y2 = object_roi_box

        min_distance = float('inf')

        for pt_x, pt_y in hand_landmarks:
            # If inside box, distance is 0
            if x1 < pt_x < x2 and y1 < pt_y < y2:
                return 0.0

            # Distance to nearest edge
            dx = max(x1 - pt_x, 0, pt_x - x2)
            dy = max(y1 - pt_y, 0, pt_y - y2)
            distance = np.sqrt(dx ** 2 + dy ** 2)

            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else float('inf')

    def draw_roi_segmentation(self, img, object_rois, face_rois, alpha=0.3):
        """
        Draw ROI bounding boxes on image for visualization.
        
        Args:
            img: Input image to draw on
            object_rois: List of object ROI dicts
            face_rois: List of face ROI dicts
            alpha: Transparency for overlay
            
        Returns:
            Modified image with ROI boxes drawn
        """
        overlay = img.copy()

        # Draw face ROIs in green
        for face_roi in face_rois:
            x1, y1, x2, y2 = face_roi["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.putText(
                overlay, f"Face-{face_roi['face_id']}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Draw object ROIs in blue
        for obj_roi in object_rois:
            x1, y1, x2, y2 = obj_roi["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
            cv2.putText(
                overlay, obj_roi["category"], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

        # Blend overlay with original
        result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return result

    def close(self):
        """Clean up resources."""
        self.object_rois = []
        self.face_rois = []
