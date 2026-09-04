import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=6
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       
            (0, 5), (5, 6), (6, 7), (7, 8),       
            (5, 9), (9, 10), (10, 11), (11, 12),  
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
        ]

    def process_and_return(self, mp_image, timestamp_ms):
        h, w = mp_image.height, mp_image.width
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = []
        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                pixel_lms = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                hands_data.append(pixel_lms)
        return hands_data

    def close(self):
        self.detector.close()
