import os
import urllib.request

def download_models():
    """Downloads AI models directly to the root folder."""
    models = {
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "efficientdet.tflite": "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite"
    }
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}... Please wait.")
            urllib.request.urlretrieve(url, filename)