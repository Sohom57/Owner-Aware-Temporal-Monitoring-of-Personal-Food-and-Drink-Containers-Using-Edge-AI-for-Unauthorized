import os
import cv2
import numpy as np
import threading
import time  
from insightface.app import FaceAnalysis
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceTracker:
    def __init__(self, user_image_path="user.jpg"):
        # 1. Initialize MediaPipe (Ultra-fast foreground tracking)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # 2. Initialize InsightFace 
        print("\nInitializing InsightFace (buffalo_l model)...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # Optimization: Reduced internal detection resolution for faster CPU execution
        self.app.prepare(ctx_id=0, det_size=(320, 320)) 
        
        # 3. State Management Variables
        self.user_embedding = None
        self.current_label = "Scanning..."
        self.current_color = (0, 255, 255)
        self.is_authenticated = False
        
        # Threading controls
        self.recognition_thread = None
        self.is_processing_identity = False
        
        # Security Cooldown controls
        self.last_auth_time = 0
        self.auth_cooldown = 3.0  # Seconds to wait before re-verifying an unauthenticated face

        # Load Master Identity
        if os.path.exists(user_image_path):
            print(f"Loading identity from {user_image_path}...")
            user_img = cv2.imread(user_image_path) 
            faces = self.app.get(user_img)
            if faces:
                self.user_embedding = faces[0].embedding
                print(f"[SUCCESS] Identity securely loaded into memory!")
            else:
                self.current_label = "BAD user.jpg"
                self.current_color = (150, 150, 150)
        else:
            print(f"[CRITICAL WARNING] '{user_image_path}' not found!")
            self.current_label = "NO user.jpg FOUND"
            self.current_color = (150, 150, 150)

    def _recognize_face(self, face_crop):
        """BACKGROUND TASK: Runs heavy math without freezing the webcam."""
        self.is_processing_identity = True
        try:
            detected_faces = self.app.get(face_crop)
                
            if detected_faces:
                current_embedding = detected_faces[0].embedding
                sim = np.dot(self.user_embedding, current_embedding) / (np.linalg.norm(self.user_embedding) * np.linalg.norm(current_embedding))
                
                if sim > 0.40:  # Threshold for InsightFace
                    self.current_label = f"Owner ({sim:.2f})"
                    self.current_color = (0, 255, 0)
                    self.is_authenticated = True
                else:
                    self.current_label = f"INTRUDER! ({sim:.2f})"
                    self.current_color = (0, 0, 255) # Red bounding box
                    self.is_authenticated = False
            else:
                self.current_label = "Scan Failed. Retrying..."
                self.current_color = (0, 165, 255) # Orange
                self.is_authenticated = False
        except Exception as e:
            print(f"Error in face recognition thread: {e}")
            self.current_label = "Scan Error"
            self.current_color = (0, 165, 255)
        finally:
            self.is_processing_identity = False

    def process_and_draw(self, img, clean_img, mp_image, timestamp_ms, frame_count):
        h, w, _ = img.shape
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        face_detected_this_frame = False
        current_time = time.time()

        if results.face_landmarks:
            for face_lms in results.face_landmarks:
                face_detected_this_frame = True
                
                # Draw Eye points
                left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
                for idx in left_eye + right_eye:
                    pt = face_lms[idx]
                    cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 2, (255, 255, 255), -1)

                # Calculate Bounding Box
                x_coords = [lm.x for lm in face_lms]
                y_coords = [lm.y for lm in face_lms]
                left = int((min(x_coords) - 0.1) * w)
                right = int((max(x_coords) + 0.1) * w)
                top = int((min(y_coords) - 0.25) * h)
                bottom = int((max(y_coords) + 0.1) * h)
                left, right = max(0, left), min(w - 1, right)
                top, bottom = max(0, top), min(h - 1, bottom)

                # --- EVENT-DRIVEN AUTHENTICATION ---
                # Trigger InsightFace if NOT authenticated AND NOT currently processing a thread
                if self.user_embedding is not None and not self.is_authenticated and not self.is_processing_identity:
                    # Enforce the 3-second cooldown before spamming the model again
                    if current_time - self.last_auth_time > self.auth_cooldown:
                        face_crop = clean_img[top:bottom, left:right]
                        
                        if face_crop.size != 0:
                            self.current_label = "Authenticating..."
                            self.current_color = (0, 255, 255)
                            self.last_auth_time = current_time # Update the timer
                            
                            # Dispatch the heavy task to a background thread
                            self.recognition_thread = threading.Thread(target=self._recognize_face, args=(face_crop,))
                            self.recognition_thread.start()
                
                # Draw the box and label seamlessly
                cv2.rectangle(img, (left, top), (right, bottom), self.current_color, 2)
                cv2.putText(img, self.current_label, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.current_color, 2, cv2.LINE_AA)

        # Reset security state if the person leaves the camera view
        if not face_detected_this_frame:
            self.is_authenticated = False
            self.last_auth_time = 0 # Reset instantly so the next face is scanned immediately
            if not self.is_processing_identity:
                self.current_label = "Scanning..."
                self.current_color = (0, 255, 255)

    def close(self):
        self.detector.close()
        if self.recognition_thread is not None and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1.0)