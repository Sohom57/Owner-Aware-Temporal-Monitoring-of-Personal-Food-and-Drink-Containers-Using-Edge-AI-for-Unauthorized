import asyncio
import os
import time
from collections import deque

import cv2
import mediapipe as mp
from telegram import Bot

from utils_download import download_models
from module_hands import HandTracker
from module_faces import FaceTracker
from module_objects import FoodDetector
from module_metrics import PerformanceEvaluator
from module_roi_segmentation import ROISegmenter
from telegram.request import HTTPXRequest

# --- Telegram Configuration ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_alert_sync(image_path, video_path):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[-] Telegram credentials are not configured; skipping alert.")
        return

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Configure request timeouts (set to 30 seconds)
        t_request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
        bot = Bot(token=TELEGRAM_TOKEN, request=t_request)

        loop.run_until_complete(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚨 ALERT: Unauthorized Tampering Detected!"))

        with open(image_path, "rb") as img:
            loop.run_until_complete(bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img))

        with open(video_path, "rb") as vid:
            loop.run_until_complete(bot.send_video(chat_id=TELEGRAM_CHAT_ID, video=vid, caption="Event Buffer (15s)"))

    except Exception as e:
        print(f"[-] Failed to send Telegram alert: {e}")
        
def main():
    download_models()

    print("Initializing AI Classes...")
    metrics_tracker = PerformanceEvaluator()
    hand_tracker = HandTracker()
    face_tracker = FaceTracker("user.jpg")
    food_tracker = FoodDetector()
    roi_segmenter = ROISegmenter()

    evidence_dir = "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    print("Opening camera...")

    # macOS built-in webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[-] Failed to open Mac webcam")
        return

    fps = 30
    video_buffer = deque(maxlen=15 * fps)

    MIRROR_WEBCAM = False
    last_timestamp_ms = 0
    frame_count = 0
    any_tampering_detected = False
    tampering_alert_sent = False

    print("Pipeline running! Press 'q' to quit.")

    while cap.isOpened():
        loop_start_time = time.time()

        success, img = cap.read()
        if not success:
            continue

        video_buffer.append(img.copy())

        if MIRROR_WEBCAM:
            img = cv2.flip(img, 1)

        clean_img = img.copy()
        img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        detected_food = food_tracker.process_and_draw(img, clean_img)
        if detected_food is None:
            detected_food = []

        # Security pipeline should only start after a valid food item is detected.
        # If no object is present, skip face authentication and hand analysis entirely.
        if not detected_food:
            metrics_tracker.update_and_draw(img, loop_start_time)
            cv2.imshow("Owner-Aware Security Pipeline", img)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face_tracker.process_and_draw(img, clean_img, mp_image, timestamp_ms, frame_count)
        user_is_authenticated = face_tracker.is_authenticated
        face_label = face_tracker.current_label

        hands_data = hand_tracker.process_and_return(mp_image, timestamp_ms)

        # Extract ROIs for objects
        object_rois = roi_segmenter.extract_object_roi(img, detected_food)

        # Tampering Logic with ROI-based checks
        for food in detected_food:
            x, y, w, h = food["box"]
            category = food["category"]

            rx1, ry1 = x, y
            rx2, ry2 = x + w, y + h
            object_bbox = (rx1, ry1, rx2, ry2)

            hand_in_object = False
            tampering_threat_level = "NONE"  # NONE, SUSPICIOUS, TAMPERING

            # Check if any hand is inside the object box
            for hand_lms in hands_data:
                if roi_segmenter.check_hand_object_interaction(hand_lms, object_bbox):
                    hand_in_object = True
                    break

            # Determine threat level based on authentication status
            if hand_in_object:
                if user_is_authenticated:
                    # Owner is touching the object - allowed, no alert
                    tampering_threat_level = "OWNER_TOUCH"
                elif "Owner" in face_label:
                    # Face recognized as owner - allow touch
                    tampering_threat_level = "OWNER_TOUCH"
                elif "Scanning" in face_label or "Authenticating" in face_label:
                    # Face detection in progress, wait for result before alerting
                    tampering_threat_level = "SUSPICIOUS"
                elif "INTRUDER" in face_label:
                    # Intruder detected and touching object - critical alert
                    tampering_threat_level = "TAMPERING"
                else:
                    # No face detected but hand is on object - suspicious
                    tampering_threat_level = "SUSPICIOUS"

            # Draw object box with appropriate color based on threat level
            if tampering_threat_level == "TAMPERING":
                # Critical alert - intruder touching
                box_color = (0, 0, 255)  # Red
                box_text = f"WARNING: {category.upper()} TAMPERING!"
                metrics_tracker.log_tamper_event(category, "YES", alert_type="INTRUDER_TOUCH")

                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), box_color, 4)
                cv2.putText(img, box_text, (rx1, ry1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)

                any_tampering_detected = True

                # Send alert only once per session
                if not tampering_alert_sent:
                    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

                    snapshot_path = os.path.join(evidence_dir, f"tamper_snapshot_{timestamp_str}.png")
                    cv2.imwrite(snapshot_path, img)
                    print(f"[!] Tamper snapshot saved to {snapshot_path}")

                    video_output_path = os.path.join(evidence_dir, f"tamper_buffer_{timestamp_str}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    actual_fps = max(1.0, float(metrics_tracker.fps))
                    out_video = cv2.VideoWriter(video_output_path, fourcc, actual_fps, (1280, 720))

                    frames_needed = int(15 * actual_fps)
                    frames_to_save = list(video_buffer)[-frames_needed:] if len(video_buffer) > frames_needed else video_buffer

                    for frame in frames_to_save:
                        out_video.write(frame)
                    out_video.release()

                    print(f"[!] Tamper video buffer saved to {video_output_path}")

                    send_telegram_alert_sync(snapshot_path, video_output_path)
                    tampering_alert_sent = True

            elif tampering_threat_level == "SUSPICIOUS":
                # Suspicious activity - hand on object but face not yet authenticated
                box_color = (0, 165, 255)  # Orange
                box_text = f"PROXIMITY WARNING: {category.upper()}"

                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), box_color, 3)
                cv2.putText(img, box_text, (rx1, ry1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

            elif tampering_threat_level == "OWNER_TOUCH":
                # Owner is touching - secure state
                box_color = (0, 255, 0)  # Green
                box_text = f"{category.upper()} (OWNER - Secure)"

                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), box_color, 2)
                cv2.putText(img, box_text, (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

            else:
                # No hand interaction - normal state
                box_color = (255, 100, 0)  # Blue-orange
                box_text = f"{category.upper()}"

                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), box_color, 2)
                cv2.putText(img, box_text, (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

        for hand_lms in hands_data:
            for connection in hand_tracker.connections:
                start_idx, end_idx = connection
                if start_idx < len(hand_lms) and end_idx < len(hand_lms):
                    cv2.line(img, hand_lms[start_idx], hand_lms[end_idx], (0, 255, 0), 2)

            key_points = [0, 4, 8, 12, 16, 20]
            for idx in key_points:
                if idx < len(hand_lms):
                    cv2.circle(img, hand_lms[idx], 6, (0, 255, 255), -1)

        metrics_tracker.update_and_draw(img, loop_start_time)

        cv2.imshow("Owner-Aware Security Pipeline", img)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    metrics_tracker.finalize_test_log(any_tampering_detected)

    hand_tracker.close()
    face_tracker.close()
    food_tracker.close()
    roi_segmenter.close()
    metrics_tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()