import os
import cv2
import time
from module_objects import FoodDetector
from module_metrics import PerformanceEvaluator

def run_dataset_test():
    # 1. Initialize your existing system components
    food_detector = FoodDetector()
    metrics_tracker = PerformanceEvaluator(
        log_dir="dataset-test-logs",
        log_file="dataset_hardware_metrics.csv",
        tamper_file="dataset_events.csv"
    )
    
    input_folder = "test_dataset/images"
    output_folder = "test_dataset/results"
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"[-] Error: '{input_folder}' directory not found. Please create it and add images.")
        return
        
    # Gather all images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    
    print(f"[+] Found {len(images)} images. Starting benchmark test...")
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        clean_img = img.copy()
        start_time = time.time()
        
        # 2. Feed the dataset image into your existing detection pipeline
        detected_items = food_detector.process_and_draw(img, clean_img)
        
        # 3. Draw bounding boxes based on predictions (Mimicking main.py behavior)
        for item in detected_items:
            x, y, w, h = item["box"]
            category = item["category"]
            
            # Draw visual box and label on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 0), 2)
            cv2.putText(img, f"{category.upper()}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            
            # Log the detected item categorized by class into your CSV file
            metrics_tracker.log_tamper_event(category, tampered_status="DETECTED")
            
        # 4. Update processing and hardware speed statistics
        metrics_tracker.update_and_draw(img, start_time)
        
        # 5. Save the processed output images to show your teacher
        output_path = os.path.join(output_folder, f"evaluated_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"[{idx + 1}/{len(images)}] Evaluated and saved: {img_name}")
        
    # 6. Generate final CSV performance report summary
    metrics_tracker.finalize_test_log(tampered_status=True)
    print("\n[SUCCESS] Dataset testing complete!")
    print(f"[->] Visual results saved to: {output_folder}")
    print(f"[->] Performance CSV records saved to: dataset-test-logs/")

if __name__ == "__main__":
    run_dataset_test()