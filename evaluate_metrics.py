import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from module_objects import FoodDetector


DATASET_CLASS_MAP = {
    # Cups (Coffee, Tea, etc.)
    10: "cup", 13: "cup", 17: "cup", 22: "cup", 63: "cup", 65: "cup", 
    131: "cup", 146: "cup", 156: "cup", 157: "cup", 186: "cup", 187: "cup", 
    225: "cup", 234: "cup", 247: "cup", 251: "cup", 253: "cup", 283: "cup", 285: "cup", 347: "cup", 385: "cup",
    
    # Bowls (Soups, Mixed Salads, Porridge, etc.)
    53: "bowl", 77: "bowl", 78: "bowl", 105: "bowl", 143: "bowl", 166: "bowl", 
    201: "bowl", 231: "bowl", 236: "bowl", 266: "bowl", 282: "bowl", 284: "bowl", 
    305: "bowl", 345: "bowl", 352: "bowl", 393: "bowl", 469: "bowl",
    
    # Bottles (Water, Juices, Beer, Wine)
    2: "bottle", 22: "bottle", 54: "bottle", 55: "bottle", 163: "bottle", 
    180: "bottle", 196: "bottle", 246: "bottle", 261: "bottle", 420: "bottle", 421: "bottle", 432: "bottle", 453: "bottle",

}

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two boxes [x, y, w, h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def read_ground_truth(label_path, img_w, img_h):
    """Parses YOLO normalized text files into absolute pixel coordinates [x, y, w, h]"""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            cls_id = int(parts[0])
            if cls_id not in DATASET_CLASS_MAP: continue
            
            category = DATASET_CLASS_MAP[cls_id]
            
            # Skip categories that aren't part of the target COCO list
            target_classes = ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"]
            if category not in target_classes:
                continue
                
            cx, cy, w_norm, h_norm = map(float, parts[1:])
            
            # Convert normalized YOLO format to absolute pixel coordinates
            w = int(w_norm * img_w)
            h = int(h_norm * img_h)
            x = int((cx * img_w) - (w / 2))
            y = int((cy * img_h) - (h / 2))
            
            gt_boxes.append({"box": [x, y, w, h], "category": category})
            
    return gt_boxes

def plot_confusion_matrix(metrics_dict):
    """Generates and saves a confusion matrix based on TP, FP, and FN."""
    # Filter out categories that didn't appear in the test set
    active_classes = [c for c, d in metrics_dict.items() if d["TP"] + d["FP"] + d["FN"] > 0]
    
    if not active_classes:
        print("[-] No data available to generate confusion matrix.")
        return

    # Matrix size: Active classes + 1 for Background
    N = len(active_classes)
    matrix = np.zeros((N + 1, N + 1), dtype=int)
    
    for i, cat in enumerate(active_classes):
        matrix[i, i] = metrics_dict[cat]["TP"]       # Correct Detections (True Positives)
        matrix[i, N] = metrics_dict[cat]["FN"]       # Missed Objects (False Negatives)
        matrix[N, i] = metrics_dict[cat]["FP"]       # False Alarms (False Positives)

    labels = [cat.capitalize() for cat in active_classes] + ["Background"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("Actual (Ground Truth) Label", fontsize=12, fontweight='bold')
    plt.title("Object Detection Confusion Matrix", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join("test_dataset", "confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[+] Confusion matrix successfully saved to: {output_path}")
    
    # Display the plot
    plt.show()

def run_evaluation(iou_threshold=0.50):
    food_detector = FoodDetector()
    
    images_dir = os.path.join("test_dataset", "images")
    labels_dir = os.path.join("test_dataset", "labels")
    
    target_classes = ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"]
    metrics = {cat: {"TP": 0, "FP": 0, "FN": 0} for cat in target_classes}
    
    if not os.path.exists(images_dir):
        print(f"[-] Directory not found: {images_dir}")
        return
        
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Starting evaluation on {len(image_files)} images...\n")
    
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_h, img_w = img.shape[:2]
        
        # 1. Get Ground Truth (Actual)
        gt_data = read_ground_truth(label_path, img_w, img_h)
        
        # 2. Get Predictions (Model Output)
        clean_img = img.copy()
        predictions = food_detector.process_and_draw(img, clean_img)
        
        # 3. Compare Predictions against Ground Truth per category
        for category in target_classes:
            gt_boxes = [g["box"] for g in gt_data if g["category"] == category]
            pred_boxes = [p["box"] for p in predictions if p["category"] == category]
            
            matched_gt_indices = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                # Find the ground truth box with the highest overlap
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt_indices: continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                # Classify the prediction based on the IoU threshold
                if best_iou >= iou_threshold:
                    metrics[category]["TP"] += 1
                    matched_gt_indices.add(best_gt_idx)
                else:
                    metrics[category]["FP"] += 1
                    
            # Any ground truth box that wasn't matched is a False Negative
            fn_count = len(gt_boxes) - len(matched_gt_indices)
            metrics[category]["FN"] += fn_count

    # 4. Calculate Final Metrics (Accuracy, Precision, Recall, F1-Score)
    print("-" * 65)
    print(f"{'Category':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    total_tp = total_fp = total_fn = 0
    
    for cat, data in metrics.items():
        tp = data["TP"]
        fp = data["FP"]
        fn = data["FN"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Calculate Accuracy for Object Detection (Detection Rate)
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Only print categories that actually appeared in the test set or predictions
        if tp + fp + fn > 0:
            print(f"{cat.capitalize():<15} | {accuracy:.4f}     | {precision:.4f}     | {recall:.4f}    | {f1:.4f}")
            
    print("-" * 65)
    
    # Calculate Global Metrics
    global_acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    
    print(f"{'OVERALL':<15} | {global_acc:.4f}     | {global_precision:.4f}     | {global_recall:.4f}    | {global_f1:.4f}")
    print("-" * 65)

    # 5. Generate and display the confusion matrix
    plot_confusion_matrix(metrics)

if __name__ == "__main__":
    run_evaluation(iou_threshold=0.50)