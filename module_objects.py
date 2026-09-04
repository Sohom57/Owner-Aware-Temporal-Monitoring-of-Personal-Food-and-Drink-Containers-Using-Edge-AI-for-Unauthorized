import cv2
import numpy as np
import onnxruntime as ort

class FoodDetector:
    def __init__(self, model_path="yolov8s.onnx"):
        print("Initializing YOLOv8 Small (ONNX)...")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # YOLOv8s pretrained COCO class IDs
        self.target_classes = {
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl"
        }

    def process_and_draw(self, img, clean_img):
        img_h, img_w = clean_img.shape[:2]

        rgb_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

        input_img = cv2.resize(rgb_img, (640, 640))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_img, axis=0)

        outputs = self.session.run(None, {self.input_name: input_tensor})[0]
        predictions = np.squeeze(outputs).T

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = (confidences > 0.50) & np.isin(class_ids, list(self.target_classes.keys()))
        filtered_boxes = boxes[mask]
        filtered_conf = confidences[mask]
        filtered_class_ids = class_ids[mask]

        x_factor = img_w / 640.0
        y_factor = img_h / 640.0

        nms_boxes = []
        nms_boxes_offset = []

        for i, row in enumerate(filtered_boxes):
            cx, cy, w, h = row
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            nms_boxes.append([left, top, width, height])

            offset = int(filtered_class_ids[i] * 4096)
            nms_boxes_offset.append([left + offset, top + offset, width, height])

        indices = cv2.dnn.NMSBoxes(nms_boxes_offset, filtered_conf.tolist(), 0.35, 0.45)

        detected_items = []

        if len(indices) > 0:
            for i in indices.flatten():
                box = nms_boxes[i]
                cls_id = filtered_class_ids[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                category = self.target_classes.get(cls_id, "unknown")

                if category in ["bottle", "cup", "bowl"]:
                    detected_items.append({"box": (x, y, w, h), "category": category})

        return detected_items

    def close(self):
        pass