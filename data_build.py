import os

from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == '__main__':
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not roboflow_api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY before downloading the training dataset.")

    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace("mds-workspace-23fmt").project("combined_cup_bottle_lunchbox")
    version = project.version(2)
    dataset = version.download("yolov8")

    model = YOLO("yolov8n.pt")
    
    model.train(data=f"{dataset.location}/data.yaml", epochs=50, imgsz=640, device=0, batch=16)
    
    model.export(format="onnx", imgsz=640)