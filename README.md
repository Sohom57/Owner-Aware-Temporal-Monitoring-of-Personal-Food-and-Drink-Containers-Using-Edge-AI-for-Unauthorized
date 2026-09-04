# Owner-Aware Temporal Monitoring of Personal Food and Drink Containers Using Edge AI for Unauthorized Interaction Detection

This project is a real-time, edge-oriented computer-vision pipeline for detecting possible unauthorized interactions with personal food and drink containers such as bottles, cups, and bowls. It combines object detection, face authentication, hand landmark tracking, region-of-interest (ROI) analysis, event logging, evidence capture, and optional Telegram notifications.

The system is designed for thesis research and controlled demonstrations. It is not a production security system and should not be used as the sole means of protecting people or property.

## Features

- Real-time webcam monitoring with OpenCV.
- YOLOv8 ONNX detection for bottles, cups, and bowls.
- MediaPipe face landmarks and hand landmarks.
- InsightFace embedding comparison against a locally supplied owner image.
- Object-detection gating: face and hand processing is skipped when no target object is visible.
- ROI-based hand/object interaction checks.
- Threat states for normal activity, owner touch, suspicious activity during authentication, and intruder touch.
- Optional Telegram alerts with a snapshot and a buffered video clip.
- CSV event and performance logs, including CPU/RAM/GPU information when available.
- Dataset benchmark and IoU-based evaluation utilities.

## Pipeline Overview

1. Capture a frame from the webcam.
2. Detect target objects with the YOLOv8 ONNX model.
3. If a target object is present, run face landmarking/authentication and hand landmarking.
4. Test whether hand landmarks enter an object bounding box.
5. Classify the interaction using the current authentication state.
6. Draw annotations, update performance metrics, and write logs.
7. For an intruder touch, save evidence and optionally send a Telegram alert once per session.

## Requirements

- Python 3.8 or newer.
- A webcam for real-time monitoring.
- Windows, macOS, or Linux. The default camera backend is `cv2.CAP_DSHOW`, which is most appropriate for Windows; change it in `main.py` for another platform if required.
- Several gigabytes of free disk space for Python packages and model assets.
- A CPU capable of running ONNX Runtime and InsightFace. A compatible GPU may help with training, but the runtime pipeline uses CPU providers by default.

## Installation

Create and activate a virtual environment, then install the runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The repository contains some model binaries, but `utils_download.py` can download the MediaPipe assets that are missing:

```bash
python -c "from utils_download import download_models; download_models()"
```

`main.py` expects these files in the project root:

- `hand_landmarker.task`
- `face_landmarker.task`
- `yolov8s.onnx`
- `user.jpg`

The owner image is intentionally ignored by Git. Add your own private `user.jpg`; do not commit it.

## Configuration

Copy the example configuration file and set values only in your local environment:

```powershell
Copy-Item .env.example .env
$env:TELEGRAM_TOKEN = "your-bot-token"
$env:TELEGRAM_CHAT_ID = "your-chat-id"
```

The application reads `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` when sending alerts. If they are absent, the pipeline logs that Telegram is not configured and continues without sending an alert.

`ROBOFLOW_API_KEY` is needed only by `data_build.py` to download the training dataset. Never put API keys in source files or notebooks.

## Running the Application

Start the real-time monitoring pipeline:

```bash
python main.py
```

Press `q` in the OpenCV window to stop. The pipeline creates runtime outputs in `evidence/` and `data-logs/`; these directories are ignored because their contents are machine- and session-specific.

### Threat states

| State | Meaning | Expected action |
| --- | --- | --- |
| Normal | A target object is detected without a hand interaction. | Blue/orange object annotation; no alert. |
| Owner touch | A recognized owner hand interacts with an object. | Green annotation; no alert. |
| Suspicious | A hand interacts while face authentication is pending or unavailable. | Orange annotation; wait for authentication. |
| Intruder touch | An unauthenticated or intruder face interacts with an object. | Red annotation, evidence capture, and optional Telegram alert. |

## Testing and Evaluation

Run the lightweight sequence-logic test:

```bash
python test_sequence_logic.py
```

Run the image benchmark after creating `test_dataset/images` and adding matching images:

```bash
python test_pipeline.py
```

The benchmark writes annotated images to `test_dataset/results` and metrics to `dataset-test-logs/`. For ground-truth evaluation, provide YOLO label files in `test_dataset/labels` and run:

```bash
python evaluate_metrics.py
```

The evaluator reports per-class and overall precision, recall, F1, and detection-rate-style accuracy at an IoU threshold of 0.50. It can also generate a confusion matrix.

The notebooks contain extended implementation notes, experiments, and demonstration cells:

- `Complete.ipynb`
- `Complete_Project.ipynb`
- `Project_Report_and_Implementation.ipynb`

Notebook cells may assume local model files, a webcam, or a dataset that is not included in this repository.

## Training and Model Export

`data_build.py` downloads a Roboflow dataset, trains a YOLO model, and exports it to ONNX. Configure the API key first:

```powershell
$env:ROBOFLOW_API_KEY = "your-roboflow-key"
python data_build.py
```

Training requires the additional `roboflow` and `ultralytics` packages and is generally best run on a machine with a supported GPU. Training outputs are written under `runs/` and are ignored by Git.

## Repository Layout

```text
ezyZip/
├── main.py                         # Real-time orchestration and alert flow
├── module_objects.py                # YOLOv8 ONNX object detector
├── module_faces.py                  # MediaPipe landmarks and InsightFace auth
├── module_hands.py                  # MediaPipe hand landmark tracker
├── module_roi_segmentation.py       # ROI extraction and interaction geometry
├── module_metrics.py                # Performance and tamper CSV logging
├── utils_download.py                # MediaPipe model download helper
├── data_build.py                    # Roboflow download and YOLO training
├── test_pipeline.py                 # Dataset inference benchmark
├── test_sequence_logic.py           # Object-gating behavior test
├── evaluate_metrics.py              # IoU and detection metric evaluation
├── requirements.txt                 # Python dependencies
├── .env.example                     # Local configuration template
├── LICENSE                           # MIT license
└── *.ipynb                           # Reports, experiments, and implementation notes
```

Model binaries may be distributed separately from the source because of their size and their upstream terms. Review the license and usage terms for YOLO, MediaPipe, InsightFace, ONNX Runtime, and any downloaded dataset before redistribution or commercial use.

## Privacy and Security

The face-authentication workflow processes a reference face image and camera frames. Keep `user.jpg`, captured evidence, logs, API keys, and Telegram credentials private. The default `.gitignore` excludes these local artifacts, but always inspect `git status` before publishing.

Previously exposed credentials should be revoked and replaced at their providers. Environment variables prevent new accidental exposure but do not invalidate credentials that were already shared.

## Known Limitations

- Authentication depends on lighting, camera quality, face pose, and the quality of `user.jpg`.
- The interaction rule is based on hand landmarks entering an object bounding box; this can produce false positives or false negatives.
- The runtime detector uses a fixed 640x640 input and CPU ONNX inference.
- The default camera backend and some diagnostic commands are platform-specific.
- Telegram delivery requires network access and correctly configured credentials.
- The included tests do not exercise the full webcam, model, Telegram, or hardware path.
- Reported FPS, memory, and accuracy values depend on the device, model, dataset, and configuration; benchmark them on your own environment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
