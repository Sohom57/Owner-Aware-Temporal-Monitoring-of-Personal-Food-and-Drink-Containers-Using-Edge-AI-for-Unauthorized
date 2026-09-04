import csv
import os
import platform
import subprocess
import time
from datetime import datetime

import cv2
import psutil

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class PerformanceEvaluator:
    def __init__(self, log_dir="data-logs", spec_file="hardware_specs.txt", log_file="hardware_logs.csv", tamper_file="tamper_events.csv"):
        self.log_dir = log_dir
        self.spec_path = os.path.join(log_dir, spec_file)
        self.log_path = os.path.join(log_dir, log_file)
        self.tamper_path = os.path.join(log_dir, tamper_file)

        os.makedirs(self.log_dir, exist_ok=True)

        self.fps_list = []
        self.fps = 0.0
        self.max_ram_used_gb = 0.0
        self.gpu_name = "N/A"
        self.max_gpu_load = 0.0
        self.max_gpu_mem = 0.0

        self._setup_tamper_csv()
        self._generate_specs_file()

    def _get_processor_name(self):
        try:
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Darwin":
                command = "sysctl -n machdep.cpu.brand_string"
                return subprocess.check_output(command, shell=True).decode().strip()
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo | grep 'model name' | uniq"
                return subprocess.check_output(command, shell=True).decode().split(':')[1].strip()
        except Exception:
            return platform.machine()
        return "Unknown Processor"

    def _get_device_model(self):
        try:
            if platform.system() == "Windows":
                cmd = "wmic csproduct get name"
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode(errors="ignore")
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                if len(lines) > 1:
                    return lines[1]
            elif platform.system() == "Linux":
                if os.path.exists("/sys/class/dmi/id/product_name"):
                    with open("/sys/class/dmi/id/product_name", "r") as f:
                        return f.read().strip()
            elif platform.system() == "Darwin":
                cmd = "sysctl -n hw.model"
                return subprocess.check_output(cmd, shell=True).decode().strip()
        except Exception:
            pass
        return "Unknown Model"

    def _generate_specs_file(self):
        device_name = platform.node()
        device_model = self._get_device_model()
        os_platform = f"{platform.system()} {platform.release()}"
        processor = self._get_processor_name()
        cpu_cores = psutil.cpu_count(logical=True)
        total_memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

        gpu_format = "N/A"
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_format = gpus[0].name
                    self.gpu_name = gpu_format
            except Exception:
                pass

        with open(self.spec_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- New Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Device Name: {device_name}\n")
            f.write(f"Device Model: {device_model}\n")
            f.write(f"Os Platform: {os_platform}\n")
            f.write(f"Processor: {processor}\n")
            f.write(f"CPU with Cores: {processor} ({cpu_cores} cores)\n")
            f.write(f"Total Memory: {total_memory_gb} GB\n")
            f.write(f"GPU: {gpu_format}\n")
            f.write(f"Total RAM: {total_memory_gb} GB\n")
            f.write(f"CPU: {processor}\n")

    def _setup_tamper_csv(self):
        if not os.path.isfile(self.tamper_path):
            with open(self.tamper_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Tamper Timestamp", "Target Object Category", "Tampered or not", "Alert Type"])

    def update_and_draw(self, img, loop_start_time):
        current_time = time.time()
        time_diff = current_time - loop_start_time

        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.fps = (self.fps * 0.9) + (current_fps * 0.1)

        self.fps_list.append(self.fps)

        ram_used_gb = round(psutil.virtual_memory().used / (1024 ** 3), 2)
        if ram_used_gb > self.max_ram_used_gb:
            self.max_ram_used_gb = ram_used_gb

        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_name = gpu.name
                    if gpu.load * 100 > self.max_gpu_load:
                        self.max_gpu_load = round(gpu.load * 100, 1)
                    if gpu.memoryUsed > self.max_gpu_mem:
                        self.max_gpu_mem = gpu.memoryUsed
            except Exception:
                pass

        cv2.putText(
            img,
            f"FPS: {int(self.fps)}",
            (img.shape[1] - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img

    def log_tamper_event(self, category, tampered_status="YES", alert_type="Direct Touch"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.tamper_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, category, tampered_status, alert_type])

    def finalize_test_log(self, tampered_status):
        if self.fps_list:
            avg_fps = round(sum(self.fps_list) / len(self.fps_list), 1)
            best_fps = round(max(self.fps_list), 1)
            lowest_fps = round(min(self.fps_list), 1)
        else:
            avg_fps, best_fps, lowest_fps = 0.0, 0.0, 0.0

        tampered_str = "YES" if tampered_status else "NO"

        file_exists = os.path.isfile(self.log_path)
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Average FPS",
                    "Best FPS",
                    "Lowest FPS",
                    "Tampered or not",
                    "Ram used (GB)",
                    "GPU_Name",
                    "GPU Load (%)",
                    "GPU memory used (MB)",
                ])

            writer.writerow([
                avg_fps,
                best_fps,
                lowest_fps,
                tampered_str,
                f"{self.max_ram_used_gb} GB",
                self.gpu_name,
                f"{self.max_gpu_load}%",
                f"{self.max_gpu_mem} MB",
            ])

    def close(self):
        pass