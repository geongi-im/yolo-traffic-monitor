from pathlib import Path
from ultralytics import YOLO
import os

DATA_YAML = f"{os.path.dirname(os.path.abspath(__file__))}/dataset_v5/data.yaml"   # Roboflow data.yaml 경로
MODEL_WEIGHTS = "model/yolo11n.pt"
EPOCHS = 50
IMG_SIZE = 352
BATCH_SIZE = 4
DEVICE = "cpu"  # "cpu" or "0"
RUN_NAME = "yolo11n_v5"
PROJECT = "runs_v5"
# =========================================================

def check_data_yaml():
    data_yaml = Path(DATA_YAML)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

def main():
    check_data_yaml()

    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
    )

if __name__ == "__main__":
    main()
