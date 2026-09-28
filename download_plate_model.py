"""
One-off helper: downloads a pretrained YOLOv8 license-plate detector so you
have something in models/plate_yolov8.pt to get started with immediately.

Important: this is a GENERIC model (trained on ~500 general vehicle images,
not Nigerian plates specifically). Treat it as a baseline to validate the
rest of the pipeline against -- once you confirm detection quality on your
own footage, plan to fine-tune on local plate samples (see README.md).

Usage:
    pip install huggingface_hub --break-system-packages
    python download_plate_model.py
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Koushim/yolov8-license-plate-detection"
FILENAME = "best.pt"
DEST = Path("models/plate_yolov8.pt")


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    shutil.copy(cached_path, DEST)
    print(f"Saved to {DEST.resolve()}")
    print("This is a generic baseline model -- see README.md's 'You still")
    print("need to provide' section for fine-tuning notes.")


if __name__ == "__main__":
    main()
