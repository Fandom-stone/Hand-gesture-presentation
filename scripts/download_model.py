"""
Download the pre-trained MediaPipe HandLandmarker model bundle.

This model is provided directly by Google (no training required for this
step - see Section 8.2 of the project documentation). It is ~5-9 MB and is
NOT bundled in the repo; run this once per machine before using any other
script in this project.

Usage:
    python scripts/download_model.py
"""

import sys
import urllib.request
from pathlib import Path

# Official Google model bundle URLs (MediaPipe Tasks, Hand Landmarker).
# "float16" is the smaller/faster variant recommended for CPU real-time use;
# "float32" is slightly larger/more precise if you ever need it instead.
MODEL_URLS = {
    "float16": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "float32": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float32/1/hand_landmarker.task",
}

DEST = Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"


def main(variant: str = "float16") -> None:
    url = MODEL_URLS[variant]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {DEST}")
    try:
        urllib.request.urlretrieve(url, DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"\nDownload failed: {exc}")
        print(
            "If your network blocks storage.googleapis.com, download the file "
            "manually from the URL above (or from "
            "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) "
            f"on any machine with internet access and place it at:\n  {DEST}"
        )
        sys.exit(1)

    size_kb = DEST.stat().st_size / 1024
    print(f"Done. Saved {size_kb:.0f} KB to {DEST}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "float16"
    main(variant)
