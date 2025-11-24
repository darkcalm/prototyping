#!/bin/bash
# Activate virtual environment and run YOLOv8 detection

cd "$(dirname "$0")"
source venv/bin/activate
python3 yolov8_webcam.py
