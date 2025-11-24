#!/usr/bin/env python3
"""
YOLOv8 Detection using ffmpeg subprocess for camera access on macOS
"""

import subprocess
import numpy as np
import cv2
import serial
import time
import logging
import threading
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FFmpegCamera:
    def __init__(self, device_name="FaceTime HD Camera", width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        
        cmd = [
            'ffmpeg',
            '-f', 'avfoundation',
            '-framerate', str(fps),
            '-i', device_name,
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-f', 'rawvideo',
            'pipe:'
        ]
        
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 2
        )
        time.sleep(2)
        logger.info(f"Camera opened: {device_name}")
    
    def read(self):
        """Read a frame"""
        try:
            frame_data = self.proc.stdout.read(self.frame_size)
            if len(frame_data) != self.frame_size:
                return False, None
            frame = np.frombuffer(frame_data, np.uint8).reshape((self.height, self.width, 3))
            return True, frame
        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return False, None
    
    def release(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except:
            self.proc.kill()


class SerialBoard:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
    def connect(self):
        try:
            if self.port is None:
                import glob
                ports = glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*')
                if ports:
                    self.port = ports[0]
                else:
                    return False
            
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            logger.info(f"Connected to serial board on {self.port}")
            return True
        except Exception as e:
            logger.warning(f"Serial connection failed: {e}")
            return False
    
    def send_detection(self, detections):
        if self.ser is None or not self.ser.is_open:
            return
        
        try:
            if detections:
                message = "DETECT:" + ";".join([
                    f"{det['class']},{det['confidence']:.2f}"
                    for det in detections
                ]) + "\n"
            else:
                message = "DETECT:NONE\n"
            
            self.ser.write(message.encode())
        except:
            pass
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def main():
    logger.info("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    # Try to connect to serial board
    serial_board = SerialBoard()
    serial_board.connect()
    
    # Open camera via ffmpeg
    cap = FFmpegCamera()
    
    logger.info("Starting detection. Press 'q' to quit.")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Run detection
            results = model(frame, conf=0.5, verbose=False)
            
            # Parse detections
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = results[0].names[cls_id]
                    detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
            
            # Send to serial
            serial_board.send_detection(detections)
            
            # Display
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            
            # Log periodically
            frame_count += 1
            if frame_count % 30 == 0:
                if detections:
                    logger.info(f"Frame {frame_count}: {', '.join([f'{d['class']}({d['confidence']:.2f})' for d in detections])}")
                else:
                    logger.info(f"Frame {frame_count}: No detections")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        serial_board.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
