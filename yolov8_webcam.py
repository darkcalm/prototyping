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
import gc
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
            bufsize=self.frame_size
        )
        time.sleep(2)
        logger.info(f"Camera opened: {device_name}")
    
    def read(self):
        """Read a frame"""
        try:
            # Drain buffer to stay current (skip old frames)
            while True:
                frame_data = self.proc.stdout.read(self.frame_size)
                if len(frame_data) != self.frame_size:
                    return False, None
                # Check if more data available without blocking
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
                # Format: DETECT:class,conf,x1,y1,x2,y2;class,conf,x1,y1,x2,y2\n
                message = "DETECT:" + ";".join([
                    f"{det['class']},{det['confidence']:.2f},{det['x1']},{det['y1']},{det['x2']},{det['y2']}"
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
    model = YOLO("yolov8m.pt")  # Medium model for better accuracy
    model.to('mps')  # Use Metal Performance Shaders on Mac
    
    # Try to connect to serial board
    serial_board = SerialBoard()
    serial_board.connect()
    
    # Open camera via ffmpeg
    cap = FFmpegCamera(width=320, height=320)  # Minimal resolution for speed
    
    logger.info("Starting detection. Press 'q' to quit.")
    
    SKIP_FRAMES = 4  # Process every 4th frame (minimum latency)
    last_annotated = None
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_count += 1
            
            # Run detection on every frame
            if frame_count % SKIP_FRAMES == 0:
                results = model(frame, conf=0.4, verbose=False)
                
                # Parse detections
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = results[0].names[cls_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'x1': int(xyxy[0]),
                            'y1': int(xyxy[1]),
                            'x2': int(xyxy[2]),
                            'y2': int(xyxy[3])
                        })
                
                last_annotated = results[0].plot()
                
                # Send to serial
                serial_board.send_detection(detections)
                
                # Log periodically (every ~2 seconds at 30fps)
                if frame_count % 120 == 0:
                    if detections:
                        det_str = ', '.join([f"{d['class']}({d['confidence']:.2f}) [{d['x1']},{d['y1']},{d['x2']},{d['y2']}]" for d in detections])
                        logger.info(f"Frame {frame_count}: {det_str}")
                    else:
                        logger.info(f"Frame {frame_count}: No detections")
                    
                    # Flush memory every 4 seconds
                    gc.collect()
            
            # Display last result
            if last_annotated is not None:
                cv2.imshow("YOLOv8 Detection", last_annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        serial_board.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
