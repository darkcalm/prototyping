#!/usr/bin/env python3
"""
YOLOv8 Detection using ffmpeg subprocess for camera access on macOS
"""

import subprocess
import numpy as np
import cv2
import serial
import serial.tools.list_ports
import time
import logging
import threading
import gc
import argparse
import sys
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CV2Camera:
    """Standard OpenCV Camera for Windows/Linux compatibility"""
    def __init__(self, device_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {device_index}")
        else:
            logger.info(f"Camera opened: {device_index}")
            
    def read(self):
        return self.cap.read()
        
    def release(self):
        self.cap.release()


class FFmpegCamera:
    def __init__(self, device_name="FaceTime HD Camera", width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        
        if sys.platform == 'darwin':
            format_flag = 'avfoundation'
            input_device = device_name
        elif sys.platform == 'linux':
            format_flag = 'v4l2'
            input_device = '/dev/video0'
        else:
            # Fallback for Windows if used, though CV2Camera is preferred
            format_flag = 'dshow' 
            input_device = f'video={device_name}'

        cmd = [
            'ffmpeg',
            '-f', format_flag,
            '-framerate', str(fps),
            '-i', input_device,
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
    def __init__(self, port=None, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.last_command_time = 0
        self.command_cooldown = 2.5  # Seconds to wait between servo commands
        self.last_detection_time = 0
        self.detection_timeout = 5.0  # Return to center after 5 seconds of no detection
        
    def connect(self):
        try:
            if self.port is None:
                ports = [p.device for p in serial.tools.list_ports.comports()]
                if ports:
                    # Prefer the last one as it's often the newly plugged device
                    self.port = ports[-1]
                    logger.info(f"Found ports: {ports}, selecting {self.port}")
                else:
                    logger.warning("No serial ports found")
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
            
        # Check cooldown
        if time.time() - self.last_command_time < self.command_cooldown:
            return

        # Priority: Cup (11) > Cellphone (10)
        command = None
        target_detected = False
        
        for det in detections:
            label = det['class'].lower()
            if label == 'cup':
                command = "11"
                target_detected = True
                logger.info("Cup detected! Sending STAY RIGHT (11)")
                break
            elif label == 'cell phone':
                command = "10"
                target_detected = True
                logger.info("Cell phone detected! Sending STAY LEFT (10)")
                break
        
        if target_detected:
            self.last_detection_time = time.time()
            if command:
                try:
                    self.ser.write(f"{command}\n".encode())
                    self.last_command_time = time.time()
                except Exception as e:
                    logger.error(f"Failed to send serial command: {e}")
        else:
            # No cup/cell phone detected - check timeout
            if time.time() - self.last_detection_time > self.detection_timeout:
                if time.time() - self.last_command_time >= self.command_cooldown:
                    try:
                        self.ser.write("0\n".encode())
                        self.last_command_time = time.time()
                        logger.info("No cup/cell phone timeout - returning to CENTER (0)")
                    except Exception as e:
                        logger.error(f"Failed to send return-to-center command: {e}")
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Webcam Detection')
    parser.add_argument('--camera', type=str, default="0", help='Camera device index (0, 1) or RTSP URL')
    parser.add_argument('--width', type=int, default=320, help='Camera width (default: 320)')
    parser.add_argument('--height', type=int, default=320, help='Camera height (default: 320)')
    args = parser.parse_args()

    # Parse camera argument: if it's a digit, convert to int, otherwise keep as string (URL)
    if args.camera.isdigit():
        camera_source = int(args.camera)
    else:
        camera_source = args.camera

    logger.info("Loading YOLOv8 model...")
    model = YOLO("yolov8m.pt")  # Medium model for better accuracy
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Try to connect to serial board
    serial_board = SerialBoard()
    serial_board.connect()
    
    # Open camera
    # Use CV2Camera for Windows/Standard compatibility, FFmpegCamera for specific macOS low-latency needs if desired
    if sys.platform == 'win32':
        cap = CV2Camera(device_index=camera_source, width=args.width, height=args.height)
    else:
        # Default to FFmpegCamera for macOS/Linux as per original script
        try:
            if isinstance(camera_source, str):
                 # If it's a URL (RTSP), FFmpegCamera might need adjustments or just use CV2 for simplicity
                 # For now, let's force CV2 for network streams to be safe/compatible
                 cap = CV2Camera(device_index=camera_source, width=args.width, height=args.height)
            else:
                 cap = FFmpegCamera(width=args.width, height=args.height)
        except Exception as e:
            logger.warning(f"FFmpegCamera failed: {e}. Falling back to CV2Camera.")
            cap = CV2Camera(device_index=camera_source, width=args.width, height=args.height)
    
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
