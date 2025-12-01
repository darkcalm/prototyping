#!/usr/bin/env python3
"""
YOLO Detection using ffmpeg subprocess for camera access on macOS
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
        self.command_cooldown = 0.5  # Seconds to wait between servo commands
        self.last_detection_time = 0
        self.detection_timeout = 5.0  # Return to center after 5 seconds of no detection
        self.cup_side = "10"  # Cup waste goes to LEFT (10) - change to "11" for RIGHT
        self.clamshell_side = "11"  # Clamshell waste goes to RIGHT (11) - change to "10" for LEFT
        self.blocker_blocks_clamshell = True  # Blocker purpose: True=clamshell, False=cups
        
        # State tracking for multi-step logic
        self.step = 0  # 0=idle, 1=blocking, 2=guiding_first, 3=unblocking, 4=guiding_second
        
        # Time-based averaging for stable signal
        self.detection_history = []  # [(timestamp, cup_left, cup_right, clam_left, clam_right)]
        self.averaging_window = 0.5  # seconds
        
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
    

    
    def get_object_side(self, det):
        """Determine if object is on left or right side of frame"""
        # Calculate center x of bounding box
        center_x = (det['x1'] + det['x2']) / 2
        # Assume frame width is 320 (from default args)
        frame_width = 320
        return 'left' if center_x < frame_width / 2 else 'right'
    
    def is_clamshell(self, det):
        """Detect clamshell by label or aspect ratio (excludes person)"""
        label = det['class'].lower()
        
        # Exclude person
        if label == 'person':
            return False
        
        width = det['x2'] - det['x1']
        height = det['y2'] - det['y1']
        
        # List of COCO labels that are clamshell-shaped
        clamshell_labels = ['cell phone', 'remote', 'spoon', 'book', 'laptop', 
                           'keyboard', 'mouse', 'credit card', 'passport']
        
        # Check if label is clamshell-like
        if any(clamshell_label in label for clamshell_label in clamshell_labels):
            return True
        
        # Also detect by aspect ratio (width > 1.5x height indicates flat/clamshell shape)
        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio > 1.5:
                return True
        
        return False
    
    def send_detection(self, detections):
        if self.ser is None or not self.ser.is_open:
            return
            
        # Check cooldown
        if time.time() - self.last_command_time < self.command_cooldown:
            return

        command = None
        target_detected = False
        cup_on_left = False
        cup_on_right = False
        clamshell_on_left = False
        clamshell_on_right = False
        
        # Waste separation logic:
        # Cup: correct side is right -> center, incorrect side (left) -> cup_side, both -> cup_side
        # Clamshell: correct side is left -> center, incorrect side (right) -> clamshell_side, both -> clamshell_side
        for det in detections:
            label = det['class'].lower()
            side = self.get_object_side(det)
            
            if label == 'cup':
                if side == 'left':
                    cup_on_left = True
                else:
                    cup_on_right = True
            elif self.is_clamshell(det):
                if side == 'left':
                    clamshell_on_left = True
                else:
                    clamshell_on_right = True
        
        # Add to detection history for time-based averaging
        now = time.time()
        self.detection_history.append((now, cup_on_left, cup_on_right, clamshell_on_left, clamshell_on_right))
        
        # Remove old entries outside averaging window
        cutoff_time = now - self.averaging_window
        self.detection_history = [(t, cl, cr, caml, camr) for t, cl, cr, caml, camr in self.detection_history if t > cutoff_time]
        
        # Average over history (majority vote)
        if self.detection_history:
            cup_on_left = sum(1 for _, cl, _, _, _ in self.detection_history if cl) > len(self.detection_history) / 2
            cup_on_right = sum(1 for _, _, cr, _, _ in self.detection_history if cr) > len(self.detection_history) / 2
            clamshell_on_left = sum(1 for _, _, _, caml, _ in self.detection_history if caml) > len(self.detection_history) / 2
            clamshell_on_right = sum(1 for _, _, _, _, camr in self.detection_history if camr) > len(self.detection_history) / 2
        
        # Debug: show detection state
        det_state = f"CUP(L:{cup_on_left},R:{cup_on_right}) CLAM(L:{clamshell_on_left},R:{clamshell_on_right})"
        
        # Complex waste separation logic with state machine
        target_detected = False
        command = None
        separator_command = None
        
        # Case 1: Clamshell left, cup right -> separator center
        if clamshell_on_left and cup_on_right and not cup_on_left and not clamshell_on_right:
            separator_command = "0"
            command = "21"
            target_detected = True
            logger.info("Clamshell-LEFT, Cup-RIGHT -> SEP-0, BLK-OFF")
        
        # Case 2: Clamshell left, clamshell right -> guide clamshell to correct location
        elif clamshell_on_left and clamshell_on_right and not cup_on_left and not cup_on_right:
            separator_command = "10"
            command = "21"
            target_detected = True
            logger.info("Clamshell-LEFT, Clamshell-RIGHT -> SEP-10, BLK-OFF")
        
        # Case 3: Cup left, cup right -> guide cup to correct location
        elif cup_on_left and cup_on_right and not clamshell_on_left and not clamshell_on_right:
            separator_command = "11"
            command = "21"
            target_detected = True
            logger.info("Cup-LEFT, Cup-RIGHT -> SEP-11, BLK-OFF")
        
        # Case 4: Cup left, clamshell right -> two-step process
        elif cup_on_left and clamshell_on_right and not cup_on_right and not clamshell_on_left:
            if self.step == 0:
                command = "20"
                separator_command = "11"
                self.step = 1
                target_detected = True
                logger.info("Cup-LEFT, Clamshell-RIGHT (Step 1) -> BLK-ON, SEP-11")
            elif self.step == 1:
                command = "21"
                separator_command = "10"
                self.step = 0
                target_detected = True
                logger.info("Cup-LEFT, Clamshell-RIGHT (Step 2) -> BLK-OFF, SEP-10")
        
        # Default: No relevant detections
        else:
            command = "21"
            separator_command = "0"
            target_detected = True
            logger.info(f"Default: SEP-0, BLK-OFF [{det_state}]")
        
        # Send commands
        if target_detected:
            self.last_detection_time = time.time()
            try:
                if command:
                    self.ser.write(f"{command}\n".encode())
                    self.last_command_time = time.time()
                if separator_command:
                    self.ser.write(f"{separator_command}\n".encode())
                    self.last_command_time = time.time()
            except Exception as e:
                logger.error(f"Failed to send serial command: {e}")
        else:
            if time.time() - self.last_detection_time > self.detection_timeout:
                if time.time() - self.last_command_time >= self.command_cooldown:
                    try:
                        self.ser.write("21\n".encode())
                        self.ser.write("0\n".encode())
                        self.last_command_time = time.time()
                        self.step = 0
                        logger.info("Timeout: SEP-0, BLK-OFF")
                    except Exception as e:
                        logger.error(f"Serial error: {e}")
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def main():
    parser = argparse.ArgumentParser(description='YOLO Webcam Detection')
    parser.add_argument('--camera', type=str, default="0", help='Camera device index (0, 1) or RTSP URL')
    parser.add_argument('--width', type=int, default=320, help='Camera width (default: 320)')
    parser.add_argument('--height', type=int, default=320, help='Camera height (default: 320)')
    args = parser.parse_args()

    # Parse camera argument: if it's a digit, convert to int, otherwise keep as string (URL)
    if args.camera.isdigit():
        camera_source = int(args.camera)
    else:
        camera_source = args.camera

    logger.info("Loading YOLO model...")
    model = YOLO("yolov11m.pt")  # Medium model for better accuracy
    
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
                cv2.imshow("YOLO Detection", last_annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        serial_board.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
