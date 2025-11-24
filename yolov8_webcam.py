#!/usr/bin/env python3
"""
YOLOv8 Real-time Detection with macOS Webcam and Serial Output
"""

import cv2
import serial
import time
import logging
from ultralytics import YOLO
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerialBoard:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        """Initialize serial connection to external board"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
    def connect(self):
        """Establish serial connection"""
        try:
            if self.port is None:
                # Auto-detect on macOS
                self.port = self._detect_port()
            
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Wait for device to initialize
            logger.info(f"Connected to serial board on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to serial board: {e}")
            return False
    
    def _detect_port(self):
        """Auto-detect serial port on macOS"""
        import glob
        ports = glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*')
        if ports:
            return ports[0]
        raise RuntimeError("No serial ports found")
    
    def send_detection(self, detections):
        """Send detection results to serial board"""
        if self.ser is None or not self.ser.is_open:
            return
        
        try:
            # Format: "DETECT:class1,conf1;class2,conf2\n"
            if detections:
                message = "DETECT:" + ";".join([
                    f"{det['class']},{det['confidence']:.2f}"
                    for det in detections
                ]) + "\n"
            else:
                message = "DETECT:NONE\n"
            
            self.ser.write(message.encode())
            logger.debug(f"Sent: {message.strip()}")
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
    
    def close(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial connection closed")


class YOLOv8Detector:
    def __init__(self, model_name="yolov8n", confidence_threshold=0.5):
        """Initialize YOLOv8 model"""
        logger.info(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(f"{model_name}.pt")
        self.confidence_threshold = confidence_threshold
    
    def detect(self, frame):
        """Run detection on frame"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        return results[0]
    
    def parse_results(self, results):
        """Parse detection results"""
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results.names[cls_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': box.xyxy[0].cpu().numpy()
                })
        return detections


def main():
    """Main execution"""
    # Initialize components
    detector = YOLOv8Detector(model_name="yolov8n", confidence_threshold=0.5)
    serial_board = SerialBoard()
    
    # Connect to serial board
    if not serial_board.connect():
        logger.warning("Continuing without serial output")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info("Starting detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break
            
            # Run detection
            results = detector.detect(frame)
            detections = detector.parse_results(results)
            
            # Send to serial board
            serial_board.send_detection(detections)
            
            # Draw detections on frame
            annotated_frame = results.plot()
            
            # Display
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            
            # Log detections
            if detections:
                logger.info(f"Detected: {', '.join([f'{d['class']}({d['confidence']:.2f})' for d in detections])}")
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting...")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        serial_board.close()


if __name__ == "__main__":
    main()
