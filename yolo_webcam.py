#!/usr/bin/env python3
"""
YOLO Detection with Finite State Machine Control
States: BOOT -> DETECT -> ACTION -> VERIFY -> DETECT
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
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FSM States
class State(IntEnum):
    BOOT = 0
    DETECT = 1
    ACTION = 2
    VERIFY = 3


# Cases for action mode
class CaseID(IntEnum):
    NONE = 0
    CASE_1 = 1
    CASE_2 = 2
    CASE_3 = 3
    CASE_4 = 4


@dataclass
class ActionStep:
    """Time-driven action step for recipe execution"""
    t_start: float  # seconds from action start
    t_end: float    # seconds from action start
    sep_cmd: int    # 0=center, 10=left, 11=right, -1=ignore
    tray_cmd: int   # 30=up, 31=down, -1=ignore
    blocker: int    # 20=on, 21=off, -1=ignore


class FSMController:
    """Finite State Machine for waste separator control"""
    
    def __init__(self, serial_board, simplified_mode=True):
        self.state = State.BOOT
        self.serial_board = serial_board
        self.simplified_mode = simplified_mode
        
        # Detection state
        self.detection_timer = 0
        self.detection_interval = 0.3  # Check every 0.3 seconds (~3 seconds for 10 samples)
        
        # Action state
        self.active_case: Optional[CaseID] = None
        self.action_start_time = 0
        self.current_step_index = 0
        
        # Current detection
        self.cup_on_left = False
        self.cup_on_right = False
        self.bottle_on_left = False
        self.bottle_on_right = False
        
        # Majority voting for detection stability
        self.detection_history = {
            'cup_on_left': [],
            'cup_on_right': [],
            'bottle_on_left': [],
            'bottle_on_right': []
        }
        self.history_size = 3  # Track 3 frames
        self.history_threshold = 2  # Require 2+ out of 3 detections
        
        # Verification state
        self.verification_received = False
        self.verification_prompted = False
        
        # Initialize case recipes
        self._init_case_recipes()
        
        logger.info("FSM initialized in BOOT state")
    
    def _init_case_recipes(self):
        """Initialize time-action recipes for each case"""
        
        # Case 1: Clamshell-LEFT, Cup-RIGHT (or variations)
        # t=0-1.5s: separator center, tray up
        # t=1.5-3s: tray down
        # t=3-6s: tray stays down
        # t=6-7.5s: tray up
        self.case_1_steps = [
            ActionStep(0.0, 1.5, 0, -1, -1),      # sep center
            ActionStep(1.5, 3.0, -1, 31, -1),     # tray down
            ActionStep(3.0, 6.0, -1, -1, -1),     # tray stays down (hold)
            ActionStep(6.0, 7.5, -1, 30, -1),     # tray up
        ]
        
        # Case 2: Clamshell-RIGHT (no cup)
        # t=0-3s: separator points to R (block clam to R)
        # t=3-4.5s: tray down
        # t=4.5-7.5s: tray stays down
        # t=7.5-9s: tray up
        # t=9-10.5s: separator center
        self.case_2_steps = [
            ActionStep(0.0, 3.0, 11, -1, -1),     # sep to right
            ActionStep(3.0, 4.5, -1, 31, -1),     # tray down
            ActionStep(4.5, 7.5, -1, -1, -1),     # tray stays down (hold)
            ActionStep(7.5, 9.0, -1, 30, -1),     # tray up
            ActionStep(9.0, 10.5, 0, -1, -1),     # sep center
        ]
        
        # Case 3: Cup-LEFT (no clamshell)
        # t=0-3s: separator points to L (block cup to L)
        # t=3-4.5s: tray down
        # t=4.5-7.5s: tray stays down
        # t=7.5-9s: tray up
        # t=9-10.5s: separator center
        self.case_3_steps = [
            ActionStep(0.0, 3.0, 10, -1, -1),     # sep to left
            ActionStep(3.0, 4.5, -1, 31, -1),     # tray down
            ActionStep(4.5, 7.5, -1, -1, -1),     # tray stays down (hold)
            ActionStep(7.5, 9.0, -1, 30, -1),     # tray up
            ActionStep(9.0, 10.5, 0, -1, -1),     # sep center
        ]
        
        # Case 4: Cup-LEFT and Clamshell-RIGHT (swap sequence)
        # t=0-3s: blocker on
        # t=3-6s: separator points opposite to BLOCKSIDE (assume blockside=left, so point right)
        # t=6-7.5s: tray down
        # t=7.5-10.5s: tray stays down
        # t=10.5-12s: tray up
        # t=12-15s: blocker off
        # t=15-18s: separator points to BLOCKSIDE (left)
        # t=18-19.5s: tray down
        # t=19.5-21s: tray stays down
        # t=21-22.5s: tray up
        # t=22.5-24s: separator center
        self.case_4_steps = [
            ActionStep(0.0, 3.0, -1, -1, 20),     # blocker on
            ActionStep(3.0, 6.0, 11, -1, -1),     # sep opposite blockside (right)
            ActionStep(6.0, 7.5, -1, 31, -1),     # tray down
            ActionStep(7.5, 10.5, -1, -1, -1),    # tray stays down (hold)
            ActionStep(10.5, 12.0, -1, 30, -1),   # tray up
            ActionStep(12.0, 15.0, -1, -1, 21),   # blocker off
            ActionStep(15.0, 18.0, 10, -1, -1),   # sep to blockside (left)
            ActionStep(18.0, 19.5, -1, 31, -1),   # tray down
            ActionStep(19.5, 21.0, -1, -1, -1),   # tray stays down (hold)
            ActionStep(21.0, 22.5, -1, 30, -1),   # tray up
            ActionStep(22.5, 24.0, 0, -1, -1),    # sep center
        ]
    
    def _determine_case(self) -> CaseID:
        """Determine which case applies based on current detection"""
        BL, BR = self.bottle_on_left, self.bottle_on_right
        CuL, CuR = self.cup_on_left, self.cup_on_right
        
        # Case 1: {CUP(L:False,R:True), BOTTLE(L:True,R:False)} or variations
        if (not CuL and CuR and BL and not BR) or \
           (not CuL and not CuR and BL and not BR) or \
           (not CuL and CuR and not BL and not BR):
            return CaseID.CASE_1
        
        # Case 2: {CUP(L:False,R:False), BOTTLE(L:False,R:True)} or {CUP(L:False,R:False), BOTTLE(L:True,R:True)}
        elif (not CuL and not CuR and not BL and BR) or \
             (not CuL and not CuR and BL and BR):
            return CaseID.CASE_2
        
        # Case 3: {CUP(L:True,R:False), BOTTLE(L:False,R:False)} or {CUP(L:True,R:True), BOTTLE(L:False,R:False)}
        elif (CuL and not CuR and not BL and not BR) or \
             (CuL and CuR and not BL and not BR):
            return CaseID.CASE_3
        
        # Case 4: {CUP(L:True,R:False), BOTTLE(L:False,R:True)}
        elif CuL and not CuR and not BL and BR:
            return CaseID.CASE_4
        
        else:
            return CaseID.NONE
    
    def _get_case_steps(self, case: CaseID) -> List[ActionStep]:
        """Get the step sequence for a case"""
        if case == CaseID.CASE_1:
            return self.case_1_steps
        elif case == CaseID.CASE_2:
            return self.case_2_steps
        elif case == CaseID.CASE_3:
            return self.case_3_steps
        elif case == CaseID.CASE_4:
            return self.case_4_steps
        return []
    
    def _execute_action_step(self, step: ActionStep, step_idx: int, total_steps: int, elapsed: float, total_time: float):
        """Send commands for a single action step and log progress"""
        commands = []
        
        if step.sep_cmd != -1:
            try:
                self.serial_board.ser.write(f"{step.sep_cmd}\n".encode())
                sep_names = {0: "center", 10: "left", 11: "right"}
                commands.append(f"SEP→{sep_names.get(step.sep_cmd, '?')}")
            except Exception as e:
                logger.error(f"Failed to send sep command: {e}")
        
        if step.tray_cmd != -1:
            try:
                self.serial_board.ser.write(f"{step.tray_cmd}\n".encode())
                tray_names = {30: "up", 31: "down"}
                commands.append(f"TRAY→{tray_names.get(step.tray_cmd, '?')}")
            except Exception as e:
                logger.error(f"Failed to send tray command: {e}")
        
        if step.blocker != -1:
            try:
                self.serial_board.ser.write(f"{step.blocker}\n".encode())
                blocker_names = {20: "on", 21: "off"}
                commands.append(f"BLK→{blocker_names.get(step.blocker, '?')}")
            except Exception as e:
                logger.error(f"Failed to send blocker command: {e}")
        
        # Progress logging
        progress_pct = int((elapsed / total_time) * 100)
        cmd_str = ", ".join(commands) if commands else "hold"
        logger.info(f"  [{step_idx}/{total_steps}] {elapsed:.1f}s: {cmd_str} [{progress_pct}%]")
    
    def update(self, detections: List[dict], dt: float):
        """Update FSM state and execute actions"""
        
        if self.state == State.BOOT:
            self._state_boot()
        
        elif self.state == State.DETECT:
            self._state_detect(detections, dt)
        
        elif self.state == State.ACTION:
            self._state_action(dt)
        
        elif self.state == State.VERIFY:
            self._state_verify()
    
    def _vote_on_detection(self, new_state: bool, history_key: str) -> bool:
        """Update detection history and return majority vote result (5+ out of 10)"""
        self.detection_history[history_key].append(new_state)
        if len(self.detection_history[history_key]) > self.history_size:
            self.detection_history[history_key].pop(0)
        
        # Return True if threshold met (5+ out of 10)
        if len(self.detection_history[history_key]) >= self.history_size:
            return sum(self.detection_history[history_key]) >= self.history_threshold
        return False
    
    def _state_boot(self):
        """Booting state: initialize hardware and move to detection"""
        logger.info("=== STATE: BOOT ===")
        logger.info("Initializing hardware...")
        
        # Ensure separator is center
        try:
            self.serial_board.ser.write(b"0\n")
            logger.info("SEP→center")
        except Exception as e:
            logger.error(f"Failed to center separator: {e}")
        
        # Ensure tray is up
        try:
            self.serial_board.ser.write(b"30\n")
            logger.info("TRAY→up")
        except Exception as e:
            logger.error(f"Failed to raise tray: {e}")
        
        # Ensure blocker is off
        try:
            self.serial_board.ser.write(b"21\n")
            logger.info("BLK→off")
        except Exception as e:
            logger.error(f"Failed to turn off blocker: {e}")
        
        logger.info("Initialization complete, moving to DETECT state")
        self.state = State.DETECT
        self.detection_timer = 0
    
    def _state_detect(self, detections: List[dict], dt: float):
        """Detection state: run detection every 0.3 seconds with majority voting"""
        self.detection_timer += dt
        
        if self.detection_timer >= self.detection_interval:
            self.detection_timer = 0
            
            if self.simplified_mode:
                self._state_detect_simplified(detections)
            else:
                self._state_detect_full(detections)
    
    def _is_bounding_box_valid(self, det: dict, min_width: int = 30, min_height: int = 30) -> bool:
        """Check if bounding box is large enough to be foreground object"""
        width = det['x2'] - det['x1']
        height = det['y2'] - det['y1']
        return width >= min_width and height >= min_height
    
    def _state_detect_simplified(self, detections: List[dict]):
        """Simplified mode: detect only cup or bottle (no left/right)"""
        # Parse current frame detections
        has_cup = False
        has_bottle = False
        
        for det in detections:
            width = det['x2'] - det['x1']
            height = det['y2'] - det['y1']
            label = det['class'].lower()
            
            if not self._is_bounding_box_valid(det):
                logger.debug(f"  {label}: {width}x{height} (too small, filtered)")
                continue
            
            logger.debug(f"  {label}: {width}x{height} (valid)")
            
            if label == 'cup':
                has_cup = True
            elif label == 'bottle':
                has_bottle = True
        
        # Apply voting (reuse fields for cup_on_left = has_cup, bottle_on_left = has_bottle)
        cup_detected = self._vote_on_detection(has_cup, 'cup_on_left')
        bottle_detected = self._vote_on_detection(has_bottle, 'bottle_on_left')
        
        # Determine case (simplified)
        case = CaseID.NONE
        det_state = f"CUP:{cup_detected}, BOTTLE:{bottle_detected}"
        
        if cup_detected and not bottle_detected:
            case = CaseID.CASE_3  # Cup -> Case 3
        elif bottle_detected and not cup_detected:
            case = CaseID.CASE_2  # Bottle -> Case 2
        
        if case == CaseID.NONE:
            logger.info(f"DETECT: No action case [{det_state}]")
        else:
            logger.info(f"DETECT: Case {case.value} detected [{det_state}]")
            self.active_case = case
            self.action_start_time = time.time()
            self.state = State.ACTION
    
    def _state_detect_full(self, detections: List[dict]):
        """Full mode: detect cup and bottle with left/right sides"""
        # Parse current frame detections (single pass)
        cup_on_left = False
        cup_on_right = False
        bottle_on_left = False
        bottle_on_right = False
        
        for det in detections:
            width = det['x2'] - det['x1']
            height = det['y2'] - det['y1']
            label = det['class'].lower()
            
            if not self._is_bounding_box_valid(det):
                logger.debug(f"  {label}: {width}x{height} (too small, filtered)")
                continue
            
            logger.debug(f"  {label}: {width}x{height} (valid)")
            side = self.serial_board.get_object_side(det)
            
            if label == 'cup':
                if side == 'left':
                    cup_on_left = True
                else:
                    cup_on_right = True
            elif label == 'bottle':
                if side == 'left':
                    bottle_on_left = True
                else:
                    bottle_on_right = True
        
        # Apply majority voting
        self.cup_on_left = self._vote_on_detection(cup_on_left, 'cup_on_left')
        self.cup_on_right = self._vote_on_detection(cup_on_right, 'cup_on_right')
        self.bottle_on_left = self._vote_on_detection(bottle_on_left, 'bottle_on_left')
        self.bottle_on_right = self._vote_on_detection(bottle_on_right, 'bottle_on_right')
        
        # Determine case
        case = self._determine_case()
        
        det_state = f"CUP(L:{self.cup_on_left},R:{self.cup_on_right}) BOTTLE(L:{self.bottle_on_left},R:{self.bottle_on_right})"
        
        if case == CaseID.NONE:
            logger.info(f"DETECT: No action case [{det_state}]")
        else:
            logger.info(f"DETECT: Case {case.value} detected [{det_state}]")
            self.active_case = case
            self.action_start_time = time.time()
            self.state = State.ACTION
    
    def _state_action(self, dt: float):
        """Action state: execute timed sequence"""
        if self.active_case is None:
            logger.error("ACTION state but no active case!")
            self._reset_detection_state()
            return
        
        elapsed = time.time() - self.action_start_time
        steps = self._get_case_steps(self.active_case)
        
        if not steps:
            logger.error(f"No steps defined for case {self.active_case.value}")
            self._reset_detection_state()
            return
        
        total_time = steps[-1].t_end
        
        # Find current step(s) and execute
        found_active_step = False
        for step_idx, step in enumerate(steps):
            if step.t_start <= elapsed < step.t_end:
                self._execute_action_step(step, step_idx + 1, len(steps), elapsed, total_time)
                found_active_step = True
        
        # Check if action is complete
        if elapsed >= total_time:
            logger.info(f"✓ ACTION Case {self.active_case.value} complete ({elapsed:.1f}s)")
            self._reset_detection_state()
    
    def _state_verify(self):
        """Verification state: wait for human input"""
        if not self.verification_prompted:
            logger.info("=== STATE: VERIFY ===")
            logger.info("Waiting for verification. Type 'ok' or 'retry' and press Enter:")
            self.verification_prompted = True
    
    def handle_verification(self, response: str):
        """Handle verification response from user"""
        response = response.lower().strip()
        if response == 'ok':
            logger.info("Verification accepted. Returning to DETECT.")
            self._reset_detection_state()
        elif response == 'retry':
            logger.info("Retry requested. Returning to DETECT.")
            self._reset_detection_state()
        else:
            logger.warning("Invalid response. Please type 'ok' or 'retry'.")
    
    def _reset_detection_state(self):
        """Reset all detection flags and history when returning to DETECT"""
        self.state = State.DETECT
        self.detection_timer = 0
        self.active_case = None
        self.verification_received = False
        self.verification_prompted = False
        
        # Clear detection flags
        self.cup_on_left = False
        self.cup_on_right = False
        self.bottle_on_left = False
        self.bottle_on_right = False
        
        # Clear detection history to avoid stale data
        self.detection_history = {
            'cup_on_left': [],
            'cup_on_right': [],
            'bottle_on_left': [],
            'bottle_on_right': []
        }


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
    def __init__(self, port=None, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.cup_side = "10"
        self.clamshell_side = "11"
        
    def connect(self):
        try:
            if self.port is None:
                ports = [p.device for p in serial.tools.list_ports.comports()]
                if ports:
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
        center_x = (det['x1'] + det['x2']) / 2
        frame_width = 320
        return 'left' if center_x < frame_width / 2 else 'right'
    

    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


def verification_input_thread(fsm):
    """Background thread for verification input"""
    while True:
        if fsm.state == State.VERIFY:
            try:
                response = input()
                fsm.handle_verification(response)
            except EOFError:
                break
        else:
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='YOLO Webcam Detection with FSM')
    parser.add_argument('--camera', type=str, default="0", help='Camera device index (0, 1) or RTSP URL')
    parser.add_argument('--width', type=int, default=320, help='Camera width (default: 320)')
    parser.add_argument('--height', type=int, default=320, help='Camera height (default: 320)')
    parser.add_argument('--full', action='store_true', help='Enable full mode (default: simplified mode)')
    args = parser.parse_args()

    if args.camera.isdigit():
        camera_source = int(args.camera)
    else:
        camera_source = args.camera

    logger.info("Loading YOLO model...")
    model = YOLO("yolo11s.pt")
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Connect to serial board
    serial_board = SerialBoard()
    serial_board.connect()
    
    # Initialize FSM
    fsm = FSMController(serial_board, simplified_mode=not args.full)
    if args.full:
        logger.info("Running in FULL mode (cup/bottle with left/right detection)")
    else:
        logger.info("Running in SIMPLIFIED mode (cup or bottle, no left/right)")
    
    # Start verification input thread
    input_thread = threading.Thread(target=verification_input_thread, args=(fsm,), daemon=True)
    input_thread.start()
    
    # Open camera
    if sys.platform == 'win32':
        cap = CV2Camera(device_index=camera_source, width=args.width, height=args.height)
    else:
        try:
            if isinstance(camera_source, str):
                cap = CV2Camera(device_index=camera_source, width=args.width, height=args.height)
            else:
                cap = FFmpegCamera(width=args.width, height=args.height)
        except Exception as e:
            logger.warning(f"FFmpegCamera failed: {e}. Falling back to CV2Camera.")
            cap = CV2Camera(device_index=camera_source, width=args.width, height=args.height)
    
    logger.info("Starting detection with FSM. Press 'q' to quit.")
    
    SKIP_FRAMES = 2
    last_annotated = None
    last_frame_time = time.time()
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time
            
            # Run detection on every frame
            if frame_count % SKIP_FRAMES == 0:
                results = model(frame, conf=0.3, imgsz=256, verbose=False)
                
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
                
                # Update FSM
                fsm.update(detections, dt)
                
                if frame_count % 36 == 0:
                    if detections:
                        det_str = ', '.join([f"{d['class']}({d['confidence']:.2f})" for d in detections])
                        logger.info(f"Frame {frame_count}: {det_str}")
                    else:
                        logger.info(f"Frame {frame_count}: No detections")
                    gc.collect()
            
            # Display last result
            if last_annotated is not None:
                cv2.imshow("YOLO Detection (FSM)", last_annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        serial_board.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
