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
    
    def __init__(self, serial_board):
        self.state = State.BOOT
        self.serial_board = serial_board
        
        # Detection state
        self.detection_timer = 0
        self.detection_interval = 3.0  # Check every 3 seconds
        
        # Action state
        self.active_case: Optional[CaseID] = None
        self.action_start_time = 0
        self.current_step_index = 0
        
        # Current detection
        self.cup_on_left = False
        self.cup_on_right = False
        self.clamshell_on_left = False
        self.clamshell_on_right = False
        
        # Verification state
        self.verification_received = False
        
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
        CL, CR = self.clamshell_on_left, self.clamshell_on_right
        CuL, CuR = self.cup_on_left, self.cup_on_right
        
        # Case 1: {CUP(L:False,R:True), CLAM(L:True,R:False)} or variations
        if (not CuL and CuR and CL and not CR) or \
           (not CuL and not CuR and CL and not CR) or \
           (not CuL and CuR and not CL and not CR):
            return CaseID.CASE_1
        
        # Case 2: {CUP(L:False,R:False), CLAM(L:False,R:True)} or {CUP(L:False,R:False), CLAM(L:True,R:True)}
        elif (not CuL and not CuR and not CL and CR) or \
             (not CuL and not CuR and CL and CR):
            return CaseID.CASE_2
        
        # Case 3: {CUP(L:True,R:False), CLAM(L:False,R:False)} or {CUP(L:True,R:True), CLAM(L:False,R:False)}
        elif (CuL and not CuR and not CL and not CR) or \
             (CuL and CuR and not CL and not CR):
            return CaseID.CASE_3
        
        # Case 4: {CUP(L:True,R:False), CLAM(L:False,R:True)}
        elif CuL and not CuR and not CL and CR:
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
    
    def _state_boot(self):
        """Booting state: initialize and move to detection"""
        logger.info("=== STATE: BOOT ===")
        logger.info("Initialization complete, moving to DETECT state")
        self.state = State.DETECT
        self.detection_timer = 0
    
    def _state_detect(self, detections: List[dict], dt: float):
        """Detection state: run detection every 3 seconds"""
        self.detection_timer += dt
        
        if self.detection_timer >= self.detection_interval:
            self.detection_timer = 0
            
            # Parse detections
            self.cup_on_left = False
            self.cup_on_right = False
            self.clamshell_on_left = False
            self.clamshell_on_right = False
            
            for det in detections:
                label = det['class'].lower()
                side = self.serial_board.get_object_side(det)
                
                if label == 'cup':
                    if side == 'left':
                        self.cup_on_left = True
                    else:
                        self.cup_on_right = True
                elif self.serial_board.is_clamshell(det):
                    if side == 'left':
                        self.clamshell_on_left = True
                    else:
                        self.clamshell_on_right = True
            
            # Determine case
            case = self._determine_case()
            
            det_state = f"CUP(L:{self.cup_on_left},R:{self.cup_on_right}) CLAM(L:{self.clamshell_on_left},R:{self.clamshell_on_right})"
            
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
            self.state = State.DETECT
            return
        
        elapsed = time.time() - self.action_start_time
        steps = self._get_case_steps(self.active_case)
        
        if not steps:
            logger.error(f"No steps defined for case {self.active_case.value}")
            self.state = State.VERIFY
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
            self.state = State.VERIFY
    
    def _state_verify(self):
        """Verification state: wait for human input"""
        logger.info("=== STATE: VERIFY ===")
        logger.info("Waiting for verification. Type 'ok' or 'retry' and press Enter:")
    
    def handle_verification(self, response: str):
        """Handle verification response from user"""
        response = response.lower().strip()
        if response == 'ok':
            logger.info("Verification accepted. Returning to DETECT.")
            self.state = State.DETECT
            self.detection_timer = 0
            self.active_case = None
            self.verification_received = False
        elif response == 'retry':
            logger.info("Retry requested. Returning to DETECT.")
            self.state = State.DETECT
            self.detection_timer = 0
            self.active_case = None
            self.verification_received = False
        else:
            logger.warning("Invalid response. Please type 'ok' or 'retry'.")


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
    
    def is_clamshell(self, det):
        """Detect clamshell by label or aspect ratio (excludes person)"""
        label = det['class'].lower()
        
        if label == 'person':
            return False
        
        width = det['x2'] - det['x1']
        height = det['y2'] - det['y1']
        
        clamshell_labels = ['cell phone', 'remote', 'spoon', 'book', 'laptop',
                           'keyboard', 'mouse', 'credit card', 'passport']
        
        if any(clamshell_label in label for clamshell_label in clamshell_labels):
            return True
        
        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio > 1.5:
                return True
        
        return False
    
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
    fsm = FSMController(serial_board)
    
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
                results = model(frame, conf=0.55, imgsz=256, verbose=False)
                
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
                
                if frame_count % 120 == 0:
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
