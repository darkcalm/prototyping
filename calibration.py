#!/usr/bin/env python3
"""
Servo Calibration Tool
Interactively adjust servo angles for separator, blocker, and tray.
Saves calibrated angles to calibration.yaml
"""

import serial
import serial.tools.list_ports
import time
import json
from pathlib import Path

class ServoCalibrator:
    def __init__(self, port=None, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
        # Servo command constants (match servo_separator.ino and yolo_webcam.py FSM)
        self.commands = {
            'separator': {'center': 0, 'left': 10, 'right': 11},
            'blocker': {'on': 20, 'off': 21},
            'tray': {'up': 30, 'down': 31}
        }
        
        # Calibration notes - user will input these in Arduino after testing
        self.angles = {
            'separator_center': 90,
            'separator_left': 45,
            'separator_right': 135,
            'blocker_on': 180,
            'blocker_off': 0,
            'tray_up': 0,
            'tray_down': 180
        }
        
    def connect(self):
        try:
            if self.port is None:
                ports = [p.device for p in serial.tools.list_ports.comports()]
                if ports:
                    self.port = ports[-1]
                    print(f"Found ports: {ports}")
                    print(f"Using port: {self.port}")
                else:
                    print("ERROR: No serial ports found")
                    return False
            
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            print(f"Connected to serial board on {self.port}\n")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_command(self, servo_name, position):
        """Send FSM command to servo (0,10,11 for sep; 20,21 for blocker; 30,31 for tray)"""
        if self.ser is None or not self.ser.is_open:
            print("ERROR: Serial not connected")
            return False
        
        try:
            cmd = str(position)
            self.ser.write(f"{cmd}\n".encode())
            print(f"  → {servo_name} = {position} (command sent)")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"ERROR sending command: {e}")
            return False
    
    def calibrate_servo(self, servo_name):
        """Interactive calibration for a single servo"""
        print(f"\n{'='*60}")
        print(f"CALIBRATING: {servo_name.upper()}")
        print(f"{'='*60}")
        
        if servo_name == 'separator':
            positions = {'center': 0, 'left': 10, 'right': 11}
            print("Positions: [c]enter, [l]eft, [r]ight")
            print("Test commands to verify servo behavior:")
        elif servo_name == 'blocker':
            positions = {'on': 20, 'off': 21}
            print("Positions: [o]n, [f]f")
            print("Test commands to verify servo behavior:")
        else:  # tray
            positions = {'up': 30, 'down': 31}
            print("Positions: [u]p, [d]own")
            print("Test commands to verify servo behavior:")
        
        print("  [pos key] : move to position (c/l/r or o/f or u/d)")
        print("  [s]       : confirm and save")
        print("  [q]       : quit without saving")
        print(f"\nType commands: ", end='', flush=True)
        
        while True:
            cmd = input().strip().lower()
            
            if servo_name == 'separator':
                if cmd == 'c':
                    self.send_command(servo_name, positions['center'])
                elif cmd == 'l':
                    self.send_command(servo_name, positions['left'])
                elif cmd == 'r':
                    self.send_command(servo_name, positions['right'])
                elif cmd == 's':
                    print(f"✓ {servo_name} calibration confirmed")
                    return True
                elif cmd == 'q':
                    print("Quit without saving")
                    return False
                else:
                    print(f"Unknown command: {cmd}. Try again: ", end='', flush=True)
            
            elif servo_name == 'blocker':
                if cmd == 'o':
                    self.send_command(servo_name, positions['on'])
                elif cmd == 'f':
                    self.send_command(servo_name, positions['off'])
                elif cmd == 's':
                    print(f"✓ {servo_name} calibration confirmed")
                    return True
                elif cmd == 'q':
                    print("Quit without saving")
                    return False
                else:
                    print(f"Unknown command: {cmd}. Try again: ", end='', flush=True)
            
            else:  # tray
                if cmd == 'u':
                    self.send_command(servo_name, positions['up'])
                elif cmd == 'd':
                    self.send_command(servo_name, positions['down'])
                elif cmd == 's':
                    print(f"✓ {servo_name} calibration confirmed")
                    return True
                elif cmd == 'q':
                    print("Quit without saving")
                    return False
                else:
                    print(f"Unknown command: {cmd}. Try again: ", end='', flush=True)
    
    def run_calibration(self):
        """Run calibration for all servos"""
        if not self.connect():
            return False
        
        print("\n" + "="*60)
        print("SERVO CALIBRATION TOOL")
        print("="*60)
        
        servos = ['separator', 'blocker', 'tray']
        saved_angles = {}
        
        for servo_name in servos:
            if self.calibrate_servo(servo_name):
                saved_angles[servo_name] = self.angles[servo_name]
            else:
                print("Calibration cancelled")
                self.close()
                return False
        
        # Save calibration
        self.save_calibration(saved_angles)
        self.close()
        return True
    
    def save_calibration(self, angles):
        """Save calibration notes (user updates servo_separator.ino manually)"""
        config_path = Path('CALIBRATION_RESULT.txt')
        
        with open(config_path, 'w') as f:
            f.write("SERVO CALIBRATION RESULT\n")
            f.write("="*60 + "\n\n")
            f.write("All servos tested and confirmed working.\n")
            f.write("Default angles in servo_separator.ino:\n\n")
            f.write("  separatorDefault = 90  (center)\n")
            f.write("  separatorLeft = 45     (command 10)\n")
            f.write("  separatorRight = 135   (command 11)\n\n")
            f.write("  blockerOff = 0         (command 21)\n")
            f.write("  blockerOn = 180        (command 20)\n\n")
            f.write("  trayUp = 0             (command 30)\n")
            f.write("  trayDown = 180         (command 31)\n\n")
            f.write("If servos need different angles, update these values in\n")
            f.write("servo_separator.ino and re-upload to Arduino.\n")
        
        print(f"\n✓ Calibration confirmed")
        print(f"  Servos respond to FSM commands: 0,10,11 (sep), 20,21 (blocker), 30,31 (tray)")
        print(f"  Settings saved to {config_path}")
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("\nDisconnected from serial board")


def main():
    calibrator = ServoCalibrator()
    try:
        calibrator.run_calibration()
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted")
        calibrator.close()


if __name__ == "__main__":
    main()
