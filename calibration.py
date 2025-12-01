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
        
        # Default servo angles (adjust based on your servo range)
        self.angles = {
            'separator': 90,      # Center position
            'blocker': 90,        # Center position
            'tray': 90            # Center position
        }
        
        # Servo command mapping (adjust if needed)
        self.servo_commands = {
            'separator': 'SEP',   # Separator servo
            'blocker': 'BLK',     # Blocker servo
            'tray': 'TRAY'        # Tray servo
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
    
    def send_angle(self, servo_name, angle):
        """Send servo angle command. Format: servo_name:angle (e.g., 'SEP:90')"""
        if self.ser is None or not self.ser.is_open:
            print("ERROR: Serial not connected")
            return False
        
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        self.angles[servo_name] = angle
        
        try:
            command = f"{servo_name}:{angle}\n"
            self.ser.write(command.encode())
            print(f"  → {servo_name} = {angle}°")
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"ERROR sending command: {e}")
            return False
    
    def calibrate_servo(self, servo_name):
        """Interactive calibration for a single servo"""
        print(f"\n{'='*60}")
        print(f"CALIBRATING: {servo_name.upper()}")
        print(f"{'='*60}")
        print(f"Current angle: {self.angles[servo_name]}°")
        print("\nControls:")
        print("  [+] or [→] : +5°")
        print("  [-] or [←] : -5°")
        print("  [1-9]      : +1° per key")
        print("  [0]        : -1°")
        print("  [d]        : default (90°)")
        print("  [m]        : move to min (0°)")
        print("  [M]        : move to max (180°)")
        print("  [s]        : save and continue to next servo")
        print("  [q]        : quit without saving")
        print(f"\nType commands: ", end='', flush=True)
        
        while True:
            cmd = input().strip().lower()
            
            if cmd in ['+', '→', 'right']:
                self.send_angle(servo_name, self.angles[servo_name] + 5)
            elif cmd in ['-', '←', 'left']:
                self.send_angle(servo_name, self.angles[servo_name] - 5)
            elif cmd in '123456789':
                self.send_angle(servo_name, self.angles[servo_name] + int(cmd))
            elif cmd == '0':
                self.send_angle(servo_name, self.angles[servo_name] - 1)
            elif cmd == 'd':
                self.send_angle(servo_name, 90)
            elif cmd == 'm':
                self.send_angle(servo_name, 0)
            elif cmd == 'M':
                self.send_angle(servo_name, 180)
            elif cmd == 's':
                print(f"Saved {servo_name} = {self.angles[servo_name]}°")
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
        """Save calibrated angles to calibration.yaml"""
        config = {
            'servos': {
                'separator': {'angle': angles['separator']},
                'blocker': {'angle': angles['blocker']},
                'tray': {'angle': angles['tray']}
            }
        }
        
        config_path = Path('calibration.yaml')
        
        # Write as YAML-like format
        with open(config_path, 'w') as f:
            f.write("# Servo Calibration Configuration\n")
            f.write("# Auto-generated by calibration.py\n\n")
            f.write("servos:\n")
            for servo, data in config['servos'].items():
                f.write(f"  {servo}:\n")
                f.write(f"    angle: {data['angle']}\n")
        
        print(f"\n✓ Calibration saved to {config_path}")
        print(f"  separator: {angles['separator']}°")
        print(f"  blocker: {angles['blocker']}°")
        print(f"  tray: {angles['tray']}°")
    
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
