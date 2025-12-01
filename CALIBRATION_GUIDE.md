# Servo Calibration Guide

## Overview
`calibration.py` verifies that servos respond correctly to FSM commands. It tests the fixed positions defined in `servo_separator.ino`:
- **Separator**: center (0), left (10), right (11)
- **Blocker**: off (21), on (20)
- **Tray**: up (30), down (31)

## Before You Start
1. Upload `servo_separator.ino` to Arduino
2. Connect all three servo motors to pins 9, 10, 11
3. Ensure servos have adequate power (external PSU recommended)
4. Attach servo arms/mechanisms loosely (allow free movement for testing)

## Running Calibration

```bash
python3 calibration.py
```

The tool will:
1. Auto-detect and connect to Arduino serial port
2. Guide you through **separator**, **blocker**, then **tray**
3. Confirm each servo responds to FSM commands
4. Generate `CALIBRATION_RESULT.txt`

## Per-Servo Calibration

### Separator Calibration
```
[c] center → moves to 90° (neutral position)
[l] left   → moves to 45° (diverts items left)
[r] right  → moves to 135° (diverts items right)
[s] save   → confirm and proceed
```

**Goal**: Verify blade angles create clean left/right/center separation paths.

### Blocker Calibration
```
[o] on  → moves to 180° (blocks one side)
[f] off → moves to 0° (fully retracted)
[s] save → confirm and proceed
```

**Goal**: Verify blocker engages fully without jamming or requiring force.

### Tray Calibration
```
[u] up   → moves to 0° (items held, ready)
[d] down → moves to 180° (tilted to discharge)
[s] save → confirm and proceed
```

**Goal**: Verify full up/down travel. Items should not bind or fall unexpectedly.

## Adjusting Servo Angles

If servos don't align correctly with your mechanical setup:

1. Open `servo_separator.ino`
2. Modify these constants:
   ```c
   int separatorDefault = 90;    // Adjust center angle
   int separatorLeft = 45;       // Adjust left angle
   int separatorRight = 135;     // Adjust right angle
   
   int blockerOn = 180;          // Adjust on position
   int blockerOff = 0;           // Adjust off position
   
   int trayUp = 0;               // Adjust up position
   int trayDown = 180;           // Adjust down position
   ```
3. Re-upload to Arduino
4. Re-run `calibration.py` to verify

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No serial port found | Check USB cable, verify Arduino IDE detects board |
| Servo doesn't move | Confirm power to servo, check pin assignments (9, 10, 11) |
| Servo stalls/stutters | Reduce `stepDelay` in servo_separator.ino (default 10 ms) |
| Wrong angle on first try | Adjust constants in Arduino, re-upload, retry calibration |

## After Calibration

Once confirmed:
- Rigidly attach servo arms/mechanisms
- Run `yolo_webcam.py`—FSM will command servos using the calibrated positions
- All servo motion during sorting should now work correctly

**Calibration is complete when all three servos move to their positions smoothly and servos stay in commanded position until next command.**
