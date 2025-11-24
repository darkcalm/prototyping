# PySerial Reference

## Basic Connection
```python
import serial
ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)  # Wait for device to initialize
```

## Send Commands
```python
ser.write(command.encode())  # Convert string to bytes
```

## Receive Data
```python
if ser.in_waiting:
    data = ser.readline().decode()
```

## Always Close
```python
ser.close()  # Close connection when done
```

## Arduino Side
- `Serial.begin(115200)` - Match baud rate with Python
- `Serial.read()` - Read incoming commands
- `Serial.println()` - Send responses back
