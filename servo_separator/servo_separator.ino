#include <Servo.h>

// Servo objects
Servo separatorServo;
Servo blockerServo;
Servo trayServo;

// Separator pins and positions
int separatorPin = 9;
int separatorDefault = 90;
int separatorLeft = 45;
int separatorRight = 135;

// Blocker pins and positions
int blockerPin = 10;
int blockerOn = 180;    // On position
int blockerOff = 0;     // Off position
boolean blockerBlocksClamshell = true;  // Set to false if blocker blocks cups instead

// Tray pins and positions
int trayPin = 11;
int trayUp = 0;         // Up position
int trayDown = 180;     // Down position

int delayTime = 1000;   // time to hold in action mode (ms)
int stepDelay = 10;     // smaller = faster movement (ms per degree)

void setup() {
  Serial.begin(9600);
  
  separatorServo.attach(separatorPin);
  blockerServo.attach(blockerPin);
  trayServo.attach(trayPin);
  
  // Initialize to neutral positions
  separatorServo.write(separatorDefault);  // Center
  blockerServo.write(blockerOff);          // Off
  trayServo.write(trayUp);                 // Up
  delay(500);
  
  Serial.println("=== Multi-Servo Control ===");
  Serial.println("Initialized: SEP-center, BLK-off, TRAY-up");
  Serial.println("Separator Commands:");
  Serial.println(" 0 = Stay center");
  Serial.println("10 = Stay left");
  Serial.println("11 = Stay right");
  Serial.println("Blocker Commands:");
  Serial.println("20 = Blocker ON");
  Serial.println("21 = Blocker OFF");
  Serial.println("Tray Commands:");
  Serial.println("30 = Tray UP");
  Serial.println("31 = Tray DOWN");
}

// Generic smooth motion helper
void moveServoSmooth(Servo &servo, int fromPos, int toPos) {
  if (fromPos < toPos) {
    for (int pos = fromPos; pos <= toPos; pos++) {
      servo.write(pos);
      delay(stepDelay);
    }
  } else {
    for (int pos = fromPos; pos >= toPos; pos--) {
      servo.write(pos);
      delay(stepDelay);
    }
  }
}

// Separator: move and stay at position
void setSeparator(int position) {
  if (position == 0) {
    moveServoSmooth(separatorServo, separatorServo.read(), separatorDefault);
  } 
  else if (position == 10) {
    moveServoSmooth(separatorServo, separatorServo.read(), separatorLeft);
  }
  else if (position == 11) {
    moveServoSmooth(separatorServo, separatorServo.read(), separatorRight);
  }
}

// Blocker control: stay on/off
void setBlocker(int position) {
  if (position == 20) {
    moveServoSmooth(blockerServo, blockerServo.read(), blockerOn);
  }
  else if (position == 21) {
    moveServoSmooth(blockerServo, blockerServo.read(), blockerOff);
  }
}

// Tray control: stay up/down
void setTray(int position) {
  if (position == 30) {
    moveServoSmooth(trayServo, trayServo.read(), trayUp);
  }
  else if (position == 31) {
    moveServoSmooth(trayServo, trayServo.read(), trayDown);
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    int command = input.toInt();

    // Separator commands
    if (command == 0 || command == 10 || command == 11)
      setSeparator(command);
    
    // Blocker commands
    else if (command == 20 || command == 21)
      setBlocker(command);
    
    // Tray commands
    else if (command == 30 || command == 31)
      setTray(command);
  }
}
