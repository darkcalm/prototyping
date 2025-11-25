#include <Servo.h>

Servo separatorServo;

int servoPin = 9;
int defaultPos = 90;
int leftPos = 45;
int rightPos = 135;
int delayTime = 1000;  // time to hold in action mode (ms)
int stepDelay = 10;    // smaller = faster movement (ms per degree)

void setup() {
  separatorServo.attach(servoPin);
  separatorServo.write(defaultPos);
  delay(500);
  Serial.begin(9600);
  Serial.println("=== Servo Separator Test ===");
  Serial.println("Commands:");
  Serial.println("-1 = Move left (45°), return to 90°");
  Serial.println(" 1 = Move right (135°), return to 90°");
  Serial.println(" 0 = Return to center (90°)");
  Serial.println("10 = Go left (45°) and stay there");
  Serial.println("11 = Go right (135°) and stay there");
}

// Smooth motion helper
void moveServoSmooth(int fromPos, int toPos) {
  if (fromPos < toPos) {
    for (int pos = fromPos; pos <= toPos; pos++) {
      separatorServo.write(pos);
      delay(stepDelay);
    }
  } else {
    for (int pos = fromPos; pos >= toPos; pos--) {
      separatorServo.write(pos);
      delay(stepDelay);
    }
  }
}

// Normal operation: go to side, wait, then return
void controlSeparator(int signal) {
  if (signal == -1) {
    moveServoSmooth(defaultPos, leftPos);
    delay(delayTime);
    moveServoSmooth(leftPos, defaultPos);
    Serial.print("DEBUG: After LEFT command, position = ");
    Serial.println(separatorServo.read());
  } 
  else if (signal == 1) {
    moveServoSmooth(defaultPos, rightPos);
    delay(delayTime);
    moveServoSmooth(rightPos, defaultPos);
    Serial.print("DEBUG: After RIGHT command, position = ");
    Serial.println(separatorServo.read());
  }
  // 0 or others → ignore
}

// Test mode: move and stay forever
void testServoPosition(int direction) {
  if (direction == -1) {
    moveServoSmooth(separatorServo.read(), leftPos);
    Serial.print("DEBUG: After STAY LEFT, position = ");
    Serial.println(separatorServo.read());
    Serial.println("→ Servo moved to LEFT (45°) and stays there.");
  } 
  else if (direction == 1) {
    moveServoSmooth(separatorServo.read(), rightPos);
    Serial.print("DEBUG: After STAY RIGHT, position = ");
    Serial.println(separatorServo.read());
    Serial.println("→ Servo moved to RIGHT (135°) and stays there.");
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    int command = input.toInt();

    if (command == -1 || command == 1)
      controlSeparator(command);
    else if (input == "0") {
      int currentRead = separatorServo.read();
      Serial.print("DEBUG: separatorServo.read() = ");
      Serial.println(currentRead);
      moveServoSmooth(currentRead, defaultPos);
      Serial.println("→ Servo returned to CENTER (90°).");
    }
    else if (command == 10)
      testServoPosition(-1);
    else if (command == 11)
      testServoPosition(1);
  }
}
