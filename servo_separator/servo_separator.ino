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
  Serial.println("10 = Go left (45°) and stay there");
  Serial.println("11 = Go right (135°) and stay there");

  Serial.println("Ready to Receive Commands!");
  pinMode(13, OUTPUT);
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
  } 
  else if (signal == 1) {
    moveServoSmooth(defaultPos, rightPos);
    delay(delayTime);
    moveServoSmooth(rightPos, defaultPos);
  }
  // 0 or others → ignore
}

// Test mode: move and stay forever
void testServoPosition(int direction) {
  if (direction == -1) {
    moveServoSmooth(separatorServo.read(), leftPos);
    Serial.println("→ Servo moved to LEFT (45°) and stays there.");
  } 
  else if (direction == 1) {
    moveServoSmooth(separatorServo.read(), rightPos);
    Serial.println("→ Servo moved to RIGHT (135°) and stays there.");
  }
}

void loop() {
  if (Serial.available() > 0) {
    String str = Serial.readStringUntil('\n');
    Serial.print("Loopback: ");
    Serial.println(str);

    // Check for TM2Arduino App Interface
    int score = -1;
    int indexTM = str.indexOf("<:>");
    if (indexTM != -1)
    {
      String scoreStr = str.substring(indexTM+3, str.length());
      scoreStr.replace("%","");
      score = scoreStr.toInt();
      str = str.substring(0, indexTM);
    }
      
    // Action for Each Class
    if ((str == "Cardboard") && (score > 90))
    {
      // Action for Class 1
      testServoPosition(-1);
    }
    else if ((str == "Plastic") && (score > 90))
    {
      // Action for Class 2
      testServoPosition(1);
    }
    else if ((str == "Metal") && (score > 90))
    {
      // Action for Class 3
      
    }

    // Clear All Serial Data in Buffer
    while(Serial.available())
      Serial.read();
  }
}