/*
  Arduino Lock Box — Oracle Challenge Response

  Protocol:
    Arduino -> PC: CHALLENGE:<0-1023>\n
    PC -> Arduino: GUESS:<0-1023>\n
    Arduino -> PC: DISARMED\n or REJECTED\n

  The Arduino emits a new random challenge every five seconds. A matching
  guess sent before the next challenge disarms the lock box.

  Hardware:
    - Arduino Uno/Nano/Mega or compatible board
    - Relay module or logic-level MOSFET driver on pin 8
    - Lock actuator powered from an appropriate external supply

  Safety:
    - Do not connect a lock solenoid directly to an Arduino pin.
    - Use a relay or MOSFET rated for the lock voltage and current.
    - Ensure common ground between the Arduino and the relay/driver module.
*/

const unsigned long CHALLENGE_INTERVAL_MS = 5000UL;
const int LOCK_RELAY_PIN = 8;

const int CHALLENGE_MIN = 0;
const int CHALLENGE_MAX = 1023;

unsigned long lastChallengeTime = 0;
int currentChallenge = -1;
bool awaitingGuess = false;
bool disarmed = false;

void setup() {
  Serial.begin(115200);

  pinMode(LOCK_RELAY_PIN, OUTPUT);
  digitalWrite(LOCK_RELAY_PIN, LOW);

  // Seed the random generator from electrical noise on an unconnected pin.
  randomSeed(analogRead(A0));

  Serial.println("LOCKBOX_READY");
}

void loop() {
  unsigned long now = millis();

  // Issue a fresh challenge every five seconds.
  if (!disarmed && now - lastChallengeTime >= CHALLENGE_INTERVAL_MS) {
    currentChallenge = random(CHALLENGE_MIN, CHALLENGE_MAX + 1);
    lastChallengeTime = now;
    awaitingGuess = true;

    Serial.print("CHALLENGE:");
    Serial.println(currentChallenge);
  }

  // Read and validate one oracle response.
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("GUESS:")) {
      int guess = line.substring(6).toInt();

      if (disarmed) {
        Serial.println("REJECTED");
      } else if (!awaitingGuess) {
        Serial.println("REJECTED");
      } else if (guess < CHALLENGE_MIN || guess > CHALLENGE_MAX) {
        Serial.println("REJECTED");
      } else if (guess == currentChallenge) {
        awaitingGuess = false;

        digitalWrite(LOCK_RELAY_PIN, HIGH);
        disarmed = true;

        Serial.println("DISARMED");
      } else {
        awaitingGuess = false;
        Serial.println("REJECTED");
      }
    }
  }
}