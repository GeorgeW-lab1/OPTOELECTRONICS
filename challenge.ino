/*
  Arduino Lock Box — 256-Bit Oracle Challenge Response

  Protocol:
    Arduino -> PC: CHALLENGE:<64 hexadecimal characters>\n
    PC -> Arduino: GUESS:<64 hexadecimal characters>\n
    Arduino -> PC: DISARMED\n or REJECTED\n

  A new 256-bit random challenge is emitted every five seconds. The Python
  bit-pooling oracle reconstructs it and returns it. The Arduino disarms only
  when the returned value exactly matches the still-current challenge.

  Hardware:
    - Arduino Uno/Nano/Mega or compatible board
    - Relay module or logic-level MOSFET driver on pin 8
    - Lock actuator powered from an appropriate external supply

  Safety:
    - Use only with a lock box you own or are authorized to control.
    - Do not connect a lock solenoid directly to an Arduino output pin.
    - Use a relay or MOSFET rated for the lock voltage and current.
    - Ensure common ground between the Arduino and relay/driver module.
*/

const unsigned long CHALLENGE_INTERVAL_MS = 500000UL;
const int LOCK_RELAY_PIN = 8;

const uint8_t PASSCODE_BYTES = 32;

unsigned long lastChallengeTime = 0;
uint8_t currentChallenge[PASSCODE_BYTES];
bool awaitingGuess = false;
bool disarmed = false;

char challengeHex[PASSCODE_BYTES * 2 + 1];
char receivedHex[PASSCODE_BYTES * 2 + 1];

// Print a byte as exactly two uppercase hexadecimal characters.
void printByteHex(uint8_t value) {
  if (value < 16) {
    Serial.print('0');
  }

  Serial.print(value, HEX);
}

// Generate 32 pseudorandom bytes.
void generateChallenge() {
  for (uint8_t i = 0; i < PASSCODE_BYTES; i++) {
    currentChallenge[i] = (uint8_t)random(0, 256);
  }
}

// Convert the current challenge bytes into a 64-character hex string.
void challengeToHex() {
  for (uint8_t i = 0; i < PASSCODE_BYTES; i++) {
    uint8_t value = currentChallenge[i];

    challengeHex[i * 2] = (value < 16) ? '0' : '\0';

    if (value < 16) {
      challengeHex[i * 2 + 1] = "0123456789ABCDEF"[value];
    } else {
      challengeHex[i * 2] = "0123456789ABCDEF"[value >> 4];
      challengeHex[i * 2 + 1] = "0123456789ABCDEF"[value & 0x0F];
    }
  }

  challengeHex[PASSCODE_BYTES * 2] = '\0';
}

// Convert a 64-character uppercase hexadecimal string into 32 bytes.
bool hexToBytes(const char *text, uint8_t *output) {
  if (strlen(text) != PASSCODE_BYTES * 2) {
    return false;
  }

  for (uint8_t i = 0; i < PASSCODE_BYTES; i++) {
    char highChar = text[i * 2];
    char lowChar = text[i * 2 + 1];

    uint8_t high;
    uint8_t low;

    if (highChar >= '0' && highChar <= '9') {
      high = highChar - '0';
    } else if (highChar >= 'A' && highChar <= 'F') {
      high = highChar - 'A' + 10;
    } else {
      return false;
    }

    if (lowChar >= '0' && lowChar <= '9') {
      low = lowChar - '0';
    } else if (lowChar >= 'A' && lowChar <= 'F') {
      low = lowChar - 'A' + 10;
    } else {
      return false;
    }

    output[i] = (high << 4) | low;
  }

  return true;
}

bool bytesEqual(
  const uint8_t *a,
  const uint8_t *b,
  uint8_t length
) {
  for (uint8_t i = 0; i < length; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }

  return true;
}

void setup() {
  Serial.begin(115200);

  pinMode(LOCK_RELAY_PIN, OUTPUT);
  digitalWrite(LOCK_RELAY_PIN, LOW);

  // Seed from electrical noise on an unconnected analogue input.
  randomSeed(analogRead(A0));

  Serial.println("LOCKBOX_READY");
}

void loop() {
  unsigned long now = millis();

  // Issue a fresh 256-bit challenge every five seconds.
  if (!disarmed && now - lastChallengeTime >= CHALLENGE_INTERVAL_MS) {
    generateChallenge();
    challengeToHex();

    lastChallengeTime = now;
    awaitingGuess = true;

    Serial.print("CHALLENGE:");
    Serial.println(challengeHex);
  }

  // Read and validate one oracle response.
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("GUESS:")) {
      String valueText = line.substring(6);
      valueText.toUpperCase();

      valueText.toCharArray(
        receivedHex,
        sizeof(receivedHex)
      );

      uint8_t receivedBytes[PASSCODE_BYTES];

      if (!hexToBytes(receivedHex, receivedBytes)) {
        Serial.println("REJECTED");
      } else if (disarmed || !awaitingGuess) {
        Serial.println("REJECTED");
      } else if (
        bytesEqual(
          receivedBytes,
          currentChallenge,
          PASSCODE_BYTES
        )
      ) {
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
