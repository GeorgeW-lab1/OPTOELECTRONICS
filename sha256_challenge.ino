/*
  Arduino Lock Box — SHA-256 Proof-of-Work Challenge Response

  Protocol:
    Arduino -> PC: POW:<64-hex challenge>:<difficulty>\n
    PC -> Arduino: SOLUTION:<16-hex nonce>:<64-hex SHA-256 hash>\n
    Arduino -> PC: DISARMED\n or REJECTED\n

  The PC must find a nonce such that:
      SHA256(challenge_bytes || nonce_bytes)
  has at least the requested number of leading zero bits.

  The Arduino independently recomputes the hash and checks:
      1. The nonce is exactly 8 bytes.
      2. The supplied hash matches its own SHA-256 calculation.
      3. The hash meets the difficulty target.
      4. The challenge is still current.

  Required Arduino library:
      Crypto by Rhys Weatherley
      Install from Arduino IDE -> Tools -> Manage Libraries.

  Safety:
      Use only with a lock box you own or are authorized to control.
      Drive the lock through a properly rated relay or MOSFET module.
      Never connect a lock solenoid directly to an Arduino output pin.
*/

#include <Crypto.h>
#include <SHA256.h>

const unsigned long CHALLENGE_INTERVAL_MS = 5000UL;
const int LOCK_RELAY_PIN = 8;

const uint8_t CHALLENGE_BYTES = 32;
const uint8_t NONCE_BYTES = 8;
const uint8_t HASH_BYTES = 32;

const uint8_t DEFAULT_DIFFICULTY = 20;

unsigned long lastChallengeTime = 0;
uint8_t currentChallenge[CHALLENGE_BYTES];
uint8_t currentDifficulty = DEFAULT_DIFFICULTY;

bool awaitingSolution = false;
bool disarmed = false;

char challengeHex[CHALLENGE_BYTES * 2 + 1];
char receivedHashHex[HASH_BYTES * 2 + 1];

void byteToHex(uint8_t value, char *output) {
  const char hexDigits[] = "0123456789ABCDEF";

  output[0] = hexDigits[value >> 4];
  output[1] = hexDigits[value & 0x0F];
}

void bytesToHex(
  const uint8_t *bytes,
  uint8_t length,
  char *output
) {
  for (uint8_t i = 0; i < length; i++) {
    byteToHex(bytes[i], output + (i * 2));
  }

  output[length * 2] = '\0';
}

bool hexToBytes(
  const char *text,
  uint8_t *output,
  uint8_t expectedBytes
) {
  if (strlen(text) != (size_t)(expectedBytes * 2)) {
    return false;
  }

  for (uint8_t i = 0; i < expectedBytes; i++) {
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

bool arraysEqual(
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

bool meetsDifficulty(
  const uint8_t *digest,
  uint8_t difficultyBits
) {
  uint8_t fullZeroBytes = difficultyBits / 8;
  uint8_t remainingBits = difficultyBits % 8;

  for (uint8_t i = 0; i < fullZeroBytes; i++) {
    if (digest[i] != 0) {
      return false;
    }
  }

  if (remainingBits == 0) {
    return true;
  }

  uint8_t mask = 0xFF << (8 - remainingBits);

  return (digest[fullZeroBytes] & mask) == 0;
}

void calculateHash(
  const uint8_t *challenge,
  uint8_t challengeLength,
  const uint8_t *nonce,
  uint8_t nonceLength,
  uint8_t *outputHash
) {
  SHA256 sha256;

  sha256.reset();
  sha256.update(challenge, challengeLength);
  sha256.update(nonce, nonceLength);
  sha256.finalize(outputHash, HASH_BYTES);
}

void setup() {
  Serial.begin(115200);

  pinMode(LOCK_RELAY_PIN, OUTPUT);
  digitalWrite(LOCK_RELAY_PIN, LOW);

  randomSeed(analogRead(A0));

  Serial.println("LOCKBOX_READY");
}

void loop() {
  unsigned long now = millis();

  if (!disarmed && now - lastChallengeTime >= CHALLENGE_INTERVAL_MS) {
    for (uint8_t i = 0; i < CHALLENGE_BYTES; i++) {
      currentChallenge[i] = (uint8_t)random(0, 256);
    }

    currentDifficulty = DEFAULT_DIFFICULTY;
    lastChallengeTime = now;
    awaitingSolution = true;

    bytesToHex(
      currentChallenge,
      CHALLENGE_BYTES,
      challengeHex
    );

    Serial.print("POW:");
    Serial.print(challengeHex);
    Serial.print(":");
    Serial.println(currentDifficulty);
  }

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (!line.startsWith("SOLUTION:")) {
      return;
    }

    String payload = line.substring(9);
    int separatorIndex = payload.indexOf(':');

    if (separatorIndex < 0) {
      Serial.println("REJECTED");
      return;
    }

    String receivedNonce = payload.substring(
      0, separatorIndex
    );
    String receivedHash = payload.substring(
      separatorIndex + 1
    );

    receivedNonce.toUpperCase();
    receivedHash.toUpperCase();

    uint8_t nonceBytes[NONCE_BYTES];
    uint8_t suppliedHash[HASH_BYTES];
    uint8_t calculatedHash[HASH_BYTES];

    bool nonceValid = hexToBytes(
      receivedNonce.c_str(),
      nonceBytes,
      NONCE_BYTES
    );

    bool hashValid = hexToBytes(
      receivedHash.c_str(),
      suppliedHash,
      HASH_BYTES
    );

    if (!nonceValid || !hashValid) {
      Serial.println("REJECTED");
      return;
    }

    if (disarmed || !awaitingSolution) {
      Serial.println("REJECTED");
      return;
    }

    calculateHash(
      currentChallenge,
      CHALLENGE_BYTES,
      nonceBytes,
      NONCE_BYTES,
      calculatedHash
    );

    bool hashesMatch = arraysEqual(
      suppliedHash,
      calculatedHash,
      HASH_BYTES
    );

    bool difficultyMet = meetsDifficulty(
      calculatedHash,
      currentDifficulty
    );

    if (hashesMatch && difficultyMet) {
      awaitingSolution = false;

      digitalWrite(LOCK_RELAY_PIN, HIGH);
      disarmed = true;

      Serial.println("DISARMED");
    } else {
      awaitingSolution = false;
      Serial.println("REJECTED");
    }
  }
}