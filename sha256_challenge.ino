/*
  DUMB SHA-256 PROOF-OF-WORK MINER
  --------------------------------
  No external libraries (own SHA-256). Serial: 115200 baud.
  Non-blocking: every pass through loop() it first listens to the serial
  port, then does exactly ONE hash, so commands are answered instantly.

  Protocol (one command per line, case-insensitive):

    PC      -> Arduino   JOB:<64-hex challenge>:<difficulty 1..32>
    Arduino -> PC        MINING_STARTED
    Arduino -> PC        MINED:<64-hex challenge>:<difficulty>:<16-hex nonce>:<64-hex hash>
    PC      -> Arduino   ACCEPTED   -> relay pin goes HIGH, replies DISARMED
    PC      -> Arduino   REJECTED   -> replies REJECTED_ACK, back to idle
    PC      -> Arduino   STOP       -> replies STOPPED (or IDLE if nothing running)

  Errors: ERR:BAD_JOB:FORMAT | CHALLENGE_LENGTH | CHALLENGE_HEX | DIFFICULTY_RANGE
          ERR:LINE_TOO_LONG | ERR:UNKNOWN_COMMAND | IGNORED:NO_PROOF_PENDING

  Proof rule:
    hash = SHA256(challenge bytes (32) + nonce bytes (8))
    The hash must start with <difficulty> zero bits.
    The nonce is 8 bytes, big-endian, counting up from 0.

  Speed note:
    challenge + nonce is always 40 bytes, which pads to exactly ONE
    64-byte SHA-256 block. The block is built once per job; each hash
    only rewrites the 8 nonce bytes and runs one transform.
*/

#include <Arduino.h>
#include <string.h>
#include <stdint.h>

// =====================================================
// CONFIGURATION
// =====================================================

#define SERIAL_BAUD 115200

const int LOCK_RELAY_PIN = 8;          // driven HIGH when a proof is ACCEPTED

const uint8_t CHALLENGE_BYTES = 32;
const uint8_t NONCE_BYTES = 8;
const uint8_t HASH_BYTES = 32;

const uint8_t MIN_DIFFICULTY = 1;
const uint8_t MAX_DIFFICULTY = 32;

const uint8_t LINE_BUFFER_SIZE = 96;

enum MinerState : uint8_t {
  STATE_IDLE,
  STATE_MINING,
  STATE_AWAITING_VERIFICATION
};

MinerState state = STATE_IDLE;

uint8_t currentChallenge[CHALLENGE_BYTES];
uint8_t currentNonce[NONCE_BYTES];
uint8_t currentHash[HASH_BYTES];
uint8_t currentDifficulty = 0;

char lineBuffer[LINE_BUFFER_SIZE];
uint8_t lineLength = 0;
bool lineOverflow = false;

// =====================================================
// SHA256 CONSTANTS
// =====================================================

static const uint32_t K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
  0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
  0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
  0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
  0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
  0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
  0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
  0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
  0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// =====================================================
// SHA256 (single 64-byte block)
// =====================================================

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32-(n))))

uint32_t H[8];
uint8_t block[64];
uint32_t W[64];

void sha256Transform() {

  for (int i = 0; i < 16; i++) {
    W[i] =
      ((uint32_t)block[i*4] << 24) |
      ((uint32_t)block[i*4+1] << 16) |
      ((uint32_t)block[i*4+2] << 8) |
      ((uint32_t)block[i*4+3]);
  }

  for (int i = 16; i < 64; i++) {

    uint32_t s0 =
      ROTR(W[i-15],7) ^
      ROTR(W[i-15],18) ^
      (W[i-15] >> 3);

    uint32_t s1 =
      ROTR(W[i-2],17) ^
      ROTR(W[i-2],19) ^
      (W[i-2] >> 10);

    W[i] = W[i-16] + s0 + W[i-7] + s1;
  }

  uint32_t a=H[0], b=H[1], c=H[2], d=H[3];
  uint32_t e=H[4], f=H[5], g=H[6], h=H[7];

  for (int i = 0; i < 64; i++) {

    uint32_t S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t temp1 = h + S1 + ch + K[i] + W[i];

    uint32_t S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = S0 + maj;

    h=g;
    g=f;
    f=e;
    e=d+temp1;
    d=c;
    c=b;
    b=a;
    a=temp1+temp2;
  }

  H[0]+=a;
  H[1]+=b;
  H[2]+=c;
  H[3]+=d;
  H[4]+=e;
  H[5]+=f;
  H[6]+=g;
  H[7]+=h;
}

// =====================================================
// HEX HELPERS
// =====================================================

void printHex(const uint8_t *bytes, uint8_t length) {
  const char hexDigits[] = "0123456789ABCDEF";
  for (uint8_t i = 0; i < length; i++) {
    Serial.print(hexDigits[bytes[i] >> 4]);
    Serial.print(hexDigits[bytes[i] & 0x0F]);
  }
}

int8_t hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

bool hexToBytes(const char *text, uint8_t *output, uint8_t expectedBytes) {
  for (uint8_t i = 0; i < expectedBytes; i++) {
    int8_t high = hexValue(text[i * 2]);
    int8_t low = hexValue(text[i * 2 + 1]);
    if (high < 0 || low < 0) return false;
    output[i] = ((uint8_t)high << 4) | (uint8_t)low;
  }
  return true;
}

// =====================================================
// MINING
// =====================================================

// Builds the fixed part of the padded block for a 40-byte message:
//   challenge(32) | nonce(8) | 0x80 | zeros | 64-bit bit length (320 = 0x0140)
void prepareBlock() {
  memset(block, 0, 64);
  memcpy(block, currentChallenge, CHALLENGE_BYTES);
  block[40] = 0x80;
  block[62] = 0x01;
  block[63] = 0x40;
}

void incrementNonce() {
  for (int i = NONCE_BYTES - 1; i >= 0; i--) {
    currentNonce[i]++;
    if (currentNonce[i] != 0) return;
  }
}

void calculateHash() {

  memcpy(block + CHALLENGE_BYTES, currentNonce, NONCE_BYTES);

  H[0]=0x6a09e667;
  H[1]=0xbb67ae85;
  H[2]=0x3c6ef372;
  H[3]=0xa54ff53a;
  H[4]=0x510e527f;
  H[5]=0x9b05688c;
  H[6]=0x1f83d9ab;
  H[7]=0x5be0cd19;

  sha256Transform();

  for (int i = 0; i < 8; i++) {
    currentHash[i*4]   = H[i] >> 24;
    currentHash[i*4+1] = H[i] >> 16;
    currentHash[i*4+2] = H[i] >> 8;
    currentHash[i*4+3] = H[i];
  }
}

bool meetsDifficulty() {

  uint8_t fullZeroBytes = currentDifficulty / 8;
  uint8_t remainingBits = currentDifficulty % 8;

  for (uint8_t i = 0; i < fullZeroBytes; i++) {
    if (currentHash[i] != 0) return false;
  }

  if (remainingBits == 0) return true;

  uint8_t mask = 0xFF << (8 - remainingBits);
  return (currentHash[fullZeroBytes] & mask) == 0;
}

void reportProof() {
  Serial.print(F("MINED:"));
  printHex(currentChallenge, CHALLENGE_BYTES);
  Serial.print(':');
  Serial.print(currentDifficulty);
  Serial.print(':');
  printHex(currentNonce, NONCE_BYTES);
  Serial.print(':');
  printHex(currentHash, HASH_BYTES);
  Serial.println();
}

// Exactly ONE hash per loop pass so serial stays responsive.
void mineOneHash() {

  calculateHash();

  if (meetsDifficulty()) {
    state = STATE_AWAITING_VERIFICATION;
    reportProof();
  } else {
    incrementNonce();
  }
}

// =====================================================
// COMMAND HANDLERS
// =====================================================

void handleJob(const char *payload) {

  const char *colon = strchr(payload, ':');

  if (colon == NULL) {
    Serial.println(F("ERR:BAD_JOB:FORMAT"));
    return;
  }

  if ((size_t)(colon - payload) != (size_t)(CHALLENGE_BYTES * 2)) {
    Serial.println(F("ERR:BAD_JOB:CHALLENGE_LENGTH"));
    return;
  }

  uint8_t receivedChallenge[CHALLENGE_BYTES];

  if (!hexToBytes(payload, receivedChallenge, CHALLENGE_BYTES)) {
    Serial.println(F("ERR:BAD_JOB:CHALLENGE_HEX"));
    return;
  }

  int difficulty = atoi(colon + 1);

  if (difficulty < MIN_DIFFICULTY || difficulty > MAX_DIFFICULTY) {
    Serial.println(F("ERR:BAD_JOB:DIFFICULTY_RANGE"));
    return;
  }

  memcpy(currentChallenge, receivedChallenge, CHALLENGE_BYTES);
  memset(currentNonce, 0, NONCE_BYTES);
  currentDifficulty = (uint8_t)difficulty;
  prepareBlock();
  state = STATE_MINING;

  Serial.println(F("MINING_STARTED"));
}

void handleAccepted() {

  if (state != STATE_AWAITING_VERIFICATION) {
    Serial.println(F("IGNORED:NO_PROOF_PENDING"));
    return;
  }

  state = STATE_IDLE;
  digitalWrite(LOCK_RELAY_PIN, HIGH);
  Serial.println(F("DISARMED"));
}

void handleRejected() {
  state = STATE_IDLE;
  Serial.println(F("REJECTED_ACK"));
}

void handleStop() {

  if (state == STATE_IDLE) {
    Serial.println(F("IDLE"));
    return;
  }

  state = STATE_IDLE;
  Serial.println(F("STOPPED"));
}

void handleLine(const char *line) {

  if (strncmp(line, "JOB:", 4) == 0) {
    handleJob(line + 4);
  } else if (strcmp(line, "ACCEPTED") == 0) {
    handleAccepted();
  } else if (strcmp(line, "REJECTED") == 0) {
    handleRejected();
  } else if (strcmp(line, "STOP") == 0) {
    handleStop();
  } else {
    Serial.println(F("ERR:UNKNOWN_COMMAND"));
  }
}

// =====================================================
// SERIAL INPUT
// =====================================================

void pollSerial() {

  while (Serial.available() > 0) {

    char c = (char)Serial.read();

    if (c == '\r') continue;

    if (c == '\n') {

      if (lineOverflow) {
        Serial.println(F("ERR:LINE_TOO_LONG"));
      } else if (lineLength > 0) {
        lineBuffer[lineLength] = '\0';
        handleLine(lineBuffer);
      }

      lineLength = 0;
      lineOverflow = false;
      continue;
    }

    if (lineOverflow) continue;

    if (lineLength < LINE_BUFFER_SIZE - 1) {
      // Commands are case-insensitive: store upper case.
      lineBuffer[lineLength++] = (c >= 'a' && c <= 'z') ? c - 'a' + 'A' : c;
    } else {
      lineOverflow = true;
    }
  }
}

// =====================================================
// SETUP / LOOP
// =====================================================

void setup() {

  Serial.begin(SERIAL_BAUD);

  // Native-USB boards (Leonardo, RP2040, ...) wait briefly for the host.
  while (!Serial && millis() < 3000) {}

  pinMode(LOCK_RELAY_PIN, OUTPUT);
  digitalWrite(LOCK_RELAY_PIN, LOW);

  Serial.println(F("MINER_READY"));
}

void loop() {

  pollSerial();

  if (state == STATE_MINING) {
    mineOneHash();
  }
}
