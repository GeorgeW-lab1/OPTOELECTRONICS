
/*
  SIMPLE SHA256 ARDUINO BOX
  -------------------------
  No external libraries.
  Serial: 115200 baud

  Commands:
    DIFF 12      sets the difficulty AND starts mining immediately
    DIFF         prints the current difficulty
    HEADER       prints HEADER:<the DATA string being hashed>
    TERMINATOR   (or STOP) aborts mining
    DATA hello
    SHA256 hello
    STATUS
    RESET
    HELP

  Mining:
    SHA256(DATA + nonce), nonce counting up from 0.
    Checks for DIFF leading zero bits.

  Output while mining:
    Nothing, until a solution is found, then one line:
      POW:<nonce>:<64 hex hash>:<padded message>:<padded nonce>:<padded hash>:<header>
    where <header> is the DATA string that was hashed (the hashed
    message is <header> followed by the decimal nonce), and each padded
    field is SHA-256 padding of its own bytes, in hex: the bytes, 0x80,
    zero bytes, then the 64-bit big-endian bit length, always a whole
    number of 64-byte blocks:
      <padded message>  header + decimal nonce
      <padded nonce>    the decimal nonce text alone
      <padded hash>     the 32 raw hash bytes
    <header> is last because it may itself contain colons.

  Note:
    This is a simple educational SHA256 proof-of-work
    implementation, not an optimized mining ASIC.
*/

#include <Arduino.h>
#include <string.h>
#include <stdint.h>

// =====================================================
// CONFIGURATION
// =====================================================

#define SERIAL_BAUD 115200
#define MAX_INPUT 160

char inputData[MAX_INPUT] = "arduino-sha256";
char serialBuffer[200];

uint32_t nonce = 0;
uint64_t hashes = 0;

uint16_t difficulty = 12;  // 0-256 (does not fit in uint8_t)

bool mining = false;
bool found = false;

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
// SHA256 FUNCTIONS
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

    uint32_t S1 =
      ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);

    uint32_t ch =
      (e & f) ^ (~e & g);

    uint32_t temp1 =
      h + S1 + ch + K[i] + W[i];

    uint32_t S0 =
      ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);

    uint32_t maj =
      (a & b) ^ (a & c) ^ (b & c);

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
// SHA256 HASH
// =====================================================

void sha256(const uint8_t *data, size_t len, uint8_t *output) {

  H[0]=0x6a09e667;
  H[1]=0xbb67ae85;
  H[2]=0x3c6ef372;
  H[3]=0xa54ff53a;
  H[4]=0x510e527f;
  H[5]=0x9b05688c;
  H[6]=0x1f83d9ab;
  H[7]=0x5be0cd19;

  size_t fullBlocks = len / 64;

  for (size_t b = 0; b < fullBlocks; b++) {
    memcpy(block, data + b*64, 64);
    sha256Transform();
  }

  size_t remaining = len % 64;

  memset(block, 0, 64);

  if (remaining > 0) {
    memcpy(block, data + fullBlocks*64, remaining);
  }

  block[remaining] = 0x80;

  if (remaining >= 56) {
    sha256Transform();
    memset(block, 0, 64);
  }

  uint64_t bitLength = (uint64_t)len * 8;

  for (int i = 0; i < 8; i++) {
    block[63-i] = (uint8_t)(bitLength >> (i*8));
  }

  sha256Transform();

  for (int i = 0; i < 8; i++) {
    output[i*4]   = H[i] >> 24;
    output[i*4+1] = H[i] >> 16;
    output[i*4+2] = H[i] >> 8;
    output[i*4+3] = H[i];
  }
}

// =====================================================
// HEX OUTPUT
// =====================================================

void printHashHex(uint8_t *hash) {

  const char hex[] = "0123456789abcdef";

  for (int i = 0; i < 32; i++) {
    Serial.print(hex[hash[i] >> 4]);
    Serial.print(hex[hash[i] & 15]);
  }
}

void printHash(uint8_t *hash) {

  printHashHex(hash);
  Serial.println();
}

// =====================================================
// DIFFICULTY CHECK
// =====================================================

bool checkDifficulty(uint8_t *hash) {

  uint16_t fullBytes = difficulty / 8;
  uint8_t remainingBits = difficulty % 8;

  for (int i = 0; i < fullBytes; i++) {
    if (hash[i] != 0) return false;
  }

  if (remainingBits > 0) {

    uint8_t mask = 0xFF << (8 - remainingBits);

    if ((hash[fullBytes] & mask) != 0) {
      return false;
    }
  }

  return true;
}

// =====================================================
// HASH INPUT + NONCE
// =====================================================

uint8_t hashOutput[32];

// Builds <inputData><decimal nonce> and returns its length.
size_t buildMessage(uint32_t n, char *message, size_t size) {

  snprintf(message, size, "%s%lu",
    inputData, (unsigned long)n);

  return strlen(message);
}

void hashNonce(uint32_t n) {

  char message[200];

  size_t len = buildMessage(n, message, sizeof(message));

  sha256(
    (const uint8_t*)message,
    len,
    hashOutput
  );
}

// Prints the SHA-256 padded form of a message as hex, streamed byte by
// byte (no extra buffer): message, 0x80, zeros, 64-bit big-endian bit length.
void printPaddedHex(const uint8_t *msg, size_t len) {

  const char hex[] = "0123456789abcdef";

  size_t total = ((len + 9 + 63) / 64) * 64;
  uint64_t bitLength = (uint64_t)len * 8;

  for (size_t i = 0; i < total; i++) {

    uint8_t b;

    if (i < len)             b = msg[i];
    else if (i == len)       b = 0x80;
    else if (i >= total - 8) b = (uint8_t)(bitLength >> ((total - 1 - i) * 8));
    else                     b = 0;

    Serial.print(hex[b >> 4]);
    Serial.print(hex[b & 15]);
  }
}

// =====================================================
// MINING
// =====================================================

void startMining() {

  // Always restart from nonce 0 so a given DATA + DIFF always
  // produces the same proof of work.
  nonce = 0;
  hashes = 0;

  found = false;
  mining = true;
}

void stopMining() {

  if (mining) {
    mining = false;

    Serial.println(F("[TERMINATED]"));
    Serial.print(F("NONCE: "));
    Serial.println(nonce);
    Serial.print(F("HASHES: "));
    Serial.println((unsigned long)hashes);
  } else {
    Serial.println(F("[IDLE]"));
  }
}

void miningStep() {

  if (!mining) return;

  // Check for serial commands before each hash.
  if (Serial.available()) return;

  hashNonce(nonce);

  hashes++;

  if (checkDifficulty(hashOutput)) {

    mining = false;
    found = true;

    char message[200];
    size_t len = buildMessage(nonce, message, sizeof(message));

    char nonceText[12];
    snprintf(nonceText, sizeof(nonceText), "%lu", (unsigned long)nonce);

    Serial.print(F("POW:"));
    Serial.print(nonce);
    Serial.print(':');
    printHashHex(hashOutput);
    Serial.print(':');
    printPaddedHex((const uint8_t*)message, len);
    Serial.print(':');
    printPaddedHex((const uint8_t*)nonceText, strlen(nonceText));
    Serial.print(':');
    printPaddedHex(hashOutput, 32);
    Serial.print(':');
    Serial.println(inputData);

    return;
  }

  nonce++;

  if (nonce == 0) {
    mining = false;
    Serial.println(F("[NONCE SPACE EXHAUSTED]"));
  }
}

// =====================================================
// COMMAND HANDLER
// =====================================================

void executeCommand(char *command) {

  while (*command == ' ') command++;

  if (strlen(command) == 0) return;

  if (
    strcmp(command, "TERMINATOR") == 0 ||
    strcmp(command, "STOP") == 0
  ) {
    stopMining();
    return;
  }

  if (strncmp(command, "DIFF", 4) == 0) {

    if (command[4] == ' ' || command[4] == '\0') {

      if (command[4] == ' ') {

        const char *arg = command + 5;
        while (*arg == ' ') arg++;

        if (*arg < '0' || *arg > '9') {
          Serial.println(F("[ERROR] Difficulty must be 0-256"));
          return;
        }

        int d = atoi(arg);

        if (d < 0 || d > 256) {
          Serial.println(F("[ERROR] Difficulty must be 0-256"));
          return;
        }

        difficulty = (uint16_t)d;

        // Receiving the difficulty is the start signal.
        startMining();

      } else {

        Serial.print(F("DIFFICULTY: "));
        Serial.println(difficulty);
      }

      return;
    }
  }

  if (strcmp(command, "HEADER") == 0) {
    Serial.print(F("HEADER:"));
    Serial.println(inputData);
    return;
  }

  if (strncmp(command, "DATA ", 5) == 0) {

    if (mining) {
      Serial.println(F("[ERROR] Stop mining first"));
      return;
    }

    strncpy(inputData, command + 5, MAX_INPUT - 1);
    inputData[MAX_INPUT - 1] = '\0';

    nonce = 0;

    Serial.print(F("[DATA SET] "));
    Serial.println(inputData);

    return;
  }

  if (strncmp(command, "SHA256 ", 7) == 0) {

    uint8_t result[32];

    sha256(
      (const uint8_t*)(command + 7),
      strlen(command + 7),
      result
    );

    printHash(result);

    return;
  }

  if (strcmp(command, "STATUS") == 0) {

    Serial.println(F("------ STATUS ------"));

    Serial.print(F("MINING: "));
    Serial.println(mining ? F("YES") : F("NO"));

    Serial.print(F("DATA: "));
    Serial.println(inputData);

    Serial.print(F("NONCE: "));
    Serial.println(nonce);

    Serial.print(F("HASHES: "));
    Serial.println((unsigned long)hashes);

    Serial.print(F("DIFF: "));
    Serial.println(difficulty);

    Serial.println(F("--------------------"));

    return;
  }

  if (strcmp(command, "RESET") == 0) {

    mining = false;
    found = false;
    nonce = 0;
    hashes = 0;

    Serial.println(F("[RESET COMPLETE]"));

    return;
  }

  if (strcmp(command, "HELP") == 0) {

    Serial.println(F("COMMANDS:"));
    Serial.println(F("TERMINATOR"));
    Serial.println(F("STOP"));
    Serial.println(F("DIFF"));
    Serial.println(F("DIFF 12   (sets difficulty and starts mining)"));
    Serial.println(F("HEADER"));
    Serial.println(F("DATA hello"));
    Serial.println(F("SHA256 hello"));
    Serial.println(F("STATUS"));
    Serial.println(F("RESET"));
    Serial.println(F("HELP"));

    return;
  }

  Serial.println(F("[UNKNOWN COMMAND]"));
}

// =====================================================
// SERIAL INPUT
// =====================================================

void readSerial() {

  static uint16_t index = 0;

  while (Serial.available()) {

    char c = Serial.read();

    if (c == '\r') continue;

    if (c == '\n') {

      serialBuffer[index] = '\0';

      // Convert command to uppercase only for
      // command matching, preserving DATA text.
      char command[200];

      strncpy(command, serialBuffer, sizeof(command)-1);
      command[sizeof(command)-1] = '\0';

      for (int i = 0; command[i]; i++) {
        if (command[i] >= 'a' && command[i] <= 'z') {
          command[i] -= 32;
        }
      }

      // DATA and SHA256 payloads are case-sensitive.
      if (strncmp(serialBuffer, "DATA ", 5) == 0) {
        executeCommand(serialBuffer);
      }
      else if (strncmp(serialBuffer, "SHA256 ", 7) == 0) {
        executeCommand(serialBuffer);
      }
      else {
        executeCommand(command);
      }

      index = 0;

      continue;
    }

    if (index < sizeof(serialBuffer)-1) {
      serialBuffer[index++] = c;
    } else {
      index = 0;
      Serial.println(F("[ERROR] Command too long"));
    }
  }
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(SERIAL_BAUD);

  while (!Serial && millis() < 3000) {}

  // Silent boot: nothing is printed until a proof of work is found.
  // Mining starts when a "DIFF <bits>" command arrives.
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {

  readSerial();

  if (mining) {
    miningStep();
  }
}
