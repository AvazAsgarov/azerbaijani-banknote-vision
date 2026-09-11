# Mobile Unit Test Suite (mobile/__tests__/)

## Purpose
Automated test suite verifying the business logic, state machines, and protocols of the mobile companion app.

## Test Specs
- `AudioService.test.ts`: Verifies Azerbaijani TTS announcements, 3-second debounce logic, and mute state.
- `WalletService.test.ts`: Verifies running balance accumulation, duplicate filtering, history logging, and reset operations.
- `BLEService.test.ts`: Verifies ESP32-S3 8-byte binary packet parsing, checksum calculation, and coordinate normalization.
- `Denominations.test.ts`: Verifies 7 canonical AZN banknote definitions, spoken text, and risk tier assignments.
- `VisionBridgeService.test.ts`: Verifies network inference payloads and mock detection generation.
