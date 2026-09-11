# Mobile Services & Business Logic (mobile/src/services/)

## Purpose
Core logic and hardware communication services managing speech output, tactile feedback,
wallet transactions, network inferences, and BLE packets.

## Key Services
- `AudioService.ts`: Text-to-Speech manager using `expo-speech` with Azerbaijani voice phrases and a 3-second debounce per denomination.
- `HapticService.ts`: Tactile vibration feedback triggering upon confirmed banknote recognition.
- `WalletService.ts`: Banknote counter and running total accumulator with transaction history.
- `BLEService.ts`: GATT client connecting to Seeed Studio XIAO ESP32-S3 smart glasses and decoding 8-byte binary packets.
- `VisionBridgeService.ts`: Network client streaming frames to local GPU inference bridge (`scripts/mobile_bridge.py`).
