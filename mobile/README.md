# AI Academy Banknote Vision - Mobile Companion App (`mobile/`)

## Purpose
Assistive mobile application for real-time recognition, audit logging, and Azerbaijani voice announcement of Azerbaijani banknotes.
Designed for visually impaired users and smart glasses integration.

## Platform Compatibility & Runtime
- **Expo SDK**: **57.0.0** (Compatible with the latest Expo Go client on iOS and Android)
- **React**: 19.2.3
- **React Native**: 0.86.3
- **TypeScript**: Strictly typed with clean modular service architecture

## Operating Modes
1. **Phone Camera Mode (Real-Time Automated Scan)**:
   - Uses device camera to continuously capture frames (every 1200ms) and send them to the local GPU inference bridge (`scripts/mobile_bridge.py`).
   - Built-in offline fallback: seamless simulation mode with interactive sample banknotes when running disconnected from the GPU bridge.
2. **Smart Glasses Mode (ESP32-S3 BLE)**:
   - Receives detection telemetry from Seeed Studio XIAO ESP32-S3 Sense smart glasses over Bluetooth Low Energy (BLE) GATT packets.
   - Real-time bounding box and audio announcement synchronized with glasses camera feed.

## Key Features & Production Resilience
- **Expo SDK 57 Full Compatibility**: Upgraded to Expo SDK 57, ensuring universal instant scanning via modern Expo Go without version mismatch errors.
- **Relational SQLite Database (`DatabaseService.ts`)**:
  - Relational SQLite schema storing all scan records with coordinates, timestamps, confidence, latency, source, and model ID.
  - Asynchronous persistent wallet balance synchronization across app restarts.
  - Aggregation statistics (`getStats()`) with breakdown per denomination.
- **Sandboxed FileSystem Archival (`ImageStorageService.ts`)**:
  - Automatically saves frame captures/thumbnails to local app storage (`scans/`), preserving database lightness.
- **4-Tier Assistive Safety Guardrail (`SafetyGuardService.ts`)**:
  - **Tier 1**: Geometric boundary verification (aspect ratio and minimum frame area).
  - **Tier 2**: Asymmetric confidence thresholds (0.50 for 1₼ up to 0.80 for 200₼) to eliminate false positives on high-value notes.
  - **Tier 3**: Temporal ring buffer majority voting (multi-frame consensus for 50₼, 100₼, 200₼).
  - **Tier 4**: Interactive voice and visual guidance in Azerbaijani ("Əsginası yaxın tutun", "Kameranı sabit saxlayın").
- **Skan Tarixçəsi & Cüzdan Audit Modalı (`ScanHistoryModal.tsx`)**:
  - Accessible high-contrast modal with color-coded banknote cards, thumbnails, and date/time metadata.
  - **"Səsləndir 🔊"** button providing Azerbaijani voice recap for visually impaired users.
  - Single-tap filtering by denomination and safe history clearing.
- **Enterprise Error Handling & Resilience**:
  - **`ErrorBoundary.tsx`**: Catches unhandled runtime exceptions and presents an elegant recovery screen with single-tap reload instead of crashing Expo Go.
  - **`ConnectionBanner.tsx`**: Real-time health check for GPU bridge connection with automatic offline fallback indicators.
- **Azerbaijani Voice Announcement (TTS)**: Spoken voice feedback ("10 Manat", "50 Manat", etc.) with 3-second anti-spam debounce.
- **Haptic Feedback**: Distinct vibration impulses upon confirmed detection.
- **AI Academy Visual Identity**: Dark navy background (`#0D0F1A`) with signature neon lime accent (`#D4F938`).

## Quick Start
```bash
# 1. Install dependencies
cd mobile
npm install --legacy-peer-deps

# 2. Configure Environment (Optional)
# Copy example configuration if overriding the default bridge endpoint
cp .env.example .env

# 3. Start Expo development server (SDK 57)
npm start

# 3. Scan the generated QR code using Expo Go on iOS or Android
```

## Running Automated Tests
```bash
cd mobile

# Jest Unit Test Suite (7 suites, 42 tests)
npm test

# Standalone Pure ESM Test Runner (23 tests)
node run_tests.mjs

# TypeScript Typecheck
npx tsc --noEmit
```
