/**
 * Central Type Definitions & Domain Models for Banknote Vision Mobile Companion.
 */

// Denominations
export type { DenominationConfig } from "../constants/denominations";

// Models
export type { ModelDescriptor } from "../constants/models";

// BLE & Smart Glasses
export type { ConnectionState, GlassesDetection } from "../services/BLEService";

// Computer Vision Bridge
export type { BoundingBoxDetection, BridgeConfig } from "../services/VisionBridgeService";

// Audio & Haptics
export type { SpeechAdapter } from "../services/AudioService";
export type { HapticAdapter } from "../services/HapticService";

// Wallet
export type { WalletTransaction } from "../services/WalletService";

// SQLite Database & Archival
export type { ScanRecord, ScanStats } from "../services/db/DatabaseService";
export type { IDatabaseAdapter } from "../services/db/DatabaseAdapters";
export type { IImageStorageAdapter } from "../services/db/ImageStorageService";

// Safety Guardrail
export type {
  BoundingBox,
  GuardrailConfig,
  RawDetectionInput,
  VerificationResult,
} from "../services/SafetyGuardService";
