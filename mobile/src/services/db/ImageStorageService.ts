/**
 * Image Storage & FileSystem Archival Service.
 *
 * Saves captured banknote image frames and thumbnails to local disk
 * within the app's sandboxed document directory (`scans/`), preserving
 * SQLite database lightness and query throughput.
 */

export interface IImageStorageAdapter {
  saveImage(filename: string, base64Data: string): Promise<string | null>;
  deleteImage(uri: string): Promise<boolean>;
  clearAll(): Promise<void>;
}

export class ImageStorageService implements IImageStorageAdapter {
  private baseDir: string = "scans";
  private isAvailable: boolean = true;

  constructor() {
    this.ensureDirectoryExists();
  }

  private async ensureDirectoryExists(): Promise<void> {
    try {
      const FileSystem = require("expo-file-system");
      if (FileSystem?.documentDirectory) {
        const dir = `${FileSystem.documentDirectory}${this.baseDir}/`;
        const dirInfo = await FileSystem.getInfoAsync(dir);
        if (!dirInfo.exists) {
          await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
        }
      }
    } catch {
      this.isAvailable = false;
    }
  }

  /**
   * Saves a base64 JPEG image buffer into persistent file storage.
   *
   * @param filename Unique target filename (e.g. scan_172000000_abcd.jpg)
   * @param base64Data Base64 encoded string of the image
   * @returns Device file URI or null if filesystem is inaccessible
   */
  public async saveImage(filename: string, base64Data: string): Promise<string | null> {
    if (!base64Data) return null;

    try {
      const FileSystem = require("expo-file-system");
      if (!FileSystem?.documentDirectory) return null;

      const cleanBase64 = base64Data.includes(",") ? base64Data.split(",")[1] : base64Data;
      const targetUri = `${FileSystem.documentDirectory}${this.baseDir}/${filename}`;

      await FileSystem.writeAsStringAsync(targetUri, cleanBase64, {
        encoding: FileSystem.EncodingType.Base64,
      });

      return targetUri;
    } catch {
      return null;
    }
  }

  /**
   * Deletes an individual image file.
   */
  public async deleteImage(uri: string): Promise<boolean> {
    if (!uri) return false;
    try {
      const FileSystem = require("expo-file-system");
      await FileSystem.deleteAsync(uri, { idempotent: true });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Cleans up all saved scan images from the filesystem.
   */
  public async clearAll(): Promise<void> {
    try {
      const FileSystem = require("expo-file-system");
      if (!FileSystem?.documentDirectory) return;
      const dir = `${FileSystem.documentDirectory}${this.baseDir}/`;
      await FileSystem.deleteAsync(dir, { idempotent: true });
      await this.ensureDirectoryExists();
    } catch {
      // Graceful fallback
    }
  }
}

export const globalImageStorageService = new ImageStorageService();
