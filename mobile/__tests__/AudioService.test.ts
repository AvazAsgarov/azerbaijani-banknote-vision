/**
 * Unit Tests for AudioService (Speech Synthesis & Debounce Logic).
 */

import { AudioService } from "../src/services/AudioService";

describe("AudioService", () => {
  let mockSpeech: jest.Mock;
  let audioService: AudioService;

  beforeEach(() => {
    mockSpeech = jest.fn();
    audioService = new AudioService(mockSpeech, 3000);
  });

  test("announces new denomination in Azerbaijani immediately", () => {
    const triggered = audioService.announceDetection(2, 1000); // 10 AZN
    expect(triggered).toBe(true);
    expect(mockSpeech).toHaveBeenCalledTimes(1);
    expect(mockSpeech).toHaveBeenCalledWith(
      "On Manat",
      expect.objectContaining({ language: "az-AZ" })
    );
  });

  test("debounces repeated announcements of the exact same denomination within 3000ms", () => {
    // First detection at t = 1000
    expect(audioService.announceDetection(4, 1000)).toBe(true); // 50 AZN
    expect(mockSpeech).toHaveBeenCalledTimes(1);
    expect(mockSpeech).toHaveBeenCalledWith("Əlli Manat", expect.anything());

    // Second detection at t = 2500 (< 3000ms elapsed)
    expect(audioService.announceDetection(4, 2500)).toBe(false);
    expect(mockSpeech).toHaveBeenCalledTimes(1); // No new call

    // Third detection at t = 4200 (> 3000ms elapsed)
    expect(audioService.announceDetection(4, 4200)).toBe(true);
    expect(mockSpeech).toHaveBeenCalledTimes(2);
  });

  test("announces immediately when denomination changes even within debounce interval", () => {
    expect(audioService.announceDetection(2, 1000)).toBe(true); // 10 AZN
    expect(mockSpeech).toHaveBeenLastCalledWith("On Manat", expect.anything());

    // Rapid change to 100 AZN at t = 1500
    expect(audioService.announceDetection(5, 1500)).toBe(true); // 100 AZN
    expect(mockSpeech).toHaveBeenLastCalledWith("Yüz Manat", expect.anything());
    expect(mockSpeech).toHaveBeenCalledTimes(2);
  });

  test("suppresses all announcements when muted", () => {
    audioService.setMuted(true);
    expect(audioService.getIsMuted()).toBe(true);

    expect(audioService.announceDetection(6, 1000)).toBe(false);
    expect(mockSpeech).not.toHaveBeenCalled();

    audioService.toggleMute();
    expect(audioService.getIsMuted()).toBe(false);
    expect(audioService.announceDetection(6, 1000)).toBe(true);
    expect(mockSpeech).toHaveBeenCalledTimes(1);
  });

  test("rejects invalid denomination id", () => {
    expect(audioService.announceDetection(99, 1000)).toBe(false);
    expect(mockSpeech).not.toHaveBeenCalled();
  });

  test("announces multiple banknotes with combined total sum in Azerbaijani (5 AZN + 10 AZN = 15 AZN)", () => {
    const triggered = audioService.announceDetections([1, 2], 1000); // 5 AZN & 10 AZN
    expect(triggered).toBe(true);
    expect(mockSpeech).toHaveBeenCalledTimes(1);
    expect(mockSpeech).toHaveBeenCalledWith(
      "5 manat və 10 manat. Cəmi 15 manat.",
      expect.objectContaining({ language: "az-AZ" })
    );
  });

  test("debounces repeated announcements of the same multi-banknote set within 3000ms", () => {
    expect(audioService.announceDetections([1, 2], 1000)).toBe(true);
    expect(mockSpeech).toHaveBeenCalledTimes(1);

    // Duplicate set at t = 2200ms (< 3000ms)
    expect(audioService.announceDetections([1, 2], 2200)).toBe(false);
    expect(mockSpeech).toHaveBeenCalledTimes(1);

    // After 3000ms (at t = 4500ms) -> allowed
    expect(audioService.announceDetections([1, 2], 4500)).toBe(true);
    expect(mockSpeech).toHaveBeenCalledTimes(2);
  });

  test("immediately announces when multi-banknote set changes (e.g. adding 20 AZN note)", () => {
    audioService.announceDetections([1, 2], 1000); // 5 AZN + 10 AZN = 15 AZN
    expect(mockSpeech).toHaveBeenLastCalledWith("5 manat və 10 manat. Cəmi 15 manat.", expect.anything());

    // New note added at t = 1500ms (5 + 10 + 20 = 35 AZN)
    expect(audioService.announceDetections([1, 2, 3], 1500)).toBe(true);
    expect(mockSpeech).toHaveBeenLastCalledWith(
      "5 manat, 10 manat və 20 manat. Cəmi 35 manat.",
      expect.anything()
    );
    expect(mockSpeech).toHaveBeenCalledTimes(2);
  });
});
