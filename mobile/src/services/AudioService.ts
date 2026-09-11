/**
 * Audio Speech Synthesis (TTS) & Debounced Announcement Service.
 *
 * Provides clear Azerbaijani language voice feedback ("10 Manat", "50 Manat")
 * for assistive banknote recognition.
 * Enforces a 3-second debounce interval to prevent announcement spamming
 * when an identical banknote remains continuously visible in front of the camera.
 */

import { getDenominationById } from "../constants/denominations";

export type SpeechAdapter = (text: string, options?: { language?: string; rate?: number; pitch?: number }) => Promise<void> | void;

export class AudioService {
  private lastAnnouncedSignature: string | null = null;
  private lastAnnouncedId: number | null = null;
  private lastAnnouncedTimestamp: number = 0;
  private debounceMs: number = 3000;
  private isMuted: boolean = false;
  private language: "en" | "az" = "az";
  private speechAdapter: SpeechAdapter;

  constructor(speechAdapter?: SpeechAdapter, debounceMs: number = 3000, language: "en" | "az" = "az") {
    this.debounceMs = debounceMs;
    this.language = language;
    this.speechAdapter = speechAdapter || this.defaultSpeechAdapter;
  }

  public setLanguage(lang: "en" | "az"): void {
    this.language = lang;
  }

  public getLanguage(): "en" | "az" {
    return this.language;
  }

  private defaultSpeechAdapter: SpeechAdapter = async (text: string, options) => {
    // 1. Browser Web Speech API
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.resume();
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = options?.rate || 0.95;
        utterance.pitch = options?.pitch || 1.0;

        const voices = window.speechSynthesis.getVoices();
        const prefLang = options?.language || (this.language === "en" ? "en-US" : "az-AZ");

        if (prefLang.startsWith("en")) {
          const enVoice = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("en"));
          if (enVoice) {
            utterance.voice = enVoice;
            utterance.lang = enVoice.lang;
          } else {
            utterance.lang = "en-US";
          }
        } else {
          const azeVoice = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("az"));
          const turkVoice = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("tr"));
          if (azeVoice) {
            utterance.voice = azeVoice;
            utterance.lang = azeVoice.lang;
          } else if (turkVoice) {
            utterance.voice = turkVoice;
            utterance.lang = "tr-TR";
          } else {
            utterance.lang = "tr-TR";
          }
        }

        window.speechSynthesis.speak(utterance);
        return;
      } catch {
        // Fallback to Expo Speech
      }
    }

    // 2. React Native / Expo Go runtime
    try {
      // Note: iOS silent-mode override (playsInSilentModeIOS) requires expo-av,
      // which needs a custom native build and is NOT compatible with Expo Go.
      // TTS works in Expo Go via expo-speech — ensure phone is NOT on silent mode.

      const Speech = require("expo-speech");
      Speech.stop();

      let targetLang = options?.language || (this.language === "en" ? "en-US" : "az-AZ");
      let voiceId: string | undefined = undefined;

      try {
        if (typeof Speech.getAvailableVoicesAsync === "function") {
          const voices = await Speech.getAvailableVoicesAsync();
          if (Array.isArray(voices) && voices.length > 0) {
            if (targetLang.startsWith("en")) {
              const en = voices.find((v: any) => v.language && v.language.toLowerCase().startsWith("en"));
              if (en) {
                targetLang = en.language || "en-US";
                voiceId = en.identifier;
              } else {
                targetLang = voices[0].language || "en-US";
                voiceId = voices[0].identifier;
              }
            } else {
              const aze = voices.find((v: any) => v.language && v.language.toLowerCase().startsWith("az"));
              const tr = voices.find((v: any) => v.language && v.language.toLowerCase().startsWith("tr"));
              if (aze) {
                targetLang = aze.language;
                voiceId = aze.identifier;
              } else if (tr) {
                targetLang = tr.language || "tr-TR";
                voiceId = tr.identifier;
              } else {
                targetLang = voices[0].language || "tr-TR";
                voiceId = voices[0].identifier;
              }
            }
          }
        }
      } catch {
        // Voice query fallback
      }

      const speakOptions: any = {
        language: targetLang,
        rate: options?.rate || 0.95,
        pitch: options?.pitch || 1.0,
        onError: () => {
          try {
            Speech.speak(text, { language: "en-US", rate: 0.95 });
          } catch {}
        },
      };

      if (voiceId) {
        speakOptions.voice = voiceId;
      }

      Speech.speak(text, speakOptions);
    } catch {
      // Graceful fallback for non-native runtime
    }
  };

  /**
   * Plays a pleasant audio chime for instant feedback upon detection.
   */
  public playChime(): void {
    if (this.isMuted) return;
    if (typeof window !== "undefined" && (window.AudioContext || (window as any).webkitAudioContext)) {
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioCtx();
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.exponentialRampToValueAtTime(1320, now + 0.12);
        gain.gain.setValueAtTime(0.18, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.25);
      } catch {}
    }
  }

  /**
   * Sets whether speech output is muted.
   */
  public setMuted(muted: boolean): void {
    this.isMuted = muted;
  }

  /**
   * Returns current mute status.
   */
  public getIsMuted(): boolean {
    return this.isMuted;
  }

  /**
   * Toggles mute state and returns the new status.
   */
  public toggleMute(): boolean {
    this.isMuted = !this.isMuted;
    return this.isMuted;
  }

  /**
   * Converts a count number to Azerbaijani word prefix.
   */
  private getAzeCountPrefix(count: number): string {
    const AZE_WORDS: Record<number, string> = {
      1: "bir",
      2: "iki",
      3: "üç",
      4: "dörd",
      5: "beş",
      6: "altı",
      7: "yeddi",
      8: "səkkiz",
      9: "doqquz",
      10: "on",
    };
    const word = AZE_WORDS[count] || `${count}`;
    return `${word} ədəd`;
  }

  /**
   * Builds an assistive spoken phrase for any combination of detected banknotes,
   * supporting English ("5 Manat", "1 Manat and 5 Manat. Total 6 Manat") and Azerbaijani.
   */
  public buildSpokenPhrase(
    denominationIds: number[],
    langOverride?: "en" | "az"
  ): { phrase: string; totalSum: number } | null {
    const validIds = denominationIds.filter((id) => id >= 0 && id <= 6);
    if (validIds.length === 0) {
      return null;
    }

    const currentLang = langOverride || this.language;

    // 1. English phrase generator (default as requested: "5 Manat", "Total 15 Manat")
    if (currentLang === "en") {
      if (validIds.length === 1) {
        const config = getDenominationById(validIds[0]);
        return {
          phrase: `${config.nominalValue} Manat`,
          totalSum: config.nominalValue,
        };
      }

      const counts: Record<number, number> = {};
      let totalSum = 0;
      for (const id of validIds) {
        counts[id] = (counts[id] || 0) + 1;
        const config = getDenominationById(id);
        totalSum += config.nominalValue;
      }

      const sortedDistinctIds = Object.keys(counts)
        .map(Number)
        .sort((a, b) => getDenominationById(a).nominalValue - getDenominationById(b).nominalValue);

      const ENG_NUMS: Record<number, string> = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten",
      };

      const isAllSame = sortedDistinctIds.length === 1;
      const parts: string[] = [];

      for (const id of sortedDistinctIds) {
        const config = getDenominationById(id);
        const count = counts[id];
        if (count === 1) {
          parts.push(`${config.nominalValue} Manat`);
        } else {
          parts.push(`${ENG_NUMS[count] || count} ${config.nominalValue} Manat`);
        }
      }

      let breakdown = "";
      if (parts.length === 1) {
        breakdown = parts[0];
      } else if (parts.length === 2) {
        breakdown = `${parts[0]} and ${parts[1]}`;
      } else {
        breakdown = `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
      }

      const phrase = `${breakdown}. Total ${totalSum} Manat.`;
      return { phrase, totalSum };
    }

    // 2. Azerbaijani phrase generator
    if (validIds.length === 1) {
      const config = getDenominationById(validIds[0]);
      return {
        phrase: config.spokenAze,
        totalSum: config.nominalValue,
      };
    }

    // Count occurrences of each denomination
    const counts: Record<number, number> = {};
    let totalSum = 0;

    for (const id of validIds) {
      counts[id] = (counts[id] || 0) + 1;
      const config = getDenominationById(id);
      totalSum += config.nominalValue;
    }

    // Sort distinct denominations by nominal value ascending
    const sortedDistinctIds = Object.keys(counts)
      .map(Number)
      .sort((a, b) => getDenominationById(a).nominalValue - getDenominationById(b).nominalValue);

    const isAllSame = sortedDistinctIds.length === 1;

    const parts: string[] = [];
    for (const id of sortedDistinctIds) {
      const config = getDenominationById(id);
      const count = counts[id];
      if (count === 1) {
        parts.push(isAllSame ? `${this.getAzeCountPrefix(count)} ${config.nominalValue} manat` : `${config.nominalValue} manat`);
      } else {
        parts.push(`${this.getAzeCountPrefix(count)} ${config.nominalValue} manat`);
      }
    }

    let breakdown = "";
    if (parts.length === 1) {
      breakdown = parts[0];
    } else if (parts.length === 2) {
      breakdown = `${parts[0]} və ${parts[1]}`;
    } else {
      breakdown = `${parts.slice(0, -1).join(", ")} və ${parts[parts.length - 1]}`;
    }

    const capitalized = breakdown.charAt(0).toLocaleUpperCase("az-AZ") + breakdown.slice(1);
    const phrase = `${capitalized}. Cəmi ${totalSum} manat.`;
    return { phrase, totalSum };
  }

  /**
   * Evaluates multiple incoming detections, calculates total sum, and announces
   * in English (or Azerbaijani) if debounce criteria are satisfied.
   *
   * @param denominationIds Array of canonical denomination integer IDs (0 to 6)
   * @param nowTimestamp Current epoch timestamp in ms (defaults to Date.now())
   * @returns boolean true if speech was triggered, false if debounced or muted
   */
  public announceDetections(denominationIds: number[], nowTimestamp: number = Date.now()): boolean {
    if (this.isMuted || !denominationIds || denominationIds.length === 0) {
      return false;
    }

    const result = this.buildSpokenPhrase(denominationIds);
    if (!result) {
      return false;
    }

    const signature = [...denominationIds].sort((a, b) => a - b).join(",");
    const isSameSignature = this.lastAnnouncedSignature === signature;
    const elapsedMs = nowTimestamp - this.lastAnnouncedTimestamp;

    if (isSameSignature && elapsedMs < this.debounceMs) {
      return false;
    }

    this.lastAnnouncedSignature = signature;
    this.lastAnnouncedId = denominationIds.length === 1 ? denominationIds[0] : null;
    this.lastAnnouncedTimestamp = nowTimestamp;

    this.playChime();
    this.speechAdapter(result.phrase, {
      language: this.language === "en" ? "en-US" : "az-AZ",
      rate: 0.95,
      pitch: 1.0,
    });

    return true;
  }

  /**
   * Evaluates single incoming detection and announces denomination in Azerbaijani.
   * Backwards-compatible shorthand for announceDetections([denominationId]).
   *
   * @param denominationId Canonical denomination integer ID (0 to 6)
   * @param nowTimestamp Current epoch timestamp in ms (defaults to Date.now())
   * @returns boolean true if speech was triggered, false if debounced or muted
   */
  public announceDetection(denominationId: number, nowTimestamp: number = Date.now()): boolean {
    return this.announceDetections([denominationId], nowTimestamp);
  }

  /**
   * Resets the debounce state (e.g. when scene is cleared or wallet reset).
   */
  public resetState(): void {
    this.lastAnnouncedSignature = null;
    this.lastAnnouncedId = null;
    this.lastAnnouncedTimestamp = 0;
  }
}

export const globalAudioService = new AudioService();
