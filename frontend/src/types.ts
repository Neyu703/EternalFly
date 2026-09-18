/** Mirrors the backend's TickResult JSON schema (see eternalfly/server.py). */
export type TickData = {
  currentWord: string | null;
  pageProgress: number;
  emotions: Record<string, number>;
  rating0To10: number;
  regionActivity: Record<string, number>;
  wantsNewBook: boolean;
};

export const EMOTION_NAMES = [
  "joy",
  "trust",
  "fear",
  "surprise",
  "sadness",
  "disgust",
  "anger",
  "anticipation",
] as const;
