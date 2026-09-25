/** Mirrors the backend's FrameResult JSON schema (see eternalfly/server.py's
 * frame_result_to_json). */
export type TickData = {
  currentWord: string | null;
  pageProgress: number;
  wordsRead: number;
  totalWords: number;
  emotions: Record<string, number>;
  rating0To10: number;
  /** The real injected teaching signal's own activity: reward (PAM), punishment
   * (PPL1), arousal (OA) - distinct from emotions/rating, which read the real
   * downstream MBON approach/avoidance activity those neurons teach. */
  regionActivity: Record<string, number>;
  /** Real descending/motor readouts (escape/feeding/backing/turn_left/turn_right), 0..1. */
  behaviors: Record<string, number>;
  /** Current word's real sensory channel drive levels (0..1), not a spike-rate readout. */
  senses: Record<string, number>;
  wantsNewBook: boolean;
  /** Per-neuropil live firing rate (0..1), keyed the same as neuropil-centroids.json. */
  neuropilActivity: Record<string, number>;
  stepsSimulated: number;
  spikesPerSecond: number;
  achievedWordsPerMinute: number;
  bookFinished: boolean;
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
