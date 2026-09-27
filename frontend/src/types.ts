/** Mirrors the backend's TickResult JSON schema (see eternalfly/server.py). */
export type TickData = {
  currentWord: string | null;
  pageProgress: number;
  wordsRead: number;
  totalWords: number;
  emotions: Record<string, number>;
  rating0To10: number;
  regionActivity: Record<string, number>;
  wantsNewBook: boolean;
  /** Per-neuropil live firing rate (0..1), keyed the same as neuropil-centroids.json. */
  neuropilActivity: Record<string, number>;
  bookFinished: boolean;
};

/** The only emotional states the simulated fly brain has circuits for (see the backend's
 * emotion_decoder.compute_emotions); no human emotion categories are projected onto it. */
export const EMOTION_NAMES = ["reward", "aversion", "arousal"] as const;

export type EmotionName = (typeof EMOTION_NAMES)[number];

/** Display label per emotional state (the backend sends the lowercase keys). */
export const EMOTION_LABELS: Record<EmotionName, string> = {
  reward: "Reward",
  aversion: "Aversion",
  arousal: "Arousal",
};

/** The brain circuit behind each emotional state, shown as its tooltip. */
export const EMOTION_CIRCUITS: Record<EmotionName, string> = {
  reward: "Reward dopamine neurons (PAM) at the mushroom body's medial lobe outweigh the punishment side",
  aversion: "Punishment dopamine neurons (PPL1) at the mushroom body's vertical lobe outweigh the reward side",
  arousal: "Octopamine neurons driving the central complex",
};
