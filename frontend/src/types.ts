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
  /** Per-neuropil live firing rate in Hz (synapse-weighted over every neuron with synapses
   * there), keyed the same as neuropil-centroids.json. */
  neuropilActivity: Record<string, number>;
  /** Mean firing rate of every simulated neuron, in Hz. */
  firingRateHz: number;
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
  reward: "Firing of the reward dopamine neurons (PAM) of the mushroom body's medial lobe above their resting level",
  aversion: "Firing of the punishment dopamine neurons (PPL1) of the mushroom body's vertical lobe above their resting level",
  arousal: "Firing of the octopamine neurons driving the central complex above their resting level",
};
