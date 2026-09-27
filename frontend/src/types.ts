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
