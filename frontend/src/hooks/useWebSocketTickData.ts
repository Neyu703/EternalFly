import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { TickData } from "../types";

/** What the fly does once it finishes reading a book. */
export type AutoplayMode = "off" | "restart" | "shuffle";

/** A playback control message sent to the backend over the same WebSocket the ticks
 * arrive on (see eternalfly/server.py's stream_ticks control-message handling). Reading
 * speed is words-per-minute only now: each word always gets a fixed amount of the
 * fly's own simulated brain time (see reading_session.py), so there is no more
 * separate "speed multiplier" - words-per-minute just changes how much simulated time
 * is advanced per real second. */
export type ControlMessage =
  | { type: "set_paused"; paused: boolean }
  | { type: "set_words_per_minute"; value: number }
  | { type: "set_autoplay_mode"; mode: AutoplayMode };

/** Raw JSON shape sent by eternalfly/server.py's frame_result_to_json (snake_case dataclass fields). */
type RawTick = {
  current_word: string | null;
  page_progress: number;
  words_read: number;
  total_words: number;
  emotions: Record<string, number>;
  rating_0_10: number;
  learned_valence: number;
  region_activity: Record<string, number>;
  behaviors: Record<string, number>;
  senses: Record<string, number>;
  neuropil_activity: Record<string, number>;
  steps_simulated: number;
  spikes_per_second: number;
  achieved_words_per_minute: number;
  wants_new_book: boolean;
  book_finished: boolean;
};

/** The most recent binary spike-cloud message's decoded neuron indices (see server.py's
 * encode_fired_neuron_indices), plus a version counter bumped on every new message -
 * lets a consumer (SpikeCloud) detect "this is a new frame's data" by comparing
 * versions, without React state/a re-render for up to 20k indices every tick. */
export type FiredNeuronIndices = { indices: Uint32Array; version: number };

/** Converts one raw server tick into the frontend's camelCase TickData shape. */
function toTickData(raw: RawTick): TickData {
  return {
    currentWord: raw.current_word,
    pageProgress: raw.page_progress,
    wordsRead: raw.words_read,
    totalWords: raw.total_words,
    emotions: raw.emotions,
    rating0To10: raw.rating_0_10,
    learnedValence: raw.learned_valence,
    regionActivity: raw.region_activity,
    behaviors: raw.behaviors,
    senses: raw.senses,
    neuropilActivity: raw.neuropil_activity,
    stepsSimulated: raw.steps_simulated,
    spikesPerSecond: raw.spikes_per_second,
    achievedWordsPerMinute: raw.achieved_words_per_minute,
    wantsNewBook: raw.wants_new_book,
    bookFinished: raw.book_finished,
  };
}

/**
 * Connects to the real eternalfly WebSocket server and returns the latest tick (or `null`
 * before the first message has arrived) alongside a function for sending playback control
 * messages back over the same socket. Reconnects automatically (with a short delay) if the
 * connection drops, so a server restart doesn't permanently strand the UI — the most
 * recently sent control message of each type is replayed on every (re)connect, so a chosen
 * pause/speed/autoplay setting survives a dropped connection.
 *
 * Each simulation frame arrives as two WebSocket messages: the JSON FrameResult (a
 * string), immediately followed by a binary message carrying that frame's fired-neuron
 * indices for the spike-cloud visualization (see server.py's stream_ticks /
 * encode_fired_neuron_indices). Routed here by payload type rather than send order.
 */
export function useWebSocketTickData(url: string): {
  tick: TickData | null;
  sendControlMessage: (message: ControlMessage) => void;
  firedNeuronIndicesRef: RefObject<FiredNeuronIndices>;
} {
  const [tick, setTick] = useState<TickData | null>(null);
  const urlRef = useRef(url);
  urlRef.current = url;
  const socketRef = useRef<WebSocket | null>(null);
  const lastControlMessageByTypeRef = useRef(new Map<ControlMessage["type"], ControlMessage>());
  const firedNeuronIndicesRef = useRef<FiredNeuronIndices>({ indices: new Uint32Array(0), version: 0 });

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;

    const connect = () => {
      socket = new WebSocket(urlRef.current);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onopen = () => {
        for (const message of lastControlMessageByTypeRef.current.values()) {
          socket?.send(JSON.stringify(message));
        }
      };
      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          const raw = JSON.parse(event.data) as RawTick;
          setTick(toTickData(raw));
        } else {
          firedNeuronIndicesRef.current = {
            indices: new Uint32Array(event.data as ArrayBuffer),
            version: firedNeuronIndicesRef.current.version + 1,
          };
        }
      };
      socket.onclose = () => {
        if (!stopped) reconnectTimeoutId = setTimeout(connect, 1000);
      };
    };
    connect();

    return () => {
      stopped = true;
      if (reconnectTimeoutId) clearTimeout(reconnectTimeoutId);
      socket?.close();
    };
  }, []);

  const sendControlMessage = useCallback((message: ControlMessage) => {
    lastControlMessageByTypeRef.current.set(message.type, message);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
    }
  }, []);

  return { tick, sendControlMessage, firedNeuronIndicesRef };
}
