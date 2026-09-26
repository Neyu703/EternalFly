import { useCallback, useEffect, useRef, useState } from "react";
import type { TickData } from "../types";

/** What the fly does once it finishes reading a book. */
export type AutoplayMode = "off" | "restart" | "shuffle";

/** A playback control message sent to the backend over the same WebSocket the ticks
 * arrive on (see eternalfly/server.py's stream_ticks control-message handling). */
export type ControlMessage =
  | { type: "set_paused"; paused: boolean }
  | { type: "set_speed_multiplier"; value: number }
  | { type: "set_words_per_minute"; value: number }
  | { type: "set_autoplay_mode"; mode: AutoplayMode };

/** Raw JSON shape sent by eternalfly/server.py's tick_result_to_json (snake_case dataclass fields). */
type RawTick = {
  current_word: string | null;
  page_progress: number;
  words_read: number;
  total_words: number;
  emotions: Record<string, number>;
  rating_0_10: number;
  region_activity: Record<string, number>;
  neuropil_activity: Record<string, number>;
  wants_new_book: boolean;
  book_finished: boolean;
};

/** Converts one raw server tick into the frontend's camelCase TickData shape. */
function toTickData(raw: RawTick): TickData {
  return {
    currentWord: raw.current_word,
    pageProgress: raw.page_progress,
    wordsRead: raw.words_read,
    totalWords: raw.total_words,
    emotions: raw.emotions,
    rating0To10: raw.rating_0_10,
    regionActivity: raw.region_activity,
    neuropilActivity: raw.neuropil_activity,
    wantsNewBook: raw.wants_new_book,
    bookFinished: raw.book_finished,
  };
}

/**
 * Connects to the real eternalfly WebSocket server and returns the latest tick (or `null`
 * before the first message has arrived), whether the socket is currently open, and a
 * function for sending playback control messages back over the same socket. Reconnects
 * automatically (with a short delay) if the connection drops, so a server restart doesn't
 * permanently strand the UI — the most recently sent control message of each type is
 * replayed on every (re)connect, so a chosen pause/speed/autoplay setting survives a
 * dropped connection.
 */
export function useWebSocketTickData(url: string): {
  tick: TickData | null;
  isConnected: boolean;
  sendControlMessage: (message: ControlMessage) => void;
} {
  const [tick, setTick] = useState<TickData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const urlRef = useRef(url);
  urlRef.current = url;
  const socketRef = useRef<WebSocket | null>(null);
  const lastControlMessageByTypeRef = useRef(new Map<ControlMessage["type"], ControlMessage>());

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;

    const connect = () => {
      const currentSocket = new WebSocket(urlRef.current);
      socket = currentSocket;
      socketRef.current = currentSocket;
      currentSocket.onopen = () => {
        setIsConnected(true);
        for (const message of lastControlMessageByTypeRef.current.values()) {
          currentSocket.send(JSON.stringify(message));
        }
      };
      currentSocket.onmessage = (event) => {
        const raw = JSON.parse(event.data) as RawTick;
        setTick(toTickData(raw));
      };
      currentSocket.onclose = () => {
        // A superseded socket (e.g. StrictMode's discarded first mount) can close after
        // its replacement already opened; only the current socket owns the status.
        if (socketRef.current === currentSocket) setIsConnected(false);
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

  return { tick, isConnected, sendControlMessage };
}
