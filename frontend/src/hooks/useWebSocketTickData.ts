import { useEffect, useRef, useState } from "react";
import type { TickData } from "../types";

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
  };
}

/**
 * Connects to the real eternalfly WebSocket server and returns the latest tick, or `null`
 * before the first message has arrived. Reconnects automatically (with a short delay) if the
 * connection drops, so a server restart doesn't permanently strand the UI.
 */
export function useWebSocketTickData(url: string): TickData | null {
  const [tick, setTick] = useState<TickData | null>(null);
  const urlRef = useRef(url);
  urlRef.current = url;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;

    const connect = () => {
      socket = new WebSocket(urlRef.current);
      socket.onmessage = (event) => {
        const raw = JSON.parse(event.data) as RawTick;
        setTick(toTickData(raw));
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

  return tick;
}
