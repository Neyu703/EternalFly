import type { ReactNode } from "react";

/** A small spinner plus what is being loaded, for a list or panel still fetching its content. */
export function LoadingText({ children }: { children: ReactNode }) {
  return (
    <p className="loading-text" role="status">
      <span className="spinner spinner--small" aria-hidden="true" />
      {children}
    </p>
  );
}
