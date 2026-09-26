/** Whether the user asked their OS for reduced motion (read once at startup). */
export const PREFERS_REDUCED_MOTION =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
