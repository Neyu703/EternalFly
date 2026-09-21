/** Formats a 0..1 fraction as a whole-number percent string, e.g. 0.42 -> "42%". */
export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}
