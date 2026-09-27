const LOCALE = "en-US";
const PERCENT_FORMAT = new Intl.NumberFormat(LOCALE, { style: "percent", maximumFractionDigits: 0 });
// Below this, a nonzero firing rate reads "<0.1 Hz" rather than rounding to "0 Hz".
const SMALLEST_SHOWN_RATE_HZ = 0.1;
// From this rate on, whole hertz are precise enough.
const WHOLE_HERTZ_FROM = 10;
const INTEGER_FORMAT = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 });

/** Formats a 0..1 fraction as a whole-number percent string, e.g. 0.42 -> "42%". */
export function formatPercent(fraction: number): string {
  return PERCENT_FORMAT.format(fraction);
}

/** Formats a firing rate in Hz with one decimal below 10 Hz and whole hertz above (e.g.
 * 5.03 -> "5.0 Hz", 168.2 -> "168 Hz"), so a region that fires never reads "0 Hz": tiny
 * nonzero rates read "<0.1 Hz". */
export function formatFiringRate(rateHz: number): string {
  if (rateHz > 0 && rateHz < SMALLEST_SHOWN_RATE_HZ) return `<${formatDecimal(SMALLEST_SHOWN_RATE_HZ, 1)} Hz`;
  return `${rateHz < WHOLE_HERTZ_FROM ? formatDecimal(rateHz, 1) : formatInteger(rateHz)} Hz`;
}

/** Formats a number with digit grouping, e.g. 10000 -> "10,000". */
export function formatInteger(value: number): string {
  return INTEGER_FORMAT.format(value);
}

/** Formats a number with a fixed count of decimals, e.g. (5.25, 1) -> "5.3". */
export function formatDecimal(value: number, fractionDigits: number): string {
  return value.toLocaleString(LOCALE, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}
