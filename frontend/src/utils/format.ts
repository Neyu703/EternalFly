const LOCALE = "en-US";
const PERCENT_FORMAT = new Intl.NumberFormat(LOCALE, { style: "percent", maximumFractionDigits: 0 });
const FINE_PERCENT_FORMAT = new Intl.NumberFormat(LOCALE, { style: "percent", maximumFractionDigits: 1 });
// Below this, a nonzero firing rate reads "<0.1%" rather than rounding to "0%".
const SMALLEST_SHOWN_RATE = 0.001;
const INTEGER_FORMAT = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 });

/** Formats a 0..1 fraction as a whole-number percent string, e.g. 0.42 -> "42%". */
export function formatPercent(fraction: number): string {
  return PERCENT_FORMAT.format(fraction);
}

/** Formats a 0..1 firing rate as a percent with one decimal below 10% (e.g. 0.004 ->
 * "0.4%"), so a region that fires never reads "0%": tiny nonzero rates read "<0.1%". */
export function formatFiringRate(rate: number): string {
  if (rate > 0 && rate < SMALLEST_SHOWN_RATE) return `<${FINE_PERCENT_FORMAT.format(SMALLEST_SHOWN_RATE)}`;
  return (rate < 0.1 ? FINE_PERCENT_FORMAT : PERCENT_FORMAT).format(rate);
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
