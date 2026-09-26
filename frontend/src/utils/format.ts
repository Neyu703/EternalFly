const PERCENT_FORMAT = new Intl.NumberFormat("de-DE", { style: "percent", maximumFractionDigits: 0 });
const INTEGER_FORMAT = new Intl.NumberFormat("de-DE", { maximumFractionDigits: 0 });

/** Formats a 0..1 fraction as a German whole-number percent string, e.g. 0.42 -> "42 %". */
export function formatPercent(fraction: number): string {
  return PERCENT_FORMAT.format(fraction);
}

/** Formats a number with German digit grouping, e.g. 10000 -> "10.000". */
export function formatInteger(value: number): string {
  return INTEGER_FORMAT.format(value);
}

/** Formats a number with a fixed count of German decimals, e.g. (5.25, 1) -> "5,3". */
export function formatDecimal(value: number, fractionDigits: number): string {
  return value.toLocaleString("de-DE", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}
