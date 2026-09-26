import { formatDecimal, formatInteger, formatPercent } from "./utils/format";

/** How one brain-activity metric is labeled, colored and scaled wherever it's charted
 * (the live brain panel and the end-of-book overview). */
export type MetricDefinition = {
    label: string;
    description: string;
    color: string;
    /** Fixed chart ceiling; undefined auto-scales to the history's own peak. */
    maxValue?: number;
    unit?: string;
    formatValue: (value: number) => string;
};

// The three colors are validated as a set (all pairs, dark surfaces) and are distinct
// from the UI accent, so a metric never reads as a control.
export const DOPAMINE_METRIC: MetricDefinition = {
    label: "Dopamin",
    description: "Wie gut der Fliege das Buch gefällt (0–10)",
    color: "#bf8800",
    maxValue: 10,
    unit: "/10",
    // Whole numbers (the 0–10 scale's own bounds) read as "10", live readings as "5,4".
    formatValue: (value) => (Number.isInteger(value) ? formatInteger(value) : formatDecimal(value, 1)),
};

export const AROUSAL_METRIC: MetricDefinition = {
    label: "Erregung",
    description: "Aktivität der oktopaminergen Arousal-Neuronen",
    color: "#d14186",
    maxValue: 1,
    formatValue: formatPercent,
};

export const FIRING_RATE_METRIC: MetricDefinition = {
    label: "Feuerrate",
    description: "Mittlere Spike-Rate über alle beobachteten Hirnregionen",
    color: "#439ccc",
    formatValue: formatPercent,
};
