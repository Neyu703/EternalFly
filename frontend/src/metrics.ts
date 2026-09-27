import { formatDecimal, formatFiringRate, formatInteger, formatPercent } from "./utils/format";

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
    label: "Dopamine",
    description: "How much the fly likes the book (0–10)",
    color: "#bf8800",
    maxValue: 10,
    unit: "/10",
    // Whole numbers (the 0–10 scale's own bounds) read as "10", live readings as "5.4".
    formatValue: (value) => (Number.isInteger(value) ? formatInteger(value) : formatDecimal(value, 1)),
};

export const AROUSAL_METRIC: MetricDefinition = {
    label: "Arousal",
    description: "Activity of the octopaminergic arousal neurons",
    color: "#d14186",
    maxValue: 1,
    formatValue: formatPercent,
};

export const FIRING_RATE_METRIC: MetricDefinition = {
    label: "Firing rate",
    description: "Mean firing rate of all simulated neurons",
    color: "#439ccc",
    formatValue: formatFiringRate,
};
