import "./NeuralActivityChart.css";

const CHART_WIDTH = 300;
const CHART_HEIGHT = 28;

/** Builds an SVG polyline's `points` attribute from a value history, scaling the y-axis
 * against maxValue (either a fixed domain ceiling or the history's own peak). */
function buildSparklinePoints(history: number[], maxValue: number): string {
  if (history.length === 0) return "";
  return history
    .map((value, index) => {
      const x = (index / Math.max(history.length - 1, 1)) * CHART_WIDTH;
      const y = CHART_HEIGHT - (value / maxValue) * CHART_HEIGHT;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/** One small labeled sparkline: a value history plotted against either a fixed domain
 * (maxValue) or, when maxValue is omitted, the history's own observed peak. Shared by
 * NeuralActivityChart's live panel and BookOverviewPanel's end-of-book charts. */
export function Sparkline({
  label,
  history,
  maxValue,
  color,
  formatDomain,
}: {
  label: string;
  history: number[];
  maxValue?: number;
  color: string;
  formatDomain: (value: number) => string;
}) {
  const peak = maxValue ?? Math.max(...history, 0.0001);
  const points = buildSparklinePoints(history, peak);
  const latestPoint = points.split(" ").at(-1)?.split(",").map(Number);
  return (
    <div className="neural-activity-sparkline-block">
      <div className="neural-activity-chart-caption">
        <span>{label}</span>
        <span>
          0–{formatDomain(peak)} {maxValue === undefined && "· auto-scale"}
        </span>
      </div>
      <svg
        className="neural-activity-sparkline"
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
      >
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        {latestPoint && <circle cx={latestPoint[0]} cy={latestPoint[1]} r="2.5" fill={color} />}
      </svg>
    </div>
  );
}
