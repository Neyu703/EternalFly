import { useId, useState, type PointerEvent } from "react";
import "./Sparkline.css";

/** One plotted point in the SVG's 0..100 × 0..100 viewBox (y grows downward). */
type ChartPoint = { x: number; y: number };

/** Scales a value history into viewBox coordinates: x spreads the history evenly across
 * 0..100, y maps 0..maxValue onto 100..0 (values beyond the domain clamp to its edge). */
function toChartPoints(history: number[], maxValue: number): ChartPoint[] {
  const lastIndex = Math.max(history.length - 1, 1);
  return history.map((value, index) => ({
    x: (index / lastIndex) * 100,
    y: 100 - Math.min(1, Math.max(0, value / maxValue)) * 100,
  }));
}

/** SVG path data for the line through points and for the area between it and the baseline. */
function buildPaths(points: ChartPoint[]): { linePath: string; areaPath: string } {
  if (points.length === 0) return { linePath: "", areaPath: "" };
  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(" ");
  const firstX = points[0].x.toFixed(2);
  const lastX = points[points.length - 1].x.toFixed(2);
  return { linePath, areaPath: `${linePath} L${lastX},100 L${firstX},100 Z` };
}

/** One small area sparkline: a value history plotted against either a fixed domain
 * (maxValue) or, when maxValue is omitted, the history's own observed peak. The caption
 * shows the domain (unless showDomain is off, e.g. for small multiples sharing one scale),
 * or the hovered point's value while the pointer is over the plot. Shared by
 * NeuralActivityChart's live tiles and BookOverviewPanel's end-of-book charts. */
export function Sparkline({
  label,
  history,
  maxValue,
  color,
  formatValue,
  height = 28,
  showDomain = true,
}: {
  label?: string;
  history: number[];
  maxValue?: number;
  color: string;
  formatValue: (value: number) => string;
  height?: number;
  showDomain?: boolean;
}) {
  const gradientId = `sparkline-fill-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const peak = maxValue ?? Math.max(...history, Number.EPSILON);
  const points = toChartPoints(history, peak);
  const { linePath, areaPath } = buildPaths(points);
  const isHovering = hoveredIndex !== null && hoveredIndex < points.length;
  const activeIndex = isHovering ? hoveredIndex : points.length - 1;
  const activePoint = points[activeIndex];
  const domainText = `0–${formatValue(peak)}${maxValue === undefined ? " · auto" : ""}`;
  const latestValue = history.at(-1);

  /** Maps the pointer's x position onto the nearest history index for the hover readout. */
  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (points.length === 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    setHoveredIndex(Math.round(fraction * (points.length - 1)));
  }

  return (
    <div className="sparkline">
      <div className="sparkline-caption">
        {label && <span className="sparkline-label">{label}</span>}
        <span className="sparkline-readout">
          {isHovering ? formatValue(history[activeIndex]) : showDomain ? domainText : null}
        </span>
      </div>
      <div className="sparkline-plot" style={{ height }}>
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          role="img"
          aria-label={`${label ?? "Verlauf"}: aktuell ${latestValue === undefined ? "–" : formatValue(latestValue)}, Skala ${domainText}`}
          onPointerMove={handlePointerMove}
          onPointerLeave={() => setHoveredIndex(null)}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.24" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={areaPath} fill={`url(#${gradientId})`} />
          <path
            d={linePath}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        {/* HTML overlays rather than SVG shapes: the SVG stretches non-uniformly, which would squash a circle. */}
        {isHovering && <span className="sparkline-crosshair" style={{ left: `${activePoint.x}%` }} />}
        {activePoint && (
          <span
            className="sparkline-marker"
            style={{ left: `${activePoint.x}%`, top: `${activePoint.y}%`, background: color }}
          />
        )}
      </div>
    </div>
  );
}
