import { useCallback, useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { BrainGlow, CLICK_SLOP_PX, type NeuropilActivity } from "./BrainGlow";
import type { BrainRegion } from "./brain/brainGeometry";
import { describeRegion, normalizedRegionActivity } from "./brain/brainRegionInfo";
import { formatPercent } from "../utils/format";
import { PREFERS_REDUCED_MOTION } from "../utils/motion";
import { CloseIcon } from "./icons";
import "./BrainScene.css";

// Pause the idle auto-rotation for a while after the user orbits, zooms or hovers.
const IDLE_RESUME_MS = 4000;
const TOOLTIP_OFFSET_PX = 14;
// Past these distances from the right/bottom edge the tooltip opens toward the inside.
const TOOLTIP_FLIP_X_PX = 260;
const TOOLTIP_FLIP_Y_PX = 90;

/** The hovered region and where its tooltip goes, relative to the scene's top-left. */
type HoverState = { code: string; style: React.CSSProperties };

/** Places the tooltip beside the pointer, flipping it inward near the scene's edges. */
function tooltipPlacement(x: number, y: number, width: number, height: number): React.CSSProperties {
  const horizontal = x > width - TOOLTIP_FLIP_X_PX ? { right: width - x + TOOLTIP_OFFSET_PX } : { left: x + TOOLTIP_OFFSET_PX };
  const vertical = y > height - TOOLTIP_FLIP_Y_PX ? { bottom: height - y + TOOLTIP_OFFSET_PX } : { top: y + TOOLTIP_OFFSET_PX };
  return { ...horizontal, ...vertical };
}

/** 3D panel: the live brain, slowly turning while idle. Hovering a region names it,
 * clicking one focuses the camera on it and opens a card with what it does and how
 * active it is; Escape, the card's close button or a click on empty space go back. */
export function BrainScene({ activity }: { activity?: NeuropilActivity }) {
  const [hover, setHover] = useState<HoverState | null>(null);
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const [regionColors, setRegionColors] = useState<Record<string, string>>({});
  const [isIdle, setIsIdle] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const idleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pointerDownRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!selectedCode) return;
    /** Clears the selection on Escape. */
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setSelectedCode(null);
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedCode]);

  useEffect(() => () => clearTimeout(idleTimeoutRef.current ?? undefined), []);

  /** Stops the idle rotation now and resumes it IDLE_RESUME_MS after the last activity. */
  const markInteraction = useCallback(() => {
    setIsIdle(false);
    clearTimeout(idleTimeoutRef.current ?? undefined);
    idleTimeoutRef.current = setTimeout(() => setIsIdle(true), IDLE_RESUME_MS);
  }, []);

  const handleHoverRegion = useCallback(
    (code: string | null, clientX: number, clientY: number) => {
      const bounds = containerRef.current?.getBoundingClientRect();
      if (!code || !bounds) {
        setHover(null);
        return;
      }
      markInteraction();
      setHover({
        code,
        style: tooltipPlacement(clientX - bounds.left, clientY - bounds.top, bounds.width, bounds.height),
      });
    },
    [markInteraction],
  );

  const handleRegionsReady = useCallback((regions: BrainRegion[]) => {
    setRegionColors(Object.fromEntries(regions.map((region) => [region.code, region.cssColor])));
  }, []);

  /** Clicking empty space (not an orbit drag) clears the selection. */
  function handlePointerMissed(event: MouseEvent) {
    const moved = Math.hypot(event.clientX - pointerDownRef.current.x, event.clientY - pointerDownRef.current.y);
    if (moved <= CLICK_SLOP_PX) setSelectedCode(null);
  }

  return (
    <div
      ref={containerRef}
      className="brain-scene"
      onPointerDown={(event) => (pointerDownRef.current = { x: event.clientX, y: event.clientY })}
      onPointerLeave={() => setHover(null)}
    >
      <Canvas camera={{ position: [0, 0, 2.6], fov: 50 }} onPointerMissed={handlePointerMissed}>
        <group scale={0.28}>
          <BrainGlow
            activity={activity}
            hoveredCode={hover?.code ?? null}
            selectedCode={selectedCode}
            onHoverRegion={handleHoverRegion}
            onSelectRegion={(code) => setSelectedCode((current) => (current === code ? null : code))}
            onRegionsReady={handleRegionsReady}
          />
        </group>
        <OrbitControls
          makeDefault
          enablePan={false}
          minDistance={1.2}
          maxDistance={6}
          autoRotate={isIdle && !selectedCode && !PREFERS_REDUCED_MOTION}
          autoRotateSpeed={0.5}
          onStart={markInteraction}
        />
      </Canvas>

      {hover && hover.code !== selectedCode && (
        <RegionTooltip code={hover.code} activity={activity?.[hover.code]} style={hover.style} />
      )}
      {selectedCode && (
        <RegionCard
          code={selectedCode}
          color={regionColors[selectedCode]}
          activity={activity?.[selectedCode]}
          onClose={() => setSelectedCode(null)}
        />
      )}
    </div>
  );
}

/** Small label that follows the pointer while it rests on a region. */
function RegionTooltip({ code, activity, style }: { code: string; activity?: number; style: React.CSSProperties }) {
  const region = describeRegion(code);
  return (
    <div className="region-tooltip" style={style} role="tooltip">
      <span className="region-tooltip-name">{region.name}</span>
      <span className="region-tooltip-meta">
        {region.areaName} · {region.side}
      </span>
      {activity !== undefined && <span className="region-tooltip-meta">Firing rate {formatPercent(activity)}</span>}
    </div>
  );
}

/** Details of the selected region: name, brain area, what that area does, and its live
 * firing rate as a bar scaled to how active regions get. */
function RegionCard({
  code,
  color,
  activity,
  onClose,
}: {
  code: string;
  color?: string;
  activity?: number;
  onClose: () => void;
}) {
  const region = describeRegion(code);
  const barFraction = activity !== undefined ? normalizedRegionActivity(activity) : 0;
  return (
    <aside className="region-card" aria-label={`Brain region ${region.name}`}>
      <div className="region-card-header">
        <div className="region-card-heading">
          <span className="overline">
            {region.areaName === region.name ? region.code : `${region.areaName} · ${region.code}`}
          </span>
          <span className="region-card-name">{region.name}</span>
          <span className="region-card-side">{region.side}</span>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Clear selection">
          <CloseIcon />
        </button>
      </div>
      {region.role && <p className="region-card-role">{region.role}</p>}
      <div className="region-card-activity">
        <span className="region-card-activity-label">Firing rate</span>
        <span className="region-card-activity-track">
          <span className="region-card-activity-fill" style={{ width: `${barFraction * 100}%`, background: color }} />
        </span>
        <span className="region-card-activity-value">{activity !== undefined ? formatPercent(activity) : "—"}</span>
      </div>
    </aside>
  );
}
