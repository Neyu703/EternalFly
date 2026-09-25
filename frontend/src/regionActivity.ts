/** Mirrors backend/eternalfly/brain_loader.py's POSITIVE_VALENCE_CEILING/
 * NEGATIVE_VALENCE_CEILING/AROUSAL_CEILING: the raw region_activity pool rate the real
 * network achieves under strong stimulation (measured against the real cached
 * connectome by scripts/audit_brain.py), which maps to a "fully active" 100% for
 * display. Real PAM/PPL1/OA pool rates are a tiny fraction of their nominal firing-rate
 * range even under maximal drive (a few hundred neurons out of ~139k - see
 * brain_loader.py's own comment), so TickData.regionActivity is deliberately left raw
 * rather than pre-normalized (unlike behaviors, which the backend already normalizes -
 * see reading_session.py's behavior_ceilings) - every display of it needs this. */
const REGION_ACTIVITY_CEILINGS: Record<"reward" | "punishment" | "arousal", number> = {
  reward: 0.022,
  punishment: 0.07,
  arousal: 0.02,
};

/** Rescales a raw regionActivity rate to 0..1 against its measured ceiling, clamped. */
export function normalizeRegionActivity(rawRate: number, key: keyof typeof REGION_ACTIVITY_CEILINGS): number {
  const ceiling = REGION_ACTIVITY_CEILINGS[key];
  if (ceiling <= 0) return 0;
  return Math.max(0, Math.min(1, rawRate / ceiling));
}
