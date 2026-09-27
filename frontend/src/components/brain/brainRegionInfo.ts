import { clamp } from "../../utils/math";

/** Plain-language descriptions of the 78 FlyWire neuropil regions (codes like "ME_R"),
 * grouped by the brain areas used in the FlyWire/hemibrain nomenclature. */

// Region firing rates span orders of magnitude (measured live: ~0.05 Hz in the lamina to
// ~170 Hz in the antennal lobes), so they're shown on a log scale from RATE_FLOOR_HZ (0) to
// RATE_CEILING_HZ (1). Any rate above zero shows at least MIN_FIRING_LEVEL, so every firing
// region stays visibly lit and only regions that don't fire at all stay dark.
const RATE_FLOOR_HZ = 0.1;
const RATE_CEILING_HZ = 200;
const MIN_FIRING_LEVEL = 0.15;

type BrainArea = { name: string; role: string };

const AREAS = {
    optic: { name: "Optic lobes", role: "Vision – motion, contrast and color" },
    ocellar: { name: "Ocellar ganglion", role: "Light sensing through the three simple eyes (ocelli)" },
    mushroomBody: { name: "Mushroom body", role: "Learning, memory and valuation" },
    centralComplex: { name: "Central complex", role: "Orientation, navigation and movement planning" },
    lateralComplex: { name: "Lateral complex", role: "Relay to the central complex and to motor control" },
    antennalLobe: { name: "Antennal lobe", role: "First stage of smell processing" },
    lateralHorn: { name: "Lateral horn", role: "Innate responses to odors" },
    superior: { name: "Superior neuropils", role: "Linking sensory input with internal states" },
    inferior: { name: "Inferior neuropils", role: "Connections between higher brain centers" },
    ventrolateral: { name: "Ventrolateral neuropils", role: "Processing of visual and auditory input" },
    ventromedial: { name: "Ventromedial neuropils", role: "Control of head and body movements" },
    periesophageal: { name: "Periesophageal neuropils", role: "Touch, hearing and taste" },
    gnathal: { name: "Gnathal ganglion", role: "Taste, feeding and motor control" },
} satisfies Record<string, BrainArea>;

type AreaKey = keyof typeof AREAS;

/** Name (FlyWire nomenclature) and brain area of each neuropil, keyed by its code without
 * the side suffix. */
const NEUROPILS: Record<string, { name: string; area: AreaKey }> = {
    LA: { name: "Lamina", area: "optic" },
    ME: { name: "Medulla", area: "optic" },
    AME: { name: "Accessory medulla", area: "optic" },
    LO: { name: "Lobula", area: "optic" },
    LOP: { name: "Lobula plate", area: "optic" },
    OCG: { name: "Ocellar ganglion", area: "ocellar" },
    MB_CA: { name: "Mushroom body calyx", area: "mushroomBody" },
    MB_PED: { name: "Mushroom body peduncle", area: "mushroomBody" },
    MB_VL: { name: "Mushroom body vertical lobe", area: "mushroomBody" },
    MB_ML: { name: "Mushroom body medial lobe", area: "mushroomBody" },
    EB: { name: "Ellipsoid body", area: "centralComplex" },
    FB: { name: "Fan-shaped body", area: "centralComplex" },
    PB: { name: "Protocerebral bridge", area: "centralComplex" },
    NO: { name: "Noduli", area: "centralComplex" },
    BU: { name: "Bulb", area: "lateralComplex" },
    LAL: { name: "Lateral accessory lobe", area: "lateralComplex" },
    GA: { name: "Gall", area: "lateralComplex" },
    AL: { name: "Antennal lobe", area: "antennalLobe" },
    LH: { name: "Lateral horn", area: "lateralHorn" },
    SLP: { name: "Superior lateral protocerebrum", area: "superior" },
    SIP: { name: "Superior intermediate protocerebrum", area: "superior" },
    SMP: { name: "Superior medial protocerebrum", area: "superior" },
    CRE: { name: "Crepine", area: "inferior" },
    SCL: { name: "Superior clamp", area: "inferior" },
    ICL: { name: "Inferior clamp", area: "inferior" },
    IB: { name: "Inferior bridge", area: "inferior" },
    ATL: { name: "Antler", area: "inferior" },
    AOTU: { name: "Anterior optic tubercle", area: "ventrolateral" },
    AVLP: { name: "Anterior ventrolateral protocerebrum", area: "ventrolateral" },
    PVLP: { name: "Posterior ventrolateral protocerebrum", area: "ventrolateral" },
    PLP: { name: "Posterior lateral protocerebrum", area: "ventrolateral" },
    WED: { name: "Wedge", area: "ventrolateral" },
    VES: { name: "Vest", area: "ventromedial" },
    EPA: { name: "Epaulette", area: "ventromedial" },
    GOR: { name: "Gorget", area: "ventromedial" },
    SPS: { name: "Superior posterior slope", area: "ventromedial" },
    IPS: { name: "Inferior posterior slope", area: "ventromedial" },
    SAD: { name: "Saddle", area: "periesophageal" },
    FLA: { name: "Flange", area: "periesophageal" },
    CAN: { name: "Cantle", area: "periesophageal" },
    PRW: { name: "Prow", area: "periesophageal" },
    AMMC: { name: "Antennal mechanosensory and motor center", area: "periesophageal" },
    GNG: { name: "Gnathal ganglion", area: "gnathal" },
};

/** Everything the UI shows about one region. */
export type RegionDescription = {
    code: string;
    name: string;
    side: string;
    areaName: string;
    role: string;
};

// How a region code's side suffix reads; codes without one are unpaired (midline).
const SIDE_NAMES = { L: "left hemisphere", R: "right hemisphere" } as const;

/** Describes a region code such as "ME_R" (fly's right medulla) or "EB" (unpaired). */
export function describeRegion(code: string): RegionDescription {
    const sideSuffix = code.match(/_(L|R)$/)?.[1] as keyof typeof SIDE_NAMES | undefined;
    const baseCode = sideSuffix ? code.slice(0, -2) : code;
    const neuropil = NEUROPILS[baseCode];
    const area: BrainArea = neuropil ? AREAS[neuropil.area] : { name: "Neuropil", role: "" };
    return {
        code,
        name: neuropil?.name ?? baseCode,
        side: sideSuffix ? SIDE_NAMES[sideSuffix] : "midline",
        areaName: area.name,
        role: area.role,
    };
}

/** How lit a region is (0..1) for its firing rate in Hz: log-scaled between RATE_FLOOR_HZ
 * and RATE_CEILING_HZ, at least MIN_FIRING_LEVEL while it fires at all, 0 only when it doesn't. */
export function regionActivityLevel(firingRateHz: number): number {
    if (!(firingRateHz > 0)) return 0;
    const logLevel = Math.log(firingRateHz / RATE_FLOOR_HZ) / Math.log(RATE_CEILING_HZ / RATE_FLOOR_HZ);
    return clamp(logLevel, MIN_FIRING_LEVEL, 1);
}
