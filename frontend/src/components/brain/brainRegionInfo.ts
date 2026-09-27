/** Plain-language descriptions of the 78 FlyWire neuropil regions (codes like "ME_R"),
 * grouped by the brain areas used in the FlyWire/hemibrain nomenclature. */

// Real per-region spike rates are small fractions of their nominal 0..1 range even for a
// genuinely very active region (measured against the real cached connectome, see
// backend/scripts/calibrate_sentiment.py) - this raw rate counts as "fully active".
const NEUROPIL_ACTIVITY_CEILING = 0.08;

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

/** Describes a region code such as "ME_R" (fly's right medulla) or "EB" (unpaired). */
export function describeRegion(code: string): RegionDescription {
    const sideSuffix = code.match(/_(L|R)$/)?.[1];
    const baseCode = sideSuffix ? code.slice(0, -2) : code;
    const neuropil = NEUROPILS[baseCode];
    const area: BrainArea = neuropil ? AREAS[neuropil.area] : { name: "Neuropil", role: "" };
    return {
        code,
        name: neuropil?.name ?? baseCode,
        side: sideSuffix === "L" ? "left hemisphere" : sideSuffix === "R" ? "right hemisphere" : "midline",
        areaName: area.name,
        role: area.role,
    };
}

/** Maps a raw region spike rate onto 0..1, where 1 means "as active as regions get". */
export function normalizedRegionActivity(rawActivity: number): number {
    return Math.max(0, Math.min(1, rawActivity / NEUROPIL_ACTIVITY_CEILING));
}
