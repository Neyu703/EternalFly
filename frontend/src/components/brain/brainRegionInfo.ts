/** Plain-language descriptions of the 78 FlyWire neuropil regions (codes like "ME_R"),
 * grouped by the brain areas used in the FlyWire/hemibrain nomenclature. */

// Real per-region spike rates are small fractions of their nominal 0..1 range even for a
// genuinely very active region (measured against the real cached connectome, see
// backend/scripts/calibrate_sentiment.py) - this raw rate counts as "fully active".
const NEUROPIL_ACTIVITY_CEILING = 0.08;

type BrainArea = { name: string; role: string };

const AREAS = {
    optic: { name: "Optische Loben", role: "Sehen – Bewegung, Kontraste und Farben" },
    ocellar: { name: "Ocellenganglion", role: "Lichtsinn der drei Punktaugen" },
    mushroomBody: { name: "Pilzkörper", role: "Lernen, Gedächtnis und Bewertung" },
    centralComplex: { name: "Zentralkomplex", role: "Orientierung, Navigation und Bewegungsplanung" },
    lateralComplex: { name: "Lateralkomplex", role: "Schaltstelle zum Zentralkomplex und zur Motorik" },
    antennalLobe: { name: "Antennallobus", role: "Erste Station der Geruchsverarbeitung" },
    lateralHorn: { name: "Lateralhorn", role: "Angeborene Reaktionen auf Gerüche" },
    superior: { name: "Obere Neuropile", role: "Verknüpfen von Sinnesreizen und inneren Zuständen" },
    inferior: { name: "Untere Neuropile", role: "Verbindungen zwischen höheren Hirnzentren" },
    ventrolateral: { name: "Ventrolaterale Neuropile", role: "Verarbeitung von Seh- und Hörreizen" },
    ventromedial: { name: "Ventromediale Neuropile", role: "Steuerung von Kopf- und Körperbewegungen" },
    periesophageal: { name: "Periösophageale Neuropile", role: "Tastsinn, Hören und Geschmack" },
    gnathal: { name: "Gnathalganglion", role: "Geschmack, Fressen und Motorik" },
} satisfies Record<string, BrainArea>;

type AreaKey = keyof typeof AREAS;

/** German name and brain area of each neuropil, keyed by its code without the side suffix. */
const NEUROPILS: Record<string, { name: string; area: AreaKey }> = {
    LA: { name: "Lamina", area: "optic" },
    ME: { name: "Medulla", area: "optic" },
    AME: { name: "Akzessorische Medulla", area: "optic" },
    LO: { name: "Lobula", area: "optic" },
    LOP: { name: "Lobulaplatte", area: "optic" },
    OCG: { name: "Ocellenganglion", area: "ocellar" },
    MB_CA: { name: "Calyx des Pilzkörpers", area: "mushroomBody" },
    MB_PED: { name: "Stiel des Pilzkörpers", area: "mushroomBody" },
    MB_VL: { name: "Vertikaler Pilzkörper-Lobus", area: "mushroomBody" },
    MB_ML: { name: "Medialer Pilzkörper-Lobus", area: "mushroomBody" },
    EB: { name: "Ellipsoidkörper", area: "centralComplex" },
    FB: { name: "Fächerförmiger Körper", area: "centralComplex" },
    PB: { name: "Protocerebralbrücke", area: "centralComplex" },
    NO: { name: "Noduli", area: "centralComplex" },
    BU: { name: "Bulbus", area: "lateralComplex" },
    LAL: { name: "Lateraler akzessorischer Lobus", area: "lateralComplex" },
    GA: { name: "Gall", area: "lateralComplex" },
    AL: { name: "Antennallobus", area: "antennalLobe" },
    LH: { name: "Lateralhorn", area: "lateralHorn" },
    SLP: { name: "Oberes laterales Protocerebrum", area: "superior" },
    SIP: { name: "Oberes intermediäres Protocerebrum", area: "superior" },
    SMP: { name: "Oberes mediales Protocerebrum", area: "superior" },
    CRE: { name: "Crepine", area: "inferior" },
    SCL: { name: "Obere Klammer", area: "inferior" },
    ICL: { name: "Untere Klammer", area: "inferior" },
    IB: { name: "Untere Brücke", area: "inferior" },
    ATL: { name: "Antler", area: "inferior" },
    AOTU: { name: "Vorderer optischer Tuberkel", area: "ventrolateral" },
    AVLP: { name: "Vorderes ventrolaterales Protocerebrum", area: "ventrolateral" },
    PVLP: { name: "Hinteres ventrolaterales Protocerebrum", area: "ventrolateral" },
    PLP: { name: "Hinteres laterales Protocerebrum", area: "ventrolateral" },
    WED: { name: "Wedge", area: "ventrolateral" },
    VES: { name: "Vest", area: "ventromedial" },
    EPA: { name: "Epaulette", area: "ventromedial" },
    GOR: { name: "Gorget", area: "ventromedial" },
    SPS: { name: "Oberer hinterer Hang", area: "ventromedial" },
    IPS: { name: "Unterer hinterer Hang", area: "ventromedial" },
    SAD: { name: "Sattel", area: "periesophageal" },
    FLA: { name: "Flange", area: "periesophageal" },
    CAN: { name: "Cantle", area: "periesophageal" },
    PRW: { name: "Prow", area: "periesophageal" },
    AMMC: { name: "Antennales Mechanozentrum", area: "periesophageal" },
    GNG: { name: "Gnathalganglion", area: "gnathal" },
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
        side: sideSuffix === "L" ? "linke Hemisphäre" : sideSuffix === "R" ? "rechte Hemisphäre" : "Mittellinie",
        areaName: area.name,
        role: area.role,
    };
}

/** Maps a raw region spike rate onto 0..1, where 1 means "as active as regions get". */
export function normalizedRegionActivity(rawActivity: number): number {
    return Math.max(0, Math.min(1, rawActivity / NEUROPIL_ACTIVITY_CEILING));
}
