/** Small colored dot keying a label to its color (the shared .swatch style). */
export function Swatch({ color }: { color: string }) {
    return <span className="swatch" style={{ background: color }} />;
}
