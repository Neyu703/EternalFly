"""One-off script: download the four public FlyWire connectome files from Zenodo (no login).

Not part of the eternalfly package (this is throwaway tooling to fetch real data once
so the loader module can be written against the real, confirmed schema).
"""

from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_FILES = {
    "proofread_connections_783.feather": "https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content",
    "per_neuron_neuropil_count_pre_783.feather": "https://zenodo.org/api/records/10676866/files/per_neuron_neuropil_count_pre_783.feather/content",
    "per_neuron_neuropil_count_post_783.feather": "https://zenodo.org/api/records/10676866/files/per_neuron_neuropil_count_post_783.feather/content",
    "proofread_root_ids_783.npy": "https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content",
}


def download_file(url: str, destination_path: Path) -> None:
    """Stream-download one file to destination_path with a progress bar, skipping if already present."""
    if destination_path.exists():
        print(f"skip (already present): {destination_path.name}")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))

    with open(destination_path, "wb") as output_file, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc=destination_path.name
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            output_file.write(chunk)
            progress_bar.update(len(chunk))


def main() -> None:
    """Download all four connectome files into backend/data/."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    for file_name, url in ZENODO_FILES.items():
        download_file(url, data_dir / file_name)


if __name__ == "__main__":
    main()
