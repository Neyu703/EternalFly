"""One-off script: download the four public FlyWire connectome files from Zenodo (no login).

Not part of the eternalfly package (this is throwaway tooling to fetch real data once
so the loader module can be written against the real, confirmed schema).
"""

from pathlib import Path

import requests
from tqdm import tqdm

from scripts.data_paths import DATA_DIR, RAW_DATA_PATHS

# The public Zenodo record holding the FlyWire files, one download URL per file name.
ZENODO_FILE_URL = "https://zenodo.org/api/records/10676866/files/{file_name}/content"


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
    DATA_DIR.mkdir(exist_ok=True)

    for destination_path in RAW_DATA_PATHS:
        download_file(ZENODO_FILE_URL.format(file_name=destination_path.name), destination_path)


if __name__ == "__main__":
    main()
