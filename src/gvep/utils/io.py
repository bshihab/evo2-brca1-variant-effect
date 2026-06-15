"""Small I/O helpers: cached file downloads.

We download large source files (the reference genome, the Findlay table) once into
data/raw/ and reuse them. Re-running is cheap and offline-friendly: if the file is
already present with the expected size, we skip the network.
"""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, dest: Path, *, expected_bytes: int | None = None) -> Path:
    """Download `url` to `dest` unless a complete copy already exists.

    If `expected_bytes` is given, an existing file of that exact size is treated as
    complete and the download is skipped (simple, dependency-free integrity gate).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_bytes is None or dest.stat().st_size == expected_bytes:
            print(f"[gvep] using cached {dest.name} ({dest.stat().st_size:,} bytes)")
            return dest
        print(f"[gvep] {dest.name} size mismatch — re-downloading")

    print(f"[gvep] downloading {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)

    print(f"[gvep] saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest
