#!/usr/bin/env python3
"""
Download the Yelp Open Dataset and extract the 5 JSON files into data/raw/.

Usage:
    python scripts/download_data.py

By running this script you agree to Yelp's terms at https://business.yelp.com/data/resources/open-dataset/"
"""

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "Yelp-JSON.zip"

RAW_DIR.mkdir(parents=True, exist_ok=True)

def show_progress(block_num, block_size, total_size):
    """
    Print download progress on a single line that updates in place.
    """
    bytes_downloaded = block_num * block_size
    downloaded_gb = bytes_downloaded / 1_000_000_000

    if total_size <= 0:
        print(f"\r  Downloaded {downloaded_gb:.2f} GB", end="")
        return

    total_gb = total_size / 1_000_000_000
    percent_done = min(100, 100 * bytes_downloaded / total_size)

    print(f"\r  Downloaded {downloaded_gb:.2f} of {total_gb:.2f} GB  ({percent_done:.1f}%)", end="")

# --- Step 1: Download the zip file to raw directory ---
print(f"Downloading {DATA_URL} -> {ZIP_PATH}")

opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

urllib.request.urlretrieve(DATA_URL, ZIP_PATH, reporthook=show_progress)
print("\nDownload complete.")

# --- Step 2: Extract the tar out of the zip ---
# The Yelp zip contains a nested tar file (e.g. "Yelp JSON/yelp_dataset.tar")
print(f"\nExtracting tar from {ZIP_PATH}")

with zipfile.ZipFile(ZIP_PATH) as zf:
    tar_members = [
        m for m in zf.namelist()
        if (m.endswith(".tar") or m.endswith(".tgz"))
        and not m.startswith("__MACOSX")
    ]
    if not tar_members:
        raise RuntimeError(f"No .tar found in zip. Contents: {zf.namelist()}")

    tar_name = tar_members[0]
    tar_path = RAW_DIR / Path(tar_name).name
    print(f"  {tar_name} -> {tar_path}")

    with zf.open(tar_name) as src, open(tar_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

# --- Step 3: Extract the 5 JSON files out of the tar ---
print(f"\nExtracting JSON files to {RAW_DIR}")

with tarfile.open(tar_path) as tf:
    for member in tf.getmembers():
        basename = Path(member.name).name
        is_yelp_json = (
            basename.startswith("yelp_academic_dataset_")
            and basename.endswith(".json")
        )
        if not is_yelp_json:
            continue

        print(f"  {basename}")
        member.name = basename 
        tf.extract(member, RAW_DIR)

# --- Step 4: Clean up the zip and tar ---
tar_path.unlink()
ZIP_PATH.unlink()
print(f"\nRemoved {tar_path.name} and {ZIP_PATH.name}")
print(f"Done. JSON files are in {RAW_DIR}")