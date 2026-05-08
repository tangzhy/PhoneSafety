#!/usr/bin/env python3
"""
PhoneSafety: One-click data setup.

Downloads the dataset from Hugging Face and organizes it into the expected structure.

Usage:
    python3 setup_data.py
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

HF_REPO = "phonesafety-anon/PhoneSafety_Data"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

FILES = [
    "phonesafety_700.jsonl",
    "phonesafety_700_minimal_protocol.jsonl",
    "screenshots.zip",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"


def download_file(url, dest):
    """Download a file with progress."""
    print(f"  Downloading: {dest.name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"OK ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("PhoneSafety: Setting up dataset")
    print("=" * 60)

    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    # Download files
    print(f"\nDownloading from: {HF_BASE}")
    print(f"Saving to: {DATA_DIR}\n")

    for fname in FILES:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  Skipping (already exists): {fname}")
            continue
        url = f"{HF_BASE}/{fname}"
        download_file(url, dest)

    # Unzip screenshots
    zip_path = DATA_DIR / "screenshots.zip"
    if zip_path.exists() and not any(SCREENSHOTS_DIR.iterdir()):
        print(f"\n  Extracting screenshots...", end=" ", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(SCREENSHOTS_DIR)
        num_files = len(list(SCREENSHOTS_DIR.iterdir()))
        print(f"OK ({num_files} files)")
        # Remove zip after extraction
        zip_path.unlink()
        print(f"  Removed: screenshots.zip")
    elif any(SCREENSHOTS_DIR.iterdir()):
        print(f"\n  Screenshots already extracted ({len(list(SCREENSHOTS_DIR.iterdir()))} files)")

    # Verify
    print("\n" + "=" * 60)
    jsonl_path = DATA_DIR / "phonesafety_700.jsonl"
    if jsonl_path.exists():
        import json
        with open(jsonl_path) as f:
            count = sum(1 for line in f if line.strip())
        num_imgs = len(list(SCREENSHOTS_DIR.iterdir()))
        print(f"Setup complete!")
        print(f"  Cases: {count}")
        print(f"  Screenshots: {num_imgs}")
        print(f"  Data dir: {DATA_DIR}")
    else:
        print("ERROR: Setup failed - jsonl not found")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
