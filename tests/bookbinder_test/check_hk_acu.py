#!/usr/bin/env python3
"""
Quick inspector for HK g3 files to see if they contain ACU data and az/el fields.

Usage:
    python check_hk_acu.py /path/to/hk_file.g3 [more.g3 ...]

Output:
    For each file: whether ACU data is seen, frame count by type, and any az/el fields found.
"""
import argparse
from collections import Counter
import numpy as np
from spt3g import core as g3

AZ_KEYS = ["az_enc", "Corrected_Azimuth", "Azimuth", "Az"]
EL_KEYS = ["el_enc", "Corrected_Elevation", "Elevation", "El"]


def inspect_file(path):
    counts = Counter()
    found_acu = False
    found_az = set()
    found_el = set()
    try:
        g3f = g3.G3File(path)
    except Exception as exc:
        return {
            "file": path,
            "error": str(exc),
        }

    for frame in g3f:
        counts[getattr(frame, "type", "UNKNOWN")] += 1

        # Heuristic: ACU HK frames often have hkagg_type == 2 and address including 'acu'
        if "hkagg_type" in frame and "address" in frame:
            addr = str(frame.get("address", ""))
            if "acu" in addr.lower():
                found_acu = True

        # Check ancil container if present
        if "ancil" in frame:
            ancil = frame["ancil"]
            for k in AZ_KEYS:
                if k in ancil:
                    found_az.add(k)
            for k in EL_KEYS:
                if k in ancil:
                    found_el.add(k)

        # Check HK aggregator blocks if present
        if "blocks" in frame:
            try:
                blocks = frame["blocks"]
            except Exception:
                blocks = []
            for block in blocks:
                keys = list(block.keys())
                for k in AZ_KEYS:
                    if k in keys:
                        found_az.add(k)
                for k in EL_KEYS:
                    if k in keys:
                        found_el.add(k)

    return {
        "file": path,
        "counts": dict(counts),
        "found_acu": found_acu,
        "found_az": sorted(found_az),
        "found_el": sorted(found_el),
    }


def main():
    ap = argparse.ArgumentParser(description="Inspect HK g3 files for ACU az/el fields")
    ap.add_argument("files", nargs="+", help="HK .g3 files")
    args = ap.parse_args()

    for path in args.files:
        info = inspect_file(path)
        print(f"File: {info['file']}")
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue
        print(f"  Frame counts: {info['counts']}")
        print(f"  ACU detected: {info['found_acu']}")
        print(f"  AZ fields: {info['found_az'] if info['found_az'] else 'none'}")
        print(f"  EL fields: {info['found_el'] if info['found_el'] else 'none'}")
        print()


if __name__ == "__main__":
    main()
