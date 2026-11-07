#!/usr/bin/env python3
"""Download the 99-map EModelS/EModelX test set from EMDB.

Reads data/test99.csv (columns: pdb_id, emdb_id, resolution) and writes
maps to --out-dir as emd_{emdb_id}.map.gz.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import pandas as pd

EMDB_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emid}/map/emd_{emid}.map.gz"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="data/test99.csv")
    p.add_argument("--out-dir", default="data/maps_test99")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of maps")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    rows = list(zip(df["pdb_id"], df["emdb_id"]))
    if args.limit:
        rows = rows[: args.limit]

    for pdb_id, emid in rows:
        emid = str(emid).lstrip("0") or "0"
        # EMDB folders use zero-padded ids in some mirrors; try both.
        candidates = [
            EMDB_URL.format(emid=str(emid)),
            EMDB_URL.format(emid=f"{int(emid):04d}"),
        ]
        out = os.path.join(args.out_dir, f"emd_{int(emid)}.map.gz")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"[skip] {pdb_id} -> {out}")
            continue
        ok = False
        for url in candidates:
            try:
                print(f"[get ] {pdb_id} {url}")
                urllib.request.urlretrieve(url, out)
                ok = True
                break
            except Exception as exc:
                print(f"[warn] {url}: {exc}")
        if not ok:
            print(f"[fail] {pdb_id} emdb={emid}")


if __name__ == "__main__":
    main()
