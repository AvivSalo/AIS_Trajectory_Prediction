"""
Turn a directory of 4-hour AIS CSVs into a UniTraj `test/` split.

This is deliberately a thin wrapper: every per-file transformation (dedup ->
1 Hz interpolation -> ego-relative Cartesian projection -> pickle) is imported
unchanged from ais_data_preprocessor_optimized, the same module that produced
train/ and val/. The only thing that differs is the tail: no 80/20 shuffle,
because a held-out test split must not be split again.

    python unitraj/tools/build_test_split.py \
        --input-dir  data/ais_4hours_2025_test_raw \
        --output-dir unitraj/data/processed_ais_4hours_optimized/test
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ais_data_preprocessor_optimized import create_dataset_files, process_file_wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-name", default="ais_dataset_test")
    ap.add_argument("--workers", type=int, default=cpu_count())
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")
    if not csv_files:
        sys.exit("nothing to process")

    args_list = [(f, args.output_dir) for f in csv_files]
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_wrapper, args_list),
                            total=len(csv_files), desc="Processing CSV files"))

    by_status = {s: [r for r in results if r[0] == s] for s in ("success", "skipped", "error")}
    print(f"\n  success: {len(by_status['success'])}"
          f"\n  skipped: {len(by_status['skipped'])}"
          f"\n  errors : {len(by_status['error'])}")
    for _, path, msg in by_status["error"][:10]:
        print(f"    {os.path.basename(path)}: {msg}")

    summary, _, _ = create_dataset_files(args.output_dir, args.dataset_name)
    print(f"\nTest split ready: {summary['meta_info']['total_frames']} scenes in {args.output_dir}")


if __name__ == "__main__":
    main()
