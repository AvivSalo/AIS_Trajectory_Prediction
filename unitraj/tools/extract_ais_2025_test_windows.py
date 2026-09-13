"""
Extract 2025 4-hour AIS windows for the held-out TEST split.

The 2024 train/val data came from s3://datasets-for-algo/ais_trajectory_dataset/
4_hours_split_query/, produced by events_engine's wrapper_main_kepler_upload_s3.py
with chunk_hours=4. That prefix holds 2024 only; the 2025 extraction that exists on
S3 (one_year_query/) was run WITHOUT chunking and is unusable here.

This script re-runs the same events_engine extractor against InfluxDB for 2025,
one 4-hour chunk at a time, so the CSVs are byte-schema-identical to the 2024 ones
and can go through unitraj/tools/ais_data_preprocessor_optimized.py unchanged.

Window selection: for each vessel, InfluxDB is asked how many gps_position points
fall in each 4-hour bucket of 2025; only dense buckets are eligible, and the sample
is spread evenly across the vessel's active months so the test set is not clustered
in a single voyage.

Run it from the events_engine environment:

    source /home/aviv/miniconda3/etc/profile.d/conda.sh && conda activate data_env
    PYTHONPATH=/home/aviv/Projects/projects \
        python unitraj/tools/extract_ais_2025_test_windows.py --out-dir <dir>
"""

import argparse
import asyncio
import logging
import random
import sys
import warnings
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

EE_ROOT = "/home/aviv/Projects/projects"
sys.path.insert(0, f"{EE_ROOT}/events_engine/src/data_extraction")
sys.path.insert(0, EE_ROOT)

import events_engine.configs.configs  # noqa: F401  (loads configs/.env -> INFLUX_* creds)
from events_engine.src.data_extraction.async_connection_manager import (  # noqa: E402
    AsyncInfluxConnectionManager as ConnMgr,
)
from events_engine.src.data_extraction.main import main_kepler  # noqa: E402


def redirect_output_dir(out_dir):
    """Point save_kepler_data at out_dir instead of events_engine/data/raw.

    main.py imports data_collector as a plain module (its directory is on
    sys.path), so the plain sys.modules entry has to be patched too - same
    approach as events_engine's own wrapper_eval_extraction.py.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("data_collector", "events_engine.src.data_extraction.data_collector"):
        mod = sys.modules.get(key)
        if mod is not None:
            mod.data_dir_path = out_dir
    return out_dir

# The 23 vessels present in the 2024 processed dataset. Six of them stopped
# reporting during 2024 and have no 2025 data at all; they are filtered out at
# runtime by the density query rather than hard-coded, so this list stays honest.
VESSELS_2024 = [
    "maran-tanker-ares", "enesel-pantelis", "neptune-dynamis", "liberty-pride",
    "nyk-moonlight", "maran-gas-lindos", "tms-manhattan", "tms-cobia",
    "tms-athena", "maran-tanker-ariadne", "irini-n-lemos", "kyrakatingo",
    "shell-macoma", "enesel-captain-lyristis", "methane-mickie-harper",
    "nordic-ace", "mg-sakura", "earth-summit", "thenamaris-seaviolet",
    "flying-dutchman", "maran-tanker-arete", "flagship-violet", "tina-pyne",
]

# A 4-hour window at 1 Hz holds 14400 seconds. Require most of it to be present so
# a window can actually yield past_len=300 + future_len=300 samples after dedup.
MIN_POINTS_PER_WINDOW = 10000


async def dense_windows(ship):
    """Return the 4-hour bucket start times in 2025 that have dense gps coverage."""
    client = await ConnMgr.get_ship_connection(ship)
    if client is None:
        return []
    query = (
        f'SELECT COUNT("latitude") FROM "{ship}"."autogen"."gps_position" '
        f"WHERE time >= '2025-01-01T00:00:00Z' AND time < '2026-01-01T00:00:00Z' "
        f"GROUP BY time(4h)"
    )
    try:
        res = await client.query(query)
    except Exception as exc:
        logging.warning("%s: density query failed: %s", ship, exc)
        return []
    if not len(res):
        return []
    counts = res.iloc[:, 0]
    return [ts for ts, n in zip(res.index, counts) if n >= MIN_POINTS_PER_WINDOW]


def spread_over_months(windows, n, rng):
    """Pick n windows spread as evenly as possible across the months present."""
    by_month = defaultdict(list)
    for w in windows:
        by_month[(w.year, w.month)].append(w)
    for bucket in by_month.values():
        rng.shuffle(bucket)

    picked, months = [], sorted(by_month)
    while len(picked) < n and any(by_month[m] for m in months):
        for m in months:
            if by_month[m]:
                picked.append(by_month[m].pop())
                if len(picked) == n:
                    break
    return sorted(picked)


async def plan(vessels, per_vessel, seed):
    """Build the (ship, start, end) extraction list without touching the CSVs yet."""
    rng = random.Random(seed)
    jobs = []
    for ship in vessels:
        windows = await dense_windows(ship)
        if not windows:
            logging.info("%-26s no dense 2025 windows - skipping", ship)
            continue
        chosen = spread_over_months(windows, per_vessel, rng)
        logging.info("%-26s %4d dense windows -> taking %d", ship, len(windows), len(chosen))
        for start in chosen:
            jobs.append((ship, start, start + timedelta(hours=4)))
    await ConnMgr.close_all()
    return jobs


def fmt(ts):
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="directory to write the 4-hour CSVs into")
    ap.add_argument("--per-vessel", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="plan the windows, extract nothing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("events_engine").setLevel(logging.WARNING)
    logging.getLogger("aioinflux").setLevel(logging.WARNING)

    out_dir = redirect_output_dir(args.out_dir)

    jobs = asyncio.run(plan(VESSELS_2024, args.per_vessel, args.seed))
    logging.info("planned %d windows across %d vessels",
                 len(jobs), len({s for s, _, _ in jobs}))
    if args.dry_run:
        for ship, start, end in jobs:
            print(f"{ship}\t{fmt(start)}\t{fmt(end)}")
        return

    ok = fail = 0
    for i, (ship, start, end) in enumerate(jobs, 1):
        try:
            main_kepler([ship], fmt(start), fmt(end), html=False)
            ok += 1
        except Exception as exc:
            fail += 1
            logging.warning("[%d/%d] %s %s failed: %s: %s",
                            i, len(jobs), ship, fmt(start), type(exc).__name__, str(exc)[:120])
        if i % 20 == 0:
            logging.info("[%d/%d] ok=%d fail=%d", i, len(jobs), ok, fail)

    logging.info("extraction done: ok=%d fail=%d -> %s", ok, fail, out_dir)


if __name__ == "__main__":
    main()
