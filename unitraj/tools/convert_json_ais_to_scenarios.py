import json
import math
from pathlib import Path
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert custom AIS JSONs into ScenarioNet-style scenarios for UniTraj using global coordinates"
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Folder containing your raw AIS JSON files"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Folder to write ScenarioNet-style JSON scenarios"
    )
    parser.add_argument(
        "--t_past", type=int, default=8,
        help="Number of past timesteps"
    )
    parser.add_argument(
        "--t_fut", type=int, default=12,
        help="Number of future timesteps"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(args.input_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {args.input_dir}")

    for json_file in tqdm(json_files, desc="Converting scenarios", unit="file"):
        print(f"\nProcessing file {json_file.name}...")
        data = json.loads(json_file.read_text())

        for scenario_id, scenario_dict in data.items():
            print(f"  Scenario ID: {scenario_id}")
            agents = {}
            dropped = 0

            for timestamp, record in scenario_dict.items():
                annots = record.get("annotations", {})
                print(f"    At timestamp {timestamp}: {len(annots)} annotation keys")

                for img_name, img_entry in annots.items():
                    for bbox in img_entry.get("bboxes", []):
                        fusion = bbox.get("fusion")
                        if not fusion:
                            dropped += 1
                            continue

                        # extract raw values
                        lat_raw  = fusion.get("latitude")
                        lon_raw  = fusion.get("longitude")
                        cog_raw  = fusion.get("cog")
                        sog_raw  = fusion.get("sog")

                        # skip if any required field is missing or None
                        if None in (lat_raw, lon_raw, cog_raw, sog_raw):
                            dropped += 1
                            continue

                        try:
                            frame = int(bbox.get("frame", 0))
                            aid   = str(bbox.get("track_id"))

                            lat = float(lat_raw)
                            lon = float(lon_raw)
                            x, y = lon, lat

                            yaw = math.radians(float(cog_raw))
                            speed = float(sog_raw) * 0.514444
                        except (ValueError, TypeError):
                            dropped += 1
                            continue

                        agents.setdefault(aid, []).append((frame, x, y, yaw, speed))

            print(f"    Dropped {dropped} invalid bboxes; kept {sum(len(v) for v in agents.values())} observations")

            # sort each agent’s data by frame
            for seq in agents.values():
                seq.sort(key=lambda tup: tup[0])

            # collect all unique frames
            all_frames = sorted({f for seq in agents.values() for f, *_ in seq})
            need = args.t_past + args.t_fut
            print(f"  → Found {len(all_frames)} unique frames (need at least {need})")

            if len(all_frames) < need:
                print(f"  Skipping {scenario_id}: only {len(all_frames)} frames (need {need})")
                continue

            # take the first window
            window = all_frames[:need]

            # build past/future sequences
            past, future = [], []
            for aid, seq in agents.items():
                seq_map = {f: (x, y, yaw, spd) for f, x, y, yaw, spd in seq}
                past.append([seq_map.get(f, [None]*4) for f in window[:args.t_past]])
                future.append([seq_map.get(f, [None]*4) for f in window[args.t_past:]])

            # assemble and write
            scenario = {
                "scenario_id": scenario_id,
                "agent_states": {"past": past, "future": future}
            }
            out_path = args.output_dir / f"{scenario_id}.json"
            with open(out_path, "w") as f:
                json.dump(scenario, f)
            print(f"  Wrote scenario {scenario_id} → {out_path}")

    print("\nAll done: your scenarios are ready for UniTraj.")

if __name__ == "__main__":
    main()
