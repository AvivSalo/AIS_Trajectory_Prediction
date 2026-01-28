import json
import os
from pathlib import Path

# 1. Config — adjust these as needed
INPUT_JSON_DIR = Path("/home/aviv/Data/UniTraj_data/MSC_CATERINA-29_05_24_11_32_50_411460-collision_risk-v2-day_center")
OUTPUT_SCENARIO_DIR = Path("/home/aviv/Data/UniTraj_data/MSC_CATERINA-29_05_24_11_32_50_411460-collision_risk-v2-day_center_ais_scenarios")
TPAST = 8      # number of past timesteps
TFUT = 12      # number of future timesteps

OUTPUT_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

def latlon_to_local(lat, lon):
    # placeholder: if you need metric coords, implement conversion here
    return float(lon), float(lat)

for json_file in INPUT_JSON_DIR.glob("*.json"):
    data = json.loads(json_file.read_text())
    # assume top-level has one scenario key
    scenario_id, scenario_data = next(iter(data.items()))
    # collect per-agent sequences
    agents = {}
    for frame_str, frame_info in scenario_data.items():
        frame = int(frame_str)
        for ann_key, ann in frame_info["annotations"].items():
            # only consider entries with fusion info
            fusion = ann.get("fusion")
            if not fusion:
                continue
            agent_id = ann["track_id"]
            x, y = float(fusion["ais_x"]), float(fusion["ais_y"])
            yaw = float(fusion["cog"])
            speed = float(fusion["sog"])
            agents.setdefault(agent_id, []).append((frame, x, y, yaw, speed))
    # sort each agent’s records by frame
    for seq in agents.values():
        seq.sort(key=lambda x: x[0])


    # build scenario: stack past and future
    # here: assume all agents share the same frame grid
    all_frames = sorted({f for seq in agents.values() for f, *rest in seq})
    # choose a time index so you have TPAST + TFUT frames
    # here: take the first window
    if len(all_frames) < TPAST + TFUT:
        continue  # not enough data for one scenario
    start_idx = 0
    window = all_frames[start_idx : start_idx + TPAST + TFUT]

    # helper to get state at a given frame for an agent (or None)
    def get_state(agent_seq, frame):
        for f, x, y, yaw, speed in agent_seq:
            if f == frame:
                return [x, y, yaw, speed]
        return None

    past = []
    future = []
    # build arrays: [N_agents × T × 4]
    agent_ids = list(agents.keys())
    for aid in agent_ids:
        seq = agents[aid]
        # collect past/future for this agent
        past_states = []
        fut_states = []
        for f in window[:TPAST]:
            s = get_state(seq, f)
            past_states.append(s if s else [None]*4)
        for f in window[TPAST:]:
            s = get_state(seq, f)
            fut_states.append(s if s else [None]*4)
        past.append(past_states)
        future.append(fut_states)

    scenario = {
        "scenario_id": scenario_id,
        "agent_states": {
            "past": past,
            "future": future
        }
        # omit "map" if you don’t have lane/context data
    }

    out_file = OUTPUT_SCENARIO_DIR / f"{scenario_id}.json"
    with out_file.open("w") as f:
        json.dump(scenario, f, indent=2)

    print(f"Wrote scenario {scenario_id} → {out_file}")

print("Conversion complete. You can now train UniTraj on the `ais_scenarios` folder.")
