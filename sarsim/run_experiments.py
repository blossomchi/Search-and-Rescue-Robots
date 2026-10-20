"""Run calibration, the main experiment (all policies x scenarios) and the sensitivity study."""
import argparse, json, os, time, itertools, numpy as np, pandas as pd
from dataclasses import asdict
from .config import *
from .env import Episode
from .sim import run_episode
from .calibrate import run_calibration

def run_block(scen, pol, n, P, sm=None, start=0):
    pid = list(POLICIES).index(pol); rows = []
    for e in range(start, start + n):
        r = run_episode(Episode(scen, e, pid, sm), POLICIES[pol], P)
        r.update(scenario=scen, policy=pol, episode=e); rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300); ap.add_argument("--sens-episodes", type=int, default=40)
    ap.add_argument("--out", default="results"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    sm = SensorModel()
    P, cal, roc = run_calibration(sm, a.quick)
    json.dump(dict(params={k: v for k, v in asdict(P).items()}, calibration=cal), open(f"{a.out}/calibration.json", "w"), indent=1)
    np.savez(f"{a.out}/roc_data.npz", **roc); print("calibrated", cal["weights"], f"{time.time()-t0:.0f}s", flush=True)
    rows = []
    for scen in SCENARIOS:
        for pol in POLICIES:
            rows += run_block(scen, pol, a.episodes, P, sm)
        print("scenario", scen, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/episodes.csv", index=False)
    # sensitivity: static vs active vs active_mt on a grid of assumed sensor parameters
    srows = []
    grid = itertools.product([0.6, 0.8, 0.95], [0.05, 0.15, 0.30], [0.8, 1.0, 1.4])
    for tpr, fpr, ts in grid:
        sm2 = SensorModel(pir_tpr=tpr, pir_fp=fpr, temp_sd_scale=ts)
        for scen in SCENARIOS:
            for pol in ("static", "active", "active_mt"):
                for r in run_block(scen, pol, a.sens_episodes, P, sm2, start=5000):
                    r.update(pir_tpr=tpr, pir_fp=fpr, temp_sd_scale=ts); srows.append(r)
        print("sens", tpr, fpr, ts, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(srows).to_csv(f"{a.out}/sensitivity.csv", index=False)
    print("done", f"{time.time()-t0:.0f}s")

if __name__ == "__main__": main()
