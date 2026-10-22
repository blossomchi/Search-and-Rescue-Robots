"""Animated side-by-side replays (GIF/MP4) and data export for the interactive HTML viewer.
Usage: python -m sarsim.visualize --scenario D --episode 17 --left static --right active --out viz"""
import argparse, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
from .config import *
from .env import Episode
from .sim import run_episode

def load_params(path="results/calibration.json"):
    return Params(**json.load(open(path))["params"])

def record(scen, ep_id, pol, P):
    pid = list(POLICIES).index(pol); ep = Episode(scen, ep_id, pid); tr = []
    res = run_episode(ep, POLICIES[pol], P, trace=tr)
    lay = dict(xy=ep.xy.tolist(), victim=ep.victim.tolist(), hazard=ep.hazard.tolist(), dheat=ep.dheat.tolist(),
               dmotion=ep.dmotion.tolist(), obst=ep.obst.tolist(), batt0=BATT_NOM * ep.sc["battery_frac"])
    return lay, tr, {k: (float(v) if v == v else None) for k, v in res.items()}

def find_episode(scen, left, right, P, rule, n=120):
    """Pick an informative episode: rule(resL, resR) -> score; highest score wins."""
    best, bs = 0, -1e9
    for e in range(n):
        _, _, a = record(scen, e, left, P); _, _, b = record(scen, e, right, P); s = rule(a, b)
        if s > bs: best, bs = e, s
    return best

def states(tr, lay, per_move=6):
    """Expand trace into drawable frames: robot xy, region status, counters, message."""
    n = len(lay["xy"]); status = ["unvisited"] * n; frames = []; pos = np.zeros(2); tp = fp = 0
    def add(xy, msg, b, en, act, t): frames.append(dict(xy=list(xy), status=list(status), msg=msg, batt=b, energy=en, act=act, tp=tp, fp=fp, time=t))
    add(pos, "start at base", lay["batt0"], 0, 0, 0)
    for ev in tr:
        if ev["t"] == "move":
            for k in range(1, per_move + 1):
                xy = pos + (np.array(ev["xy"]) - pos) * k / per_move
                add(xy, "travelling" if ev["i"] >= 0 else "exploring", ev["batt"], ev["energy"], ev["act"], ev["time"])
            pos = np.array(ev["xy"])
        elif ev["t"] == "sense":
            if status[ev["i"]] == "unvisited": status[ev["i"]] = "sensing"
            add(pos, f"sample {ev['kind']}", ev["batt"], ev["energy"], ev["act"], ev["time"])
        elif ev["t"] in ("log", "reject"):
            if ev["t"] == "log":
                status[ev["i"]] = "tp" if ev["tp"] else "fp"; tp += int(ev["tp"]); fp += int(not ev["tp"])
                msg = "VICTIM CONFIRMED" if ev["tp"] else "FALSE ALARM LOGGED"
            else:
                status[ev["i"]] = "rejected"; msg = "rejected at PIR gate" if ev.get("gate") else "rejected"
            add(pos, msg, ev["batt"], ev["energy"], ev["act"], ev["time"])
        elif ev["t"] == "return":
            msg = "returned to base" if ev["ok"] else "STRANDED: battery exhausted"
            for k in range(1, per_move + 1):
                xy = pos * (1 - k / per_move) if ev["ok"] else pos
                add(xy, msg, ev["batt"], ev["energy"], ev["act"], ev["time"])
            pos = np.zeros(2) if ev["ok"] else pos
    return frames

COL = dict(unvisited="#c8c8c8", sensing="#f2c200", rejected="#8a8a8a", tp="#1b9e3b", fp="#d62728")

def draw(ax, lay, f, trail, title, batt0):
    ax.clear(); ax.set_xlim(-11.5, 11.5); ax.set_ylim(-11.5, 11.5); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0, 0), ARENA_R, fill=False, ec="#999", lw=0.8, ls="--"))
    for i, (x, y) in enumerate(lay["xy"]):
        c = COL[f["status"][i]]; ax.plot(x, y, "o", ms=11, color=c, mec="k", mew=0.5, zorder=3)
        if lay["hazard"][i]: ax.plot(x, y, "^", ms=8, color="#ff7f0e", mec="k", mew=0.5, zorder=4)
        if lay["victim"][i]: ax.plot(x, y, "*", ms=11, color="#2166ac", mec="k", mew=0.4, zorder=4)
        if lay["dheat"][i] or lay["dmotion"][i]: ax.plot(x, y, "D", ms=6, color="#9467bd", mec="k", mew=0.4, zorder=4)
        if lay["obst"][i]: ax.plot(x, y, "s", ms=15, mfc="none", mec="k", mew=1.0, zorder=2)
    ax.plot(0, 0, "ks", ms=7); ax.text(0.3, -0.9, "base", fontsize=6)
    if len(trail) > 1: t = np.array(trail); ax.plot(t[:, 0], t[:, 1], "-", color="#4c72b0", lw=1.0, alpha=0.7)
    ax.plot(*f["xy"], "o", ms=8, color="#e377c2", mec="k", zorder=5)
    ax.set_title(title, fontsize=10, pad=16); frac = max(f["batt"], 0) / batt0
    ax.add_patch(plt.Rectangle((-11, -12.6), 22, 0.7, fc="none", ec="k", clip_on=False)); ax.add_patch(plt.Rectangle((-11, -12.6), 22 * frac, 0.7, fc="#1b9e3b" if frac > 0.2 else "#d62728", clip_on=False))
    ax.text(0, -13.9, f"battery left {100*frac:.0f}% of start   energy {f['energy']:.0f}   sensor samples {f['act']}", ha="center", fontsize=7)
    ax.text(0, 12.1, f"{f['msg']}   |   victims found {f['tp']}   false alarms {f['fp']}", ha="center", fontsize=7.5)

def animate(scen, ep_id, left, right, P, out, fps=10):
    L = [record(scen, ep_id, p, P) for p in (left, right)]; S = [states(t, l) for l, t, _ in L]
    n = max(len(s) for s in S) + 15; fig, axs = plt.subplots(1, 2, figsize=(9, 5.2))
    names = {"static": "Static fusion", "active": "Active (proposed)", "active_mt": "Active + mandatory temp.", "static_RB": "Static + reserve + battery weights",
             "frontier_R": "Frontier + reserve", "frontier": "Frontier"}
    def upd(k):
        for a, (lay, tr, res), st, pol in zip(axs, L, S, (left, right)):
            f = st[min(k, len(st) - 1)]; trail = [g["xy"] for g in st[: min(k, len(st) - 1) + 1]]
            draw(a, lay, f, trail, names.get(pol, pol), lay["batt0"])
        fig.suptitle(f"Illustrative episode: scenario {scen}, #{ep_id}. Blue star = victim, orange triangle = hazard, purple diamond = distractor, square = obstacle", fontsize=7, y=0.99)
    ani = FuncAnimation(fig, upd, frames=n, interval=1000 / fps)
    os.makedirs(out, exist_ok=True); base = f"{out}/{scen}{ep_id}_{left}_vs_{right}"
    ani.save(base + ".gif", writer=PillowWriter(fps=fps), dpi=90)
    try: ani.save(base + ".mp4", writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=130)
    except Exception as e: print("mp4 skipped:", e)
    plt.close(fig); return base, [r for _, _, r in L]

def export_case(scen, ep_id, pols, P, title):
    out = dict(title=title, scenario=scen, episode=ep_id, runs=[])
    for p in pols:
        lay, tr, res = record(scen, ep_id, p, P); out["runs"].append(dict(policy=p, layout=lay, trace=tr, result=res))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", default="D"); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--left", default="static"); ap.add_argument("--right", default="active"); ap.add_argument("--out", default="viz")
    a = ap.parse_args(); P = load_params(); print(animate(a.scenario, a.episode, a.left, a.right, P, a.out))
