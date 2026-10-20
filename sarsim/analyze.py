"""Statistics, tables and (black-and-white) figures from results/*.csv."""
import argparse, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 600})
RNG = np.random.default_rng(1); NB = 4000

def boot_ratio(df, num, den, n=NB):
    """Ratio of sums with percentile CI over episodes (no exclusion of zero-detection episodes)."""
    x, y = df[num].to_numpy(float), df[den].to_numpy(float); m = len(x)
    idx = RNG.integers(0, m, (n, m)); r = x[idx].sum(1) / np.maximum(y[idx].sum(1), 1e-9)
    return x.sum() / max(y.sum(), 1e-9), *np.percentile(r, [2.5, 97.5])

def boot_diff(a, b, n=NB, ratio=None):
    """Paired (shared layouts) bootstrap CI of mean(a)-mean(b)."""
    d = np.asarray(a, float) - np.asarray(b, float); idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1); return d.mean(), *np.percentile(m, [2.5, 97.5])

def cliffs(a, b):
    u = mannwhitneyu(a, b).statistic; return 2 * u / (len(a) * len(b)) - 1

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results"); a = ap.parse_args(); D = a.dir
    df = pd.read_csv(f"{D}/episodes.csv"); out = {}
    # ---- overall table (pooled over scenarios) ----
    rows = []
    for pol, g in df.groupby("policy", sort=False):
        e, lo, hi = boot_ratio(g, "energy", "tp")
        rows.append(dict(policy=pol, detection=g.detection.mean(), fp_per_ep=g.fp.mean(), coverage=g.coverage.mean(),
                         distance=g.distance.mean(), energy_per_ep=g.energy.mean(), energy_per_tp=e, ept_lo=lo, ept_hi=hi,
                         activations=g.activations.mean(), hazard_exposure=g.hazard_exposure.mean(), n=len(g)))
    T = pd.DataFrame(rows); T.to_csv(f"{D}/table_overall.csv", index=False)
    # non-battery scenarios only (battery not binding) for the main comparison
    nb = df[df.scenario != "E"]; rows = []
    for pol, g in nb.groupby("policy", sort=False):
        e, lo, hi = boot_ratio(g, "energy", "tp")
        rows.append(dict(policy=pol, detection=g.detection.mean(), fp_per_ep=g.fp.mean(), coverage=g.coverage.mean(), distance=g.distance.mean(),
                         energy_per_ep=g.energy.mean(), energy_per_tp=e, ept_lo=lo, ept_hi=hi, activations=g.activations.mean(), n=len(g)))
    pd.DataFrame(rows).to_csv(f"{D}/table_overall_noE.csv", index=False)
    # ---- Scenario E ----
    E = df[df.scenario == "E"].groupby("policy", sort=False).agg(success=("success", "mean"), stranded=("stranded", "mean"),
            detection=("detection", "mean"), fp=("fp", "mean"), energy=("energy", "mean")).reset_index(); E.to_csv(f"{D}/table_scenarioE.csv", index=False)
    # ---- per-scenario active vs static + stats ----
    srows, tests = [], []
    for sc in sorted(df.scenario.unique()):
        s = df[(df.scenario == sc) & (df.policy == "static")].sort_values("episode"); ac = df[(df.scenario == sc) & (df.policy == "active")].sort_values("episode")
        es, els, ehs = boot_ratio(s, "energy", "tp"); ea, ela, eha = boot_ratio(ac, "energy", "tp")
        srows.append(dict(scenario=sc, static_ept=es, static_lo=els, static_hi=ehs, active_ept=ea, active_lo=ela, active_hi=eha, reduction_pct=100 * (1 - ea / es),
                          static_det=s.detection.mean(), active_det=ac.detection.mean(), static_fp=s.fp.mean(), active_fp=ac.fp.mean()))
        for met in ["detection", "fp", "energy", "activations", "distance", "time"]:
            d, lo, hi = boot_diff(ac[met], s[met]); p = mannwhitneyu(ac[met], s[met]).pvalue
            tests.append(dict(scenario=sc, metric=met, diff_active_minus_static=d, ci_lo=lo, ci_hi=hi, mw_p=p, cliffs_delta=cliffs(ac[met].to_numpy(), s[met].to_numpy())))
    S = pd.DataFrame(srows); S.to_csv(f"{D}/table_per_scenario.csv", index=False)
    Tt = pd.DataFrame(tests); Tt["holm_p"] = holm(Tt.mw_p.values); Tt.to_csv(f"{D}/table_tests.csv", index=False)
    # ---- ablation (pooled over A-D,F; E reported separately) ----
    ab = pd.DataFrame([dict(policy=p, **{k: g[k].mean() for k in ["detection", "fp", "energy", "hazard_exposure"]},
                            ept=boot_ratio(g, "energy", "tp")[0]) for p, g in nb.groupby("policy", sort=False)
                       if p in ("active", "active_mt", "active_nohaz", "active_nobat", "active_noreserve", "static")]); ab.to_csv(f"{D}/table_ablation.csv", index=False)
    # ---- sensitivity ----
    sv = pd.read_csv(f"{D}/sensitivity.csv"); rows = []
    for (tpr, fpr, ts), g in sv.groupby(["pir_tpr", "pir_fp", "temp_sd_scale"]):
        r = dict(pir_tpr=tpr, pir_fp=fpr, temp_sd_scale=ts)
        stt = g[g.policy == "static"]; 
        for pol in ("active", "active_mt"):
            x = g[g.policy == pol]; r[f"{pol}_det_diff"] = x.detection.mean() - stt.detection.mean(); r[f"{pol}_fp_diff"] = x.fp.mean() - stt.fp.mean()
            r[f"{pol}_ept_red_pct"] = 100 * (1 - (x.energy.sum() / max(x.tp.sum(), 1)) / (stt.energy.sum() / max(stt.tp.sum(), 1)))
        rows.append(r)
    SV = pd.DataFrame(rows); SV.to_csv(f"{D}/table_sensitivity.csv", index=False)
    # ================= figures =================
    # Fig: energy per verified detection by scenario
    fig, ax = plt.subplots(figsize=(4.4, 2.8)); x = np.arange(len(S)); w = 0.36
    ax.bar(x - w / 2, S.static_ept, w, color="0.65", edgecolor="k", hatch="//", label="Static fusion",
           yerr=[S.static_ept - S.static_lo, S.static_hi - S.static_ept], capsize=2)
    ax.bar(x + w / 2, S.active_ept, w, color="white", edgecolor="k", label="Active (proposed)",
           yerr=[S.active_ept - S.active_lo, S.active_hi - S.active_ept], capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(S.scenario); ax.set_xlabel("Scenario"); ax.set_ylabel("Energy per verified detection\n(relative units)"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{D}/fig_energy.png"); fig.savefig(f"{D}/fig_energy.pdf"); plt.close(fig)
    # Fig: ROC
    z = np.load(f"{D}/roc_data.npz"); y = z["y"]; fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for key, ls, lab in (("cP", ":", "PIR (gate)"), ("cT", "--", "Temperature"), ("V", "-", "Fused V")):
        sc = z[key]; th = np.linspace(0, 1, 201); tpr = [(sc[y] >= t).mean() for t in th]; fpr = [(sc[~y] >= t).mean() for t in th]
        ax.plot(fpr, tpr, ls, color="k", label=lab)
    g = z["cP"] >= 0.5; ax.plot((g[~y]).mean(), g[y].mean(), "ks", ms=5, label="PIR gate at 0.5")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.6); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{D}/fig_roc.png"); fig.savefig(f"{D}/fig_roc.pdf"); plt.close(fig)
    # Fig: Pareto (recall vs energy per episode), scenarios other than E
    keep = ["random", "frontier", "static", "active", "active_mt", "frontier_R", "utility_R", "static_RB", "fmc_R"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2)); mk = dict(zip(keep, "osD^v<>PX"))
    for p_, g in nb.groupby("policy", sort=False):
        if p_ in keep: ax.plot(g.energy.mean(), g.detection.mean(), mk[p_], color="k", mfc="k" if p_.startswith("active") else "white", ms=6, label=p_)
    ax.set_xlabel("Mean energy per episode (relative units)"); ax.set_ylabel("Detection rate (recall)")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5)); fig.tight_layout()
    fig.savefig(f"{D}/fig_pareto.png"); fig.savefig(f"{D}/fig_pareto.pdf"); plt.close(fig)
    # Fig: sensitivity (PIR false-trigger = 0.15)
    sub = SV[SV.pir_fp == 0.15]; fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.6)); st = {0.8: ("-", "o"), 1.0: ("--", "s"), 1.4: (":", "^")}
    for ts, g in sub.groupby("temp_sd_scale"):
        for ax, col, yl in ((axs[0], "active_det_diff", "Detection rate: active minus static"), (axs[1], "active_ept_red_pct", "Energy-per-detection reduction (%)")):
            ax.plot(g.pir_tpr, g[col], st[ts][0], marker=st[ts][1], color="k", ms=4, label=f"temp. noise x{ts}")
            if col == "active_det_diff": ax.plot(g.pir_tpr, g["active_mt_det_diff"], st[ts][0], marker=st[ts][1], color="0.55", ms=4)
            ax.set_xlabel("Assumed PIR true-trigger probability"); ax.set_ylabel(yl)
    axs[0].axhline(0, color="0.7", lw=0.6); axs[0].legend(frameon=False, fontsize=6); fig.tight_layout(); fig.savefig(f"{D}/fig_sensitivity.png"); fig.savefig(f"{D}/fig_sensitivity.pdf"); plt.close(fig)

    # ---- extra: Scenario E success CIs / Fisher tests, pooled comparisons, gate operating point ----
    from scipy.stats import fisher_exact
    def wilson(k, n, z=1.96):
        ph = k / n; c = (ph + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h
    e = df[df.scenario == "E"]; ex = {}
    for pol, g in e.groupby("policy", sort=False):
        k, n = int(g.success.sum()), len(g); ex[pol] = dict(k=k, n=n, rate=k / n, lo=wilson(k, n)[0], hi=wilson(k, n)[1])
    pairs = [("active", "static_RB"), ("active", "active_nobat"), ("active_mt", "active"), ("active", "frontier_R"), ("active", "fmc_R"), ("static_RB", "static_R"), ("active", "active_nohaz")]
    ps = []
    for a_, b_ in pairs:
        ka, kb = ex[a_]["k"], ex[b_]["k"]; ps.append(fisher_exact([[ka, 300 - ka], [kb, 300 - kb]])[1])
    adj = holm(ps); ex["tests"] = [dict(a=a_, b=b_, diff=ex[a_]["rate"] - ex[b_]["rate"], p=p_, holm_p=h_) for (a_, b_), p_, h_ in zip(pairs, ps, adj)]
    cmp_ = []
    for a_, b_ in (("active", "static"), ("active_mt", "active"), ("active_mt", "static")):
        x = nb[nb.policy == a_].sort_values(["scenario", "episode"]); yy = nb[nb.policy == b_].sort_values(["scenario", "episode"])
        for met in ("detection", "fp", "energy", "activations"):
            d, lo, hi = boot_diff(x[met], yy[met]); cmp_.append(dict(a=a_, b=b_, metric=met, diff=d, lo=lo, hi=hi, mw_p=mannwhitneyu(x[met], yy[met]).pvalue, cliffs=cliffs(x[met].to_numpy(), yy[met].to_numpy())))
        ra, rb = boot_ratio(x, "energy", "tp"), boot_ratio(yy, "energy", "tp"); cmp_.append(dict(a=a_, b=b_, metric="energy_per_tp_ratio_of_sums", diff=ra[0] - rb[0], lo=np.nan, hi=np.nan, mw_p=np.nan, cliffs=np.nan, pct=100 * (1 - ra[0] / rb[0])))
    ex["pooled_noE"] = cmp_
    gm = z["cP"] >= 0.5; ex["gate"] = dict(tpr=float(gm[y].mean()), fpr=float((gm[~y]).mean()), precision=float(y[gm].mean()))
    json.dump(ex, open(f"{D}/extra_stats.json", "w"), indent=1, default=float); print(json.dumps(ex["tests"], indent=1, default=float)); print(ex["gate"]); print(pd.DataFrame(cmp_).round(4).to_string())
    print(T.round(3).to_string()); print(E.round(3).to_string()); print(S.round(3).to_string()); print(ab.round(3).to_string())
    print(Tt[Tt.metric.isin(["detection", "fp", "energy"])].round(4).to_string()); print(SV.round(2).to_string())

if __name__ == "__main__": main()
