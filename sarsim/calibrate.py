"""Calibration on episodes DISJOINT from evaluation (seed base CAL_SEED): fusion weights, IG(a) table, battery constants."""
import json, numpy as np
from dataclasses import asdict
from .config import *
from . import sensors as S
from .env import Episode
from .sim import run_episode

def population(sm, n_per_scen=300):
    """Region flags pooled over all six scenarios (disjoint calibration seeds)."""
    keys = ("victim", "heat", "motion", "hazard"); out = {k: [] for k in keys}
    for sc in SCENARIOS:
        for e in range(n_per_scen):
            ep = Episode(sc, e, 0, sm, base_seed=CAL_SEED)
            for k in keys: out[k].append(getattr(ep, k))
    return {k: np.concatenate(v) for k, v in out.items()}

def f1(pred, y):
    tp = (pred & y).sum(); p = tp / max(pred.sum(), 1); r = tp / max(y.sum(), 1)
    return 0 if p + r == 0 else 2 * p * r / (p + r)

def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def fit_weights(sm, pop, P, rng):
    n = len(pop["victim"]); y = pop["victim"]
    cP = S.pir(rng, sm, pop["motion"], ATT_SITE); cT = S.temp(rng, sm, pop["heat"], ATT_SITE)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    f = [f1(w * cP + (1 - w) * cT > P.tau_hi, y) for w in grid]
    w1 = float(grid[int(np.argmax(f))])
    V = w1 * cP + (1 - w1) * cT
    res = dict(w1=w1, w2=round(1 - w1, 2), f1_fused=float(max(f)), f1_pir=float(f1(cP > P.tau_hi, y)),
               f1_temp=float(f1(cT > P.tau_hi, y)), auc_pir=float(auc(cP, y)), auc_temp=float(auc(cT, y)),
               auc_fused=float(auc(V, y)), n_regions=int(n), prevalence=float(y.mean()))
    return res, dict(y=y, cP=cP, cT=cT, V=V)

def calibrate_ig(sm, pop, P, rng):
    """IG(a): empirical reduction in squared error of V vs. the noise-free class value, in the states where the action is used."""
    m, a0 = pop["motion"], ATT_SITE
    cP1 = S.pir(rng, sm, m, a0); keep = cP1 >= P.gate
    m, h = m[keep], pop["heat"][keep]; cP1 = cP1[keep]
    pt = np.where(m, sm.pir_fp + (sm.pir_tpr - sm.pir_fp) * a0, sm.pir_fp)
    eP = pt * sm.pir_trig_mu + (1 - pt) * sm.pir_quiet_mu
    eT = np.where(h, sm.temp_cold_mu + (sm.temp_hot_mu - sm.temp_cold_mu) * a0, sm.temp_cold_mu)
    Vs = P.w1 * eP + P.w2 * eT
    cT1 = S.temp(rng, sm, h, a0); mT = cT1.mean()
    cP2, cT2 = S.pir(rng, sm, m, a0), S.temp(rng, sm, h, a0)
    cP3, cT3 = S.pir(rng, sm, m, ATT_CLOSE), S.temp(rng, sm, h, ATT_CLOSE)
    w1, w2 = P.w1, P.w2; mse = lambda V, idx=slice(None): float(np.mean((V[idx] - Vs[idx]) ** 2))
    b = w1 * cP1 + w2 * mT
    ig = {"noT|temp": mse(b) - mse(w1 * cP1 + w2 * cT1),
          "noT|pir": 1e-6,   # not selectable: V is undefined without a temperature sample
          "noT|closer": mse(b) - mse(w1 * (cP1 + cP3) / 2 + w2 * cT3)}
    V1 = w1 * cP1 + w2 * cT1; band = (V1 >= P.tau_lo) & (V1 < P.tau_hi)
    ig.update({"T|temp": mse(V1, band) - mse(w1 * cP1 + w2 * (cT1 + cT2) / 2, band),
               "T|pir": mse(V1, band) - mse(w1 * (cP1 + cP2) / 2 + w2 * cT1, band),
               "T|closer": mse(V1, band) - mse(w1 * (cP1 + cP3) / 2 + w2 * (cT1 + cT3) / 2, band)})
    return {k: max(float(v), 1e-6) for k, v in ig.items()}

def fit_battery(sm, P, n_draw=50, n_ep=100, seed=1):
    """Random search on calibration episodes (E: success; A and C: detection at full battery)."""
    rng = np.random.default_rng(seed); best, bs = None, -1e9; hist = []
    for t in range(n_draw):
        Q = Params(**{**asdict(P)}); Q.ig = P.ig
        if t > 0:
            Q.alpha0 = rng.uniform(0.5, 1.5); Q.alpha_max = Q.alpha0 + rng.uniform(0.3, 2.0)
            Q.delta0 = rng.uniform(0.0, 0.05); Q.delta_max = Q.delta0 + rng.uniform(0.03, 0.4)
            Q.k1, Q.k2 = rng.uniform(5, 25, 2); Q.b_mid = rng.uniform(0.15, 0.6); Q.b_min = rng.uniform(0.04, 0.2)
        sE = np.mean([run_episode(Episode("E", e, 3, sm, CAL_SEED + 1), POLICIES["active"], Q)["success"] for e in range(n_ep)])
        dF = np.mean([run_episode(Episode(s, e, 3, sm, CAL_SEED + 1), POLICIES["active"], Q)["detection"]
                      for s in "AC" for e in range(n_ep // 2)])
        sc = 0.7 * sE + 0.3 * dF; hist.append(sc)
        if sc > bs: bs, best = sc, Q
    return best, dict(objective=float(bs), n_draw=n_draw, n_ep=n_ep)

def run_calibration(sm=None, quick=False):
    sm = sm or SensorModel(); P = Params(); rng = np.random.default_rng(CAL_SEED)
    pop = population(sm, 60 if quick else 300)
    wres, roc = fit_weights(sm, pop, P, rng); P.w1, P.w2 = wres["w1"], wres["w2"]
    P.ig = calibrate_ig(sm, pop, P, rng)
    P, bres = fit_battery(sm, P, 12 if quick else 50, 40 if quick else 100)
    return P, dict(weights=wres, battery_search=bres), roc
