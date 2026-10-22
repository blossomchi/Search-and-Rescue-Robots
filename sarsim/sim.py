"""Policy runner implementing Algorithm 1 and the baselines."""
import numpy as np
from types import SimpleNamespace as NS
from .config import *
from .env import Episode, sigmoid

def alpha(P, B): return P.alpha0 + (P.alpha_max - P.alpha0) / (1 + np.exp(P.k1 * (B - P.b_mid)))
def delta(P, B): return P.delta0 + (P.delta_max - P.delta0) / (1 + np.exp(P.k2 * (B - P.b_mid)))

def run_episode(ep: Episode, cfg: PolicyCfg, P: Params, trace=None):
    n = N_REG
    S = NS(pos=np.zeros(2), batt=BATT_NOM * ep.sc["battery_frac"], visited=np.zeros(n, bool), logged=[],
           energy=0.0, dist=0.0, time=0.0, act=0, steps=0, expo=0, first_tp=None, depleted=False)

    def spend(e, t=0.0, acts=0):
        if e > S.batt: S.energy += S.batt; S.batt = 0.0; S.depleted = True; return False
        S.batt -= e; S.energy += e; S.time += t; S.act += acts; return True

    def T(ev, **kw):
        if trace is not None:
            trace.append(dict(t=ev, xy=[float(S.pos[0]), float(S.pos[1])], batt=float(S.batt), energy=float(S.energy), act=int(S.act), time=float(S.time), **kw))

    def verify_and_log(i, V_scr, cG_scr):
        if not spend(MOVE_COST * CLOSER_DIST + MOVE_OVERHEAD, T_MOVE_STEP + CLOSER_DIST): return
        cP, cT, cG = ep.read(i, ATT_CLOSE)
        if not spend(C_PIR + C_TEMP + C_GAS, T_PIR + T_TEMP + T_GAS, 3): return
        T('sense', i=int(i), kind='verify')
        ok = (V_scr + P.w1 * cP + P.w2 * cT) / 2 >= P.tau_hi
        if ok and cfg.hazard:
            gmean = cG if cG_scr is None else (cG + cG_scr) / 2
            if gmean >= 0.5 and not (cP >= 0.5 and cT >= 0.6): ok = False   # hazard cross-check
        T('log' if ok else 'reject', i=int(i), tp=bool(ep.victim[i]))
        if ok:
            S.logged.append(i)
            if ep.victim[i] and S.first_tp is None: S.first_tp = S.time

    def sense_static(i):
        if not spend(C_PIR + C_TEMP + C_GAS, T_PIR + T_TEMP + T_GAS, 3): return
        cP, cT, cG = ep.read(i, ATT_SITE)
        T('sense', i=int(i), kind='PIR+TEMP+GAS')
        V = P.w1 * cP + P.w2 * cT
        if V >= P.tau_hi: verify_and_log(i, V, cG)
        else: T('reject', i=int(i), tp=bool(ep.victim[i]))

    def sense_active(i, Hhat):
        if not spend(C_PIR, T_PIR, 1): return
        Ps = [ep.read_pir(i, ATT_SITE)]; Ts = []; cur = ATT_SITE; T('sense', i=int(i), kind='PIR')
        if Ps[0] < P.gate:
            if cfg.sensing != "active_mt":
                T('reject', i=int(i), tp=bool(ep.victim[i]), gate=True); return          # PIR gate: confident negative
            if not spend(C_TEMP, T_TEMP, 1): return
            Ts.append(ep.read_temp(i, cur))                # mandatory confirmatory temperature sample
        for _ in range(P.action_budget):
            if Ts:
                V = P.w1 * np.mean(Ps) + P.w2 * np.mean(Ts)
                if V >= P.tau_hi or V < P.tau_lo: break
            state = "T" if Ts else "noT"; best, bA = None, -1
            for a in ("temp", "pir", "closer"):
                if a == "closer" and cur >= ATT_CLOSE: continue
                if a == "pir" and state == "noT": continue   # PIR alone cannot separate a person from a distractor
                e = {"temp": C_TEMP, "pir": C_PIR, "closer": MOVE_COST * CLOSER_DIST + MOVE_OVERHEAD + C_TEMP}[a]
                A = P.ig[f"{state}|{a}"] * REL[a] / (e + EPS_COST + P.lam * (Hhat if a == "closer" else 0.0))
                if A > bA: best, bA = a, A
            if bA < P.a_min: break                          # abandon region
            if best == "temp":
                if not spend(C_TEMP, T_TEMP, 1): return
                Ts.append(ep.read_temp(i, cur)); T('sense', i=int(i), kind='TEMP')
            elif best == "pir":
                if not spend(C_PIR, T_PIR, 1): return
                Ps.append(ep.read_pir(i, cur)); T('sense', i=int(i), kind='PIR')
            else:
                if not spend(MOVE_COST * CLOSER_DIST + MOVE_OVERHEAD + C_TEMP, T_MOVE_STEP + CLOSER_DIST + T_TEMP, 2): return
                cur = ATT_CLOSE; Ps.append(ep.read_pir(i, cur)); Ts.append(ep.read_temp(i, cur)); T('sense', i=int(i), kind='CLOSER')
        if not Ts:
            T('reject', i=int(i), tp=bool(ep.victim[i])); return
        V = P.w1 * np.mean(Ps) + P.w2 * np.mean(Ts)
        if V >= P.tau_hi: verify_and_log(i, V, None)
        else: T('reject', i=int(i), tp=bool(ep.victim[i]))

    def travel_cost(d, i): return MOVE_COST * d * ep.leg_mult(i) + MOVE_OVERHEAD
    def ret_cost(pos): 
        r = np.linalg.norm(pos); return MOVE_COST * r + MOVE_OVERHEAD if r > 1e-6 else 0.0

    def pick(cand, rng):
        D = np.linalg.norm(ep.xy - S.pos, axis=1)
        if cfg.select == "random": return int(rng.choice(cand))
        if cfg.select == "frontier": return int(cand[np.argmin(D[cand])])
        V, H, R = ep.estimates(P)
        if cfg.select == "utility": return int(cand[np.argmax(((V + 0.25) / (D + 1))[cand])])
        B = S.batt / BATT_NOM
        a = alpha(P, B) if cfg.battery_w else P.alpha0
        d = delta(P, B) if cfg.battery_w else P.delta0
        be, ga = (P.beta, P.gamma) if cfg.hazard else (0.0, P.gamma)
        score = a * V - be * H - ga * R - d * D
        return int(cand[np.argmax(score[cand])])

    def pick_fmc(cand, rng, K=10):
        q = sigmoid(np.log(0.15 / 0.85) + ep.lr)
        best, bv = None, -1e9
        for j in cand:
            tot = 0.0
            for _ in range(K):
                truth = rng.random(n) < q; b = S.batt; pos = S.pos.copy()
                order = [j] + list(rng.permutation([c for c in cand if c != j])); found = 0.0
                for t in order:
                    c = travel_cost(np.linalg.norm(ep.xy[t] - pos), t) + C_TEMP + C_GAS
                    if b - c < ret_cost(ep.xy[t]) + P.b_min * BATT_NOM * 0.5: break
                    b -= c; pos = ep.xy[t]; found += 0.67 * truth[t]
                tot += found
            v = tot / K - 0.001 * np.linalg.norm(ep.xy[j] - S.pos)
            if v > bv: best, bv = int(j), v
        feasible = travel_cost(np.linalg.norm(ep.xy[best] - S.pos), best) + ret_cost(ep.xy[best]) + C_TEMP + C_GAS < S.batt
        return best if feasible else None

    rng = ep.rng
    ep.update_remote(S.pos, S.visited)
    for _ in range(80):
        if cfg.reserve and S.batt / BATT_NOM <= P.b_min: break
        cand = np.where(ep.discovered & ~S.visited)[0]
        if len(cand) == 0:
            if S.visited.all(): break
            th = rng.uniform(0, 2 * np.pi); tgt = S.pos + 3 * np.array([np.cos(th), np.sin(th)])
            if np.linalg.norm(tgt) > ARENA_R: tgt *= ARENA_R / np.linalg.norm(tgt)
            d = np.linalg.norm(tgt - S.pos)
            if not spend(travel_cost(d, None), d + T_MOVE_STEP): break
            S.dist += d; S.pos = tgt; T('move', i=-1); ep.update_remote(S.pos, S.visited); continue
        i = pick_fmc(cand, rng) if cfg.select == "fmc" else pick(cand, rng)
        if i is None: break
        d = float(np.linalg.norm(ep.xy[i] - S.pos))
        if not spend(travel_cost(d, i), d * ep.leg_mult(i) + T_MOVE_STEP): break
        S.dist += d; S.pos = ep.xy[i].copy(); S.visited[i] = True; S.steps += 1; T('move', i=int(i))
        S.expo += int(ep.hazard[i])
        if ep.sc["dynamic"] and S.steps == 4: ep.trigger_block(S.visited)
        Hh = float(np.clip(ep.sG[i] / max(ep.cnt[i], 1), 0, 1))
        ep.update_remote(S.pos, S.visited)
        if cfg.sensing == "static": sense_static(i)
        else: sense_active(i, Hh)
        if S.depleted: break
    returned = (not S.depleted) and S.batt >= ret_cost(S.pos)
    T('return', ok=bool(returned))
    L = list(dict.fromkeys(S.logged)); tp = sum(ep.victim[j] for j in L); fp = len(L) - tp
    return dict(n_victims=ep.n_victims, tp=int(tp), fp=int(fp), detection=tp / ep.n_victims,
                coverage=float(S.visited.mean()), distance=S.dist, energy=S.energy, activations=S.act,
                time=S.time, first_tp_time=(S.first_tp if S.first_tp is not None else np.nan),
                success=int(tp >= 1 and returned), stranded=int(not returned), hazard_exposure=S.expo)
