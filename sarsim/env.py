"""Episode generator: layout (shared across policies via layout seed) and noisy sensing."""
import numpy as np
from .config import *
from . import sensors as S

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

class Episode:
    def __init__(self, scen, ep_id, policy_id, sm=None, base_seed=BASE_SEED):
        self.sm = sm or SensorModel(); self.sc = SCENARIOS[scen]; sc = self.sc
        lay = np.random.default_rng([base_seed, ord(scen), ep_id])
        self.rng = np.random.default_rng([base_seed, ord(scen), ep_id, 1 + policy_id])
        n = N_REG
        ang = lay.uniform(0, 2 * np.pi, n); rad = ARENA_R * np.sqrt(lay.uniform(0, 1, n))
        self.xy = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
        perm = lay.permutation(n); k = 0
        z = lambda: np.zeros(n, bool)
        self.victim, self.hazard, self.dheat, self.dmotion, self.obst = z(), z(), z(), z(), z()
        self.victim[perm[k:k + sc["n_victims"]]] = True; k += sc["n_victims"]
        self.hazard[perm[k:k + sc["n_hazards"]]] = True; k += sc["n_hazards"]
        if sc["distractors"]:
            self.dheat[perm[k]] = True; self.dmotion[perm[k + 1]] = True; k += 2
        if sc["n_obstacles"]:
            self.obst[lay.choice(n, sc["n_obstacles"], replace=False)] = True
        self.hazhot = self.hazard & (lay.random(n) < 0.5)      # some hazards are also hot (fire)
        self.heat = self.victim | self.dheat | self.hazhot
        self.motion = self.victim | self.dmotion
        self.blocked = z(); self.block_order = lay.permutation(n)
        self.n_victims = int(self.victim.sum())
        self.discovered = z(); self.cnt = np.zeros(n)
        self.sP, self.sT, self.sG, self.sR = (np.zeros(n) for _ in range(4))
        self.lr = np.zeros(n)   # naive-Bayes log-LR (used only by the belief planner)

    # --- sensing -------------------------------------------------------
    def read(self, i, a):
        return (float(S.pir(self.rng, self.sm, self.motion[i], a)),
                float(S.temp(self.rng, self.sm, self.heat[i], a)),
                float(S.gas(self.rng, self.sm, self.hazard[i], a)))
    def read_pir(self, i, a): return float(S.pir(self.rng, self.sm, self.motion[i], a))
    def read_temp(self, i, a): return float(S.temp(self.rng, self.sm, self.heat[i], a))
    def read_gas(self, i, a): return float(S.gas(self.rng, self.sm, self.hazard[i], a))

    def update_remote(self, pos, visited):
        """Free background (baseline) readings of regions within DISC_R of the robot."""
        d = np.linalg.norm(self.xy - pos, axis=1); sm = self.sm
        for i in np.where((d <= DISC_R) & ~visited)[0]:
            a = float(S.att(sm, d[i])); self.discovered[i] = True
            cP, cT, cG = self.read(i, a)
            r = np.clip(0.1 + 0.6 * self.obst[i] + 0.3 * self.blocked[i] + self.rng.normal(0, 0.08), 0, 1)
            self.cnt[i] += 1; self.sP[i] += cP; self.sT[i] += cT; self.sG[i] += cG; self.sR[i] += r
            pv = sm.pir_fp + (sm.pir_tpr - sm.pir_fp) * a
            self.lr[i] += np.log(pv / sm.pir_fp) if cP >= 0.5 else np.log((1 - pv) / (1 - sm.pir_fp))
            mu_h = sm.temp_cold_mu + (sm.temp_hot_mu - sm.temp_cold_mu) * a
            sd_h = sm.temp_hot_sd * sm.temp_sd_scale * (2 - a); sd_c = sm.temp_cold_sd * sm.temp_sd_scale * (2 - a)
            self.lr[i] += (-0.5 * ((cT - mu_h) / sd_h) ** 2 - np.log(sd_h)) - (-0.5 * ((cT - sm.temp_cold_mu) / sd_c) ** 2 - np.log(sd_c))

    def estimates(self, P):
        c = np.maximum(self.cnt, 1)
        return (P.w1 * self.sP + P.w2 * self.sT) / c, np.clip(self.sG / c, 0, 1), self.sR / c

    def trigger_block(self, visited):
        for j in self.block_order:
            if not visited[j]:
                self.blocked[j] = True; return

    def leg_mult(self, i):
        if i is None: return 1.0
        return (1.4 if self.obst[i] else 1.0) * (2.5 if self.blocked[i] else 1.0)
