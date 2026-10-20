"""Synthetic sensor models (vectorised). Inputs are ground-truth flags and an attenuation factor a in (0,1]."""
import numpy as np

def att(sm, d):
    return np.clip(1.0 - np.asarray(d, float) / sm.att_range, sm.att_floor, 1.0)

def pir(rng, sm, motion, a):
    motion = np.asarray(motion); a = np.asarray(a, float)
    p = np.where(motion, sm.pir_fp + (sm.pir_tpr - sm.pir_fp) * a, sm.pir_fp)
    p, _ = np.broadcast_arrays(p, motion)
    trig = rng.random(p.shape) < p
    c = np.where(trig, rng.normal(sm.pir_trig_mu, sm.pir_trig_sd, p.shape),
                 rng.normal(sm.pir_quiet_mu, sm.pir_quiet_sd, p.shape))
    return np.clip(c, 0, 1)

def _gauss(rng, flag, a, hot_mu, hot_sd, cold_mu, cold_sd, scale):
    flag = np.asarray(flag); a = np.asarray(a, float)
    mu = np.where(flag, cold_mu + (hot_mu - cold_mu) * a, cold_mu)
    sd = np.where(flag, hot_sd, cold_sd) * scale * (1.0 + (1.0 - a))
    mu, sd = np.broadcast_arrays(mu, sd)
    return np.clip(rng.normal(mu, sd), 0, 1)

def temp(rng, sm, heat, a):
    return _gauss(rng, heat, a, sm.temp_hot_mu, sm.temp_hot_sd, sm.temp_cold_mu, sm.temp_cold_sd, sm.temp_sd_scale)

def gas(rng, sm, haz, a):
    return _gauss(rng, haz, a, sm.gas_haz_mu, sm.gas_haz_sd, sm.gas_cold_mu, sm.gas_cold_sd, 1.0)
