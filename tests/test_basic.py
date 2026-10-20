import numpy as np
from sarsim.config import *
from sarsim.env import Episode
from sarsim.sim import run_episode, alpha, delta

def test_layout_shared_across_policies():
    a, b = Episode("C", 5, 0), Episode("C", 5, 3)
    assert np.allclose(a.xy, b.xy) and (a.victim == b.victim).all()

def test_battery_functions_monotone():
    P = Params()
    assert alpha(P, 1.0) < alpha(P, 0.1) and delta(P, 1.0) < delta(P, 0.1)

def test_reproducible():
    P = Params(); P.ig = {f"{s}|{a}": 0.01 for s in ("T", "noT") for a in ("temp", "pir", "closer")}
    r1 = run_episode(Episode("B", 1, 3), POLICIES["active"], P)
    r2 = run_episode(Episode("B", 1, 3), POLICIES["active"], P)
    assert r1 == r2 or all((r1[k] == r2[k]) or (r1[k] != r1[k]) for k in r1)
