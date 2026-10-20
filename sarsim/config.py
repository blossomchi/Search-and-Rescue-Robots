"""All simulation parameters. Sensor numbers are MODELLING ASSUMPTIONS, not measurements."""
from dataclasses import dataclass, field

ARENA_R = 10.0        # regions lie within this distance of the base (units ~ metres)
N_REG = 12            # candidate regions per episode
DISC_R = 6.0          # remote-sensing (discovery) radius
BATT_NOM = 600.0      # nominal full-battery budget (relative energy units)
MOVE_COST = 3.0       # energy per unit distance
MOVE_OVERHEAD = 1.0   # fixed energy per move
C_PIR, C_TEMP, C_GAS = 0.0, 2.0, 6.0   # energy per on-site sample
T_PIR, T_TEMP, T_GAS = 0.1, 0.3, 1.5   # time per on-site sample (MQ-2 settles slowly)
T_MOVE_STEP = 0.5     # fixed time per move (time per unit distance = 1)
ATT_SITE = 0.90       # signal attenuation factor on arrival at a region
ATT_CLOSE = 1.00      # after the "move 0.5 closer" action
CLOSER_DIST = 0.5
EPS_COST = 0.5        # control overhead added to every action cost in A(a)
REL = {"temp": 1.0, "pir": 0.6, "closer": 0.8}   # Rel(a)
BASE_SEED = 20260913
CAL_SEED = 777000

@dataclass
class SensorModel:
    pir_tpr: float = 0.80; pir_fp: float = 0.15
    pir_trig_mu: float = 0.75; pir_trig_sd: float = 0.15
    pir_quiet_mu: float = 0.20; pir_quiet_sd: float = 0.12
    temp_hot_mu: float = 0.72; temp_hot_sd: float = 0.14
    temp_cold_mu: float = 0.20; temp_cold_sd: float = 0.13
    gas_haz_mu: float = 0.78; gas_haz_sd: float = 0.14
    gas_cold_mu: float = 0.15; gas_cold_sd: float = 0.10
    att_range: float = 6.0; att_floor: float = 0.15
    temp_sd_scale: float = 1.0   # sensitivity knob

@dataclass
class Params:
    w1: float = 0.10; w2: float = 0.90
    tau_lo: float = 0.50; tau_hi: float = 0.65; gate: float = 0.50
    beta: float = 0.6; gamma: float = 0.4; lam: float = 1.0
    alpha0: float = 1.0; alpha_max: float = 2.0
    delta0: float = 0.02; delta_max: float = 0.12
    k1: float = 10.0; k2: float = 10.0; b_mid: float = 0.4; b_min: float = 0.10
    action_budget: int = 4; a_min: float = 0.001
    ig: dict = field(default_factory=dict)

SCENARIOS = {
    "A": dict(n_victims=1, n_hazards=0, n_obstacles=0, distractors=False, battery_frac=1.0, dynamic=False),
    "B": dict(n_victims=1, n_hazards=2, n_obstacles=3, distractors=False, battery_frac=1.0, dynamic=False),
    "C": dict(n_victims=3, n_hazards=2, n_obstacles=0, distractors=False, battery_frac=1.0, dynamic=False),
    "D": dict(n_victims=1, n_hazards=2, n_obstacles=0, distractors=True,  battery_frac=1.0, dynamic=False),
    "E": dict(n_victims=2, n_hazards=2, n_obstacles=0, distractors=False, battery_frac=0.30, dynamic=False),
    "F": dict(n_victims=1, n_hazards=2, n_obstacles=2, distractors=False, battery_frac=1.0, dynamic=True),
}

@dataclass
class PolicyCfg:
    name: str
    select: str = "score"      # random | frontier | utility | score | fmc
    sensing: str = "static"    # static | active | active_mt
    hazard: bool = True        # beta term + gas cross-check
    battery_w: bool = False    # alpha(B), delta(B)
    reserve: bool = False      # stop and return when B_t <= b_min

POLICIES = {p.name: p for p in [
    PolicyCfg("random", select="random"),
    PolicyCfg("frontier", select="frontier"),
    PolicyCfg("static", select="score"),
    PolicyCfg("active", select="score", sensing="active", battery_w=True, reserve=True),
    PolicyCfg("active_mt", select="score", sensing="active_mt", battery_w=True, reserve=True),
    PolicyCfg("active_nohaz", select="score", sensing="active", hazard=False, battery_w=True, reserve=True),
    PolicyCfg("active_nobat", select="score", sensing="active", battery_w=False, reserve=True),
    PolicyCfg("active_noreserve", select="score", sensing="active", battery_w=True, reserve=False),
    PolicyCfg("frontier_R", select="frontier", reserve=True),
    PolicyCfg("utility_R", select="utility", reserve=True),
    PolicyCfg("static_R", select="score", reserve=True),
    PolicyCfg("static_RB", select="score", battery_w=True, reserve=True),
    PolicyCfg("fmc_R", select="fmc", reserve=True),
]}
