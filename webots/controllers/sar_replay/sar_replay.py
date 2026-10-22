"""Webots controller: replays a planned search (plan.json) in 3-D with geometry-aware virtual PIR / thermal / gas sensors.
The DECISIONS come from the Python simulation; Webots supplies physics, motion and sensor geometry. This is not yet closed-loop.
UNTESTED: written without access to Webots. Expect to adjust device names and gains (see webots/README.md)."""
import json, math, os, sys
from controller import Robot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
from sarsim.config import SensorModel
from sarsim import sensors as S

robot = Robot(); dt = int(robot.getBasicTimeStep()); sm = SensorModel(); rng = np.random.default_rng(1)
plan = json.load(open(os.path.join(os.path.dirname(__file__), "plan.json"))); L = plan["regions"]
lm, rm = robot.getDevice("left wheel"), robot.getDevice("right wheel")
for m in (lm, rm): m.setPosition(float("inf")); m.setVelocity(0.0)
gps, imu = robot.getDevice("gps"), robot.getDevice("inertial unit"); gps.enable(dt); imu.enable(dt)
MAXV, THERMAL_FOV, SPEED_STILL = 5.0, math.radians(35), 0.02      # thermal cone half-angle; PIR only valid when stationary

def pose():
    x, y, _ = gps.getValues(); yaw = imu.getRollPitchYaw()[2]; return x, y, yaw
def drive_to(tx, ty):
    while robot.step(dt) != -1:
        x, y, yaw = pose(); dx, dy = tx - x, ty - y; d = math.hypot(dx, dy)
        if d < 0.25: lm.setVelocity(0); rm.setVelocity(0); return
        err = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw)); fwd = MAXV * min(1, d) * max(0, math.cos(err))
        lm.setVelocity(fwd - 2.5 * err); rm.setVelocity(fwd + 2.5 * err)
def settle(seconds):
    t0 = robot.getTime()
    while robot.step(dt) != -1 and robot.getTime() - t0 < seconds: pass
def virtual_read(i, kind):
    """Geometry-aware reading: attenuation from real distance, thermal narrow cone, PIR needs a stationary robot."""
    x, y, yaw = pose(); rx, ry = L["xy"][i]; d = math.hypot(rx - x, ry - y); a = float(S.att(sm, d))
    bearing = abs(math.atan2(math.sin(math.atan2(ry - y, rx - x) - yaw), math.cos(math.atan2(ry - y, rx - x) - yaw)))
    heat = L["victim"][i] or L["dheat"][i]; motion = L["victim"][i] or L["dmotion"][i]
    if kind == "PIR": return float(S.pir(rng, sm, motion, a))
    if kind == "TEMP": return float(S.temp(rng, sm, heat and bearing < THERMAL_FOV, a))   # outside the cone it sees only ambient
    return float(S.gas(rng, sm, L["hazard"][i], a))

for ev in plan["events"]:
    if ev["t"] == "move": drive_to(*ev["xy"])
    elif ev["t"] == "sense":
        settle(0.5); r = {k: virtual_read(ev["i"], k) for k in ("PIR", "TEMP", "GAS")}; print("sense", ev["i"], ev["kind"], {k: round(v, 2) for k, v in r.items()})
    elif ev["t"] in ("log", "reject"): print(ev["t"], ev["i"], "true victim" if ev["tp"] else "not a victim")
    elif ev["t"] == "return": drive_to(0, 0); print("returned" if ev["ok"] else "stranded")
print("replay finished; simulated outcome:", plan["result"])
