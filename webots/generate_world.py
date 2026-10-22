"""Create a Webots world and a replay plan from one simulated episode, so the 3-D scene matches the Python simulation.
Usage (from the repository root): python webots/generate_world.py --scenario D --episode 74 --policy active
UNTESTED against a Webots installation: see webots/README.md."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sarsim.visualize import load_params, record

HDR = """#VRML_SIM R2023b utf8
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/adept/pioneer3/protos/Pioneer3dx.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
WorldInfo { basicTimeStep 16 }
Viewpoint { orientation -0.5 0.5 0.7 2.1 position 0 -22 22 }
TexturedBackground {}
TexturedBackgroundLight {}
RectangleArena { floorSize 26 26 }
"""
def solid(name, x, y, shape, color):
    return f'Solid {{ translation {x:.3f} {y:.3f} 0.25 name "{name}" children [ Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }} geometry {shape} }} ] }}\n'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", default="D"); ap.add_argument("--episode", type=int, default=74); ap.add_argument("--policy", default="active")
    a = ap.parse_args(); P = load_params(); lay, tr, res = record(a.scenario, a.episode, a.policy, P)
    w = HDR
    for i, (x, y) in enumerate(lay["xy"]):
        if lay["victim"][i]: w += solid(f"victim{i}", x, y, "Capsule { height 0.5 radius 0.25 }", "0.2 0.4 1")      # heated mannequin stand-in
        elif lay["hazard"][i]: w += solid(f"hazard{i}", x, y, "Cylinder { height 0.5 radius 0.35 }", "1 0.45 0")
        elif lay["dheat"][i] or lay["dmotion"][i]: w += solid(f"distractor{i}", x, y, "Box { size 0.5 0.5 0.5 }", "0.6 0.3 0.8")
        else: w += solid(f"region{i}", x, y, "Cylinder { height 0.05 radius 0.4 }", "0.7 0.7 0.7")
        if lay["obst"][i]: w += solid(f"obstacle{i}", x + 0.9, y, "Box { size 0.2 1.6 0.6 }", "0.4 0.3 0.2")
    w += 'Pioneer3dx { translation 0 0 0.1 name "sar_robot" controller "sar_replay" extensionSlot [ GPS {} InertialUnit {} ] }\n'
    root = os.path.dirname(__file__); name = f"sar_{a.scenario}{a.episode}_{a.policy}"
    open(f"{root}/worlds/{name}.wbt", "w").write(w)
    plan = dict(regions=lay, events=[e for e in tr if e["t"] in ("move", "sense", "log", "reject", "return")], result=res)
    json.dump(plan, open(f"{root}/controllers/sar_replay/plan.json", "w"))
    print("wrote", f"webots/worlds/{name}.wbt", "and controllers/sar_replay/plan.json")
if __name__ == "__main__": main()
