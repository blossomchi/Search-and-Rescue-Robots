"""Generate a Gazebo SDF world (and plan.json) from one simulated episode. Run from the repository root:
   python gazebo/sar_gazebo/sar_gazebo/generate_world.py --scenario D --episode 74 --policy active
UNTESTED against Gazebo: see gazebo/README.md."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); sys.path.insert(0, os.path.join(PKG, "..", ".."))
from sarsim.visualize import load_params, record

def model(name, x, y, geom, rgb, z=0.25, static=True):
    r, g, b = rgb
    return (f'<model name="{name}"><static>{str(static).lower()}</static><pose>{x:.3f} {y:.3f} {z} 0 0 0</pose><link name="l"><visual name="v"><geometry>{geom}</geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material></visual><collision name="c"><geometry>{geom}</geometry></collision></link></model>\n')
ROBOT = """<model name="sar_robot"><pose>0 0 0.12 0 0 0</pose>
 <link name="base_link"><inertial><mass>4</mass><inertia><ixx>0.05</ixx><iyy>0.08</iyy><izz>0.1</izz></inertia></inertial>
  <visual name="v"><geometry><box><size>0.5 0.3 0.1</size></box></geometry><material><ambient>0.9 0.3 0.7 1</ambient><diffuse>0.9 0.3 0.7 1</diffuse></material></visual>
  <collision name="c"><geometry><box><size>0.5 0.3 0.1</size></box></geometry></collision></link>
 WHEELS
 <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
  <left_joint>fl_j</left_joint><left_joint>rl_j</left_joint><right_joint>fr_j</right_joint><right_joint>rr_j</right_joint>
  <wheel_separation>0.36</wheel_separation><wheel_radius>0.08</wheel_radius><topic>cmd_vel</topic><odom_topic>odom</odom_topic>
  <frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id><odom_publisher_frequency>30</odom_publisher_frequency></plugin>
</model>"""
def wheel(n, x, y):
    return (f'<link name="{n}_w"><pose relative_to="base_link">{x} {y} 0 -1.5708 0 0</pose><inertial><mass>0.4</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.003</izz></inertia></inertial>'
            f'<visual name="v"><geometry><cylinder><radius>0.08</radius><length>0.05</length></cylinder></geometry></visual><collision name="c"><geometry><cylinder><radius>0.08</radius><length>0.05</length></cylinder></geometry></collision></link>'
            f'<joint name="{n}_j" type="revolute"><parent>base_link</parent><child>{n}_w</child><axis><xyz>0 0 1</xyz></axis></joint>')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", default="D"); ap.add_argument("--episode", type=int, default=74); ap.add_argument("--policy", default="active")
    a = ap.parse_args(); lay, tr, res = record(a.scenario, a.episode, a.policy, load_params())
    w = ('<?xml version="1.0"?><sdf version="1.9"><world name="sar_arena"><physics type="ode"><max_step_size>0.004</max_step_size></physics>'
         '<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/><plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>'
         '<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>'
         '<light type="directional" name="sun"><direction>-0.5 0.1 -0.9</direction><diffuse>1 1 1 1</diffuse></light>'
         '<model name="ground"><static>true</static><link name="l"><collision name="c"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry></collision>'
         '<visual name="v"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry><material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual></link></model>\n')
    for i, (x, y) in enumerate(lay["xy"]):
        if lay["victim"][i]: w += model(f"victim{i}", x, y, "<capsule><radius>0.25</radius><length>0.5</length></capsule>", (0.2, 0.4, 1))
        elif lay["hazard"][i]: w += model(f"hazard{i}", x, y, "<cylinder><radius>0.35</radius><length>0.5</length></cylinder>", (1, 0.45, 0))
        elif lay["dheat"][i] or lay["dmotion"][i]: w += model(f"distractor{i}", x, y, "<box><size>0.5 0.5 0.5</size></box>", (0.6, 0.3, 0.8))
        else: w += model(f"region{i}", x, y, "<cylinder><radius>0.4</radius><length>0.05</length></cylinder>", (0.6, 0.6, 0.6), z=0.03)
        if lay["obst"][i]: w += model(f"obstacle{i}", x + 0.9, y, "<box><size>0.2 1.6 0.6</size></box>", (0.4, 0.3, 0.2), z=0.3)
    w += ROBOT.replace("WHEELS", wheel("fl", 0.17, 0.2) + wheel("fr", 0.17, -0.2) + wheel("rl", -0.17, 0.2) + wheel("rr", -0.17, -0.2)) + "\n</world></sdf>"
    open(f"{PKG}/worlds/sar_arena.sdf", "w").write(w)
    json.dump(dict(regions=lay, events=[e for e in tr if e["t"] in ("move", "sense", "log", "reject", "return")], result=res), open(f"{PKG}/worlds/plan.json", "w"))
    print("wrote", f"{PKG}/worlds/sar_arena.sdf", "and plan.json")
if __name__ == "__main__": main()
