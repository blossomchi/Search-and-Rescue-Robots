"""ROS 2 node: replays plan.json (decisions from the Python simulation) through Gazebo physics and publishes geometry-aware virtual sensors.
Topics: /sar/pir, /sar/temp, /sar/gas (std_msgs/Float32), /sar/events (std_msgs/String). Needs PYTHONPATH to include the repository root.
UNTESTED: written without access to ROS 2 or Gazebo."""
import json, math, sys
import numpy as np, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from sarsim.config import SensorModel
from sarsim import sensors as S

class Replay(Node):
    def __init__(self):
        super().__init__("sar_replay"); self.declare_parameter("plan", "plan.json"); self.declare_parameter("fov_deg", 35.0)
        self.plan = json.load(open(self.get_parameter("plan").value)); self.L = self.plan["regions"]; self.ev = list(self.plan["events"]); self.k = 0
        self.fov = math.radians(self.get_parameter("fov_deg").value); self.sm = SensorModel(); self.rng = np.random.default_rng(1)
        self.x = self.y = self.yaw = 0.0; self.got_odom = False; self.wait_until = None; self.speed = 0.0
        self.cmd = self.create_publisher(Twist, "cmd_vel", 10); self.pub = {k: self.create_publisher(Float32, f"sar/{k}", 10) for k in ("pir", "temp", "gas")}
        self.log = self.create_publisher(String, "sar/events", 10); self.create_subscription(Odometry, "odom", self.on_odom, 10); self.create_timer(0.05, self.step)
    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation; self.x, self.y = p.x, p.y; self.got_odom = True
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)); self.speed = abs(m.twist.twist.linear.x)
    def read(self, i):
        rx, ry = self.L["xy"][i]; d = math.hypot(rx - self.x, ry - self.y); a = float(S.att(self.sm, d)); b = math.atan2(ry - self.y, rx - self.x) - self.yaw; b = abs(math.atan2(math.sin(b), math.cos(b)))
        heat = self.L["victim"][i] or self.L["dheat"][i]; motion = self.L["victim"][i] or self.L["dmotion"][i]
        pir = float(S.pir(self.rng, self.sm, motion and self.speed < 0.02, a))                 # PIR only valid when stationary
        return pir, float(S.temp(self.rng, self.sm, heat and b < self.fov, a)), float(S.gas(self.rng, self.sm, self.L["hazard"][i], a))
    def goto(self, tx, ty):
        dx, dy = tx - self.x, ty - self.y; d = math.hypot(dx, dy); t = Twist()
        if d < 0.25: self.cmd.publish(t); return True
        e = math.atan2(math.sin(math.atan2(dy, dx) - self.yaw), math.cos(math.atan2(dy, dx) - self.yaw)); t.linear.x = 0.6 * min(1.0, d) * max(0.0, math.cos(e)); t.angular.z = 1.8 * e; self.cmd.publish(t); return False
    def step(self):
        if not self.got_odom or self.k >= len(self.ev): return
        e = self.ev[self.k]; now = self.get_clock().now().nanoseconds * 1e-9
        if e["t"] == "move":
            if self.goto(*e["xy"]): self.k += 1
        elif e["t"] == "return":
            if self.goto(0.0, 0.0): self.get_logger().info("returned" if e["ok"] else "stranded"); self.k += 1
        elif e["t"] == "sense":
            if self.wait_until is None: self.wait_until = now + 0.8
            elif now >= self.wait_until:
                p, t, g = self.read(e["i"]); [self.pub[n].publish(Float32(data=v)) for n, v in zip(("pir", "temp", "gas"), (p, t, g))]
                self.log.publish(String(data=f"sense region {e['i']} ({e['kind']})")); self.wait_until = None; self.k += 1
        else:
            self.log.publish(String(data=f"{e['t']} region {e['i']}: {'true victim' if e['tp'] else 'not a victim'}")); self.k += 1

def main():
    rclpy.init(); n = Replay(); rclpy.spin(n); n.destroy_node(); rclpy.shutdown()
