# Gazebo + ROS 2 replay (experimental, untested)

Status: **untested.** Written without access to ROS 2 or Gazebo. The Python files compile and the generated SDF world is well-formed XML, but nothing has been launched. Expect small fixes.

## Requirements
Ubuntu 24.04 (a Windows laptop can use WSL2 with a graphical window), ROS 2 Jazzy, Gazebo Harmonic, and the packages `ros-jazzy-ros-gz-sim` and `ros-jazzy-ros-gz-bridge`.

## Run
```bash
source /opt/ros/jazzy/setup.bash
cd <repository root>
pip install -r requirements.txt
python gazebo/sar_gazebo/sar_gazebo/generate_world.py --scenario D --episode 74 --policy active   # writes worlds/sar_arena.sdf and plan.json
cd gazebo && colcon build --packages-select sar_gazebo && source install/setup.bash
export PYTHONPATH=<repository root>:$PYTHONPATH
ros2 launch sar_gazebo sar.launch.py
```
A Gazebo window opens with the arena. The four-wheel-drive robot (four driven wheels through the DiffDrive plugin) drives to each region. Watch `ros2 topic echo /sar/events` and `/sar/pir`, `/sar/temp`, `/sar/gas`.

## What is and is not modeled
* Four driven wheels, wheel-ground physics, odometry-based position (so drift and slip appear in the motion).
* Virtual PIR, thermal and gas sensors with distance attenuation, a thermal field-of-view cone, and PIR valid only when stationary. They are not Gazebo sensor plugins.
* Decisions come from the Python simulation (trace replay). It is not closed-loop yet, and there are no ultrasonic or inertial sensor plugins.

## Next steps
Add ultrasonic-like ray sensors and an IMU, call the policy from the node at each decision point using the odometry pose, and replace the virtual sensors with recorded traces from the real HC-SR501, MLX90614 and MQ-2.
