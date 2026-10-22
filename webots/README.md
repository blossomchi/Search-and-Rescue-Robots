# Webots 3-D replay (experimental, untested)

Status: **untested.** These files were written without access to a Webots installation. Treat them as a starting point and expect to fix small issues.

## What it does
`generate_world.py` turns one simulated episode into a Webots world (arena, victim, hazard, distractor and obstacle objects, a Pioneer 3-DX robot with GPS and inertial unit) and a replay plan. `controllers/sar_replay/sar_replay.py` then drives the robot to each region and takes virtual PIR, thermal and gas readings that use real geometry in the 3-D scene.
Decisions still come from the Python simulation (trace replay), so this is a visual and geometric sanity check, not yet a closed-loop test.

## Run
1. Install Webots R2023b or later and the Python requirements of this repository.
2. `python webots/generate_world.py --scenario D --episode 74 --policy active`
3. Open `webots/worlds/sar_D74_active.wbt` in Webots and press Play.

## Known differences from the planned hardware
* Pioneer 3-DX is a two-wheel differential-drive robot, not the planned four-wheel-drive chassis. Replace it with a custom four-wheel robot for wheel-slip studies.
* PIR, MLX90614 and MQ-2 are virtual (sensor models from `sarsim/sensors.py` with a distance, a thermal field-of-view cone and a stationary-only PIR rule). They are not Webots devices.
* Positions use GPS, so dead-reckoning drift is not modeled.

## Next steps
1. Closed loop: call the policy logic of `sarsim/sim.py` from the controller at each decision point, using Webots poses and ultrasonic ranging instead of ground-truth positions.
2. Replace the virtual sensors with recorded traces from the real HC-SR501, MLX90614 and MQ-2 once the hardware is assembled.
3. Add PIR self-motion triggers and wheel slip, then re-run the comparison.
