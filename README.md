# sar-sim: active, risk-aware multimodal search for low-cost SAR robots

Stochastic simulation accompanying the chapter *Active, Risk-Aware Multimodal Search for
Resource-Constrained Autonomous Search-and-Rescue Robots* (Ejaita, Oyem and Chiemeke, 2026).

**Scope.** This code tests the *decision logic* (Search Priority Score, energy-cost active sensing,
battery-aware weighting) under *assumed* sensor models. The sensor parameters in `sarsim/config.py`
are modelling assumptions, not hardware measurements. No claim about real HC-SR501 / MLX90614 / MQ-2
hardware follows from these results; hardware validation is pending.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests            # optional (pip install pytest)
python -m sarsim.run_experiments --episodes 300 --sens-episodes 60 --out results   # ~4 min on one CPU core
python -m sarsim.analyze --dir results                                              # tables + figures
```
Use `--quick --episodes 20 --sens-episodes 2` for a 15-second smoke test.

## Layout
| File | Purpose |
|---|---|
| `sarsim/config.py` | Sensor model, costs, scenarios A-F, policy definitions |
| `sarsim/sensors.py`, `sarsim/env.py` | Synthetic sensors; episode generator (layouts shared across policies) |
| `sarsim/sim.py` | Algorithm 1, the active-sensing rule A(a), all baselines |
| `sarsim/calibrate.py` | Calibration on **disjoint seeds**: fusion weights, IG(a) table, battery constants |
| `sarsim/run_experiments.py` | Calibration, main experiment, sensitivity study |
| `sarsim/analyze.py` | Bootstrap CIs, Mann-Whitney + Holm, Cliff's delta, Fisher tests, figures |
| `results/` | Outputs reported in the chapter (CSV, JSON, figures) |

## Visual replays
```bash
python -m sarsim.visualize --scenario D --episode 74 --left static --right active --out viz   # GIF + MP4
python -m sarsim.make_viewer viz/viewer.html                                                  # interactive HTML viewer
```
Episodes shown in `viz/` were chosen to illustrate differences between policies; they are not typical. Averages are in the chapter.
`webots/` (Webots) and `gazebo/` (Gazebo Harmonic with ROS 2) contain experimental, untested 3-D replays; see their READMEs.

## Reproducibility
* Evaluation seed base `20260913`; calibration seed base `777000` (disjoint). Environment layouts depend only on
  (scenario, episode) so all policies face identical layouts; sensor noise is seeded per (scenario, episode, policy).
* 300 episodes per policy per scenario (13 policies x 6 scenarios = 23,400 episodes); sensitivity study:
  27 sensor-parameter settings x 6 scenarios x 3 policies x 60 episodes.

## Policies
`random`, `frontier`, `static` (paper baseline: same score, no battery terms, no stop rule), `active` (proposed),
`active_mt` (adds a mandatory confirmatory temperature sample), ablations `active_nohaz`, `active_nobat`,
`active_noreserve`, and baselines with the same reserve stop rule: `frontier_R`, `utility_R`, `static_R`,
`static_RB`, `fmc_R` (Monte-Carlo belief planner, a lightweight POMDP-style reference; not a full MCTS).

## Metrics
Detection rate, false positives per episode, coverage (fraction of regions visited), distance, energy per episode,
energy per verified detection (ratio of sums over *all* episodes, no exclusion of zero-detection episodes),
sensor activations, hazard exposures, search time, mission success and stranding (defined for every scenario;
reported for E).

## Limitations
Simulation only; assumed sensor models; regions are abstract points (no occupancy-grid SLAM error, no PIR
self-motion, no field-of-view geometry); `fmc_R` is a simple planner, not a tuned POMDP solver.

## Citation / licence
MIT licence (see `LICENSE`). Cite the chapter above.
