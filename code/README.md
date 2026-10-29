# PeMS agentic world model (preliminary)

Two-class (human-driven / agentic) traffic world model on a real route pair in San Jose, calibrated from detector
speeds only, and a closed loop in which a traffic management agent coordinates ramp metering with agentic vehicles.

## Network
South San Jose to the US-101 / SR-85 interchange in Mountain View, northbound:
- Route A: SR-85 N (PEMS-BAY, 34 detectors, 30.5 km)
- Route B: SR-87 N then US-101 N (43 detectors, 28.7 km)
Data: PEMS-BAY 5-min speeds, Jan-Jun 2017 (`~/Desktop/CGT/CGT-DB/Traffic/PEMS-BAY`); from v4 also LargeST 5-min
mainline flows (PeMS, CC BY-NC 4.0, `~/Desktop/CGT/CGT-DB/Traffic/LargeST`).

## Layout
| file | what |
| --- | --- |
| `src/data.py` | PEMS-BAY loading, route geometry, observed route travel times |
| `src/ctm.py` | two-class cell transmission model (PyTorch, batched, differentiable) |
| `src/network.py` | builds the cell network from detector positions and lane counts |
| `src/calibrate.py` | gradient calibration to mean weekday speeds; 30% detectors hidden |
| `src/agentic.py` | vehicle agency levels L0-L3, the true world, ALINEA, CEM-MPC operator, daily runner |
| `src/run_experiments.py` | X1 operator ladder, X2 vehicle ladder, X3 agency x autonomy, paired by day |
| `src/fig*.py` | figures into `figs/` |
| `tests/test_ctm.py` | gate tests: CFL, conservation, free flow, spillback, finite gradients |
| `src/flows.py` | LargeST 5-min mainline flows for the route detectors (cached `route_flows_2017H1.pkl`) |
| `src/calibrate_flow.py` | v4 calibration: speeds + flows, per-ramp demand, measured US-101 boundary, queue penalty |
| `src/estimate.py` | moving-horizon state estimation + real-data forecasting vs persistence / historical (E2) |
| `src/operator_est.py`, `src/run_est.py` | detector-only operator in the closed loop (X5) |
| `src/metanet.py` | second-order METANET truth, same interface as `ctm.step` (`World(truth_model="metanet")`) |
| `tests/test_estimator.py` | gate test: one re-estimation cuts state error after a density jump |

## Run
```bash
cd ~/Desktop/Study2026/pems_agentic
.venv/bin/python -m pytest -q tests/test_ctm.py
.venv/bin/python src/fig1_real.py
.venv/bin/python src/calibrate.py 400 0.08 0.5 _v3 25 4500
.venv/bin/python src/fig2_calibration.py _v3
.venv/bin/python src/run_experiments.py results/wm_params_v3.pt
.venv/bin/python src/fig3_closed_loop.py
# v4 (2026-10-21)
.venv/bin/python src/flows.py                       # needs LargeST ca_his_raw_2017.h5 once; then cached
.venv/bin/python src/calibrate_flow.py 300 0.05 1.0 _v4 25 1.0
.venv/bin/python src/estimate.py results/wm_params_v4.pt _v4 no 80
.venv/bin/python src/run_experiments.py results/wm_params_v4.pt ctm _v4
.venv/bin/python src/run_est.py results/wm_params_v4.pt results/closed_loop_runs_v4.csv _v4 ctm
.venv/bin/python src/run_experiments.py results/wm_params_v4.pt metanet _v4_metanet
.venv/bin/python src/run_est.py results/wm_params_v4.pt results/closed_loop_runs_v4_metanet.csv _v4_metanet metanet
```

## Known simplifications (this version)
- Speeds only: demand and capacity are identified only up to the fundamental-diagram assumptions (2000 veh/h/lane).
- The true world shares the CTM structure; mismatch comes from capacity drop, incidents, demand noise, value-of-time
  misspecification and route-choice bias. A second-order METANET truth is the next step.
- The MPC operator observes the true current state; a state estimator is the next step.
- Agentic vehicles are a scenario parameter (30% share); they do not exist in the 2017 data.
