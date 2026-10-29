# Coordinating Agentic Fleets with a Calibrated Physics–Learning Traffic World Model

> **Status (2026-10-30): work in progress.** All numbers are in `RESULTS.md`, which is the source of truth.
> Known issue: on three June test days (06-12, 06-19, 06-30) the US-101 boundary detector reports zero flow, so
> those replayed days are partly unrealistic; results are also reported without them, and a recalibration is pending.
> Reproduce with the project environment (`code/.venv`, torch 2.14); other library versions can change single-day
> planner outcomes. Raw data are not in the repository (see `data/README.md`).

A traffic management agent coordinates with a ride-hailing fleet on two parallel freeway routes in the South Bay
(SR-85 vs SR-87 + US-101, northbound), using raw Caltrans PeMS District 4 data for January–June 2026.

- **World model**: a differentiable six-class cell transmission model (human-driven, local, and fleet vehicles with
  and without passengers, plus a depot queue), calibrated by gradient descent to measured detector speeds, flows and
  ramp counts (`code/src/ctm.py`, `code/src/calibrate_daily_2026.py`).
- **Operator**: sees only noisy detectors, re-estimates the state with a moving horizon, and plans ramp meters and
  fleet offers with sampling-based MPC (`code/src/operator_est.py`, `code/src/fleet.py`, `code/src/fleet_est.py`).
- **Fleet agency levels**: self-interested, obedient, negotiating, and negotiating for empty vehicles only.
- **Evaluation**: each held-out day is replayed from its own calibration; chronological split (train Jan–Apr, tune
  May, test June once), pre-registered readings written into the scripts (`code/src/run_replay.py`).
- **Learned components**: STID forecasting benchmark (`code/src/stid_wm.py`) and DemandCast, which forecasts the
  physics model's demand inputs as of each decision and re-anchors the operator's world model
  (`code/src/demand_forecast.py`, `code/src/run_demandcast.py`).
- **New York**: download and aggregation of TLC high-volume FHV trips around the January 2025 congestion charge
  (`code/nyc/fetch_hvfhv.py`); analysis in progress.

## Layout
| Path | Contents |
| --- | --- |
| `code/src/` | data loaders, world model, calibration, operator and fleet agents, experiments |
| `code/tests/` | unit and smoke tests |
| `code/results/` | experiment outputs (csv/json/pt/npz) |
| `code/figs/` | figures |
| `code/docs/` | design notes |
| `data/README.md` | where the raw data come from |

## Run
```bash
cd code
.venv/bin/python -m pytest -q tests
PEMS_SPLIT=chrono .venv/bin/python src/run_replay.py test 6.0
```
