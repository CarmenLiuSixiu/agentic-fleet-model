# Preliminary results (2026-10-20)

## Figure 1: observed routes (PEMS-BAY, 126 weekdays, Jan-Jun 2017) — `figs/fig1_real_routes.png`
| | Route A: SR-85 N | Route B: SR-87 N + US-101 N |
| --- | --- | --- |
| free-flow travel time | 16.8 min | 15.2 min |
| peak median travel time | 36.2 min at 08:00 | 32.5 min at 08:35 |
- Route B faster in 89% of 5-min slots; the faster route changes at least once between 06:00 and 10:00 on 64% of days.

## Figure 2: detector-only calibration — `figs/fig2_calibration.png` (mean profile), `figs/fig2_daily.png` (day-to-day)
Speed MAE, km/h, 30% detectors hidden:
| calibration | visible, train | visible, test | hidden, test | interpolation, hidden test | own history, hidden test |
| --- | --- | --- | --- | --- | --- |
| mean weekday (v1, used downstream) | 11.2 | 13.3 | 17.5 | 7.1 | — |
| day-to-day (30 train / 20 test days) | 13.2 | 13.5 | 17.4 | 8.8 | 10.4 |
- The model reproduces the main recurrent bottlenecks but does not beat interpolation at hidden detectors (0/20 test days).
- Diagnosis: free-flow cells (47% of observations) are 10 km/h too slow because one free-flow speed per route had to absorb
  partially congested readings; congested cells are not congested deeply enough (model 64 vs observed 42 km/h).
  Per-cell free-flow speeds were added afterwards; a data-driven bottleneck initialisation made the fit worse (train MAE 15.3).
- PEMS-BAY speeds are partly imputed by PeMS from neighbouring detectors, which favours interpolation at "hidden" detectors.

## Figure 3: closed loop — `figs/fig3_closed_loop.png`, `results/closed_loop_runs.csv`
30% agentic vehicles, 10 simulated days (5 with incidents), paired by day; change in total time spent vs no control, mean [90% bootstrap CI]:
| operator / vehicles | TTS change | worst controlled ramp mean wait | incentives per day |
| --- | --- | --- | --- |
| A1 ALINEA | +0.56% [+0.47, +0.64] | 2.5 min | — |
| A2 world-model MPC, L0 | −0.17% [−0.23, −0.12] | 3.2 min | — |
| A2, L1 self-interested | −0.47% [−0.72, −0.26] | 3.3 min | — |
| A2, L2 obedient | −3.69% [−4.61, −2.77] | 2.9 min | — |
| A2, L3 negotiating | −3.15% [−3.88, −2.38] | 1.3 min | $2,728 |
| A2, L4 negotiating + Caltrain park-and-ride | −3.75% [−4.52, −2.94] | 1.4 min | $3,897 (96 riders) |
| A2, L0, autonomy (kappa 0.3) | −12.80% [−13.35, −12.26] | 3.1 min | — |
| A2, L3 + autonomy | −16.40% [−17.26, −15.54] | 0.6 min | $2,784 |
- L2, L3, L4 each beat A2-L0 on 10/10 days; L3 keeps 85% of the L2 gain; L3 is worse than L2 on 9/10 days by 0.6% on average.
- Agency x autonomy interaction: −0.64% [−1.21, −0.02], i.e. nearly additive with slight synergy.
- The autonomy effect size is set by the assumed capacity gain and is not a finding.

## Caveats
Operator observes the true state; truth shares the CTM structure (perturbed by demand noise, capacity drop, incidents,
value-of-time misspecification, route-choice bias); agentic share is a scenario parameter; speeds only; 10 days.

---
# Update 2026-10-21

## X5: detector-only operator (closes caveat "operator observes the true state") — `src/operator_est.py`, `src/run_est.py`
The operator now sees only noisy 5-min detector speeds (all route detectors, noise sd 3 km/h). It keeps an internal
world-model copy advanced with its own applied controls, and every 15 min re-estimates the state from the last 30 min
of detectors by gradient descent through the CTM (per-cell density multipliers, queue scale, 3 demand scales; 30 Adam
steps, ~3 s). Same v1 params, truth and 10 days as Figure 3; paired with the oracle-state runs.
| vehicles | TTS change, detector-only [90% CI] | oracle state | detector-only minus oracle [90% CI] | share of oracle gain kept | worst ramp wait (det / oracle) |
| --- | --- | --- | --- | --- | --- |
| L0 | −0.09% [−0.14, −0.05] | −0.17% | +0.08 [+0.05, +0.11] | — | 2.6 / 3.1 min |
| L1 | −0.39% [−0.63, −0.18] | −0.47% | +0.08 [−0.00, +0.17] | — | 4.0 / 3.3 min |
| L2 | −3.58% [−4.48, −2.69] | −3.69% | +0.11 [−0.00, +0.25] | 97% | 3.8 / 2.9 min |
| L3 | −3.12% [−3.82, −2.40] | −3.15% | +0.02 [−0.05, +0.10] | 99% | 1.4 / 1.3 min |
| L4 | −3.84% [−4.69, −2.98] | −3.75% | −0.09 [−0.27, +0.11] | 102% | 0.9 / 1.4 min |
- Mean state error after estimation ≈ 5.5 veh/km per cell. L2/L3/L4 still beat metering-only on 10/10 days.
- Reading: within a world that shares the model's structure, state knowledge is not what limits the gain. This does
  not test structural mismatch (next: METANET truth).
- Gate test `tests/test_estimator.py`: after a 40% density jump, one re-estimation cuts state error by >30%.

## E2: real-data short-term forecasting from detectors — `src/estimate.py`, `results/nowcast_*.json`
Test weekdays May–Jun 2017 (44 days), decisions 06:30–09:30 every 30 min, MHE on the last 30 min of all detectors,
then roll forward. "anomaly" = historical mean + (model day − model mean day) at each detector.
| params | horizon | speed MAE wm / persistence / historical (km/h) | route TT MAE wm / pers / hist (min) | faster route correct wm / pers / hist |
| --- | --- | --- | --- | --- |
| v1 raw | 15 | 19.6 / 6.8 / 12.7 | 7.6 / 2.1 / 5.6 | 31% / 93% / 91% |
| v1 raw | 60 | 21.7 / 17.4 / 13.8 | 8.0 / 6.4 / 6.1 | 28% / 85% / 91% |
| v1 anomaly | 60 | 14.4 / 17.4 / 13.8 | 5.2 / 6.4 / 6.1 | 65% / 85% / 91% |
| v3 raw | 15 | 18.1 / 6.8 / 12.7 | 4.0 / 2.1 / 5.6 | 91% / 93% / 91% |
| v3 raw | 60 | 20.5 / 17.4 / 13.8 | 5.1 / 6.4 / 6.1 | 91% / 85% / 91% |
| v3 anomaly | 60 | 14.0 / 17.4 / 13.8 | 5.7 / 6.4 / 6.1 | 85% / 85% / 91% |
- Persistence wins at 15 min, as expected. At 60 min the world model has the lowest route travel-time error
  (v3 raw 5.1, v1 anomaly 5.2 vs 6.1–6.4), but no version beats the historical profile at detector level.
- **v1 (the version used in the closed loop) ranks the routes wrongly on real data 69–72% of the time**: it is
  self-consistent in simulation but biased towards route A on the real corridor. "Always B" scores 89%.
- **v3 matches the real route ranking (91%) but its simulated world is implausible**: with no control a regular
  on-ramp queues ~1,200 vehicles, because calibration pushed a ramp demand above what can merge. v3 is therefore
  not used for closed-loop runs. The flow-calibrated v4 adds a ramp-storage penalty and must pass both checks.

---
# v4: flow-calibrated world model (2026-10-21) — supersedes the v1 closed-loop numbers above

## Why v1 numbers are superseded
With no control, the v1 world builds point queues at the two freeway-to-freeway connectors that never clear
(US-101 N into the SR-87/US-101 junction 8,900 veh and I-280 into SR-85 4,700 veh at 09:30, still growing at 11:00):
speed-only calibration used connector demand as the knob to create downstream congestion. Most of the v1 baseline TTS
(~76k veh-h) is these queues, so the v1 percentages in Figure 3 rest on an implausible world.
v1 also ranks the routes wrongly on real data (E2: 28–31% correct).

## Calibration v4 — `src/flows.py`, `src/calibrate_flow.py`, `results/calibration_v4.json`
Data added: LargeST (Liu et al., NeurIPS 2023 D&B; CC BY-NC 4.0) 5-min mainline flows from PeMS for 76 of the 77 route
detectors plus US-101 N upstream of the SR-87 junction, Jan–Jun 2017, 100% timestamp-aligned with PEMS-BAY.
Changes: speed + flow loss; per-ramp 15-min demand; measured US-101 inflow as boundary; per-cell capacity and free-flow
speed initialised from each detector's empirical fundamental diagram; penalty on ramp queues beyond storage
(60 veh regular, 600 veh connector). Same train/test split and hidden detectors as v1/v3.
| | v1 | v3 | v4 | reference |
| --- | --- | --- | --- | --- |
| hidden-detector speed MAE, test (km/h) | 17.5 | 18.6 | 17.9 | interpolation 7.1 |
| hidden-detector flow MAE, test (veh/h) | — | — | 924 | interpolation 851 |
| route TT MAE on test mean day, A / B (min) | 5.2 / 2.3 | 3.0 / 2.5 | 1.4 / 2.7 | train-period mean day 1.5 / 1.2 |
| calibrated free-flow speed (km/h) | — | 90 / 91 | 102 / 100 | |
| worst ramp queue / storage, no control | ~15 (connector) | ~20 (regular) | 2.1 | |
- Flows do not improve detector-level speed fit. They fix the demand side: no runaway queues, realistic free-flow speed,
  route A travel time as good as the historical profile. Route B is still under-congested (peak 26 vs 30 min).
- Known remaining issue: two regular on-ramps still queue 300–570 veh with no control (one on SR-87 N with 1,426 veh/h
  at 08:00 is probably the I-280/SR-87 connector and should be marked as a connector).

## E2 with v4 (real data, 44 test days) — `results/nowcast_v4*.json`
Route travel-time MAE (min) at +15 / +30 / +60 min:
| method | +15 | +30 | +60 | faster route correct (+60) |
| --- | --- | --- | --- | --- |
| world model v4, MHE on speeds | 3.1 | 3.8 | 5.0 | 85% |
| world model v4, MHE on speeds + flows | 3.7 | 4.5 | 6.1 | 85% |
| world model v4, anomaly mode | 3.8 | 4.1 | 5.1 | 74% |
| persistence | 2.1 | 3.6 | 6.4 | 85% |
| historical profile | 5.6 | 5.9 | 6.1 | 91% |
- v4 beats both baselines at +60 min and ties persistence at +30. Adding flows to the estimator makes it worse: the model's
  flows are still biased (~600 veh/h MAE), so fitting them pulls the state off. Detector-level speed still loses to
  the historical profile (16–21 vs 13–14 km/h).

## Closed loop with v4 — `results/closed_loop_*_v4*.csv/json`
TTS change vs no control, mean over 10 paired days [90% CI]. "oracle" = operator sees the true state; "detector" =
MHE from noisy detectors (`operator_est.py`). Truth "CTM" = calibrated CTM + capacity drop, incidents, demand noise;
truth "METANET" = second-order model (`src/metanet.py`) with the same parameters, not calibrated to data
(its peak route times are 29/24 min vs 37/33 observed), so it is a stress test of model error, not a more realistic world.
| operator / vehicles | CTM, oracle | CTM, detector | METANET, oracle | METANET, detector |
| --- | --- | --- | --- | --- |
| A1 ALINEA | +0.31 [+0.05, +0.60] | — | +1.50 [+1.34, +1.66] | — |
| A2 MPC, L0 | +0.07 [−0.15, +0.32] | +0.44 [+0.13, +0.75] | −0.06 [−0.17, +0.03] | −0.11 [−0.26, +0.01] |
| A2, L1 self-interested | **+5.17** [+3.63, +6.83] | +6.16 [+4.27, +8.14] | **−2.00** [−2.98, −1.09] | −2.10 [−3.10, −1.18] |
| A2, L2 obedient | **−6.70** [−10.04, −3.54] | −5.55 [−9.22, −2.05] | **+4.53** [+3.23, +5.65] | +4.73 [+3.47, +5.85] |
| A2, L3 negotiating | −4.15 [−7.10, −1.41] | −3.01 [−5.88, −0.30] | +1.15 [+0.64, +1.72] | +1.67 [+0.91, +2.37] |
| A2, L4 + Caltrain | −4.07 [−7.34, −1.08] | −4.19 [−7.05, −1.53] | +1.23 [+0.70, +1.80] | +1.63 [+0.83, +2.40] |
| A2, L0 + autonomy κ0.3 | −25.1 | | −9.3 | |
| A2, L3 + autonomy | −25.7 | | −8.7 | |
| agency × autonomy interaction | +3.62 [+1.57, +5.84] | | −0.58 [−0.92, −0.27] | |
- Baseline TTS: CTM truth 27.9k veh-h/day, METANET truth 20.2k (v1 world: ~76k).
- CTM truth: L2 beats metering-only on 9/10 days, L3 6/10, L4 7/10. L3 keeps 62% of the L2 gain and is worse than L2
  on 9/10 days. Incentives: L3 $21.8k/day = $14.9 per veh-h saved (v1: $1.2), below the mean value of time ($25/h).
  Worst controlled ramp wait rises with coordination (L3 7.2 min vs 5.8 with no control): the v1 claim "negotiation
  halves the worst ramp wait" does not hold in v4. Self-interested rerouting on live times makes things worse (+5.2%).
- Detector-only operator keeps 83% (L2), 72% (L3), 103% (L4) of the oracle gain under CTM truth; under METANET truth
  the detector-only results are within ~0.5 points of oracle. State error ~9 veh/km (METANET), larger under v1.
- **Under METANET truth the sign flips**: coordinating with obedient vehicles costs +4.5% (0/10 days better than
  metering-only), negotiation limits the damage to +1.2% because vehicles can refuse offers that do not match the
  travel times they actually see, and self-interested vehicles help (−2.0%). With the v1 model the same truth gave
  gains (L2 −2.5%, L3 −2.6%), so the value of vehicle agency depends on how well the operator's model matches the world.
  This is the motivation for A3 (learn how vehicles respond) and A4 (fall back when the model's own detector
  residuals say it is wrong).
- The autonomy effect size is set by the assumed capacity gain and is not a finding.

---
# April 2026, raw PeMS (2026-10-25) — `src/pems_raw.py`, `src/compare_2026.py`, `calibrate_flow.py ... 2026`
Data: PeMS Clearinghouse Station 5-Minute, District 4, all 22 weekdays of April 2026 (+ 4 May days not used yet),
`~/Desktop/CGT/CGT-DB/Traffic/PeMS_raw/d04/station_5min/` (29 MB/day). Each row carries "% Observed"; rows with
< 50% observed are PeMS imputations and are dropped everywhere below. No 2026 station metadata yet, so stations are
matched by their 2017 IDs and on/off-ramp stations cannot be placed yet.

## Data quality on the corridor (2017 station IDs)
| route | stations (2017) | present 2026 | measured in ≥ 50% of AM rows | never measured |
| --- | --- | --- | --- | --- |
| A: SR-85 N | 34 | 31 | 20 | 9 |
| B: SR-87 N + US-101 N | 43 | 39 | 22 | 17 |
- In 17% of 5-min slots a route has fewer than two measured stations; route times then fall back to published values.
- On/off-ramp stations with measured counts exist (e.g. SR-85 N 17 of 28 on-ramps, US-101 N 90 of 131 in D4) but
  need the station metadata file to be located.

## 2017 vs April 2026 (figs/fig1b_2017_vs_2026.png, results/compare_2026.json)
| | 2017 Jan–Jun (126 d) | Apr 2026, as published (22 d) | Apr 2026, measured only |
| --- | --- | --- | --- |
| peak median TT, A | 36.2 min at 08:00 | 28.0 at 08:45 | 28.6 at 07:50 |
| peak median TT, B | 32.5 min at 08:35 | 25.8 at 07:55 | 26.4 at 08:50 |
| free-flow TT A / B | 16.8 / 15.2 | 16.7 / 15.2 | 16.6 / 14.9 |
| B faster (share of 5-min slots) | 89% | 79% | 76% |
| days on which the faster route flips 06:00–10:00 | 64% | 86% | 91% |
- Peak congestion is ~8 min lower on A and ~6 min lower on B than in 2017; free-flow times are unchanged.
- The two routes are much closer in 2026: which route is faster changes on almost every morning, so routing advice
  matters more, not less.

## Calibration on April 2026 (train Apr 1–21, 15 weekdays; test Apr 22–30, 7 weekdays) — `results/calibration_2026apr.json`
Same model and settings as v4; loss and evaluation use measured rows of valid detectors only (42 of 77; 11 of them
hidden). US-101 boundary from station 402364 (95% measured).
| | 2017 v4 | Apr 2026 |
| --- | --- | --- |
| speed MAE visible, train / test (km/h) | 11.7 / 13.3 | 10.9 / 18.4 |
| speed MAE hidden, test: model / interpolation / train-period profile | 17.9 / 7.1 / — | 25.5 / 9.1 / 8.8 |
| flow MAE visible train / hidden test (veh/h) | 597 / 924 | 371 / 810 (interpolation 773) |
| route TT MAE on test mean day, A / B (min): model / train-period profile | 1.4, 2.7 / 1.5, 1.2 | 3.6, 3.5 / 3.8, 2.7 |
| worst ramp queue / storage, no control | 2.1 | 1.3 |
- Flows fit better than in 2017 and the world stays plausible (no runaway queues).
- Speed generalisation to the test week is poor (visible 10.9 → 18.4 km/h): one mean profile from 15 days does not
  carry over to the next 7 days; the train-period profile itself is a strong reference (8.8 at hidden detectors).
- On route travel time the model beats the train-period profile on A (3.6 vs 3.8) and loses on B (3.5 vs 2.7).
- Refactoring check: rerunning the 2017 v4 calibration through the new code reproduces every v4 number exactly.

## April 2026 with ramps and a fair split (2026-10-25) — `results/calibration_2026i*.json`
On/off-ramp stations placed from the 2026-04-08 metadata (`results/ramps_2026_meta.csv`; `build_default_net(ramps="2026")`:
31 model on-ramps, 33 exit cells). Measured on-ramp inflow (21 model ramps, 6 hidden) and off-ramp flow (6 exit cells,
2 hidden) added to the loss. Split: days with < 30% measured mainline rows dropped (Apr 21–24, an outage); of the 18
remaining weekdays every third is a test day (Apr 3, 8, 13, 16, 27, 30). (The earlier date split put Apr 22–30 in test:
3 outage days plus Apr 28–30, the three worst mornings of the month, peak A 39–48 min.)
| | no ramps | ramps, w_q 1 | ramps, w_q 0.3 | train-period mean day | interpolation |
| --- | --- | --- | --- | --- | --- |
| speed MAE hidden detectors, test (km/h) | 17.0 | 15.6 | 15.8 | 3.8 | 6.6 |
| flow MAE hidden detectors, test (veh/h) | 772 | 849 | 787 | — | 614 |
| route TT MAE on test mean day A / B (min) | 2.0 / 2.4 | 2.9 / 2.8 | 2.9 / 2.4 | 0.4 / 0.7 | — |
| modelled peak route TT A / B (obs 27 / 26) | 24.8 / 21.7 | 19.1 / 18.8 | 24.0 / 21.2 | | |
| modelled free-flow route TT A / B (obs 16.8 / 15.1) | 19.1 / 17.7 | 19.1 / 17.4 | 18.9 / 17.3 | | |
| on-ramp MAE visible / hidden (veh/h) | — | 123 / 183 | 192 / 170 | hidden 31 | |
- Ramp counts improve hidden-detector speeds a little but flatten the peak (A 19 vs 27 min): matching measured ramp
  inflows leaves the model without the demand excess it used to create congestion. Down-weighting flows restores most
  of the peak. Hidden ramps cannot be predicted from the others (183 vs 31 for their own train-period profile).
- With an interleaved split the train and test mean days are almost identical, so the mean-day profile is a very strong
  baseline (route TT error under 1 min) and every model version is far behind it. All versions are also ~13% too slow
  in free flow. Reading: the limit is calibrating one CTM day to an average of days (averaging stop-and-go across days
  gives mild slowdowns everywhere, which no single CTM day reproduces). Next: day-specific calibration with shared
  parameters (calibrate_daily.py extended to flows + ramps) on April–June 2026.

---
# Fleet agent (2026-10-25) — design `docs/fleet_agent_design.md`, code `src/fleet.py`, `src/run_fleet.py`
CTM now has 6 classes (adds fleet occupied / empty through vehicles) and a depot queue for held empties.
Checks: with fleet share 0 the closed loop reproduces the v4 CTM runs bit for bit (A0, A2-L3, A2-L4, detector-only L3);
METANET differs by 4e-6 relative because the downstream density at the diverge is now composition-weighted
(was a plain average of two route probabilities). `tests/test_fleet.py`: conservation with fleet + hold (CTM and
METANET), held empties released, fleet route probabilities move only fleet classes. 11/11 tests pass.
Setting: v4 world model, private agentic vehicles off, empty share 41% (Waymo 2026Q2 statewide), c_E $20/h,
mean hold 10 min, 10 paired days (5 with incidents). Outcome = all vehicle-hours incl. held empties, change vs metering
only (A2-F0), mean [90% CI], days better than metering-only.

## CTM truth — `results/fleet_summary_v4_ctm.json`
| fleet share | F1 obedient | F2 negotiating | F3 empties only | F3 + hold | F2 $/day ($ per veh-h saved) |
| --- | --- | --- | --- | --- | --- |
| 1% | −0.62 [−1.09, −0.20] 7/10 | −0.60 [−0.94, −0.27] 8/10 | −0.16 [−0.40, +0.07] | −0.23 [−0.43, −0.04] | 626 (3.5) |
| 5% | −3.59 [−5.14, −2.20] 10/10 | −3.11 [−4.57, −1.77] 9/10 | −1.22 [−1.66, −0.81] | −1.54 [−1.97, −1.14] | 3,464 (3.2) |
| 10% | −5.78 [−8.34, −3.44] 9/10 | −5.33 [−7.78, −3.08] 9/10 | −1.90 [−2.57, −1.27] | −2.14 [−2.78, −1.49] | 7,192 (3.6) |
| 30% | −11.00 [−14.89, −7.41] 9/10 | −8.76 [−12.33, −5.41] 10/10 | −2.13 [−2.63, −1.65] | −2.96 [−3.60, −2.34] | 24,050 (6.9) |
- Pre-registered reading F1 (threshold): F2 is already significant at 1%; no threshold above 1% in this world. Gains grow
  roughly linearly to 10% and flatten at 30%.
- F2 (empties as main lever): F3 gets 36–40% of the F2 gain at 5–10% share and 24% at 30%, below the pre-registered 70%
  → not supported: empties alone are not the main lever in the CTM world.
- F3 (hold): adds 0.1–0.8 points; at 30% it costs $11.4 per veh-h saved (F3 without hold $5.8).
- Who pays: with F1/F2 the fleet's riders take ~4–10 min longer (occupied 26 → 30–37 min) and human through drivers
  1–4 min longer; the total still falls, so the gains go to on-ramp (local) traffic. F3 leaves riders unchanged
  (26.2 vs 26.4 min). F2 riders are compensated by construction (accepted only when paid ≥ value of time).
- Payment efficiency $3–7 per veh-h saved, well below value of time ($25/h); private-vehicle negotiation (L3, v4) was $14.9.
- Metering alone with a self-interested fleet: −0.0 to −1.2% vs no control.

## METANET truth (model-mismatch stress test) — `results/fleet_summary_v4_metanet.json`
| fleet share | F1 obedient | F2 negotiating | F3 empties only | F3 + hold |
| --- | --- | --- | --- | --- |
| 1% | +0.22 [+0.09, +0.34] | +0.04 [−0.07, +0.14] | +0.02 [−0.04, +0.07] | −0.00 |
| 5% | +1.05 [+0.34, +1.70] | +0.36 [−0.16, +0.83] | +0.14 [−0.09, +0.36] | +0.06 |
| 10% | +2.16 [+0.90, +3.38] | +1.08 [+0.24, +1.87] | +0.20 [−0.18, +0.56] | +0.09 |
| 30% | +6.69 [+4.50, +8.69] 0/10 | +3.66 [+2.75, +4.59] 0/10 | +0.69 [+0.45, +0.93] | +0.31 [−0.18, +0.75] |
- Pre-registered reading F4: F2 harm < F1 harm at every share (about half), replicating the L3 mechanism for a fleet.
  Empties-only negotiation is nearly harmless (≤ +0.7%) because occupied vehicles keep routing on live travel times.
- Together: F2 gives the largest gain when the operator's model is right and half the obedient fleet's damage when it is
  wrong; F3 gives a third of the gain but almost no damage. Choosing between them is a robustness–performance trade-off
  that depends on how far the operator trusts its own model (motivation for the A4 fallback).

## Day-by-day calibration, April–June 2026 — `src/calibrate_daily_2026.py`, `results/calibration_daily2026.json`
53 usable weekdays (outage days dropped; 5/12, 6/1, 6/11 not downloaded); every third day is a test day (17), 36 train.
Per day: 15-min multipliers on origin and route-A / route-B ramp demand, route bias, measured US-101 inflow; road
parameters shared (warm start from the April mean-day fit). Test days: road frozen, day parameters fitted on visible
detectors only; scored on 12 hidden valid detectors.
| test days (17) | world model | same-day interpolation | train-period mean day |
| --- | --- | --- | --- |
| speed MAE hidden detectors (km/h) | 14.8 | 7.8 | 9.5 |
| route TT MAE (min) | 2.7 | 1.0 | 3.2 |
| days model better: speed / route TT | | 0 / 0 | 2 / 11 |
- Better than the mean-day fits (hidden speed 17.0 → 14.8) and better than the historical profile on route travel time on
  11 of 17 days, but far from same-day interpolation.
- Even with 3 × 26 free demand numbers per day the visible-detector error stays at 12 km/h (train 12.2, test 11.6) and the
  free-flow speed stays at 98–100 km/h (observed ~109). The remaining gap is structural (1 km cells, trapezoidal
  fundamental diagram, no capacity drop in the world model), not missing data or the calibration design.

## Model-structure ablations on the day-by-day calibration (2026-10-25) — `results/calibration_d26_*.json`
CTM refactor: `ctm.step` = `net.sub` substeps of `_substep(dt)` (1 for 1 km cells, bit-identical to before; 3 for 0.5 km),
optional learnable capacity drop on the sending side (`drop_logit`, 0.2·sigmoid per route), free-flow speed clamped at
the CFL bound, `network.remap_params` for 1 km → 0.5 km warm starts. Calibration options: cell size, capacity drop,
detector observation operator (speed × 1 ± 0.15 per detector), incident layer (day × cell × 90-min capacity reductions,
L1-sparse). Same 53 days, 36 train / 17 test, 12 hidden valid detectors.
| version | hidden speed (km/h) | route TT (min) | visible speed train / test |
| --- | --- | --- | --- |
| 1 km (baseline) | 14.8 | 2.70 | 12.2 / 11.6 |
| + capacity drop | 15.0 | 2.80 | 12.7 / 12.2 |
| + detector operator | 14.5 | 2.66 | 12.1 / 11.6 |
| + detector operator + incident layer | 16.0 | 2.83 | 12.3 / 12.7 |
| 0.5 km | 17.0 | 3.23 | 14.4 / 14.0 |
| 0.5 km + drop + detector | 16.9 | 3.05 | 11.7 / 11.1 |
| reference: same-day interpolation / train mean day | 7.8 / 9.5 | 0.99 / 3.19 | |
- None helps; the learned capacity drop shrinks to 1–5%; the incident layer switches on in 7.5% of cell-windows yet the
  visible fit stays at ~12 km/h.
- Error anatomy (baseline, visible detectors): free flow (55% of samples) model 97 vs observed 105 km/h, 41% of all
  error; 40–70 km/h model 12 km/h too fast; jams (< 40) 22 km/h too fast; the model misses ~45% of observed congested
  samples (speed < 70).
- Diagnosis: with a trapezoidal fundamental diagram a cell in free flow has speed = vf whatever its capacity, so the
  gradient of a missed jam with respect to that cell's capacity (or an incident reduction) is zero; the annealed
  smoothing only provides it early in training. More parameters or finer cells add more zero-gradient directions.
  Next: keep a smoothing floor during fitting (τ 6 and 12, same read-out at evaluation), then a smooth FD if needed.
| version (continued) | hidden speed | route TT | visible train / test | free-flow bias | congested bias | missed congestion |
| --- | --- | --- | --- | --- | --- | --- |
| baseline 1 km | 14.8 | 2.70 | 12.2 / 11.6 | −7.9 | +14.1 | 45% |
| + detector operator, smoothing floor τ = 6 | **14.4** | **2.41** | 11.7 / 11.1 | −7.5 | +11.6 | 44% |
| + detector operator, τ = 12 | 14.9 | 2.45 | 11.3 / 10.7 | | | |
| + detector operator, τ = 6, Huber 10 km/h | 15.0 | 3.27 | 13.6 / 13.3 | −5.9 | +25.1 | 70% |
- The smoothing floor gives the best version so far (route TT 2.41 vs 2.70; beats the train mean day on 11/17 days) but
  does not remove the free-flow bias. Huber loss lets free-flow speed rise a little but the fit gives up on congestion.
- Reading: within the CTM family and gradient calibration the gap to same-day interpolation (7.8 km/h, 1.0 min) is not
  closing; the remaining error is mostly congestion the model does not reproduce. Candidates: physics + learned
  residual (hybrid world model), or a smooth fundamental diagram.

## Fleet experiments on the April 2026 world model (2026-10-26) — `results/fleet_summary_2026i_*.json`
Same design as the v4 fleet runs, world model = April 2026 mean-day calibration (`wm_params_2026i.pt`).
Change vs metering only, mean [90% CI], days better / 10:
| share | F1 obedient | F2 negotiating | F3 empties only | F2 $/day |
| --- | --- | --- | --- | --- |
| CTM 1% | −0.24 [−0.33, −0.16] 10 | −0.25 [−0.37, −0.16] 10 | −0.17 [−0.29, −0.08] 10 | 131 |
| CTM 5% | −1.10 [−1.46, −0.77] 10 | −0.91 [−1.19, −0.65] 10 | −0.38 [−0.50, −0.27] 10 | 567 |
| CTM 10% | −2.35 [−3.07, −1.69] 10 | −2.12 [−2.75, −1.55] 10 | −0.74 [−0.99, −0.52] 10 | 1,126 |
| CTM 30% | −5.82 [−7.47, −4.17] 10 | −5.96 [−7.79, −4.17] 10 | −1.79 [−2.31, −1.30] 10 | 2,703 |
| METANET 10% | −0.06 [−0.13, +0.01] 7 | −0.07 [−0.13, −0.02] 8 | −0.06 [−0.08, −0.03] 9 | 1,740 |
| METANET 30% | −0.17 [−0.29, −0.05] 8 | −0.04 [−0.15, +0.09] 6 | −0.15 [−0.19, −0.10] 9 | 3,099 |
- CTM truth: same ordering as v4 (F1 ≈ F2 > F3, gains grow with share), about 40–65% of the v4 sizes because 2026 is
  less congested; riders no longer pay (occupied 24.7 min with or without coordination); cheaper.
- METANET truth: everything is within ±0.2%: this world is barely congested with the 2026 parameters, so the v4
  finding "obedience amplifies model error" is not tested here (no harm, no gain). The stress test needs a
  congested mismatched truth (e.g. METANET calibrated to the 2026 days) before that claim can be carried to 2026.

---
# Replay of 2026 test days (2026-10-26) — `src/replay.py`, `src/fleet_est.py`, `src/run_replay.py`
Truth = CTM reconstruction of each 2026 day from its calibrated demand multipliers (simulation built from that day's
data, not the real day). Operator = typical-day model (shared road + mean training-day demand), detector-only (noisy,
moving-horizon estimation), MPC every 5 min. A4 threshold pre-registered on 12 training days → 6.0 km/h (mean-first
rule; at 6.0 A4 almost never falls back). 17 test days never used for tuning.
| share | arm | vs metering-only [90% CI] | days better | worst day | $/day | riders' avg min | route-choice regret veh-h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | no control | +0.46 [+0.21, +0.72] | 5/17 | +1.81 | — | 30.3 | 337 |
| 10% | F1 obedient | −2.24 [−2.94, −1.63] | 17/17 | −0.31 | 0 | 31.4 | 596 |
| 10% | F2 negotiating | −1.73 [−2.41, −1.09] | 15/17 | +1.30 | 1,535 | 30.4 | 468 |
| 10% | F3 empties only | −0.86 [−1.16, −0.59] | 15/17 | +0.01 | 620 | 30.0 | 377 |
| 10% | A4 @ 6.0 | −1.77 [−2.43, −1.18] | 16/17 | +0.78 | 1,562 | 30.4 | 468 |
| 30% | no control | +0.54 [+0.21, +0.86] | 4/17 | +2.06 | — | 30.9 | 186 |
| 30% | F1 obedient | −5.70 [−6.68, −4.70] | 17/17 | −0.72 | 0 | 33.0 | 1,067 |
| 30% | F2 negotiating | −4.02 [−4.76, −3.27] | 17/17 | −0.82 | 4,786 | 30.9 | 474 |
| 30% | F3 empties only | −2.16 [−2.81, −1.58] | 17/17 | −0.21 | 1,475 | 30.4 | 249 |
| 30% | A4 @ 6.0 | −3.90 [−4.64, −3.15] | 17/17 | −0.82 | 4,712 | 30.9 | 476 |
- Under realistic day-to-day model error, obedience gives the largest gain and no harmful day: the 2017/METANET
  finding "obedience amplifies model error" does not carry over to 2026 replay. A4 as pre-registered adds nothing.
- The value of refusal here is distributional: obedience makes the fleet's riders pay (+2.3 min each at 30%) and
  doubles route-choice regret; negotiation keeps riders whole at ~30% lower system gain plus ~$4.8k/day;
  empties-only leaves riders untouched and keeps ~40% of the obedient gain.
- Tuning-day trade-off (descriptive only): a 3.0 km/h threshold cut the 30% worst day from +0.66% to +0.15% at a
  0.5-point mean cost.

---
# Chronological replication (2026-10-27): train Jan–Apr (75 weekdays), tune May (20), test June (18) — `PEMS_SPLIT=chrono`
Data: raw PeMS District 4, all 181 days Jan–Jun 2026 (weekends used for STID training only); metadata 2025-10-16,
2026-02-14, 2026-04-08 identical on the corridor. 2026 federal holidays excluded. All pre-registered rules unchanged.
## Forecasting, June test days (route travel-time MAE, min, +15 / +30 / +60)
| STID ensemble (5 seeds) | 1.26 / 1.85 / 2.32 |
| --- | --- |
| physics typical-day model + MHE | 2.72 / 3.33 / 4.02 |
| persistence | 1.56 / 3.04 / 4.84 |
| historical mean day | 3.49 / 3.75 / 4.04 |
Decision-relevant cases (|gap| > 2 min), +60 min correct route: STID 75%, historical 80%, persistence 66%, physics 52%.
## Fleet coordination on June replay days (vs metering only)
| share | F1 obedient | F2 negotiating | F3 empties only | A4 (threshold 6.0, chosen on May) |
| --- | --- | --- | --- | --- |
| 10% | −1.40 [−1.91, −0.88], 14/18, worst +1.12 | −1.43 [−1.87, −0.99], 15/18, worst +0.16 | −0.66 [−0.88, −0.42] | −1.42 (≈ F2) |
| 30% | −2.46 [−3.23, −1.70], 15/18, worst +1.96 | −2.20 [−2.98, −1.40], 15/18, worst +1.95 | −1.03 [−1.46, −0.57] | −2.13 (≈ F2) |
Riders' average minutes (F1 / F2 / metering): 10% 35.0 / 34.1 / 33.9; 30% 36.8 / 34.6 / 34.3. Route-choice regret
(veh-h, F1 / F2): 10% 558 / 422; 30% 1,039 / 420. F2 $/day: 1,307 (10%), 3,873 (30%).
## Negotiation as sensing (June, 30%, F2) — NOT replicated
Judgement change 100% / 50% / 25% detectors: −0.018 [−0.044, +0.007], −0.051 [−0.13, +0.04], −0.18 [−0.30, −0.07];
TTS change: +0.08 [−0.20, +0.40], +0.16 [−0.32, +0.66], −0.20 [−0.44, +0.02]. Pre-registered verdict: not supported
(interleaved run: supported at 100%/50%, not at 25%). Gap of the detector-only operator to the oracle: 1.4–1.9%.
## What holds across both splits
- STID ≫ physics model for forecasting; physics needed for interventions.
- Coordination gains are significant and grow with fleet share (smaller in the lighter June).
- Obedience shifts the cost onto the fleet's riders (+2.3–2.5 min each at 30%, ~2x route-choice regret); negotiation
  keeps riders whole at a system cost of 0–30% of the obedient gain (June: ≈0 at 10%, ~10% at 30%).
- A4 as pre-registered never adds value; negotiation-as-sensing is not robust; oracle gap 1.4–2.3%.

## Fleet with a forecast of its own trip (2026-10-29) — `src/fleet_forecast.py`, `results/fleet_forecast_chrono.csv`
Negotiating fleet (F2, 30% share) judges routes and offers with the experienced travel time of a vehicle departing
now instead of instantaneous route times; 18 June test days, `.venv`, paired with the 10-27 F2 runs. Reading
pre-registered in the script header.
| | perfect forecast | noisy forecast (sd 2.3 min) |
| --- | --- | --- |
| total veh-h vs F2-instantaneous (% of metering) | −0.01 [−0.28, +0.23] | +1.35 [+0.99, +1.72] |
| riders' minutes vs F2-instantaneous | −0.02 [−0.10, +0.05] | +0.45 [+0.27, +0.64] |
| total veh-h vs metering | −2.21% | −0.85% |
| payments per day | $3,967 (F2: $3,873) | $2,535 |
- Pre-registered verdict: perfect foresight by the fleet does **not** help (both intervals contain 0), so the fleet's
  information does not matter on this corridor; the noisy arm is not interpreted. (It is also not a fair comparison:
  the instantaneous fleet sees noiseless current times.)
- Contrast with the operator: an operator that knows the day's demand does 1.4–2.3% better. On this corridor the
  value of information sits with the planner, and the value of the fleet's agency is protecting its riders.

## DATA ISSUE found 2026-10-29: zero US-101 boundary inflow on some replay days
Station 402364 reports 0 veh/h (rows flagged measured) on train 2026-01-06 (blocks 15-20), 2026-02-05 (17-19) and
test 2026-06-12, 2026-06-19 (blocks 22-24, ~10:00-10:45, both Fridays) and **2026-06-30 (all day)**. The replayed
worlds of those days have no US-101 inflow in those blocks, so 3 of the 18 June test days are partly unrealistic.
All arms share the same worlds, so paired comparisons are less affected, but every June result should be re-reported
without these 3 days and the boundary fixed (treat 0 as missing, fill from the typical profile, recalibrate).

## DemandCast: operator re-anchors its physics model on a learned demand forecast (2026-10-29)
`src/demand_forecast.py` predicts, as of each decision (inputs up to clock − 10 min, last 1 h, 135 nodes), the day's
deviation from the training-mean multipliers for the next 4 blocks (z0, zA, zB, log US-101 inflow). Selected on May:
M2 (neural ensemble, 5 seeds) over ridge and typical. June RMSE typical → DemandCast: z0 0.34 → 0.24, zA 0.30 → 0.21,
zB 0.38 → 0.29 (log inflow 2.24 → 2.01, dominated by the zero-inflow detector days).
`src/run_demandcast.py`: same June closed loop, operator parameters for the next 4 blocks reset from the forecast
before every decision. Reading pre-registered in the script header. Paired with the 10-27 typical-operator runs.
| 30% share | DemandCast − typical (% of metering veh-h) | days better | vs metering (typical → DemandCast) | riders' min | pay/day |
| --- | --- | --- | --- | --- | --- |
| F2, 18 days | **−0.36 [−0.60, −0.11]** | 15/18 | −2.20 → −2.56% | +0.12 [−0.05, +0.29] | $3,873 → $4,420 |
| F2, without 3 faulty days | −0.38 [−0.65, −0.10] | 13/15 | −2.57 → −2.94% | +0.18 [+0.01, +0.36] | |
| F1, 18 days (secondary) | −0.45 [−0.89, +0.07] | 14/18 | −2.46 → −2.92% | −0.16 | |
- Pre-registered verdict: helps (F2 interval below 0). Closes ~25% of the typical-to-oracle gap (1.36% → 1.01%).
- Small rider cost with negotiation (+0.1 to +0.2 min) and 14% higher payments: the better-informed operator buys
  more rerouting.
- Main-table ordering without the 3 faulty days (vs metering, riders): 10% F1 −1.59% +1.29 min, F2 −1.62% +0.27,
  F3 −0.71% −0.15; 30% F1 −2.94% +2.92, F2 −2.57% +0.30, F3 −1.07% −0.28. Same ordering as with all 18 days.
