"""X5: the same closed loop with a detector-only operator (operator_est), paired with the oracle-state runs.

Reads the oracle runs from results/closed_loop_runs.csv (same params, same 10 days) and reports, per vehicle level,
the TTS change vs no control with the estimated state, and how much of the oracle-state gain is retained.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from network import build_default_net
from run_experiments import controlled_ramps, boot_ci, DAYS, P_AG
from operator_est import run_day_est

ROOT = os.path.join(os.path.dirname(__file__), "..")
LEVELS = (0, 1, 2, 3, 4)


def _job(args):
    params_path, level, seed, incident, truth_model = args
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    p = torch.load(params_path)
    return run_day_est(net, p, p, level, P_AG, 0.0, seed, incident, controlled_ramps(net, p), truth_model=truth_model)


def main(params_path, oracle_csv, tag="", workers=8, truth_model="ctm"):
    jobs = [(params_path, lv, s, inc, truth_model) for lv, (s, inc) in itertools.product(LEVELS, DAYS)]
    with Pool(workers) as pool:
        est = pd.DataFrame(pool.map(_job, jobs))
    est.to_csv(os.path.join(ROOT, "results", f"closed_loop_est_runs{tag}.csv"), index=False)
    orc = pd.read_csv(oracle_csv)
    base = orc[(orc.agent == "A0") & (orc.kappa == 0)].set_index("day").tts_vh
    rows = []
    for lv in LEVELS:
        e = est[est.level == lv].set_index("day")
        o = orc[(orc.agent == "A2") & (orc.level == lv) & (orc.kappa == 0)].set_index("day")
        de = (e.tts_vh - base) / base * 100
        do = (o.tts_vh - base) / base * 100
        m, lo, hi = boot_ci(de.values)
        mo = boot_ci(do.values)[0]
        gap = boot_ci((de - do).values)
        rows.append({"level": lv, "tts_change_est_pct": m, "ci90": [lo, hi], "tts_change_oracle_pct": mo,
                     "est_minus_oracle_pct": gap[0], "gap_ci90": [gap[1], gap[2]],
                     "retained_share": (m / mo) if mo < -0.5 else None,
                     "days_est_beats_A2L0_est": int((e.tts_vh < est[est.level == 0].set_index("day").tts_vh).sum()) if lv else None,
                     "worst_ramp_wait_est": float(e.worst_ramp_wait_min.mean()),
                     "worst_ramp_wait_oracle": float(o.worst_ramp_wait_min.mean()),
                     "spend_est": float(e.spend_usd.mean()), "state_err_vehkm": float(e.state_err_vehkm.mean()),
                     "est_sec": float(e.est_sec.mean())})
    json.dump(rows, open(os.path.join(ROOT, "results", f"closed_loop_est_summary{tag}.json"), "w"), indent=2)
    for r in rows:
        print(r)


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else os.path.join(ROOT, "results", "wm_params.pt"),
         a[2] if len(a) > 2 else os.path.join(ROOT, "results", "closed_loop_runs.csv"),
         a[3] if len(a) > 3 else "", truth_model=a[4] if len(a) > 4 else "ctm")
