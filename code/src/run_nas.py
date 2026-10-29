"""Negotiation-as-sensing and oracle experiments on the 17 replayed 2026 test days (fleet share 30%, F2).

Pre-registered reading (written before the run):
- Supported: adding the fleet's realised acceptance shares to the estimator lowers the operator's route-time judgement
  error (paired by day, 90% bootstrap CI below 0) AND lowers all vehicle-hours for at least one detector density
  (90% CI below 0).
- Only the first holds: "responses sharpen the operator's judgement but this does not yet translate into decisions".
- Neither: not supported.
- The oracle planner (true day demand and true state) gives the decision gap of the detector-only operator.
Arms: oracle; detector density 100% / 50% / 25% x responses off / on.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from network import build_default_net
from run_experiments import controlled_ramps, boot_ci
from replay import load_replay, day_slice, day_folded_params, typical_path
import split
from fleet_est import run_day_replay

ROOT = os.path.join(os.path.dirname(__file__), "..")
S_F = 0.30
ARMS = [("oracle", 1.0, False)] + [("est", d, n) for d in (1.0, 0.5, 0.25) for n in (False, True)]


def _job(args):
    i, (kind, det, nas) = args
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    shared, days = load_replay("test")
    day = day_slice(days, i)
    p_typ = torch.load(typical_path())
    p_op = day_folded_params(shared, net, day) if kind == "oracle" else p_typ
    r = run_day_replay(net, shared, day, p_op, "A2", "F2", S_F, controlled_ramps(net, p_typ), seed=i,
                       oracle=(kind == "oracle"), nas=nas, det_frac=det)
    r["kind"] = kind
    return r


def main():
    _, days = load_replay("test")
    jobs = list(itertools.product(range(len(days["days"])), ARMS))
    with Pool(8) as pool:
        df = pd.DataFrame(pool.map(_job, jobs))
    df.to_csv(os.path.join(ROOT, "results", f"nas_runs{split.tag()}.csv"), index=False)
    out = {}
    orc = df[df.kind == "oracle"].set_index("date")
    for det in (1.0, 0.5, 0.25):
        off = df[(df.kind == "est") & (df.det_frac == det) & (~df.nas)].set_index("date")
        on = df[(df.kind == "est") & (df.det_frac == det) & (df.nas)].set_index("date")
        dj = (on.judge_err_min - off.judge_err_min)
        dt = (on.tts_all_vh - off.tts_all_vh) / off.tts_all_vh * 100
        gap_off = (off.tts_all_vh - orc.tts_all_vh) / orc.tts_all_vh * 100
        gap_on = (on.tts_all_vh - orc.tts_all_vh) / orc.tts_all_vh * 100
        out[str(det)] = {"judge_err_off": float(off.judge_err_min.mean()), "judge_err_on": float(on.judge_err_min.mean()),
                         "judge_change": boot_ci(dj.values), "tts_change_pct": boot_ci(dt.values),
                         "days_tts_better": int((dt < 0).sum()), "gap_to_oracle_off_pct": boot_ci(gap_off.values),
                         "gap_to_oracle_on_pct": boot_ci(gap_on.values), "pay_off": float(off.pay_usd.mean()),
                         "pay_on": float(on.pay_usd.mean()), "pay_oracle": float(orc.pay_usd.mean()),
                         "regret_off": float(off.regret_vh.mean()), "regret_on": float(on.regret_vh.mean()),
                         "regret_oracle": float(orc.regret_vh.mean())}
    json.dump(out, open(os.path.join(ROOT, "results", f"nas_summary{split.tag()}.json"), "w"), indent=2)
    for k, v in out.items():
        print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else [round(x, 3) for x in b] if isinstance(b, (list, tuple)) else b) for a, b in v.items()}))


if __name__ == "__main__":
    main()
