"""Replay experiments: detector-only operator on reconstructed 2026 days, fleet levels F0-F3 and the A4 fallback.

Pre-registered (written before any tuning run):
- Tuning days: every third training day (12 days); fleet shares 10% and 30%.
- Threshold candidates (km/h of realised 5-15 min prediction error): 3.0, 3.5, 4.0, 5.0, 6.0.
- Selection: the threshold with the lowest mean change in all vehicle-hours vs metering-only (A2-F0), averaged over
  tuning days and both shares; if two are within 0.05 points, the one with the smaller worst-day change wins.
- The selected threshold is then locked and used once on the 17 test days (never tuned on them).
- Test report per share and arm: mean change vs metering-only and vs no control with 90% day-bootstrap CI, days better
  than metering, worst day, payments, riders' and human drivers' average time, route-choice regret (all / when the
  routes differ by > 2 min), share of decisions in F3 for A4.
Usage: run_replay.py tune | test <threshold>
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from network import build_default_net
from run_experiments import controlled_ramps, boot_ci
from replay import load_replay, day_slice, typical_path
import split
from fleet_est import run_day_replay

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHARES = (0.10, 0.30)
THRESHOLDS = (3.0, 3.5, 4.0, 5.0, 6.0)


def _job(args):
    split, i, agent, level, s_f, thr = args
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    shared, days = load_replay(split)
    p_op = torch.load(typical_path())
    return run_day_replay(net, shared, day_slice(days, i), p_op, agent, level, s_f, controlled_ramps(net, p_op),
                          threshold=thr, seed=i)


def run(jobs, workers=8):
    with Pool(workers) as pool:
        return pd.DataFrame(pool.map(_job, jobs))


def arm_name(r):
    return f"{r['agent']}-{r['level']}" + (f"@{r['threshold']}" if r["level"] == "A4" else "")


def summarize(df):
    df = df.copy(); df["arm"] = df.apply(arm_name, axis=1)
    out = []
    for s_f, g in df.groupby("s_f"):
        meter = g[g.arm == "A2-F0"].set_index("date").tts_all_vh
        base = g[g.arm == "A0-F0"].set_index("date").tts_all_vh if (g.arm == "A0-F0").any() else None
        for arm, a in g.groupby("arm"):
            a = a.set_index("date")
            dm = (a.tts_all_vh - meter) / meter * 100
            m, lo, hi = boot_ci(dm.values)
            row = {"s_f": s_f, "arm": arm, "vs_metering_pct": m, "ci90": [lo, hi], "worst_day_pct": float(dm.max()),
                   "worst_day": str(dm.idxmax()), "days_better": int((dm < 0).sum()), "n_days": int(len(dm)),
                   "pay_usd_day": float(a.pay_usd.mean()), "occ_avg_min": float(a.occ_avg_min.mean()),
                   "human_avg_min": float(a.human_avg_min.mean()), "regret_vh": float(a.regret_vh.mean()),
                   "regret_decisive_vh": float(a.regret_decisive_vh.mean()), "f3_share": float(a.f3_share.mean())}
            if base is not None:
                d0 = (a.tts_all_vh - base) / base * 100
                row["vs_no_control_pct"] = float(d0.mean())
            out.append(row)
    return out


def main(mode, thr=None):
    if mode == "tune":
        part = "tune" if split.MODE == "chrono" else "train"          # chrono: all May days; else every third train day
        shared, days = load_replay(part)
        idx = list(range(len(days["days"]))) if part == "tune" else list(range(0, len(days["days"]), 3))
        arms = [("A2", "F0", None), ("A2", "F2", None), ("A2", "F3", None)] + [("A2", "A4", t) for t in THRESHOLDS]
        jobs = [(part, i, a, l, s, t) for i, (a, l, t), s in itertools.product(idx, arms, SHARES)]
        df = run(jobs)
        df.to_csv(os.path.join(ROOT, "results", f"replay_tune_runs{split.tag()}.csv"), index=False)
        summ = summarize(df)
        a4 = [r for r in summ if r["arm"].startswith("A2-A4")]
        score = {}
        for t in THRESHOLDS:
            rs = [r for r in a4 if r["arm"] == f"A2-A4@{t}"]
            score[t] = (float(np.mean([r["vs_metering_pct"] for r in rs])), float(max(r["worst_day_pct"] for r in rs)))
        best = min(score.values(), key=lambda v: v[0])[0]
        cands = [t for t, v in score.items() if v[0] - best <= 0.05]
        chosen = min(cands, key=lambda t: score[t][1])
        json.dump({"summary": summ, "score": {str(k): v for k, v in score.items()}, "chosen_threshold": chosen},
                  open(os.path.join(ROOT, "results", f"replay_tune_summary{split.tag()}.json"), "w"), indent=2)
        print("scores (mean vs metering %, worst day %):", score, "-> chosen", chosen)
    else:
        shared, days = load_replay("test")
        idx = list(range(len(days["days"])))
        arms = [("A0", "F0", None), ("A2", "F0", None), ("A2", "F1", None), ("A2", "F2", None), ("A2", "F3", None),
                ("A2", "A4", thr)]
        jobs = [("test", i, a, l, s, t) for i, (a, l, t), s in itertools.product(idx, arms, SHARES)]
        df = run(jobs)
        df.to_csv(os.path.join(ROOT, "results", f"replay_test_runs{split.tag()}.csv"), index=False)
        summ = summarize(df)
        json.dump({"threshold": thr, "summary": summ}, open(os.path.join(ROOT, "results", f"replay_test_summary{split.tag()}.json"), "w"), indent=2)
    for r in summ:
        print(f"s_F={r['s_f']:.2f} {r['arm']:12s} vsMeter {r['vs_metering_pct']:+.2f}% [{r['ci90'][0]:+.2f},{r['ci90'][1]:+.2f}] "
              f"better {r['days_better']}/{r['n_days']} worst {r['worst_day_pct']:+.2f}% ({r['worst_day']}) pay ${r['pay_usd_day']:.0f} "
              f"occ {r['occ_avg_min']:.1f} human {r['human_avg_min']:.1f} regret {r['regret_vh']:.0f}/{r['regret_decisive_vh']:.0f} F3 {r['f3_share']:.2f}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else None)
