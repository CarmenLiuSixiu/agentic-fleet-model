"""Fleet experiments (docs/fleet_agent_design.md section 4).

Arms per fleet share s_F: A0-F0 (no control, self-interested fleet), A2-F0 (MPC metering only), and A2 with
F1 obedient / F2 negotiating / F3 empties-only negotiation, each without and with the depot-hold lever.
Main outcome: all vehicle-hours including held empties (tts_all), paired by day against A0-F0 and against A2-F0.
Usage: run_fleet.py <params.pt> <truth: ctm|metanet> <tag> [shares comma-separated] [empty share]
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from network import build_default_net
from run_experiments import controlled_ramps, boot_ci, DAYS
from fleet import run_day_fleet, EMPTY_SHARE

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARMS = [("A0", "F0", False), ("A2", "F0", False)] + [("A2", lv, h) for lv in ("F1", "F2", "F3") for h in (False, True)]


def _job(args):
    params_path, truth, s_f, e, (agent, level, hold), (seed, incident) = args
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    p = torch.load(params_path)
    return run_day_fleet(net, p, p, agent, level, s_f, hold, seed, incident, controlled_ramps(net, p),
                         truth_model=truth, empty_share=e)


def summarize(df):
    rows = []
    for s_f, g in df.groupby("s_f"):
        base = g[(g.agent == "A0")].set_index("day")
        meter = g[(g.agent == "A2") & (g.level == "F0")].set_index("day")
        for (agent, level, hold), a in g.groupby(["agent", "level", "hold"]):
            a = a.set_index("day")
            d0 = (a.tts_all_vh - base.tts_all_vh) / base.tts_all_vh * 100
            dm = (a.tts_all_vh - meter.tts_all_vh) / base.tts_all_vh * 100
            saved = (base.tts_all_vh - a.tts_all_vh).sum()
            pay = (a.pay_route_usd + a.pay_hold_usd)
            m0, lo0, hi0 = boot_ci(d0.values)
            mm, lom, him = boot_ci(dm.values)
            rows.append({"s_f": s_f, "agent": agent, "level": level, "hold": bool(hold),
                         "tts_change_vs_A0_pct": m0, "ci90_vs_A0": [lo0, hi0],
                         "tts_change_vs_metering_pct": mm, "ci90_vs_metering": [lom, him],
                         "days_better_than_metering": int((a.tts_all_vh < meter.tts_all_vh).sum()),
                         "pay_usd_day": float(pay.mean()), "usd_per_vh_saved": float(pay.sum() / saved) if saved > 0 else None,
                         "hold_vh_day": float(a.hold_vh.mean()),
                         "occ_avg_min": float(a.occ_avg_min.mean()), "occ_avg_min_A0": float(base.occ_avg_min.mean()),
                         "human_avg_min": float(a.human_avg_min.mean()), "human_avg_min_A0": float(base.human_avg_min.mean()),
                         "emp_vkt_change_pct": float(((a.emp_vkt - base.emp_vkt) / base.emp_vkt.clip(lower=1e-6)).mean() * 100),
                         "worst_ramp_wait_min": float(a.worst_ramp_wait_min.mean())})
    return rows


def main(params_path, truth, tag, shares=(0.01, 0.05, 0.10, 0.30), e=EMPTY_SHARE, workers=8):
    jobs = [(params_path, truth, s, e, arm, day) for s, arm, day in itertools.product(shares, ARMS, DAYS)]
    with Pool(workers) as pool:
        df = pd.DataFrame(pool.map(_job, jobs))
    df.to_csv(os.path.join(ROOT, "results", f"fleet_runs{tag}.csv"), index=False)
    rows = summarize(df)
    json.dump(rows, open(os.path.join(ROOT, "results", f"fleet_summary{tag}.json"), "w"), indent=2)
    for r in rows:
        print(f"s_F={r['s_f']:.2f} {r['agent']}-{r['level']}{'+hold' if r['hold'] else '':5s} vsA0 {r['tts_change_vs_A0_pct']:+.2f}% "
              f"vsMeter {r['tts_change_vs_metering_pct']:+.2f}% [{r['ci90_vs_metering'][0]:+.2f},{r['ci90_vs_metering'][1]:+.2f}] "
              f"better {r['days_better_than_metering']}/10 pay ${r['pay_usd_day']:.0f} "
              f"occ {r['occ_avg_min']:.1f}/{r['occ_avg_min_A0']:.1f} human {r['human_avg_min']:.1f}/{r['human_avg_min_A0']:.1f}")


if __name__ == "__main__":
    a = sys.argv
    shares = tuple(float(x) for x in a[4].split(",")) if len(a) > 4 else (0.01, 0.05, 0.10, 0.30)
    main(a[1], a[2], a[3], shares, float(a[5]) if len(a) > 5 else EMPTY_SHARE)
