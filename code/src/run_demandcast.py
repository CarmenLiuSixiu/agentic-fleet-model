"""Closed loop with DemandCast: the operator re-anchors its world model's demand on the forecast before each decision.

Same as the 10-27 June test (fleet_est.run_day_replay, A2, detector-only operator, same seeds), except that at every
decision the operator's parameters for the next H blocks (log_d0, log_r, US-101 inflow) are set to the typical-day
values plus DemandCast's predicted deviation for that day and decision slot (results/demandcast_chrono.pt). Blocks
before the current one keep the last forecast made for them; later blocks keep the typical day.

Pre-registered reading (written before the run), 18 June test days, paired by day, 90% day-bootstrap:
- Primary: total vehicle-hours, DemandCast operator minus typical-day operator (10-27 runs), F2 at 30% share.
  "Helps" if the interval is below 0.
- Secondary: share of the typical-to-oracle gap that is closed (oracle = run_nas.py chrono, F2 30%, full detectors),
  riders' minutes, payments; F1 at 30% reported the same way.
- If the chosen forecaster on May is M0 (typical), there is nothing to test and the run is skipped.
Usage: PEMS_SPLIT=chrono .venv/bin/python src/run_demandcast.py [workers]
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PEMS_SPLIT", "chrono")
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import ctm, fleet_est
from replay import load_replay, day_slice, typical_path
from network import build_default_net
from run_experiments import controlled_ramps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DC = os.path.join(ROOT, "results", "demandcast_chrono.pt")


def anchor(p_op, typ, shared, net, dc, day, k):
    """Set p_op (in place, shared with the estimator and the planner) for blocks b..b+H-1 from the forecast."""
    dev = dc["forecast"].get((day, k))
    if dev is None:
        return
    b = (30 + 5 * k) // ctm.BLOCK_MIN
    route_a = (net.ramp_route == 1)
    for h in range(dc["H"]):
        bb = b + h
        if bb >= ctm.N_BLOCKS:
            break
        p_op["log_d0"][bb] = typ["log_d0"][bb] + dev[0, h]
        p_op["log_r"][:, bb] = typ["log_r"][:, bb] + torch.where(route_a, dev[1, h], dev[2, h])
        p_op["bnd_major_B"][bb] = typ["bnd_major_B"][bb] * torch.exp(dev[3, h])


def job(a):
    i, level, s_f = a
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    shared, days = load_replay("test")
    typ = torch.load(typical_path())
    p_op = {k: v.clone() for k, v in typ.items()}
    dc = torch.load(DC)
    day = day_slice(days, i)
    dstr = str(day["days"])[:10]
    orig = fleet_est.fleet_mpc_plan

    def planner(shim, wm, *aa, **kk):
        anchor(p_op, typ, shared, net, dc, dstr, (shim.t - 60) // 10)
        return orig(shim, wm, *aa, **kk)
    fleet_est.fleet_mpc_plan = planner
    r = fleet_est.run_day_replay(net, shared, day, p_op, "A2", level, s_f, controlled_ramps(net, typ), seed=i)
    r["operator"] = "demandcast"
    return r


if __name__ == "__main__":
    dc = torch.load(DC)
    if dc["chosen"] == "M0":
        print("May selected the typical day (M0); nothing to test"); sys.exit(0)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    shared, days = load_replay("test")
    jobs = list(itertools.product(range(len(days["days"])), ("F2", "F1"), (0.30,)))
    with Pool(workers) as p:
        df = pd.DataFrame(p.map(job, jobs))
    df.to_csv(os.path.join(ROOT, "results", "demandcast_closed_loop_chrono.csv"), index=False)
    print("saved", len(df), flush=True)
