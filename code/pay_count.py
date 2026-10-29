"""June chrono test days, F2/F3: count fleet through vehicles at the diverge and those paid (induced), to get
payment per paid vehicle and the share of fleet vehicles paid. Writes results/pay_count_chrono.csv."""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ["PEMS_SPLIT"] = "chrono"
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import fleet, replay, fleet_est
from network import build_default_net
from run_experiments import controlled_ramps
from replay import load_replay, day_slice, typical_path

_orig = fleet.FleetWorld.step
def step(self, meters, lever):
    pay0 = self.fm["pay_route"]
    t0 = self.t
    info = _orig(self, meters, lever)
    c = self.__dict__.setdefault("cnt", {"fleet_occ": 0.0, "fleet_emp": 0.0, "paid_occ": 0.0, "paid_emp": 0.0})
    if t0 >= 60:
        oF = info["o_F"][0]; ind = info["ind"][0]
        c["fleet_occ"] += float(oF[0]); c["fleet_emp"] += float(oF[1])
        c["paid_occ"] += float(oF[0] * ind[0]); c["paid_emp"] += float(oF[1] * ind[1])
    return info
fleet.FleetWorld.step = step

def job(a):
    i, level, s_f = a
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    shared, days = load_replay("test")
    p_op = torch.load(typical_path())
    holder = {}
    orig_init = replay.ReplayFleetWorld.__init__
    def init(self, *aa, **kk):
        orig_init(self, *aa, **kk); holder["w"] = self
    replay.ReplayFleetWorld.__init__ = init
    fleet_est.ReplayFleetWorld = replay.ReplayFleetWorld
    r = fleet_est.run_day_replay(net, shared, day_slice(days, i), p_op, "A2", level, s_f, controlled_ramps(net, p_op), seed=i)
    r.update(holder["w"].cnt)
    r["torch"] = torch.__version__
    return r

if __name__ == "__main__":
    shared, days = load_replay("test")
    levels = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("F2", "F3")
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    jobs = list(itertools.product(range(len(days["days"])), levels, (0.10, 0.30)))
    with Pool(4) as p:
        df = pd.DataFrame(p.map(job, jobs))
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"pay_count_chrono{tag}.csv"), index=False)
    g = df.groupby(["level", "s_f"])[["tts_all_vh", "pay_usd", "fleet_occ", "fleet_emp", "paid_occ", "paid_emp"]].mean()
    g["usd_per_paid"] = g.pay_usd / (g.paid_occ + g.paid_emp)
    g["paid_share"] = (g.paid_occ + g.paid_emp) / (g.fleet_occ + g.fleet_emp)
    print(g.round(2).to_string())
