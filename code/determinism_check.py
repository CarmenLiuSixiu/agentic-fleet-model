"""Run the same June day / arm twice in separate processes; identical outputs => planner is deterministic in this
environment, so differences from the 10-27 run come from a code or library change, not run-to-run noise."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ["PEMS_SPLIT"] = "chrono"
import torch
from multiprocessing import Pool
from network import build_default_net
from run_experiments import controlled_ramps
from replay import load_replay, day_slice, typical_path
from fleet_est import run_day_replay

def job(i):
    torch.set_num_threads(1)
    net, _, _ = build_default_net()
    shared, days = load_replay("test")
    p_op = torch.load(typical_path())
    r = run_day_replay(net, shared, day_slice(days, i), p_op, "A2", "F2", 0.30, controlled_ramps(net, p_op), seed=i)
    return r["date"], r["tts_all_vh"], r["pay_usd"]

if __name__ == "__main__":
    i = 3            # 2026-06 day with a large gap between the two runs is picked by the caller; default day 3
    i = int(sys.argv[1]) if len(sys.argv) > 1 else i
    with Pool(2) as p:
        print(p.map(job, [i, i]), torch.__version__, flush=True)
