import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, ctm, agentic
from network import build_default_net
net, _, _ = build_default_net()
p = torch.load(os.path.join(os.path.dirname(__file__), "..", "results", "wm_params.pt"))
ctrl = torch.tensor([i for i in range(net.R) if not net.ramp_major[i]][:6])
print("park-and-ride cell", agentic.pr_cell(net))
for level in (3, 4):
    t0 = time.time()
    r = agentic.run_day(net, p, p, "A2", level, 0.3, 0.0, day_seed=7, incident=True, ctrl_idx=ctrl)
    print(level, {k: round(v, 2) for k, v in r.items() if isinstance(v, float)}, f"{time.time()-t0:.1f}s")
