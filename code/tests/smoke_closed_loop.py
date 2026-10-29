import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, ctm, agentic
from network import build_default_net
net, _, _ = build_default_net()
p = ctm.default_params(net)
ctrl = torch.tensor([i for i in range(net.R) if not net.ramp_major[i]][:6])
for agent, level in [("A0", 0), ("A1", 0), ("A2", 0), ("A2", 3)]:
    t0 = time.time()
    r = agentic.run_day(net, p, p, agent, level, 0.3, 0.0, day_seed=1, incident=True, ctrl_idx=ctrl)
    print(agent, level, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("agent", "level")}, f"{time.time()-t0:.1f}s", flush=True)
