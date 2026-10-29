"""DemandCast: forecast the physics model's demand inputs, as of each decision time, for the operator's world model.

The typical-day operator plans with the mean demand multipliers of the training days. DemandCast predicts, from
detector data available at the decision time, the day's deviation from that mean for the next H = 4 blocks (1 h):
mainline origin z0, route-A ramps zA, route-B ramps zB, and log US-101 boundary inflow. Targets are the daily
calibration's multipliers (results/day_params_d26_tau6s_chrono_*.pt), which also drive the replayed test worlds.

As-of rule: a decision at clock c (05:00 + 5k min) uses 5-min slots stamped up to c - 10 min (one slot of lag), the
last L = 12 slots (1 h), all corridor nodes (mainline, ramps, boundary): speed, flow, measured flag.
Models: M0 typical (zero deviation), M1 ridge on the last-3-slot means, M2 small neural ensemble (per-node temporal
encoder + node, time-of-day and day-of-week embeddings, attention pooling over nodes; 5 seeds).
Split: train Jan-Apr, select (model, ridge alpha, epochs) on May, report June once.
Usage: PEMS_SPLIT=chrono .venv/bin/python src/demand_forecast.py   -> results/demandcast_chrono.pt (+ log)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PEMS_SPLIT", "chrono")
import numpy as np, pandas as pd, torch
from torch import nn
import ctm
from stid_wm import build_nodes, build_tensor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L_IN, H, N_SLOTS = 12, 4, 72
SERIES = ("z0", "zA", "zB", "lbnd")


def load_targets():
    out = {}
    for part in ("train", "tune", "test"):
        d = torch.load(os.path.join(ROOT, "results", f"day_params_d26_tau6s_chrono_{part}.pt"))
        out[part] = {"days": list(d["days"]), "Z": torch.stack([d["z0"], d["zA"], d["zB"], d["bnd"].clamp_min(1).log()], 1).numpy()}
    return out                                                           # Z: (D, 4, N_BLOCKS)


def cur_block(k):
    return (30 + 5 * k) // ctm.BLOCK_MIN                                 # blocks start at 04:30


def build(parts, idx, S, F, M, zbar):
    """X (n, L, N, 3), tod (n,), dow (n,), Y (n, 4, H) deviation from the train mean, W (n, 4, H) valid mask."""
    pos = {t: i for i, t in enumerate(idx)}
    Xs, tod, dow, Ys, Ws, key = [], [], [], [], [], []
    for di, day in enumerate(parts["days"]):
        Z = parts["Z"][di]
        for k in range(N_SLOTS):
            clock = pd.Timestamp(f"{day} 05:00") + pd.Timedelta(minutes=5 * k)
            end = pos.get(clock - pd.Timedelta(minutes=10))
            if end is None or end < L_IN - 1:
                continue
            sl = slice(end - L_IN + 1, end + 1)
            x = np.stack([np.nan_to_num(S[sl]) / 100.0, np.nan_to_num(F[sl]) / 2000.0, M[sl]], -1)
            b = cur_block(k)
            y = np.zeros((4, H)); w = np.zeros((4, H))
            for h in range(H):
                if b + h < ctm.N_BLOCKS:
                    y[:, h] = Z[:, b + h] - zbar[:, b + h]; w[:, h] = 1.0
            Xs.append(x); tod.append(k); dow.append(pd.Timestamp(day).weekday()); Ys.append(y); Ws.append(w)
            key.append((day, k))
    return (torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(tod), torch.tensor(dow),
            torch.tensor(np.array(Ys), dtype=torch.float32), torch.tensor(np.array(Ws), dtype=torch.float32), key)


class DemandNet(nn.Module):
    def __init__(self, n_nodes, d=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(L_IN * 3, d), nn.GELU(), nn.Linear(d, d))
        self.node = nn.Parameter(torch.randn(n_nodes, d) * 0.1)
        self.tod = nn.Embedding(N_SLOTS, d); self.dow = nn.Embedding(5, d)
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, 4 * H))

    def forward(self, x, tod, dow):
        B, L, N, C = x.shape
        h = self.enc(x.permute(0, 2, 1, 3).reshape(B, N, L * C)) + self.node[None]
        a = torch.softmax(self.att(torch.tanh(h)), 1)
        g = (a * h).sum(1)
        return self.head(torch.cat([g, self.tod(tod) + self.dow(dow)], 1)).reshape(B, 4, H)


def wmse(p, y, w):
    return (((p - y) ** 2) * w).sum() / w.sum()


def ridge_feats(X):
    return torch.cat([X[:, -3:].mean(1).flatten(1), X[:, -12:-9].mean(1).flatten(1)], 1)


def fit_ridge(Xtr, Ytr, alpha):
    A = torch.cat([ridge_feats(Xtr), torch.ones(len(Xtr), 1)], 1).double()
    B = Ytr.flatten(1).double()
    Wt = torch.linalg.solve(A.T @ A + alpha * torch.eye(A.shape[1], dtype=torch.float64), A.T @ B)
    return lambda X: (torch.cat([ridge_feats(X), torch.ones(len(X), 1)], 1).double() @ Wt).float().reshape(-1, 4, H)


def train_net(seed, tr, va, epochs=60):
    torch.manual_seed(seed)
    net = DemandNet(tr[0].shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    best, best_state, curve = 1e9, None, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr[0]))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            opt.zero_grad(); wmse(net(tr[0][j], tr[1][j], tr[2][j]), tr[3][j], tr[4][j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = wmse(net(*va[:3]), va[3], va[4]).item()
        curve.append(v)
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best, int(np.argmin(curve))


def per_series(p, y, w):
    return {s: float(((((p - y) ** 2) * w)[:, i].sum() / w[:, i].sum()).sqrt()) for i, s in enumerate(SERIES)}


def main():
    torch.set_num_threads(2)
    t0 = time.time()
    T = load_targets()
    zbar = T["train"]["Z"].mean(0)
    nodes, _ = build_nodes()
    idx, S, F, M = build_tensor(nodes, months=(1, 2, 3, 4, 5, 6))
    tr, va, te = (build(T[p], idx, S, F, M, zbar) for p in ("train", "tune", "test"))
    print("samples", len(tr[0]), len(va[0]), len(te[0]), "nodes", tr[0].shape[2], f"{time.time()-t0:.0f}s", flush=True)
    res = {"M0": {"tune": wmse(torch.zeros_like(va[3]), va[3], va[4]).item()}}
    best_a, best_r = None, 1e9
    for a in (1, 10, 100, 1000, 10000):
        r = wmse(fit_ridge(tr[0], tr[3], a)(va[0]), va[3], va[4]).item()
        if r < best_r:
            best_a, best_r = a, r
    ridge = fit_ridge(tr[0], tr[3], best_a)
    res["M1"] = {"tune": best_r, "alpha": best_a}
    nets = [train_net(s, tr, va) for s in range(5)]
    with torch.no_grad():
        ens_va = torch.stack([n(*va[:3]) for n, _, _ in nets]).mean(0)
    res["M2"] = {"tune": wmse(ens_va, va[3], va[4]).item(), "best_epochs": [e for _, _, e in nets]}
    chosen = min(res, key=lambda m: res[m]["tune"])
    with torch.no_grad():
        preds = {"M0": torch.zeros_like(te[3]), "M1": ridge(te[0]),
                 "M2": torch.stack([n(*te[:3]) for n, _, _ in nets]).mean(0)}
    for m in res:
        res[m]["test"] = wmse(preds[m], te[3], te[4]).item()
        res[m]["test_rmse_by_series"] = per_series(preds[m], te[3], te[4])
    res["chosen_on_may"] = chosen
    print(json.dumps(res, indent=1), flush=True)
    fc = {key: preds[chosen][i] for i, key in enumerate(te[5])}
    torch.save({"chosen": chosen, "forecast": fc, "zbar": torch.tensor(zbar), "H": H, "res": res,
                "nets": [n.state_dict() for n, _, _ in nets]}, os.path.join(ROOT, "results", "demandcast_chrono.pt"))
    print("saved", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
