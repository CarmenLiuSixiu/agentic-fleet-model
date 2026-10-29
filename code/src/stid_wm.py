"""STID-based world model for the corridor (step 1: background forecasting from PeMS only).

STID (Shao et al., CIKM 2022, https://arxiv.org/abs/2208.05233): MLP encoder over the flattened input window plus
learnable node, time-of-day and day-of-week identities. Re-implemented here (same architecture as the official
stid_arch.py) with two extensions used in later steps: static road-structure features per node, and optional
per-node exogenous channels (events at decision time, actions) concatenated to the input window.

Data: raw PeMS 2026, corridor mainline stations (2017 IDs present in 2026) + on/off-ramp stations + US-101 boundary.
Channels: speed (km/h), flow (veh/h), measured flag; imputed rows are zero with flag 0, never targets.
Targets: mainline speeds at +15/+30/+60 min (steps 3, 6, 12), masked MAE on measured values.
Split: identical to calibrate_daily_2026 / estimate_2026 (17 test days = every third usable weekday) so results are
directly comparable with the physics world model; validation = last 20% of training days.
Evaluation at 06:30-09:30 every 30 min with the same metrics as estimate_2026 (speed MAE on valid detectors, route
travel-time MAE, faster-route accuracy, wrong-choice time loss, decision-relevant cases with |gap| > 2 min).
"""
from __future__ import annotations
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
from torch import nn
from data import load_meta, route_sensors, day_window, weekdays
from flows import boundary_sensors
from pems_raw import load_month, observed_mask
from calibrate_daily_2026 import load_days, route_tt
from calibrate_flow import interp_nan

ROOT = os.path.join(os.path.dirname(__file__), "..")
STEPS_PER_DAY = 288
L_IN = 12
HORIZONS = (3, 6, 12)                     # 5-min steps: +15, +30, +60 min


class MLPRes(nn.Module):
    def __init__(self, d, p=0.15):
        super().__init__()
        self.fc1, self.fc2 = nn.Conv2d(d, d, 1), nn.Conv2d(d, d, 1)
        self.act, self.drop = nn.ReLU(), nn.Dropout(p)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(x))))


class STIDX(nn.Module):
    """STID with optional static node features and exogenous per-node channels.
    x: (B, L, N, C) history; tod: (B,) int slot of the last input step; dow: (B,) int; exo: (B, N, E) or None."""

    def __init__(self, n_nodes, in_ch, n_static, n_exo=0, embed=32, node_dim=32, tid=32, diw=32, layers=3,
                 out_len=len(HORIZONS)):
        super().__init__()
        self.node_emb = nn.Parameter(torch.empty(n_nodes, node_dim)); nn.init.xavier_uniform_(self.node_emb)
        self.tid_emb = nn.Parameter(torch.empty(STEPS_PER_DAY, tid)); nn.init.xavier_uniform_(self.tid_emb)
        self.diw_emb = nn.Parameter(torch.empty(7, diw)); nn.init.xavier_uniform_(self.diw_emb)
        self.ts_emb = nn.Conv2d(in_ch * L_IN, embed, 1)
        self.static_emb = nn.Conv2d(n_static, 16, 1)
        self.exo_emb = nn.Conv2d(n_exo, 16, 1) if n_exo else None
        hid = embed + node_dim + tid + diw + 16 + (16 if n_exo else 0)
        self.enc = nn.Sequential(*[MLPRes(hid) for _ in range(layers)])
        self.head = nn.Conv2d(hid, out_len, 1)

    def forward(self, x, tod, dow, static, exo=None):
        B, L, N, C = x.shape
        h = [self.ts_emb(x.permute(0, 2, 1, 3).reshape(B, N, L * C).transpose(1, 2).unsqueeze(-1)),
             self.node_emb.T.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1),
             self.tid_emb[tod].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1),
             self.diw_emb[dow].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1),
             self.static_emb(static.T.unsqueeze(0).expand(B, -1, -1).unsqueeze(-1))]
        if self.exo_emb is not None:
            h.append(self.exo_emb(exo.transpose(1, 2).unsqueeze(-1)))
        return self.head(self.enc(torch.cat(h, 1))).squeeze(-1)          # (B, out_len, N)


def build_nodes():
    meta17 = load_meta()
    ml = pd.concat([route_sensors(meta17, r).assign(route=i + 1) for i, r in enumerate("AB")], ignore_index=True)
    rm = pd.read_csv(os.path.join(ROOT, "results", "ramps_2026_meta.csv"))
    bnd = boundary_sensors()[:3]
    nodes = pd.DataFrame({
        "id": list(ml.sensor_id.astype(int)) + list(rm.ID.astype(int)) + bnd,
        "kind": ["ML"] * len(ml) + list(rm.Type) + ["BND"] * len(bnd),
        "route": list(ml.route) + [1 if r == "A" else 2 for r in rm.route] + [2] * len(bnd),
        "x_km": list(ml.x_km) + list(rm.x_km) + [-0.5, -0.8, -1.1]})
    return nodes, len(ml)


def build_tensor(nodes, months=(4, 5, 6)):
    """(T, N, 3) [speed, flow, measured] on the full 5-min index of the given months, plus timestamps."""
    parts = []
    ml_ids = [i for i, k in zip(nodes.id, nodes.kind) if k in ("ML", "BND")]
    rm_ids = [i for i, k in zip(nodes.id, nodes.kind) if k not in ("ML", "BND")]
    for m in months:
        a = load_month(2026, m, ml_ids, "corridor")
        b = load_month(2026, m, rm_ids, "corridor_ramps")
        idx = pd.date_range(f"2026-{m:02d}-01", periods=pd.Timestamp(f"2026-{m:02d}-01").days_in_month * STEPS_PER_DAY, freq="5min")
        sp = pd.concat([a["speed"], b["speed"]], axis=1).reindex(index=idx, columns=nodes.id)
        fl = pd.concat([a["flow"], b["flow"]], axis=1).reindex(index=idx, columns=nodes.id)
        ob = pd.concat([observed_mask(a), observed_mask(b)], axis=1).reindex(index=idx, columns=nodes.id).fillna(False)
        sp = sp.where(ob); fl = fl.where(ob)
        parts.append((idx, sp.values, fl.values, ob.values.astype(np.float32)))
    idx = np.concatenate([p[0] for p in parts])
    S = np.concatenate([p[1] for p in parts]); F = np.concatenate([p[2] for p in parts]); M = np.concatenate([p[3] for p in parts])
    return pd.DatetimeIndex(idx), S, F, M


def samples_for_days(idx, days, t_lo="00:00", t_hi="23:55"):
    """End indices (last input step) for every 5-min step of the given days with full history and horizon."""
    pos = []
    dset = {pd.Timestamp(d).date() for d in days}
    for k in range(L_IN - 1, len(idx) - max(HORIZONS)):
        if idx[k].date() in dset:
            pos.append(k)
    return np.array(pos)


class Scaler:
    def __init__(self, S, F, M, tr_pos):
        sel = np.zeros(len(S), bool); sel[tr_pos] = True
        self.sm, self.ss = np.nanmean(S[sel]), np.nanstd(S[sel])
        self.fm, self.fs = np.nanmean(F[sel]), np.nanstd(F[sel])

    def x(self, S, F, M):
        s = np.nan_to_num((S - self.sm) / self.ss) * M
        f = np.nan_to_num((F - self.fm) / self.fs) * M
        return np.stack([s, f, M], -1).astype(np.float32)


_TOD, _DOW = None, None


def batch(X, S, M, idx, pos, ml_n, dev):
    """Vectorised window gather: X (T, N, C) -> (B, L_IN, N, C); targets (B, 3, ml_n)."""
    global _TOD, _DOW
    if _TOD is None or len(_TOD) != len(idx):
        _TOD = ((idx.hour * 60 + idx.minute) // 5).values.astype(np.int64); _DOW = idx.weekday.values.astype(np.int64)
    pos = np.asarray(pos)
    win = pos[:, None] + np.arange(-L_IN + 1, 1)[None, :]
    hz = pos[:, None] + np.array(HORIZONS)[None, :]
    xs = torch.from_numpy(X[win]).to(dev)
    ys = torch.from_numpy(np.nan_to_num(S[hz][:, :, :ml_n]).astype(np.float32)).to(dev)
    ms = torch.from_numpy(M[hz][:, :, :ml_n] > 0).to(dev)
    return xs, ys, ms, torch.from_numpy(_TOD[pos]).to(dev), torch.from_numpy(_DOW[pos]).to(dev)


def train(seed=0, epochs=60, lr=2e-3, bs=64, dev="cpu", tag="_stid", months=None):
    import split
    months = months or split.months(); tag = tag + split.tag()
    torch.manual_seed(seed); np.random.seed(seed)
    nodes, ml_n = build_nodes()
    idx, S, F, M = build_tensor(nodes, months)
    net, sens, days, V, Q, MM, B = load_days()                       # same usable days and split as the physics model
    import split
    tr_i, tu_i, te_i = split.indices(days)
    tr_days = [days[i] for i in tr_i]; te_days = [days[i] for i in te_i]
    if tu_i:                                   # chrono: validate on the tuning month; train on all days (incl. weekends) up to April
        val_days = [days[i] for i in tu_i]
        all_days = pd.DatetimeIndex(sorted(set(idx.normalize())))
        fit_days = [d for d in all_days if d < val_days[0]]
    else:
        n_val = max(1, len(tr_days) // 5)
        fit_days, val_days = tr_days[:-n_val], tr_days[-n_val:]
    p_fit, p_val = samples_for_days(idx, fit_days), samples_for_days(idx, val_days)
    sc = Scaler(S, F, M, p_fit)
    X = sc.x(S, F, M)
    static = torch.tensor(np.stack([nodes.kind.eq("ML"), nodes.kind.eq("OR"), nodes.kind.isin(["FR", "FF"]),
                                    nodes.kind.eq("BND"), nodes.route.eq(2), nodes.x_km / 30.0], 1).astype(np.float32), device=dev)
    model = STIDX(len(nodes), 3, static.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(epochs * 0.4), int(epochs * 0.7)], 0.3)
    best, best_state, log = 1e9, None, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(p_fit); tl = []
        for k in range(0, len(perm), bs):
            xs, ys, ms, tod, dow = batch(X, S, M, idx, perm[k:k + bs], ml_n, dev)
            pred = model(xs, tod, dow, static)[:, :, :ml_n] * sc.ss + sc.sm
            loss = (pred - ys).abs()[ms].mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl.append(loss.item())
        sched.step()
        model.eval(); vl = []
        with torch.no_grad():
            for k in range(0, len(p_val), 256):
                xs, ys, ms, tod, dow = batch(X, S, M, idx, p_val[k:k + 256], ml_n, dev)
                pred = model(xs, tod, dow, static)[:, :, :ml_n] * sc.ss + sc.sm
                vl.append((pred - ys).abs()[ms].mean().item())
        v = float(np.mean(vl)); log.append({"epoch": ep, "train_mae": float(np.mean(tl)), "val_mae": v})
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(log[-1], flush=True)
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "scaler": [sc.sm, sc.ss, sc.fm, sc.fs], "nodes": nodes.to_dict("list"), "ml_n": ml_n},
               os.path.join(ROOT, "results", f"wm{tag}_seed{seed}.pt"))
    return model, sc, nodes, ml_n, idx, S, F, M, X, static, days, te_i, tr_i, sens, V, MM, log


def evaluate(models, sc, nodes, ml_n, idx, S, M, X, static, days, te_i, tr_i, sens, V, MM, tag, dev="cpu"):
    valid = MM[tr_i].mean((0, 1)) >= 0.5                      # valid detectors, same rule as the physics evaluation
    rows = []
    for d_i in te_i:
        day = days[d_i]
        for hhmm in ("06:30", "07:00", "07:30", "08:00", "08:30", "09:00", "09:30"):
            t_dec = pd.Timestamp(f"{day.date()} {hhmm}")
            p = int(np.where(idx == t_dec)[0][0]) - 1           # last input step = slot ending at the decision time
            with torch.no_grad():
                xs, ys, ms, tod, dow = batch(X, S, M, idx, np.array([p]), ml_n, dev)
                preds = [m(xs, tod, dow, static)[0, :, :ml_n] * sc.ss + sc.sm for m in models]
                pr = torch.stack(preds).mean(0).cpu().numpy()   # (3, ml_n), ensemble mean
                spread = torch.stack(preds).std(0).mean().item() if len(preds) > 1 else 0.0
            for j, h in enumerate(HORIZONS):
                truth = np.where(M[p + h, :ml_n] > 0, S[p + h, :ml_n], np.nan)
                truth_full = interp_nan(truth[None], sens, valid)
                tt_true = route_tt(truth_full, sens)[0]
                if not np.isfinite(tt_true).all():
                    continue
                pv = pr[j].copy(); pv[~valid] = np.nan
                tt_p = route_tt(interp_nan(pv[None], sens, valid), sens)[0]
                ok = valid & np.isfinite(truth)
                gap = float(abs(tt_true[0] - tt_true[1])); wrong = tt_p.argmin() != tt_true.argmin()
                rows.append({"day": str(day.date()), "time": hhmm, "horizon_min": h * 5, "method": "stid",
                             "speed_mae": float(np.abs(pr[j] - truth)[ok].mean()), "tt_mae_min": float(np.abs(tt_p - tt_true).mean()),
                             "faster_route_correct": float(not wrong), "true_gap_min": gap, "choice_loss_min": gap if wrong else 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", f"nowcast{tag}.csv"), index=False)
    summ = df.groupby("horizon_min")[["speed_mae", "tt_mae_min", "faster_route_correct", "choice_loss_min"]].mean().round(3)
    dec = df[df.true_gap_min > 2.0].groupby("horizon_min")[["faster_route_correct", "choice_loss_min"]].mean().round(3)
    print(summ.to_string()); print("decision-relevant (|gap| > 2 min):"); print(dec.to_string())
    json.dump({"summary": summ.reset_index().to_dict(orient="records"), "decision_relevant": dec.reset_index().to_dict(orient="records")},
              open(os.path.join(ROOT, "results", f"nowcast{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    a = sys.argv
    n_seeds = int(a[1]) if len(a) > 1 else 1
    epochs = int(a[2]) if len(a) > 2 else 60
    torch.set_num_threads(int(a[3]) if len(a) > 3 else 2)
    dev = a[4] if len(a) > 4 else "cpu"
    models = []
    for sd in range(n_seeds):
        out = train(seed=sd, epochs=epochs, dev=dev)
        models.append(out[0])
        print(f"seed {sd} done", flush=True)
    model, sc, nodes, ml_n, idx, S, F, M, X, static, days, te_i, tr_i, sens, V, MM, log = out
    for m in models:
        m.eval()
    import split
    evaluate(models, sc, nodes, ml_n, idx, S, M, X, static, days, te_i, tr_i, sens, V, MM, "_stid" + split.tag(), dev=dev)
