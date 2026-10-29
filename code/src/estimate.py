"""Detector-only state estimation and short-term forecasting with the calibrated world model.

Moving-horizon estimation (MHE): at decision time t, start from the model's mean-day state at t - W, then fit
per-cell density multipliers (initial state) and three day-level demand scales (origin, route-A ramps, route-B
ramps) to the last W minutes of detector speeds (and flows when available) by gradient descent through the CTM.
The estimated state at t is then rolled forward H minutes with the demand scales held fixed.
All test days at one decision time are estimated in one batch.

Forecast targets: detector speeds at +15/+30/+60 min and route travel times, on test weekdays (May-Jun 2017).
Baselines: persistence (last observed 5-min value) and the historical mean profile of training weekdays.
This is the state the operator would see in the closed loop instead of the true state.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
import ctm
from data import load_speeds, weekdays, day_window, instantaneous_tt_min, route_length_km
from network import build_default_net

ROOT = os.path.join(os.path.dirname(__file__), "..")
SLOT0_STEP = 60                          # 05:00 is step 60 from the 04:30 start
W_MIN, H_MIN = 30, 60
HORIZONS = (15, 30, 60)


def load_params(path):
    p = torch.load(path)
    return {k: v.float() for k, v in p.items()}


def route_prob_fn(p, bias):
    return lambda a, b: torch.sigmoid(p["b0"] + bias + p["b1"] * (a - b))[:, None].expand(-1, 2)


def demand(p, net, t, s):
    """Batched demand with day scales s = (s0, sA, sB), each (B,)."""
    blk = min(ctm.N_BLOCKS - 1, t // (2 * ctm.BLOCK_MIN))
    d0 = torch.exp(p["log_d0"][blk] + s[0])
    lr = ctm.ramp_log_demand(p, net, blk)
    route_s = torch.stack([s[1], s[2]], 1)[:, net.ramp_route - 1]
    return d0, torch.exp(lr[None, :] + route_s)


def rollout(p, net, st, t0, n_steps, s, bias, record_every=ctm.STEPS_PER_5MIN, smooth=0.0):
    """Roll the CTM forward; returns final state and per-5-min mean cell speeds and outflows (B, S, M)."""
    meters = net.ramp_cap.expand(st["n"].shape[0], -1)
    vf = ctm.vf_cells(p, net)
    rp = route_prob_fn(p, bias)
    Vs, Qs, av, aq = [], [], 0.0, 0.0
    for k in range(n_steps):
        t = t0 + k
        st, info = ctm.step(st, p, net, t, meters, rp, demand=demand(p, net, t, s))
        av = av + ctm.fd_speed(st["n"].sum(1), info["cap"], vf, net, smooth=smooth)
        aq = aq + info["out_tot"] / ctm.DT_H
        if k % record_every == record_every - 1:
            Vs.append(av / record_every); Qs.append(aq / record_every); av, aq = 0.0, 0.0
    return st, torch.stack(Vs, 1), torch.stack(Qs, 1)


def prior_state(p, net, t):
    """Mean-day model state at step t (batch 1)."""
    st = ctm.init_state(net, 1)
    z = torch.zeros(1)
    with torch.no_grad():
        st, _, _ = rollout(p, net, st, 0, t, (z, z, z), z)
    return st


def expand(st, B):
    return {k: v.expand(B, *v.shape[1:]).clone() for k, v in st.items()}


def mhe(p, net, t_dec, obsV, visV, cell_of, obsQ=None, visQ=None, iters=60, lr=0.08, smooth=2.0, offV=None):
    """Estimate the state at step t_dec for B days. obsV: (B, W/5, n_sens) speeds over the window."""
    B = obsV.shape[0]
    W = W_MIN * 2
    st0 = expand(prior_state(p, net, t_dec - W), B)
    x = {"logm": torch.zeros(B, net.M), "s0": torch.zeros(B), "sA": torch.zeros(B), "sB": torch.zeros(B),
         "bias": torch.zeros(B), "logq": torch.zeros(B)}
    x = {k: v.requires_grad_(True) for k, v in x.items()}
    opt = torch.optim.Adam(x.values(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        st = {"n": st0["n"] * torch.exp(x["logm"])[:, None, :], "qo": st0["qo"] * torch.exp(x["logq"])[:, None],
              "qr": st0["qr"] * torch.exp(x["logq"])[:, None, None]}
        _, V, Q = rollout(p, net, st, t_dec - W, W, (x["s0"], x["sA"], x["sB"]), x["bias"], smooth=smooth)
        Vd = V[:, :, cell_of] + (offV if offV is not None else 0.0)
        loss = (((Vd - obsV) / 10.0)[:, :, visV] ** 2).mean()
        if obsQ is not None:
            loss = loss + (((Q[:, :, cell_of] - obsQ) / 500.0)[:, :, visQ] ** 2).mean()
        loss = loss + 0.05 * (x["logm"] ** 2).mean() + 0.5 * sum((x[k] ** 2).mean() for k in ("s0", "sA", "sB", "bias", "logq")) \
            + 0.5 * (x["logm"].diff(dim=1) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x["logm"].clamp_(-2.0, 2.0)
            for k in ("s0", "sA", "sB", "bias", "logq"):
                x[k].clamp_(-1.0, 1.0)
    with torch.no_grad():
        st = {"n": st0["n"] * torch.exp(x["logm"])[:, None, :], "qo": st0["qo"] * torch.exp(x["logq"])[:, None],
              "qr": st0["qr"] * torch.exp(x["logq"])[:, None, None]}
        st, Vw, _ = rollout(p, net, st, t_dec - W, W, (x["s0"], x["sA"], x["sB"]), x["bias"])
    return st, {k: v.detach() for k, v in x.items()}, Vw


def route_tt(Vsens, sens):
    """Route travel times (min) from detector speeds, Vsens (..., n_sens) -> (..., 2)."""
    out = []
    for r, name in ((1, "A"), (2, "B")):
        m = (sens.route == r).values
        s = sens[m]
        flat = Vsens[..., m].reshape(-1, m.sum())
        tt = instantaneous_tt_min(pd.DataFrame(flat, columns=s.sensor_id.values), s, route_length_km(name))
        out.append(tt.reshape(Vsens.shape[:-1]))
    return np.stack(out, -1)


def model_mean_speeds(p, net, cell_of):
    """Mean-day model speeds at detectors per 5-min slot from 05:00, (72, n_sens)."""
    st = ctm.init_state(net, 1)
    z = torch.zeros(1)
    with torch.no_grad():
        st, _, _ = rollout(p, net, st, 0, SLOT0_STEP, (z, z, z), z)
        _, V, _ = rollout(p, net, st, SLOT0_STEP, 72 * ctm.STEPS_PER_5MIN, (z, z, z), z)
    return V[0][:, cell_of].numpy()


def main(params="results/wm_params_v3.pt", tag="_v3", use_flows=False, iters=60, anomaly=False):
    torch.manual_seed(0)
    p = load_params(os.path.join(ROOT, params))
    net, sA, sB = build_default_net()
    sens = pd.concat([sA.assign(route=1), sB.assign(route=2)], ignore_index=True)
    ids = sens.sensor_id.values
    cell_of = torch.tensor([net.sensor_cell[int(s)] for s in ids])
    sp = load_speeds()
    days = weekdays(sp.index)
    tr = [d for d in days if d.month <= 4]
    te = [d for d in days if d.month >= 5]
    Y = np.array([a for a in (day_window(sp, d)[ids].values for d in te) if a.shape[0] == 72], dtype=np.float32)
    hist = np.mean([a for a in (day_window(sp, d)[ids].values for d in tr) if a.shape[0] == 72], 0)
    obsQ_all = None
    if use_flows:
        from flows import route_flows
        fl = route_flows(); fl.columns = fl.columns.astype(int)
        flq = fl.reindex(columns=ids)
        obsQ_all = np.array([day_window(flq, d).values for d in te if len(day_window(sp, d)) == 72], dtype=np.float32)
        visQ = torch.tensor(np.isfinite(obsQ_all).all((0, 1)))
        obsQ_all = np.nan_to_num(obsQ_all)
    visV = torch.ones(len(ids), dtype=torch.bool)        # all detectors observed in operation
    # anomaly mode: detector speed = historical mean + (model day - model mean day); the offset is learned on train days
    off = (hist - model_mean_speeds(p, net, cell_of)).astype(np.float32) if anomaly else np.zeros_like(hist, dtype=np.float32)
    rows, t0 = [], time.time()
    for hhmm in ("06:30", "07:00", "07:30", "08:00", "08:30", "09:00", "09:30"):
        h, m = map(int, hhmm.split(":"))
        slot = (h * 60 + m - 300) // 5                   # index of the 5-min slot starting at the decision time
        t_dec = SLOT0_STEP + slot * ctm.STEPS_PER_5MIN
        w = W_MIN // 5
        obsV = torch.tensor(Y[:, slot - w:slot])
        oQ = torch.tensor(obsQ_all[:, slot - w:slot]) if use_flows else None
        st, x, _ = mhe(p, net, t_dec, obsV, visV, cell_of, oQ, visQ if use_flows else None, iters=iters,
                       offV=torch.tensor(off[slot - w:slot]))
        with torch.no_grad():
            _, Vf, _ = rollout(p, net, st, t_dec, H_MIN * 2, (x["s0"], x["sA"], x["sB"]), x["bias"])
        Vf = Vf[:, :, cell_of].numpy() + off[slot:slot + 12][None]   # (D, 12, n_sens): slots slot .. slot+11
        for hz in HORIZONS:
            k = hz // 5 - 1                               # forecast for the slot ending at t + hz
            truth = Y[:, slot + k]
            preds = {"world_model": Vf[:, k], "persistence": Y[:, slot - 1], "historical": np.broadcast_to(hist[slot + k], truth.shape)}
            tt_true = route_tt(truth, sens)
            for name, pr in preds.items():
                rows.append({"time": hhmm, "horizon_min": hz, "method": name,
                             "speed_mae": float(np.abs(pr - truth).mean()),
                             "tt_mae_min": float(np.abs(route_tt(np.asarray(pr), sens) - tt_true).mean()),
                             "faster_route_correct": float(((route_tt(np.asarray(pr), sens).argmin(-1)) == tt_true.argmin(-1)).mean())})
        print(hhmm, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", f"nowcast{tag}.csv"), index=False)
    summ = df.groupby(["horizon_min", "method"])[["speed_mae", "tt_mae_min", "faster_route_correct"]].mean().round(3)
    print(summ.to_string())
    json.dump({"n_test_days": int(Y.shape[0]), "params": params, "use_flows": use_flows, "iters": iters, "anomaly": anomaly,
               "summary": summ.reset_index().to_dict(orient="records")},
              open(os.path.join(ROOT, "results", f"nowcast{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    a = sys.argv
    main(params=a[1] if len(a) > 1 else "results/wm_params_v3.pt", tag=a[2] if len(a) > 2 else "_v3",
         use_flows=(len(a) > 3 and a[3] == "flows"), iters=int(a[4]) if len(a) > 4 else 60,
         anomaly=(len(a) > 5 and a[5] == "anomaly"))
