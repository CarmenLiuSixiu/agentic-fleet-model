"""Calibrate the world model to observed PEMS-BAY speeds by gradient descent through the CTM.

Train: mean weekday (Jan-Apr 2017). Test: mean weekday (May-Jun 2017).
30% of detectors are hidden from calibration and used only for evaluation.
Baseline for hidden detectors: linear interpolation along the route from visible detectors.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
import ctm
from data import load_speeds, weekdays, day_window
from network import build_default_net

ROOT = os.path.join(os.path.dirname(__file__), "..")
torch.manual_seed(0)


def targets():
    net, sA, sB = build_default_net()
    sp = load_speeds()
    days = weekdays(sp.index)
    sens = pd.concat([sA.assign(route=1), sB.assign(route=2)], ignore_index=True)
    ids = sens.sensor_id.values

    def mean_profile(sel):
        arr = [day_window(sp, d)[ids].values for d in sel]
        arr = [a for a in arr if a.shape[0] == 72]
        return np.mean(arr, 0), len(arr)

    train_days = [d for d in days if d.month <= 4]
    test_days = [d for d in days if d.month >= 5]
    Ytr, ntr = mean_profile(train_days)
    Yte, nte = mean_profile(test_days)
    rng = np.random.default_rng(0)
    hidden = np.zeros(len(ids), bool)
    hidden[rng.choice(len(ids), int(round(0.3 * len(ids))), replace=False)] = True
    return net, sens, Ytr, Yte, hidden, ntr, nte


def simulate_speeds(p, net, smooth=0.0):
    """Cell speeds averaged per 5-min slot from 05:00 to 11:00, shape (72, M)."""
    st = ctm.init_state(net, 1)
    meters = net.ramp_cap.expand(1, -1)
    route_prob = lambda a, b: torch.sigmoid(p["b0"] + p["b1"] * (a - b)).reshape(1, 1).expand(1, 2)
    slots, acc = [], 0.0
    for t in range(ctm.N_STEPS):
        st, info = ctm.step(st, p, net, t, meters, route_prob)
        if t >= 60:
            vf_cell = ctm.vf_cells(p, net)
            v = ctm.fd_speed(st["n"].sum(1), info["cap"], vf_cell, net, smooth=smooth)[0]
            acc = acc + v
            if (t - 60) % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                slots.append(acc / ctm.STEPS_PER_5MIN); acc = 0.0
    return torch.stack(slots)


def interp_baseline(Y, sens, hidden):
    """Linear interpolation along each route from visible detectors."""
    out = Y.copy()
    for r in (1, 2):
        m = (sens.route == r).values
        x = sens.x_km.values[m]; vis = ~hidden[m]
        idx = np.where(m)[0]
        for t in range(Y.shape[0]):
            out[t, idx] = np.interp(x, x[vis], Y[t, idx][vis])
    return out


def main(iters=160, lr=0.05, cap_reg=2.0, tag="", anneal=0.0, major_init=0.0):
    net, sens, Ytr, Yte, hidden, ntr, nte = targets()
    cell_of = torch.tensor([net.sensor_cell[int(s)] for s in sens.sensor_id])
    Y = torch.tensor(Ytr, dtype=torch.float32)
    vis = torch.tensor(~hidden)
    p0 = ctm.default_params(net)
    if major_init > 0:
        p0["log_a"] = torch.where(net.ramp_major, torch.tensor(float(np.log(major_init))), p0["log_a"])
    p = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    opt = torch.optim.Adam(p.values(), lr=lr)
    log = []
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        tau = max(2.0, anneal * (1 - it / (0.6 * iters))) if anneal > 0 else 2.0
        V = simulate_speeds(p, net, smooth=tau)[:, cell_of]
        fit = ((V - Y)[:, vis] ** 2).mean()
        reg = cap_reg * (p["cap_logit"] ** 2).mean() + 5.0 * (p["log_g"].diff(dim=1) ** 2).mean() \
            + 5.0 * (p["log_d0"].diff() ** 2).mean()
        loss = fit + 100.0 * reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            p["vf"].clamp_(70.0, 115.0); p["b1"].clamp_(0.0, 1.0)
        if it % 10 == 0 or it == iters - 1:
            mae = (V - Y)[:, vis].abs().mean().item()
            log.append({"iter": it, "rmse_vis": fit.sqrt().item(), "mae_vis": mae, "sec": round(time.time() - t0, 1)})
            print(log[-1], flush=True)

    with torch.no_grad():
        V = simulate_speeds(p, net, smooth=0.0)[:, cell_of].numpy()
    base_tr = interp_baseline(Ytr, sens, hidden)
    base_te = interp_baseline(Yte, sens, hidden)
    hv = hidden
    res = {
        "n_train_days": ntr, "n_test_days": nte, "n_detectors": int(len(hidden)), "n_hidden": int(hidden.sum()),
        "wm_mae_visible_train": float(np.abs(V - Ytr)[:, ~hv].mean()),
        "wm_mae_hidden_train": float(np.abs(V - Ytr)[:, hv].mean()),
        "wm_mae_visible_test": float(np.abs(V - Yte)[:, ~hv].mean()),
        "wm_mae_hidden_test": float(np.abs(V - Yte)[:, hv].mean()),
        "interp_mae_hidden_train": float(np.abs(base_tr - Ytr)[:, hv].mean()),
        "interp_mae_hidden_test": float(np.abs(base_te - Yte)[:, hv].mean()),
        "vf": p["vf"].tolist(), "b0": float(p["b0"]), "b1": float(p["b1"]),
        "cap_mult_min": float(ctm.cap_mult(p).min()), "cap_mult_max": float(ctm.cap_mult(p).max()),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    torch.save({k: v.detach() for k, v in p.items()}, os.path.join(ROOT, "results", f"wm_params{tag}.pt"))
    json.dump({"result": res, "log": log}, open(os.path.join(ROOT, "results", f"calibration{tag}.json"), "w"), indent=2)
    np.savez(os.path.join(ROOT, "results", f"calibration_arrays{tag}.npz"), V=V, Ytr=Ytr, Yte=Yte, hidden=hidden,
             base_te=base_te, x=sens.x_km.values, route=sens.route.values)


if __name__ == "__main__":
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    cr = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    an = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    mi = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    main(iters=it, lr=lr, cap_reg=cr, tag=tag, anneal=an, major_init=mi)
