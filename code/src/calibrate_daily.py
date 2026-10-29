"""Day-to-day calibration (in the spirit of computational-graph calibration from system-level data).

Shared across days: free-flow speeds, per-cell capacities, off-ramp exit fractions, demand time shapes,
ramp demand scales, route-choice sensitivity. Day-specific: origin, route-A ramp and route-B ramp demand
scales and a route-choice bias (4 numbers per day).
Train: random weekdays Jan-Apr 2017. Test: weekdays May-Jun 2017, where shared parameters are frozen and
only the 4 day-specific numbers are estimated from visible detectors; hidden detectors (30%) are never used.
Baselines on hidden detectors: linear interpolation from that day's visible detectors, and the detector's
own historical mean profile (uses the hidden detector's history, so it is a stronger reference).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch
import ctm
from data import load_speeds, weekdays, day_window
from network import build_default_net
from calibrate import interp_baseline

ROOT = os.path.join(os.path.dirname(__file__), "..")
torch.manual_seed(0)


def day_arrays(sp, ids, days):
    out, kept = [], []
    for d in days:
        a = day_window(sp, d)[ids].values
        if a.shape[0] == 72:
            out.append(a); kept.append(d)
    return np.array(out, dtype=np.float32), kept


def simulate(shared, day, net, smooth):
    """Batched over days. Returns cell speeds per 5-min slot, shape (D, 72, M)."""
    D = day["s0"].shape[0]
    st = ctm.init_state(net, D)
    meters = net.ramp_cap.expand(D, -1)
    vf_cell = ctm.vf_cells(shared, net)
    rp = lambda a, b: torch.sigmoid(shared["b0"] + day["b0"] + shared["b1"] * (a - b))[:, None].expand(D, 2)
    route_scale = torch.stack([day["sA"], day["sB"]], 1)            # (D,2)
    slots, acc = [], 0.0
    for t in range(ctm.N_STEPS):
        blk = min(ctm.N_BLOCKS - 1, t // (2 * ctm.BLOCK_MIN))
        d0 = torch.exp(shared["log_d0"][blk] + day["s0"])
        dr = torch.exp(shared["log_a"][None, :] + shared["log_g"][net.ramp_route - 1, blk][None, :]
                       + route_scale[:, net.ramp_route - 1])
        st, info = ctm.step(st, shared, net, t, meters, rp, demand=(d0, dr))
        if t >= 60:
            acc = acc + ctm.fd_speed(st["n"].sum(1), info["cap"], vf_cell, net, smooth=smooth)
            if (t - 60) % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                slots.append(acc / ctm.STEPS_PER_5MIN); acc = 0.0
    return torch.stack(slots, 1)


def new_day_params(D):
    return {k: torch.zeros(D, requires_grad=True) for k in ("s0", "sA", "sB", "b0")}


def fit(shared, day, net, Y, vis, cell_of, iters, lr, anneal, train_shared, log_every=10, tag=""):
    params = list(day.values()) + ([v for v in shared.values()] if train_shared else [])
    opt = torch.optim.Adam(params, lr=lr)
    Yt = torch.tensor(Y)
    log = []
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        tau = max(2.0, anneal * (1 - it / (0.6 * iters)))
        V = simulate(shared, day, net, tau)[:, :, cell_of]
        fit_ = ((V - Yt)[:, :, vis] ** 2).mean()
        reg = 0.05 * sum((day[k] ** 2).mean() for k in day)
        if train_shared:
            reg = reg + 0.5 * (shared["cap_logit"] ** 2).mean() + 0.2 * (shared["vf_logit"] ** 2).mean() + 5.0 * (shared["log_g"].diff(dim=1) ** 2).mean() \
                + 5.0 * (shared["log_d0"].diff() ** 2).mean()
        (fit_ + 100.0 * reg).backward()
        opt.step()
        with torch.no_grad():
            shared["vf"].clamp_(70.0, 115.0); shared["b1"].clamp_(0.0, 1.0)
            for k in day:
                day[k].clamp_(-1.0, 1.0)
        if it % log_every == 0 or it == iters - 1:
            log.append({"iter": it, "tau": round(tau, 1), "mae_vis": (V - Yt)[:, :, vis].abs().mean().item(),
                        "sec": round(time.time() - t0, 1)})
            print(tag, log[-1], flush=True)
    return log


def main(n_train=30, n_test=20, iters=240, test_iters=80):
    net, sA, sB = build_default_net()
    sens = pd.concat([sA.assign(route=1), sB.assign(route=2)], ignore_index=True)
    ids = sens.sensor_id.values
    cell_of = torch.tensor([net.sensor_cell[int(s)] for s in ids])
    sp = load_speeds()
    days = weekdays(sp.index)
    rng = np.random.default_rng(0)
    tr_pool = [d for d in days if d.month <= 4]
    te_pool = [d for d in days if d.month >= 5]
    tr_days = sorted(rng.choice(tr_pool, n_train, replace=False))
    te_days = sorted(rng.choice(te_pool, n_test, replace=False))
    Ytr, tr_days = day_arrays(sp, ids, tr_days)
    Yte, te_days = day_arrays(sp, ids, te_days)
    hidden = np.zeros(len(ids), bool)
    hidden[np.random.default_rng(0).choice(len(ids), int(round(0.3 * len(ids))), replace=False)] = True
    vis = torch.tensor(~hidden)

    p0 = ctm.default_params(net)
    p0["log_a"] = torch.where(net.ramp_major, torch.tensor(float(np.log(4500.0))), p0["log_a"])
    # data-driven bottleneck initialisation: cells where the AM-peak mean speed recovers just downstream
    peak = Ytr[:, 24:54, :].mean((0, 1))                     # 07:00-09:30, per detector
    cell_speed = np.full(net.M, np.nan)
    for c in range(net.M):
        m = (cell_of.numpy() == c)
        if m.any():
            cell_speed[c] = peak[m].mean()
    for idx in (net.a_idx.numpy(), net.b_idx.numpy()):
        v = pd.Series(cell_speed[idx]).interpolate(limit_direction="both").values
        score = np.where(v[:-1] < 75, v[1:] - v[:-1], -np.inf)
        for j in np.argsort(-score)[:3]:
            if np.isfinite(score[j]) and score[j] > 5:
                p0["cap_logit"][int(idx[j])] = -0.45
    print("bottleneck init cells:", torch.where(p0["cap_logit"] < 0)[0].tolist(), flush=True)
    shared = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    day_tr = new_day_params(len(tr_days))
    log_tr = fit(shared, day_tr, net, Ytr, vis, cell_of, iters, 0.08, 25.0, True, tag="train")
    shared = {k: v.detach() for k, v in shared.items()}
    torch.save(shared, os.path.join(ROOT, "results", "wm_params_daily2.pt"))

    day_te = new_day_params(len(te_days))
    log_te = fit(shared, day_te, net, Yte, vis, cell_of, test_iters, 0.08, 10.0, False, tag="test")
    with torch.no_grad():
        Vtr = simulate(shared, {k: v.detach() for k, v in day_tr.items()}, net, 0.0)[:, :, cell_of].numpy()
        Vte = simulate(shared, {k: v.detach() for k, v in day_te.items()}, net, 0.0)[:, :, cell_of].numpy()
    hist_mean = np.mean(day_arrays(sp, ids, tr_pool)[0], 0)          # (72, S) from Jan-Apr
    interp_te = np.stack([interp_baseline(Yte[i], sens, hidden) for i in range(len(te_days))])
    h = hidden
    per_day = {
        "wm_hidden": np.abs(Vte - Yte)[:, :, h].mean((1, 2)),
        "interp_hidden": np.abs(interp_te - Yte)[:, :, h].mean((1, 2)),
        "histmean_hidden": np.abs(hist_mean[None] - Yte)[:, :, h].mean((1, 2)),
    }
    res = {
        "n_train_days": len(tr_days), "n_test_days": len(te_days), "n_hidden": int(h.sum()), "n_detectors": int(len(h)),
        "wm_mae_visible_train": float(np.abs(Vtr - Ytr)[:, :, ~h].mean()),
        "wm_mae_hidden_train": float(np.abs(Vtr - Ytr)[:, :, h].mean()),
        "wm_mae_visible_test": float(np.abs(Vte - Yte)[:, :, ~h].mean()),
        "wm_mae_hidden_test": float(per_day["wm_hidden"].mean()),
        "interp_mae_hidden_test": float(per_day["interp_hidden"].mean()),
        "histmean_mae_hidden_test": float(per_day["histmean_hidden"].mean()),
        "days_wm_beats_interp": int((per_day["wm_hidden"] < per_day["interp_hidden"]).sum()),
        "vf": shared["vf"].tolist(), "b0": float(shared["b0"]), "b1": float(shared["b1"]),
        "cap_mult_range": [float(ctm.cap_mult(shared).min()), float(ctm.cap_mult(shared).max())],
    }
    print(json.dumps(res, indent=2))
    json.dump({"result": res, "log_train": log_tr, "log_test": log_te,
               "per_day": {k: v.tolist() for k, v in per_day.items()},
               "test_days": [str(d.date()) for d in te_days]},
              open(os.path.join(ROOT, "results", "calibration_daily2.json"), "w"), indent=2)
    np.savez(os.path.join(ROOT, "results", "calibration_daily2_arrays.npz"), Vte=Vte, Yte=Yte, interp_te=interp_te,
             hidden=hidden, route=sens.route.values, x=sens.x_km.values)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
