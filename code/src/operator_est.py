"""Closed loop with a detector-only operator: moving-horizon state estimation feeding the MPC.

The operator never sees the true state. It keeps an internal copy of the world model that it advances every
30 s with the controls it actually applied (meters, routing lever) and mean demand. Every 15 min it re-estimates
the state from the last 30 min of noisy 5-min detector speeds (all route detectors, noise sd 3 km/h): starting
from its own internal state at t - 30 min, it fits per-cell density multipliers, a queue scale and three demand
scales by gradient descent through the CTM while replaying its applied controls, then resets its internal state
to the fitted state at t. The MPC (agentic.mpc_plan, unchanged) plans from this estimated state.
"""
from __future__ import annotations
import sys, os, time, types
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import ctm, agentic
from agentic import route_probs, transit_share, VOT_BELIEF, World
from estimate import demand as scaled_demand

NOISE_SD = 3.0
W_STEPS = 60          # 30 min window
EST_EVERY = 30        # re-estimate every 15 min


class Estimator:
    def __init__(self, net, p, level, p_agentic, kappa, cell_of, iters=30, lr=0.08, seed=0):
        self.net, self.level, self.pa = net, level, p_agentic
        self.route_level = 3 if level == 4 else level
        self.p = dict(p); self.p["kappa"] = torch.tensor(float(kappa))
        self.cell_of, self.iters, self.lr = cell_of, iters, lr
        self.pr = agentic.pr_cell(net)
        self.state = ctm.init_state(net, 1)
        self.snap, self.ctrl, self.obs = {}, {}, {}
        self.rng = np.random.default_rng(seed)
        self.acc, self.n_est, self.sec = 0.0, 0, 0.0

    def _step(self, st, t, meters, lever, s=None):
        net, p = self.net, self.p
        lev_all = torch.as_tensor(lever, dtype=torch.float32).reshape(-1)
        lev, lev_t = lev_all[:1], (lev_all[1:2] if self.level == 4 else None)
        rp = lambda a, b: route_probs(self.route_level, lev, a, b, p["b0"], p["b1"], VOT_BELIEF)[0]
        tr = (lambda v: (self.pr, transit_share(lev_t, v, net, self.pr, VOT_BELIEF)[0])) if self.level == 4 else None
        z = torch.zeros(1)
        dem = scaled_demand(p, net, t, s if s is not None else (z, z, z))
        return ctm.step(st, p, net, t, meters.reshape(1, -1), rp, demand=dem, p_agentic=self.pa, transit=tr)

    def observe(self, t, true_speed_cells):
        """Called after each true-world step with the true cell speeds (1, M); stores noisy 5-min detector means."""
        if t < 60:
            return
        self.acc = self.acc + true_speed_cells[0, self.cell_of]
        if (t - 60) % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
            v = self.acc / ctm.STEPS_PER_5MIN
            self.obs[t + 1] = v + torch.tensor(self.rng.normal(0, NOISE_SD, v.shape), dtype=torch.float32)
            self.acc = 0.0

    @torch.no_grad()
    def predict(self, t, meters, lever):
        self.snap[t] = {k: v.clone() for k, v in self.state.items()}
        self.ctrl[t] = (meters.clone(), lever)
        self.state, _ = self._step(self.state, t, meters, lever)

    def update(self, t):
        """Re-estimate the state at step t (before applying the step-t control)."""
        t0 = t - W_STEPS
        ends = [e for e in range(t0 + ctm.STEPS_PER_5MIN, t + 1, ctm.STEPS_PER_5MIN) if e in self.obs]
        if t0 < 60 or len(ends) < W_STEPS // ctm.STEPS_PER_5MIN or t0 not in self.snap:
            return
        c0 = time.time()
        prior = self.snap[t0]
        Y = torch.stack([self.obs[e] for e in ends])                       # (6, n_sens)
        x = {"logm": torch.zeros(self.net.M), "logq": torch.zeros(1), "s0": torch.zeros(1), "sA": torch.zeros(1),
             "sB": torch.zeros(1)}
        x = {k: v.requires_grad_(True) for k, v in x.items()}
        opt = torch.optim.Adam(x.values(), lr=self.lr)
        vf = ctm.vf_cells(self.p, self.net)

        def run(grad_smooth):
            st = {"n": prior["n"] * torch.exp(x["logm"])[None, None, :], "qo": prior["qo"] * torch.exp(x["logq"]),
                  "qr": prior["qr"] * torch.exp(x["logq"])}
            Vs, acc = [], 0.0
            for k in range(W_STEPS):
                m, lv = self.ctrl[t0 + k]
                st, info = self._step(st, t0 + k, m, lv, (x["s0"], x["sA"], x["sB"]))
                acc = acc + ctm.fd_speed(st["n"].sum(1), info["cap"], vf, self.net, smooth=grad_smooth)[0, self.cell_of]
                if k % ctm.STEPS_PER_5MIN == ctm.STEPS_PER_5MIN - 1:
                    Vs.append(acc / ctm.STEPS_PER_5MIN); acc = 0.0
            return st, torch.stack(Vs)

        for _ in range(self.iters):
            opt.zero_grad()
            _, V = run(2.0)
            loss = (((V - Y) / 10.0) ** 2).mean() + 0.05 * (x["logm"] ** 2).mean() + 0.5 * (x["logm"].diff() ** 2).mean() \
                + 0.5 * sum((x[k] ** 2).mean() for k in ("logq", "s0", "sA", "sB"))
            loss.backward()
            opt.step()
            with torch.no_grad():
                x["logm"].clamp_(-2.0, 2.0)
                for k in ("logq", "s0", "sA", "sB"):
                    x[k].clamp_(-1.0, 1.0)
        with torch.no_grad():
            st, _ = run(0.0)
        self.state = {k: v.detach() for k, v in st.items()}
        self.n_est += 1; self.sec += time.time() - c0


def run_day_est(net, wm_params, truth_params, level, p_agentic, kappa, day_seed, incident, ctrl_idx,
                decide_every=10, truth_model="ctm"):
    """Like agentic.run_day with agent A2, but the MPC plans from the estimator's state."""
    torch.set_num_threads(1)
    world = World(net, truth_params, level, p_agentic, kappa, truth=True, day_seed=day_seed, incident=incident,
                  truth_model=truth_model)
    cell_of = torch.tensor(sorted(set(int(c) for c in net.sensor_cell.values())))
    est = Estimator(net, wm_params, level, p_agentic, kappa, cell_of, seed=day_seed)
    meters = net.ramp_cap.clone(); meters[ctrl_idx] = 1200.0
    lever, init, n_dec, t_dec = ([0.0, 0.0] if level == 4 else 0.0), None, 0, 0.0
    err = []
    for t in range(ctm.N_STEPS):
        if t >= 60 and (t - 60) % EST_EVERY == 0:
            est.update(t)
            Nt, Ne = world.state["n"].sum(1)[0], est.state["n"].sum(1)[0]
            err.append(float(((Ne - Nt).abs() / net.L).mean()))          # veh/km, mean over cells
        if t >= 60 and (t - 60) % decide_every == 0:
            c0 = time.time()
            shim = types.SimpleNamespace(net=net, state=est.state, t=t, pr=world.pr)
            meters, lever, init = agentic.mpc_plan(shim, wm_params, ctrl_idx, level, p_agentic, kappa,
                                                   seed=day_seed * 1000 + t, init=init)
            t_dec += time.time() - c0; n_dec += 1
        est.predict(t, meters, lever)
        info = world.step(meters, lever)
        est.observe(t, world.speed_cells(info))
    m = world.m
    tts_total = m["tts"] + world.stock() * (10.0 / 60.0)
    ramp_avg_wait_min = m["ramp_wait"][ctrl_idx] / np.maximum(m["ramp_served"][ctrl_idx], 1.0) * 60.0
    return {"agent": "A2est", "truth": truth_model, "level": level, "p": p_agentic, "kappa": kappa, "day": day_seed, "incident": incident,
            "tts_vh": tts_total, "through_avg_min": m["time_O"] / max(m["served_O"], 1.0) * 60.0,
            "worst_ramp_wait_min": float(ramp_avg_wait_min.max()), "max_ramp_queue": m["max_q"],
            "spill_vh": m["spill"], "spend_usd": m["spend"], "vkt_agentic": m["vkt_A"], "transit_riders": m["transit"],
            "decision_sec": t_dec / max(n_dec, 1), "est_sec": est.sec / max(est.n_est, 1),
            "state_err_vehkm": float(np.mean(err)) if err else float("nan")}
