"""Vehicle agency levels, the 'true world', infrastructure agents, and the closed-loop runner.

Vehicle levels for agentic through-travellers (O_A); human through-travellers (O_H) always follow the
calibrated logit P(B) = sig(b0 + b1 (TT_A - TT_B)).
  L0  non-agentic: behaves like a human driver.
  L1  context-aware, self-interested: P(B) = sig(c1 (TT_A - TT_B)), no response to the operator.
  L2  obedient: follows the operator's requested share of route B, theta in [0, 1].
  L3  negotiating: self-interested choice, plus accepts a paid offer sigma ($ per vehicle) to take the
      route the operator asks for when VOT * extra_time <= |sigma| (VOT lognormal per passenger).
  L4  L3 plus a multimodal offer: park at San Jose Diridon and ride Caltrain to Mountain View, accepted
      when it saves time or when the paid offer covers VOT * extra time (incl. a transfer penalty).
Autonomy is separate: kappa raises cell capacity in proportion to the agentic share.
"""
from __future__ import annotations
import math, copy
import numpy as np
import torch
import ctm

C1 = 1.0               # selfish sensitivity, 1/min
VOT_TRUE = (25.0, 0.6)  # $/h mean, log-sd of passenger value of time
VOT_BELIEF = (18.0, 0.6)
PI_MAX = 8.0           # $ per vehicle
SPILL_Q = 60.0         # ramp storage before spillback, veh
T_RAIL_MIN = 32.0      # Diridon -> Mountain View incl. access, average wait and egress
TRANSFER_PEN_MIN = 8.0 # perceived inconvenience of a mode change
PR_CELL_KM = 9.66      # park-and-ride access near SR-87 PM 6.0 (San Jose Diridon), on route B
PI_T_MAX = 12.0        # $ per park-and-ride offer
N_LEVER = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}


def _lognorm_cdf(x, mean, s):
    mu = math.log(mean) - 0.5 * s * s
    return 0.5 * (1 + torch.erf((torch.log(x.clamp_min(1e-6)) - mu) / (s * math.sqrt(2))))


def route_probs(level, lever, ttA, ttB, b0, b1, vot):
    """Returns P(route B) for (O_H, O_A), shape (B, 2), and the induced share of O_A (paid diversions)."""
    pH = torch.sigmoid(b0 + b1 * (ttA - ttB))
    induced = torch.zeros_like(pH)
    if level == 0:
        pA = pH
    elif level == 1:
        pA = torch.sigmoid(C1 * (ttA - ttB))
    elif level == 2:
        pA = lever.clamp(0.0, 1.0)
    else:
        sB = torch.sigmoid(C1 * (ttA - ttB))
        pos, neg = lever.clamp_min(0.0), (-lever).clamp_min(0.0)
        extra_B = (ttB - ttA).clamp_min(0.1)
        extra_A = (ttA - ttB).clamp_min(0.1)
        F_B = _lognorm_cdf(60.0 * pos / extra_B, *vot) * (pos > 0)
        F_A = _lognorm_cdf(60.0 * neg / extra_A, *vot) * (neg > 0)
        to_B = (1 - sB) * F_B
        to_A = sB * F_A
        pA = sB + to_B - to_A
        induced = to_B + to_A
    return torch.stack([pH, pA], 1), induced


def pr_cell(net):
    Lb = float(net.L[net.b_idx[0]])
    return int(net.b_idx[min(len(net.b_idx) - 1, int(PR_CELL_KM // Lb))])


def transit_share(lever_t, v, net, cell, vot):
    """Share of agentic through vehicles leaving at the park-and-ride cell, and the paid (induced) part."""
    rem = net.b_idx[net.b_idx > cell]
    tt_rem = (net.L[rem] / v[:, rem].clamp_min(3.0)).sum(1) * 60.0
    sel = torch.sigmoid(C1 * (tt_rem - T_RAIL_MIN - TRANSFER_PEN_MIN))
    extra = (T_RAIL_MIN + TRANSFER_PEN_MIN - tt_rem).clamp_min(0.1)
    pos = lever_t.clamp_min(0.0)
    F = _lognorm_cdf(60.0 * pos / extra, *vot) * (pos > 0)
    return sel + (1 - sel) * F, (1 - sel) * F


# --------------------------------------------------------------------------------------------
class World:
    """Truth (or world-model) simulator around ctm.step with metrics accounting."""

    def __init__(self, net, p, level, p_agentic, kappa, truth=False, day_seed=0, incident=False, truth_model="ctm"):
        self.net, self.level, self.pa = net, level, p_agentic
        self.multimodal = level == 4
        self.route_level = 3 if level == 4 else level
        self.pr = pr_cell(net)
        self.p = dict(p); self.p["kappa"] = torch.tensor(float(kappa))
        self.truth = truth
        self.vot = VOT_TRUE if truth else VOT_BELIEF
        self.truth_model = truth_model
        if truth_model == "metanet":
            import metanet
            self.sim, self.state = metanet.step, metanet.init_state(net, 1, self.p)
        else:
            self.sim, self.state = ctm.step, ctm.init_state(net, 1)
        self.t = 0
        self.m = {"tts": 0.0, "spend": 0.0, "vkt_A": 0.0, "served_O": 0.0, "time_O": 0.0, "transit": 0.0,
                  "ramp_wait": np.zeros(net.R), "ramp_served": np.zeros(net.R), "spill": 0.0, "max_q": 0.0}
        rng = np.random.default_rng(day_seed)
        nb = ctm.N_BLOCKS
        def ar1(n, sd=0.07, rho=0.8):
            e = np.zeros(n); e[0] = rng.normal(0, sd)
            for i in range(1, n):
                e[i] = rho * e[i - 1] + rng.normal(0, sd * math.sqrt(1 - rho ** 2))
            return e
        self.eps0 = torch.tensor(ar1(nb), dtype=torch.float32) if truth else torch.zeros(nb)
        self.epsg = torch.tensor(np.stack([ar1(nb), ar1(nb)]), dtype=torch.float32) if truth else torch.zeros(2, nb)
        self.b0 = p["b0"] + (float(rng.normal(0, 0.2)) if truth else 0.0)
        self.incident = None
        if truth and incident:
            route_cells = net.a_idx if rng.random() < 0.5 else net.b_idx
            cell = int(route_cells[rng.integers(5, len(route_cells) - 3)])
            start = int((7 * 60 + 15 - ctm.T0_MIN) * 2 + rng.integers(0, 120))
            self.incident = (cell, start, start + 60)   # 30 min, steps of 30 s

    def demand(self, t):
        blk = min(ctm.N_BLOCKS - 1, t // (2 * ctm.BLOCK_MIN))
        d0 = torch.exp(self.p["log_d0"][blk] + self.eps0[blk]).reshape(1)
        dr = torch.exp(ctm.ramp_log_demand(self.p, self.net, blk) + self.epsg[self.net.ramp_route - 1, blk]).reshape(1, -1)
        return d0, dr

    def cap_factor(self):
        if not self.truth:
            return None
        net = self.net
        N = self.state["n"].sum(1)
        kc = ctm.Q_LANE / 105.0 * net.lanes * net.L
        if self.truth_model == "metanet":     # capacity drop emerges from the speed dynamics
            f = torch.ones_like(N)
        else:
            f = 1.0 - 0.10 * torch.sigmoid((N - kc) / (0.1 * kc + 1e-3))
        if self.incident and self.incident[1] <= self.t < self.incident[2]:
            f = f.clone(); f[0, self.incident[0]] *= 0.5
        return f

    def step(self, meters, lever):
        net = self.net
        lev_all = torch.as_tensor(lever, dtype=torch.float32).reshape(-1)
        lev = lev_all[:1]
        lev_t = lev_all[1:2] if self.multimodal else None
        holder = {}
        def rp(a, b):
            probs, ind = route_probs(self.route_level, lev, a, b, self.b0, self.p["b1"], self.vot)
            holder["ind"] = ind
            return probs
        def tr(v):
            share, ind = transit_share(lev_t, v, net, self.pr, self.vot)
            holder["t_share"], holder["t_ind"] = share, ind
            return self.pr, share
        st, info = self.sim(self.state, self.p, net, self.t, meters.reshape(1, -1), rp,
                            cap_factor=self.cap_factor(), demand=self.demand(self.t), p_agentic=self.pa,
                            transit=tr if self.multimodal else None)
        if self.t >= 60:   # account from 05:00
            stock = st["n"].sum().item() + st["qo"].sum().item() + st["qr"].sum().item()
            m = self.m
            m["tts"] += stock * ctm.DT_H
            if self.route_level == 3:
                m["spend"] += (info["o_A"] * holder["ind"]).sum().item() * abs(float(lev))
            if self.multimodal:
                riders = info["transit_exit"].sum().item()
                induced = riders * float(holder["t_ind"] / holder["t_share"].clamp_min(1e-6))
                m["spend"] += induced * float(lev_t)
                m["transit"] += riders
                m["tts"] += riders * T_RAIL_MIN / 60.0
                m["time_O"] += riders * T_RAIL_MIN / 60.0
            prev = self.state["n"][0]
            m["vkt_A"] += (info["out_tot"][0] * (prev[1] + prev[3]) / prev.sum(0).clamp_min(1e-6) * net.L).sum().item()
            m["time_O"] += (st["n"][0, 0:2].sum() + st["qo"].sum()).item() * ctm.DT_H
            m["served_O"] += info["exit"][0, 0:2].sum().item()
            q = st["qr"][0].sum(0).numpy() * (~net.ramp_major).numpy()   # regular on-ramps only
            m["ramp_wait"] += q * ctm.DT_H
            arr = self.demand(self.t)[1][0].numpy() * ctm.DT_H
            q_all = st["qr"][0].sum(0).numpy()
            m["ramp_served"] += self.state["qr"][0].sum(0).numpy() + arr - q_all
            m["spill"] += float(np.clip(q - SPILL_Q, 0, None).sum()) * ctm.DT_H
            m["max_q"] = max(m["max_q"], float(q.max()))
        self.state, self.t = st, self.t + 1
        return info

    def speed_cells(self, info):
        """Speeds a detector in each cell would read after the last step, (1, M)."""
        if self.truth_model == "metanet":
            return self.state["v"]
        return ctm.fd_speed(self.state["n"].sum(1), info["cap"], ctm.vf_cells(self.p, self.net), self.net)

    def stock(self):
        st = self.state
        return st["n"].sum().item() + st["qo"].sum().item() + st["qr"].sum().item()


# --------------------------------------------------------------------------------------------
def alinea(world, meters, ctrl_idx, gain=60.0):
    net = world.net
    cells = net.ramp_cell[ctrl_idx]
    k_lane = world.state["n"][0].sum(0)[cells] / (net.L[cells] * net.lanes[cells])
    kc = ctm.Q_LANE / 105.0
    new = meters.clone()
    new[ctrl_idx] = (meters[ctrl_idx] + gain * (kc - k_lane)).clamp(300.0, 1800.0)
    # queue override: release the ramp when its queue approaches storage
    q = world.state["qr"][0].sum(0)[ctrl_idx]
    new[ctrl_idx] = torch.where(q > 0.8 * SPILL_Q, torch.tensor(1800.0), new[ctrl_idx])
    return new


def mpc_plan(world, wm_params, ctrl_idx, level, p_agentic, kappa, horizon_steps=60, block_steps=20,
             pop=48, iters=3, elite=8, seed=0, init=None):
    """CEM over metering rates of controlled ramps and the routing lever, using the world model."""
    net = world.net
    nb = horizon_steps // block_steps
    nm = len(ctrl_idx)
    nlev = N_LEVER[level]
    dim = nb * (nm + nlev)
    g = torch.Generator().manual_seed(seed)
    lev_lo = {0: [], 1: [], 2: [0.0], 3: [-PI_MAX], 4: [-PI_MAX, 0.0]}[level]
    lev_hi = {0: [], 1: [], 2: [1.0], 3: [PI_MAX], 4: [PI_MAX, PI_T_MAX]}[level]
    lo = torch.tensor([300.0] * nm + lev_lo)
    hi = torch.tensor([1800.0] * nm + lev_hi)
    lo, hi = lo.repeat(nb), hi.repeat(nb)
    mean = init.repeat(nb) if init is not None else (lo + hi) / 2
    std = (hi - lo) / 3
    p = dict(wm_params); p["kappa"] = torch.tensor(float(kappa))
    best = None
    for it in range(iters):
        X = (mean + std * torch.randn(pop, dim, generator=g)).clamp(lo, hi)
        X[0] = mean.clamp(lo, hi)
        cost = rollout_cost(world, p, X, nb, nm, block_steps, ctrl_idx, level, p_agentic)
        el = X[cost.argsort()[:elite]]
        mean, std = el.mean(0), el.std(0) + 1e-3
        best = X[cost.argmin()]
    per = nm + nlev
    first = best[:per]
    meters = net.ramp_cap.clone()
    meters[ctrl_idx] = first[:nm]
    lever = 0.0 if nlev == 0 else (float(first[nm]) if nlev == 1 else first[nm:nm + 2].tolist())
    return meters, lever, best[:per]


@torch.no_grad()
def rollout_cost(world, p, X, nb, nm, block_steps, ctrl_idx, level, p_agentic):
    net = world.net
    P = X.shape[0]
    st = {k: v.expand(P, *v.shape[1:]).clone() for k, v in world.state.items()}
    cost = torch.zeros(P)
    per = X.shape[1] // nb
    for b in range(nb):
        u = X[:, b * per:(b + 1) * per]
        meters = net.ramp_cap.expand(P, -1).clone()
        meters[:, ctrl_idx] = u[:, :nm]
        lever = u[:, nm] if per > nm else torch.zeros(P)
        lever_t = u[:, nm + 1] if per > nm + 1 else None
        route_level = 3 if level == 4 else level
        for s in range(block_steps):
            t = world.t + b * block_steps + s
            holder = {}
            def rp(a, bb):
                probs, ind = route_probs(route_level, lever, a, bb, p["b0"], p["b1"], VOT_BELIEF)
                holder["ind"] = ind
                return probs
            def tr(v):
                share, ind = transit_share(lever_t, v, net, world.pr, VOT_BELIEF)
                holder["t_share"], holder["t_ind"] = share, ind
                return world.pr, share
            d0, dr = ctm.demand_at(p, net, min(t, ctm.N_STEPS - 1))
            st, info = ctm.step(st, p, net, t, meters, rp, demand=(d0.expand(P), dr.expand(P, -1)), p_agentic=p_agentic,
                                transit=tr if level == 4 else None)
            stock = st["n"].sum((1, 2)) + st["qo"].sum(1) + st["qr"].sum((1, 2))
            q = st["qr"].sum(1)
            cost += stock * ctm.DT_H + 10.0 * ((q - SPILL_Q).clamp_min(0) * (~net.ramp_major)).sum(1) * ctm.DT_H
            if route_level == 3:
                cost += info["o_A"] * holder["ind"] * lever.abs() / 25.0
            if level == 4:
                riders = info["transit_exit"]
                cost += riders * T_RAIL_MIN / 60.0
                cost += riders * holder["t_ind"] / holder["t_share"].clamp_min(1e-6) * lever_t / 25.0
    cost += (st["n"].sum((1, 2)) + st["qo"].sum(1) + st["qr"].sum((1, 2))) * (10.0 / 60.0)
    return cost


def run_day(net, wm_params, truth_params, agent, level, p_agentic, kappa, day_seed, incident, ctrl_idx,
            decide_every=10, truth_model="ctm"):
    torch.set_num_threads(1)
    world = World(net, truth_params, level, p_agentic, kappa, truth=True, day_seed=day_seed, incident=incident,
                  truth_model=truth_model)
    meters = net.ramp_cap.clone()
    if agent in ("A1", "A2"):
        meters[ctrl_idx] = 1200.0
    lever, init, n_dec, t_dec = ([0.0, 0.0] if level == 4 else 0.0), None, 0, 0.0
    import time
    for t in range(ctm.N_STEPS):
        if agent == "A1" and t % 2 == 0:
            meters = alinea(world, meters, ctrl_idx)
        if agent == "A2" and t >= 60 and (t - 60) % decide_every == 0:
            t0 = time.time()
            meters, lever, init = mpc_plan(world, wm_params, ctrl_idx, level, p_agentic, kappa, seed=day_seed * 1000 + t, init=init)
            t_dec += time.time() - t0; n_dec += 1
        world.step(meters, lever)
    m = world.m
    tts_total = m["tts"] + world.stock() * (10.0 / 60.0)
    ramp_avg_wait_min = m["ramp_wait"][ctrl_idx] / np.maximum(m["ramp_served"][ctrl_idx], 1.0) * 60.0
    return {"agent": agent, "level": level, "p": p_agentic, "kappa": kappa, "day": day_seed, "incident": incident,
            "tts_vh": tts_total, "through_avg_min": m["time_O"] / max(m["served_O"], 1.0) * 60.0,
            "worst_ramp_wait_min": float(ramp_avg_wait_min.max()), "max_ramp_queue": m["max_q"],
            "spill_vh": m["spill"], "spend_usd": m["spend"], "vkt_agentic": m["vkt_A"], "transit_riders": m["transit"],
            "decision_sec": t_dec / max(n_dec, 1)}
