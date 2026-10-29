"""Two-class (human-driven H / agentic A) cell transmission model on the SR-85 vs SR-87+US-101 pair.

Commodities: 0 = O_H, 1 = O_A (through travellers from the origin who choose a route),
             2 = L_H, 3 = L_A (local traffic entering at on-ramps and leaving at off-ramps or the end),
             4 = F_occ, 5 = F_emp (through vehicles of a ride-hailing / robotaxi fleet, occupied and empty).
Fleet empties can be held at the origin (state "qh", a depot queue off the road) and released with mean delay T_HOLD.
Cell 0 is the origin cell upstream of the diverge; route A cells follow, then route B cells.
Everything is batched over a leading dimension B (scenarios or CEM candidates) and differentiable.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

K = 6
IS_A = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])      # automated vehicles (capacity effect kappa)
IS_LOCAL = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
THROUGH = torch.tensor([0, 1, 4, 5])                      # classes that enter at the origin and choose a route
T_HOLD_MIN = 10.0
VF_MAX = 130.0     # km/h, bound used to choose substeps for short cells
Q_LANE = 2000.0  # veh/h/lane
KJ_LANE = 130.0  # veh/km/lane
W_KMH = 18.0     # backward wave speed
DT_H = 30.0 / 3600.0
T0_MIN = 4 * 60 + 30   # simulation starts 04:30
T1_MIN = 11 * 60       # ends 11:00
N_STEPS = int((T1_MIN - T0_MIN) * 60 / 30)
STEPS_PER_5MIN = 10
BLOCK_MIN = 15
N_BLOCKS = (T1_MIN - T0_MIN) // BLOCK_MIN


@dataclass
class Net:
    M: int
    L: torch.Tensor        # (M,) km
    lanes: torch.Tensor    # (M,)
    route: torch.Tensor    # (M,) 0 origin, 1 A, 2 B
    a_idx: torch.Tensor    # route A cells in order
    b_idx: torch.Tensor
    merge_cells: torch.Tensor  # cells j fed by j-1 (all route cells except the first of each route)
    last_cells: torch.Tensor   # [last of A, last of B]
    cell_ramp: torch.Tensor    # (M,) ramp index or -1
    ramp_cell: torch.Tensor    # (R,)
    ramp_route: torch.Tensor   # (R,) 1 or 2
    ramp_major: torch.Tensor   # (R,) bool, freeway-to-freeway connector
    ramp_cap: torch.Tensor     # (R,) veh/h
    exit_mask: torch.Tensor    # (M,) 1 where an off-ramp exists
    sensor_cell: dict
    sensor_route: dict
    sub: int = 1              # internal substeps per 30 s step (CFL); 1 for the default 1 km cells

    @property
    def R(self):
        return int(self.ramp_cell.numel())


def build_net(sens_A: pd.DataFrame, sens_B: pd.DataFrame, len_A: float, len_B: float,
              major_A_km: float, major_B_km: float, cell_km: float = 1.0, ramp_x=None, exit_x=None, sub=None) -> Net:
    """ramp_x / exit_x: optional {1: [x_km...], 2: [...]} on-ramp and off-ramp positions per route (e.g. from PeMS
    OR/FR station metadata). Default: on-ramps and exits at cells with a 'rm-n'-named mainline detector."""
    cells_L, cells_lanes, cells_route, sensor_cell, sensor_route = [1.0], [3.0], [0], {}, {}
    ramp_cell, ramp_route, ramp_major = [], [], []
    a_idx, b_idx = [], []
    extra_exit = []
    for r_id, (sens, length, major_km, idx) in enumerate([(sens_A, len_A, major_A_km, a_idx), (sens_B, len_B, major_B_km, b_idx)], start=1):
        n = max(1, round(length / cell_km))
        Lc = length / n
        start = len(cells_L)
        for c in range(n):
            lo, hi = c * Lc, (c + 1) * Lc
            inside = sens[(sens.x_km >= lo) & (sens.x_km < hi)]
            if len(inside):
                lanes = float(np.median(inside.Lanes))
            else:
                lanes = float(sens.iloc[(sens.x_km - (lo + hi) / 2).abs().argmin()].Lanes)
            cells_L.append(Lc); cells_lanes.append(lanes); cells_route.append(r_id)
            idx.append(start + c)
        for _, s in sens.iterrows():
            cid = start + min(n - 1, int(s.x_km // Lc))
            sensor_cell[int(s.sensor_id)] = cid
            sensor_route[int(s.sensor_id)] = r_id
        # on/off ramps at cells with a ramp-named detector (skip the first cell of each route)
        xs = sens.loc[sens.is_ramp, "x_km"] if ramp_x is None else ramp_x[r_id]
        rc = sorted({start + min(n - 1, int(x // Lc)) for x in xs} - {start})
        if exit_x is not None:
            extra_exit += [start + min(n - 1, int(x // Lc)) for x in exit_x[r_id]]
        major = start + min(n - 1, int(major_km // Lc))
        for c in rc:
            if c == major:
                continue
            ramp_cell.append(c); ramp_route.append(r_id); ramp_major.append(False)
        ramp_cell.append(major); ramp_route.append(r_id); ramp_major.append(True)
    M = len(cells_L)
    route = torch.tensor(cells_route)
    cell_ramp = torch.full((M,), -1, dtype=torch.long)
    for r, c in enumerate(ramp_cell):
        cell_ramp[c] = r
    exit_mask = torch.zeros(M)
    exit_mask[torch.tensor(ramp_cell)] = 1.0
    if extra_exit:
        exit_mask[torch.tensor(extra_exit)] = 1.0
    a_idx, b_idx = torch.tensor(a_idx), torch.tensor(b_idx)
    merge = torch.cat([a_idx[1:], b_idx[1:]])
    ramp_major_t = torch.tensor(ramp_major)
    ramp_cap = torch.where(ramp_major_t, torch.tensor(7000.0), torch.tensor(1800.0))
    return Net(M=M, L=torch.tensor(cells_L), lanes=torch.tensor(cells_lanes), route=route,
               a_idx=a_idx, b_idx=b_idx, merge_cells=merge, last_cells=torch.stack([a_idx[-1], b_idx[-1]]),
               cell_ramp=cell_ramp, ramp_cell=torch.tensor(ramp_cell), ramp_route=torch.tensor(ramp_route),
               ramp_major=ramp_major_t, ramp_cap=ramp_cap, exit_mask=exit_mask,
               sensor_cell=sensor_cell, sensor_route=sensor_route,
               sub=sub if sub is not None else (1 if cell_km >= 1.0 else int(math.ceil(VF_MAX * DT_H / min(cells_L)))))


def default_params(net: Net) -> dict:
    """Learnable parameters (unconstrained) of the world model, in a dict of tensors."""
    R = net.R
    log_a = torch.where(net.ramp_major, torch.tensor(math.log(2500.0)), torch.tensor(math.log(450.0)))
    return {
        "vf": torch.tensor([105.0, 105.0]),                       # free-flow speed per route
        "cap_logit": torch.zeros(net.M),                          # capacity multiplier in [0.6, 1.4]
        "vf_logit": torch.zeros(net.M),                           # per-cell free-flow multiplier in [0.8, 1.2]
        "log_d0": torch.full((N_BLOCKS,), math.log(3800.0)),      # origin demand profile, veh/h
        "log_a": log_a.clone(),                                   # ramp demand scale
        "log_g": torch.zeros(2, N_BLOCKS),                        # route-level ramp demand shape
        "exit_logit": torch.full((net.M,), math.log(0.10 / 0.90)),  # off-ramp exit fraction (local traffic)
        "b0": torch.tensor(0.0), "b1": torch.tensor(0.10),       # human route choice: P(B)=sig(b0+b1*(TT_A-TT_B))
    }


def vf_cells(p, net):
    """Per-cell free-flow speed: route value times a learnable per-cell multiplier in [0.8, 1.2]."""
    base = torch.where(net.route == 2, p["vf"][1], p["vf"][0])
    if "vf_logit" in p:
        base = base * (1.0 + 0.2 * torch.tanh(p["vf_logit"]))
    return torch.minimum(base, net.L * net.sub / DT_H)       # CFL: vf * dt <= L


def cap_mult(p):
    return 1.0 + 0.4 * torch.tanh(p["cap_logit"])


def demand_at(p, net: Net, step: int):
    """Origin demand (scalar veh/h) and ramp demands (R,) at a simulation step."""
    blk = min(N_BLOCKS - 1, step * 30 // 60 // BLOCK_MIN)
    d0 = torch.exp(p["log_d0"][blk])
    dr = torch.exp(ramp_log_demand(p, net, blk))
    return d0, dr


def ramp_log_demand(p, net: Net, blk: int):
    """Log ramp demands (R,) in a 15-min block. v1-v3 params: per-ramp scale x route-level shape.
    v4 params: per-ramp profile `log_r`, with the route-B major ramp fixed to the measured US-101 inflow."""
    if "log_r" not in p:
        return p["log_a"] + p["log_g"][net.ramp_route - 1, blk]
    lr = p["log_r"][:, blk]
    if "bnd_major_B" in p:
        majB = int(torch.where(net.ramp_major & (net.ramp_route == 2))[0])
        lr = torch.cat([lr[:majB], torch.log(p["bnd_major_B"][blk]).reshape(1), lr[majB + 1:]])
    return lr


def init_state(net: Net, Bsz: int = 1):
    return {
        "n": torch.zeros(Bsz, K, net.M),
        "qo": torch.zeros(Bsz, 4),          # origin queue (O_H, O_A, F_occ, F_emp)
        "qr": torch.zeros(Bsz, 2, net.R),   # ramp queues (L_H, L_A)
        "qh": torch.zeros(Bsz),             # fleet empties held at the origin depot (off the road)
    }


def _mid(a, b, c):
    return torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))


def fd_speed(N, cap, vf, net: Net, smooth: float = 0.0):
    """Trapezoidal fundamental-diagram speed (km/h) of each cell. N: (B, M) vehicles."""
    k = N / net.L                      # veh/km
    kj = KJ_LANE * net.lanes
    kk = k.clamp_min(1e-3)
    c1 = cap / kk
    c2 = W_KMH * (kj - k).clamp_min(0.0) / kk
    if smooth > 0:
        x = torch.stack([vf.expand_as(c1), c1, c2], 0)
        return -smooth * torch.logsumexp(-x / smooth, 0)
    return torch.minimum(vf.expand_as(c1), torch.minimum(c1, c2))


def route_tt_min(v, net: Net):
    tt = (net.L / v.clamp_min(3.0)) * 60.0
    return tt[:, net.a_idx].sum(1), tt[:, net.b_idx].sum(1)


def _substep(state, p, net: Net, t: int, meters, route_prob_B, cap_factor, demand, p_agentic, transit, fleet, hold, dt):
    """One substep of length dt (hours). See step().

    meters: (B, R) metering rates in veh/h (use ramp_cap for unmetered ramps).
    route_prob_B: callable(TT_A, TT_B) -> (B, 2) probability of choosing route B for (O_H, O_A).
    cap_factor: optional (B, M) multiplier on capacity (truth: capacity drop, incidents).
    demand: optional (d0 (B,), dr (B,R)) overriding the parameter demand (truth: noisy days).
    transit: optional callable(speeds (B,M)) -> (cell, share (B,)): share of agentic through vehicles that
             leave the freeway at `cell` to park and ride transit (multimodal option).
    route_prob_B may return (B, 2) for (O_H, O_A), fleet classes then follow O_A, or (B, 4) for (O_H, O_A, F_occ, F_emp).
    fleet: (share of origin demand served by the fleet, empty share of fleet through vehicles).
    hold: optional (B,) share of newly arriving fleet empties sent to the depot queue instead of the road.
    Returns new state and a dict of flows for metrics.
    """
    n, qo, qr = state["n"], state["qo"], state["qr"]
    qh = state.get("qh", torch.zeros(n.shape[0]))
    Bsz = n.shape[0]
    vf_cell = vf_cells(p, net)
    N = n.sum(1)
    shareA = (n * IS_A[None, :, None]).sum(1) / N.clamp_min(1e-6)
    kappa = p.get("kappa", torch.tensor(0.0))
    cap = Q_LANE * net.lanes * cap_mult(p) * (1.0 + kappa * shareA)
    if cap_factor is not None:
        cap = cap * cap_factor
    kj_tot = KJ_LANE * net.lanes * net.L
    cap_send = cap
    if "drop_logit" in p:      # discharge from a congested cell is lower than capacity (capacity drop)
        delta = (0.2 * torch.sigmoid(p["drop_logit"]))[(net.route == 2).long()]
        kc = cap / vf_cell
        cap_send = cap * (1.0 - delta * torch.sigmoid((N / net.L - kc) / (0.1 * kc)))
    S = torch.minimum(vf_cell / net.L * N, cap_send) * dt
    Rcv = torch.minimum(cap, W_KMH / net.L * (kj_tot - N).clamp_min(0.0)) * dt
    frac = n / N.clamp_min(1e-6)[:, None, :]
    v_now = fd_speed(N, cap, vf_cell, net)

    exit_frac = torch.sigmoid(p["exit_logit"]) * net.exit_mask                # (M,)
    e_k = exit_frac[None, None, :] * IS_LOCAL[None, :, None]                  # (1,K,M)
    t_cell = None
    if transit is not None:
        t_cell, t_share = transit(v_now)
        e_k = e_k.expand(Bsz, K, net.M).clone()
        e_k[:, 1, t_cell] = t_share
    ebar = (frac * e_k).sum(1)                                                # (B,M)

    if demand is None:
        d0, dr = demand_at(p, net, t)
        d0 = d0.expand(Bsz); dr = dr.expand(Bsz, -1)
    else:
        d0, dr = demand
    # arrivals into queues
    s_f, e_f = fleet
    arr_emp = s_f * e_f * d0 * dt
    held = arr_emp * hold if hold is not None else torch.zeros_like(arr_emp)
    released = qh * (dt * 60.0 / T_HOLD_MIN)
    qh = qh + held - released
    qo = qo + torch.stack([(1 - p_agentic - s_f) * d0 * dt, p_agentic * d0 * dt,
                           s_f * (1 - e_f) * d0 * dt, arr_emp - held + released], 1)
    qr = qr + torch.stack([(1 - p_agentic) * dr, p_agentic * dr], 1) * dt

    # ramp sending
    qr_tot = qr.sum(1)                                                        # (B,R)
    Dr = torch.minimum(qr_tot, torch.minimum(meters, net.ramp_cap.expand_as(meters)) * dt)
    Dr_cell = torch.zeros(Bsz, net.M)
    has_r = net.cell_ramp >= 0
    Dr_cell[:, has_r] = Dr[:, net.cell_ramp[has_r]]

    out_tot = torch.zeros(Bsz, net.M)       # total outflow of each cell (incl. exits)
    inflow = torch.zeros(Bsz, K, net.M)
    fr_cell = torch.zeros(Bsz, net.M)

    # merges j-1 -> j with optional on-ramp into j
    j = net.merge_cells
    up = j - 1
    Dm = S[:, up] * (1.0 - ebar[:, up])
    Drj = Dr_cell[:, j]
    Rj = Rcv[:, j]
    free = (Dm + Drj) <= Rj
    alpha = 0.7
    fm = torch.where(free, Dm, _mid(Dm, Rj - Drj, alpha * Rj))
    fr = torch.where(free, Drj, _mid(Drj, Rj - Dm, (1 - alpha) * Rj))
    fm = fm.clamp_min(0.0); fr = fr.clamp_min(0.0)
    out_tot[:, up] = fm / (1.0 - ebar[:, up]).clamp_min(1e-3)
    fr_cell[:, j] = fr

    # last cells exit to the destination unconstrained
    out_tot[:, net.last_cells] = S[:, net.last_cells]

    # diverge at origin cell 0 into first cells of A and B
    a0, b0 = net.a_idx[0], net.b_idx[0]
    ttA, ttB = route_tt_min(v_now, net)
    pB = route_prob_B(ttA, ttB)                      # (B,2) for O_H, O_A, or (B,4) incl. F_occ, F_emp
    if pB.shape[1] == 2:
        pB = torch.cat([pB, pB[:, 1:2], pB[:, 1:2]], 1)
    pB_k = torch.zeros(Bsz, K)
    pB_k[:, THROUGH] = pB
    pbar = (frac[:, :, 0] * pB_k).sum(1)
    f0 = torch.minimum(S[:, 0], torch.minimum(Rcv[:, a0] / (1 - pbar).clamp_min(1e-3), Rcv[:, b0] / pbar.clamp_min(1e-3)))
    out_tot[:, 0] = f0

    # per-commodity flows
    out_k = out_tot[:, None, :] * frac                                       # (B,K,M)
    exit_k = out_k * e_k
    fwd_k = out_k - exit_k
    inflow[:, :, j] += fwd_k[:, :, up]
    inflow[:, :, a0] += fwd_k[:, :, 0] * (1 - pB_k)
    inflow[:, :, b0] += fwd_k[:, :, 0] * pB_k
    # ramp inflow by class
    fr_r = torch.zeros(Bsz, net.R)
    fr_r[:, net.cell_ramp[has_r]] = fr_cell[:, has_r]
    rshare = qr / qr_tot.clamp_min(1e-6)[:, None, :]
    ramp_in = fr_r[:, None, :] * rshare                                       # (B,2,R)
    inflow[:, 2:4, net.ramp_cell] += ramp_in
    qr = (qr - ramp_in).clamp_min(0.0)
    # origin queue into cell 0
    qo_tot = qo.sum(1)
    fin0 = torch.minimum(qo_tot, Rcv[:, 0])
    oshare = qo / qo_tot.clamp_min(1e-6)[:, None]
    o_in = fin0[:, None] * oshare
    inflow[:, THROUGH, 0] += o_in
    qo = (qo - o_in).clamp_min(0.0)

    n_new = (n - out_k + inflow).clamp_min(0.0)
    arrivals_dest = fwd_k[:, :, net.last_cells].sum(2)                       # (B,K) reach the destination
    info = {"out_tot": out_tot, "exit": exit_k.sum(2) + arrivals_dest, "fwd": fwd_k, "pB": pB,
            "o_A": fwd_k[:, 1, 0], "speed": v_now, "ttA": ttA, "ttB": ttB, "cap": cap,
            "transit_exit": exit_k[:, 1, t_cell] if t_cell is not None else torch.zeros(Bsz),
            "ramp_in": fr_r, "exit_cell": exit_k.sum(1)}
    info["o_F"] = fwd_k[:, 4:6, 0]
    info["held"], info["released"] = held, released
    return {"n": n_new, "qo": qo, "qr": qr, "qh": qh}, info


def step(state, p, net: Net, t: int, meters, route_prob_B, cap_factor=None, demand=None, p_agentic=0.0, transit=None,
         fleet=(0.0, 0.41), hold=None):
    """One 30 s step, made of net.sub substeps (route choice and transit offers re-evaluated each substep).

    meters: (B, R) metering rates in veh/h (use ramp_cap for unmetered ramps).
    route_prob_B: callable(TT_A, TT_B) -> (B, 2) probability of choosing route B for (O_H, O_A).
    cap_factor: optional (B, M) multiplier on capacity (truth: capacity drop, incidents).
    demand: optional (d0 (B,), dr (B,R)) overriding the parameter demand (truth: noisy days).
    transit: optional callable(speeds (B,M)) -> (cell, share (B,)): share of agentic through vehicles that
             leave the freeway at `cell` to park and ride transit (multimodal option).
    route_prob_B may return (B, 2) for (O_H, O_A), fleet classes then follow O_A, or (B, 4) for (O_H, O_A, F_occ, F_emp).
    fleet: (share of origin demand served by the fleet, empty share of fleet through vehicles).
    hold: optional (B,) share of newly arriving fleet empties sent to the depot queue instead of the road.
    Returns new state and a dict of flows for metrics (flows summed over substeps; speeds etc. from the first).
    """
    if net.sub == 1:
        return _substep(state, p, net, t, meters, route_prob_B, cap_factor, demand, p_agentic, transit, fleet, hold, DT_H)
    dt = DT_H / net.sub
    info = None
    for _ in range(net.sub):
        state, inf = _substep(state, p, net, t, meters, route_prob_B, cap_factor, demand, p_agentic, transit, fleet, hold, dt)
        if info is None:
            info = dict(inf)
        else:
            for k in ("out_tot", "exit", "fwd", "o_A", "o_F", "transit_exit", "ramp_in", "exit_cell", "held", "released"):
                info[k] = info[k] + inf[k]
    return state, info


def total_vehicles(state):
    return state["n"].sum((1, 2)) + state["qo"].sum(1) + state["qr"].sum((1, 2)) + state.get("qh", 0.0)
