"""Second-order METANET truth with the same network, commodities and interface as ctm.step.

Used only as the 'true world' so that the operator's CTM world model is structurally wrong: speed has its own
dynamics (relaxation, convection, anticipation, on-ramp merging), so capacity drop and stop-and-go fronts emerge
instead of being imposed. Each 30 s call runs three 10 s substeps (CFL: 125 km/h x 10 s < 1 km cells).

Per cell i: lane density rho_i (veh/km/lane), speed v_i (km/h), total flow q_i = rho_i v_i lanes_i.
  v+ = v + T/tau (V(rho) - v) + T/L v (v_up - v) - nu T / (tau L) (rho_dn - rho) / (rho + kappa)
       - delta T r v / (L lanes (rho + kappa))
  V(rho) = vf exp(-(1/a) (rho / rho_c)^a), rho_c chosen so that max flow equals the CTM capacity of the cell.
On-ramp inflow r = min(demand + queue / T, meter, Q_r (rho_max - rho) / (rho_max - rho_c)) (Papageorgiou et al.).
Parameters follow common METANET calibrations (Kotsialos et al. 2002): tau 18 s, nu 60 km^2/h, kappa 40, a 1.867.
"""
from __future__ import annotations
import math
import torch
import ctm
from ctm import K, IS_A, IS_LOCAL, Q_LANE, KJ_LANE, THROUGH, T_HOLD_MIN

TAU_H = 18.0 / 3600.0
NU = 60.0
KAPPA = 40.0
A_EXP = 1.867
DELTA = 0.0122
SUB = 3
T_H = ctm.DT_H / SUB
RHO_MAX = KJ_LANE


def init_state(net, Bsz=1, p=None):
    st = ctm.init_state(net, Bsz)
    vf = ctm.vf_cells(p, net) if p is not None else torch.full((net.M,), 105.0)
    st["v"] = vf.expand(Bsz, -1).clone()
    return st


def _V(rho, rho_c, vf):
    return vf * torch.exp(-(1.0 / A_EXP) * (rho / rho_c).clamp_min(0.0) ** A_EXP)


def step(state, p, net, t, meters, route_prob_B, cap_factor=None, demand=None, p_agentic=0.0, transit=None,
         fleet=(0.0, 0.41), hold=None):
    n, qo, qr, v = state["n"], state["qo"], state["qr"], state["v"]
    qh = state.get("qh", torch.zeros(n.shape[0]))
    Bsz = n.shape[0]
    vf = ctm.vf_cells(p, net).expand(Bsz, -1)
    lanes, L = net.lanes, net.L
    if demand is None:
        d0, dr = ctm.demand_at(p, net, t)
        d0 = d0.expand(Bsz); dr = dr.expand(Bsz, -1)
    else:
        d0, dr = demand
    out_acc = torch.zeros(Bsz, net.M)
    exit_acc = torch.zeros(Bsz, K)
    fwd_acc = torch.zeros(Bsz, K, net.M)
    oA_acc = torch.zeros(Bsz)
    tr_acc = torch.zeros(Bsz)
    v_start = v.clone()
    ttA0, ttB0 = ctm.route_tt_min(v_start, net)
    pB = route_prob_B(ttA0, ttB0)
    if pB.shape[1] == 2:
        pB = torch.cat([pB, pB[:, 1:2], pB[:, 1:2]], 1)
    pB_k = torch.zeros(Bsz, K); pB_k[:, THROUGH] = pB
    s_f, e_f = fleet
    held_acc = torch.zeros(Bsz); rel_acc = torch.zeros(Bsz)
    exit_frac = torch.sigmoid(p["exit_logit"]) * net.exit_mask
    e_base = (exit_frac[None, None, :] * IS_LOCAL[None, :, None]).expand(Bsz, K, net.M).clone()
    t_cell = None
    if transit is not None:
        t_cell, t_share = transit(v_start)
        e_base[:, 1, t_cell] = t_share
    has_r = net.cell_ramp >= 0
    for _ in range(SUB):
        N = n.sum(1)
        shareA = (n * IS_A[None, :, None]).sum(1) / N.clamp_min(1e-6)
        cap = Q_LANE * lanes * ctm.cap_mult(p) * (1.0 + p.get("kappa", torch.tensor(0.0)) * shareA)
        if cap_factor is not None:
            cap = cap * cap_factor
        rho_c = cap / lanes / (vf * math.exp(-1.0 / A_EXP))
        rho = N / (L * lanes)
        # flows over the substep (veh), limited by the vehicles present
        out_tot = torch.minimum(rho * v * lanes * T_H, N)
        frac = n / N.clamp_min(1e-6)[:, None, :]
        out_k = out_tot[:, None, :] * frac
        exit_k = out_k * e_base
        fwd_k = out_k - exit_k
        # on-ramps (queue model) and origin
        arr_emp = s_f * e_f * d0 * T_H
        held = arr_emp * hold if hold is not None else torch.zeros_like(arr_emp)
        released = qh * (T_H * 60.0 / T_HOLD_MIN)
        qh = qh + held - released
        held_acc = held_acc + held; rel_acc = rel_acc + released
        qo = qo + torch.stack([(1 - p_agentic - s_f) * d0 * T_H, p_agentic * d0 * T_H,
                               s_f * (1 - e_f) * d0 * T_H, arr_emp - held + released], 1)
        qr = qr + torch.stack([(1 - p_agentic) * dr, p_agentic * dr], 1) * T_H
        rcell = net.ramp_cell
        rho_r = rho[:, rcell]
        room = ((RHO_MAX - rho_r) / (RHO_MAX - rho_c[:, rcell])).clamp(0.0, 1.0)
        qr_tot = qr.sum(1)
        r_flow = torch.minimum(qr_tot, torch.minimum(meters, net.ramp_cap.expand_as(meters) * room) * T_H)
        rshare = qr / qr_tot.clamp_min(1e-6)[:, None, :]
        ramp_in = r_flow[:, None, :] * rshare
        qr = (qr - ramp_in).clamp_min(0.0)
        room0 = ((RHO_MAX - rho[:, 0]) / (RHO_MAX - rho_c[:, 0])).clamp(0.0, 1.0)
        qo_tot = qo.sum(1)
        f_in0 = torch.minimum(qo_tot, cap[:, 0] * room0 * T_H)
        o_in = f_in0[:, None] * qo / qo_tot.clamp_min(1e-6)[:, None]
        qo = (qo - o_in).clamp_min(0.0)
        # densities
        inflow = torch.zeros(Bsz, K, net.M)
        j = net.merge_cells
        inflow[:, :, j] += fwd_k[:, :, j - 1]
        a0, b0 = net.a_idx[0], net.b_idx[0]
        inflow[:, :, a0] += fwd_k[:, :, 0] * (1 - pB_k)
        inflow[:, :, b0] += fwd_k[:, :, 0] * pB_k
        inflow[:, 2:4, rcell] += ramp_in
        inflow[:, THROUGH, 0] += o_in
        n = (n - out_k + inflow).clamp_min(0.0)
        # speeds
        v_up = v.clone()
        v_up[:, j] = v[:, j - 1]
        v_up[:, a0] = v[:, 0]; v_up[:, b0] = v[:, 0]
        rho_dn = rho.clone()
        rho_dn[:, j - 1] = rho[:, j]
        pbar0 = (n[:, :, 0] / n[:, :, 0].sum(1, keepdim=True).clamp_min(1e-6) * pB_k).sum(1)
        rho_dn[:, 0] = (1 - pbar0) * rho[:, a0] + pbar0 * rho[:, b0]
        r_cell = torch.zeros(Bsz, net.M)
        r_cell[:, rcell] = r_flow / T_H
        Vr = _V(rho, rho_c, vf)
        v = v + T_H / TAU_H * (Vr - v) + T_H / L * v * (v_up - v) \
            - NU * T_H / (TAU_H * L) * (rho_dn - rho) / (rho + KAPPA) \
            - DELTA * T_H * r_cell * v / (L * lanes * (rho + KAPPA))
        v = torch.minimum(v.clamp_min(5.0), vf * 1.05)
        out_acc = out_acc + out_tot
        exit_acc = exit_acc + exit_k.sum(2) + fwd_k[:, :, net.last_cells].sum(2)
        fwd_acc = fwd_acc + fwd_k
        oA_acc = oA_acc + fwd_k[:, 1, 0]
        if t_cell is not None:
            tr_acc = tr_acc + exit_k[:, 1, t_cell]
    info = {"out_tot": out_acc, "exit": exit_acc, "fwd": fwd_acc, "pB": pB, "o_A": oA_acc, "speed": v_start,
            "ttA": ttA0, "ttB": ttB0, "cap": cap, "transit_exit": tr_acc, "v_new": v,
            "o_F": fwd_acc[:, 4:6, 0], "held": held_acc, "released": rel_acc}
    return {"n": n, "qo": qo, "qr": qr, "v": v, "qh": qh}, info
