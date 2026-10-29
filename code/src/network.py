"""Build the default network from PEMS-BAY geometry."""
from data import load_meta, route_sensors, route_length_km, MI_TO_KM, ROUTES
from ctm import build_net

MAJOR_A_KM = (18.726 - ROUTES["A"][0][2]) * MI_TO_KM   # I-280 N joins SR-85 N (Cupertino)
MAJOR_B_KM = (ROUTES["B"][0][3] - ROUTES["B"][0][2]) * MI_TO_KM  # SR-87 N meets US-101 N mainline


RAMPS_2026 = "/Users/carmenliu/Desktop/Study2026/AgenticMobility_Yu/code/results/ramps_2026_meta.csv"


def build_default_net(cell_km: float = 1.0, ramps: str = "2017"):
    """ramps='2017': on-ramps/exits at 'rm-n'-named mainline detectors (PEMS-BAY). ramps='2026': at the on-ramp (OR)
    and off-ramp (FR/FF) stations of the April 2026 PeMS metadata (results/ramps_2026_meta.csv)."""
    meta = load_meta()
    sA, sB = route_sensors(meta, "A"), route_sensors(meta, "B")
    ramp_x = exit_x = None
    if ramps == "2026":
        import pandas as pd
        rm = pd.read_csv(RAMPS_2026)
        ramp_x = {1: rm[(rm.route == "A") & (rm.Type == "OR")].x_km.tolist(), 2: rm[(rm.route == "B") & (rm.Type == "OR")].x_km.tolist()}
        exit_x = {1: rm[(rm.route == "A") & rm.Type.isin(["FR", "FF"])].x_km.tolist(), 2: rm[(rm.route == "B") & rm.Type.isin(["FR", "FF"])].x_km.tolist()}
    net = build_net(sA, sB, route_length_km("A"), route_length_km("B"), MAJOR_A_KM, MAJOR_B_KM, cell_km, ramp_x, exit_x)
    return net, sA, sB


def remap_params(p_old: dict, net_old, net_new) -> dict:
    """Map per-cell and per-ramp parameters from one cell size to another by position (warm start).
    Per-cell values are copied from the old cell containing the new cell's centre. A ramp's demand is copied from the
    old ramp in the old cell containing it, split evenly when several new ramps map to the same old ramp."""
    import math, torch
    p = {k: v.clone() for k, v in p_old.items()}
    def old_cell_of(new_c):
        for idx_n, idx_o in ((net_new.a_idx, net_old.a_idx), (net_new.b_idx, net_old.b_idx)):
            pos = (idx_n == new_c).nonzero()
            if len(pos):
                k = int(pos[0]); Ln = float(net_new.L[idx_n[0]]); Lo = float(net_old.L[idx_o[0]])
                x = (k + 0.5) * Ln
                return int(idx_o[min(len(idx_o) - 1, int(x // Lo))])
        return 0                                   # origin cell
    cmap = torch.tensor([old_cell_of(c) for c in range(net_new.M)])
    for k in ("cap_logit", "vf_logit", "exit_logit"):
        if k in p_old:
            p[k] = p_old[k][cmap].clone()
    if "log_r" in p_old:
        old_r = [int(net_old.cell_ramp[cmap[int(c)]]) for c in net_new.ramp_cell]
        # majors map to majors on the same route
        for i in range(net_new.R):
            if bool(net_new.ramp_major[i]):
                old_r[i] = int(((net_old.ramp_major) & (net_old.ramp_route == net_new.ramp_route[i])).nonzero()[0])
        counts = {r: old_r.count(r) for r in set(old_r)}
        rows = []
        for i, r in enumerate(old_r):
            if r >= 0:
                rows.append(p_old["log_r"][r] - math.log(counts[r]))
            else:
                rows.append(torch.full_like(p_old["log_r"][0], math.log(450.0)))
        p["log_r"] = torch.stack(rows)
    return p
