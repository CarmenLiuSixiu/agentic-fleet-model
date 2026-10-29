"""Gate tests for the fleet classes: conservation with fleet + depot hold (CTM and METANET), held empties are
released, and fleet route probabilities are applied to fleet classes only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import ctm, metanet
from network import build_default_net

P = torch.load(os.path.join(os.path.dirname(__file__), "..", "results", "wm_params_v4.pt"))


def _run(sim, init, steps, hold_until):
    net, _, _ = build_default_net()
    st = init(net)
    meters = net.ramp_cap.expand(1, -1)
    rp = lambda a, b: torch.tensor([[0.4, 0.4, 0.9, 0.1]])
    arrived = left = 0.0
    held_total = 0.0
    for t in range(steps):
        d0, dr = ctm.demand_at(P, net, t)
        hold = torch.tensor([0.5]) if t < hold_until else torch.tensor([0.0])
        st, info = sim(st, P, net, t, meters, rp, fleet=(0.2, 0.41), hold=hold)
        arrived += float((d0 + dr.sum()) * ctm.DT_H)
        left += float(info["exit"].sum())
        held_total += float(info["held"].sum())
    return net, st, arrived, left, held_total


def test_ctm_conservation_with_fleet_and_hold():
    net, st, arrived, left, held = _run(ctm.step, lambda n: ctm.init_state(n, 1), 400, 300)
    assert held > 0
    assert abs(float(ctm.total_vehicles(st)) - (arrived - left)) < 1e-2


def test_metanet_conservation_with_fleet_and_hold():
    net, st, arrived, left, held = _run(metanet.step, lambda n: metanet.init_state(n, 1, P), 400, 300)
    total = float(st["n"].sum() + st["qo"].sum() + st["qr"].sum() + st["qh"].sum())
    assert held > 0
    assert abs(total - (arrived - left)) < 1e-2


def test_held_empties_are_released():
    net, st, *_ = _run(ctm.step, lambda n: ctm.init_state(n, 1), 780, 300)
    # 480 steps (4 h) after holding stops, the depot is essentially empty (mean hold 10 min)
    assert float(st["qh"].sum()) < 1e-3


def test_fleet_route_probability_only_moves_fleet():
    net, _, _ = build_default_net()
    st = ctm.init_state(net, 1)
    meters = net.ramp_cap.expand(1, -1)
    for t in range(200):
        st, info = ctm.step(st, P, net, t, meters, lambda a, b: torch.tensor([[0.5, 0.5, 1.0, 1.0]]), fleet=(0.2, 0.41))
    nA_fleet = float(st["n"][0, 4:6][:, net.a_idx].sum())
    nB_fleet = float(st["n"][0, 4:6][:, net.b_idx].sum())
    assert nA_fleet < 1e-6 and nB_fleet > 1.0
