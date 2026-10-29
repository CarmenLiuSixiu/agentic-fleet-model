"""Gate test for the moving-horizon estimator: with a perturbed true state and noise-free detectors, the
re-estimated state must be closer to the truth than the operator's prior, and the vehicle total must stay finite."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
import ctm, operator_est
from network import build_default_net
from operator_est import Estimator


def test_mhe_reduces_state_error():
    torch.manual_seed(0)
    operator_est.NOISE_SD = 0.0
    net, _, _ = build_default_net()
    p = torch.load(os.path.join(os.path.dirname(__file__), "..", "results", "wm_params.pt"))
    cell_of = torch.tensor(sorted(set(int(c) for c in net.sensor_cell.values())))
    est = Estimator(net, p, 0, 0.0, 0.0, cell_of, iters=40)
    truth = Estimator(net, p, 0, 0.0, 0.0, cell_of)        # used only as a simulator
    meters = net.ramp_cap.clone()
    t_pert = 7 * 120 - ctm.T0_MIN * 2 + 60                 # step 360 = 07:30
    for t in range(t_pert + 61):
        if t == t_pert:                                   # the truth jumps away from the operator's belief
            truth.state["n"] = truth.state["n"] * 1.4
        est.predict(t, meters, 0.0)
        with torch.no_grad():
            truth.state, info = truth._step(truth.state, t, meters, 0.0)
        est.observe(t, ctm.fd_speed(truth.state["n"].sum(1), info["cap"], ctm.vf_cells(p, net), net))
    err = lambda: float(((est.state["n"].sum(1) - truth.state["n"].sum(1)).abs() / net.L).mean())
    # advance both to the next re-estimation time
    t = t_pert + 61
    while (t - 60) % operator_est.EST_EVERY:
        est.predict(t, meters, 0.0)
        with torch.no_grad():
            truth.state, info = truth._step(truth.state, t, meters, 0.0)
        est.observe(t, ctm.fd_speed(truth.state["n"].sum(1), info["cap"], ctm.vf_cells(p, net), net))
        t += 1
    before = err()
    est.update(t)
    after = err()
    assert est.n_est == 1
    assert np.isfinite(after)
    assert after < 0.7 * before, (before, after)
