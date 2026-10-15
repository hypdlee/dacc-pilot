"""Section 14 correctness checks and Appendix A arithmetic."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from dacc.analysis import controller_metrics, evaluate_decisions, gather_actions, headroom_report
from dacc.controller import (AdditiveController, BilinearController, JointController, SequentialController,
                             center, decide, descriptor, descriptor_dim, score_loss)
from dacc.data import DeviceData, stratified_split
from dacc.deploy import DeployModel
from dacc.evaluate import deploy_forward
from dacc.model import CorrectedWRN, action_index, action_pair, action_table, bn_state
from dacc.targets import collect_targets, single_action_losses
from dacc.train import predict_table

TOL = 1e-5


class FloatData:
    """Minimal loader yielding the given float tensors (identity transform)."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def batches(self, indices, batch_size, shuffle=False, augment=False, gen=None, drop_last=False):
        idx = torch.as_tensor(indices)
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            yield self.x[b], self.y[b], b


# ---------------------------------------------------------------- equivalences
def test_zero_action_reproduces_shared_model(small_model, batch):
    x, _ = batch
    m = small_model
    zero = torch.zeros(len(x), 2, dtype=torch.long)
    assert torch.equal(m(x), m(x, zero))
    # the correction for choice 0 is exactly zero
    h = m.prefix(x)
    assert torch.count_nonzero(m.corr_a.forward_choice(h, torch.zeros(len(x), dtype=torch.long))) == 0
    # and equals a network with the correction modules removed from the computation
    ref = copy.deepcopy(m)
    h2 = ref.prefix(x)
    z = ref.suffix(ref.stage3[-1](ref.middle(ref.stage2[-1](h2))))
    assert torch.allclose(z, m(x), atol=0, rtol=0)


def test_grouped_matches_looped(small_model, batch):
    x, _ = batch
    m = small_model
    h = m.prefix(x)
    choice = torch.tensor([0, 1, 2, 3, 4, 1, 1, 0, 4, 3, 2, 2])
    g = m.corr_a.forward_choice(h, choice)
    l = m.corr_a.forward_choice_looped(h, choice)
    assert torch.allclose(g, l, atol=TOL)
    assert g.abs().sum() > 0


def test_dense_onehot_matches_grouped(small_model, batch):
    x, _ = batch
    m = small_model
    acts = torch.randint(0, 5, (len(x), 2), generator=torch.Generator().manual_seed(4))
    oh_a = F.one_hot(acts[:, 0], 5).float()
    oh_b = F.one_hot(acts[:, 1], 5).float()
    assert torch.allclose(m(x, acts), m(x, weights_a=oh_a, weights_b=oh_b), atol=TOL)


def test_kernel_fusion_equals_output_mixing(small_model, batch):
    x, _ = batch
    m = small_model
    h = m.prefix(x)
    w = torch.softmax(torch.randn(len(x), 5, generator=torch.Generator().manual_seed(2)), 1)
    a = m.corr_a.forward_weights(h, w)
    b = m.corr_a.forward_kernel_fusion(h, w)
    # explicit linear output mixing of the individual experts
    c = sum(w[:, e:e + 1, None, None] * m.corr_a.forward_choice(h, torch.full((len(x),), e, dtype=torch.long))
            for e in range(1, 5))
    assert torch.allclose(a, b, atol=TOL)
    assert torch.allclose(a, c, atol=TOL)


def test_untied_dense_matches_grouped():
    m = CorrectedWRN(10, depth=16, widen=1, untied=True, init_seed=5).eval()
    x = torch.randn(6, 3, 32, 32)
    acts = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [2, 2]])
    oh_a, oh_b = F.one_hot(acts[:, 0], 5).float(), F.one_hot(acts[:, 1], 5).float()
    assert torch.allclose(m(x, acts), m(x, weights_a=oh_a, weights_b=oh_b), atol=TOL)
    h = m.prefix(x)
    assert torch.allclose(m.corr_a.forward_choice(h, acts[:, 0]), m.corr_a.forward_choice_looped(h, acts[:, 0]), atol=TOL)


def test_batch_shuffle_invariance(small_model, batch):
    x, _ = batch
    m = small_model
    acts = torch.randint(0, 5, (len(x), 2), generator=torch.Generator().manual_seed(7))
    out = m(x, acts)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(8))
    out_p = m(x[perm], acts[perm])
    inv = torch.argsort(perm)
    assert torch.allclose(out_p[inv], out, atol=TOL)


# ---------------------------------------------------------------- target collection
def test_collector_matches_full_forward_and_leaves_state(small_model, batch):
    x, y = batch
    m = small_model
    m.train()                                   # collector must switch to eval itself
    bn0 = bn_state(m)
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    T = collect_targets(m, FloatData(x, y), np.arange(len(y)), 5, torch.device("cpu"), "full")
    for k, v in bn_state(m).items():
        assert torch.equal(v, bn0[k])
    for k, v in m.state_dict().items():
        assert torch.equal(v, sd0[k])
    m.eval()
    for i in range(5):
        for j in range(5):
            a = torch.tensor([i, j]).repeat(len(y), 1)
            direct = single_action_losses(m, x, y, a)
            assert torch.allclose(T["losses"][:, i, j], direct, atol=1e-5), (i, j)
            pred = m(x, a).argmax(1)
            assert torch.equal(T["preds"][:, i, j].long(), pred)
    # B-input descriptors depend on the A choice (no reuse of the zero-correction input)
    assert not torch.allclose(T["desc_b"][:, 0], T["desc_b"][:, 1])
    assert T["desc_h"].shape == (len(y), descriptor_dim(m.widths[1]))


def test_action_index_convention():
    tbl = action_table(5)
    assert len(tbl) == 25
    for a, (i, j) in enumerate(tbl):
        assert a == 5 * i + j == action_index(i, j)
        assert action_pair(a) == (i, j)
    L = torch.arange(25.0).view(1, 5, 5)
    acts = torch.tensor([[2, 3]])
    assert gather_actions(L, acts).item() == 13.0
    q = torch.zeros(1, 5, 5)
    q[0, 3, 1] = -1
    assert decide(q).tolist() == [[3, 1]]


def test_tie_rule_lexicographic():
    q = torch.zeros(2, 5, 5)
    assert decide(q).tolist() == [[0, 0], [0, 0]]
    q = torch.ones(1, 5, 5)
    q[0, 2, 1] = 0
    q[0, 1, 2] = 0
    assert decide(q).tolist() == [[1, 2]]


# ---------------------------------------------------------------- gradients
def test_gradient_reaches_every_expert_first_update(small_model, batch):
    m = small_model
    m.train()
    x = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 10, (50,))
    acts = torch.stack(torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij"), -1).view(25, 2).repeat(2, 1)
    m.zero_grad()
    F.cross_entropy(m(x, acts), y).backward()
    for corr in (m.corr_a, m.corr_b):
        for e in range(4):
            assert corr.depthwise.grad[e].abs().sum() > 0, f"expert {e} got no gradient"
        assert corr.reduce[0].weight.grad.abs().sum() > 0
        assert corr.expand[0].weight.grad.abs().sum() > 0
        assert corr.gain.grad is not None and corr.gain.grad.abs() > 0


def test_nonzero_initial_gain_passes_gradient():
    m = CorrectedWRN(10, depth=16, widen=1, gain_init=0.05, init_seed=1)
    m.train()
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    F.cross_entropy(m(x, torch.ones(8, 2, dtype=torch.long)), y).backward()
    assert m.corr_a.depthwise.grad[0].abs().sum() > 0


# ---------------------------------------------------------------- controllers / labels
def test_deploy_uses_no_labels_and_no_unselected_futures(small_model, batch):
    x, _ = batch
    m = small_model
    d = descriptor_dim(m.widths[1])
    db = descriptor_dim(m.widths[2])
    calls = {"middle": 0}
    orig = m.middle

    def counting(hh):
        calls["middle"] += 1
        return orig(hh)

    m.middle = counting
    try:
        ctrl = JointController(d, 16).eval()
        deploy_forward(m, ctrl, x, "full")
        assert calls["middle"] == 1            # only the executed A choice is run forward
        calls["middle"] = 0
        seq = SequentialController(d, db, 16, 16).eval()
        deploy_forward(m, seq, x, "full")
        assert calls["middle"] == 1
    finally:
        m.middle = orig
    import inspect

    assert list(inspect.signature(deploy_forward).parameters) == ["model", "ctrl", "x", "desc_kind", "cost", "lam"]
    assert list(inspect.signature(DeployModel.forward).parameters) == ["self", "x01"]


def test_controller_decisions_are_per_example(small_model, batch):
    x, _ = batch
    m = small_model
    ctrl = JointController(descriptor_dim(m.widths[1]), 16).eval()
    _, a_full = deploy_forward(m, ctrl, x, "full")
    _, a_single = deploy_forward(m, ctrl, x[:1], "full")
    assert torch.equal(a_full[:1], a_single)


def test_predict_table_matches_real_execution(small_model, batch):
    x, y = batch
    m = small_model
    T = collect_targets(m, FloatData(x, y), np.arange(len(y)), 4, torch.device("cpu"), "full")
    d, db = descriptor_dim(m.widths[1]), descriptor_dim(m.widths[2])
    for ctrl in (JointController(d, 16), AdditiveController(d, 16), BilinearController(d, 16, rank=2),
                 SequentialController(d, db, 16, 16)):
        torch.manual_seed(0)
        for p in ctrl.parameters():
            p.data.normal_(0, 0.5)
        ctrl.eval()
        _, a_tab = predict_table(ctrl, T, torch.device("cpu"))
        _, a_real = deploy_forward(m, ctrl, x, "full")
        assert torch.equal(a_tab, a_real), ctrl.kind


def test_descriptor_dims():
    h = torch.randn(3, 128, 16, 16)
    assert descriptor(h, "full").shape == (3, 768)
    assert descriptor(h, "meanstd").shape == (3, 256)
    assert descriptor(h, "mean").shape == (3, 128)


def test_additive_has_zero_interaction_and_bilinear_index0():
    d = 20
    add = AdditiveController(d, 8)
    bil = BilinearController(d, 8, rank=2)
    for p in list(add.parameters()) + list(bil.parameters()):
        p.data.normal_()
    z = torch.randn(4, d)
    q = add(z)
    I = q[:, :, :1] + q[:, :1, :] - q - q[:, :1, :1]
    assert I.abs().max() < 1e-5
    qb = bil(z)
    Ib = qb[:, :, :1] + qb[:, :1, :] - qb - qb[:, :1, :1]
    assert Ib[:, 0, :].abs().max() < 1e-5 and Ib[:, :, 0].abs().max() < 1e-5
    assert Ib[:, 1:, 1:].abs().max() > 1e-4      # can express interactions


def test_centering_removes_per_example_offset():
    q = torch.randn(5, 5, 5)
    L = torch.randn(5, 5, 5)
    shift = torch.randn(5, 1, 1)
    assert torch.allclose(score_loss(q + shift, L), score_loss(q, L), atol=1e-6)
    assert torch.equal(decide(q + shift), decide(q))
    assert torch.allclose(center(q).flatten(1).mean(1), torch.zeros(5), atol=1e-6)


# ---------------------------------------------------------------- theory and arithmetic
def test_proposition1_regret_bound():
    g = torch.Generator().manual_seed(0)
    L = torch.rand(2000, 5, 5, generator=g)
    q = L + 0.1 * torch.randn(2000, 5, 5, generator=g)
    a = decide(q)
    reg = gather_actions(L, a) - L.flatten(1).min(1).values
    delta = (q - L).flatten(1).abs().max(1).values
    assert torch.all(reg <= 2 * delta + 1e-6)
    T = {"losses": L, "correct": torch.zeros(2000, 5, 5, dtype=torch.bool)}
    assert controller_metrics(q, T)["prop1_bound_holds_frac"] == 1.0


def test_interaction_identity_and_worked_example():
    L = torch.zeros(1, 5, 5)
    L[0, 0, 0], L[0, 1, 0], L[0, 0, 1], L[0, 1, 1] = 1.20, 1.15, 1.16, 0.80
    U = L[:, :1, :1] - L
    I = L[:, :, :1] + L[:, :1, :] - L - L[:, :1, :1]
    assert abs(U[0, 1, 0].item() - 0.05) < 1e-6
    assert abs(U[0, 0, 1].item() - 0.04) < 1e-6
    assert abs(U[0, 1, 1].item() - 0.40) < 1e-6
    assert abs(I[0, 1, 1].item() - 0.31) < 1e-6
    g = torch.Generator().manual_seed(1)
    L = torch.rand(10, 5, 5, generator=g)
    U = L[:, :1, :1] - L
    I = L[:, :, :1] + L[:, :1, :] - L - L[:, :1, :1]
    assert torch.allclose(U, U[:, :, :1] + U[:, :1, :] + I, atol=1e-6)


def test_parameter_and_mac_arithmetic():
    m = CorrectedWRN(100)
    assert m.corr_a.params_excluding_gain() == 9344
    assert m.corr_b.params_excluding_gain() == 35072
    assert m.corr_a.params_excluding_gain() + m.corr_b.params_excluding_gain() == 44416
    assert m.corr_a.macs_selected(16, 16) == 2170880
    assert m.corr_b.macs_selected(8, 8) == 2134016
    ctrl = JointController(descriptor_dim(128), 128)
    assert sum(p.numel() for p in ctrl.parameters()) == 101657
    assert ctrl.mlp.fc1.in_features * 128 + 128 * 25 == 101504
    assert m.widths == [64, 128, 256]
    assert m.site_shapes() == {"A": (128, 16, 16), "B": (256, 8, 8)}


def test_site_blocks_are_stride_one_and_shape_preserving():
    m = CorrectedWRN(100)
    assert m.stage2[-1].stride == 1 and m.stage2[-1].equal
    assert m.stage3[-1].stride == 1 and m.stage3[-1].equal
    assert m.stage2[0].stride == 2 and m.stage3[0].stride == 2
    assert len(m.stage1) == len(m.stage2) == len(m.stage3) == 4


# ---------------------------------------------------------------- data
def test_stratified_split_exact_and_disjoint():
    labels = torch.arange(100).repeat_interleave(500)
    sp = stratified_split(labels, {"bank": 40000, "route": 5000, "val": 5000}, seed=0)
    assert [len(sp[k]) for k in ("bank", "route", "val")] == [40000, 5000, 5000]
    allidx = np.concatenate([sp["bank"], sp["route"], sp["val"]])
    assert len(np.unique(allidx)) == 50000
    for k, per in (("bank", 400), ("route", 50), ("val", 50)):
        counts = np.bincount(labels.numpy()[sp[k]], minlength=100)
        assert np.all(counts == per)
    sp2 = stratified_split(labels, {"bank": 40000, "route": 5000, "val": 5000}, seed=0)
    assert all(np.array_equal(sp[k], sp2[k]) for k in sp)


def test_augmentation_deterministic_and_shape():
    x = torch.randint(0, 255, (8, 3, 32, 32), dtype=torch.uint8)
    a = DeviceData.augment(x, torch.Generator().manual_seed(3))
    b = DeviceData.augment(x, torch.Generator().manual_seed(3))
    assert a.shape == (8, 3, 32, 32) and torch.equal(a, b)
    # every output is some (dy, dx, flip) crop of the reflection-padded input
    padded = F.pad(x.float(), (4, 4, 4, 4), mode="reflect")
    for n in range(8):
        found = False
        for dy in range(9):
            for dx in range(9):
                crop = padded[n, :, dy:dy + 32, dx:dx + 32]
                if torch.equal(a[n], crop) or torch.equal(a[n], crop.flip(2)):
                    found = True
        assert found


# ---------------------------------------------------------------- diagnostics
def test_headroom_report_and_oracles():
    g = torch.Generator().manual_seed(0)
    L = torch.rand(100, 5, 5, generator=g)
    C = torch.rand(100, 5, 5, generator=g) > 0.5
    T = {"losses": L, "correct": C}
    rep = headroom_report(T, (0, 0))
    assert rep["accuracy_oracle_accuracy"] >= rep["best_fixed_accuracy"]
    assert rep["loss_oracle_mean_loss"] <= rep["best_fixed_mean_loss"]
    ev = evaluate_decisions(T, torch.zeros(100, 2, dtype=torch.long))
    assert abs(ev["accuracy"] - rep["static_accuracy"]) < 1e-6
    assert ev["regret_mean"] >= 0


def test_small_overfit():
    torch.manual_seed(0)
    m = CorrectedWRN(4, depth=16, widen=1, init_seed=0)
    x = torch.randn(16, 3, 32, 32)
    y = torch.arange(16) % 4
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    m.train()
    for _ in range(60):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    m.train()
    assert (m(x).argmax(1) == y).float().mean() == 1.0
