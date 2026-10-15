"""Appendix B initial pilot values must be the defaults."""

import pytest

from dacc.config import Config, load_config


def test_appendix_b_defaults():
    c = Config()
    assert c.data.dataset == "cifar100"
    assert (c.data.bank_size, c.data.route_size, c.data.val_size) == (40000, 5000, 5000)
    assert c.seeds_screening == [11, 22, 33]
    assert c.seeds_finalist == [11, 22, 33, 44, 55]
    assert (c.model.depth, c.model.widen) == (28, 4)
    assert c.model.num_experts == 4                      # four experts + zero correction
    assert c.model.rank_divisor == 4                     # ranks 32 and 64
    assert c.model.gain_init == 0.05
    assert c.controller.hidden == 128
    assert c.controller.descriptor == "full"             # 768-dim descriptor at 128 channels
    # correction warm-up
    assert c.bank.warm_epochs == 10 and c.bank.warm_lr == 0.01
    # bank refinement
    assert c.bank.refine_epochs == 30 and c.bank.refine_corr_lr == 0.01 and c.bank.refine_shared_lr == 0.001
    assert c.bank.anchor_eta == 0.25
    # controller training
    assert c.controller.lr == 3e-4 and c.controller.weight_decay == 1e-4
    assert c.controller.batch_size == 256 and c.controller.epochs == 30
    # alternating refinement
    assert c.alternate.rounds == 3 and c.alternate.bank_epochs == 10
    assert c.controller.score_loss == "huber" and c.controller.huber_delta == 0.1
    assert c.controller.tau == 0.1 and c.controller.beta_kl == 0.0
    assert c.alternate.eps == 0.2 and c.alternate.eps_final == 0.05
    assert c.eval.cost_lambda == 0.0
    # static screening recipe (Section 9.1)
    s = c.static
    assert (s.epochs, s.batch_size, s.momentum, s.lr, s.weight_decay, s.warmup_epochs) == (200, 128, 0.9, 0.1, 5e-4, 5)
    # gates (Section 15.4)
    assert c.eval.headroom_gate_pp == 0.5 and c.eval.coordination_gate_pp == 0.3


def test_overrides_and_validation(tmp_path):
    cfg = load_config(None, ["model.widen=2", "controller.hidden=64"])
    assert cfg.model.widen == 2 and cfg.controller.hidden == 64
    with pytest.raises(KeyError):
        load_config(None, ["model.nonexistent=1"])
    with pytest.raises(ValueError):
        load_config(None, ["bank.refine_shared_lr=0.005"])   # must be <= corr lr / 10
    with pytest.raises(ValueError):
        load_config(None, ["model.depth=10"])                 # one block per stage: no stride-one site


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files
    for f in files:
        load_config(str(f))
