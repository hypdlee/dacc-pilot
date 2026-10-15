"""End-to-end pilot pipeline (work packages WP2-WP7 for one seed).

    python -m dacc.pipeline --config configs/pilot_cifar100.yaml --seed 11 --stages all

Stages (each resumable; outputs under <out_root>/<run_name>/seed<seed>/):

  static       B0 static WRN backbone on the bank split            ckpt/static.pt
  b1           B1 capacity control: always-on corrections with extra
               parameters matched to bank + joint controller,
               trained from scratch for the static + bank budget  ckpt/b1.pt
  bank         correction bank (frozen warm-up, joint refinement)  ckpt/bank.pt
  collect      25-action target tables on route / val (/ test)     targets/*.pt
  analyze      headroom, interactions, coverage, redundancy (H1)    metrics/headroom_r0.json
  controllers  B5 additive (+wide), B6 sequential (+wide),
               P1 joint, P2 bilinear on the frozen bank             ckpt/ctrl_*.pt
  routers      B3 soft dynamic conv, B4 hard routing (frozen bank)  ckpt/router_*.pt
  evaluate     real-execution evaluation of every method, D0
               oracle diagnostics, intervention controls, gates     metrics/results.json
  alternate    alternating refinement of P1 with matched-budget
               B4 and static controls                               metrics/alternate.json
  profile      parameters, MACs, latency, training cost              metrics/profile.json
  export       deployment bundle, ONNX, TorchScript, equivalence     deploy/

The official test split is only evaluated when ``eval.final_test: true``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from . import __version__
from .alternate import run_alternation
from .analysis import (best_fixed_pair, confusion_counts, controller_metrics, corrected_overlap,
                       evaluate_decisions, headroom_report, intervention_controls, recovery)
from .config import Config, load_config, save_config
from .controller import descriptor_dim, mlp_params
from .data import DataBundle
from .deploy import (DeployModel, controller_build_spec, controller_from_build, export_onnx,
                     export_torchscript, save_bundle, verify_exports)
from .evaluate import deploy_forward, eval_with_policy, strip_private
from .model import CorrectedWRN, build_model
from .profiling import macs_report, measure_latency, param_report, target_collection_cost
from .routers import OrdinaryRouter, eval_router, train_router
from .targets import collect_targets, load_targets, save_targets, single_action_losses
from .train import b1_site_ranks, fit_controller, predict_table, train_bank, train_static
from .utils import (JsonlLogger, hash_state_dict, read_json, resolve_device, seed_everything,
                    software_versions, write_json)

STAGES = ["static", "b1", "bank", "collect", "analyze", "controllers", "routers", "evaluate",
          "alternate", "profile", "export"]
CONTROLLER_KINDS = ["joint", "additive", "bilinear", "sequential"]
METHOD_KEYS = ["B0_static", "B1_capacity", "B2_best_fixed", "B3_soft_dynamic", "B4_hard_routing",
               "B5_additive", "B6_sequential", "P1_joint", "P2_bilinear"]


class Run:
    def __init__(self, cfg: Config, seed: int, force: bool = False):
        self.cfg, self.seed, self.force = cfg, seed, force
        self.dir = Path(cfg.out_root) / cfg.run_name / f"seed{seed}"
        for sub in ("ckpt", "targets", "metrics", "deploy", "logs"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.device)
        seed_everything(seed, cfg.deterministic)
        self.logger = JsonlLogger(self.dir / "logs" / "metrics.jsonl")
        cfg_hash = hashlib.sha256(cfg.to_json().encode()).hexdigest()[:10]
        self.run_id = f"{cfg.run_name}-s{seed}-{cfg_hash}"
        save_config(cfg, self.dir / "config.json")
        write_json(self.dir / "versions.json", {**software_versions(), "dacc": __version__})
        self.bundle = DataBundle(cfg, self.device)
        self.bundle.assert_disjoint()
        write_json(self.dir / "splits.json", {
            "hashes": self.bundle.hashes,
            "sizes": {k: int(len(v)) for k, v in self.bundle.splits.items()},
            "normalisation": {"mean": self.bundle.mean, "std": self.bundle.std, "source": "bank split"},
            "split_seed": cfg.data.split_seed,
        })
        write_json(self.dir / "run_info.json", {"run_id": self.run_id, "seed": seed, "config_hash": cfg_hash,
                                                "device": str(self.device)})
        self.timings_path = self.dir / "metrics" / "timings.json"
        self.timings: Dict[str, float] = read_json(self.timings_path) if self.timings_path.exists() else {}
        self.K = self.bundle.num_classes

    # ------------------------------------------------------------- helpers
    def p(self, rel: str) -> Path:
        return self.dir / rel

    def need(self, rel: str) -> bool:
        return self.force or not self.p(rel).exists()

    def save_model(self, model: nn.Module, rel: str, meta: Dict[str, Any]) -> None:
        sd = model.state_dict()
        torch.save({"state": sd, "hash": hash_state_dict(sd), "meta": meta}, self.p(rel))

    def load_model(self, rel: str, **build_kw) -> CorrectedWRN:
        ck = torch.load(self.p(rel), map_location=self.device, weights_only=False)
        model = build_model(self.cfg, self.K, **build_kw).to(self.device)
        model.load_state_dict(ck["state"])
        model.eval()
        return model

    def ckpt_hash(self, rel: str) -> str:
        return torch.load(self.p(rel), map_location="cpu", weights_only=False)["hash"]

    def save_ctrl(self, ctrl: nn.Module, rel: str, meta: Dict[str, Any]) -> None:
        torch.save({"state": ctrl.state_dict(), "build": controller_build_spec(ctrl), "meta": meta}, self.p(rel))

    def load_ctrl(self, rel: str) -> nn.Module:
        ck = torch.load(self.p(rel), map_location=self.device, weights_only=False)
        c = controller_from_build(ck["build"]).to(self.device)
        c.load_state_dict(ck["state"])
        return c.eval()

    def load_router(self, rel: str) -> OrdinaryRouter:
        ck = torch.load(self.p(rel), map_location=self.device, weights_only=False)
        r = OrdinaryRouter(**ck["build"]).to(self.device)
        r.load_state_dict(ck["state"])
        return r.eval()

    def time_stage(self, name: str, fn: Callable[[], None]) -> None:
        t0 = time.perf_counter()
        fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timings[name] = time.perf_counter() - t0
        write_json(self.timings_path, self.timings)

    # ------------------------------------------------------------- stages
    def stage_static(self) -> None:
        if not self.need("ckpt/static.pt"):
            return
        model = build_model(self.cfg, self.K, init_seed=self.seed).to(self.device)
        info = train_static(self.cfg, model, self.bundle, self.device, self.seed, self.logger, tag="static")
        self.save_model(model, "ckpt/static.pt", {"selected_epoch": info["selected_epoch"],
                                                  "checkpoint_rule": info["checkpoint_rule"]})
        write_json(self.p("metrics/static.json"), info)

    def _joint_ctrl_params(self) -> int:
        d = descriptor_dim(build_model(self.cfg, self.K).widths[1], self.cfg.controller.descriptor)
        return mlp_params(d, self.cfg.controller.hidden, (self.cfg.model.num_experts + 1) ** 2)

    def stage_b1(self) -> None:
        if not self.need("ckpt/b1.pt"):
            return
        ref = build_model(self.cfg, self.K)
        target_extra = ref.corr_a.params_excluding_gain() + ref.corr_b.params_excluding_gain() + self._joint_ctrl_params()
        ranks = b1_site_ranks(self.cfg, target_extra, ref.widths)
        model = build_model(self.cfg, self.K, site_ranks=ranks, num_experts=1, init_seed=self.seed).to(self.device)
        epochs = self.cfg.static.epochs + self.cfg.bank.warm_epochs + self.cfg.bank.refine_epochs
        info = train_static(self.cfg, model, self.bundle, self.device, self.seed, self.logger, tag="b1",
                            epochs=epochs, fixed_action=(1, 1))
        extra = model.corr_a.params_excluding_gain() + model.corr_b.params_excluding_gain()
        meta = {"site_ranks": list(ranks), "extra_params": extra, "target_extra_params": target_extra,
                "epochs": epochs, "action": [1, 1]}
        self.save_model(model, "ckpt/b1.pt", meta)
        write_json(self.p("metrics/b1.json"), {**info, **meta})

    def stage_bank(self) -> None:
        if not self.need("ckpt/bank.pt"):
            return
        model = self.load_model("ckpt/static.pt")
        info = train_bank(self.cfg, model, self.bundle, self.device, self.seed, self.logger)
        self.save_model(model, "ckpt/bank.pt", {"source_static_hash": self.ckpt_hash("ckpt/static.pt")})
        write_json(self.p("metrics/bank.json"), info)

    def _collect(self, model: CorrectedWRN, split: str, tag: str, ckpt: str) -> Dict[str, Any]:
        data, idx = self.bundle.source(split)
        T = collect_targets(model, data, idx, 250, self.device, self.cfg.controller.descriptor, self.cfg.amp,
                            extra_meta={"split": split, "split_hash": self.bundle.hashes[split],
                                        "source_checkpoint": ckpt, "round": 0, "run_id": self.run_id})
        save_targets(T, self.p(f"targets/{split}_{tag}.pt"))
        return T

    def stage_collect(self) -> None:
        if not self.need("targets/val_r0.pt"):
            if self.cfg.eval.final_test and not self.p("targets/test_r0.pt").exists():
                self._collect(self.load_model("ckpt/bank.pt"), "test", "r0", "ckpt/bank.pt")
            return
        model = self.load_model("ckpt/bank.pt")
        T_route = self._collect(model, "route", "r0", "ckpt/bank.pt")
        T_val = self._collect(model, "val", "r0", "ckpt/bank.pt")
        if self.cfg.eval.final_test:
            self._collect(model, "test", "r0", "ckpt/bank.pt")
        # verify cached collector against independent complete forward passes
        g = torch.Generator().manual_seed(self.seed)
        sel = torch.randperm(len(T_val["indices"]), generator=g)[:64]
        idx = T_val["indices"][sel].numpy()
        x = self.bundle.train.normalize(self.bundle.train.x[torch.as_tensor(idx).to(self.bundle.train.x.device)])
        y = self.bundle.train.y[torch.as_tensor(idx).to(self.bundle.train.y.device)].to(self.device)
        n = model.n_choices
        acts = torch.randint(0, n, (len(idx), 2), generator=g).to(self.device)
        direct = single_action_losses(model, x, y, acts).cpu()
        cached = T_val["losses"][sel, acts[:, 0].cpu(), acts[:, 1].cpu()]
        diff = float((direct - cached).abs().max())
        write_json(self.p("metrics/collect_check.json"), {
            "max_abs_diff_vs_full_forward": diff, "examples_checked": len(idx),
            "route_meta": T_route["meta"], "val_meta": T_val["meta"]})
        if diff > 1e-4:
            raise RuntimeError(f"cached targets disagree with full forward pass: {diff}")

    def stage_analyze(self) -> None:
        if not self.need("metrics/headroom_r0.json"):
            return
        T_route, T_val = load_targets(self.p("targets/route_r0.pt")), load_targets(self.p("targets/val_r0.pt"))
        pair = best_fixed_pair(T_route)
        rep = headroom_report(T_val, pair, self.cfg.eval.coverage_thresholds)
        rep["best_fixed_selected_on"] = "route split (labelled), evaluated on validation"
        rep["gate_headroom"] = {
            "criterion_pp": self.cfg.eval.headroom_gate_pp,
            "value_pp": rep["oracle_headroom_pp_vs_best_fixed"],
            "pass": rep["oracle_headroom_pp_vs_best_fixed"] >= self.cfg.eval.headroom_gate_pp,
            "on_fail": "revise the bank or stop the routing study"}
        write_json(self.p("metrics/headroom_r0.json"), rep)
        self.logger.log({"stage": "analyze", "oracle_headroom_pp": rep["oracle_headroom_pp_vs_best_fixed"],
                         "headroom_gate_pass": rep["gate_headroom"]["pass"],
                         "frac_decision_changed_by_interaction": rep["frac_decision_changed_by_interaction"]})

    def stage_controllers(self) -> None:
        if not self.need("metrics/controllers.json"):
            return
        T_route, T_val = load_targets(self.p("targets/route_r0.pt")), load_targets(self.p("targets/val_r0.pt"))
        out: Dict[str, Any] = {}
        trained: Dict[str, nn.Module] = {}
        for kind in CONTROLLER_KINDS:
            ctrl, info = fit_controller(kind, self.cfg, T_route, T_val, self.device, self.seed, self.logger, tag=kind)
            trained[kind] = ctrl
            q, a = predict_table(ctrl, T_val, self.device)
            out[kind] = {**info, "val": controller_metrics(q, T_val, a)}
            self.save_ctrl(ctrl, f"ckpt/ctrl_{kind}.pt", {"kind": kind, "bank_hash": T_route["meta"]["checkpoint_hash"]})
        if self.cfg.controller.match_capacity:
            target = sum(p.numel() for p in trained["joint"].parameters())
            for kind in ("additive", "sequential"):
                ctrl, info = fit_controller(kind, self.cfg, T_route, T_val, self.device, self.seed, self.logger,
                                            tag=f"{kind}_wide", wide_target=target)
                q, a = predict_table(ctrl, T_val, self.device)
                out[f"{kind}_wide"] = {**info, "val": controller_metrics(q, T_val, a), "matched_to_params": target}
                self.save_ctrl(ctrl, f"ckpt/ctrl_{kind}_wide.pt", {"kind": kind, "wide": True})
        write_json(self.p("metrics/controllers.json"), out)

    def stage_routers(self) -> None:
        if not self.need("metrics/routers.json"):
            return
        out = {}
        for mode in ("soft", "hard"):
            model = self.load_model("ckpt/bank.pt")
            router, info = train_router(mode, self.cfg, model, self.bundle, self.device, self.seed, self.logger)
            build = {"d_a": router.head_a.fc1.in_features, "d_b": router.head_b.fc1.in_features,
                     "hidden": router.head_a.fc1.out_features, "n": router.n, "mode": mode,
                     "desc_kind": router.desc_kind}
            torch.save({"state": router.state_dict(), "build": build}, self.p(f"ckpt/router_{mode}.pt"))
            out[mode] = {**info, "val": eval_router(router, model, self.bundle, "val")}
        write_json(self.p("metrics/routers.json"), out)

    # ------------------------------------------------------------- evaluation
    def _evaluate_split(self, split: str) -> Dict[str, Any]:
        cfg, dev = self.cfg, self.device
        data, idx = self.bundle.source(split)
        T = load_targets(self.p(f"targets/{split}_r0.pt"))
        T_route = load_targets(self.p("targets/route_r0.pt"))
        bank = self.load_model("ckpt/bank.pt")
        static = self.load_model("ckpt/static.pt")
        dk = cfg.controller.descriptor
        pair = best_fixed_pair(T_route)
        res: Dict[str, Any] = {}
        raw: Dict[str, Dict[str, Any]] = {}

        def record(key: str, ev: Dict[str, Any], **extra) -> None:
            raw[key] = ev
            res[key] = {**strip_private(ev), **extra}

        record("B0_static", eval_with_policy(static, data, idx, 500, lambda x: (static(x), None), self.K),
               params=sum(p.numel() for p in static.shared_parameters()), note="static backbone checkpoint")
        record("B0_bank_zero_path", eval_with_policy(bank, data, idx, 500, lambda x: (bank(x), None), self.K),
               note="(0,0) path of the refined bank model")
        if self.p("ckpt/b1.pt").exists():
            m1 = torch.load(self.p("ckpt/b1.pt"), map_location=dev, weights_only=False)["meta"]
            b1 = self.load_model("ckpt/b1.pt", site_ranks=tuple(m1["site_ranks"]), num_experts=1)
            a11 = torch.tensor([1, 1], device=dev)
            record("B1_capacity", eval_with_policy(b1, data, idx, 500, lambda x: (b1(x, a11.repeat(x.size(0), 1)), None), self.K),
                   params=sum(p.numel() for p in b1.parameters()), extra_params=m1["extra_params"],
                   epochs=m1["epochs"])
        pa = torch.tensor(pair, device=dev)
        record("B2_best_fixed", eval_with_policy(bank, data, idx, 500, lambda x: (bank(x, pa.repeat(x.size(0), 1)), pa.repeat(x.size(0), 1)), self.K),
               pair=list(pair), selected_on="route split")
        for mode, key in (("soft", "B3_soft_dynamic"), ("hard", "B4_hard_routing")):
            r = self.load_router(f"ckpt/router_{mode}.pt")

            def pol(x, r=r):
                z, a, _ = r(bank, x)
                return z, (a if mode == "hard" else None)
            ev = eval_with_policy(bank, data, idx, 500, pol, self.K)
            extra = {"router_params": sum(p.numel() for p in r.parameters())}
            if mode == "hard":
                tb = evaluate_decisions(T, ev["_actions"])
                extra["table"] = {k: tb[k] for k in ("accuracy", "mean_loss", "regret_mean", "regret_p90")}
            record(key, ev, **extra)

        ctrl_map = {"B5_additive": "additive", "B5_additive_wide": "additive_wide", "B6_sequential": "sequential",
                    "B6_sequential_wide": "sequential_wide", "P1_joint": "joint", "P2_bilinear": "bilinear"}
        table_agree = {}
        for key, name in ctrl_map.items():
            rel = f"ckpt/ctrl_{name}.pt"
            if not self.p(rel).exists():
                continue
            ctrl = self.load_ctrl(rel)
            ev = eval_with_policy(bank, data, idx, 500, lambda x, c=ctrl: deploy_forward(bank, c, x, dk), self.K)
            q, a = predict_table(ctrl, T, dev)
            tm = controller_metrics(q, T, a)
            agree = bool(torch.equal(a, ev["_actions"])) and abs(tm["accuracy"] - ev["accuracy"]) < 1e-6
            table_agree[key] = agree
            record(key, ev, params=sum(p.numel() for p in ctrl.parameters()),
                   table={k: tm[k] for k in ("regret_mean", "regret_p90", "regret_max", "top1_matches_loss_oracle",
                                             "spearman_mean", "centered_abs_err_mean", "centered_abs_err_p99",
                                             "uniform_err_delta_mean", "prop1_bound_holds_frac")})

        # D0 diagnostics: label-informed, never ranked with deployable methods
        Lf, Cf = T["losses"].flatten(1), T["correct"].flatten(1)
        diag = {"D0_loss_oracle": {"mean_loss": float(Lf.min(1).values.mean()),
                                   "accuracy": float(Cf.gather(1, Lf.argmin(1)[:, None]).float().mean() * 100)},
                "D0_accuracy_oracle": {"accuracy": float(Cf.any(1).float().mean() * 100)},
                "note": "label-informed, nondeployable diagnostics"}

        # recovery (Eq. 26) vs ordinary hard routing; flag if soft mixture beats the discrete oracle
        rec = recovery(res["B4_hard_routing"]["mean_loss"], res["P1_joint"]["mean_loss"],
                       diag["D0_loss_oracle"]["mean_loss"])
        soft_beats = res["B3_soft_dynamic"]["mean_loss"] < diag["D0_loss_oracle"]["mean_loss"]
        diag["recovery_P1_vs_B4"] = rec
        diag["soft_mixture_beats_discrete_oracle"] = soft_beats
        if soft_beats:
            diag["recovery_note"] = "B3 beats every discrete action: recovery comparison inappropriate, reconsider bank"

        interventions = intervention_controls(T, raw["P1_joint"]["_actions"], pair, self.seed)

        deployable = {k: v["accuracy"] for k, v in res.items()}
        strongest = sorted(deployable, key=lambda k: -deployable[k])[:3]
        conf = {k: confusion_counts(raw[k]["_preds"], raw[k]["_labels"], self.K) for k in strongest}
        overlap = corrected_overlap({k: raw[k]["_correct"] for k in strongest}, raw["B0_bank_zero_path"]["_correct"])

        e = cfg.eval
        simpler = [k for k in ("B4_hard_routing", "B5_additive", "B5_additive_wide", "B6_sequential",
                               "B6_sequential_wide") if k in res]
        best_simpler = max(simpler, key=lambda k: res[k]["accuracy"])
        gates = {
            "shared_path": {"value": res["B0_static"]["accuracy"], "reference": e.reference_static_acc,
                            "tolerance_pp": e.reference_tolerance_pp,
                            "pass": None if e.reference_static_acc is None else
                            abs(res["B0_static"]["accuracy"] - e.reference_static_acc) <= e.reference_tolerance_pp,
                            "note": "set eval.reference_static_acc to your reproduced baseline to evaluate"},
            "predictability": {"P1": res["P1_joint"]["accuracy"], "B2": res["B2_best_fixed"]["accuracy"],
                               "pass": res["P1_joint"]["accuracy"] > res["B2_best_fixed"]["accuracy"],
                               "note": "must hold across all screening seeds (see aggregate)"},
            "coordination": {"strongest_simpler": best_simpler,
                             "advantage_pp": res["P1_joint"]["accuracy"] - res[best_simpler]["accuracy"],
                             "criterion_pp": e.coordination_gate_pp,
                             "pass_this_seed": res["P1_joint"]["accuracy"] - res[best_simpler]["accuracy"] >= e.coordination_gate_pp,
                             "note": "final decision needs the mean over finalist seeds and a positive paired interval"},
        }
        hr = self.p("metrics/headroom_r0.json")
        if hr.exists() and split == "val":
            gates["headroom"] = read_json(hr)["gate_headroom"]
        return {"split": split, "methods": res, "diagnostics": diag, "interventions_P1": interventions,
                "confusion_strongest": conf, "corrected_overlap": overlap, "gates": gates,
                "real_vs_table_actions_agree": table_agree, "best_fixed_pair": list(pair)}

    def stage_evaluate(self) -> None:
        if not self.need("metrics/results.json"):
            return
        out = self._evaluate_split("val")
        write_json(self.p("metrics/results.json"), out)
        if not all(out["real_vs_table_actions_agree"].values()):
            raise RuntimeError(f"real execution disagrees with table lookup: {out['real_vs_table_actions_agree']}")
        if self.cfg.eval.final_test:
            write_json(self.p("metrics/results_test.json"), self._evaluate_split("test"))

    def stage_alternate(self) -> None:
        if not self.need("metrics/alternate.json"):
            return
        cfg, dev = self.cfg, self.device
        bank = self.load_model("ckpt/bank.pt")
        ctrl = self.load_ctrl("ckpt/ctrl_joint.pt")
        T_route = load_targets(self.p("targets/route_r0.pt"))
        val_data, val_idx = self.bundle.source("val")
        dk = cfg.controller.descriptor

        def save_cb(r, m, c, T_r, T_v):
            self.save_model(m, f"ckpt/bank_alt_r{r}.pt", {"round": r})
            self.save_ctrl(c, f"ckpt/ctrl_joint_alt_r{r}.pt", {"round": r})
            save_targets(T_r, self.p(f"targets/route_r{r}.pt"))
            save_targets(T_v, self.p(f"targets/val_r{r}.pt"))

        m_alt, c_alt, info = run_alternation(cfg, bank, ctrl, "joint", self.bundle, T_route, dev, self.seed,
                                             self.logger, save_cb=save_cb)
        ev = eval_with_policy(m_alt, val_data, val_idx, 500, lambda x: deploy_forward(m_alt, c_alt, x, dk), self.K)
        out = {"P1_alternated": strip_private(ev), **info}
        extra_epochs = cfg.alternate.rounds * cfg.alternate.bank_epochs
        if cfg.alternate.baselines and extra_epochs > 0:
            b4_model = self.load_model("ckpt/bank.pt")
            r = self.load_router("ckpt/router_hard.pt")
            r, rinfo = train_router("hard", cfg, b4_model, self.bundle, dev, self.seed + 7, self.logger,
                                    finetune_bank=True, epochs=extra_epochs, split="bank", router=r)
            evb = eval_with_policy(b4_model, val_data, val_idx, 500, lambda x: r(b4_model, x)[:2], self.K)
            out["B4_extra_budget"] = {**strip_private(evb), "extra_epochs": extra_epochs}
            st = self.load_model("ckpt/static.pt")
            train_static(cfg, st, self.bundle, dev, self.seed + 9, self.logger, tag="static_extra",
                                 epochs=extra_epochs, lr=cfg.alternate.corr_lr, warmup=0)
            evs = eval_with_policy(st, val_data, val_idx, 500, lambda x: (st(x), None), self.K)
            out["B0_extra_budget"] = {**strip_private(evs), "extra_epochs": extra_epochs,
                                      "lr": cfg.alternate.corr_lr}
            out["budget_note"] = "fixed bank-optimizer-epoch comparison; P1 additionally spends target collection"
        write_json(self.p("metrics/alternate.json"), out)
        if self.p("metrics/results.json").exists():
            res = read_json(self.p("metrics/results.json"))
            for k in ("P1_alternated", "B4_extra_budget", "B0_extra_budget"):
                if k in out:
                    res["methods"][k] = out[k]
            write_json(self.p("metrics/results.json"), res)

    def stage_profile(self) -> None:
        if not self.need("metrics/profile.json"):
            return
        cfg, dev = self.cfg, self.device
        bank = self.load_model("ckpt/bank.pt")
        ctrls = {}
        for name in ("joint", "additive", "bilinear", "sequential", "additive_wide", "sequential_wide"):
            if self.p(f"ckpt/ctrl_{name}.pt").exists():
                ctrls[name] = self.load_ctrl(f"ckpt/ctrl_{name}.pt")
        joint = ctrls["joint"]
        dk = cfg.controller.descriptor
        rs = self.load_router("ckpt/router_soft.pt")
        dm_grouped = DeployModel(bank, joint, dk, self.bundle.mean, self.bundle.std, "grouped").to(dev)
        dm_dense = DeployModel(bank, joint, dk, self.bundle.mean, self.bundle.std, "dense").to(dev)
        fns = {
            "static_path": lambda x: bank(x),
            "P1_grouped_deploy": lambda x: dm_grouped(x),
            "P1_dense_export_path": lambda x: dm_dense(x),
            "B3_soft_dynamic": lambda x: rs(bank, x),
        }
        pc = cfg.profile
        lat: Dict[str, Any] = {}
        for name, fn in fns.items():
            lat[name] = [measure_latency(fn, dev, bs, max(pc.latency_images, bs), pc.warmup_iters,
                                         precision=pc.precision) for bs in pc.batch_sizes]
        T_route = load_targets(self.p("targets/route_r0.pt"))
        out = {"params": param_report(bank, {**ctrls, "router_soft": rs}),
               "macs": macs_report(bank, joint, dev),
               "latency": lat,
               "target_collection": {**target_collection_cost(bank.n_choices),
                                     "route_r0_seconds": T_route["meta"]["seconds"],
                                     "route_r0_images_per_second": T_route["meta"]["images_per_second"],
                                     "route_r0_gpu_hours": T_route["meta"]["gpu_hours"]},
               "stage_seconds": self.timings,
               "total_device_hours": sum(self.timings.values()) / 3600.0,
               "versions": software_versions()}
        write_json(self.p("metrics/profile.json"), out)

    def stage_export(self) -> None:
        if not self.need("deploy/export_check.json"):
            return
        cfg = self.cfg
        bank = self.load_model("ckpt/bank.pt").cpu()
        ctrl = self.load_ctrl("ckpt/ctrl_joint.pt").cpu()
        dk = cfg.controller.descriptor
        save_bundle(self.p("deploy/bundle.pt"), cfg, bank, ctrl, "joint", controller_build_spec(ctrl),
                    self.bundle.mean, self.bundle.std, self.bundle.class_names,
                    extra={"run_id": self.run_id, "bank_hash": self.ckpt_hash("ckpt/bank.pt")})
        dm = DeployModel(bank, ctrl, dk, self.bundle.mean, self.bundle.std, "grouped").eval()
        export_onnx(dm, self.p("deploy/model.onnx"))
        export_torchscript(dm, self.p("deploy/model_ts.pt"))
        check = verify_exports(dm, self.p("deploy/model.onnx"), self.p("deploy/model_ts.pt"))
        rounds = sorted(self.dir.glob("ckpt/bank_alt_r*.pt"))
        if rounds:
            r = rounds[-1].stem.split("_r")[-1]
            ba = self.load_model(f"ckpt/bank_alt_r{r}.pt").cpu()
            ca = self.load_ctrl(f"ckpt/ctrl_joint_alt_r{r}.pt").cpu()
            save_bundle(self.p("deploy/bundle_alternated.pt"), cfg, ba, ca, "joint", controller_build_spec(ca),
                        self.bundle.mean, self.bundle.std, self.bundle.class_names,
                        extra={"run_id": self.run_id, "round": int(r)})
        write_json(self.p("deploy/export_check.json"), check)
        bad = [k for k, v in check.items() if (k.endswith("max_abs") and v > 1e-4) or (k.endswith("equal") and not v)]
        if bad:
            raise RuntimeError(f"export mismatch: {bad} {check}")

    # ------------------------------------------------------------- driver
    def run(self, stages: List[str]) -> None:
        for s in stages:
            fn = getattr(self, f"stage_{s}")
            print(f"=== stage {s} ({self.run_id}) ===", flush=True)
            self.time_stage(s, fn)


def parse_stages(text: str) -> List[str]:
    if text == "all":
        return list(STAGES)
    st = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in st if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    return [s for s in STAGES if s in st]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stages", default="all", help=f"'all' or comma list of {STAGES}")
    ap.add_argument("--force", action="store_true", help="recompute stages even if outputs exist")
    ap.add_argument("--override", nargs="*", default=[], help="section.key=value overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override)
    Run(cfg, args.seed, args.force).run(parse_stages(args.stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
