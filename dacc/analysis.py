"""Functional diagnostics on 25-action loss tables (Sections 11-12).

Everything here operates on a target table collected from a frozen
checkpoint: ``losses`` (N, n, n) float32, ``correct`` (N, n, n) bool.
Label-informed quantities (oracles) are diagnostic only and are always
reported under keys containing ``oracle`` so they never enter a ranking of
deployable methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(1)


def gather_actions(table: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """table (N, n, n), actions (N, 2) -> (N,) values at the chosen actions."""
    n = table.shape[-1]
    a = actions[:, 0] * n + actions[:, 1]
    return _flat(table).gather(1, a[:, None].long()).squeeze(1)


# ------------------------------------------------------------------ decisions
def evaluate_decisions(T: Dict[str, Any], actions: torch.Tensor) -> Dict[str, Any]:
    L, C = T["losses"], T["correct"]
    n = L.shape[-1]
    actions = actions.cpu().long()
    loss = gather_actions(L, actions)
    corr = gather_actions(C.float(), actions).bool()
    oracle = _flat(L).min(1).values
    base_c = C[:, 0, 0]
    usage = torch.bincount(actions[:, 0] * n + actions[:, 1], minlength=n * n).view(n, n)
    return {
        "accuracy": float(corr.float().mean() * 100),
        "mean_loss": float(loss.mean()),
        "regret_mean": float((loss - oracle).mean()),
        "regret_p90": float(torch.quantile(loss - oracle, 0.9)),
        "regret_max": float((loss - oracle).max()),
        "corrected_vs_static": int((corr & ~base_c).sum()),
        "damaged_vs_static": int((~corr & base_c).sum()),
        "n": int(len(loss)),
        "usage": usage.tolist(),
        "frac_static_action": float(((actions[:, 0] == 0) & (actions[:, 1] == 0)).float().mean()),
    }


def best_fixed_pair(T_select: Dict[str, Any]) -> Tuple[int, int]:
    """Best globally fixed action chosen on a labelled selection table.

    Primary criterion: accuracy; tie-break: lower mean loss; then lexicographic.
    """
    L, C = T_select["losses"], T_select["correct"]
    n = L.shape[-1]
    acc = _flat(C.float()).mean(0)
    loss = _flat(L).mean(0)
    order = sorted(range(n * n), key=lambda a: (-round(float(acc[a]), 9), float(loss[a]), a))
    return divmod(order[0], n)


def fixed_actions(N: int, pair: Tuple[int, int]) -> torch.Tensor:
    return torch.tensor(pair, dtype=torch.long).repeat(N, 1)


def uniform_random_actions(N: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, n, (N, 2), generator=g)


def shuffled_actions(actions: torch.Tensor, seed: int) -> torch.Tensor:
    """Decisions shuffled between images; preserves the route-frequency distribution."""
    g = torch.Generator().manual_seed(seed)
    return actions[torch.randperm(len(actions), generator=g)]


# ------------------------------------------------------------------ headroom
def headroom_report(T_val: Dict[str, Any], best_pair: Tuple[int, int],
                    coverage_thresholds: Sequence[float] = (0.0, 0.01, 0.05, 0.1)) -> Dict[str, Any]:
    L, C = T_val["losses"], T_val["correct"]
    N, n, _ = L.shape
    Lf, Cf = _flat(L), _flat(C)
    static_c = C[:, 0, 0]
    bi, bj = best_pair
    rep: Dict[str, Any] = {}
    rep["static_accuracy"] = float(static_c.float().mean() * 100)
    rep["static_mean_loss"] = float(L[:, 0, 0].mean())
    rep["best_fixed_pair"] = [bi, bj]
    rep["best_fixed_accuracy"] = float(C[:, bi, bj].float().mean() * 100)
    rep["best_fixed_mean_loss"] = float(L[:, bi, bj].mean())
    rep["per_action_accuracy"] = (Cf.float().mean(0).view(n, n) * 100).tolist()
    rep["per_action_mean_loss"] = Lf.mean(0).view(n, n).tolist()

    # oracles (diagnostic, label-informed, non-deployable)
    a_loss = Lf.argmin(1)
    rep["loss_oracle_mean_loss"] = float(Lf.min(1).values.mean())
    rep["loss_oracle_accuracy"] = float(Cf.gather(1, a_loss[:, None]).float().mean() * 100)
    acc_oracle = Cf.any(1)
    rep["accuracy_oracle_accuracy"] = float(acc_oracle.float().mean() * 100)
    rep["oracle_headroom_pp_vs_best_fixed"] = rep["accuracy_oracle_accuracy"] - rep["best_fixed_accuracy"]
    rep["oracle_headroom_pp_vs_static"] = rep["accuracy_oracle_accuracy"] - rep["static_accuracy"]
    wrong = ~static_c
    rep["frac_static_errors_correctable"] = float((Cf[wrong].any(1)).float().mean()) if wrong.any() else 0.0
    rep["frac_static_correct_damageable"] = float((~Cf[static_c]).any(1).float().mean()) if static_c.any() else 0.0
    rep["count_static_errors"] = int(wrong.sum())
    rep["count_static_errors_correctable"] = int(Cf[wrong].any(1).sum()) if wrong.any() else 0

    # utilities (Eq. 8) and interactions (Eq. 9)
    U = L[:, :1, :1] - L                                              # U_ij = L00 - Lij
    I = L[:, :, :1] + L[:, :1, :] - L - L[:, :1, :1]                   # I_ij
    Iin = I[:, 1:, 1:]
    rep["utility_mean"] = U.mean(0).tolist()
    rep["utility_frac_positive"] = (U > 0).float().mean(0).tolist()
    rep["interaction_mean"] = I.mean(0).tolist()
    rep["interaction_abs_mean"] = float(Iin.abs().mean())
    rep["interaction_frac_positive"] = float((Iin > 0).float().mean())
    rep["interaction_std"] = float(Iin.std())

    # does the interaction change the optimal decision? (Section 11.2)
    S = L[:, :, :1] + L[:, :1, :] - L[:, :1, :1]                       # exact additive table
    a_add = _flat(S).argmin(1)
    rep["frac_decision_changed_by_interaction"] = float((a_add != a_loss).float().mean())
    rep["additive_exact_regret_mean"] = float((Lf.gather(1, a_add[:, None]).squeeze(1) - Lf.min(1).values).mean())
    rep["additive_exact_accuracy"] = float(Cf.gather(1, a_add[:, None]).float().mean() * 100)
    rep["max_abs_interaction"] = float(Iin.abs().max())

    # coverage (Section 12.1)
    Unz = _flat(U)[:, 1:]                                               # exclude (0,0)
    cov = {}
    for m in coverage_thresholds:
        material = (Unz > m)
        unique = material & (material.sum(1, keepdim=True) == 1)
        cov[str(m)] = {"material": material.float().mean(0).tolist(),
                       "unique": unique.float().mean(0).tolist(),
                       "any_material": float(material.any(1).float().mean())}
    rep["coverage"] = cov

    # benefit-profile redundancy, raw and centred per image
    def mean_offdiag_corr(M: torch.Tensor) -> float:
        M = M - M.mean(0, keepdim=True)
        sd = M.std(0).clamp_min(1e-12)
        R = (M.t() @ M) / (M.shape[0] - 1) / (sd[:, None] * sd[None, :])
        k = R.shape[0]
        off = R[~torch.eye(k, dtype=torch.bool)]
        return float(off.mean())

    rep["benefit_corr_raw_mean_offdiag"] = mean_offdiag_corr(Unz)
    rep["benefit_corr_centered_mean_offdiag"] = mean_offdiag_corr(Unz - Unz.mean(1, keepdim=True))

    # corrected-error-set agreement (Jaccard) between non-zero actions
    if wrong.any():
        S_err = Cf[wrong][:, 1:].float()                                 # (n_err, n*n-1)
        inter = S_err.t() @ S_err
        sizes = S_err.sum(0)
        union = sizes[:, None] + sizes[None, :] - inter
        jac = torch.where(union > 0, inter / union.clamp_min(1), torch.zeros_like(inter))
        k = jac.shape[0]
        rep["corrected_set_jaccard_mean_offdiag"] = float(jac[~torch.eye(k, dtype=torch.bool)].mean())
    else:
        rep["corrected_set_jaccard_mean_offdiag"] = 0.0

    # leave-one-expert-out change in best achievable loss
    base = Lf.min(1).values.mean()
    loo = {}
    for site, axis in (("A", 1), ("B", 2)):
        for e in range(1, n):
            mask = torch.ones(n, n, dtype=torch.bool)
            if axis == 1:
                mask[e, :] = False
            else:
                mask[:, e] = False
            m = L[:, mask].min(1).values.mean()
            loo[f"{site}{e}"] = float(m - base)
    rep["leave_one_expert_out_delta_best_loss"] = loo
    return rep


# ------------------------------------------------------------------ controllers
def ranks(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(1).argsort(1).float()


def controller_metrics(q: torch.Tensor, T: Dict[str, Any], actions: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Regret, ranking accuracy, tail errors and outcome counts for predictions q."""
    from .controller import decide

    L = T["losses"]
    q = q.detach().cpu().float()
    if actions is None:
        actions = decide(q)
    out = evaluate_decisions(T, actions)
    qf, Lf = _flat(q), _flat(L)
    out["top1_matches_loss_oracle"] = float((qf.argmin(1) == Lf.argmin(1)).float().mean())
    rq, rl = ranks(qf), ranks(Lf)
    rq = rq - rq.mean(1, keepdim=True)
    rl = rl - rl.mean(1, keepdim=True)
    sp = (rq * rl).sum(1) / (rq.norm(dim=1) * rl.norm(dim=1)).clamp_min(1e-12)
    out["spearman_mean"] = float(sp.mean())
    err = (qf - qf.mean(1, keepdim=True)) - (Lf - Lf.mean(1, keepdim=True))
    out["centered_abs_err_mean"] = float(err.abs().mean())
    out["centered_abs_err_p99"] = float(torch.quantile(err.abs().flatten(), 0.99))
    per_ex_max = err.abs().max(1).values
    out["uniform_err_delta_mean"] = float(per_ex_max.mean())
    # Proposition 1 check on centred scores: regret <= 2 * delta_x per example
    loss_sel = gather_actions(L, actions)
    reg = loss_sel - Lf.min(1).values
    out["prop1_bound_holds_frac"] = float((reg <= 2 * per_ex_max + 1e-5).float().mean())
    return out


def recovery(ordinary_loss: float, learned_loss: float, oracle_loss: float, eps: float = 1e-4) -> Optional[float]:
    """Eq. 26; None when the denominator is near zero (not reported)."""
    denom = ordinary_loss - oracle_loss
    if abs(denom) < eps:
        return None
    return (ordinary_loss - learned_loss) / denom


def intervention_controls(T: Dict[str, Any], learned_actions: torch.Tensor, best_pair: Tuple[int, int],
                          seed: int) -> Dict[str, Any]:
    N, n = T["losses"].shape[0], T["losses"].shape[-1]
    return {
        "learned": evaluate_decisions(T, learned_actions),
        "best_fixed": evaluate_decisions(T, fixed_actions(N, best_pair)),
        "uniform_random": evaluate_decisions(T, uniform_random_actions(N, n, seed)),
        "shuffled_between_images": evaluate_decisions(T, shuffled_actions(learned_actions, seed)),
    }


def confusion_counts(preds: torch.Tensor, labels: torch.Tensor, k: int) -> List[List[int]]:
    cm = torch.zeros(k, k, dtype=torch.long)
    cm.index_put_((labels.long(), preds.long()), torch.ones_like(labels, dtype=torch.long), accumulate=True)
    return cm.tolist()


def corrected_overlap(correct_sets: Dict[str, torch.Tensor], static_correct: torch.Tensor) -> Dict[str, Any]:
    """Counts of static errors corrected by each method and pairwise overlaps."""
    wrong = ~static_correct
    out: Dict[str, Any] = {"static_errors": int(wrong.sum())}
    names = list(correct_sets.keys())
    for a in names:
        out[f"corrected[{a}]"] = int((correct_sets[a] & wrong).sum())
        out[f"damaged[{a}]"] = int((~correct_sets[a] & static_correct).sum())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"both_corrected[{a}|{b}]"] = int((correct_sets[a] & correct_sets[b] & wrong).sum())
    return out
