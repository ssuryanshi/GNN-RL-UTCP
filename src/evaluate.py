"""
evaluate.py
-----------
Computes the 5 evaluation metrics from the blueprint:

  HCSR — Hard Constraint Satisfaction Rate
  SCP  — Soft Constraint Penalty
  TTFS — Time to First Feasible Solution
  SSD  — Solution Stability under Disruption
  SI   — Scalability Index

Usage:
    python src/evaluate.py \
        --model checkpoints/ppo_final.zip \
        --data data/indian_synthetic/ \
        --baselines all
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import time
import random
import numpy as np
from typing import Any


# ------------------------------------------------------------------ #
# Metric calculations                                                  #
# ------------------------------------------------------------------ #

def compute_hcsr(assignment: dict[str, dict], instance: dict) -> float:
    """
    Hard Constraint Satisfaction Rate.
    HCSR = 1 - (hard_violations / total_hard_constraint_slots)

    Parameters
    ----------
    assignment : {course_id: {slot_id, room_id}}
    instance   : full timetabling instance dict

    Returns
    -------
    float in [0, 1], higher is better
    """
    from timetable_env import _check_hard_violations
    n_constraints = len(instance["hard_constraints"]) * len(instance["courses"])
    violations    = _check_hard_violations(assignment, instance)
    if n_constraints == 0:
        return 1.0
    return max(0.0, 1.0 - violations / n_constraints)


def compute_scp(assignment: dict[str, dict], instance: dict) -> float:
    """
    Soft Constraint Penalty.
    SCP = Σ wᵢ × soft_violations  (lower is better)
    """
    from timetable_env import _check_soft_penalties
    return _check_soft_penalties(assignment, instance)


def compute_ttfs(solve_fn, instance: dict) -> float:
    """
    Time to First Feasible Solution (seconds).
    Calls solve_fn(instance) and records wall-clock time until
    first zero-hard-violation solution. Returns inf if not found.

    Parameters
    ----------
    solve_fn : callable -> dict with key 'ttfs' (or computes internally)
    instance : timetabling instance
    """
    result = solve_fn(instance)
    return result.get("ttfs") or float("inf")


def compute_ssd(
    assignment: dict[str, dict],
    instance: dict,
    n_disruptions: int = 1,
    seed: int = 42,
) -> float:
    """
    Solution Stability under Disruption.
    Apply n_disruptions random single-course reassignments, then re-evaluate HCSR.
    No retraining — measures robustness of the solution.

    Returns float in [0, 1], higher is better.
    """
    rng      = random.Random(seed)
    disrupted = dict(assignment)
    courses   = [c["id"] for c in instance["courses"]]
    slots     = [s["id"] for s in instance["timeslots"]]
    rooms     = [r["id"] for r in instance["rooms"]]

    for _ in range(n_disruptions):
        cid = rng.choice(courses)
        disrupted[cid] = {
            "slot_id": rng.choice(slots),
            "room_id": rng.choice(rooms),
        }

    return compute_hcsr(disrupted, instance)


def compute_si(
    results_small: dict,
    results_large: dict,
) -> float:
    """
    Scalability Index.
    SI = HCSR_large / HCSR_small  (aim close to 1.0)

    Parameters
    ----------
    results_small : eval output for smallest instance
    results_large : eval output for largest instance
    """
    hcsr_small = results_small.get("hcsr", 0.0)
    hcsr_large = results_large.get("hcsr", 0.0)
    if hcsr_small == 0:
        return 0.0
    return hcsr_large / hcsr_small


# ------------------------------------------------------------------ #
# Aggregated evaluation run                                            #
# ------------------------------------------------------------------ #

def evaluate_assignment(
    assignment: dict[str, dict],
    instance: dict,
    ttfs: float | None = None,
) -> dict[str, Any]:
    """
    Compute all 5 metrics for a given assignment.

    Returns
    -------
    dict with keys: hcsr, scp, ttfs, ssd  (SI computed across instances separately)
    """
    hcsr = compute_hcsr(assignment, instance)
    scp  = compute_scp(assignment, instance)
    ssd  = compute_ssd(assignment, instance)

    return {
        "hcsr":       round(hcsr, 4),
        "scp":        round(scp,  2),
        "ttfs":       round(ttfs, 3) if ttfs is not None else None,
        "ssd":        round(ssd,  4),
    }


def run_full_evaluation(
    instance: dict,
    model_path: str | None = None,
    run_baselines: bool = True,
    n_rollouts: int = 100,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Run full evaluation: GNN+RL model + all baseline solvers.

    Parameters
    ----------
    instance      : timetabling instance dict
    model_path    : path to saved MaskablePPO model
    run_baselines : whether to also run GA/SA/Tabu/CP baselines
    n_rollouts    : number of agent rollouts to average over
    seed          : random seed

    Returns
    -------
    dict[solver_name → metric_dict]
    """
    results: dict[str, dict] = {}

    # GNN+RL Agent
    if model_path and pathlib.Path(model_path).exists():
        from sb3_contrib import MaskablePPO
        from ppo_agent import make_masked_env
        env   = make_masked_env(instance)
        agent = MaskablePPO.load(model_path, env=env)

        rollout_scores = []
        for i in range(n_rollouts):
            obs, _ = env.reset()
            done = trunc = False
            start = time.time()
            ttfs_i = None
            while not (done or trunc):
                action, _ = agent.predict(obs, action_masks=env.action_masks())
                obs, rew, done, trunc, info = env.step(int(action))
                if ttfs_i is None and info.get("hard_violations", 1) == 0:
                    ttfs_i = time.time() - start
            # Get final assignment from env
            asgn = env.env.assignment  # unwrap ActionMasker
            metrics = evaluate_assignment(asgn, instance, ttfs=ttfs_i)
            rollout_scores.append(metrics)

        # Average across rollouts
        results["GNN+RL"] = {
            k: round(float(np.mean([r[k] for r in rollout_scores if r[k] is not None])), 4)
            for k in ["hcsr", "scp", "ssd"]
        }
        ttfs_vals = [r["ttfs"] for r in rollout_scores if r["ttfs"] is not None]
        results["GNN+RL"]["ttfs"] = round(float(np.mean(ttfs_vals)), 3) if ttfs_vals else None

    if run_baselines:
        from baselines.ga_baseline    import GABaseline
        from baselines.sa_baseline    import SABaseline
        from baselines.tabu_baseline  import TabuBaseline
        from baselines.cp_baseline    import CPBaseline

        for name, solver in [
            ("GA",   GABaseline(population_size=200, generations=500, seed=seed)),
            ("SA",   SABaseline(T_init=1000, T_min=0.1, cooling=0.995, seed=seed)),
            ("Tabu", TabuBaseline(tabu_tenure=20, max_iter=10_000, seed=seed)),
            ("CP",   CPBaseline(time_limit_sec=300.0, seed=seed)),
        ]:
            print(f"Running {name} baseline...")
            res  = solver.solve(instance)
            asgn = res.get("assignment", {})
            metrics = evaluate_assignment(asgn, instance, ttfs=res.get("ttfs"))
            metrics["runtime_sec"] = round(res.get("runtime_sec", 0.0), 2)
            results[name] = metrics

    return results


def print_results_table(results: dict[str, dict]):
    """Pretty-print a comparison table."""
    col_w = 14
    header = f"{'Solver':<12}" + "".join(f"{m:>{col_w}}" for m in ["HCSR↑", "SCP↓", "TTFS↓(s)", "SSD↑"])
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for solver, metrics in results.items():
        row = f"{solver:<12}"
        for m in ["hcsr", "scp", "ttfs", "ssd"]:
            val = metrics.get(m)
            row += f"{str(val) if val is not None else 'N/A':>{col_w}}"
        print(row)
    print("=" * len(header))


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN+RL vs baselines")
    parser.add_argument("--model",      type=str, default=None,
                        help="Path to saved MaskablePPO model (.zip)")
    parser.add_argument("--data",       type=str,
                        default="data/indian_synthetic/",
                        help="Path to instance JSON or directory of instances")
    parser.add_argument("--baselines",  choices=["all", "none"], default="all")
    parser.add_argument("--rollouts",   type=int, default=100)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--save_csv",   type=str, default=None,
                        help="Save results to CSV path")
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    json_files = ([data_path] if data_path.is_file()
                  else sorted(data_path.glob("*.json"))[:5])

    all_results = {}
    for jf in json_files:
        print(f"\n--- Evaluating on: {jf.name} ---")
        with open(jf) as f:
            instance = json.load(f)
        res = run_full_evaluation(
            instance,
            model_path=args.model,
            run_baselines=(args.baselines == "all"),
            n_rollouts=args.rollouts,
            seed=args.seed,
        )
        all_results[jf.name] = res
        print_results_table(res)

    if args.save_csv:
        import csv
        rows = []
        for fname, res in all_results.items():
            for solver, metrics in res.items():
                rows.append({"instance": fname, "solver": solver, **metrics})
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved → {args.save_csv}")


if __name__ == "__main__":
    main()
