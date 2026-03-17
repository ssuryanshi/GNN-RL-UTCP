"""
sa_baseline.py
--------------
Simulated Annealing baseline.

Hyperparameters (from blueprint):
  T_init   = 1000
  T_min    = 0.1
  cooling  = 0.995

Encoding: assignment dict {course_id: {slot_id, room_id}}
Neighbour: randomly reassign one course to a new (slot, room) pair.
"""

from __future__ import annotations
import random
import time
import math
import json


def _random_assignment(instance: dict, rng: random.Random) -> dict[str, dict]:
    """Create a random initial assignment."""
    slot_ids = [s["id"] for s in instance["timeslots"]]
    room_ids = [r["id"] for r in instance["rooms"]]
    return {
        c["id"]: {
            "slot_id": rng.choice(slot_ids),
            "room_id": rng.choice(room_ids),
        }
        for c in instance["courses"]
    }


def _neighbour(assignment: dict[str, dict], instance: dict,
               rng: random.Random) -> dict[str, dict]:
    """Generate a neighbour by reassigning a random course to a random (slot, room)."""
    slot_ids = [s["id"] for s in instance["timeslots"]]
    room_ids = [r["id"] for r in instance["rooms"]]
    new_asgn = dict(assignment)  # shallow copy (values are also dicts so copy them)
    cid = rng.choice(list(new_asgn.keys()))
    new_asgn[cid] = {
        "slot_id": rng.choice(slot_ids),
        "room_id": rng.choice(room_ids),
    }
    return new_asgn


def _cost(assignment: dict[str, dict], instance: dict) -> float:
    """Total cost = 1000 × hard_violations + soft_penalty (lower = better)."""
    from timetable_env import _check_hard_violations, _check_soft_penalties
    hard = _check_hard_violations(assignment, instance)
    soft = _check_soft_penalties(assignment, instance)
    return 1000.0 * hard + soft


class SABaseline:
    """
    Simulated Annealing solver for UTCP.

    Parameters
    ----------
    T_init   : initial temperature
    T_min    : stopping temperature
    cooling  : multiplicative cooling rate
    seed     : random seed
    """

    def __init__(
        self,
        T_init:  float = 1000.0,
        T_min:   float = 0.1,
        cooling: float = 0.995,
        seed:    int   = 42,
    ):
        self.T_init  = T_init
        self.T_min   = T_min
        self.cooling = cooling
        self.seed    = seed

    def solve(self, instance: dict) -> dict:
        """
        Solve the timetabling instance.

        Returns
        -------
        dict with keys: assignment, hard_violations, soft_penalty,
                        runtime_sec, iterations, best_cost
        """
        rng   = random.Random(self.seed)
        start = time.time()

        current = _random_assignment(instance, rng)
        current_cost = _cost(current, instance)
        best, best_cost = current, current_cost
        T = self.T_init
        iters = 0
        ttfs: float | None = None  # time to first feasible solution

        while T > self.T_min:
            candidate      = _neighbour(current, instance, rng)
            candidate_cost = _cost(candidate, instance)
            delta = candidate_cost - current_cost

            if delta < 0 or rng.random() < math.exp(-delta / T):
                current, current_cost = candidate, candidate_cost

            if current_cost < best_cost:
                best, best_cost = current, current_cost

            # Check first feasible
            from timetable_env import _check_hard_violations
            if ttfs is None and _check_hard_violations(best, instance) == 0:
                ttfs = time.time() - start

            T     *= self.cooling
            iters += 1

        elapsed = time.time() - start
        from timetable_env import _check_hard_violations, _check_soft_penalties
        hard = _check_hard_violations(best, instance)
        soft = _check_soft_penalties(best, instance)

        return {
            "assignment":     best,
            "hard_violations": hard,
            "soft_penalty":   soft,
            "runtime_sec":    elapsed,
            "iterations":     iters,
            "best_cost":      best_cost,
            "ttfs":           ttfs,
        }


if __name__ == "__main__":
    import pathlib
    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if sample.exists():
        with open(sample) as f:
            inst = json.load(f)
        sa = SABaseline(T_init=1000, T_min=0.1, cooling=0.995)
        result = sa.solve(inst)
        print(f"SA  ->  hard={result['hard_violations']}, "
              f"soft={result['soft_penalty']:.1f}, "
              f"time={result['runtime_sec']:.1f}s, "
              f"iters={result['iterations']}")
