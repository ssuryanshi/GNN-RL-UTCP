"""
tabu_baseline.py
----------------
Tabu Search baseline.

Hyperparameters (from blueprint):
  tabu_tenure = 20
  max_iter    = 10000

Encoding: assignment dict {course_id: {slot_id, room_id}}
Tabu list: set of (course_id, slot_id, room_id) forbidden moves.
"""

from __future__ import annotations
import random
import time
from collections import deque
import json


def _cost(assignment: dict[str, dict], instance: dict) -> float:
    from timetable_env import _check_hard_violations, _check_soft_penalties
    return 1000.0 * _check_hard_violations(assignment, instance) + \
           _check_soft_penalties(assignment, instance)


class TabuBaseline:
    """
    Tabu Search solver for UTCP.

    Parameters
    ----------
    tabu_tenure : int    — how many iterations a move stays forbidden
    max_iter    : int    — maximum number of iterations
    seed        : int
    """

    def __init__(
        self,
        tabu_tenure: int = 20,
        max_iter:    int = 10_000,
        seed:        int = 42,
    ):
        self.tabu_tenure = tabu_tenure
        self.max_iter    = max_iter
        self.seed        = seed

    def solve(self, instance: dict) -> dict:
        rng = random.Random(self.seed)
        slot_ids = [s["id"] for s in instance["timeslots"]]
        room_ids = [r["id"] for r in instance["rooms"]]
        course_ids = [c["id"] for c in instance["courses"]]

        # Greedy initial solution
        from train import greedy_csp_solve
        current = greedy_csp_solve(instance, seed=self.seed)
        if not current:
            current = {
                cid: {"slot_id": rng.choice(slot_ids), "room_id": rng.choice(room_ids)}
                for cid in course_ids
            }

        current_cost = _cost(current, instance)
        best, best_cost = dict(current), current_cost
        tabu: deque = deque(maxlen=self.tabu_tenure)
        ttfs: float | None = None
        start = time.time()

        for iteration in range(self.max_iter):
            # Generate neighbourhood: reassign a random course to a new (slot, room)
            best_neighbor     = None
            best_neighbor_cost = float("inf")
            best_move         = None

            candidates = rng.sample(course_ids, k=min(10, len(course_ids)))
            for cid in candidates:
                new_sid = rng.choice(slot_ids)
                new_rid = rng.choice(room_ids)
                move = (cid, new_sid, new_rid)
                if move in tabu:
                    continue
                neighbor = dict(current)
                neighbor[cid] = {"slot_id": new_sid, "room_id": new_rid}
                cost = _cost(neighbor, instance)
                if cost < best_neighbor_cost:
                    best_neighbor_cost = cost
                    best_neighbor      = neighbor
                    best_move          = move

            if best_neighbor is None:
                continue

            current      = best_neighbor
            current_cost = best_neighbor_cost
            tabu.append(best_move)

            if current_cost < best_cost:
                best, best_cost = dict(current), current_cost
                from timetable_env import _check_hard_violations
                if ttfs is None and _check_hard_violations(best, instance) == 0:
                    ttfs = time.time() - start

        elapsed = time.time() - start
        from timetable_env import _check_hard_violations, _check_soft_penalties
        hard = _check_hard_violations(best, instance)
        soft = _check_soft_penalties(best, instance)

        return {
            "assignment":      best,
            "hard_violations": hard,
            "soft_penalty":    soft,
            "runtime_sec":     elapsed,
            "iterations":      self.max_iter,
            "best_cost":       best_cost,
            "ttfs":            ttfs,
        }


if __name__ == "__main__":
    import pathlib
    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if sample.exists():
        with open(sample) as f:
            inst = json.load(f)
        ts = TabuBaseline(tabu_tenure=20, max_iter=500)
        result = ts.solve(inst)
        print(f"TS  ->  hard={result['hard_violations']}, "
              f"soft={result['soft_penalty']:.1f}, "
              f"time={result['runtime_sec']:.1f}s")
