"""
cp_baseline.py
--------------
Constraint Programming baseline using Google OR-Tools CP-SAT solver.

Encodes all hard constraints H1-H8 as CP-SAT constraints.
Reports time to first feasible solution and final hard/soft violations.
"""

from __future__ import annotations
import time
import json


try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: OR-Tools not installed. Install with: pip install ortools")


class CPBaseline:
    """
    CP-SAT solver for UTCP.

    Parameters
    ----------
    time_limit_sec : float  — wall-clock time limit for the solver
    seed           : int    — solver random seed
    """

    def __init__(self, time_limit_sec: float = 300.0, seed: int = 42):
        assert ORTOOLS_AVAILABLE, "Install OR-Tools: pip install ortools"
        self.time_limit = time_limit_sec
        self.seed       = seed

    def solve(self, instance: dict) -> dict:
        """
        Solve the timetabling instance using CP-SAT.

        Returns
        -------
        dict with keys: assignment, hard_violations, soft_penalty,
                        runtime_sec, ttfs, cp_status
        """
        model = cp_model.CpModel()
        courses  = instance["courses"]
        slots    = instance["timeslots"]
        rooms    = instance["rooms"]
        faculty  = {f["id"]: f for f in instance["faculty"]}

        n_c = len(courses)
        n_s = len(slots)
        n_r = len(rooms)

        slot_ids   = [s["id"] for s in slots]
        room_ids   = [r["id"] for r in rooms]
        course_ids = [c["id"] for c in courses]

        slot_idx = {s["id"]: i for i, s in enumerate(slots)}
        room_idx = {r["id"]: i for i, r in enumerate(rooms)}

        # Decision variables: x[c][s][r] = 1 iff course c is in slot s, room r
        x = {}
        for ci, c in enumerate(courses):
            for si in range(n_s):
                for ri in range(n_r):
                    x[ci, si, ri] = model.NewBoolVar(f"x_c{ci}_s{si}_r{ri}")

        # Each course must be assigned exactly once
        for ci in range(n_c):
            model.AddExactlyOne(
                x[ci, si, ri] for si in range(n_s) for ri in range(n_r)
            )

        # H1: No room double-booking at same slot
        for si in range(n_s):
            for ri in range(n_r):
                model.AddAtMostOne(x[ci, si, ri] for ci in range(n_c))

        # H2: Faculty double-booking — same faculty in same slot
        faculty_courses: dict[str, list[int]] = {}
        for ci, c in enumerate(courses):
            fid = c.get("faculty_id", "")
            faculty_courses.setdefault(fid, []).append(ci)

        for fid, cis in faculty_courses.items():
            if len(cis) > 1:
                for si in range(n_s):
                    model.AddAtMostOne(
                        x[ci, si, ri] for ci in cis for ri in range(n_r)
                    )

        # H3: Student group double-booking
        group_courses: dict[str, list[int]] = {}
        for ci, c in enumerate(courses):
            gid = c.get("group_id", "")
            group_courses.setdefault(gid, []).append(ci)

        for gid, cis in group_courses.items():
            if len(cis) > 1:
                for si in range(n_s):
                    model.AddAtMostOne(
                        x[ci, si, ri] for ci in cis for ri in range(n_r)
                    )

        # H5: Room type must match course requirement
        for ci, c in enumerate(courses):
            ctype = c.get("required_room_type", "lecture")
            forbidden_rooms = [ri for ri, r in enumerate(rooms) if r["type"] != ctype]
            for si in range(n_s):
                for ri in forbidden_rooms:
                    model.Add(x[ci, si, ri] == 0)

        # H4: Capacity — coarse filter, exclude clearly too-small rooms
        for ci, c in enumerate(courses):
            enrol = c.get("enrolment", 0)
            too_small = [ri for ri, r in enumerate(rooms) if r.get("capacity", 0) < enrol]
            for si in range(n_s):
                for ri in too_small:
                    model.Add(x[ci, si, ri] == 0)

        # H6: Faculty unavailability
        fac_unavail: dict[str, set] = {
            f["id"]: set(f.get("unavailable_slots", []))
            for f in instance["faculty"]
        }
        slot_labels = {s["id"]: s.get("label", "") for s in slots}

        for ci, c in enumerate(courses):
            fid   = c.get("faculty_id", "")
            unavail = fac_unavail.get(fid, set())
            for si, s in enumerate(slots):
                if slot_labels[s["id"]] in unavail:
                    for ri in range(n_r):
                        model.Add(x[ci, si, ri] == 0)

        # H7: Medium compliance — only allow slots where faculty medium matches
        for ci, c in enumerate(courses):
            fid  = c.get("faculty_id", "")
            f    = faculty.get(fid, {})
            if f.get("medium") and f["medium"] != c.get("medium"):
                # Entire course is infeasible — block all assignments
                for si in range(n_s):
                    for ri in range(n_r):
                        model.Add(x[ci, si, ri] == 0)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.random_seed = self.seed

        ttfs_ref = [None]
        start = time.time()

        class SolutionCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                super().__init__()
                self._solutions = 0

            def on_solution_callback(self):
                if ttfs_ref[0] is None:
                    ttfs_ref[0] = time.time() - start
                self._solutions += 1

        cb     = SolutionCallback()
        status = solver.Solve(model, cb)
        elapsed = time.time() - start

        assignment: dict[str, dict] = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for ci, c in enumerate(courses):
                for si in range(n_s):
                    for ri in range(n_r):
                        if solver.BooleanValue(x[ci, si, ri]):
                            assignment[c["id"]] = {
                                "slot_id": slot_ids[si],
                                "room_id": room_ids[ri],
                            }
                            break

        status_str = {
            cp_model.OPTIMAL:   "OPTIMAL",
            cp_model.FEASIBLE:  "FEASIBLE",
            cp_model.INFEASIBLE:"INFEASIBLE",
            cp_model.UNKNOWN:   "UNKNOWN",
        }.get(status, "UNKNOWN")

        from timetable_env import _check_hard_violations, _check_soft_penalties
        hard = _check_hard_violations(assignment, instance) if assignment else 999
        soft = _check_soft_penalties(assignment, instance) if assignment else 999.0

        return {
            "assignment":      assignment,
            "hard_violations": hard,
            "soft_penalty":    soft,
            "runtime_sec":     elapsed,
            "ttfs":            ttfs_ref[0],
            "cp_status":       status_str,
        }


if __name__ == "__main__":
    import pathlib
    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if sample.exists():
        with open(sample) as f:
            inst = json.load(f)
        cp = CPBaseline(time_limit_sec=30.0)
        result = cp.solve(inst)
        print(f"CP  ->  hard={result['hard_violations']}, "
              f"soft={result['soft_penalty']:.1f}, "
              f"time={result['runtime_sec']:.1f}s, "
              f"status={result['cp_status']}")
