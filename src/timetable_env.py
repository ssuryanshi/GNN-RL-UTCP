"""
timetable_env.py
----------------
Gymnasium-compatible RL environment for the University Course Timetabling Problem.

State  : HeteroData (updated graph) + partial_assignment_matrix + violation_vector
Action : flat integer index decoded to (course_id, slot_id, room_id)
         Faculty is pre-assigned (from instance data) so action is 3-tuple.
         Action masking: hard-constraint-violating actions are flagged invalid.

Reward :
  r_hard       = -1000 × hard_violations_introduced
  r_soft       = +10   × soft_constraints_newly_satisfied
  r_progress   = +1    × (courses_assigned / total_courses)
  r_completion = +500  if all courses assigned with zero hard violations

Episode terminates when all courses are assigned OR max_steps exceeded.
max_steps = 3 × n_courses  (budget for backtracking and reassignment)
"""

from __future__ import annotations
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any

from graph_builder import build_hetero_graph

# ------------------------------------------------------------------ #
# Hard-constraint checkers (H1-H8)                                    #
# ------------------------------------------------------------------ #

def _check_hard_violations(
    assignment: dict[str, dict],  # course_id -> {slot_id, room_id}
    instance: dict,
) -> int:
    """Return total number of unique hard constraint violations in assignment."""
    violations = 0
    courses  = {c["id"]: c for c in instance["courses"]}
    rooms    = {r["id"]: r for r in instance["rooms"]}
    slots    = {s["id"]: s for s in instance["timeslots"]}
    faculty_map = {c["id"]: c.get("faculty_id") for c in instance["courses"]}

    slot_room_used: dict[tuple, str] = {}   # (slot_id, room_id) -> course_id  H1
    slot_faculty_used: dict[tuple, str] = {}  # (slot_id, faculty_id) -> course  H2
    slot_group_used: dict[tuple, str] = {}   # (slot_id, group_id) -> course   H3
    faculty_avail: dict[str, set] = {
        f["id"]: set(f.get("unavailable_slots", []))
        for f in instance["faculty"]
    }

    for cid, asgn in assignment.items():
        sid = asgn["slot_id"]
        rid = asgn["room_id"]
        fid = faculty_map.get(cid)
        c   = courses[cid]
        r   = rooms.get(rid, {})
        s   = slots.get(sid, {})

        # H1: Room double-booking
        key1 = (sid, rid)
        if key1 in slot_room_used:
            violations += 1
        else:
            slot_room_used[key1] = cid

        # H2: Faculty double-booking
        if fid:
            key2 = (sid, fid)
            if key2 in slot_faculty_used:
                violations += 1
            else:
                slot_faculty_used[key2] = cid

        # H3: Student group double-booking
        gid = c.get("group_id", "")
        key3 = (sid, gid)
        if key3 in slot_group_used:
            violations += 1
        else:
            slot_group_used[key3] = cid

        # H4: Room capacity
        if r.get("capacity", 999) < c.get("enrolment", 0):
            violations += 1

        # H5: Room type match
        if r.get("type") != c.get("required_room_type"):
            violations += 1

        # H6: Faculty availability
        if fid and s.get("label") in faculty_avail.get(fid, set()):
            violations += 1

        # H7: Medium compliance
        faculty_medium = next(
            (f["medium"] for f in instance["faculty"] if f["id"] == fid), None
        )
        if faculty_medium and faculty_medium != c.get("medium"):
            violations += 1

    # H8: Lab batching — lab courses must occupy consecutive slots
    slot_label_to_idx = {s["label"]: i for i, s in enumerate(instance["timeslots"])}
    for cid, asgn in assignment.items():
        c = courses[cid]
        if c.get("is_lab") and c.get("lab_duration_slots", 1) > 1:
            # Check whether a consecutive companion slot is also assigned
            sid = asgn["slot_id"]
            s_obj = slots.get(sid, {})
            label = s_obj.get("label", "")
            idx = slot_label_to_idx.get(label, -1)
            # Look for consecutive companion
            companion_ok = False
            for other_cid, other_asgn in assignment.items():
                if other_cid == cid:
                    continue
                other_s = slots.get(other_asgn["slot_id"], {})
                other_label = other_s.get("label", "")
                other_idx = slot_label_to_idx.get(other_label, -1)
                if (other_s.get("day") == s_obj.get("day") and
                        abs(other_idx - idx) == 1):
                    companion_ok = True
                    break
            if not companion_ok:
                violations += 1

    return violations


def _check_soft_penalties(
    assignment: dict[str, dict],
    instance: dict,
) -> float:
    """Return total weighted soft constraint penalty (lower = better)."""
    weights = instance.get("soft_constraint_weights",
                           {"S1": 5, "S2": 3, "S3": 4, "S4": 6, "S5": 2, "S6": 3})
    courses  = {c["id"]: c for c in instance["courses"]}
    slots    = {s["id"]: s for s in instance["timeslots"]}
    penalty  = 0.0

    faculty_day_slots: dict[str, dict[str, list]] = {}
    for cid, asgn in assignment.items():
        c  = courses[cid]
        fid = c.get("faculty_id", "")
        s  = slots.get(asgn["slot_id"], {})
        day = s.get("day", "")
        faculty_day_slots.setdefault(fid, {}).setdefault(day, []).append(s.get("period", 0))

        # S2: Avoid last period
        if s.get("is_last_period", False):
            penalty += weights["S2"]

        # S6: Morning preference for first-year students
        gid = c.get("group_id", "")
        group = next((g for g in instance["student_groups"] if g["id"] == gid), {})
        if group.get("is_first_year") and not s.get("is_morning", False):
            penalty += weights["S6"]

    # S1: Faculty idle time
    for fid, days in faculty_day_slots.items():
        for day, periods in days.items():
            if len(periods) > 1:
                periods_sorted = sorted(periods)
                idle = sum(
                    periods_sorted[i + 1] - periods_sorted[i] - 1
                    for i in range(len(periods_sorted) - 1)
                )
                penalty += weights["S1"] * idle

    # S4: Faculty workload imbalance (std dev across days)
    for fid, days in faculty_day_slots.items():
        loads = [len(p) for p in days.values()]
        if len(loads) > 1:
            imbalance = float(np.std(loads))
            penalty += weights["S4"] * imbalance

    return penalty


# ------------------------------------------------------------------ #
# Environment                                                          #
# ------------------------------------------------------------------ #

class TimetableEnv(gym.Env):
    """
    RL environment for university course timetabling.

    Action space  : Discrete(n_courses × n_slots × n_rooms)
    Observation   : dict with graph node features (flattened) and auxiliary vectors
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, instance: dict, render_mode: str | None = None):
        super().__init__()
        self.instance     = instance
        self.render_mode  = render_mode

        self.courses  = instance["courses"]
        self.rooms    = instance["rooms"]
        self.slots    = instance["timeslots"]

        self.n_courses = len(self.courses)
        self.n_rooms   = len(self.rooms)
        self.n_slots   = len(self.slots)

        self.course_ids = [c["id"] for c in self.courses]
        self.room_ids   = [r["id"] for r in self.rooms]
        self.slot_ids   = [s["id"] for s in self.slots]

        self.n_actions = self.n_courses * self.n_slots * self.n_rooms
        self.max_steps = 3 * self.n_courses

        # Action & observation spaces
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation: flattened partial assignment matrix + violation vector
        # partial_assignment_matrix: shape (n_courses, n_slots, n_rooms) binary
        # violation_vector: length 8 (one per hard constraint, normalised count)
        obs_dim = self.n_courses * self.n_slots * self.n_rooms + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.assignment: dict[str, dict] = {}
        self.step_count  = 0
        self.total_hard_before = 0

    # -------------------------------------------------------------- #
    # Gym API                                                          #
    # -------------------------------------------------------------- #

    def reset(
        self, *, seed: int | None = None, options: Any = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.assignment  = {}
        self.step_count  = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        course_idx, slot_idx, room_idx = self._decode_action(action)
        cid = self.course_ids[course_idx]
        sid = self.slot_ids[slot_idx]
        rid = self.room_ids[room_idx]

        # Compute violations before assignment
        hard_before = _check_hard_violations(self.assignment, self.instance)
        soft_before = _check_soft_penalties(self.assignment, self.instance)

        # Apply assignment (overwrite if course already assigned — allows reassignment)
        already_assigned = cid in self.assignment
        self.assignment[cid] = {"slot_id": sid, "room_id": rid}

        hard_after = _check_hard_violations(self.assignment, self.instance)
        soft_after = _check_soft_penalties(self.assignment, self.instance)

        # Reward components
        violations_introduced = max(0, hard_after - hard_before)
        soft_improved = max(0.0, soft_before - soft_after)
        progress = len(self.assignment) / self.n_courses

        r_hard       = -1000.0 * violations_introduced
        r_soft       = +10.0   * soft_improved
        r_progress   = +1.0    * progress
        r_completion = 0.0

        all_assigned = len(self.assignment) == self.n_courses
        if all_assigned and hard_after == 0:
            r_completion = 500.0

        reward = r_hard + r_soft + r_progress + r_completion

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = all_assigned

        obs  = self._get_obs()
        info = {
            "hard_violations": hard_after,
            "soft_penalty": soft_after,
            "courses_assigned": len(self.assignment),
            "progress": progress,
        }
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return boolean mask of shape (n_actions,).
        True  = action is valid (no hard constraint violation introduced).
        False = action would introduce at least one hard constraint violation.

        Uses precomputed conflict sets for O(n_actions) efficiency instead of
        calling _check_hard_violations per action.
        """
        # Precompute occupied sets from current assignment
        slot_room_used:    set[tuple] = set()
        slot_faculty_used: set[tuple] = set()
        slot_group_used:   set[tuple] = set()
        faculty_map  = {c["id"]: c.get("faculty_id") for c in self.courses}
        group_map    = {c["id"]: c.get("group_id", "") for c in self.courses}
        room_type_map = {r["id"]: r["type"] for r in self.rooms}
        course_type_map = {c["id"]: c.get("required_room_type", "lecture") for c in self.courses}
        faculty_unavail  = {
            f["id"]: set(f.get("unavailable_slots", []))
            for f in self.instance["faculty"]
        }
        slot_label_map = {s["id"]: s.get("label", "") for s in self.slots}

        for cid, asgn in self.assignment.items():
            sid = asgn["slot_id"]
            rid = asgn["room_id"]
            fid = faculty_map.get(cid)
            gid = group_map.get(cid, "")
            slot_room_used.add((sid, rid))
            if fid:
                slot_faculty_used.add((sid, fid))
            if gid:
                slot_group_used.add((sid, gid))

        masks = np.ones(self.n_actions, dtype=bool)
        for ci in range(self.n_courses):
            cid = self.course_ids[ci]
            fid = faculty_map.get(cid)
            gid = group_map.get(cid, "")
            required_type = course_type_map.get(cid, "lecture")

            # Check if this course is already assigned — if reassigning, remove old conflicts
            old_asgn = self.assignment.get(cid)

            for si in range(self.n_slots):
                sid = self.slot_ids[si]
                slot_label = slot_label_map.get(sid, "")

                # H2: Faculty double-booking (skip if reassigning same course)
                fac_conflict = (
                    fid and (sid, fid) in slot_faculty_used
                    and not (old_asgn and old_asgn["slot_id"] == sid)
                )
                # H3: Group double-booking
                grp_conflict = (
                    gid and (sid, gid) in slot_group_used
                    and not (old_asgn and old_asgn["slot_id"] == sid)
                )
                # H6: Faculty unavailability
                fac_unavail = fid and slot_label in faculty_unavail.get(fid, set())

                for ri in range(self.n_rooms):
                    rid = self.room_ids[ri]
                    # H1: Room double-booking
                    room_conflict = (
                        (sid, rid) in slot_room_used
                        and not (old_asgn and old_asgn["slot_id"] == sid
                                 and old_asgn["room_id"] == rid)
                    )
                    # H5: Room type mismatch
                    type_conflict = room_type_map.get(rid) != required_type

                    a = self._encode_action(ci, si, ri)
                    masks[a] = not (room_conflict or fac_conflict or
                                    grp_conflict or fac_unavail or type_conflict)

        # Ensure at least one action is valid (fallback: unmask all if all masked)
        if not masks.any():
            masks[:] = True
        return masks

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.step_count}: "
                  f"{len(self.assignment)}/{self.n_courses} courses assigned")
            hard = _check_hard_violations(self.assignment, self.instance)
            print(f"  Hard violations: {hard}")

    # -------------------------------------------------------------- #
    # Helpers                                                          #
    # -------------------------------------------------------------- #

    def _decode_action(self, action: int) -> tuple[int, int, int]:
        room_idx   = action % self.n_rooms
        slot_idx   = (action // self.n_rooms) % self.n_slots
        course_idx = action // (self.n_rooms * self.n_slots)
        return course_idx, slot_idx, room_idx

    def _encode_action(self, course_idx: int, slot_idx: int, room_idx: int) -> int:
        return course_idx * self.n_slots * self.n_rooms + slot_idx * self.n_rooms + room_idx

    def _get_obs(self) -> np.ndarray:
        # Partial assignment matrix
        mat = np.zeros((self.n_courses, self.n_slots, self.n_rooms), dtype=np.float32)
        for cid, asgn in self.assignment.items():
            ci = self.course_ids.index(cid)
            si = self.slot_ids.index(asgn["slot_id"])
            ri = self.room_ids.index(asgn["room_id"])
            mat[ci, si, ri] = 1.0

        # Violation vector (8 hard constraints, each as 0/1 flag)
        viol = np.zeros(8, dtype=np.float32)
        if self.assignment:
            hard = _check_hard_violations(self.assignment, self.instance)
            viol[:] = min(hard / max(self.n_courses, 1), 1.0)

        return np.concatenate([mat.flatten(), viol])

    def get_hetero_graph(self):
        """Return current HeteroData graph (for GNN-based policies)."""
        return build_hetero_graph(self.instance)


# ------------------------------------------------------------------ #
# Quick test                                                           #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import pathlib
    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if not sample.exists():
        print("Run generate_indian_data.py first.")
    else:
        with open(sample) as f:
            inst = json.load(f)
        env = TimetableEnv(inst)
        obs, info = env.reset()
        print(f"Obs shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        for step in range(5):
            action = env.action_space.sample()
            obs, rew, done, trunc, info = env.step(action)
            env.render()
            print(f"  reward={rew:.1f}, done={done}, info={info}")
            if done or trunc:
                break
