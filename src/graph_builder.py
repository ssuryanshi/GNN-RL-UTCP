"""
graph_builder.py
----------------
Converts a timetabling instance (dict, e.g. from JSON) into a
torch_geometric.data.HeteroData heterogeneous graph.

Node types:
  course    — one node per course
  faculty   — one node per faculty member
  room      — one node per room
  timeslot  — one node per time slot

Edge types:
  (faculty, teaches, course)          — faculty is assigned to course
  (course,  conflicts_with, course)   — same student group → must not share slot
  (course,  requires, room)           — course needs a specific room type
  (timeslot, consecutive_with, timeslot) — adjacent slots (needed for H8)
  (faculty, locked_during, timeslot)  — faculty unavailable in slot

Node features:
  course:   [is_lab, enrolment_norm, weekly_hours_norm, medium_enc(3)]
  faculty:  [medium_enc(3), max_hours_norm]
  room:     [capacity_norm, type_enc(3)]
  timeslot: [day_enc(6), period_norm, is_morning, is_last_period]
"""

from __future__ import annotations
import torch
from torch_geometric.data import HeteroData

MEDIUMS = ["english", "hindi", "regional"]
ROOM_TYPES = ["lecture", "lab", "seminar"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _one_hot(value: str, categories: list[str]) -> list[float]:
    return [1.0 if value == c else 0.0 for c in categories]


def build_hetero_graph(instance: dict) -> HeteroData:
    """
    Parameters
    ----------
    instance : dict  (loaded from generate_indian_data.py JSON output)

    Returns
    -------
    HeteroData with node features and edge indices.
    """
    courses   = instance["courses"]
    faculty   = instance["faculty"]
    rooms     = instance["rooms"]
    slots     = instance["timeslots"]

    # ------------------------------------------------------------------ #
    # Build index maps                                                     #
    # ------------------------------------------------------------------ #
    course_idx  = {c["id"]: i for i, c in enumerate(courses)}
    faculty_idx = {f["id"]: i for i, f in enumerate(faculty)}
    room_idx    = {r["id"]: i for i, r in enumerate(rooms)}
    slot_idx    = {s["id"]: i for i, s in enumerate(slots)}

    max_enrol   = max(c["enrolment"]    for c in courses) or 1
    max_hours   = max(c["weekly_hours"] for c in courses) or 1
    max_cap     = max(r["capacity"]     for r in rooms)   or 1
    max_fachrs  = max(f["max_hours_per_day"] for f in faculty) or 1
    n_slots     = len(slots)

    # ------------------------------------------------------------------ #
    # Node features                                                        #
    # ------------------------------------------------------------------ #
    # Course nodes
    course_feats = []
    for c in courses:
        feat = (
            [float(c["is_lab"]),
             c["enrolment"]    / max_enrol,
             c["weekly_hours"] / max_hours]
            + _one_hot(c["medium"], MEDIUMS)      # 3 dims medium (H7)
        )
        course_feats.append(feat)  # dim = 6

    # Faculty nodes
    faculty_feats = []
    for f in faculty:
        feat = (
            _one_hot(f["medium"], MEDIUMS)
            + [f["max_hours_per_day"] / max_fachrs]
        )
        faculty_feats.append(feat)  # dim = 4

    # Room nodes
    room_feats = []
    for r in rooms:
        feat = (
            [r["capacity"] / max_cap]
            + _one_hot(r["type"], ROOM_TYPES)
        )
        room_feats.append(feat)  # dim = 4

    # TimeSlot nodes
    slot_feats = []
    for s in slots:
        feat = (
            _one_hot(s["day"], DAYS)              # 6 dims
            + [s["period"] / 8,
               float(s["is_morning"]),
               float(s["is_last_period"])]
        )
        slot_feats.append(feat)  # dim = 9

    data = HeteroData()
    data["course"].x   = torch.tensor(course_feats,  dtype=torch.float)
    data["faculty"].x  = torch.tensor(faculty_feats, dtype=torch.float)
    data["room"].x     = torch.tensor(room_feats,    dtype=torch.float)
    data["timeslot"].x = torch.tensor(slot_feats,    dtype=torch.float)

    # ------------------------------------------------------------------ #
    # Edges: (faculty, teaches, course)                                   #
    # ------------------------------------------------------------------ #
    teaches_src, teaches_dst = [], []
    for c in courses:
        fid = c.get("faculty_id")
        if fid and fid in faculty_idx:
            teaches_src.append(faculty_idx[fid])
            teaches_dst.append(course_idx[c["id"]])

    data["faculty", "teaches", "course"].edge_index = torch.tensor(
        [teaches_src, teaches_dst], dtype=torch.long
    )

    # ------------------------------------------------------------------ #
    # Edges: (course, conflicts_with, course) — same student group        #
    # ------------------------------------------------------------------ #
    from itertools import combinations
    group_to_courses: dict[str, list[int]] = {}
    for c in courses:
        gid = c.get("group_id", "")
        group_to_courses.setdefault(gid, []).append(course_idx[c["id"]])

    cf_src, cf_dst = [], []
    for group_courses in group_to_courses.values():
        for a, b in combinations(group_courses, 2):
            cf_src += [a, b]
            cf_dst += [b, a]  # undirected

    data["course", "conflicts_with", "course"].edge_index = torch.tensor(
        [cf_src, cf_dst], dtype=torch.long
    ) if cf_src else torch.zeros((2, 0), dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Edges: (course, requires, room) — matching room type (H5)           #
    # ------------------------------------------------------------------ #
    req_src, req_dst = [], []
    for c in courses:
        ctype = c.get("required_room_type", "lecture")
        for r in rooms:
            if r["type"] == ctype:
                req_src.append(course_idx[c["id"]])
                req_dst.append(room_idx[r["id"]])

    data["course", "requires", "room"].edge_index = torch.tensor(
        [req_src, req_dst], dtype=torch.long
    ) if req_src else torch.zeros((2, 0), dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Edges: (timeslot, consecutive_with, timeslot) — H8 lab batching     #
    # Consecutive slots on the same day                                   #
    # ------------------------------------------------------------------ #
    consec_src, consec_dst = [], []
    slot_by_day: dict[str, list[tuple[int, int]]] = {}
    for s in slots:
        slot_by_day.setdefault(s["day"], []).append((s["period"], slot_idx[s["id"]]))

    for day_slots in slot_by_day.values():
        day_slots_sorted = sorted(day_slots, key=lambda x: x[0])
        for k in range(len(day_slots_sorted) - 1):
            a = day_slots_sorted[k][1]
            b = day_slots_sorted[k + 1][1]
            consec_src += [a, b]
            consec_dst += [b, a]

    data["timeslot", "consecutive_with", "timeslot"].edge_index = torch.tensor(
        [consec_src, consec_dst], dtype=torch.long
    ) if consec_src else torch.zeros((2, 0), dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Edges: (faculty, locked_during, timeslot) — unavailability (H6)    #
    # ------------------------------------------------------------------ #
    locked_src, locked_dst = [], []
    slot_by_label = {s["label"]: slot_idx[s["id"]] for s in slots}

    for f in faculty:
        for label in f.get("unavailable_slots", []):
            if label in slot_by_label:
                locked_src.append(faculty_idx[f["id"]])
                locked_dst.append(slot_by_label[label])

    data["faculty", "locked_during", "timeslot"].edge_index = torch.tensor(
        [locked_src, locked_dst], dtype=torch.long
    ) if locked_src else torch.zeros((2, 0), dtype=torch.long)

    return data


if __name__ == "__main__":
    import json, pathlib
    sample_path = pathlib.Path("data/indian_synthetic/instance.json")
    if sample_path.exists():
        with open(sample_path) as f:
            inst = json.load(f)
        g = build_hetero_graph(inst)
        print(g)
        print("\nNode feature dims:")
        for nt in g.node_types:
            print(f"  {nt}: {g[nt].x.shape}")
        print("\nEdge types:", g.edge_types)
    else:
        print("Run generate_indian_data.py first.")
