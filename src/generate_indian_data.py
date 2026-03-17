"""
generate_indian_data.py
-----------------------
Generates synthetic Indian university timetabling instances as JSON files.

Encodes India-specific constraints:
  H7 — Medium compliance: faculty medium must match course medium tag
       (English-medium, Hindi-medium, Regional-medium)
  H8 — Lab batching: lab sessions occupy consecutive slots of required duration

Usage:
    python src/generate_indian_data.py \
        --n_courses 30 --n_faculty 15 --n_rooms 10 \
        --n_slots 40 --n_groups 8 \
        --output data/indian_synthetic/instance_30c.json
"""

import argparse
import json
import random
import os
from pathlib import Path


MEDIUMS = ["english", "hindi", "regional"]
ROOM_TYPES = ["lecture", "lab", "seminar"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
PERIODS_PER_DAY = 8  # 9am–5pm, 1-hour slots


def generate_timeslots(n_slots: int) -> list[dict]:
    """Generate time slot objects spread across a week."""
    slots = []
    day_idx = 0
    period = 1
    for i in range(n_slots):
        slots.append({
            "id": f"slot_{i}",
            "day": DAYS[day_idx % len(DAYS)],
            "period": period,
            "label": f"{DAYS[day_idx % len(DAYS)]}_P{period}",
            "is_morning": period <= 3,   # periods 1-3 = morning
            "is_last_period": period == PERIODS_PER_DAY,
        })
        period += 1
        if period > PERIODS_PER_DAY:
            period = 1
            day_idx += 1
    return slots


def generate_rooms(n_rooms: int) -> list[dict]:
    """Generate rooms with type (lecture/lab/seminar) and capacity."""
    rooms = []
    for i in range(n_rooms):
        rtype = ROOM_TYPES[i % len(ROOM_TYPES)]
        rooms.append({
            "id": f"room_{i}",
            "type": rtype,
            "capacity": random.choice([40, 60, 80, 120]) if rtype == "lecture"
                        else random.choice([20, 30]) if rtype == "lab"
                        else 30,
            "building": f"Block_{chr(65 + i % 5)}",  # Building A–E
        })
    return rooms


def generate_faculty(n_faculty: int) -> list[dict]:
    """Generate faculty with medium capability and weekly availability."""
    faculty = []
    for i in range(n_faculty):
        medium = random.choice(MEDIUMS)
        # unavailable_slots: random ~10% of all week periods
        unavailable = [
            f"{random.choice(DAYS)}_P{random.randint(1, PERIODS_PER_DAY)}"
            for _ in range(random.randint(2, 6))
        ]
        faculty.append({
            "id": f"faculty_{i}",
            "name": f"Prof_{i}",
            "medium": medium,
            "unavailable_slots": list(set(unavailable)),
            "max_hours_per_day": random.randint(4, 6),
        })
    return faculty


def generate_student_groups(n_groups: int) -> list[dict]:
    """Generate student groups with year and rough size."""
    groups = []
    for i in range(n_groups):
        year = (i % 4) + 1
        groups.append({
            "id": f"group_{i}",
            "name": f"Group_{i}",
            "year": year,
            "size": random.randint(30, 60),
            "is_first_year": year == 1,
        })
    return groups


def generate_courses(n_courses: int, faculty: list, groups: list) -> list[dict]:
    """Generate courses with constraints H5 (room type), H7 (medium), H8 (lab batching)."""
    courses = []
    for i in range(n_courses):
        is_lab = (i % 5 == 0)  # every 5th course is a lab
        medium = random.choice(MEDIUMS)
        # find faculty that can teach this medium (H7)
        eligible_faculty = [f["id"] for f in faculty if f["medium"] == medium]
        if not eligible_faculty:
            eligible_faculty = [random.choice(faculty)["id"]]
        assigned_faculty = random.choice(eligible_faculty)
        assigned_group = random.choice(groups)["id"]
        courses.append({
            "id": f"course_{i}",
            "name": f"Course_{i}",
            "is_lab": is_lab,
            "medium": medium,                        # H7: must match faculty medium
            "required_room_type": "lab" if is_lab else "lecture",  # H5
            "lab_duration_slots": 2 if is_lab else 1,  # H8: labs need 2 consecutive slots
            "enrolment": random.randint(20, 60),
            "faculty_id": assigned_faculty,          # assigned faculty
            "group_id": assigned_group,
            "weekly_hours": 3 if is_lab else random.randint(2, 4),
        })
    return courses


def generate_instance(
    n_courses: int,
    n_faculty: int,
    n_rooms: int,
    n_slots: int,
    n_groups: int,
    seed: int = 42,
) -> dict:
    """Generate one complete synthetic Indian timetabling instance."""
    random.seed(seed)
    timeslots = generate_timeslots(n_slots)
    rooms = generate_rooms(n_rooms)
    faculty = generate_faculty(n_faculty)
    groups = generate_student_groups(n_groups)
    courses = generate_courses(n_courses, faculty, groups)

    return {
        "meta": {
            "source": "synthetic_indian",
            "seed": seed,
            "n_courses": n_courses,
            "n_faculty": n_faculty,
            "n_rooms": n_rooms,
            "n_slots": n_slots,
            "n_groups": n_groups,
        },
        "timeslots": timeslots,
        "rooms": rooms,
        "faculty": faculty,
        "student_groups": groups,
        "courses": courses,
        "hard_constraints": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
        "soft_constraint_weights": {
            "S1": 5, "S2": 3, "S3": 4, "S4": 6, "S5": 2, "S6": 3
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Indian timetabling instances")
    parser.add_argument("--n_courses", type=int, default=30)
    parser.add_argument("--n_faculty", type=int, default=15)
    parser.add_argument("--n_rooms", type=int, default=10)
    parser.add_argument("--n_slots", type=int, default=40)
    parser.add_argument("--n_groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/indian_synthetic/instance.json")
    parser.add_argument("--batch", type=int, default=1,
                        help="Generate N instances with different seeds")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.batch == 1:
        instance = generate_instance(
            args.n_courses, args.n_faculty, args.n_rooms,
            args.n_slots, args.n_groups, args.seed
        )
        with open(out_path, "w") as f:
            json.dump(instance, f, indent=2)
        print(f"Generated instance → {out_path}")
        print(f"  Courses: {args.n_courses}, Faculty: {args.n_faculty}, "
              f"Rooms: {args.n_rooms}, Slots: {args.n_slots}, Groups: {args.n_groups}")
    else:
        stem = out_path.stem
        for i in range(args.batch):
            seed_i = args.seed + i
            instance = generate_instance(
                args.n_courses, args.n_faculty, args.n_rooms,
                args.n_slots, args.n_groups, seed_i
            )
            fp = out_path.parent / f"{stem}_{i:03d}.json"
            with open(fp, "w") as f:
                json.dump(instance, f, indent=2)
        print(f"Generated {args.batch} instances in {out_path.parent}/")


if __name__ == "__main__":
    main()
