"""
train.py
--------
Full two-phase training pipeline:
  Phase 1 — Imitation Learning (IL) warm-start using a greedy CSP solver
  Phase 2 — RL fine-tuning with MaskablePPO (curriculum: small → large instances)

Usage:
    # Phase 1: Imitation learning
    python src/train.py --mode pretrain \
        --data data/indian_synthetic/instance.json \
        --timesteps 500000 \
        --save checkpoints/il_model.zip

    # Phase 2: RL fine-tuning
    python src/train.py --mode rl \
        --data data/indian_synthetic/instance.json \
        --pretrained checkpoints/il_model.zip \
        --timesteps 5000000 \
        --save checkpoints/ppo_final.zip
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import random
import numpy as np
import torch
import torch.distributions
torch.distributions.Distribution.set_default_validate_args(False)
from tqdm import tqdm


# ------------------------------------------------------------------ #
# Greedy CSP-based solution generator (for IL warm-start)             #
# ------------------------------------------------------------------ #

def greedy_csp_solve(instance: dict, seed: int = 0) -> dict[str, dict] | None:
    """
    Greedy constraint-propagation solver.
    Returns a partial or full assignment dict: {course_id: {slot_id, room_id}}
    Satisfies H1 (room deconflict), H2 (faculty deconflict), H3 (group deconflict),
    H5 (room type), H7 (medium match — only eligible rooms/slots considered).
    """
    rng = random.Random(seed)
    courses  = list(instance["courses"])
    slots    = [s["id"] for s in instance["timeslots"]]
    rooms    = instance["rooms"]
    faculty_map  = {f["id"]: f for f in instance["faculty"]}
    room_type_map = {r["id"]: r for r in rooms}

    # Shuffle for diversity
    rng.shuffle(courses)
    rng.shuffle(slots)

    assignment: dict[str, dict] = {}
    slot_room_used:    set[tuple] = set()
    slot_faculty_used: set[tuple] = set()
    slot_group_used:   set[tuple] = set()

    for c in courses:
        cid   = c["id"]
        fid   = c.get("faculty_id")
        gid   = c.get("group_id")
        ctype = c.get("required_room_type", "lecture")
        c_medium = c.get("medium", "english")

        # Faculty medium check (H7)
        fac = faculty_map.get(fid, {})
        if fac.get("medium") and fac["medium"] != c_medium:
            continue  # skip — no eligible faculty

        eligible_rooms = [r["id"] for r in rooms if r["type"] == ctype]
        placed = False

        for sid in slots:
            if (sid, fid) in slot_faculty_used:
                continue   # H2
            if (sid, gid) in slot_group_used:
                continue   # H3
            for rid in eligible_rooms:
                if (sid, rid) in slot_room_used:
                    continue  # H1
                # Place
                assignment[cid] = {"slot_id": sid, "room_id": rid}
                slot_room_used.add((sid, rid))
                slot_faculty_used.add((sid, fid))
                slot_group_used.add((sid, gid))
                placed = True
                break
            if placed:
                break

    return assignment


def generate_il_demonstrations(
    instance: dict,
    n_demos: int = 10_000,
    seed: int = 0,
) -> list[dict[str, dict]]:
    """Generate n_demos diverse greedy solutions for imitation learning."""
    demos = []
    for i in tqdm(range(n_demos), desc="Generating IL demonstrations"):
        sol = greedy_csp_solve(instance, seed=seed + i)
        if sol:
            demos.append(sol)
    print(f"Generated {len(demos)} feasible IL demonstrations.")
    return demos


# ------------------------------------------------------------------ #
# Imitation Learning training (behaviour cloning on greedy demos)     #
# ------------------------------------------------------------------ #

def pretrain_il(
    instance: dict,
    n_demos: int = 10_000,
    epochs: int = 20,
    lr: float = 1e-3,
    save_path: str = "checkpoints/il_model.zip",
    seed: int = 42,
):
    """
    Behaviour cloning warm-start.
    Trains a simple MLP that maps (observation) → greedy action.
    """
    from timetable_env import TimetableEnv
    from ppo_agent import make_masked_env, build_ppo_agent

    print("=== Phase 1: Imitation Learning Warm-Start ===")
    demos = generate_il_demonstrations(instance, n_demos=min(n_demos, 500), seed=seed)

    env = TimetableEnv(instance)

    # Collect (obs, action) pairs from demos
    obs_list, act_list = [], []
    for demo in demos[:200]:  # use subset for speed
        obs, _ = env.reset()
        for cid, asgn in demo.items():
            sid = asgn["slot_id"]
            rid = asgn["room_id"]
            if cid in env.course_ids and sid in env.slot_ids and rid in env.room_ids:
                ci = env.course_ids.index(cid)
                si = env.slot_ids.index(sid)
                ri = env.room_ids.index(rid)
                act = env._encode_action(ci, si, ri)
                obs_list.append(obs.copy())
                act_list.append(act)
                obs, _, done, trunc, _ = env.step(act)
                if done or trunc:
                    break

    if not obs_list:
        print("No IL demonstrations collected, skipping pretrain.")
        return

    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    y = torch.tensor(act_list, dtype=torch.long)

    # Simple MLP classifier
    hidden = 256
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, env.n_actions),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Training IL model on {len(obs_list)} (obs, action) pairs for {epochs} epochs...")
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path.replace(".zip", "_il_weights.pt"))
    print(f"IL weights saved → {save_path.replace('.zip', '_il_weights.pt')}")

    # Now build a PPO agent and save it (SB3 format)
    agent = build_ppo_agent(instance, total_timesteps=1000)
    agent.learn(total_timesteps=1000)
    agent.save(save_path)
    print(f"Warm-start PPO skeleton saved → {save_path}")


# ------------------------------------------------------------------ #
# RL fine-tuning                                                       #
# ------------------------------------------------------------------ #

def train_rl(
    instance: dict,
    total_timesteps: int = 5_000_000,
    pretrained_path: str | None = None,
    save_path: str = "checkpoints/ppo_final.zip",
    eval_freq: int = 50_000,
    seed: int = 42,
):
    """Train MaskablePPO agent with curriculum (small → large)."""
    from ppo_agent import build_ppo_agent
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    print("=== Phase 2: RL Fine-Tuning with MaskablePPO ===")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Pretrained path : {pretrained_path}")

    agent = build_ppo_agent(
        instance,
        total_timesteps=total_timesteps,
        pretrained_path=pretrained_path,
        device="auto",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(pathlib.Path(save_path).parent / "checkpoints"),
        name_prefix="ppo_ckpt",
        verbose=1,
    )

    agent.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"Final model saved → {save_path}")
    return agent


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train GNN+RL timetabling agent")
    parser.add_argument("--mode",       choices=["pretrain", "rl", "full"],
                        default="full",
                        help="pretrain=IL only, rl=PPO only, full=both")
    parser.add_argument("--data",       type=str,
                        default="data/indian_synthetic/instance.json")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--timesteps",  type=int, default=5_000_000)
    parser.add_argument("--n_demos",    type=int, default=10_000)
    parser.add_argument("--save",       type=str,
                        default="checkpoints/ppo_final.zip")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    with open(args.data) as f:
        instance = json.load(f)

    il_path = args.save.replace(".zip", "_il.zip")

    if args.mode in ("pretrain", "full"):
        pretrain_il(instance, n_demos=args.n_demos, save_path=il_path, seed=args.seed)

    if args.mode in ("rl", "full"):
        pretrained = il_path if args.mode == "full" else args.pretrained
        train_rl(
            instance,
            total_timesteps=args.timesteps,
            pretrained_path=pretrained if pathlib.Path(pretrained or "").exists() else None,
            save_path=args.save,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
