# UTCP-GNN-RL: GNN-Augmented Deep Reinforcement Learning for University Timetable Scheduling

> **Paper Title**: "GNN-Augmented Deep Reinforcement Learning for Constraint-Aware University Timetable Scheduling: A Framework for Indian Academic Contexts"
>
> **Target Journals**: Expert Systems with Applications · Applied Soft Computing · IEEE Access

---

## Overview

This repository implements a novel **GNN + Deep RL (PPO)** framework for solving the University Course Timetabling Problem (UTCP), with a specific focus on Indian academic constraints. It accompanies the research paper of the same name.

### Key Contributions
- First GNN + RL framework for UTCP
- Formal encoding of Indian-specific constraints (medium compliance, lab batching)
- Comprehensive evaluation on ITC-2007, ITC-2019, and a self-generated Indian synthetic benchmark
- Four classical baselines for fair comparison (GA, SA, Tabu, CP-SAT)

---

## Repository Structure

```
utcp-gnn-rl/
├── data/
│   ├── itc2007/              # ITC-2007 XML instances (download separately)
│   ├── itc2019/              # ITC-2019 XML instances (register at itc2019.ugent.be)
│   └── indian_synthetic/     # Auto-generated Indian instances (JSON)
├── src/
│   ├── generate_indian_data.py   # Synthetic Indian instance generator
│   ├── graph_builder.py          # Instance → HeteroData graph
│   ├── gnn_encoder.py            # HGTConv heterogeneous GNN
│   ├── timetable_env.py          # Gymnasium RL environment
│   ├── ppo_agent.py              # MaskablePPO + custom GNN policy
│   ├── train.py                  # Full training pipeline (IL + RL)
│   ├── evaluate.py               # 5-metric evaluation (HCSR, SCP, TTFS, SSD, SI)
│   └── baselines/
│       ├── ga_baseline.py        # Genetic Algorithm (DEAP)
│       ├── sa_baseline.py        # Simulated Annealing
│       ├── tabu_baseline.py      # Tabu Search
│       └── cp_baseline.py        # CP-SAT (OR-Tools)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_visualization.ipynb
│   ├── 03_training_curves.ipynb
│   └── 04_results_tables.ipynb
├── paper/
│   ├── main.tex                  # Full LaTeX paper
│   └── references.bib            # 40+ BibTeX entries
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# For GPU (PyG sparse ops):
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. Generate Synthetic Indian Instances

```bash
python src/generate_indian_data.py \
  --n_courses 30 --n_faculty 15 --n_rooms 10 \
  --n_slots 40 --n_groups 8 \
  --output data/indian_synthetic/instance_30c.json
```

### 3. Train the GNN+RL Agent

```bash
# Phase 1: Imitation learning warm-start
python src/train.py --mode pretrain \
  --data data/indian_synthetic/instance_30c.json \
  --timesteps 500000

# Phase 2: RL fine-tuning
python src/train.py --mode rl \
  --data data/indian_synthetic/instance_30c.json \
  --pretrained checkpoints/il_model.zip \
  --timesteps 5000000
```

### 4. Evaluate

```bash
python src/evaluate.py \
  --model checkpoints/ppo_final.zip \
  --data data/indian_synthetic/ \
  --baselines all
```

---

## Constraints Modelled

### Hard Constraints (H1–H8)

| ID | Description | Type |
|----|------------|------|
| H1 | No room double-booking | Universal |
| H2 | No faculty double-booking | Universal |
| H3 | No student-group double-booking | Universal |
| H4 | Room capacity ≥ enrolment | Universal |
| H5 | Room type matches course requirement | Universal |
| H6 | Faculty availability respected | Universal |
| H7 | Medium compliance: faculty medium matches course | India-specific |
| H8 | Lab batching: consecutive slots for lab sessions | India-specific |

### Soft Constraints (S1–S6)

| ID | Description | Weight |
|----|------------|--------|
| S1 | Minimize faculty idle time | 5 |
| S2 | Avoid last-period scheduling | 3 |
| S3 | Minimize student inter-building travel | 4 |
| S4 | Distribute faculty workload evenly | 6 |
| S5 | Preferred rooms assigned where possible | 2 |
| S6 | Morning slots preferred for first-year students | 3 |

---

## Evaluation Metrics

| Metric | Definition | Direction |
|--------|-----------|-----------|
| HCSR | 1 − violations/total_hard | Higher better |
| SCP | Σ wᵢ × soft_violations | Lower better |
| TTFS | Seconds to first feasible solution | Lower better |
| SSD | HCSR after single random disruption | Higher better |
| SI | HCSR_large / HCSR_small | Higher better (aim ~1) |

---

## Citation

```bibtex
@article{utcp_gnn_rl_2025,
  title   = {GNN-Augmented Deep Reinforcement Learning for Constraint-Aware
             University Timetable Scheduling: A Framework for Indian Academic Contexts},
  author  = {Suryanshi},
  journal = {Expert Systems with Applications},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
