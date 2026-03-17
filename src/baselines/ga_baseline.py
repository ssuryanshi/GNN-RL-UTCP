"""
ga_baseline.py
--------------
Genetic Algorithm baseline using DEAP.

Hyperparameters (from blueprint):
  population_size = 200
  generations     = 500
  mutation_rate   = 0.01
  crossover_rate  = 0.7

Encoding: chromosome = list of (slot_idx, room_idx) per course
"""

from __future__ import annotations
import random
import time
import numpy as np

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("WARNING: DEAP not installed. Install with: pip install deap")


def _decode_chromosome(individual: list, n_slots: int, n_rooms: int) -> list[tuple]:
    """Decode DEAP individual → list of (slot_idx, room_idx) per course."""
    result = []
    for gene in individual:
        slot_idx = gene % n_slots
        room_idx = (gene // n_slots) % n_rooms
        result.append((slot_idx, room_idx))
    return result


def _evaluate_chromosome(individual, instance: dict) -> tuple[float]:
    """Evaluate a chromosome; returns (fitness,) where higher = better."""
    from timetable_env import _check_hard_violations, _check_soft_penalties

    courses  = instance["courses"]
    slot_ids = [s["id"] for s in instance["timeslots"]]
    room_ids = [r["id"] for r in instance["rooms"]]
    n_slots  = len(slot_ids)
    n_rooms  = len(room_ids)

    assignment: dict[str, dict] = {}
    decoded = _decode_chromosome(individual, n_slots, n_rooms)
    for i, c in enumerate(courses):
        si, ri = decoded[i]
        assignment[c["id"]] = {
            "slot_id": slot_ids[si],
            "room_id": room_ids[ri],
        }

    hard = _check_hard_violations(assignment, instance)
    soft = _check_soft_penalties(assignment, instance)
    # Fitness = penalize hard violations heavily, then minimize soft
    fitness = -1000.0 * hard - soft
    return (fitness,)


class GABaseline:
    """
    Genetic Algorithm solver for UTCP.

    Parameters
    ----------
    population_size : int
    generations     : int
    mutation_rate   : float
    crossover_rate  : float
    seed            : int
    """

    def __init__(
        self,
        population_size: int = 200,
        generations:     int = 500,
        mutation_rate:   float = 0.01,
        crossover_rate:  float = 0.7,
        seed:            int = 42,
    ):
        assert DEAP_AVAILABLE, "Install DEAP: pip install deap"
        self.pop_size   = population_size
        self.n_gen      = generations
        self.mut_rate   = mutation_rate
        self.cx_rate    = crossover_rate
        self.seed       = seed
        self._toolbox   = None
        self._best_assignment: dict | None = None
        self._stats: dict = {}

    def solve(self, instance: dict) -> dict:
        """
        Solve the timetabling instance.

        Returns
        -------
        dict with keys: assignment, hard_violations, soft_penalty, runtime_sec, best_cost
        """
        import sys
        random.seed(self.seed)
        np.random.seed(self.seed)

        courses  = instance["courses"]
        n_courses = len(courses)
        n_slots   = len(instance["timeslots"])
        n_rooms   = len(instance["rooms"])
        gene_max  = n_slots * n_rooms

        # DEAP setup (create once per call to avoid re-registration errors)
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_gene", random.randint, 0, gene_max - 1)
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_gene, n=n_courses)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _evaluate_chromosome, instance=instance)
        toolbox.register("mate",     tools.cxTwoPoint)
        toolbox.register("mutate",   tools.mutUniformInt, low=0, up=gene_max - 1,
                         indpb=self.mut_rate)
        toolbox.register("select",   tools.selTournament, tournsize=5)

        pop  = toolbox.population(n=self.pop_size)
        hof  = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        start = time.time()
        pop, log = algorithms.eaSimple(
            pop, toolbox,
            cxpb=self.cx_rate,
            mutpb=self.mut_rate,
            ngen=self.n_gen,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )
        elapsed = time.time() - start

        # Decode best individual
        best = hof[0]
        slot_ids = [s["id"] for s in instance["timeslots"]]
        room_ids = [r["id"] for r in instance["rooms"]]
        decoded = _decode_chromosome(best, n_slots, n_rooms)
        assignment: dict[str, dict] = {}
        for i, c in enumerate(courses):
            si, ri = decoded[i]
            assignment[c["id"]] = {
                "slot_id": slot_ids[si],
                "room_id": room_ids[ri],
            }

        from timetable_env import _check_hard_violations, _check_soft_penalties
        hard = _check_hard_violations(assignment, instance)
        soft = _check_soft_penalties(assignment, instance)

        self._best_assignment = assignment
        self._stats = {
            "hard_violations": hard,
            "soft_penalty":    soft,
            "runtime_sec":     elapsed,
            "best_fitness":    float(best.fitness.values[0]),
            "logbook":         log,
        }
        return self._stats | {"assignment": assignment}


if __name__ == "__main__":
    import json, pathlib
    sample = pathlib.Path("data/indian_synthetic/instance.json")
    if sample.exists():
        with open(sample) as f:
            inst = json.load(f)
        ga = GABaseline(population_size=50, generations=20)
        result = ga.solve(inst)
        print(f"GA  ->  hard={result['hard_violations']}, "
              f"soft={result['soft_penalty']:.1f}, "
              f"time={result['runtime_sec']:.1f}s")
