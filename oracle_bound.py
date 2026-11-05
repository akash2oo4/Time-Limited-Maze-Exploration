"""
oracle_bound.py
Is the budgeted exploration gap closable, or is it information-theoretic?

The 250-maze sweep showed two hand-designed policies leaving 0.2-0.5 s on the table under
a time budget -- and leaving exactly the SAME amount, statistically indistinguishable at
every budget. That tells us the two heuristics fail identically. It does not tell us
whether ANY policy could do better.

This measures the ceiling. `OracleGuidedPolicy` cheats: it reads the true maze, computes
the genuinely optimal route, and spends its budget verifying only the cells on that route.
It is not a method and could never run on hardware -- it is an upper bound on what perfect
exploration decisions could achieve with the same sensor and the same clock.

Two outcomes, both decisive:

  * Oracle reaches ~0 gap where classical sits at 0.3  ->  that 0.3 is learnable headroom.
    Phases 5-6 have a concrete target, and the paper reports classical / learned / bound
    on one axis.
  * Oracle is barely better than classical  ->  the gap is information-theoretic. There
    simply are not enough sensor readings available in the time allowed, no policy can fix
    it, and the finding is that budgeted exploration is sensor-bound rather than
    policy-bound. Also publishable, and known in a day instead of six weeks.

Run:
    python3 oracle_bound.py           # 100 mazes
    python3 oracle_bound.py 250
"""

import random
import sys
import time

import numpy as np

from maze import Maze, DIR_DELTA
from belief import BeliefMap, NORTH
from solver import truth_passable, belief_passable, time_to_goal, extract_path
from dataset_generator import generate_maze, T_CELL, turn_cost, UNREACHABLE, NUM_HEADINGS
from episode import (EpisodeConfig, ExplorationPolicy, _CachedFieldPolicy,
                     FloodFillPolicy, OptimisticShortcutPolicy, run_episode)

NO_ROUTE = 1e5
BUDGETS = [None, 25, 22, 20, 18, 15, 12]


class OracleGuidedPolicy(ExplorationPolicy):
    """
    UPPER BOUND, NOT A METHOD. Reads ground truth. Cannot run on hardware.

    Computes the true optimal route once, then drives to verify exactly that route and
    nothing else. This is the best any exploration policy could possibly do with this
    sensor and this clock: it never wastes a single move on a cell that does not matter.

    Note it is not trivially perfect either -- it still has to physically drive to the
    cells it wants to sense, and those drives cost budget. So the bound it produces is
    achievable-in-principle, not merely optimistic.
    """

    name = "oracle_guided(BOUND)"

    def __init__(self):
        self._route = None
        self._true = None

    def bind(self, true_maze: Maze):
        """The driver never does this for real policies. That is the whole point."""
        self._true = true_maze
        p = truth_passable(true_maze)
        cost = time_to_goal(p, sorted(true_maze.goal_indices))
        _, path = extract_path(cost, p, (true_maze.start_x, true_maze.start_y),
                               NORTH, sorted(true_maze.goal_indices))
        self._route = path

    def reset(self, belief):
        pass

    def choose(self, belief, x, y, heading, targets):
        # Follow the true optimal route directly. An earlier version instead applied a
        # discount to route cells inside the ordinary greedy rule -- and produced numbers
        # identical to flood-fill to three decimals, because flood-fill ALREADY drives the
        # optimal route. The discount never changed a single decision. Identical output
        # from a supposedly different policy is the tell that a "bound" is not binding.
        route = self._route
        outbound = set(targets) != {self._true.start_index}

        seq = route if outbound else list(reversed(route))
        here = [i for i, (cx, cy, _) in enumerate(seq) if (cx, cy) == (x, y)]
        if here:
            i = here[0]
            if i + 1 < len(seq):
                nx, ny = seq[i + 1][0], seq[i + 1][1]
                for h in range(NUM_HEADINGS):
                    d = DIR_DELTA[h]
                    if (x + d["x"], y + d["y"]) == (nx, ny):
                        if belief.is_passable(x, y, h, optimistic=True):
                            return h

        # Off-route (or route exhausted): fall back to ordinary greedy travel so the mouse
        # can rejoin. The bound stays valid because rejoining costs it budget too.
        p = belief_passable(belief, optimistic=True)
        field = time_to_goal(p, targets, belief.width, belief.height)
        best = None
        for h in range(NUM_HEADINGS):
            if not belief.is_passable(x, y, h, optimistic=True):
                continue
            d = DIR_DELTA[h]
            nx, ny = x + d["x"], y + d["y"]
            if not belief.in_bounds(nx, ny):
                continue
            c = float(field[h, ny * belief.width + nx])
            if c >= UNREACHABLE:
                continue
            s = turn_cost(heading, h) + T_CELL + c
            if best is None or s < best[0] - 1e-12:
                best = (s, h)
        return None if best is None else best[1]


def run(n_mazes: int = 100, seed0: int = 0):
    mazes = []
    for i in range(n_mazes):
        random.seed(seed0 + i)
        mazes.append(generate_maze())

    print(f"\n{'budget':>8}  {'policy':<24}{'expl-gap':>10}{'median':>9}{'p90':>8}"
          f"{'known':>7}{'no-route':>10}")

    table = {}
    for b in BUDGETS:
        for factory in (FloodFillPolicy, OptimisticShortcutPolicy, OracleGuidedPolicy):
            gaps, known, fail = [], [], 0
            for m in mazes:
                pol = factory()
                if isinstance(pol, OracleGuidedPolicy):
                    pol.bind(m)
                r = run_episode(m.clone(), pol, EpisodeConfig(search_time_budget=b))
                known.append(r.known_fraction)
                if r.fast_run_time >= NO_ROUTE:
                    fail += 1
                    # Failures are NOT dropped: excluding them flatters tight budgets,
                    # which is exactly where they happen. Charged as a large gap instead.
                    gaps.append(5.0)
                else:
                    gaps.append(r.exploration_gap)

            g = np.array(gaps)
            name = factory().name
            print(f"{str(b):>8}  {name:<24}{g.mean():>10.3f}{np.median(g):>9.3f}"
                  f"{np.percentile(g, 90):>8.3f}{np.mean(known):>7.2f}{fail:>10}")
            table[(b, name)] = g
        print()

    verdict(table)
    return table


def verdict(table):
    print("closable headroom = classical gap - oracle gap (paired, same mazes):")
    found = False
    for b in BUDGETS:
        cls = table.get((b, "flood_fill"))
        orc = table.get((b, "oracle_guided(BOUND)"))
        if cls is None or orc is None:
            continue
        d = cls - orc
        se = d.std(ddof=1) / np.sqrt(len(d))
        real = d.mean() > 2 * se
        found |= bool(real)
        print(f"  budget {str(b):>5}: {d.mean():+.3f} +/- {2*se:.3f} (95% CI)"
              f"{'   CLOSABLE' if real else ''}")

    print()
    if found:
        print(">> Real headroom: perfect exploration decisions beat the classical policy by")
        print(">> a margin larger than noise. Phases 5-6 have a concrete target. Report")
        print(">> classical / learned / oracle-bound on the same axes -- the bound tells a")
        print(">> reviewer how much of the available gap the learned method actually took.")
    else:
        print(">> No closable headroom. Even perfect route knowledge cannot beat the")
        print(">> classical policy under budget: the mouse still has to physically drive to")
        print(">> the cells it wants to sense, and the clock, not the decision rule, is the")
        print(">> binding constraint. The finding is that budgeted exploration is")
        print(">> SENSOR-BOUND, not policy-bound -- and no learned method can change that.")
        print(">> Write that paper. It is a stronger negative result than a failed DQN.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    run(n)
    print(f"\n({time.time() - t0:.0f}s)")
