"""
episode.py
The competition episode driver.

One attempt, in the order the rules impose:

    search run (one or more sorties)  ->  2 s stop  ->  fast run

    Score = FastRunTime + SearchTime / 30

A second saved on the fast run is worth thirty saved while exploring, which is why the
search penalty is small but not ignorable: exploration typically runs far longer than the
fast run, so it lands at a meaningful fraction of the final score. Reporting fast-run time
alone optimises the wrong thing.

The search run is structured as SORTIES -- out to the goal, back to the start -- repeated
until the intended route is verified. That mirrors the rules (the mouse must return to the
start cell before each fast run) and it is also what makes the thing terminate: each leg is
driven toward a concrete target, so it ends. An earlier design ran exploration as one
undirected phase; the mouse reached the goal, ran out of anything to aim at, and oscillated
in place until the step cap.

Everything pluggable sits behind two interfaces:

    ExplorationPolicy  - "which way do I drive next, given a target?"
    StopRule           - "have I explored enough?"   (fixed in Scope A, learned in Scope B)

Both read the belief map only. The driver is the sole thing that touches ground truth, and
only to answer sensor queries and execute moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from maze import Maze, DIR_DELTA
from belief import BeliefMap, sense, fresh_belief, OPEN, UNKNOWN, NORTH
from solver import (belief_passable, truth_passable, time_to_goal, extract_path,
                    oracle_time, exploration_gap, unknown_edges_on_optimistic_path)
from dataset_generator import T_CELL, turn_cost, UNREACHABLE, NUM_HEADINGS

SEARCH_PENALTY_DIVISOR = 30.0     # APEC/NTF handicap: 1/30 of accumulated search time
START_CELL_STOP = 2.0             # NTF s.3: full stop in the start cell before a fast run


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EpisodeConfig:
    # Stop once at most this many walls on the intended route remain unverified.
    # 0 means "refuse to drive anything you have not personally checked".
    stop_max_unknown_on_path: int = 0

    # Sorties are capped: a mouse that keeps finding new unknowns forever would burn its
    # whole maze-time allowance. Real mice rarely need more than two or three.
    max_sorties: int = 6

    # Seconds of pretend-discount on a move entering an unvisited cell. Makes information
    # look cheaper than it is, so the mouse detours to collect it on a journey it has to
    # make anyway. MUST stay below T_CELL or edge costs go negative and Dijkstra breaks.
    info_bonus: float = 0.04

    max_leg_steps: int = 600

    # Hard cap on simulated search seconds. None = explore until the route is verified.
    # A finite budget is what breaks the optimality guarantee below: the mouse is forced
    # to commit to whatever map it has, so exploration QUALITY starts to differ between
    # policies instead of only exploration TIME.
    search_time_budget: float = None

    # Stall fallback. A guided policy can drive the same out-and-back trip forever: it
    # avoids the cells the stop rule needs checked, so it never learns anything new and
    # the search only ends at max_sorties. If a whole sortie reveals no new walls, the
    # remaining sorties are driven by flood-fill. A flood-fill run never stalls this way,
    # so the rule leaves the classical baseline unchanged.
    stall_fallback: bool = True

    def __post_init__(self):
        if self.info_bonus >= T_CELL:
            raise ValueError(
                f"info_bonus ({self.info_bonus}) must be < T_CELL ({T_CELL:.4f}); "
                "Dijkstra requires non-negative edge costs")


@dataclass
class EpisodeResult:
    search_time: float
    fast_run_time: float
    score: float
    oracle_time: float                 # true optimum on the complete maze
    belief_best_time: float            # best achievable on the map the mouse ended up with
    reached_goal: bool
    sorties: int
    search_steps: int
    cells_visited: int
    known_fraction: float
    fast_path: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def exploration_gap(self) -> float:
        """Cost of an imperfect map. Blame the exploration policy."""
        return self.belief_best_time - self.oracle_time

    @property
    def planning_gap(self) -> float:
        """Cost of imperfect planning given that map. Blame the planner."""
        return self.fast_run_time - self.belief_best_time

    def summary(self) -> str:
        return (f"score {self.score:6.3f} = fast {self.fast_run_time:6.3f} "
                f"+ search {self.search_time:6.2f}/30 | oracle {self.oracle_time:6.3f} "
                f"expl {self.exploration_gap:+.3f} plan {self.planning_gap:+.3f} "
                f"known {self.known_fraction:.2f} sorties {self.sorties}")


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class ExplorationPolicy:
    """
    Decide the next move during the search run.

    `choose` returns a heading (0=N 1=E 2=S 3=W) to drive one cell, or None to give up.
    `targets` is the set of cell indices this leg aims for -- the goal cells on the way
    out, the start cell on the way back -- so one policy serves both legs.

    It may read only the belief map. Anything touching `true_maze` is cheating and will
    produce numbers that do not survive contact with hardware.

    Flood-fill, optimistic-shortcut, the CNN heuristic and the DQN all subclass this,
    which is what lets Phases 3, 5 and 6 reuse this driver unchanged.
    """

    name = "base"

    def reset(self, belief: BeliefMap) -> None:
        pass

    def choose(self, belief: BeliefMap, x: int, y: int, heading: int,
               targets: Sequence[int]) -> Optional[int]:
        raise NotImplementedError


class _CachedFieldPolicy(ExplorationPolicy):
    """
    Shared machinery: the turn-weighted cost field is recomputed only when the belief map
    has actually changed.

    Cache key is (unknown edges remaining, cells visited, targets) -- the first two change
    exactly when the mouse learns something, and both are cheap numpy reductions. Without
    this the driver runs a fresh 1024-node Dijkstra every single step and an episode takes
    ~15 s instead of well under one.
    """

    def __init__(self):
        self._cache: Dict[tuple, np.ndarray] = {}

    def reset(self, belief):
        self._cache.clear()

    def _key(self, belief, targets):
        return (belief.unknown_count(), int(belief.visited.sum()), tuple(targets))

    def _field(self, belief, targets, edge_bonus=None) -> np.ndarray:
        key = self._key(belief, targets)
        f = self._cache.get(key)
        if f is None:
            p = belief_passable(belief, optimistic=True)
            f = time_to_goal(p, targets, belief.width, belief.height, edge_bonus=edge_bonus)
            self._cache = {key: f}          # only the current belief matters; keep it small
        return f

    def _greedy(self, belief, x, y, heading, field, bonus_amount=0.0):
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
            step = T_CELL
            if bonus_amount and not belief.visited[ny, nx]:
                step -= bonus_amount
            s = turn_cost(heading, h) + step + c
            if best is None or s < best[0] - 1e-12:
                best = (s, h)
        return None if best is None else best[1]


class FloodFillPolicy(_CachedFieldPolicy):
    """
    The classical micromouse baseline: head for the target, treating unknown walls as open,
    recomputing whenever a new wall turns up.

    Purely greedy. It learns only about cells it happens to drive through, so the map ends
    up thin -- exactly the weakness the optimistic policy below is built to expose.
    """

    name = "flood_fill"

    def choose(self, belief, x, y, heading, targets):
        return self._greedy(belief, x, y, heading, self._field(belief, targets))


class OptimisticShortcutPolicy(_CachedFieldPolicy):
    """
    The strong classical baseline, and the one the DQN actually has to beat.

    Same optimistic solve, but moves entering an unvisited cell get a discount, so the
    mouse is pulled through unmapped territory that might hide a shortcut instead of only
    re-walking the route it already knows. Roughly what competitive mice do. Benchmarking
    RL against plain flood-fill alone is a strawman.
    """

    name = "optimistic_shortcut"

    def __init__(self, info_bonus: float = 0.04):
        super().__init__()
        if info_bonus >= T_CELL:
            raise ValueError("info_bonus must be < T_CELL to keep edge costs non-negative")
        self.info_bonus = info_bonus

    def choose(self, belief, x, y, heading, targets):
        field = self._field(belief, targets,
                            edge_bonus=_unvisited_bonus(belief, self.info_bonus))
        return self._greedy(belief, x, y, heading, field, bonus_amount=self.info_bonus)


def _unvisited_bonus(belief: BeliefMap, amount: float):
    def f(x: int, y: int, h: int) -> float:
        d = DIR_DELTA[h]
        nx, ny = x + d["x"], y + d["y"]
        if not belief.in_bounds(nx, ny):
            return 0.0
        return amount if not belief.visited[ny, nx] else 0.0
    return f


# ---------------------------------------------------------------------------
# Stopping rule (Scope A: fixed. Scope B swaps in a learned one here.)
# ---------------------------------------------------------------------------

StopRule = Callable[[BeliefMap, Maze, bool], bool]


def make_path_verified_stop_rule(max_unknown: int = 0) -> StopRule:
    """
    Stop once the route the mouse intends to drive is fully verified.

    Solve optimistically, take the resulting path, count how many of its walls the mouse
    has never checked. At zero, the optimistic and conservative plans are the same plan:
    optimism has nothing left to offer on this route, so more searching cannot buy back
    its 1/30 penalty.

    The tempting alternative -- stop when (conservative_best - optimistic_best) falls below
    a threshold -- does NOT work, and it is worth knowing why. The optimistic bound assumes
    every unchecked wall in the whole maze is missing, so it stays wildly loose while a
    single unknown edge survives near any plausible route. Measured: it never converged;
    the mouse hit its step cap having mapped 27 % of the maze, 723 s of search against a
    3.2 s fast run. Ask about the route, not about the maze.
    """
    def rule(belief: BeliefMap, maze: Maze, reached_goal: bool) -> bool:
        if not reached_goal:
            return False
        n = unknown_edges_on_optimistic_path(belief, maze, NORTH)
        return 0 <= n <= max_unknown
    return rule


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _step(true_maze: Maze, belief: BeliefMap, x: int, y: int,
          heading: int, new_heading: int) -> Tuple[int, int, int, float]:
    """Execute one physical move. Returns (x, y, heading, seconds_spent)."""
    d = DIR_DELTA[new_heading]
    nx, ny = x + d["x"], y + d["y"]

    # The mouse just drove through this edge, so it is open by demonstration. Recording it
    # is what lets the mouse retrace: the sensor never looks backwards.
    belief.set(x, y, new_heading, OPEN)

    dt = turn_cost(heading, new_heading) + T_CELL
    sense(true_maze, belief, nx, ny, new_heading)
    return nx, ny, new_heading, dt


def _drive_leg(true_maze, belief, x, y, heading, policy, targets, max_steps,
               budget_left=None):
    """Drive until a target cell is reached, the policy gives up, or a cap trips."""
    t = 0.0
    steps = 0
    target_set = set(targets)
    while steps < max_steps:
        if (y * belief.width + x) in target_set:
            break
        if budget_left is not None and t >= budget_left:
            break
        h2 = policy.choose(belief, x, y, heading, targets)
        if h2 is None:
            break
        x, y, heading, dt = _step(true_maze, belief, x, y, heading, h2)
        t += dt
        steps += 1
    return x, y, heading, t, steps


def run_episode(true_maze: Maze,
                policy: ExplorationPolicy,
                cfg: EpisodeConfig = EpisodeConfig(),
                stop_rule: Optional[StopRule] = None) -> EpisodeResult:
    goals = sorted(true_maze.goal_indices)
    start = [true_maze.start_index]
    stop_rule = stop_rule or make_path_verified_stop_rule(cfg.stop_max_unknown_on_path)

    belief, heading = fresh_belief(true_maze)
    policy.reset(belief)

    x, y = true_maze.start_x, true_maze.start_y
    search_time = 0.0
    search_steps = 0
    reached_goal = False
    sorties = 0

    # ---------------- search run: out-and-back sorties ----------------
    for _ in range(cfg.max_sorties):
        sorties += 1
        unknown_before = belief.unknown_count()

        left = None if cfg.search_time_budget is None else cfg.search_time_budget - search_time
        if left is not None and left <= 0:
            break
        x, y, heading, t, s = _drive_leg(true_maze, belief, x, y, heading,
                                         policy, goals, cfg.max_leg_steps, left)
        search_time += t
        search_steps += s
        if (y * belief.width + x) in true_maze.goal_indices:
            reached_goal = True

        left = None if cfg.search_time_budget is None else cfg.search_time_budget - search_time
        x, y, heading, t, s = _drive_leg(true_maze, belief, x, y, heading,
                                         policy, start, cfg.max_leg_steps, left)
        search_time += t
        search_steps += s

        if stop_rule(belief, true_maze, reached_goal):
            break

        if (cfg.stall_fallback and belief.unknown_count() == unknown_before
                and not isinstance(policy, FloodFillPolicy)):
            policy = FloodFillPolicy()
            policy.reset(belief)

    search_time += START_CELL_STOP

    # ---------------- fast run ----------------
    # Planned on the CONSERVATIVE belief: routing through a wall the mouse never verified
    # is how real mice crash. Unknown counts as wall here, by design.
    p_con = belief_passable(belief, optimistic=False)
    cost_con = time_to_goal(p_con, goals, belief.width, belief.height)
    belief_best, fast_path = extract_path(cost_con, p_con,
                                          (true_maze.start_x, true_maze.start_y),
                                          NORTH, goals, belief.width, belief.height)

    fast_run_time = belief_best
    if fast_run_time < UNREACHABLE:
        p_true = truth_passable(true_maze)
        for i in range(1, len(fast_path)):
            px, py, _ = fast_path[i - 1]
            if not p_true(px, py, fast_path[i][2]):
                raise AssertionError(
                    f"conservative plan crossed a real wall at ({px},{py}) heading "
                    f"{fast_path[i][2]} -- the belief map is corrupt")

    score = fast_run_time + search_time / SEARCH_PENALTY_DIVISOR

    return EpisodeResult(
        search_time=search_time,
        fast_run_time=fast_run_time,
        score=score,
        oracle_time=oracle_time(true_maze),
        belief_best_time=belief_best,
        reached_goal=reached_goal,
        sorties=sorties,
        search_steps=search_steps,
        cells_visited=int(belief.visited.sum()),
        known_fraction=belief.known_fraction(),
        fast_path=fast_path,
    )


# ---------------------------------------------------------------------------

def benchmark(policy_factories, n_mazes: int = 20, seed: int = 0, verbose: bool = True):
    """
    Run every policy over the SAME mazes and print the comparison.

    `policy_factories` is a list of zero-arg callables, so each maze gets a fresh policy
    instance and no state leaks between runs.
    """
    import random
    from dataset_generator import generate_maze

    rows: Dict[str, List[EpisodeResult]] = {}
    for i in range(n_mazes):
        random.seed(seed + i)
        m = generate_maze()
        for f in policy_factories:
            p = f()
            rows.setdefault(p.name, []).append(run_episode(m.clone(), p))

    if verbose:
        print(f"\n{'policy':<22}{'score':>8}{'fast':>8}{'search':>8}{'expl-gap':>10}"
              f"{'plan-gap':>10}{'known':>7}{'sortie':>8}{'goal%':>7}")
        for name, rs in rows.items():
            print(f"{name:<22}"
                  f"{np.mean([r.score for r in rs]):>8.3f}"
                  f"{np.mean([r.fast_run_time for r in rs]):>8.3f}"
                  f"{np.mean([r.search_time for r in rs]):>8.2f}"
                  f"{np.mean([r.exploration_gap for r in rs]):>10.3f}"
                  f"{np.mean([r.planning_gap for r in rs]):>10.3f}"
                  f"{np.mean([r.known_fraction for r in rs]):>7.2f}"
                  f"{np.mean([r.sorties for r in rs]):>8.2f}"
                  f"{100 * np.mean([r.reached_goal for r in rs]):>7.0f}")
    return rows


if __name__ == "__main__":
    benchmark([FloodFillPolicy, OptimisticShortcutPolicy], n_mazes=20)
