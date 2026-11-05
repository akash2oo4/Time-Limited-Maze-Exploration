"""
solver.py
Turn-weighted time-optimal planning over (cell, heading) states.

One solver, three jobs, decided entirely by which `passable` function you hand it:

    truth_passable(maze)                -> the oracle. True optimal time on the full maze.
    belief_passable(belief, False)      -> conservative. Best time the mouse can guarantee.
    belief_passable(belief, True)       -> optimistic. Best time the maze could possibly
                                           allow if every unchecked wall turns out open.

The gap between the last two is exactly "how much is still worth exploring", which is
what drives both exploration targeting and the stopping rule.

Heading has to be part of the state. Without it there is no way to charge for a turn,
and the whole time-optimal claim silently collapses back to distance-optimal.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from maze import Maze, DIR_DELTA, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST
from belief import BeliefMap

# Kinematics live in dataset_generator so the labels and the planner cannot drift apart.
from dataset_generator import T_CELL, turn_cost, UNREACHABLE, NUM_HEADINGS

_WALL_OF_HEADING = (WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST)

# A `passable` function answers: may the mouse leave cell (x, y) heading `h`?
Passable = Callable[[int, int, int], bool]


def truth_passable(maze: Maze) -> Passable:
    def f(x: int, y: int, h: int) -> bool:
        return not maze.has_wall(x, y, _WALL_OF_HEADING[h], False)
    return f


def belief_passable(belief: BeliefMap, optimistic: bool) -> Passable:
    def f(x: int, y: int, h: int) -> bool:
        return belief.is_passable(x, y, h, optimistic)
    return f


# ---------------------------------------------------------------------------

def time_to_goal(passable: Passable,
                 goals: Sequence[int],
                 width: int = 16,
                 height: int = 16,
                 edge_bonus: Optional[Callable[[int, int, int], float]] = None
                 ) -> np.ndarray:
    """
    Dijkstra run backwards from the goal cells over (heading, cell) states.

    Returns float32 [4, width*height]: the minimum time from "sitting in that cell facing
    that heading" to any goal cell. Unreachable states hold UNREACHABLE.

    `edge_bonus(x, y, h)` optionally subtracts from the cost of the move out of (x, y)
    heading h. Used by the return trip to make routes through unmapped territory look
    cheaper than they are, so the mouse picks up information on a journey it has to make
    anyway. Keep it strictly smaller than T_CELL or costs go negative and Dijkstra breaks.
    """
    n = width * height
    dist = np.full((NUM_HEADINGS, n), math.inf, dtype=np.float64)
    pq: List[Tuple[float, int, int]] = []

    for g in goals:
        for h in range(NUM_HEADINGS):
            dist[h, g] = 0.0
            heapq.heappush(pq, (0.0, g, h))

    while pq:
        d, c2, h2 = heapq.heappop(pq)
        if d > dist[h2, c2]:
            continue

        # Who could have arrived at (c2, h2)? Whoever sat one cell back along h2.
        delta = DIR_DELTA[h2]
        x2, y2 = c2 % width, c2 // width
        px, py = x2 - delta["x"], y2 - delta["y"]
        if not (0 <= px < width and 0 <= py < height):
            continue
        if not passable(px, py, h2):
            continue

        step = T_CELL
        if edge_bonus is not None:
            step -= edge_bonus(px, py, h2)

        c = py * width + px
        for h in range(NUM_HEADINGS):
            nd = d + step + turn_cost(h, h2)
            if nd < dist[h, c]:
                dist[h, c] = nd
                heapq.heappush(pq, (nd, c, h))

    dist[~np.isfinite(dist)] = UNREACHABLE
    return dist.astype(np.float32)


def extract_path(cost: np.ndarray,
                 passable: Passable,
                 start: Tuple[int, int],
                 start_heading: int,
                 goals: Sequence[int],
                 width: int = 16,
                 height: int = 16,
                 max_steps: int = 2048
                 ) -> Tuple[float, List[Tuple[int, int, int]]]:
    """
    Walk downhill through a cost field produced by `time_to_goal`, using the SAME
    `passable` it was built with.

    Returns (total_time, [(x, y, heading_used_to_enter), ...]) with the start cell first.
    Total time is read straight off the cost field rather than re-summed, so it cannot
    disagree with the planner.
    """
    x, y = start
    h = start_heading
    c = y * width + x
    total = float(cost[h, c])
    path = [(x, y, h)]

    if total >= UNREACHABLE:
        return UNREACHABLE, path

    goal_set = set(goals)
    for _ in range(max_steps):
        if c in goal_set:
            return total, path

        best = None
        for h2 in range(NUM_HEADINGS):
            if not passable(x, y, h2):
                continue
            d = DIR_DELTA[h2]
            nx, ny = x + d["x"], y + d["y"]
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nc = ny * width + nx
            step = turn_cost(h, h2) + T_CELL + float(cost[h2, nc])
            if best is None or step < best[0] - 1e-12:
                best = (step, nx, ny, nc, h2)

        if best is None:
            break
        _, x, y, c, h = best
        path.append((x, y, h))

    return total, path


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def oracle_time(maze: Maze) -> float:
    """True optimal start-to-goal time on the fully known maze. The scoring reference."""
    p = truth_passable(maze)
    cost = time_to_goal(p, sorted(maze.goal_indices), maze.width, maze.height)
    return float(cost[0, maze.start_index])       # start faces NORTH by IEEE rule


def belief_time(belief: BeliefMap, maze: Maze, optimistic: bool) -> float:
    p = belief_passable(belief, optimistic)
    cost = time_to_goal(p, sorted(maze.goal_indices), belief.width, belief.height)
    return float(cost[0, maze.start_index])


def exploration_gap(belief: BeliefMap, maze: Maze) -> float:
    """
    Conservative best minus optimistic best.

    Reported as a diagnostic, NOT used as a stopping criterion -- see
    `unknown_edges_on_optimistic_path` for why. The optimistic bound stays wildly loose
    while any unknown edge survives anywhere near a plausible route, so this quantity only
    collapses once nearly the whole maze is mapped, which is precisely what the mouse is
    trying to avoid doing.
    """
    opt = belief_time(belief, maze, optimistic=True)
    con = belief_time(belief, maze, optimistic=False)
    if con >= UNREACHABLE:
        return UNREACHABLE
    return con - opt


def unknown_edges_on_optimistic_path(belief: BeliefMap,
                                     maze: Maze,
                                     start_heading: int = 0) -> int:
    """
    How many walls on the mouse's currently-believed-best route it has never actually
    checked. Returns -1 if no optimistic route exists at all.

    This is the criterion that works. The mouse does not care about unknowns in general --
    a corridor in the far corner is irrelevant no matter how unexplored. It cares only
    about unknowns on the route it intends to drive. When that count reaches zero, the
    optimistic plan is fully verified: it IS the conservative plan, optimism has nothing
    left to offer on this route, and further searching cannot buy back its 1/30 penalty.

    This is also why it terminates in a few hundred steps rather than mapping everything:
    verifying one route is a far smaller job than surveying 256 cells.
    """
    p = belief_passable(belief, optimistic=True)
    cost = time_to_goal(p, sorted(maze.goal_indices), belief.width, belief.height)
    total, path = extract_path(cost, p, (maze.start_x, maze.start_y),
                               start_heading, sorted(maze.goal_indices),
                               belief.width, belief.height)
    if total >= UNREACHABLE:
        return -1

    from belief import UNKNOWN
    unknown = 0
    for i in range(1, len(path)):
        px, py, _ = path[i - 1]
        if belief.get(px, py, path[i][2]) == UNKNOWN:
            unknown += 1
    return unknown
