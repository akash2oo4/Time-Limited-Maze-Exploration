"""
test_belief.py
Tests for BeliefMap and the sensor model. Run: python3 test_belief.py
"""

import random

import numpy as np

from maze import Maze, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST, DIR_DELTA
from belief import (BeliefMap, sense, fresh_belief,
                    UNKNOWN, OPEN, WALL, NORTH, EAST, SOUTH, WEST)
from dataset_generator import generate_maze

_FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        _FAILS.append(name)


# ---------------------------------------------------------------------------

def test_shared_edge_identity():
    """A cell's north wall and its neighbour's south wall must be ONE stored value."""
    b = BeliefMap()
    b.set(5, 5, NORTH, WALL)
    check("north of (5,5) == south of (5,4)",
          b.get(5, 4, SOUTH) == WALL, f"got {b.get(5,4,SOUTH)}")

    b.set(5, 4, SOUTH, OPEN)
    check("writing from the other side updates the same edge",
          b.get(5, 5, NORTH) == OPEN, f"got {b.get(5,5,NORTH)}")

    b.set(5, 5, EAST, WALL)
    check("east of (5,5) == west of (6,5)",
          b.get(6, 5, WEST) == WALL, f"got {b.get(6,5,WEST)}")


def test_initial_state():
    b = BeliefMap()
    check("outer boundary starts known", b.get(0, 0, NORTH) == WALL and b.get(0, 0, WEST) == WALL)
    check("interior starts unknown", b.get(5, 5, NORTH) == UNKNOWN)

    interior_h = b.h_edges[1:16, :]
    interior_v = b.v_edges[:, 1:16]
    check("every interior edge unknown at t=0",
          bool((interior_h == UNKNOWN).all() and (interior_v == UNKNOWN).all()))


def test_optimistic_vs_conservative():
    b = BeliefMap()
    b.set(5, 5, NORTH, UNKNOWN)
    check("unknown is passable when optimistic", b.is_passable(5, 5, NORTH, True))
    check("unknown is blocked when conservative", not b.is_passable(5, 5, NORTH, False))

    b.set(5, 5, EAST, WALL)
    check("known wall blocks even when optimistic", not b.is_passable(5, 5, EAST, True))

    b.set(5, 5, WEST, OPEN)
    check("known open passes when conservative", b.is_passable(5, 5, WEST, False))


def test_forward_lookahead_when_clear():
    """Way ahead open -> the beam carries into the next cell. Two boundaries resolved."""
    m = Maze()
    m.clear(True)
    x, y = 5, 5
    m.set_wall(x, y, WALL_NORTH, False)          # front of current cell open
    m.set_wall(x, y - 1, WALL_NORTH, True)       # far wall of the cell ahead

    b = BeliefMap()
    sense(m, b, x, y, NORTH)

    check("front wall of current cell resolved", b.get(x, y, NORTH) == OPEN)
    check("far wall of NEXT cell resolved (lookahead)",
          b.get(x, y - 1, NORTH) == WALL, f"got {b.get(x, y-1, NORTH)}")


def test_no_lookahead_when_blocked():
    """Way ahead walled -> beam stops. The next cell stays unknown."""
    m = Maze()
    m.clear(True)
    x, y = 5, 5
    m.set_wall(x, y, WALL_NORTH, True)           # front blocked
    m.set_wall(x, y - 1, WALL_NORTH, False)      # would be visible if beam passed

    b = BeliefMap()
    sense(m, b, x, y, NORTH)

    check("front wall resolved as WALL", b.get(x, y, NORTH) == WALL)
    check("cell beyond a wall stays UNKNOWN",
          b.get(x, y - 1, NORTH) == UNKNOWN, f"got {b.get(x, y-1, NORTH)}")


def test_nothing_behind():
    m = Maze()
    m.clear(True)
    x, y = 5, 5
    m.set_wall(x, y, WALL_NORTH, False)
    b = BeliefMap()
    sense(m, b, x, y, NORTH)
    check("wall behind the mouse stays UNKNOWN",
          b.get(x, y, SOUTH) == UNKNOWN, f"got {b.get(x, y, SOUTH)}")


def test_sides_only_current_cell():
    m = Maze()
    m.clear(True)
    x, y = 5, 5
    m.set_wall(x, y, WALL_NORTH, False)
    b = BeliefMap()
    sense(m, b, x, y, NORTH)
    check("left wall of current cell resolved", b.get(x, y, WEST) != UNKNOWN)
    check("right wall of current cell resolved", b.get(x, y, EAST) != UNKNOWN)
    check("side wall of the NEXT cell stays unknown",
          b.get(x, y - 1, WEST) == UNKNOWN, f"got {b.get(x, y-1, WEST)}")


def test_newly_resolved_counter():
    m = Maze()
    m.clear(True)
    m.set_wall(5, 5, WALL_NORTH, False)
    b = BeliefMap()
    first = sense(m, b, 5, 5, NORTH)
    again = sense(m, b, 5, 5, NORTH)
    check("first reading resolves edges", first > 0, f"got {first}")
    check("repeat reading resolves nothing new", again == 0, f"got {again}")


def test_belief_never_contradicts_truth():
    """Drive a long random walk; every resolved edge must match ground truth."""
    random.seed(42)
    m = generate_maze()
    belief, heading = fresh_belief(m)
    x, y = m.start_x, m.start_y

    for _ in range(3000):
        opts = list(belief.neighbors(x, y, optimistic=False))
        if not opts:
            break
        nx, ny, h = random.choice(opts)
        x, y, heading = nx, ny, h
        sense(m, belief, x, y, heading)

    bad = 0
    for yy in range(m.height):
        for xx in range(m.width):
            for h, flag in ((NORTH, WALL_NORTH), (EAST, WALL_EAST),
                            (SOUTH, WALL_SOUTH), (WEST, WALL_WEST)):
                s = belief.get(xx, yy, h)
                if s == UNKNOWN:
                    continue
                truth = WALL if m.has_wall(xx, yy, flag, False) else OPEN
                if s != truth:
                    bad += 1

    check("no resolved edge contradicts ground truth", bad == 0, f"{bad} mismatches")
    check("random walk actually learned something",
          belief.known_fraction() > 0.5, f"known={belief.known_fraction():.2f}")


def test_lookahead_is_worth_something():
    """
    The two-boundary sensor should resolve more of the maze per step than a
    one-cell-only sensor. If it does not, the model is not doing what the hardware does.
    """
    def walk(with_lookahead, seed):
        random.seed(seed)
        m = generate_maze()
        random.seed(seed + 10_000)
        belief = BeliefMap(m.width, m.height)
        x, y, heading = m.start_x, m.start_y, NORTH
        for _ in range(300):
            if with_lookahead:
                sense(m, belief, x, y, heading)
            else:
                belief.visited[y, x] = True
                for h in (heading, (heading + 1) % 4, (heading + 3) % 4):
                    flag = (WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST)[h]
                    belief.set(x, y, h, WALL if m.has_wall(x, y, flag, False) else OPEN)
            opts = list(belief.neighbors(x, y, optimistic=False))
            if not opts:
                break
            x, y, heading = random.choice(opts)
        return belief.known_fraction()

    with_la = np.mean([walk(True, s) for s in range(8)])
    without = np.mean([walk(False, s) for s in range(8)])
    check("forward lookahead maps more per step",
          with_la > without, f"with={with_la:.3f} without={without:.3f}")
    print(f"        (with lookahead {with_la:.3f} vs without {without:.3f} "
          f"after 300 steps)")


def test_channels_shape():
    m = generate_maze()
    belief, _ = fresh_belief(m)
    ch = belief.to_channels()
    check("channel tensor is [9,16,16]", ch.shape == (9, 16, 16), f"got {ch.shape}")
    overlap = (ch[0:4] * ch[4:8]).sum()
    check("wall and unknown planes never both set", overlap == 0, f"overlap={overlap}")


if __name__ == "__main__":
    for fn in [test_shared_edge_identity, test_initial_state,
               test_optimistic_vs_conservative, test_forward_lookahead_when_clear,
               test_no_lookahead_when_blocked, test_nothing_behind,
               test_sides_only_current_cell, test_newly_resolved_counter,
               test_belief_never_contradicts_truth, test_lookahead_is_worth_something,
               test_channels_shape]:
        print(f"\n{fn.__name__}")
        fn()

    print("\n" + ("ALL TESTS PASSED" if not _FAILS else f"{len(_FAILS)} FAILED: {_FAILS}"))
