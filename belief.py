"""
belief.py
Three-state, edge-keyed belief map + hardware-matched sensor model.

Why this file exists
--------------------
`Maze.walls` is ground truth: where the walls actually are. The mouse never gets to
read it. What the mouse has is a *belief* -- what it has personally checked so far --
and that needs three states per wall, not two:

    UNKNOWN  never checked
    OPEN     checked, no wall
    WALL     checked, wall present

A two-state bitmask (which is what `Maze.known_walls` is) cannot express UNKNOWN: an
unvisited region reads as wide open, so a planner will route straight through it. Every
optimistic-map solve, the stopping rule, and the CNN's known/unknown input channel
depend on the distinction, so it is built in here rather than bolted on later.

Each wall is stored ONCE, keyed by the wall itself rather than by cell. The wall between
(5,5) and (4,5) is one piece of plastic; storing it twice invites the two copies to drift
apart, which makes paths silently direction-dependent.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from maze import (Maze, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST, DIR_DELTA)

UNKNOWN = 0
OPEN = 1
WALL = 2

# Heading indices match DIR_DELTA: 0=N 1=E 2=S 3=W
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
_WALL_OF_HEADING = (WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST)


class BeliefMap:
    """
    Edge-keyed three-state wall knowledge.

    Storage is two arrays of canonical edges:
        h_edges[y][x]  the wall on the NORTH side of cell (x, y)   -> y in 0..H, x in 0..W-1
        v_edges[y][x]  the wall on the WEST  side of cell (x, y)   -> y in 0..H-1, x in 0..W

    A cell's south wall is just the next row's north edge, and its east wall is the next
    column's west edge, so every wall has exactly one home.
    """

    def __init__(self, width: int = 16, height: int = 16):
        self.width = width
        self.height = height

        # +1 in one dimension each: 17 rows of horizontal edges, 17 columns of vertical.
        self.h_edges = np.full((height + 1, width), UNKNOWN, dtype=np.uint8)
        self.v_edges = np.full((height, width + 1), UNKNOWN, dtype=np.uint8)

        self.visited = np.zeros((height, width), dtype=bool)

        # The mouse knows the maze has an outer boundary before it moves.
        self.h_edges[0, :] = WALL
        self.h_edges[height, :] = WALL
        self.v_edges[:, 0] = WALL
        self.v_edges[:, width] = WALL

    # ------------------------------------------------------------------
    # Edge addressing
    # ------------------------------------------------------------------

    def _edge_ref(self, x: int, y: int, heading: int) -> Tuple[np.ndarray, int, int]:
        """Return (array, row, col) for the wall on `heading` side of cell (x, y)."""
        if heading == NORTH:
            return self.h_edges, y, x
        if heading == SOUTH:
            return self.h_edges, y + 1, x
        if heading == WEST:
            return self.v_edges, y, x
        return self.v_edges, y, x + 1              # EAST

    def get(self, x: int, y: int, heading: int) -> int:
        """State of the wall on the `heading` side of cell (x, y)."""
        if not self.in_bounds(x, y):
            return WALL
        arr, r, c = self._edge_ref(x, y, heading)
        return int(arr[r, c])

    def set(self, x: int, y: int, heading: int, state: int) -> None:
        if not self.in_bounds(x, y):
            return
        arr, r, c = self._edge_ref(x, y, heading)
        arr[r, c] = state

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # ------------------------------------------------------------------
    # Queries used by the planners
    # ------------------------------------------------------------------

    def is_passable(self, x: int, y: int, heading: int, optimistic: bool) -> bool:
        """
        Can the mouse move from (x, y) in direction `heading`?

        optimistic=True  -> UNKNOWN counts as OPEN. This is the "assume every unchecked
                            wall is missing" solve: it yields the best time the maze could
                            possibly allow, which is what drives exploration targeting and
                            the stopping rule.
        optimistic=False -> UNKNOWN counts as WALL. The conservative solve: the best time
                            the mouse can actually guarantee with what it has checked.

        The two solves run on the same Dijkstra; only this flag differs. When they
        converge, there is nothing left worth exploring.
        """
        state = self.get(x, y, heading)
        if state == OPEN:
            return True
        if state == WALL:
            return False
        return optimistic

    def unknown_count(self) -> int:
        return int((self.h_edges == UNKNOWN).sum() + (self.v_edges == UNKNOWN).sum())

    def known_fraction(self) -> float:
        total = self.h_edges.size + self.v_edges.size
        return 1.0 - self.unknown_count() / total

    def neighbors(self, x: int, y: int, optimistic: bool) -> Iterator[Tuple[int, int, int]]:
        """Yield (nx, ny, heading) for each direction the mouse may move."""
        for h in range(4):
            if not self.is_passable(x, y, h, optimistic):
                continue
            d = DIR_DELTA[h]
            nx, ny = x + d["x"], y + d["y"]
            if self.in_bounds(nx, ny):
                yield nx, ny, h

    # ------------------------------------------------------------------
    # CNN input plane
    # ------------------------------------------------------------------

    def to_channels(self) -> np.ndarray:
        """
        Belief as CNN input: [9, H, W] float32.

        Planes 0-3: wall present (N, E, S, W)      -- 1.0 where known WALL
        Planes 4-7: wall unknown (N, E, S, W)      -- 1.0 where UNKNOWN
        Plane 8:    cell visited

        Two planes per direction rather than one three-valued plane: a single plane with
        values {0, 0.5, 1} would imply UNKNOWN sits numerically between OPEN and WALL,
        which is meaningless and something a convolution will try to interpolate across.
        """
        out = np.zeros((9, self.height, self.width), dtype=np.float32)
        for h in range(4):
            for y in range(self.height):
                for x in range(self.width):
                    s = self.get(x, y, h)
                    if s == WALL:
                        out[h, y, x] = 1.0
                    elif s == UNKNOWN:
                        out[4 + h, y, x] = 1.0
        out[8] = self.visited.astype(np.float32)
        return out

    def copy(self) -> "BeliefMap":
        b = BeliefMap(self.width, self.height)
        b.h_edges = self.h_edges.copy()
        b.v_edges = self.v_edges.copy()
        b.visited = self.visited.copy()
        return b


# ---------------------------------------------------------------------------
# Sensor model
# ---------------------------------------------------------------------------

def sense(true_maze: Maze, belief: BeliefMap, x: int, y: int, heading: int) -> int:
    """
    Reveal what the TQD-Micromouse-JL's six IR sensors can see from cell (x, y)
    facing `heading`, writing the result into `belief`. Returns the number of edges
    newly resolved (0 if the cell was already fully known -- useful for reward shaping
    and for logging exploration value).

    The sensor set is asymmetric, and that asymmetry matters:

      * front pair, calibrated out to ~165 mm -- longer than the 180 mm cell pitch, so
        when the way ahead is clear the beam carries into the next cell and reads the
        wall at its far end. Two boundaries of forward lookahead.
      * side pair (90 deg), effective to ~80 mm -- reads the current cell's own side
        walls and nothing beyond.
      * 45 deg pair -- short range, used by the firmware for corridor centring, not for
        mapping. Deliberately excluded; state this simplification in Methodology.

    Nothing is read behind the mouse.
    """
    if not belief.in_bounds(x, y):
        return 0

    newly = 0
    belief.visited[y, x] = True

    left = (heading + 3) % 4
    right = (heading + 1) % 4

    # --- current cell: front, left, right -------------------------------
    for h in (heading, left, right):
        truth = WALL if true_maze.has_wall(x, y, _WALL_OF_HEADING[h], False) else OPEN
        if belief.get(x, y, h) == UNKNOWN:
            newly += 1
        belief.set(x, y, h, truth)

    # --- forward lookahead: only if the way ahead is actually clear ------
    if belief.get(x, y, heading) == OPEN:
        d = DIR_DELTA[heading]
        ax, ay = x + d["x"], y + d["y"]
        if belief.in_bounds(ax, ay):
            truth = WALL if true_maze.has_wall(ax, ay, _WALL_OF_HEADING[heading], False) else OPEN
            if belief.get(ax, ay, heading) == UNKNOWN:
                newly += 1
            belief.set(ax, ay, heading, truth)

    return newly


def fresh_belief(true_maze: Maze) -> Tuple[BeliefMap, int]:
    """
    A belief map at the start of a run: outer boundary known, everything else unknown,
    then one sensor reading taken from the start cell.

    The IEEE start cell is sealed on three sides with its single exit facing north, so
    the mouse begins facing north by rule, not by choice.
    """
    belief = BeliefMap(true_maze.width, true_maze.height)
    sense(true_maze, belief, true_maze.start_x, true_maze.start_y, NORTH)
    return belief, NORTH
