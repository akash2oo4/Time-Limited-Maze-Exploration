"""
maze_generator.py
IEEE Standard Micromouse Random Maze Generator in Python

Rules enforced on every generated maze:
1. 16x16 grid with a complete outer perimeter wall.
2. Start cell at (0, 15) sealed on East/South/West, single exit facing North.
3. 2x2 centre goal (7,7)/(8,7)/(7,8)/(8,8) with no interior walls and exactly ONE entrance.
4. Guaranteed solvability: every cell reachable, no isolated pockets.
5. Anti-wall-hugging guard: the goal ring is a free-standing wall island, so a simple
   left-hand or right-hand wall-following robot can never reach the centre.
6. Controllable loop density to provide multiple alternative routes.
"""

import random
from maze import Maze, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST, DIR_DELTA

# Turn offsets in preference order for a wall-following robot.
_WALL_FOLLOWER_ORDER = {
    "left": (3, 0, 1, 2),   # left, straight, right, back
    "right": (1, 0, 3, 2),  # right, straight, left, back
}


class IEEEMazeGenerator:
    # ------------------------------------------------------------------
    # Wall-segment bookkeeping
    #
    # A wall lives on the lattice of posts, not on a cell. Post (i, j) is the
    # top-left corner of cell (i, j). A segment is keyed canonically as
    # ('H', x, y) for the horizontal segment from post (x, y) to (x+1, y), and
    # ('V', x, y) for the vertical segment from post (x, y) to (x, y+1).
    # ------------------------------------------------------------------

    @staticmethod
    def goal_ring_segments(maze: Maze):
        """The 8 wall segments that enclose the 2x2 centre goal."""
        mx, my = maze.width // 2, maze.height // 2  # 8, 8
        return {
            ('H', mx - 1, my - 1),  # north of (7,7)
            ('H', mx,     my - 1),  # north of (8,7)
            ('V', mx + 1, my - 1),  # east  of (8,7)
            ('V', mx + 1, my),      # east  of (8,8)
            ('H', mx,     my + 1),  # south of (8,8)
            ('H', mx - 1, my + 1),  # south of (7,8)
            ('V', mx - 1, my),      # west  of (7,8)
            ('V', mx - 1, my - 1),  # west  of (7,7)
        }

    @staticmethod
    def goal_ring_posts(maze: Maze):
        """The 8 lattice posts on the goal ring (the centre post is excluded)."""
        mx, my = maze.width // 2, maze.height // 2
        return [(i, j)
                for i in (mx - 1, mx, mx + 1)
                for j in (my - 1, my, my + 1)
                if not (i == mx and j == my)]

    @staticmethod
    def clear_segment(maze: Maze, seg):
        """Remove one canonical wall segment from the maze."""
        kind, x, y = seg
        if kind == 'H':
            maze.set_wall(x, y, WALL_NORTH, False)
        else:
            maze.set_wall(x, y, WALL_WEST, False)

    @classmethod
    def isolate_goal_island(cls, maze: Maze):
        """
        Anti-wall-hugging guard.

        A wall-following robot traces the surface of whichever connected wall structure it
        is touching. If the goal ring shares even one post with a wall that chains back to
        the outer boundary, the hugger can walk that chain straight into the centre. So
        every wall segment that touches a ring post -- but is not part of the ring itself --
        is removed, leaving the ring as a free-standing island the hugger can never reach.

        Only removes walls, so reachability and the single-entrance rule are untouched.
        """
        ring = cls.goal_ring_segments(maze)
        for (i, j) in cls.goal_ring_posts(maze):
            incident = [
                ('V', i, j - 1),  # upward
                ('V', i, j),      # downward
                ('H', i - 1, j),  # leftward
                ('H', i, j),      # rightward
            ]
            for seg in incident:
                if seg in ring:
                    continue
                kind, x, y = seg
                if not (0 <= x < maze.width and 0 <= y < maze.height):
                    continue
                cls.clear_segment(maze, seg)

    @staticmethod
    def wall_follower_reaches_goal(maze: Maze, hand: str = "left", max_steps: int = 4000) -> bool:
        """Simulate a naive wall-hugging robot. Must return False for a legal IEEE maze."""
        order = _WALL_FOLLOWER_ORDER[hand]
        x, y, heading = maze.start_x, maze.start_y, 0
        seen = set()

        for _ in range(max_steps):
            if maze.is_goal(x, y):
                return True
            state = (x, y, heading)
            if state in seen:
                return False  # closed circuit: it will never enter the centre
            seen.add(state)

            for turn in order:
                nh = (heading + turn) % 4
                d = DIR_DELTA[nh]
                if not maze.has_wall(x, y, d["wall"], False):
                    x, y, heading = x + d["x"], y + d["y"], nh
                    break
            else:
                return False  # fully boxed in
        return False

    @classmethod
    def is_wall_hugger_proof(cls, maze: Maze) -> bool:
        return not (cls.wall_follower_reaches_goal(maze, "left")
                    or cls.wall_follower_reaches_goal(maze, "right"))

    # ------------------------------------------------------------------
    # Structural setup
    # ------------------------------------------------------------------

    @staticmethod
    def setup_center_goal(maze: Maze) -> dict:
        """
        Set up the 2x2 IEEE centre goal at (7,7), (8,7), (7,8), (8,8):
        strip every interior wall and open exactly ONE entrance.
        """
        mid_x, mid_y = maze.width // 2, maze.height // 2  # 8, 8
        g00 = maze.coord_to_index(mid_x - 1, mid_y - 1)   # (7,7)
        g10 = maze.coord_to_index(mid_x,     mid_y - 1)   # (8,7)
        g01 = maze.coord_to_index(mid_x - 1, mid_y)       # (7,8)
        g11 = maze.coord_to_index(mid_x,     mid_y)       # (8,8)

        # Clear all internal walls inside the 2x2 box
        maze.walls[g00] = (maze.walls[g00] | (WALL_NORTH | WALL_WEST)) & ~(WALL_EAST | WALL_SOUTH)
        maze.walls[g10] = (maze.walls[g10] | (WALL_NORTH | WALL_EAST)) & ~(WALL_WEST | WALL_SOUTH)
        maze.walls[g01] = (maze.walls[g01] | (WALL_SOUTH | WALL_WEST)) & ~(WALL_EAST | WALL_NORTH)
        maze.walls[g11] = (maze.walls[g11] | (WALL_SOUTH | WALL_EAST)) & ~(WALL_WEST | WALL_NORTH)

        # Seal the outer ring of the 2x2 goal box
        maze.set_wall(mid_x - 1, mid_y - 1, WALL_NORTH, True)
        maze.set_wall(mid_x,     mid_y - 1, WALL_NORTH, True)
        maze.set_wall(mid_x - 1, mid_y,     WALL_SOUTH, True)
        maze.set_wall(mid_x,     mid_y,     WALL_SOUTH, True)
        maze.set_wall(mid_x - 1, mid_y - 1, WALL_WEST,  True)
        maze.set_wall(mid_x - 1, mid_y,     WALL_WEST,  True)
        maze.set_wall(mid_x,     mid_y - 1, WALL_EAST,  True)
        maze.set_wall(mid_x,     mid_y,     WALL_EAST,  True)

        # Open exactly 1 entrance out of the 8 ring walls
        possible_entrances = [
            {"x": mid_x - 1, "y": mid_y - 1, "wall": WALL_NORTH},
            {"x": mid_x,     "y": mid_y - 1, "wall": WALL_NORTH},
            {"x": mid_x,     "y": mid_y - 1, "wall": WALL_EAST},
            {"x": mid_x,     "y": mid_y,     "wall": WALL_EAST},
            {"x": mid_x,     "y": mid_y,     "wall": WALL_SOUTH},
            {"x": mid_x - 1, "y": mid_y,     "wall": WALL_SOUTH},
            {"x": mid_x - 1, "y": mid_y,     "wall": WALL_WEST},
            {"x": mid_x - 1, "y": mid_y - 1, "wall": WALL_WEST},
        ]
        chosen = random.choice(possible_entrances)
        maze.set_wall(chosen["x"], chosen["y"], chosen["wall"], False)
        return chosen

    @staticmethod
    def fill_solid(maze: Maze):
        """Reset to a fully walled grid, the starting point for the carver."""
        maze.clear(True)
        for i in range(maze.total_cells):
            maze.walls[i] = WALL_NORTH | WALL_EAST | WALL_SOUTH | WALL_WEST

    @staticmethod
    def prepare_start_enclosure(maze: Maze) -> int:
        """
        Build the IEEE start enclosure at (0, 15): East/South/West sealed, single exit North.

        This must run *before* carving, and the returned seed cell (the one north of the
        start) is where the carver has to begin. Sealing the start's East wall *after*
        carving -- as opposed to before -- can cut the start off from the spanning tree it
        was attached through, which silently produces unsolvable mazes.
        """
        sx, sy = maze.start_x, maze.start_y
        maze.set_wall(sx, sy, WALL_EAST, True)
        maze.set_wall(sx, sy, WALL_SOUTH, True)
        maze.set_wall(sx, sy, WALL_WEST, True)
        maze.set_wall(sx, sy, WALL_NORTH, False)  # mandatory forward exit
        return maze.coord_to_index(sx, sy - 1)

    @staticmethod
    def add_loops(maze: Maze, probability: float = 0.15):
        """Randomly add alternative bypass paths to create realistic tournament mazes"""
        goal_set = set(maze.goal_indices)
        for y in range(1, maze.height - 1):
            for x in range(1, maze.width - 1):
                idx = maze.coord_to_index(x, y)
                if idx in goal_set:
                    continue
                if random.random() >= probability:
                    continue

                # Never punch a bypass into the centre goal box: IEEE rules allow it
                # exactly one entrance, already chosen by setup_center_goal().
                choices = []
                for wall, dx, dy in ((WALL_EAST, 1, 0), (WALL_SOUTH, 0, 1)):
                    n_idx = maze.coord_to_index(x + dx, y + dy)
                    if n_idx >= 0 and n_idx not in goal_set:
                        choices.append(wall)
                if choices:
                    maze.set_wall(x, y, random.choice(choices), False)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @classmethod
    def carve_dfs(cls, maze: Maze, loop_prob: float):
        """One DFS recursive-backtracker pass, plus goal setup, loops and the guard."""
        cls.fill_solid(maze)

        visited = [False] * maze.total_cells
        goal_set = set(maze.goal_indices)

        # The start cell is pre-enclosed and already linked North, so the carver seeds
        # from that neighbour and never reconsiders the start itself.
        seed_idx = cls.prepare_start_enclosure(maze)
        visited[maze.start_index] = True
        visited[seed_idx] = True
        stack = [seed_idx]

        while stack:
            curr_idx = stack[-1]
            cx, cy = maze.index_to_coord(curr_idx)

            neighbors = []
            for d in DIR_DELTA:
                nx, ny = cx + d["x"], cy + d["y"]
                n_idx = maze.coord_to_index(nx, ny)
                if n_idx >= 0 and not visited[n_idx] and n_idx not in goal_set:
                    neighbors.append((nx, ny, n_idx, d["wall"], d["opp"]))

            if neighbors:
                nx, ny, n_idx, w_wall, w_opp = random.choice(neighbors)
                maze.walls[curr_idx] &= ~w_wall
                maze.walls[n_idx] &= ~w_opp
                visited[n_idx] = True
                stack.append(n_idx)
            else:
                stack.pop()

        cls.setup_center_goal(maze)
        if loop_prob > 0:
            cls.add_loops(maze, loop_prob)

        # Must run last: it only removes walls, and it needs the final wall layout.
        cls.isolate_goal_island(maze)

    @classmethod
    def generate_random_dfs(cls, maze: Maze, loop_prob: float = 0.14, max_attempts: int = 20) -> Maze:
        """
        Generate a random IEEE-legal maze via DFS recursive backtracker.

        Retries until the wall-hugger guard verifies, so the returned maze is guaranteed
        to satisfy every rule rather than merely being likely to.
        """
        for _ in range(max_attempts):
            cls.carve_dfs(maze, loop_prob)
            if cls.is_wall_hugger_proof(maze):
                maze.reset_exploration()
                return maze

        # Should not happen in practice; surface it rather than return an illegal maze.
        raise RuntimeError(
            f"Could not generate a wall-hugger-proof IEEE maze in {max_attempts} attempts"
        )
