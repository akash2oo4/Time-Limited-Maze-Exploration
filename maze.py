"""
maze.py
16x16 IEEE Standard Micromouse Maze Data Structure in Python

IEEE Rules:
- 16x16 grid (256 cells) of equal unit squares
- Start cell: (0, 15) bottom-left with starting enclosure (exit North)
- Goal: 4 center cells (7,7), (8,7), (7,8), (8,8) with no interior walls and exactly 1 entrance
- Bitmask walls: NORTH=1, EAST=2, SOUTH=4, WEST=8
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# IEEE Micromouse physical specification. The solver is purely topological, but
# these are the dimensions the grid represents and they are reported to the UI.
# ---------------------------------------------------------------------------
GRID_SIZE = 16                                             # 16 x 16 unit squares
CELL_SIZE_CM = 18.0                                        # 18 cm x 18 cm per unit cell
WALL_THICKNESS_CM = 1.2                                    # wall thickness
WALL_HEIGHT_CM = 5.0                                       # wall height
WALL_HEIGHT_TOLERANCE = 0.05                               # +/- 5 %
CORRIDOR_WIDTH_CM = CELL_SIZE_CM - WALL_THICKNESS_CM       # 16.8 cm clear path

IEEE_SPEC = {
    "grid": f"{GRID_SIZE} x {GRID_SIZE} unit squares",
    "cell_size_cm": CELL_SIZE_CM,
    "wall_thickness_cm": WALL_THICKNESS_CM,
    "wall_height_cm": WALL_HEIGHT_CM,
    "wall_height_tolerance_cm": round(WALL_HEIGHT_CM * WALL_HEIGHT_TOLERANCE, 3),
    "corridor_width_cm": CORRIDOR_WIDTH_CM,
    "goal_zone": "2 x 2 central block: (7,7), (8,7), (7,8), (8,8)",
    "goal_entrances": 1,
    "wall_follower_can_reach_goal": False,
}

WALL_NORTH = 1
WALL_EAST = 2
WALL_SOUTH = 4
WALL_WEST = 8

DIR_DELTA = [
    {"x": 0, "y": -1, "wall": WALL_NORTH, "opp": WALL_SOUTH, "name": "NORTH"},
    {"x": 1, "y": 0,  "wall": WALL_EAST,  "opp": WALL_WEST,  "name": "EAST"},
    {"x": 0, "y": 1,  "wall": WALL_SOUTH, "opp": WALL_NORTH, "name": "SOUTH"},
    {"x": -1, "y": 0, "wall": WALL_WEST,  "opp": WALL_EAST,  "name": "WEST"}
]

class Maze:
    def __init__(self, width: int = 16, height: int = 16):
        self.width = width
        self.height = height
        self.total_cells = width * height

        # Ground truth walls (bitmask)
        self.walls = [0] * self.total_cells

        # Discovered / known walls in Fog-of-War exploration
        self.known_walls = [0] * self.total_cells
        self.visited = [False] * self.total_cells

        # Standard IEEE Start (bottom-left)
        self.start_x = 0
        self.start_y = height - 1
        self.start_index = self.coord_to_index(self.start_x, self.start_y)

        # Standard IEEE 4-cell Center Goals
        mid_x, mid_y = width // 2, height // 2
        self.goals = [
            {"x": mid_x - 1, "y": mid_y - 1, "index": self.coord_to_index(mid_x - 1, mid_y - 1)},
            {"x": mid_x,     "y": mid_y - 1, "index": self.coord_to_index(mid_x,     mid_y - 1)},
            {"x": mid_x - 1, "y": mid_y,     "index": self.coord_to_index(mid_x - 1, mid_y)},
            {"x": mid_x,     "y": mid_y,     "index": self.coord_to_index(mid_x,     mid_y)}
        ]
        self.goal_indices = {g["index"] for g in self.goals}

        self.init_perimeter()

    def coord_to_index(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        return -1

    def index_to_coord(self, index: int) -> Tuple[int, int]:
        return (index % self.width, index // self.width)

    def is_goal(self, x: int, y: int) -> bool:
        idx = self.coord_to_index(x, y)
        return idx in self.goal_indices

    def init_perimeter(self):
        """Initialize outer perimeter boundary walls without wiping internal walls"""
        for y in range(self.height):
            for x in range(self.width):
                idx = self.coord_to_index(x, y)
                if y == 0:
                    self.walls[idx] |= WALL_NORTH
                    self.known_walls[idx] |= WALL_NORTH
                if y == self.height - 1:
                    self.walls[idx] |= WALL_SOUTH
                    self.known_walls[idx] |= WALL_SOUTH
                if x == 0:
                    self.walls[idx] |= WALL_WEST
                    self.known_walls[idx] |= WALL_WEST
                if x == self.width - 1:
                    self.walls[idx] |= WALL_EAST
                    self.known_walls[idx] |= WALL_EAST

        # Start cell IEEE standard: Eastern wall closed so mouse is forced forward (North)
        self.set_wall(0, self.height - 1, WALL_EAST, True)

    def clear(self, keep_perimeter: bool = True):
        self.walls = [0] * self.total_cells
        self.known_walls = [0] * self.total_cells
        self.visited = [False] * self.total_cells
        if keep_perimeter:
            self.init_perimeter()

    def set_wall(self, x: int, y: int, wall_flag: int, state: bool = True):
        idx = self.coord_to_index(x, y)
        if idx < 0:
            return

        if state:
            self.walls[idx] |= wall_flag
        else:
            self.walls[idx] &= ~wall_flag

        # Synchronize with adjacent neighbor
        for d in DIR_DELTA:
            if d["wall"] == wall_flag:
                nx, ny = x + d["x"], y + d["y"]
                n_idx = self.coord_to_index(nx, ny)
                if n_idx >= 0:
                    if state:
                        self.walls[n_idx] |= d["opp"]
                    else:
                        self.walls[n_idx] &= ~d["opp"]
                break

    def has_wall(self, x: int, y: int, wall_flag: int, use_known: bool = False) -> bool:
        idx = self.coord_to_index(x, y)
        if idx < 0:
            return True
        w = self.known_walls[idx] if use_known else self.walls[idx]
        return (w & wall_flag) != 0

    def toggle_wall(self, x: int, y: int, wall_flag: int):
        curr = self.has_wall(x, y, wall_flag)
        self.set_wall(x, y, wall_flag, not curr)
        idx = self.coord_to_index(x, y)
        self.known_walls[idx] = self.walls[idx]

    def sense_and_discover(self, x: int, y: int):
        """Simulate hardware IR sensors discovering surrounding walls"""
        idx = self.coord_to_index(x, y)
        if idx < 0:
            return
        self.visited[idx] = True
        self.known_walls[idx] = self.walls[idx]

        for d in DIR_DELTA:
            nx, ny = x + d["x"], y + d["y"]
            n_idx = self.coord_to_index(nx, ny)
            if n_idx >= 0:
                if (self.walls[idx] & d["wall"]) != 0:
                    self.known_walls[n_idx] |= d["opp"]
                else:
                    self.known_walls[n_idx] &= ~d["opp"]

    def reset_exploration(self):
        self.known_walls = [0] * self.total_cells
        self.visited = [False] * self.total_cells
        for y in range(self.height):
            for x in range(self.width):
                idx = self.coord_to_index(x, y)
                w = 0
                if y == 0: w |= WALL_NORTH
                if y == self.height - 1: w |= WALL_SOUTH
                if x == 0: w |= WALL_WEST
                if x == self.width - 1: w |= WALL_EAST
                self.known_walls[idx] = w
        self.sense_and_discover(self.start_x, self.start_y)

    def compute_ground_truth_bfs(self, use_known: bool = False) -> List[int]:
        dist = [9999] * self.total_cells
        queue = []

        for g in self.goals:
            dist[g["index"]] = 0
            queue.append(g["index"])

        head = 0
        while head < len(queue):
            curr_idx = queue[head]
            head += 1
            curr_dist = dist[curr_idx]
            cx, cy = self.index_to_coord(curr_idx)

            for d in DIR_DELTA:
                if not self.has_wall(cx, cy, d["wall"], use_known):
                    nx, ny = cx + d["x"], cy + d["y"]
                    n_idx = self.coord_to_index(nx, ny)
                    if n_idx >= 0 and dist[n_idx] > curr_dist + 1:
                        dist[n_idx] = curr_dist + 1
                        queue.append(n_idx)
        return dist

    def to_string_format(self) -> str:
        lines = []
        top = "+" + "---+" * self.width
        lines.append(top)

        for y in range(self.height):
            row_str = "|"
            bottom_str = "+"
            for x in range(self.width):
                row_str += "   "
                row_str += "|" if self.has_wall(x, y, WALL_EAST) else " "
                bottom_str += "---+" if self.has_wall(x, y, WALL_SOUTH) else "   +"
            lines.append(row_str)
            lines.append(bottom_str)
        return "\n".join(lines)

    def load_from_string_format(self, text: str) -> bool:
        self.clear(False)
        raw_lines = [l for l in text.splitlines() if l.strip()]
        if len(raw_lines) < self.height * 2 + 1:
            return False

        for y in range(self.height):
            top_line = raw_lines[y * 2]
            cell_line = raw_lines[y * 2 + 1]
            bottom_line = raw_lines[y * 2 + 2]

            for x in range(self.width):
                col = x * 4
                idx = self.coord_to_index(x, y)
                if col + 2 < len(top_line) and (top_line[col + 1] == '-' or top_line[col + 2] == '-'):
                    self.walls[idx] |= WALL_NORTH
                if col + 2 < len(bottom_line) and (bottom_line[col + 1] == '-' or bottom_line[col + 2] == '-'):
                    self.walls[idx] |= WALL_SOUTH
                if col < len(cell_line) and cell_line[col] == '|':
                    self.walls[idx] |= WALL_WEST
                if col + 4 < len(cell_line) and cell_line[col + 4] == '|':
                    self.walls[idx] |= WALL_EAST

        self.reset_exploration()
        return True

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "walls": list(self.walls),
            "start": {"x": self.start_x, "y": self.start_y, "index": self.start_index},
            "goals": self.goals,
            "spec": dict(IEEE_SPEC),
            "physical_extent_cm": {
                "width": round(self.width * CELL_SIZE_CM, 2),
                "height": round(self.height * CELL_SIZE_CM, 2)
            }
        }

    def clone(self) -> 'Maze':
        copy_maze = Maze(self.width, self.height)
        copy_maze.walls = list(self.walls)
        copy_maze.known_walls = list(self.known_walls)
        copy_maze.visited = list(self.visited)
        return copy_maze
