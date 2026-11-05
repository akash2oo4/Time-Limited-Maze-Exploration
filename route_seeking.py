"""
route_seeking.py
Phase 5b: use the route head as a SENSING TARGET, not an edge discount.

WHY THE FIRST POLICY DID NOT WORK
---------------------------------
The route head is good: AUC 0.948 overall, and 0.909 in the earliest third of the
trajectory, when ~85 % of the maze is still unknown. It ranks route cells above non-route
cells almost always, and it does so early, when the budget can still be spent well.

But `CNNExplorationPolicy` used it as a small discount on the cost of entering a cell, and
that produced numbers IDENTICAL to three decimals across a 4x sweep of `route_bonus`
(0.02 to 0.088). Measurement explains it: the margin between the best and second-best move
in the greedy rule has a median of 0.410 s, while `route_bonus` is capped below T_CELL
(0.09) to keep Dijkstra edge costs non-negative. A 0.088 bonus can flip only 3 % of
decisions. The head was informative and the mechanism could not act on it.

Identical output across a hyperparameter sweep is the same tell that exposed the broken
oracle earlier: if a knob changes nothing, it is not connected to anything.

WHAT THIS DOES INSTEAD
----------------------
Rather than nudging a goal-directed rule, the mouse picks an explicit destination: the
nearest unvisited cell the network believes lies on the true optimal route, and drives
there to sense it. When no such cell remains, it heads for the leg target.

That is structurally how the oracle achieves 0.25 coverage against flood-fill's 0.44 -- it
goes to the cells that matter and ignores the rest. With AUC 0.91 early, the network can
name those cells; this policy is what lets it act on them.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from maze import DIR_DELTA
from belief import BeliefMap
from dataset_generator import T_CELL, turn_cost, UNREACHABLE, NUM_HEADINGS
from solver import belief_passable, time_to_goal
from episode import ExplorationPolicy
from cnn_heuristic import MazeNet, DEVICE, belief_planes


class RouteSeekingPolicy(ExplorationPolicy):
    """
    Learned exploration by explicit sensing targets.

    Reads only the belief map -- same contract as every other policy.

    `route_threshold` sets how confident the network must be before a cell is worth a
    detour. `max_detour` caps how far the mouse will travel for one sensing target, so a
    single confident-but-distant cell cannot eat the whole budget.
    """

    name = "cnn_route_seeking"

    def __init__(self, ckpt: str = "cnn_heuristic.pt",
                 route_threshold: float = 0.5,
                 max_detour: float = 1.5):
        blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        self.net = MazeNet().to(DEVICE)
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.route_threshold = route_threshold
        self.max_detour = max_detour
        self._cache_key = None
        self._route = None

    def reset(self, belief):
        self._cache_key = None
        self._route = None

    # ------------------------------------------------------------------

    def _route_probs(self, belief: BeliefMap, x, y, heading) -> np.ndarray:
        """One forward pass per belief change, not per step."""
        key = (belief.unknown_count(), int(belief.visited.sum()))
        if key != self._cache_key:
            packed = np.concatenate([belief.h_edges.ravel(), belief.v_edges.ravel()])
            planes = belief_planes(packed, np.array([x, y, heading]))
            with torch.no_grad():
                _, pr = self.net(torch.from_numpy(planes)[None].to(DEVICE))
            self._route = torch.sigmoid(pr)[0].cpu().numpy()
            self._cache_key = key
        return self._route

    def _pick_targets(self, belief: BeliefMap, x, y, heading, leg_targets):
        """
        Unvisited cells the network is confident lie on the true optimal route, close
        enough to be worth the detour.

        Returns leg_targets unchanged when nothing qualifies, so the mouse always makes
        progress toward the goal (or home) rather than wandering.
        """
        route = self._route_probs(belief, x, y, heading)
        w = belief.width

        cand = np.argwhere((route > self.route_threshold) & (~belief.visited))
        if len(cand) == 0:
            return leg_targets, False

        # Distance to each candidate, on the optimistic belief, from where we stand.
        idxs = [int(cy) * w + int(cx) for cy, cx in cand]
        p = belief_passable(belief, optimistic=True)
        field = time_to_goal(p, idxs, w, belief.height)
        here = float(field[heading, y * w + x])
        if here >= UNREACHABLE or here > self.max_detour:
            return leg_targets, False

        return idxs, True

    def choose(self, belief, x, y, heading, targets):
        sub_targets, seeking = self._pick_targets(belief, x, y, heading, targets)

        p = belief_passable(belief, optimistic=True)
        field = time_to_goal(p, sub_targets, belief.width, belief.height)

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

        if best is not None:
            return best[1]

        # Sub-target unreachable on the optimistic map: fall back to the leg target so the
        # mouse never stalls.
        if seeking:
            field = time_to_goal(p, targets, belief.width, belief.height)
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


# ---------------------------------------------------------------------------

def sweep(n_mazes: int = 60, budget: float = 18.0):
    """
    Tune threshold and detour cap. Watch `known` as much as the gap.

    If coverage walks down toward the oracle's 0.25 as the threshold rises, the mechanism
    is connected and only needs a setting. If `known` stays pinned near flood-fill's 0.40
    regardless, this approach is no better than the discount and the problem is elsewhere.
    """
    import random
    from dataset_generator import generate_maze
    from episode import EpisodeConfig, run_episode, FloodFillPolicy

    print(f"budget {budget}s, {n_mazes} mazes\n")
    print(f"{'policy':<34}{'mean':>8}{'p90':>8}{'known':>8}{'no-route':>10}")

    def bench(factory, label):
        g, k, fail = [], [], 0
        for i in range(n_mazes):
            random.seed(i)
            m = generate_maze()
            r = run_episode(m.clone(), factory(), EpisodeConfig(search_time_budget=budget))
            k.append(r.known_fraction)
            if r.fast_run_time >= 1e5:
                fail += 1
                g.append(5.0)
            else:
                g.append(r.exploration_gap)
        g = np.array(g)
        print(f"{label:<34}{g.mean():>8.3f}{np.percentile(g, 90):>8.3f}"
              f"{np.mean(k):>8.2f}{fail:>10}")

    bench(FloodFillPolicy, "flood_fill (baseline)")
    for thr in (0.3, 0.5, 0.7, 0.9):
        for det in (1.0, 2.0):
            bench(lambda t=thr, d=det: RouteSeekingPolicy(route_threshold=t, max_detour=d),
                  f"route_seek thr={thr} detour={det}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    b = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0
    sweep(n, b)


# ---------------------------------------------------------------------------
# Third mechanism: route-weighted edge costs
# ---------------------------------------------------------------------------

class RouteWeightedPolicy(ExplorationPolicy):
    """
    Make NON-route cells expensive, rather than making route cells slightly cheap.

    Two mechanisms failed before this one, and both failures were informative:

      * additive bonus (`CNNExplorationPolicy`) -- capped below T_CELL to keep Dijkstra
        edge costs non-negative, so at most 0.088 against a median decision margin of
        0.410. It could flip 3 % of decisions and produced identical results across a 4x
        sweep.
      * explicit sensing targets (`RouteSeekingPolicy`) -- unbounded above, and it
        overcorrected: the mouse chased isolated route cells, coverage rose to 0.51 (worse
        than flood-fill's 0.40, against the oracle's 0.25) and it failed to find any route
        on 8 of 15 mazes. Chasing route cells is not the same as following the route.

    A multiplicative penalty avoids both traps. Entering an unvisited cell costs
    T_CELL * (1 + alpha * (1 - p_route)), so a confident non-route cell can cost several
    times a route cell -- enough to clear the 0.410 margin -- while the search stays
    goal-directed and every edge cost stays positive.
    """

    name = "cnn_route_weighted"

    def __init__(self, ckpt: str = "cnn_heuristic.pt", alpha: float = 3.0):
        blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        self.net = MazeNet().to(DEVICE)
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.alpha = alpha
        self._key = None
        self._route = None

    def reset(self, belief):
        self._key = None

    def _route_probs(self, belief, x, y, heading):
        key = (belief.unknown_count(), int(belief.visited.sum()))
        if key != self._key:
            packed = np.concatenate([belief.h_edges.ravel(), belief.v_edges.ravel()])
            planes = belief_planes(packed, np.array([x, y, heading]))
            with torch.no_grad():
                _, pr = self.net(torch.from_numpy(planes)[None].to(DEVICE))
            self._route = torch.sigmoid(pr)[0].cpu().numpy()
            self._key = key
        return self._route

    def choose(self, belief, x, y, heading, targets):
        route = self._route_probs(belief, x, y, heading)
        w = belief.width

        def penalty(px, py, h):
            """Extra seconds charged for stepping into a cell off the believed route."""
            d = DIR_DELTA[h]
            nx, ny = px + d["x"], py + d["y"]
            if not belief.in_bounds(nx, ny) or belief.visited[ny, nx]:
                return 0.0
            return self.alpha * T_CELL * (1.0 - float(route[ny, nx]))

        # edge_bonus subtracts, so pass the negated penalty.
        p = belief_passable(belief, optimistic=True)
        field = time_to_goal(p, targets, w, belief.height,
                             edge_bonus=lambda px, py, h: -penalty(px, py, h))

        best = None
        for h in range(NUM_HEADINGS):
            if not belief.is_passable(x, y, h, optimistic=True):
                continue
            d = DIR_DELTA[h]
            nx, ny = x + d["x"], y + d["y"]
            if not belief.in_bounds(nx, ny):
                continue
            c = float(field[h, ny * w + nx])
            if c >= UNREACHABLE:
                continue
            s = turn_cost(heading, h) + T_CELL + penalty(x, y, h) + c
            if best is None or s < best[0] - 1e-12:
                best = (s, h)
        return None if best is None else best[1]


def sweep_weighted(n_mazes: int = 60, budget: float = 18.0):
    import random
    from dataset_generator import generate_maze
    from episode import EpisodeConfig, run_episode, FloodFillPolicy

    print(f"budget {budget}s, {n_mazes} mazes\n")
    print(f"{'policy':<34}{'mean':>8}{'p90':>8}{'known':>8}{'no-route':>10}")

    def bench(factory, label):
        g, k, fail = [], [], 0
        for i in range(n_mazes):
            random.seed(i)
            m = generate_maze()
            r = run_episode(m.clone(), factory(), EpisodeConfig(search_time_budget=budget))
            k.append(r.known_fraction)
            if r.fast_run_time >= 1e5:
                fail += 1
                g.append(5.0)
            else:
                g.append(r.exploration_gap)
        g = np.array(g)
        print(f"{label:<34}{g.mean():>8.3f}{np.percentile(g, 90):>8.3f}"
              f"{np.mean(k):>8.2f}{fail:>10}")

    bench(FloodFillPolicy, "flood_fill (baseline)")
    for a in (0.5, 1.0, 2.0, 3.0, 5.0):
        bench(lambda aa=a: RouteWeightedPolicy(alpha=aa), f"route_weighted alpha={a}")
