"""A*, Minimax and Alpha-beta. Algorithm implementation uses only the stdlib."""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import inf, isfinite
from typing import Callable

Coord = tuple[int, int]
Grid = list[list[int | float | None]]
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left


@dataclass
class SearchStats:
    visited: int = 0
    expanded: int = 0
    generated: int = 0
    evaluated_leaves: int = 0
    cutoffs: int = 0
    skipped_children: int = 0
    cache_hits: int = 0
    stale_pops: int = 0
    peak_frontier: int = 0


@dataclass
class PathResult:
    cost: float
    path: list[Coord]
    stats: SearchStats
    trace: list[dict] = field(default_factory=list)


def validate_grid(grid: Grid, start: Coord, goal: Coord) -> None:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be non-empty and rectangular")
    for row in grid:
        for value in row:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not isfinite(value) or value < 0):
                raise ValueError("cell costs must be finite non-negative numbers or None")
    for r, c in (start, goal):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            raise ValueError("start/goal is outside the grid")


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_heuristic(grid: Grid, goal: Coord, mode: str) -> Callable[[Coord], float]:
    """Construct a heuristic, including a raw Manhattan comparison mode.

    For n != goal, D steps require at least D-1 non-goal entries plus the
    goal entry. Thus max(D-1,0)*m + cost(goal) is admissible and consistent.
    m is the minimum cost of a traversable non-goal cell. If all cells cost
    one this reduces to ordinary Manhattan distance.
    """
    if mode == "zero":
        return lambda _: 0
    if mode == "manhattan":
        return lambda n: manhattan(n, goal)
    if mode != "safe_manhattan":
        raise ValueError("heuristic must be zero, manhattan or safe_manhattan")
    ordinary = [v for r, row in enumerate(grid) for c, v in enumerate(row)
                if v is not None and (r, c) != goal]
    minimum = min(ordinary, default=0)
    goal_cost = grid[goal[0]][goal[1]]
    if goal_cost is None:
        return lambda _: 0
    return lambda n: 0 if n == goal else max(manhattan(n, goal) - 1, 0) * minimum + goal_cost


def astar(grid: Grid, start: Coord, goal: Coord, heuristic: str = "safe_manhattan",
          tie_break: str = "lower_h", record_trace: bool = False) -> PathResult:
    """Search by g+h. Entering a cell incurs its weight.

    The start is free. Walls are None. Return cost=inf and [] if unreachable.
    Heap records are lazy; improvements can reopen nodes, stale records are
    discarded. Stop when a valid goal record is popped, not when generated.
    zero/safe_manhattan supply admissible heuristics. Raw manhattan is an
    explicit experimental mode; its admissibility depends on cell costs.
    """
    validate_grid(grid, start, goal)
    if tie_break not in ("lower_h", "fifo"):
        raise ValueError("tie_break must be lower_h or fifo")
    stats = SearchStats()
    trace = []
    if grid[start[0]][start[1]] is None or grid[goal[0]][goal[1]] is None:
        return PathResult(inf, [], stats, trace)
    h = make_heuristic(grid, goal, heuristic)
    serial = count()
    best = {start: 0}
    parent: dict[Coord, Coord] = {}
    queue = [(h(start), h(start) if tie_break == "lower_h" else 0, next(serial), 0, start)]
    stats.generated = stats.peak_frontier = 1
    while queue:
        f, _, _, g, current = heappop(queue)
        if g != best.get(current, inf):
            stats.stale_pops += 1
            continue
        stats.visited += 1
        if record_trace:
            trace.append(dict(step=stats.visited, row=current[0], col=current[1],
                              g=g, h=h(current), f=f, frontier_after_pop=len(queue)))
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return PathResult(g, path[::-1], stats, trace)
        stats.expanded += 1
        for dr, dc in DIRECTIONS:
            nxt = current[0] + dr, current[1] + dc
            r, c = nxt
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] is None:
                continue
            new_g = g + grid[r][c]
            if new_g < best.get(nxt, inf):
                best[nxt] = new_g
                parent[nxt] = current
                hn = h(nxt)
                heappush(queue, (new_g + hn, hn if tie_break == "lower_h" else 0,
                                 next(serial), new_g, nxt))
                stats.generated += 1
        stats.peak_frontier = max(stats.peak_frontier, len(queue))
    return PathResult(inf, [], stats, trace)


def maze_grid(raw: list[list[str]]) -> Grid:
    allowed = {"S", "E", ".", "X"}
    if any(cell not in allowed for row in raw for cell in row):
        raise ValueError("maze contains an unknown symbol")
    return [[None if cell == "X" else 1 for cell in row] for row in raw]


def weighted_grid(raw: list[list]) -> Grid:
    def convert(cell):
        if cell is None or cell == "X":
            return None
        if cell == "S":
            return 1
        if cell == "E":
            return 0
        if isinstance(cell, str):
            return int(cell)
        return cell
    return [[convert(cell) for cell in row] for row in raw]


def solve_maze(start: Coord, goal: Coord, raw: list[list[str]],
               heuristic: str = "safe_manhattan") -> list[Coord]:
    return astar(maze_grid(raw), start, goal, heuristic).path


def solve_weighted(start: Coord, goal: Coord, raw: list[list],
                   heuristic: str = "safe_manhattan") -> int | float:
    result = astar(weighted_grid(raw), start, goal, heuristic)
    return -1 if result.cost == inf else result.cost


@dataclass
class TreeNode:
    value: int | float | None = None
    children: list["TreeNode"] = field(default_factory=list)


def tree_from_dict(data: dict, maximizing: bool = True) -> TreeNode:
    """Parse the JSON sample, checking its optional player labels."""
    if "tree_structure" in data:
        data = data["tree_structure"]
    if "is_maximizing_player" in data and data["is_maximizing_player"] != maximizing:
        raise ValueError("players must alternate; root is MAX")
    children = data.get("children", [])
    if children:
        return TreeNode(children=[tree_from_dict(c, not maximizing) for c in children])
    if data.get("value") is None:
        raise ValueError("a leaf requires a value")
    return TreeNode(value=data["value"])


def search_tree(root: TreeNode, algorithm: str = "alphabeta", order: str = "left",
                record_trace: bool = False) -> tuple[float, SearchStats, list[dict]]:
    if algorithm not in ("minimax", "alphabeta") or order not in ("left", "reverse"):
        raise ValueError("unknown algorithm/order")
    stats, trace = SearchStats(), []

    def visit(node, maximizing, alpha, beta, path):
        stats.visited += 1
        if not node.children:
            if node.value is None:
                raise ValueError("a leaf requires a value")
            stats.evaluated_leaves += 1
            if record_trace:
                trace.append(dict(event="leaf", node=path, value=node.value))
            return node.value
        stats.expanded += 1
        children = list(enumerate(node.children))
        if order == "reverse":
            children.reverse()
        value = -inf if maximizing else inf
        for index, (original_index, child) in enumerate(children):
            candidate = visit(child, not maximizing, alpha, beta, path + str(original_index))
            value = max(value, candidate) if maximizing else min(value, candidate)
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            remaining = len(children) - index - 1
            if algorithm == "alphabeta" and alpha >= beta and remaining:
                stats.cutoffs += 1
                stats.skipped_children += remaining
                if record_trace:
                    trace.append(dict(event="cutoff", node=path, alpha=alpha, beta=beta,
                                      skipped_children=remaining))
                break
        if record_trace:
            trace.append(dict(event="return", node=path, value=value))
        return value

    return visit(root, True, -inf, inf, "R"), stats, trace


def alpha_beta(root: TreeNode) -> float:
    return search_tree(root)[0]


@dataclass(frozen=True)
class GameState:
    position_a: int = 0
    position_b: int = 1
    remaining: tuple[int, ...] = (2, 3, 4, 5)
    score_a: float = 0
    score_b: float = 0
    player: int = 0  # 0=MAX/A, 1=MIN/B

    @property
    def utility(self):
        return self.score_a - self.score_b


@dataclass
class GameResult:
    best_target: int | None
    best_value: float
    principal_variation: list[int]
    stats: SearchStats


def transition(state: GameState, target: int, distances: list[list], values: dict) -> GameState:
    if target not in state.remaining:
        raise ValueError("target has already been collected")
    remaining = tuple(t for t in state.remaining if t != target)
    position = state.position_a if state.player == 0 else state.position_b
    distance = distances[position][target]
    if not isfinite(distance):
        raise ValueError("unreachable treasure is not a legal action")
    gain = values[target] - distance
    if state.player == 0:
        return GameState(target, state.position_b, remaining, state.score_a + gain, state.score_b, 1)
    return GameState(state.position_a, target, remaining, state.score_a, state.score_b + gain, 0)


def search_game(distances: list[list], values: dict, horizon: int,
                algorithm: str = "alphabeta", order: str = "natural",
                state: GameState | None = None, memoize: bool = False) -> GameResult:
    """Depth-limited zero-sum search, returning absolute Score_A - Score_B.

    Memoization is deliberately restricted to full Minimax: cut-off Alpha-beta
    values can be bounds. Cache future utility (value-current U), so identical
    positions/remnants can share an exact entry despite different past scores.
    No reachable actions: evaluate current utility (extension; sample connected).
    """
    if horizon < 0 or not isinstance(horizon, int):
        raise ValueError("horizon must be a non-negative integer")
    if algorithm not in ("minimax", "alphabeta") or order not in ("natural", "reverse", "greedy"):
        raise ValueError("unknown algorithm/order")
    if memoize and algorithm != "minimax":
        raise ValueError("only exact Minimax values may be cached")
    state = state or GameState(remaining=tuple(sorted(values)))
    stats, cache = SearchStats(), {}

    def visit(s, depth, alpha, beta):
        stats.visited += 1
        if depth == 0 or not s.remaining:
            stats.evaluated_leaves += 1
            return s.utility, []
        key = (s.position_a, s.position_b, s.remaining, s.player, depth)
        if memoize and key in cache:
            stats.cache_hits += 1
            future_value, pv = cache[key]
            return s.utility + future_value, list(pv)
        position = s.position_a if s.player == 0 else s.position_b
        actions = [t for t in s.remaining if isfinite(distances[position][t])]
        if order == "reverse":
            actions.reverse()
        elif order == "greedy":
            # Both players first try their largest immediate personal gain.
            actions.sort(key=lambda t: (-(values[t] - distances[position][t]), t))
        if not actions:
            stats.evaluated_leaves += 1
            return s.utility, []
        stats.expanded += 1
        maximizing = s.player == 0
        value, pv = (-inf if maximizing else inf), []
        for index, target in enumerate(actions):
            child = transition(s, target, distances, values)
            candidate, child_pv = visit(child, depth - 1, alpha, beta)
            if (candidate > value if maximizing else candidate < value):
                value, pv = candidate, [target] + child_pv
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            remaining = len(actions) - index - 1
            if algorithm == "alphabeta" and alpha >= beta and remaining:
                stats.cutoffs += 1
                stats.skipped_children += remaining
                break
        if memoize:
            cache[key] = value - s.utility, tuple(pv)
        return value, pv

    value, pv = visit(state, horizon, -inf, inf)
    return GameResult(pv[0] if pv else None, value, pv, stats)


def pairwise_distances(grid: Grid, positions: list[Coord]) -> list[list[float]]:
    # Compute directed entries independently: destination-weight graphs need
    # not have symmetric distances. Symmetry happens to hold for PDF endpoints.
    return [[astar(grid, a, b).cost for b in positions] for a in positions]
