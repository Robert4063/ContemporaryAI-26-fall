"""Correctness oracles deliberately do not use the production search logic."""
import math
import random
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_algorithms import (GameState, TreeNode, alpha_beta, astar, make_heuristic,
    maze_grid, pairwise_distances, search_game, search_tree, solve_maze,
    solve_weighted, transition, tree_from_dict, weighted_grid)
from fixtures import *

RANDOM_GRIDS, RANDOM_TREES, RANDOM_GAMES = 120, 80, 60


def bellman_ford(grid, start, goal):
    if grid[start[0]][start[1]] is None or grid[goal[0]][goal[1]] is None:
        return math.inf
    vertices = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v is not None]
    distances = {v: math.inf for v in vertices}
    distances[start] = 0
    edges = [(u, v, grid[v[0]][v[1]]) for u in vertices for v in vertices
             if abs(u[0]-v[0]) + abs(u[1]-v[1]) == 1]
    for _ in range(max(len(vertices)-1, 0)):
        changed = False
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                changed = True
        if not changed:
            break
    return distances[goal]


def independent_game(d, values, a, b, remaining, player, depth, utility=0):
    if not remaining or depth == 0:
        return utility
    scores = []
    for target in remaining:
        rest = tuple(t for t in remaining if t != target)
        if player == 0:
            scores.append(independent_game(d, values, target, b, rest, 1, depth-1,
                                          utility + values[target] - d[a][target]))
        else:
            scores.append(independent_game(d, values, a, target, rest, 0, depth-1,
                                          utility - values[target] + d[b][target]))
    return (max if player == 0 else min)(scores)


def independent_tree(node, player=0):
    if not node.children:
        return node.value
    scores = [independent_tree(child, 1-player) for child in node.children]
    return (max if player == 0 else min)(scores)


class SearchTests(unittest.TestCase):
    def test_pdf_maze(self):
        path = solve_maze(MAZE_START, MAZE_GOAL, MAZE)
        self.assertEqual(len(path)-1, 7)
        self.assertEqual(path[0], MAZE_START)
        self.assertEqual(path[-1], MAZE_GOAL)

    def test_pdf_weighted(self):
        self.assertEqual(solve_weighted(WEIGHTED_START, WEIGHTED_GOAL, WEIGHTED), 7)

    def test_raw_manhattan_comparison(self):
        grid=weighted_grid(WEIGHTED)
        raw=make_heuristic(grid,WEIGHTED_GOAL,'manhattan')
        safe=make_heuristic(grid,WEIGHTED_GOAL,'safe_manhattan')
        self.assertEqual(raw(WEIGHTED_GOAL),0)
        self.assertEqual(raw((4,3)),1)
        self.assertEqual(safe((4,3)),0)
        self.assertEqual(bellman_ford(grid,(4,3),WEIGHTED_GOAL),0)
        expected=bellman_ford(grid,WEIGHTED_START,WEIGHTED_GOAL)
        for mode in ('zero','manhattan','safe_manhattan'):
            result=astar(grid,WEIGHTED_START,WEIGHTED_GOAL,mode)
            self.assertEqual(result.cost,expected)
            self.assertEqual(sum(grid[r][c] for r,c in result.path[1:]),expected)
        self.assertEqual(solve_weighted(WEIGHTED_START,WEIGHTED_GOAL,WEIGHTED,'manhattan'),7)

    def test_no_path(self):
        self.assertEqual(solve_maze((0,0), (0,2), [list('SXE')]), [])
        self.assertEqual(solve_weighted((0,0), (0,2), [['S','X','E']]), -1)

    def test_start_equals_goal(self):
        result = astar([[5]], (0,0), (0,0))
        self.assertEqual((result.cost, result.path), (0, [(0,0)]))

    def test_blocked_endpoint(self):
        self.assertEqual(astar([[None,1]], (0,0), (0,1)).path, [])
        self.assertEqual(astar([[1,None]], (0,0), (0,1)).cost, math.inf)

    def test_free_goal_adjacent(self):
        grid = weighted_grid([['S','E']])
        self.assertEqual(astar(grid, (0,0), (0,1)).cost, 0)
        self.assertEqual(make_heuristic(grid, (0,1), 'safe_manhattan')((0,0)), 0)

    def test_cheaper_detour(self):
        result = astar([[1,9,1],[1,1,1]], (0,0), (0,2))
        self.assertEqual(result.cost, 4)
        self.assertEqual(len(result.path)-1, 4)

    def test_zero_cost_cycles(self):
        result = astar([[0,0,0],[0,None,1]], (0,0), (1,2))
        self.assertEqual(result.cost, 1)
        self.assertLessEqual(result.stats.visited, 5)

    def test_invalid_grids(self):
        for grid in ([], [[]], [[1],[1,2]], [[1,-1]], [[1,float('nan')]], [[True]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                astar(grid, (0,0), (0,0))
        with self.assertRaises(ValueError):
            astar([[1]], (0,0), (1,0))

    def test_random_grids_against_bellman_ford(self):
        rng = random.Random(20261105)
        for case in range(RANDOM_GRIDS):
            rows, cols = rng.randint(2,6), rng.randint(2,6)
            grid = [[None if rng.random()<.25 else rng.choice([0,1,1,3,8])
                     for _ in range(cols)] for _ in range(rows)]
            grid[0][0] = 1
            grid[-1][-1] = rng.choice([0,1,3])
            start, goal = (0,0), (rows-1,cols-1)
            expected = bellman_ford(grid, start, goal)
            for heuristic in ('zero','safe_manhattan'):
                for tie in ('fifo','lower_h'):
                    result = astar(grid,start,goal,heuristic,tie)
                    with self.subTest(case=case, heuristic=heuristic, tie=tie):
                        self.assertEqual(result.cost, expected)
                        if result.path:
                            self.assertEqual(result.path[0], start)
                            self.assertEqual(result.path[-1], goal)
                            self.assertEqual(sum(grid[r][c] for r,c in result.path[1:]), expected)
                            self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1
                                                for a,b in zip(result.path,result.path[1:])))

    def test_heuristic_admissibility_and_consistency(self):
        rng = random.Random(50116)
        for _ in range(20):
            grid = [[rng.choice([1,1,3,8]) for _ in range(4)] for _ in range(4)]
            grid[-1][-1] = rng.choice([0,1,3])
            goal=(3,3)
            h=make_heuristic(grid,goal,'safe_manhattan')
            for r in range(4):
                for c in range(4):
                    u=(r,c)
                    self.assertLessEqual(h(u), bellman_ford(grid,u,goal))
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        v=(r+dr,c+dc)
                        if 0<=v[0]<4 and 0<=v[1]<4:
                            self.assertLessEqual(h(u),grid[v[0]][v[1]]+h(v))

    def test_pdf_tree(self):
        for algorithm in ('minimax','alphabeta'):
            for order in ('left','reverse'):
                self.assertEqual(search_tree(example_tree(),algorithm,order)[0],5)
        self.assertEqual(alpha_beta(example_tree()),5)
        self.assertEqual(search_tree(example_tree(),'minimax')[1].visited,15)
        self.assertEqual(search_tree(example_tree())[1].visited,14)

    def test_tree_leaf_negative_and_json(self):
        self.assertEqual(alpha_beta(TreeNode(-9)), -9)
        root=tree_from_dict({'tree_structure':{'is_maximizing_player':True,
                                              'children':[{'value':-9},{'value':-2}]}})
        self.assertEqual(alpha_beta(root), -2)
        with self.assertRaises(ValueError):
            tree_from_dict({'is_maximizing_player':False,'value':1})
        with self.assertRaises(ValueError):
            alpha_beta(TreeNode())

    def test_random_trees(self):
        rng=random.Random(20261106)
        def build(depth):
            if depth==0 or rng.random()<.15:
                return TreeNode(rng.randint(-20,20))
            return TreeNode(children=[build(depth-1) for _ in range(rng.randint(1,4))])
        for _ in range(RANDOM_TREES):
            root=build(4)
            expected=independent_tree(root)
            for algorithm in ('minimax','alphabeta'):
                for order in ('left','reverse'):
                    self.assertEqual(search_tree(root,algorithm,order)[0],expected)

    def test_distance_matrix(self):
        self.assertEqual(pairwise_distances(TREASURE_GRID,POSITIONS),EXPECTED_DISTANCES)
        # Weighted destination costs are directed in the general case.
        self.assertEqual(pairwise_distances([[1,3]],[(0,0),(0,1)]), [[0,3],[1,0]])

    def test_pdf_game(self):
        for depth,target,value in ((2,3,-1),(4,4,3)):
            for algorithm in ('minimax','alphabeta'):
                for order in ('natural','reverse','greedy'):
                    result=search_game(EXPECTED_DISTANCES,VALUES,depth,algorithm,order)
                    self.assertEqual((result.best_target,result.best_value),(target,value))

    def test_game_transition_and_negative_gain(self):
        s=transition(GameState(),4,EXPECTED_DISTANCES,VALUES)
        self.assertEqual((s.position_a,s.position_b,s.score_a,s.score_b,s.player),(4,1,-1,0,1))
        self.assertNotIn(4,s.remaining)
        self.assertEqual(GameState().remaining,(2,3,4,5))
        with self.assertRaises(ValueError):
            transition(s,4,EXPECTED_DISTANCES,VALUES)

    def test_game_terminal_and_ties(self):
        self.assertEqual(search_game(EXPECTED_DISTANCES,VALUES,0).best_value,0)
        s=GameState(remaining=(),score_a=3,score_b=8)
        self.assertEqual(search_game(EXPECTED_DISTANCES,VALUES,4,state=s).best_value,-5)
        d=[[0]*4 for _ in range(4)]
        self.assertEqual(search_game(d,{2:1,3:1},2).best_target,2)

    def test_unreachable_game_extension(self):
        d=[[0,1,math.inf],[1,0,math.inf],[math.inf,math.inf,0]]
        self.assertIsNone(search_game(d,{2:9},2).best_target)

    def test_random_games_and_exact_cache(self):
        rng=random.Random(20261107)
        for _ in range(RANDOM_GAMES):
            d=[[0 if i==j else rng.randint(1,12) for j in range(6)] for i in range(6)]
            values={i:rng.randint(1,10) for i in range(2,6)}
            depth=rng.randint(1,4)
            s=GameState(score_a=5,score_b=2)
            expected=independent_game(d,values,0,1,(2,3,4,5),0,depth,3)
            for algorithm in ('minimax','alphabeta'):
                for order in ('natural','reverse','greedy'):
                    self.assertEqual(search_game(d,values,depth,algorithm,order,s).best_value,expected)
            self.assertEqual(search_game(d,values,depth,'minimax',state=s,memoize=True).best_value,expected)
        with self.assertRaises(ValueError):
            search_game(EXPECTED_DISTANCES,VALUES,4,memoize=True)


if __name__ == '__main__':
    import json
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SearchTests))
    output=Path(__file__).resolve().parents[1]/'results'/'correctness.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(test_methods=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),successful=result.wasSuccessful(),random_grids=RANDOM_GRIDS,
        random_trees=RANDOM_TREES,random_games=RANDOM_GAMES,
        grid_configurations=RANDOM_GRIDS*4,tree_configurations=RANDOM_TREES*4,
        game_configurations=RANDOM_GAMES*7),indent=2),encoding='utf-8')
    sys.exit(not result.wasSuccessful())
