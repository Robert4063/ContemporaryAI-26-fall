"""Independent tests for move-then-evaluate inertia and A* restoration."""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from direction_search import inertia_astar
from search_algorithms import astar
from test_search import bellman_ford


class DirectionTests(unittest.TestCase):
    def check_path(self,grid,start,goal,result):
        self.assertEqual(result.cost,bellman_ford(grid,start,goal))
        if result.path:
            self.assertEqual(result.path[0],start);self.assertEqual(result.path[-1],goal)
            self.assertEqual(sum(grid[r][c] for r,c in result.path[1:]),result.cost)
            self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(result.path,result.path[1:])))
            self.assertTrue(all(grid[r][c] is not None for r,c in result.path))

    def test_random_weighted_and_zero_cost(self):
        rng=random.Random(20261112)
        for _ in range(120):
            grid=[[None if rng.random()<.2 else rng.choice([0,1,1,3,8]) for _ in range(5)] for _ in range(5)]
            grid[0][0]=1;grid[-1][-1]=rng.choice([0,1,3])
            for enabled in (False,True):
                result=inertia_astar(grid,(0,0),(4,4),enable_inertia=enabled)
                self.check_path(grid,(0,0),(4,4),result)
                if not enabled:
                    original=astar(grid,(0,0),(4,4))
                    self.assertEqual((result.path,result.stats.expanded,result.stats.generated),
                        (original.path,original.stats.expanded,original.stats.generated))

    def test_first_step_selected_before_inertia_and_same_h_calls(self):
        grid=[[1] for _ in range(8)]
        result=inertia_astar(grid,(0,0),(7,0),True)
        trace=result.trace
        first=next(i for i,e in enumerate(trace) if e['event']=='probe')
        self.assertEqual(trace[first-2]['p'],(1,0))
        self.assertEqual(trace[first]['q'],(2,0))
        self.assertEqual(trace[first+1]['event'],'accept')
        self.assertEqual(result.stats.inertia_accepts,6)
        baseline=inertia_astar(grid,(0,0),(7,0),enable_inertia=False)
        self.assertEqual(result.stats.heuristic_calls,baseline.stats.heuristic_calls)
        self.assertEqual((baseline.stats.queue_pops,result.stats.queue_pops),(8,2))

    def test_failed_probe_returns_and_does_not_discard_required_cell(self):
        grid=[[1],[1],[8],[1],[1]]
        result=inertia_astar(grid,(0,0),(4,0),True)
        self.check_path(grid,(0,0),(4,0),result)
        event=next(e for e in result.trace if e['event']=='rollback')
        self.assertEqual((event['p'],event['q'],event['previous_f'],event['f']),((1,0),(2,0),4,11))
        self.assertEqual((result.stats.backtracks,result.stats.rollback_cost),(1,9))
        self.assertIn((2,0),result.path)

    def test_saved_side_branch_restored_after_wrong_inertia(self):
        grid=[[1,1,1,1,1],[None,1,None,None,None],[None,1,None,None,None],
              [None,1,None,None,None],[None,1,1,1,1]]
        result=inertia_astar(grid,(0,0),(4,4),True)
        self.check_path(grid,(0,0),(4,4),result)
        self.assertGreater(result.stats.resumed_states,0)
        self.assertIn((1,1),result.path)

    def test_goal_inside_run_and_zero_fee(self):
        for grid,goal in [([[1,1,1,1,1]],(0,2)),([[1,0,0,0]],(0,3)),([[1,3,8,0,1]],(0,3))]:
            self.check_path(grid,(0,0),goal,inertia_astar(grid,(0,0),goal))

    def test_start_goal_blocked_unreachable_and_bad_grid(self):
        for enabled in (False,True):
            self.assertEqual(inertia_astar([[1]],(0,0),(0,0),enable_inertia=enabled).path,[(0,0)])
            self.assertEqual(inertia_astar([[1,None,1]],(0,0),(0,2),enable_inertia=enabled).path,[])
            self.assertEqual(inertia_astar([[None]],(0,0),(0,0),enable_inertia=enabled).path,[])
        with self.assertRaises(ValueError):inertia_astar([[1,-1]],(0,0),(0,1))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DirectionTests))
    output=Path(__file__).resolve().parents[1]/'results/direction_correctness.json'
    output.write_text(json.dumps(dict(test_methods=result.testsRun,successful=result.wasSuccessful(),
        failures=len(result.failures),errors=len(result.errors),random_grids=120),indent=2),encoding='utf-8')
    sys.exit(not result.wasSuccessful())
