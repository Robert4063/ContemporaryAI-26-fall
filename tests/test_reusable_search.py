"""Independent checks for persistent values, bounds, hints and node budgets."""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import EXPECTED_DISTANCES,VALUES
from search_algorithms import GameState,transition
from reusable_search import GameSearchSession
from test_search import independent_game


def truth(d,values,s,h):
    return independent_game(d,values,s.position_a,s.position_b,s.remaining,s.player,h,s.utility)


class ReuseTests(unittest.TestCase):
    def check_action(self,d,values,s,h,r):
        self.assertEqual(r.best_value,truth(d,values,s,h))
        if h and s.remaining:
            child=transition(s,r.best_target,d,values)
            self.assertEqual(truth(d,values,child,h-1),r.best_value)

    def test_known_depths_and_score_offsets(self):
        for algorithm in ('minimax','alphabeta'):
            session=GameSearchSession(EXPECTED_DISTANCES,VALUES,algorithm=algorithm)
            for h in (2,4,1,3,4,2):
                for offset in (0,17,-11):
                    s=GameState(score_a=offset)
                    self.check_action(EXPECTED_DISTANCES,VALUES,s,h,session.search(h,s))

    def test_cross_turn_exact_reuse(self):
        session=GameSearchSession(EXPECTED_DISTANCES,VALUES,algorithm='minimax',use_hints=False)
        result=session.search(4)
        child=transition(GameState(),result.best_target,EXPECTED_DISTANCES,VALUES)
        reused=session.search(3,child)
        self.check_action(EXPECTED_DISTANCES,VALUES,child,3,reused)
        self.assertEqual(reused.stats.visited,1)
        self.assertEqual(reused.stats.cross_call_hits,1)

    def test_depth_is_part_of_value_key(self):
        session=GameSearchSession(EXPECTED_DISTANCES,VALUES,algorithm='minimax',use_hints=False)
        self.assertEqual(session.search(2).best_value,-1)
        self.assertEqual(session.search(4).best_value,3)

    def test_random_persistent_bounds_and_hints(self):
        rng=random.Random(20261111)
        for case in range(80):
            d=[[0 if i==j else rng.randint(1,12) for j in range(7)] for i in range(7)]
            values={i:rng.randint(1,15) for i in range(2,7)}
            for algorithm in ('minimax','alphabeta'):
                for hints in (False,True):
                    session=GameSearchSession(d,values,algorithm=algorithm,use_hints=hints)
                    s=GameState(remaining=tuple(values),score_a=3,score_b=-2)
                    while s.remaining:
                        for h in (min(2,len(s.remaining)),len(s.remaining)):
                            self.check_action(d,values,s,h,session.search(h,s))
                        # Real play can deviate from the previously predicted PV.
                        s=transition(s,rng.choice(s.remaining),d,values)

    def test_budget_and_interrupted_search_recovery(self):
        for budget in (1,5,20,60,200):
            session=GameSearchSession(EXPECTED_DISTANCES,VALUES)
            limited=session.iterative(4,budget)
            self.assertLessEqual(limited.result.stats.visited,budget)
            if limited.completed_depth:
                self.check_action(EXPECTED_DISTANCES,VALUES,GameState(),limited.completed_depth,limited.result)
            self.check_action(EXPECTED_DISTANCES,VALUES,GameState(),4,session.search(4))

    def test_environment_snapshot(self):
        d=[row.copy() for row in EXPECTED_DISTANCES];v=VALUES.copy()
        session=GameSearchSession(d,v)
        d[0][2]=1000;v[2]=1000
        self.assertEqual(session.search(4).best_value,3)

    def test_paired_depth_budget(self):
        self.assertEqual(GameSearchSession(EXPECTED_DISTANCES,VALUES).iterative(0,100,depth_step=2).completed_depth,0)
        for budget in (1,5,20,60,200):
            session=GameSearchSession(EXPECTED_DISTANCES,VALUES)
            limited=session.iterative(4,budget,depth_step=2)
            self.assertIn(limited.completed_depth,(0,1,2,4))
            self.assertLessEqual(limited.result.stats.visited,budget)
            if limited.completed_depth:
                self.check_action(EXPECTED_DISTANCES,VALUES,GameState(),limited.completed_depth,limited.result)

    def test_hint_only_and_external_deviation(self):
        session=GameSearchSession(EXPECTED_DISTANCES,VALUES,cache_values=False,use_hints=True)
        s=GameState()
        for h,target in [(2,2),(3,5),(2,3),(1,4)]:
            self.check_action(EXPECTED_DISTANCES,VALUES,s,h,session.search(h,s))
            s=transition(s,target,EXPECTED_DISTANCES,VALUES)
        self.assertFalse(session.table)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReuseTests))
    output=Path(__file__).resolve().parents[1]/'results/reuse_correctness.json'
    output.write_text(json.dumps(dict(test_methods=result.testsRun,successful=result.wasSuccessful(),
        failures=len(result.failures),errors=len(result.errors),random_games=80),indent=2),encoding='utf-8')
    sys.exit(not result.wasSuccessful())
