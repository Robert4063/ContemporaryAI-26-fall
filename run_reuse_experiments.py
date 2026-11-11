"""Controlled cross-decision reuse and fixed-node-budget experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

from run_experiments import benchmark,connected_random_grid,save_csv,save_json
from reusable_search import GameSearchSession,ReuseStats
from search_algorithms import DIRECTIONS,GameState,pairwise_distances,transition

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_search import independent_game

SEEDS=tuple(range(20261111,20261117))
BUDGETS=(64,256,1024)
EXACT_METHODS=('fresh_mm','reuse_mm','fresh_ab','reuse_ab','fresh_ab_hints','reuse_ab_hints')
BUDGET_METHODS=('fresh_ab_hints','reuse_hints_only','reuse_ab','reuse_ab_hints','fresh_ab_pairs','reuse_ab_pairs')


def session_for(data,method):
    return GameSearchSession(data['distances'],{int(k):v for k,v in data['values'].items()},
        algorithm='minimax' if method.endswith('_mm') else 'alphabeta',order='greedy',
        use_hints=method.endswith(('_hints','_pairs')) or method=='reuse_hints_only')


def make_input(seed):
    grid=connected_random_grid(10,seed)
    reachable={(0,0)};queue=deque([(0,0)])
    while queue:
        r,c=queue.popleft()
        for dr,dc in DIRECTIONS:
            p=(r+dr,c+dc)
            if 0<=p[0]<10 and 0<=p[1]<10 and grid[p[0]][p[1]] is not None and p not in reachable:
                reachable.add(p);queue.append(p)
    rng=random.Random(seed+1000)
    positions=[(0,0),(9,9)]+rng.sample(sorted(reachable-{(0,0),(9,9)}),8)
    values={i:rng.randint(5,15) for i in range(2,10)}
    distances=pairwise_distances(grid,positions)
    return dict(seed=seed,grid=grid,positions=positions,values=values,distances=distances)


def reference_queries(data):
    d=data['distances'];v={int(k):x for k,x in data['values'].items()}
    s=GameState(remaining=tuple(v));queries=[]
    for turn in range(1,9):
        action_values={}
        for t in s.remaining:
            child=transition(s,t,d,v)
            action_values[t]=independent_game(d,v,child.position_a,child.position_b,child.remaining,
                child.player,len(child.remaining),child.utility)
        best=(max if s.player==0 else min)(action_values.values())
        played=next(t for t,value in action_values.items() if value==best)
        queries.append(dict(turn=turn,state=asdict(s),full_value=best,action_values=action_values,played=played))
        s=transition(s,played,d,v)
    return queries


def game_state(raw):
    return GameState(**{**raw,'remaining':tuple(raw['remaining'])})


def exact_episode(data,queries,method,profile,trace=False):
    session=session_for(data,method);total=ReuseStats();rows=[]
    for query in queries:
        state=game_state(query['state'])
        if method.startswith('fresh_'):session=session_for(data,method)
        depth=len(state.remaining) if profile=='full' else min(4,len(state.remaining))
        result=session.search(depth,state)
        for field,value in asdict(result.stats).items():setattr(total,field,getattr(total,field)+value)
        if trace:rows.append(dict(turn=query['turn'],depth=depth,target=result.best_target,value=result.best_value,
            stats=asdict(result.stats),table_entries=len(session.table)))
    return dict(stats=asdict(total),trace=rows)


def budget_episode(data,queries,method,budget):
    session=session_for(data,method);rows=[]
    for query in queries:
        state=game_state(query['state'])
        if method.startswith('fresh_'):session=session_for(data,method)
        elif method=='reuse_hints_only':session.table.clear()
        limited=session.iterative(len(state.remaining),budget,state,depth_step=2 if method.endswith('_pairs') else 1)
        result=limited.result
        candidate=query['action_values'][result.best_target]
        regret=(query['full_value']-candidate) if state.player==0 else (candidate-query['full_value'])
        rows.append(dict(turn=query['turn'],target=result.best_target,completed_depth=limited.completed_depth,
            estimate=result.best_value,full_action_value=candidate,full_optimum=query['full_value'],regret=regret,
            optimal=regret==0,exhausted=limited.exhausted,stats=asdict(result.stats),table_entries=len(session.table)))
    return dict(mean_depth=statistics.mean(r['completed_depth'] for r in rows),
        mean_regret=statistics.mean(r['regret'] for r in rows),optimal_decisions=sum(r['optimal'] for r in rows),
        visited=sum(r['stats']['visited'] for r in rows),cross_call_hits=sum(r['stats']['cross_call_hits'] for r in rows),
        cross_call_hints=sum(r['stats']['cross_call_hints'] for r in rows),trace=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=21)
    parser.add_argument('--cases',type=int,default=6)
    parser.add_argument('--output',type=Path,default=ROOT/'results/reuse')
    args=parser.parse_args()
    if not 1<=args.cases<=len(SEEDS) or args.repeats<3:parser.error('invalid cases/repeats')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):parser.error('output must stay in experiment_2')
    exact=[];budget_rows=[];all_details={};inputs=[]
    for seed in SEEDS[:args.cases]:
        data=make_input(seed);queries=reference_queries(data)
        save_json(output/'inputs'/f'{seed}.json',dict(**data,queries=queries));inputs.append(seed)
        for profile in ('full','rolling4'):
            for method in EXACT_METHODS:
                result=exact_episode(data,queries,method,profile,True)
                timing=benchmark(lambda:exact_episode(data,queries,method,profile),args.repeats,batch=1)
                key=f'{seed}_{profile}_{method}'
                exact.append(dict(seed=seed,profile=profile,method=method,**result['stats'],
                    **{k:v for k,v in timing.items() if k!='timing_samples_us'}))
                all_details[key]=dict(trace=result['trace'],timing_samples_us=timing['timing_samples_us'])
        for budget in BUDGETS:
            for method in BUDGET_METHODS:
                result=budget_episode(data,queries,method,budget)
                key=f'{seed}_{budget}_{method}'
                budget_rows.append(dict(seed=seed,budget=budget,method=method,
                    **{k:v for k,v in result.items() if k!='trace'}))
                all_details[key]=result['trace']
        print(json.dumps(dict(seed=seed,exact=len(exact),budget=len(budget_rows))),flush=True)
    sources=['reusable_search.py','run_reuse_experiments.py','tests/test_reusable_search.py',
        'search_algorithms.py','run_experiments.py']
    summary=dict(setup=dict(seeds=inputs,budgets=BUDGETS,repeats=args.repeats,warmups=3,queries_per_episode=8,
        measurement='each timed episode starts cold; reuse only between its eight queries; matrix generation excluded',
        trajectories='independent full-Minimax canonical moves; all methods receive identical observed states',
        budget_quality='exact full-horizon regret of selected action; all completed/aborted iterations charged',
        depth_schedules='step 1; or fallback depth 1 then paired depths 2,4,... and exact terminal depth',
        sampling='10x10, obstacle probability .27, connected component, 8 random treasure positions, values 5..15',
        inputs_sha256={f'inputs/{seed}.json':hashlib.sha256((output/'inputs'/f'{seed}.json').read_bytes()).hexdigest() for seed in inputs},
        sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}),
        exact=exact,budget=budget_rows,details=all_details)
    save_json(output/'summary.json',summary)
    save_csv(output/'exact_metrics.csv',exact);save_csv(output/'budget_metrics.csv',budget_rows)
    print(json.dumps(dict(output=str(output),exact=len(exact),budget=len(budget_rows))),flush=True)


if __name__=='__main__':main()
