"""Run deterministic experiments and save raw counts, timings, paths and traces."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from fixtures import *
from search_algorithms import *

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
SEED=20261105


def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_csv(path,rows):
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def benchmark(fn,repeats=21,batch=10):
    for _ in range(3):
        fn()
    durations=[]
    for _ in range(repeats):
        begin=perf_counter_ns()
        for _ in range(batch):
            fn()
        durations.append((perf_counter_ns()-begin)/batch/1000)
    return dict(median_us=statistics.median(durations),min_us=min(durations),
                max_us=max(durations),timing_samples_us=durations)


def generated_grid(size,seed,weighted=False):
    rng=random.Random(seed)
    grid=[[None if rng.random()<.22 else (rng.choices([1,3,8],[.75,.15,.10])[0] if weighted else 1)
           for _ in range(size)] for _ in range(size)]
    # A controlled guaranteed route. The remaining interior is still random.
    for i in range(size):
        grid[i][0]=1
        grid[-1][i]=1
    if weighted:
        grid[-1][-1]=0
    return grid


def connected_random_grid(size, seed, weighted=False):
    """Rejection-sample reachable random maps; do not carve a cheap corridor."""
    rng=random.Random(seed)
    while True:
        grid=[[None if rng.random()<.27 else (rng.choices([1,3,8],[.75,.15,.10])[0] if weighted else 1)
               for _ in range(size)] for _ in range(size)]
        grid[0][0]=1
        grid[-1][-1]=0 if weighted else 1
        seen={(0,0)}
        queue=deque([(0,0)])
        while queue:
            r,c=queue.popleft()
            for dr,dc in DIRECTIONS:
                v=(r+dr,c+dc)
                if 0<=v[0]<size and 0<=v[1]<size and grid[v[0]][v[1]] is not None and v not in seen:
                    seen.add(v)
                    queue.append(v)
        if (size-1,size-1) in seen:
            return grid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=21)
    args=parser.parse_args()
    if args.repeats<3:
        parser.error('use at least 3 timing repetitions')
    RESULTS.mkdir(exist_ok=True)
    records=[]
    details={}
    for label,grid,start,goal in [
        ('q1',maze_grid(MAZE),MAZE_START,MAZE_GOAL),
        ('q2',weighted_grid(WEIGHTED),WEIGHTED_START,WEIGHTED_GOAL)]:
        modes=('zero','safe_manhattan') if label=='q1' else ('zero','manhattan','safe_manhattan')
        for heuristic in modes:
            fn=lambda:astar(grid,start,goal,heuristic)
            result=astar(grid,start,goal,heuristic,record_trace=True)
            timing=benchmark(fn,args.repeats)
            row=dict(problem=label,algorithm='astar',setting=heuristic,horizon='',value=result.cost,
                     best_target='',**asdict(result.stats),**{k:v for k,v in timing.items() if k!='timing_samples_us'})
            records.append(row)
            details[label+'_'+heuristic]=dict(path=result.path,cost=result.cost,stats=asdict(result.stats),**timing)
            save_csv(RESULTS/'traces'/f'{label}_{heuristic}.csv',result.trace)
    for algorithm in ('minimax','alphabeta'):
        for order in ('left','reverse'):
            fn=lambda:search_tree(example_tree(),algorithm,order)
            value,stats,trace=search_tree(example_tree(),algorithm,order,True)
            timing=benchmark(fn,args.repeats)
            records.append(dict(problem='q3',algorithm=algorithm,setting=order,horizon='',value=value,
                best_target='',**asdict(stats),**{k:v for k,v in timing.items() if k!='timing_samples_us'}))
            details[f'q3_{algorithm}_{order}']=dict(value=value,stats=asdict(stats),**timing)
            save_json(RESULTS/'traces'/f'q3_{algorithm}_{order}.json',trace)
    distances=pairwise_distances(TREASURE_GRID,POSITIONS)
    assert distances==EXPECTED_DISTANCES
    save_csv(RESULTS/'distance_matrix.csv',[dict(source=NAMES[i],**dict(zip(NAMES,row))) for i,row in enumerate(distances)])
    decisions=[]
    for horizon in (2,4):
        for algorithm in ('minimax','alphabeta'):
            for order in ('natural','reverse','greedy'):
                fn=lambda:search_game(distances,VALUES,horizon,algorithm,order)
                result=fn()
                timing=benchmark(fn,args.repeats)
                expected=(3,-1) if horizon==2 else (4,3)
                assert (result.best_target,result.best_value)==expected
                records.append(dict(problem='q4',algorithm=algorithm,setting=order,horizon=horizon,
                    value=result.best_value,best_target=NAMES[result.best_target],**asdict(result.stats),
                    **{k:v for k,v in timing.items() if k!='timing_samples_us'}))
                details[f'q4_{algorithm}_{order}_h{horizon}']=dict(
                    best_target=NAMES[result.best_target],best_value=result.best_value,
                    principal_variation=[NAMES[t] for t in result.principal_variation],
                    stats=asdict(result.stats),**timing)
        for target in VALUES:
            state=transition(GameState(),target,distances,VALUES)
            result=search_game(distances,VALUES,horizon-1,'minimax',state=state)
            decisions.append(dict(horizon=horizon,first_target=NAMES[target],value=result.best_value,
                                  principal_variation=' -> '.join([NAMES[target]]+[NAMES[t] for t in result.principal_variation])))
    # Evaluate order sensitivity on the exact same larger game tree.
    rng=random.Random(SEED)
    positions8=[(0,0),(9,9),(0,9),(2,2),(2,7),(4,4),(5,8),(7,1),(8,6),(9,0)]
    grid8=generated_grid(10,SEED+99)
    for r,c in positions8:
        grid8[r][c]=1
    distances8=pairwise_distances(grid8,positions8)
    values8={i:rng.randint(5,15) for i in range(2,10)}
    large=[]
    for algorithm,order,memo in [('minimax','natural',False),('minimax','natural',True),
        ('alphabeta','natural',False),('alphabeta','reverse',False),('alphabeta','greedy',False)]:
        fn=lambda:search_game(distances8,values8,8,algorithm,order,memoize=memo)
        result=fn()
        timing=benchmark(fn,args.repeats,batch=1)
        large.append(dict(algorithm=algorithm,order=order,memoize=memo,
            best_target=result.best_target,best_value=result.best_value,**asdict(result.stats),
            **{k:v for k,v in timing.items() if k!='timing_samples_us'}))
    assert len({r['best_value'] for r in large})==1
    synthetic=[]
    for weighted in (False,True):
        for index in range(20):
            grid=connected_random_grid(40,SEED+index,weighted)
            label='weighted40' if weighted else 'maze40'
            save_json(RESULTS/'inputs'/f'{label}_{index:02d}.json',grid)
            for heuristic in ('zero','safe_manhattan'):
                fn=lambda:astar(grid,(0,0),(39,39),heuristic)
                result=fn()
                timing=benchmark(fn,args.repeats,batch=1)
                synthetic.append(dict(dataset=label,index=index,seed=SEED+index,heuristic=heuristic,
                    cost=result.cost,path_steps=len(result.path)-1,**asdict(result.stats),
                    **{k:v for k,v in timing.items() if k!='timing_samples_us'}))
    for index in range(0,len(synthetic),2):
        assert synthetic[index]['cost']==synthetic[index+1]['cost']
    save_csv(RESULTS/'core_metrics.csv',records)
    save_csv(RESULTS/'root_decisions.csv',decisions)
    save_csv(RESULTS/'large_game_metrics.csv',large)
    save_csv(RESULTS/'synthetic_metrics.csv',synthetic)
    save_json(RESULTS/'large_game_input.json',dict(grid=grid8,positions=positions8,values=values8,distances=distances8))
    setup=dict(seed=SEED,repeats=args.repeats,warmups=3,core_batch=10,large_batch=1,
        python=sys.version,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),
        directions=DIRECTIONS,tie_break='lower_h then insertion order',grid_size=40,
        q2_heuristics=['zero','manhattan','safe_manhattan'],
        synthetic_maps_per_type=20,obstacle_probability=.27,weights=[1,3,8],weight_probabilities=[.75,.15,.10],
        synthetic_sampling='resample until endpoints connected by BFS; no corridor carved; weighted goal cost 0',
        large_game_sampling='10x10; obstacle probability .22; left column and bottom row carved; endpoints set to 1',
        measurement='median microseconds per call; three warmups; validation and stats included; no file I/O',
        grid_stat_definitions={'visited':'valid pops including goal','expanded':'valid non-goal pops whose neighbors are scanned',
                              'generated':'heap pushes including start; not unique states','peak_frontier':'heap length including stale records'},
        game_stat_definitions={'visited':'recursive calls including terminals and cache hits',
                              'expanded':'non-terminal calls with children actually processed',
                              'cutoffs':'events that skip at least one unvisited immediate child',
                              'skipped_children':'skipped immediate children, not entire skipped subtree size'})
    save_json(RESULTS/'summary.json',dict(setup=setup,core=records,details=details,distance_matrix=distances,
        root_decisions=decisions,large_game=large,synthetic=synthetic))
    experiment_sources=['search_algorithms.py','fixtures.py','run_experiments.py','solve.py','tests/test_search.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in experiment_sources}
    hashes['docs/实验二要求_new.pdf']=hashlib.sha256((ROOT/'docs'/'实验二要求_new.pdf').read_bytes()).hexdigest()
    save_json(RESULTS/'run_manifest.json',dict(inputs_sha256=hashes,setup=setup))
    print(json.dumps(dict(core_configurations=len(records),synthetic_configurations=len(synthetic),
        large_game_configurations=len(large),all_expected_values_matched=True),ensure_ascii=False))


if __name__=='__main__':
    main()
