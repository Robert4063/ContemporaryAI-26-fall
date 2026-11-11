"""Counterexamples and controlled tests of direction persistence."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from audit_reuse_experiments import dijkstra
from direction_search import DirectionStats,inertia_astar
from fixtures import MAZE,MAZE_START,MAZE_GOAL,WEIGHTED,WEIGHTED_START,WEIGHTED_GOAL
from run_experiments import benchmark,connected_random_grid,save_csv,save_json
from search_algorithms import astar,maze_grid,weighted_grid

ROOT=Path(__file__).resolve().parent
METHODS=('astar','inertia_astar')


def cases():
    ring=[[1 if r in (0,6) or c in (0,6) else None for c in range(7)] for r in range(7)]
    weighted=[row.copy() for row in ring]
    for r in range(1,6):weighted[r][0]=8
    dead=[[1]*7]+[[1]+[None]*6 for _ in range(5)]+[[None]*7]
    snake=[[1 if r%2==0 or c==(40 if ((r-1)//2)%2==0 else 0) else None for c in range(41)] for r in range(41)]
    named=[('ring_shortcut',ring,(0,0),(0,6)),('weighted_ring',weighted,(0,0),(6,6)),
        ('dead_end',dead,(0,0),(0,6)),('vertical_corridor',[[1] for _ in range(401)],(0,0),(400,0)),
        ('horizontal_corridor',[[1]*401],(0,0),(0,400)),('snake_corridor',snake,(0,0),(40,40)),
        ('open_grid',[[1]*41 for _ in range(41)],(0,0),(40,40))]
    # A costly forward probe, a compulsory move away from goal, a comb,
    # and a staircase distinguish rollback from wall handling and turns.
    costly=[[1,1,1,1,1],[1,None,None,None,1],[8,None,None,None,1],[1,1,1,1,1],[1,1,1,1,1]]
    detour=[[1,None,1,1,1],[1,None,1,None,1],[1,None,1,None,1],[1,1,1,None,1],[None]*5]
    comb=[[1]*21 if r==0 else [1 if c%2==0 else None for c in range(21)] for r in range(9)]
    stairs=[[None]*21 for _ in range(21)]
    for i in range(21):
        stairs[i][i]=1
        if i<20:stairs[i][i+1]=1
    named.extend([('costly_probe',costly,(0,0),(4,0)),('forced_detour',detour,(0,0),(0,4)),
        ('comb',comb,(0,0),(8,20)),('staircase',stairs,(0,0),(20,20)),
        ('course_maze',maze_grid(MAZE),MAZE_START,MAZE_GOAL),
        ('course_weighted',weighted_grid(WEIGHTED),WEIGHTED_START,WEIGHTED_GOAL)])
    for name,grid,start,goal in named:yield dict(name=name,category='designed',grid=grid,start=start,goal=goal)
    for weighted in (False,True):
        for index in range(20):
            seed=20261201+index;name=f"{'weighted' if weighted else 'unit'}_{seed}"
            yield dict(name=name,category='weighted40' if weighted else 'unit40',seed=seed,
                grid=connected_random_grid(40,seed,weighted),start=(0,0),goal=(39,39))


def solve(method,data):
    grid,start,goal=data['grid'],tuple(data['start']),tuple(data['goal'])
    return astar(grid,start,goal) if method=='astar' else inertia_astar(grid,start,goal)


def normalized_stats(result,method):
    stats=asdict(DirectionStats());stats.update(asdict(result.stats))
    if method=='astar':
        stats.update(heuristic_calls=result.stats.generated+1,geometry_checks=result.stats.expanded*4,
            queue_pushes=result.stats.generated,queue_pops=result.stats.visited+result.stats.stale_pops)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=21)
    p.add_argument('--output',type=Path,default=ROOT/'results/direction')
    args=p.parse_args();folder=args.output.resolve()
    if args.repeats<3 or not folder.is_relative_to(ROOT):p.error('invalid repeats/output')
    rows=[];details={};names=[]
    for data in cases():
        name=data['name'];truth=dijkstra(data['grid'],tuple(data['start']))[tuple(data['goal'])]
        save_json(folder/'inputs'/f'{name}.json',dict(**data,optimal_cost=truth));names.append(name)
        for method in METHODS:
            result=solve(method,data)
            timing=benchmark(lambda:solve(method,data),args.repeats,batch=10)
            stats=normalized_stats(result,method)
            rows.append(dict(case=name,category=data['category'],method=method,optimal_cost=truth,cost=result.cost,
                path_steps=len(result.path)-1,optimal=result.cost==truth,**stats,
                **{k:v for k,v in timing.items() if k!='timing_samples_us'}))
            traced=(astar(data['grid'],tuple(data['start']),tuple(data['goal']),record_trace=True)
                if method=='astar' else inertia_astar(data['grid'],tuple(data['start']),tuple(data['goal']),record_trace=True))
            details[name+'_'+method]=dict(path=result.path,stats=stats,trace=traced.trace,
                timing_samples_us=timing['timing_samples_us'])
        print(json.dumps(dict(case=name,rows=len(rows))),flush=True)
    sources=['direction_search.py','run_direction_experiments.py','tests/test_direction_search.py',
        'search_algorithms.py','run_experiments.py','audit_reuse_experiments.py']
    summary=dict(setup=dict(repeats=args.repeats,warmups=3,batch=10,methods=METHODS,input_count=len(names),
        named_cases=names[:13],random_seeds=list(range(20261201,20261221)),
        timing='original course A* vs inertia implementation; entire search including validation, h construction, saved states and rollback; trace off; no preprocessing excluded',
        acceptance='probe q=p+incoming before evaluating f(q); accept f(q)<=f(p), otherwise roll back and resume A*; preserve deferred branches in a saved stack whose top competes with OPEN by (f,h)',
        movement='rollback_cost counts failed probe q and return p entering costs, separate from final path cost; search OPEN changes are not a continuous physical walk',
        sources_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
        inputs_sha256={f'inputs/{name}.json':hashlib.sha256((folder/'inputs'/f'{name}.json').read_bytes()).hexdigest() for name in names}),
        metrics=rows,details=details)
    save_json(folder/'summary.json',summary);save_csv(folder/'metrics.csv',rows)
    print(json.dumps(dict(inputs=len(names),configurations=len(rows))),flush=True)


if __name__=='__main__':main()
