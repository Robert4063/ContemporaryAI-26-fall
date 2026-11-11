"""Audit every new input, query, reused decision and budget record."""
from __future__ import annotations
import csv
import hashlib
import json
from heapq import heappop,heappush
from pathlib import Path
import sys

from run_reuse_experiments import exact_episode,budget_episode,game_state
from search_algorithms import transition

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_search import independent_game


def dijkstra(grid,start):
    d={start:0};queue=[(0,start)]
    while queue:
        cost,(r,c)=heappop(queue)
        if cost!=d[(r,c)]:continue
        for u in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
            rr,cc=u
            if 0<=rr<len(grid) and 0<=cc<len(grid[0]) and grid[rr][cc] is not None:
                v=cost+grid[rr][cc]
                if v<d.get(u,float('inf')):d[u]=v;heappush(queue,(v,u))
    return d


def verify():
    folder=ROOT/'results/reuse';s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    check_count=0
    def check(condition,description):
        nonlocal check_count
        check_count+=1
        if not condition:raise AssertionError(description)
    check(len(s['exact'])==72 and len(s['budget'])==108,'72 exact + 108 budget configurations')
    for filename,digest in s['setup']['sources_sha256'].items():
        check(hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,'source '+filename)
    for filename,digest in s['setup']['inputs_sha256'].items():
        check(hashlib.sha256((folder/filename).read_bytes()).hexdigest()==digest,'input '+filename)
    tests=json.loads((ROOT/'results/reuse_correctness.json').read_text(encoding='utf-8'))
    check(tests['successful'] and tests['test_methods']==8 and tests['failures']==tests['errors']==0,'eight new tests')
    data_by_seed={};truth_cache={}
    def oracle(data,state,depth):
        key=(data['seed'],state,depth)
        if key not in truth_cache:
            truth_cache[key]=independent_game(data['distances'],data['values'],state.position_a,state.position_b,
                state.remaining,state.player,depth,state.utility)
        return truth_cache[key]
    for seed in s['setup']['seeds']:
        data=json.loads((folder/'inputs'/f'{seed}.json').read_text(encoding='utf-8'))
        data['values']={int(k):v for k,v in data['values'].items()}
        for q in data['queries']:q['action_values']={int(k):v for k,v in q['action_values'].items()}
        matrix=[]
        for position in data['positions']:
            distance=dijkstra(data['grid'],tuple(position));matrix.append([distance[tuple(p)] for p in data['positions']])
        check(matrix==data['distances'],f'{seed} 100 independent distances')
        previous=None
        for q in data['queries']:
            state=game_state(q['state'])
            if previous is not None:check(state==previous,'shared chronological input')
            expected={t:oracle(data,transition(state,t,matrix,data['values']),len(state.remaining)-1) for t in state.remaining}
            optimal=(max if state.player==0 else min)(expected.values())
            check(expected==q['action_values'] and optimal==q['full_value'],'full-action quality oracle')
            check(q['played']==next(t for t,v in expected.items() if v==optimal),'canonical observed move')
            previous=transition(state,q['played'],matrix,data['values'])
        data_by_seed[seed]=data
    for row in s['exact']:
        data=data_by_seed[row['seed']];queries=data['queries']
        fresh=exact_episode(data,queries,row['method'],row['profile'],True)
        key=f"{row['seed']}_{row['profile']}_{row['method']}";detail=s['details'][key]
        check(fresh['trace']==detail['trace'] and all(row[k]==v for k,v in fresh['stats'].items()),'exact counts '+key)
        for q,t in zip(queries,detail['trace']):
            state=game_state(q['state']);expected=oracle(data,state,t['depth'])
            action_value=oracle(data,transition(state,t['target'],data['distances'],data['values']),t['depth']-1)
            check(t['value']==expected==action_value,'exact decision '+key)
        times=detail['timing_samples_us']
        check(len(times)==21 and min(times)==row['min_us'] and max(times)==row['max_us'] and
              sorted(times)[len(times)//2]==row['median_us'],'21 measured episode times')
    for row in s['budget']:
        data=data_by_seed[row['seed']];queries=data['queries']
        fresh=budget_episode(data,queries,row['method'],row['budget'])
        key=f"{row['seed']}_{row['budget']}_{row['method']}"
        check(fresh['trace']==s['details'][key] and all(row[k]==v for k,v in fresh.items() if k!='trace'),'budget counts '+key)
        for q,t in zip(queries,fresh['trace']):
            state=game_state(q['state']);depth=t['completed_depth']
            expected=oracle(data,state,depth)
            check(t['estimate']==expected and t['stats']['visited']<=row['budget'],'budget/depth '+key)
            check(oracle(data,transition(state,t['target'],data['distances'],data['values']),depth-1)==expected,'completed-depth action')
            full=q['action_values'][t['target']]
            regret=q['full_value']-full if state.player==0 else full-q['full_value']
            check(t['regret']==regret>=0 and t['optimal']==(regret==0),'full-depth regret '+key)
    for filename,key in [('exact_metrics.csv','exact'),('budget_metrics.csv','budget')]:
        with (folder/filename).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        check(len(rows)==len(s[key]) and all(all(str(v)==row[k] for k,v in record.items())
              for row,record in zip(rows,s[key])),'CSV '+filename)
    result=dict(passed=True,checks=check_count,inputs=6,exact_configurations=72,budget_configurations=108,
        queries_per_episode=8,summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest())
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(verify()))
