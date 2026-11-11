"""Verify all experiment data without PDF, font or rendering dependencies."""
from __future__ import annotations
import csv
import hashlib
import json
from heapq import heappop,heappush
from math import inf
from pathlib import Path
import sys
from dataclasses import asdict
from math import isfinite

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_search import independent_game,independent_tree
from fixtures import *
from search_algorithms import *


def reference_dijkstra(grid,start):
    """Independent implementation: no production heuristic, validator or stats."""
    distances={start:0}
    queue=[(0,start)]
    while queue:
        cost,u=heappop(queue)
        if cost!=distances[u]:continue
        for r,c in ((u[0]+1,u[1]),(u[0]-1,u[1]),(u[0],u[1]+1),(u[0],u[1]-1)):
            if 0<=r<len(grid) and 0<=c<len(grid[0]) and grid[r][c] is not None:
                v=(r,c);candidate=cost+grid[r][c]
                if candidate<distances.get(v,inf):
                    distances[v]=candidate;heappush(queue,(candidate,v))
    return distances


def main():
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    tests=json.loads((ROOT/'results/correctness.json').read_text(encoding='utf-8'))
    checks=[]
    def check(name,condition,detail):
        checks.append(dict(requirement=name,passed=bool(condition),detail=detail))
        if not condition:raise AssertionError(name+': '+str(detail))
    check('自动化测试',tests['successful'] and tests['failures']==tests['errors']==0,tests)
    check('实验配置数量',len(s['core'])==21 and len(s['synthetic'])==80 and len(s['large_game'])==5,'21+80+5=106')
    from audit_reuse_experiments import verify as verify_reuse
    from audit_direction_experiments import verify as verify_direction
    reuse_audit=verify_reuse();direction_audit=verify_direction()
    check('跨回合复用实验全部核对',reuse_audit['passed'],reuse_audit)
    check('惯性跳棋策略实验全部核对',direction_audit['passed'],direction_audit)
    configuration_count=106+reuse_audit['exact_configurations']+reuse_audit['budget_configurations']+direction_audit['configurations']
    check('实验总配置392组',configuration_count==392,'106 baseline + 180 reuse + 106 inertia')
    extra_tests=[json.loads((ROOT/f'results/{name}_correctness.json').read_text(encoding='utf-8'))
                 for name in ['reuse','direction']]
    check('三十五项测试全部通过',tests['test_methods']+sum(r['test_methods'] for r in extra_tests)==35 and
          all(r['successful'] and r['failures']==r['errors']==0 for r in extra_tests),'21 baseline + 8 reuse + 6 direction')
    for key,raw,start,goal in [('q1',maze_grid(MAZE),MAZE_START,MAZE_GOAL),
                              ('q2',weighted_grid(WEIGHTED),WEIGHTED_START,WEIGHTED_GOAL)]:
        truth=reference_dijkstra(raw,start)[goal]
        modes=['zero','safe_manhattan'] if key=='q1' else ['zero','manhattan','safe_manhattan']
        for h in modes:
            record=s['details'][key+'_'+h];path=[tuple(v) for v in record['path']]
            legal=path[0]==start and path[-1]==goal and all(
                abs(a[0]-b[0])+abs(a[1]-b[1])==1 and raw[b[0]][b[1]] is not None
                for a,b in zip(path,path[1:]))
            cost=sum(raw[r][c] for r,c in path[1:])
            check(key+' '+h+'路径合法且最优',legal and cost==record['cost']==truth==7,record['path'])
            with (ROOT/'results/traces'/f'{key}_{h}.csv').open(encoding='utf-8-sig',newline='') as f:
                trace=list(csv.DictReader(f))
            check(key+' '+h+'轨迹计数',len(trace)==record['stats']['visited'] and
                  all(float(r['g'])+float(r['h'])==float(r['f']) for r in trace),'valid pops and f=g+h')
    independent_matrix=[]
    for source in POSITIONS:
        distances=reference_dijkstra(TREASURE_GRID,source)
        independent_matrix.append([distances[target] for target in POSITIONS])
    check('6×6矩阵独立核对',s['distance_matrix']==independent_matrix==EXPECTED_DISTANCES,independent_matrix)
    alternative=[(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4),(4,4)]
    weighted=weighted_grid(WEIGHTED)
    alternative_cost=sum(weighted[r][c] for r,c in alternative[1:])
    check('实验二等长路线代价对照',len(alternative)-1==8 and alternative_cost==9,
          dict(path=alternative,moves=8,cost=alternative_cost,optimal_cost=7))
    q2_rows=[row for row in s['core'] if row['problem']=='q2']
    check('第二题三种启发对照齐全',len(q2_rows)==3 and
          {row['setting'] for row in q2_rows}=={'zero','manhattan','safe_manhattan'},q2_rows)
    with (ROOT/'results/traces/q2_manhattan.csv').open(encoding='utf-8-sig',newline='') as f:
        raw_trace=list(csv.DictReader(f))
    with (ROOT/'results/traces/q2_safe_manhattan.csv').open(encoding='utf-8-sig',newline='') as f:
        safe_trace=list(csv.DictReader(f))
    check('第二题原始与修正启发逐次对照',len(raw_trace)==len(safe_trace)==11 and all(
        all(a[k]==b[k] for k in ['row','col','g']) and
        float(a['h'])-float(b['h'])==(0 if (int(a['row']),int(a['col']))==WEIGHTED_GOAL else 1) and
        float(a['f'])-float(b['f'])==(0 if (int(a['row']),int(a['col']))==WEIGHTED_GOAL else 1)
        for a,b in zip(raw_trace,safe_trace)),'11 pops, same coordinates/g; non-goal h/f shift by 1')
    for row in s['core']:
        expected=7 if row['problem'] in ('q1','q2') else 5 if row['problem']=='q3' else -1 if row['horizon']==2 else 3
        check('课程配置 '+str((row['problem'],row['algorithm'],row['setting'],row['horizon'])),
              row['value']==expected and row['min_us']<=row['median_us']<=row['max_us'],row)
        if row['problem'] in ('q1','q2'):
            grid,start,goal=(maze_grid(MAZE),MAZE_START,MAZE_GOAL) if row['problem']=='q1' else (
                weighted_grid(WEIGHTED),WEIGHTED_START,WEIGHTED_GOAL)
            fresh=astar(grid,start,goal,row['setting'])
            stats=fresh.stats
        elif row['problem']=='q3':
            _,stats,_=search_tree(example_tree(),row['algorithm'],row['setting'])
        else:
            fresh=search_game(independent_matrix,VALUES,row['horizon'],row['algorithm'],row['setting'])
            stats=fresh.stats
            check('宝藏首步 '+str((row['algorithm'],row['setting'],row['horizon'])),
                  NAMES[fresh.best_target]==row['best_target'],row['best_target'])
        check('课程节点统计 '+str((row['problem'],row['algorithm'],row['setting'],row['horizon'])),
              all(row[k]==v for k,v in asdict(stats).items()),asdict(stats))
    for row in s['root_decisions']:
        t=NAMES.index(row['first_target']);remaining=tuple(i for i in VALUES if i!=t)
        truth=independent_game(independent_matrix,VALUES,t,1,remaining,1,row['horizon']-1,
                               VALUES[t]-independent_matrix[0][t])
        check('首步独立穷举 '+str((row['horizon'],row['first_target'])),row['value']==truth,truth)
    # Every one of the 80 synthetic rows is checked, not just group means.
    for kind in ['maze40','weighted40']:
        for index in range(20):
            grid=json.loads((ROOT/'results/inputs'/f'{kind}_{index:02d}.json').read_text(encoding='utf-8'))
            truth=reference_dijkstra(grid,(0,0))[(39,39)]
            for h in ['zero','safe_manhattan']:
                row=next(r for r in s['synthetic'] if r['dataset']==kind and r['index']==index and r['heuristic']==h)
                result=astar(grid,(0,0),(39,39),h)
                check(f'{kind} #{index:02d} {h}',row['cost']==result.cost==truth and
                      all(row[k]==v for k,v in asdict(result.stats).items()),
                      dict(reference_cost=truth,expanded=row['expanded']))
    data=json.loads((ROOT/'results/large_game_input.json').read_text(encoding='utf-8'))
    vals={int(k):v for k,v in data['values'].items()}
    truth=independent_game(data['distances'],vals,0,1,tuple(vals),0,8)
    check('八宝藏五配置独立穷举',all(row['best_value']==truth for row in s['large_game']),truth)
    for row in s['large_game']:
        fresh=search_game(data['distances'],vals,8,row['algorithm'],row['order'],memoize=row['memoize'])
        t=row['best_target']
        target_truth=independent_game(data['distances'],vals,t,1,tuple(i for i in vals if i!=t),1,7,
                                      vals[t]-data['distances'][0][t])
        check('八宝藏首步与统计 '+str((row['algorithm'],row['order'],row['memoize'])),
              target_truth==truth and fresh.best_target==t and
              all(row[k]==v for k,v in asdict(fresh.stats).items()),dict(target=t,value=target_truth))
    distances8=[]
    for source in data['positions']:
        distances=reference_dijkstra(data['grid'],tuple(source))
        distances8.append([distances[tuple(t)] for t in data['positions']])
    check('八宝藏距离矩阵独立核对',distances8==data['distances'],'100 directed entries')
    check('106组计时记录完整且有序',all(
        all(isfinite(row[k]) and row[k]>=0 for k in ['min_us','median_us','max_us']) and
        row['min_us']<=row['median_us']<=row['max_us']
        for row in s['core']+s['synthetic']+s['large_game']),'min <= median <= max')
    # CSV and JSON must represent the same exact rows and timing records.
    for filename,key in [('core_metrics.csv','core'),('synthetic_metrics.csv','synthetic'),
                          ('large_game_metrics.csv','large_game'),('root_decisions.csv','root_decisions')]:
        with (ROOT/'results'/filename).open(encoding='utf-8-sig',newline='') as f:csv_rows=list(csv.DictReader(f))
        check(filename+'与JSON一致',len(csv_rows)==len(s[key]) and all(
            all(str(v)==row[k] for k,v in record.items()) for row,record in zip(csv_rows,s[key])),len(csv_rows))
    manifest=json.loads((ROOT/'results/run_manifest.json').read_text(encoding='utf-8'))
    check('运行时输入哈希',all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
          for name,digest in manifest['inputs_sha256'].items()),list(manifest['inputs_sha256']))
    result=dict(passed=True,checks=len(checks),experiment_configurations=configuration_count,
                test_methods=35,reuse_checks=reuse_audit['checks'],direction_checks=direction_audit['checks'],
                summary_sha256=hashlib.sha256((ROOT/'results/summary.json').read_bytes()).hexdigest(),
                requirements=checks)
    (ROOT/'results/experiment_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='requirements'},ensure_ascii=False))


if __name__=='__main__':main()
