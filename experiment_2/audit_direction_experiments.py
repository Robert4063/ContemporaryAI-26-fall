"""Check every direction experiment against independent shortest paths."""
import csv
import hashlib
import json
from pathlib import Path

from audit_reuse_experiments import dijkstra
from run_direction_experiments import solve,normalized_stats
from compare_direction_results import comparisons

ROOT=Path(__file__).resolve().parent


def verify():
    folder=ROOT/'results/direction';s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    check(len(s['metrics'])==106 and s['setup']['input_count']==53,'53 inputs x 2 methods')
    for name,digest in s['setup']['sources_sha256'].items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'source '+name)
    data={}
    for name,digest in s['setup']['inputs_sha256'].items():
        check(hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,'input '+name)
        item=json.loads((folder/name).read_text(encoding='utf-8'));data[item['name']]=item
        truth=dijkstra(item['grid'],tuple(item['start']))[tuple(item['goal'])]
        check(truth==item['optimal_cost'],'independent optimal cost '+item['name'])
    for row in s['metrics']:
        item=data[row['case']];key=row['case']+'_'+row['method'];detail=s['details'][key]
        fresh=solve(row['method'],item);stats=normalized_stats(fresh,row['method'])
        check(detail['path']==[list(p) for p in fresh.path] and all(row[k]==v for k,v in stats.items()),'fresh trace '+key)
        path=[tuple(p) for p in detail['path']];grid=item['grid']
        check(path[0]==tuple(item['start']) and path[-1]==tuple(item['goal']) and
              all(0<=r<len(grid) and 0<=c<len(grid[0]) and grid[r][c] is not None for r,c in path) and
              all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:])),'legal full cell path '+key)
        cost=sum(grid[r][c] for r,c in path[1:])
        check(cost==row['cost']==fresh.cost and row['path_steps']==len(path)-1 and
              row['optimal']==(cost==item['optimal_cost']),'cost '+key)
        check(row['optimal'],'optimal restored search '+key)
        if row['method']=='inertia_astar':
            trace=detail['trace']
            probes=[e for e in trace if e['event']=='probe']
            accepts=[e for e in trace if e['event']=='accept']
            rollbacks=[e for e in trace if e['event']=='rollback']
            check(len(probes)==row['inertia_probes']==len(accepts)+len(rollbacks) and
                len(accepts)==row['inertia_accepts'] and len(rollbacks)==row['backtracks'],
                'every trial accepted or reversed '+key)
            check(all(e['f']>e['previous_f'] for e in rollbacks) and
                all(e['f']<=next(p['previous_f'] for p in probes if p['p']==e['p'] and p['q']==e['q'] and p['f']==e['f']) for e in accepts),
                'f threshold '+key)
            check(sum(e['extra_cost'] for e in rollbacks)==row['rollback_cost'] and
                sum(grid[e['q'][0]][e['q'][1]] for e in probes)+sum(grid[e['p'][0]][e['p'][1]] for e in rollbacks)==row['probe_walk_cost'],
                'failed trial movement charges '+key)
            for i,e in enumerate(trace):
                if e['event']=='probe':
                    check(i>0 and trace[i-1]['event']=='save' and trace[i-1]['p']==e['p'] and
                        trace[i+1]['event'] in ('accept','rollback'),'save then trial then evaluate '+key)
                    goal=tuple(item['goal']);q=tuple(e['q']);p=tuple(e['p'])
                    minimum=min(v for r,line in enumerate(grid) for c,v in enumerate(line) if v is not None and (r,c)!=goal)
                    hn=0 if q==goal else max(abs(q[0]-goal[0])+abs(q[1]-goal[1])-1,0)*minimum+grid[goal[0]][goal[1]]
                    check(abs(q[0]-p[0])+abs(q[1]-p[1])==1 and
                        e['g']==trace[i-1]['g']+grid[q[0]][q[1]] and e['h']==hn and e['f']==e['g']+hn,
                        'independent entering cost and heuristic '+key)
        times=detail['timing_samples_us']
        check(len(times)==21 and sorted(times)[10]==row['median_us'] and min(times)==row['min_us'] and
              max(times)==row['max_us'],'21 entire-search timings '+key)
    def value(case,method,key):return next(r[key] for r in s['metrics'] if r['case']==case and r['method']==method)
    check(value('ring_shortcut','inertia_astar','cost')==6 and value('weighted_ring','inertia_astar','cost')==12,'A* chooses first direction')
    check(value('forced_detour','inertia_astar','backtracks')==2 and value('course_maze','inertia_astar','backtracks')==1,'f increase triggers return')
    check(value('vertical_corridor','astar','queue_pops')==401 and
          value('vertical_corridor','inertia_astar','queue_pops')==2 and
          value('vertical_corridor','inertia_astar','heuristic_calls')==401,'each new cell still computes h')
    tests=json.loads((ROOT/'results/direction_correctness.json').read_text(encoding='utf-8'))
    check(tests['successful'] and tests['test_methods']==6 and tests['random_grids']==120,'six new direction tests')
    with (folder/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    check(len(rows)==106 and all(all(str(v)==row[k] for k,v in item.items()) for row,item in zip(rows,s['metrics'])),'CSV consistency')
    with (folder/'comparison.csv').open(encoding='utf-8-sig',newline='') as f:paired=list(csv.DictReader(f))
    check(len(paired)==53 and all(all(str(v)==row[k] for k,v in item.items()) for row,item in zip(paired,comparisons(s))),
        '53 per-input paired comparisons')
    result=dict(passed=True,checks=checks,inputs=53,configurations=106,
        summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest())
    (folder/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(verify()))
