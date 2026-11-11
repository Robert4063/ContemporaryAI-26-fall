"""Speculative inertial moves with saved A* branches and f-based rollback."""
from dataclasses import dataclass
from heapq import heappop,heappush
from itertools import count
from math import inf

from search_algorithms import DIRECTIONS,PathResult,SearchStats,validate_grid,make_heuristic

@dataclass
class DirectionStats(SearchStats):
    heuristic_calls: int = 0
    geometry_checks: int = 0
    backtracks: int = 0
    probe_walk_cost: float = 0
    inertia_probes: int = 0
    inertia_accepts: int = 0
    queue_pushes: int = 0
    queue_pops: int = 0
    resumed_states: int = 0
    rollback_cost: float = 0
    peak_saved_states: int = 0


def inertia_astar(grid,start,goal,record_trace=False,enable_inertia=True):
    """A* selects a step, then physically speculates one cell in that direction.

    At p, save its unexpanded alternatives and try q=p+incoming. Evaluate
    f(q) after that trial: accept if f(q)<=f(p), otherwise return to p and
    resume A*. A saved-state stack with lower bound f(p) preserves all
    deferred siblings; its top competes with OPEN. Accepted nodes bypass
    OPEN; failed trials are not
    discarded and their computed h is reused in the restored expansion.

    Search probes are simulated; rollback_cost charges entering q and p.
    It is extra movement, separate from the final start-to-goal path cost.
    """
    validate_grid(grid,start,goal);stats=DirectionStats();trace=[]
    if grid[start[0]][start[1]] is None or grid[goal[0]][goal[1]] is None:
        return PathResult(inf,[],stats,trace)
    heuristic=make_heuristic(grid,goal,'safe_manhattan')
    height,width=len(grid),len(grid[0]);serial=count()
    best={start:0};parents={};saved=[]
    def h(p):stats.heuristic_calls+=1;return heuristic(p)
    def legal(q):
        stats.geometry_checks+=1
        return 0<=q[0]<height and 0<=q[1]<width and grid[q[0]][q[1]] is not None
    def log(event,**values):
        if record_trace:trace.append(dict(event=event,**values))
    def push(entry):
        heappush(queue,entry);stats.queue_pushes+=1
        stats.peak_frontier=max(stats.peak_frontier,len(queue)+len(saved))
    hn=h(start)
    # f, h, serial, g, p, incoming, suspended, already_probed_neighbor
    queue=[(hn,hn,next(serial),0,start,None,False,None)]
    stats.generated=stats.peak_frontier=stats.queue_pushes=1
    active=None
    while queue or saved or active is not None:
        if active is None:
            if saved and (not queue or saved[-1][:2]<=queue[0][:2]):entry=saved.pop()
            else:entry=heappop(queue);stats.queue_pops+=1
        else:entry=active;active=None
        fp,hp,_,gp,p,incoming,suspended,skip=entry
        if gp!=best.get(p,inf):stats.stale_pops+=1;continue
        cached=None
        if suspended:
            stats.resumed_states+=1;log('resume',p=p,g=gp,f=fp,skip=skip)
        else:
            stats.visited+=1;log('select',p=p,g=gp,h=hp,f=fp,incoming=incoming)
            if p==goal:
                path=[p]
                while path[-1]!=start:path.append(parents[path[-1]])
                return PathResult(gp,path[::-1],stats,trace)
            stats.expanded+=1
            if incoming is not None and enable_inertia:
                q=p[0]+incoming[0],p[1]+incoming[1]
                if legal(q) and gp+grid[q[0]][q[1]]<best.get(q,inf):
                    log('save',p=p,g=gp,f=fp)
                    # Enter first, then calculate exactly once at the new cell.
                    stats.inertia_probes+=1;stats.probe_walk_cost+=grid[q[0]][q[1]]
                    gq=gp+grid[q[0]][q[1]];hq=h(q);fq=gq+hq
                    log('probe',p=p,q=q,g=gq,h=hq,f=fq,previous_f=fp)
                    if fq<=fp:
                        stats.inertia_accepts+=1;stats.generated+=1
                        best[q]=gq;parents[q]=p
                        # Consistency gives f(q)>=f(p); acceptance therefore
                        # keeps f equal and makes h non-increasing. The stack
                        # top is the smallest saved (f,h) lower bound.
                        saved.append((fp,hp,next(serial),gp,p,incoming,True,q))
                        stats.peak_saved_states=max(stats.peak_saved_states,len(saved))
                        stats.peak_frontier=max(stats.peak_frontier,len(queue)+len(saved))
                        log('accept',p=p,q=q,f=fq)
                        active=(fq,hq,next(serial),gq,q,incoming,False,None)
                        continue
                    stats.backtracks+=1
                    fee=grid[q[0]][q[1]]+grid[p[0]][p[1]]
                    stats.rollback_cost+=fee;stats.probe_walk_cost+=grid[p[0]][p[1]]
                    log('rollback',p=p,q=q,f=fq,previous_f=fp,extra_cost=fee)
                    cached=(q,gq,hq,fq)
        for dr,dc in DIRECTIONS:
            q=p[0]+dr,p[1]+dc
            if q==skip:continue
            if not legal(q):continue
            gq=gp+grid[q[0]][q[1]]
            if gq>=best.get(q,inf):continue
            hq=cached[2] if cached is not None and q==cached[0] else h(q)
            best[q]=gq;parents[q]=p;stats.generated+=1
            push((gq+hq,hq,next(serial),gq,q,(dr,dc),False,None))
    return PathResult(inf,[],stats,trace)

