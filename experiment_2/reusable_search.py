"""Reuse game-search work across decisions in one fixed environment.

Each session owns an immutable distance/value snapshot. Values are indexed by
the exact remaining horizon, while best-action hints may order another horizon.
The table stores future utility increments and EXACT/LOWER/UPPER bounds, so
different past scores and alpha-beta cutoffs can be reused correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf,isfinite
from types import MappingProxyType

from search_algorithms import GameState,GameResult,SearchStats,transition


@dataclass
class ReuseStats(SearchStats):
    exact_hits: int = 0
    bound_hits: int = 0
    cross_call_hits: int = 0
    hint_uses: int = 0
    cross_call_hints: int = 0


@dataclass(frozen=True)
class TableEntry:
    future_value: float
    bound: str
    pv: tuple[int,...]
    generation: int


@dataclass
class BudgetResult:
    result: GameResult
    completed_depth: int
    exhausted: bool


class NodeBudgetReached(Exception):
    pass


class GameSearchSession:
    """A persistent transposition table and move hints for one treasure game."""

    def __init__(self,distances,values,algorithm='alphabeta',order='greedy',
                 cache_values=True,use_hints=True):
        if algorithm not in ('minimax','alphabeta') or order not in ('natural','reverse','greedy'):
            raise ValueError('unknown algorithm/order')
        self.distances=tuple(tuple(row) for row in distances)
        self.values=MappingProxyType(dict(values))
        self.algorithm,self.order=algorithm,order
        self.cache_values,self.use_hints=cache_values,use_hints
        self.table={}
        self.hints={}
        self.generation=0

    @staticmethod
    def position_key(state):
        return state.position_a,state.position_b,state.remaining,state.player

    def clear(self):
        self.table.clear();self.hints.clear()

    def _run(self,state,horizon,stats,node_budget=None):
        def visit(s,depth,alpha,beta,is_root=False):
            if node_budget is not None and stats.visited>=node_budget:
                raise NodeBudgetReached
            stats.visited+=1
            if depth==0 or not s.remaining:
                stats.evaluated_leaves+=1
                return s.utility,[]
            base=self.position_key(s)
            key=base+(depth,)
            entry=self.table.get(key) if self.cache_values else None
            if entry is not None:
                stats.cache_hits+=1
                stats.cross_call_hits+=entry.generation<self.generation
                value=s.utility+entry.future_value
                if entry.bound=='EXACT':
                    stats.exact_hits+=1
                    return value,list(entry.pv)
                stats.bound_hits+=1
                # A root must return an optimal action, not only a valid bound.
                if not is_root and self.algorithm=='alphabeta':
                    if entry.bound=='LOWER':alpha=max(alpha,value)
                    else:beta=min(beta,value)
                    if alpha>=beta:
                        return value,list(entry.pv)
            alpha_start,beta_start=alpha,beta
            position=s.position_a if s.player==0 else s.position_b
            actions=[t for t in s.remaining if isfinite(self.distances[position][t])]
            if self.order=='reverse':actions.reverse()
            elif self.order=='greedy':
                actions.sort(key=lambda t:(-(self.values[t]-self.distances[position][t]),t))
            hint=self.hints.get(base) if self.use_hints else None
            if hint and hint[0] in actions:
                stats.hint_uses+=1
                stats.cross_call_hints+=hint[1]<self.generation
                actions.remove(hint[0]);actions.insert(0,hint[0])
            if not actions:
                stats.evaluated_leaves+=1
                return s.utility,[]
            stats.expanded+=1
            maximizing=s.player==0
            value,pv=(-inf if maximizing else inf),[]
            for index,target in enumerate(actions):
                candidate,tail=visit(transition(s,target,self.distances,self.values),depth-1,alpha,beta)
                if candidate>value if maximizing else candidate<value:
                    value,pv=candidate,[target]+tail
                if self.algorithm=='alphabeta':
                    if maximizing:alpha=max(alpha,value)
                    else:beta=min(beta,value)
                    rest=len(actions)-index-1
                    if alpha>=beta and rest:
                        stats.cutoffs+=1;stats.skipped_children+=rest
                        break
            # An interrupted parent never reaches this write. Completed child
            # entries remain valid for the next iteration or the next decision.
            if self.algorithm=='minimax':bound='EXACT'
            elif value<=alpha_start:bound='UPPER'
            elif value>=beta_start:bound='LOWER'
            else:bound='EXACT'
            if self.cache_values:
                old=self.table.get(key)
                if old is None or old.bound!='EXACT' or bound=='EXACT':
                    self.table[key]=TableEntry(value-s.utility,bound,tuple(pv),self.generation)
            if self.use_hints:
                self.hints[base]=(pv[0],self.generation)
            return value,pv
        value,pv=visit(state,horizon,-inf,inf,True)
        return GameResult(pv[0] if pv else None,value,pv,stats)

    def search(self,horizon,state=None):
        if not isinstance(horizon,int) or horizon<0:
            raise ValueError('horizon must be a nonnegative integer')
        state=state or GameState(remaining=tuple(sorted(self.values)))
        self.generation+=1
        return self._run(state,min(horizon,len(state.remaining)),ReuseStats())

    def iterative(self,max_depth,node_budget,state=None,depth_step=1):
        """Return the deepest completed iteration within a total node budget."""
        if (not isinstance(max_depth,int) or max_depth<0 or not isinstance(node_budget,int) or node_budget<1
                or depth_step not in (1,2)):
            raise ValueError('invalid depth/budget')
        state=state or GameState(remaining=tuple(sorted(self.values)))
        self.generation+=1
        stats=ReuseStats();completed=0;exhausted=False
        result=GameResult(None,state.utility,[],stats)
        maximum=min(max_depth,len(state.remaining))
        depths=(list(range(1,maximum+1)) if depth_step==1 else sorted({1,*range(2,maximum+1,2),maximum})) if maximum else []
        for depth in depths:
            try:
                candidate=self._run(state,depth,stats,node_budget)
            except NodeBudgetReached:
                exhausted=True
                break
            result=candidate;completed=depth
        return BudgetResult(result,completed,exhausted)
