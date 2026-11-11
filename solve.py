"""Run PDF samples or a JSON input. See README for schemas."""
import argparse
import json
from pathlib import Path
from fixtures import *
from search_algorithms import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--question',type=int,choices=range(1,5),required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--horizon',type=int,default=4)
    p.add_argument('--algorithm',choices=['minimax','alphabeta'],default='alphabeta')
    p.add_argument('--heuristic',choices=['zero','manhattan','safe_manhattan'],default='safe_manhattan')
    args=p.parse_args()
    data=json.loads(args.input.read_text(encoding='utf-8-sig')) if args.input else {}
    if args.question==1:
        out=solve_maze(tuple(data.get('start',MAZE_START)),tuple(data.get('goal',MAZE_GOAL)),data.get('grid',MAZE),args.heuristic)
    elif args.question==2:
        out=solve_weighted(tuple(data.get('start',WEIGHTED_START)),tuple(data.get('goal',WEIGHTED_GOAL)),data.get('grid',WEIGHTED),args.heuristic)
    elif args.question==3:
        root=tree_from_dict(data) if data else example_tree()
        out=search_tree(root,args.algorithm)[0]
    else:
        grid=data.get('grid',TREASURE_GRID)
        positions=[tuple(x) for x in data.get('positions',POSITIONS)]
        names=data.get('names',NAMES)
        values={int(k):v for k,v in data.get('values',VALUES).items()}
        distances=pairwise_distances(grid,positions)
        result=search_game(distances,values,data.get('horizon',args.horizon),args.algorithm)
        out={'distance_order':names,'distance_matrix':distances,
             'best_target':names[result.best_target] if result.best_target is not None else None,
             'best_value':result.best_value}
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
