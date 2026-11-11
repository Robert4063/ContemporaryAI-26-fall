"""Export one comparison row per input, including every random map."""
import json
from pathlib import Path
from run_experiments import save_csv

ROOT=Path(__file__).resolve().parent


def comparisons(summary):
    baseline={r['case']:r for r in summary['metrics'] if r['method']=='astar'}
    rows=[]
    for b in summary['metrics']:
        if b['method']!='inertia_astar':continue
        a=baseline[b['case']]
        row=dict(case=b['case'],category=b['category'],optimal_cost=b['optimal_cost'],
            cost_astar=a['cost'],cost_inertia=b['cost'],time_astar_us=a['median_us'],time_inertia_us=b['median_us'],
            ratio_inertia_over_astar=b['median_us']/a['median_us'],winner='惯性跳棋策略' if b['median_us']<a['median_us'] else 'A*',
            backtracks=b['backtracks'],rollback_cost=b['rollback_cost'])
        for key in ['expanded','heuristic_calls','queue_pops','peak_frontier']:
            row[key+'_astar']=a[key];row[key+'_inertia']=b[key]
        rows.append(row)
    return rows


def main():
    folder=ROOT/'results/direction'
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    rows=comparisons(summary);save_csv(folder/'comparison.csv',rows)
    lines=['# 惯性跳棋策略与 A*：53 张地图逐项对照','',
        '惯性跳棋策略：A* 选定一步后，保存当前位置 p，沿刚才方向先前进到 q，再计算 f(q)=g(q)+h(q)。'
        'f(q)≤f(p) 则继续；f(q)>f(p) 则退回 p 并恢复 A*。保存全部侧枝，栈顶与 OPEN 按 (f,h) 竞争。','',
        '两法全部 53/53 达到独立 Dijkstra 最优费用。耗时为预热 3 次后、21 批各 10 次完整搜索的单次中位数。'
        '比值为惯性跳棋策略/A*，小于 1 表示惯性跳棋策略更快；额外往返费为被退回的移动进入 q 和退回 p 的费用之和，另计于最终路线费用。'
        '搜索中的 OPEN 切换是算法状态切换，未建模为机器人连续运动。','',
        '| 地图 | 最优费 | A*/μs | 惯性跳棋策略/μs | 比值 | 更快 | 回退次数/额外往返费 |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        lines.append(f"| {r['case']} | {r['optimal_cost']} | {r['time_astar_us']:.2f} | {r['time_inertia_us']:.2f} | "
            f"{r['ratio_inertia_over_astar']:.3f} | {r['winner']} | {r['backtracks']}/{r['rollback_cost']} |")
    (folder/'地图逐项对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(maps=len(rows),inertia_faster=sum(r['winner']=='惯性跳棋策略' for r in rows)),ensure_ascii=False))


if __name__=='__main__':main()
