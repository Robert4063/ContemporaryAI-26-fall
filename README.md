# ContemporaryAI-26-fall
26年秋季当代人工智能
# 实验二：搜索算法

实现 A*、Minimax、Alpha-beta，以及跨回合搜索复用和惯性跳棋策略。实验代码位于 [lab_2 分支](https://github.com/Robert4063/ContemporaryAI-26-fall/tree/lab_2)。

## 环境与一键复现

需要 **Python 3.10 或更新版本**。全部算法、测试、计时及结果核对仅使用 Python 标准库，无需安装第三方包，支持 Windows、Linux 和 macOS。本次实验使用 Python 3.13.2。

首次拉取并运行：

```bash
git clone --branch lab_2 --single-branch https://github.com/Robert4063/ContemporaryAI-26-fall.git
cd ContemporaryAI-26-fall/projects/experiment_2
python reproduce.py
```

已在仓库根目录时，也可以直接运行：

```bash
python projects/experiment_2/reproduce.py
```

若系统命令为 `python3`，将以上 `python` 替换为 `python3`。脚本始终在自身所在目录运行，结果写入本目录的 `results/`。

默认命令依次执行 35 项测试、全部 392 组实验、逐图对照导出和独立结果核对。任一步失败立即停止并返回非零退出码；成功时最后显示 `All requested steps passed.`。复现会覆盖已有的实验 CSV/JSON，最终核对记录为 `results/experiment_audit.json`。计时在不同机器上会变化，固定输入下的费用、最优值和节点统计用于核对算法结果。

只运行测试，或核对当前源码生成的已有结果：

```bash
python reproduce.py --tests-only
python reproduce.py --verify-only
```

## 直接运行课程四题

```bash
python solve.py --question 1
python solve.py --question 2
python solve.py --question 2 --heuristic manhattan
python solve.py --question 3 --algorithm minimax
python solve.py --question 3 --algorithm alphabeta
python solve.py --question 4 --horizon 2
python solve.py --question 4 --horizon 4
```

输出为标准 JSON。默认采用题目给定输入，预期结果如下：

| 实验 | 预期结果 |
| --- | --- |
| 迷宫寻路 | 最短 7 步，返回含起终点的 8 个坐标 |
| 加权地形 | 最小费用 7；进入终点 E 收费 0 |
| 极大极小博弈树 | 根节点最优值 5 |
| 宝藏分配 H=2 | 首选 T2，深度截断时收益差 -1 |
| 宝藏分配 H=4 | 首选 T3，完整终局收益差 3 |

`--heuristic` 支持 `zero`、`manhattan`、`safe_manhattan`，默认使用费用下界 `safe_manhattan`。`--algorithm` 支持 `minimax`、`alphabeta`，默认后者。

## 自定义输入

将 JSON 保存为本目录下的文件，例如 `input.json`，再运行：

```bash
python solve.py --question 1 --input input.json
```

第一题接受 `S`、`E`、`.`、`X`：

```json
{"start":[0,0],"goal":[0,2],"grid":[["S",".","E"]]}
```

第二题将普通格写为非负进入费用，`X` 或 `null` 表示障碍，`E` 收费 0：

```json
{"start":[0,0],"goal":[0,2],"grid":[["S",3,"E"]]}
```

第三题以 MAX 为根、逐层交替，可直接使用根对象或外包 `tree_structure`：

```json
{"tree_structure":{"is_maximizing_player":true,"children":[{"value":3},{"value":5}]}}
```

第四题的 `positions` 依次为 A、B、各宝藏位置，`values` 的键是位置索引，宝藏从索引 2 开始。所有可通行格均按数字计费，起点不收费；`horizon` 是双方行动总步数：

```json
{"grid":[[1,1,1],[1,1,1]],"positions":[[0,0],[1,2],[0,2],[1,0]],"names":["A","B","T1","T2"],"values":{"2":9,"3":6},"horizon":2}
```

```bash
python solve.py --question 4 --input input.json --algorithm alphabeta
```

第一题无解返回 `[]`，第二题无解返回 `-1`。第四题双方累计净收益为宝藏价值减移动费用，优化目标为 `U=Score_A-Score_B`；收益可为负，无跳过回合动作。

## 分步复现实验

以下命令与 `python reproduce.py` 等价：

```bash
python tests/test_search.py
python tests/test_reusable_search.py
python tests/test_direction_search.py
python run_experiments.py
python run_reuse_experiments.py
python run_direction_experiments.py
python compare_direction_results.py
python audit_experiments.py
```

| 实验脚本 | 配置数 | 输入与对照 |
| --- | ---: | --- |
| `run_experiments.py` | 106 | 21 组课程样例、80 组随机地图启发函数对照、5 组八宝藏搜索对照 |
| `run_reuse_experiments.py` | 180 | 6 张八宝藏地图；72 组完整/滚动深度搜索、108 组固定预算搜索 |
| `run_direction_experiments.py` | 106 | 11 张专设图、2 张课程图、40 张随机图，各比较 A* 与惯性跳棋策略 |
| 合计 | 392 | 每组结果均进行独立核对 |

所有计时预热 3 次、测量 21 批，取每次调用耗时的中位数。课程样例和惯性跳棋每批 10 次，原随机地图、八宝藏及跨回合复用每批 1 次。记录包含校验、搜索统计和状态保存开销；惯性跳棋计时关闭逐事件轨迹记录，完整轨迹另行生成。

### 第四题的收益排序

每次取宝后的得分变化为 `G=宝藏价值-移动费用`。因此用当前玩家的即时净收益 G 安排搜索顺序：优先搜索 G 大的动作，期望 A 更早找到较高 U 的分支、抬高 α，B 更早找到较低 U 的分支、压低 β，从而提早触发剪枝。

排序只改变访问次序，最终动作仍由 Minimax 递归回传确定；当 `α≥β` 时，当前分支剩余子节点已不能改善祖先的选择，才跳过它们。课程地图初始时 T2 的即时净收益为 `6-4=2`，因此先搜索 T2；完整 H=4 搜索仍选终局收益差为 3 的 T3。H=4 的 Alpha-beta 自然顺序访问 47 个节点，收益排序访问 37 个，减少约 21.3%，最优值一致。

### 跨回合树复用

**本实验是在原有 Minimax 记忆化缓存上增加跨回合保存。** 基础思想仍是保存已求解的子问题结果，遇到相同子问题时直接读取，与动态规划复用子问题结果的思想相通。原实现每次调用 `search_game(..., memoize=True)` 都新建缓存；新实现由 `GameSearchSession` 保存缓存，供同一会话的后续搜索继续使用。

例如，第一回合已经计算过“A取T3后，B如何选择”的后续局面；实际取走T3后，下一回合就可能命中这些记录，直接读取已算出的价值或界限，省去相同局面的重复展开。

种子 `20261111` 至 `20261116` 分别生成 6 张 10×10 地图及各自的 8 个宝藏位置与价值。**每张地图单独建立搜索会话，只在同一地图连续的 8 个回合之间复用；换地图后从空缓存开始。** 各策略接收相同的、由独立完整 Minimax 生成的局面序列。

缓存键包含双方位置、剩余宝藏、当前玩家和剩余深度。存储相对当前收益差的未来增量及 `EXACT/LOWER/UPPER` 类型；同深度复用数值，异深度只使用历史行动排序。比较完整剩余深度、滚动 H=4，以及每次决策 64、256、1024 节点的预算；已完成和中断的迭代均计入预算。每段计时从冷启动开始，计入首回合的搜索成本。

本节以是否跨回合保留记忆化缓存为对照变量，记录省去的重复搜索，并比较固定节点预算下的搜索深度和决策质量。

### 惯性跳棋策略

A* 先选定一步。在格子 p 保存状态，沿上一移动方向进入 q 后计算 `f(q)=g(q)+h(q)`；`f(q)≤f(p)` 则继续，`f(q)>f(p)` 则退回 p 并恢复 A*。所有延迟侧枝保留，保存栈顶与 OPEN 按 `(f,h)` 竞争。遇到墙、边界或已有更优 g 时恢复常规选点。

输入包括环形捷径、加权环、死路、横竖直廊、蛇形走廊、开放地图、费用突增、强制绕行、梳形和阶梯地图，以及种子 `20261201` 至 `20261220` 生成的单位权、加权随机地图。开放地图为 41×41、无障碍、每格费用 1，起终点位于左上和右下。

`results/direction/comparison.csv` 和 `地图逐项对照.md` 对全部 53 张地图分别列出费用、耗时、时间比值、优者和回退费用。最终路径费用与被退回移动的额外往返费用分开记录；OPEN 的状态切换未建模为机器人连续运动。

## 结果与目录

| 路径 | 内容 |
| --- | --- |
| `search_algorithms.py` | A*、Minimax、Alpha-beta、状态转移、距离矩阵 |
| `reusable_search.py` | 持久转置表、行动提示和限节点迭代加深 |
| `direction_search.py` | 惯性跳棋策略与分支恢复 |
| `fixtures.py`、`solve.py` | 课程输入及四题 JSON 接口 |
| `reproduce.py` | 跨平台一键测试、实验和核对入口 |
| `tests/` | 35 项测试及独立参考算法 |
| `audit_experiments.py` | 课程和随机基线核对，汇总两个拓展实验核对 |
| `audit_reuse_experiments.py`、`audit_direction_experiments.py` | 拓展实验的数值、路径、预算、轨迹、计时与哈希核对 |
| `results/core_metrics.csv` | 四题课程配置 |
| `results/synthetic_metrics.csv`、`large_game_metrics.csv` | 随机地图及八宝藏配置 |
| `results/summary.json`、`traces/` | 基线计时、统计、路径和搜索轨迹 |
| `results/inputs/`、`large_game_input.json` | 基线实验输入 |
| `results/reuse/` | 跨回合复用的输入、完整/限额指标、逐回合记录及核对 |
| `results/direction/` | 惯性跳棋的输入、逐图对照、轨迹及核对 |
| `results/experiment_audit.json` | 全部 392 组配置的最终核对结果 |
| `docs/实验二要求_new.pdf` | 课程实验要求 |
| `reports/实验二报告.pdf` | 最终五页报告，保留提交时的实测数据 |

复现只更新实验数据，不修改最终报告。目录中不包含 PDF 编写、排版、渲染或打包脚本。
