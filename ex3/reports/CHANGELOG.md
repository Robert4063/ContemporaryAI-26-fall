# 报告修订记录

## 2026-10-15：按评分标准整理GitHub提交目录

- 依据课程PDF第16、18页补充评分材料对应表、固定依赖和README复现命令。
- 最终Notebook、Markdown、PDF、prediction.csv保持原件；旧报告备份和历史保护基准移入本地忽略的 `.local/archive/`，以下历史记录中的旧路径据此理解。
- Git仅保留最终LinearSVC模型、完整指标/loss/对比图与原始CSV；其余训练模型和可再生预处理缓存保留本地并忽略。旧交接、草稿、日志及两个失效脚本归档。
- 预测脚本支持不带预处理缓存的仓库副本，仍检查存在的缓存文本/顺序；新增提交核查工具，已知局限移入统一文档并继续列入TODO。

## 2026-10-15：改为论文字体与黑白三线表

- 仅修改PDF导出样式：中文正文使用宋体，英文与数字使用Times New Roman；标题与节标题用黑体，主标题居中。正文9.5pt，图注8.1pt，表格8.3pt，适配两页篇幅。
- 正文、图注、页脚统一为纯黑色；三张表去除蓝色底纹，采用黑色三线表（上下线0.8pt、表头下线0.45pt），表头和数值居中，参数说明保留左对齐。插图配色保持原样。
- 为Times New Roman未覆盖的数学符号使用中文字体回退，确认集合符号∈等正常显示。
- Notebook与Markdown内容及文件哈希均未改变，保留全部措辞、前三部分、表格数据与图片。重新导出两页PDF并逐页检查，更新 `assets/report_qa.json` 和TODO；样式保存在 `../scripts/reports/export_report.py` 中供后续复现。

## 2026-10-15：以用户最终Notebook同步并导出两页PDF

- 主源为用户当前 `实验一报告.ipynb`，原稿SHA256为 `080d038af402ffc5bc7a1976568ce14fbcbdc11b9d49d8ad454ab185a5f6c243`，备份至 `archive/实验一报告_最终版导出前.ipynb`；本次基准为 `assets/report_final_export_baseline.json`。
- 原文直接导出为四页，两页只有少量溢出内容。仅精简第四至七部分的重复表达，Markdown单元源字符数由3510减至3252，减少258个字符；标题、仓库说明、前三部分单元及所有表格数值保持不变。
- 保留loss分类与趋势、三种模型/清洗效果对比、表达能力与泛化的讨论、划分问题、加宽/加深与过拟合、最终预测的原有顺序；沿用“与此相比”“相比来说”“但是很奇怪”“继续分析会发现”“那么”“最后”等连接方式。没有用历史分析文档替换用户新写的正文。
- Notebook同步导出同文Markdown和两页PDF，仍包含三张图片；逐页渲染检查无裁切、重叠或溢出。导出脚本与绘图数据未改动，无模型重训，prediction.csv哈希不变。
- 更新README、HANDOVER、TODO、要求清单与 `assets/report_qa.json`。用户最终Notebook未保留历史版局限说明，相关事实继续保留在背景文档和TODO，未擅自补回正文。

## 2026-10-15：按用户Notebook重排第四、五部分

- 以用户最新 `.ipynb` 为主，完整保留标题/仓库说明及前三部分单元。修改前备份至 `archive/实验一报告_第四五部分修改前.ipynb`，校验基准为 `assets/report_revision_baseline.json`；未替用户修正其原文措辞或标点。
- 第四部分集中放置SGD SVM与MLP隐藏层loss合并图、三模型Accuracy图，并重写loss、加宽/加深和SVM/MLP比较分析；完整loss合并图增加9组关键轮次数值表。
- 第五部分按用户确认合并SVM与MLP两张版本效果图，比较三个划分下的验证Accuracy和宏F1；明确均值与第四部分最佳配置的口径区别。
- 修复原版本图raw/body_only前两个点与横轴标签错位：按规范版本顺序重排数据，重新生成两张原图及合并图。
- 为两页限制压缩第六、七部分并调整排版，保留重复文本、历史test访问、单种子等局限及最终预测说明；Notebook、Markdown、PDF、TODO同步。
- 详细分析见 `docs/REPORT_SECTIONS_4_5_ANALYSIS.md`，新图和均值摘要见 `results/evaluation/`；训练结果与prediction.csv未改变。

## 2026-10-15：隐藏层loss对比与实验思考1、2

- 新增 `results/evaluation/figures/mlp_split_B_hidden_size_comparison.png`：固定B和subject_body_author，以两种α分别比较三种h；采用鲜明的蓝、橙、品红、不同线型和关键轮次标记。
- 图内明确训练条件、loss名称与数据来源，并列出关键轮次loss及train/val Accuracy；配套保存完整300轮CSV、6组摘要CSV及文件哈希清单。
- 新增 `docs/EXPERIMENT_THINKING_Q1_Q2.md`：解释稀疏高维文本为何适合线性分类器，并结合已保存的6组MLP结果讨论加宽、加深、增加epochs及过拟合风险。逻辑回归未实测，超过300轮、早停与多种子实验也未实施，均已明确区分。
- 同步修改 `实验一报告.ipynb`，重新导出同文Markdown和不超过两页的PDF；保留SGD SVM/MLP loss和三模型Accuracy比较，改为显式回答思考题1、2。
- 图表仅读取已有数据，没有重新训练模型或改变prediction.csv；本地test未重新用于本次结构比较。
