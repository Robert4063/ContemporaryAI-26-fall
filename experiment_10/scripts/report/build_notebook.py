"""Build the editable report; report_page metadata controls the two-page export."""
from pathlib import Path
import sys
import nbformat as nbf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_utils import ROOT, load_evidence, markdown_table
ctx = load_evidence()
nb = nbf.v4.new_notebook()
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "report": {"max_pages": 2}})
cells = []
def md(source, page=1):
    cells.append(nbf.v4.new_markdown_cell(source, metadata={"report_page": page}))
def code(source, page=1, hidden=False):
    tags = ["remove-input"] + (["report-hidden", "remove-cell"] if hidden else [])
    cells.append(nbf.v4.new_code_cell(source, metadata={"report_page": page, "tags": tags}))
code('''# PDF显示正文与输出，计算代码保留在Notebook中。
from pathlib import Path
import sys
from IPython.display import Markdown, Image, display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data-Project1/train_data.csv").exists())
sys.path.insert(0, str(ROOT / "scripts/report"))
from report_utils import load_evidence, results_table, make_comparison, verify_and_predict, write_manifest
ctx = load_evidence()
scores = verify_and_predict(ctx)  # 依据验证日志锁定配置后，重新拟合最终SVM
''', hidden=True)
md("# 实验一：文本分类\n\n"
   "## 1. 任务、问题与假设\n\n"
   "任务是用课程提供的英文新闻文本训练10分类模型，并预测无标签文本。流程为加载与划分、文本清洗、TF-IDF、训练、验证调参、测试评价和预测。"
   "我们检验三个假设：增加训练比例可能改善泛化；保留标题可能提供主题线索；增加MLP容量未必优于线性模型。以原文、80:10:10、SVM(C=1)为基线，按验证Accuracy排序，并列时比较宏F1。")
md("## 2. 数据加载、探索与划分\n\n"
   "来源为课程data-Project1：train_data.csv有7368条text/target记录，标签为0-9；test_data_unlabeled.csv有2457条text记录。"
   "每类705-747条，文本长度中位数1139字符，无缺失或完全重复文本。先按类别分层划分，再处理各子集，随机种子42；未删除样本，也未做数据增强。"
   "训练集拟合参数，验证集调参，本地测试集评价；课程无标签数据只用于最终预测。\n\n" +
   markdown_table(["划分", "训练:验证:测试", "训练数", "验证数", "测试数"], ctx["split_rows"]))
md("## 3. 文本预处理与特征工程\n\n"
   "四种输入为raw（原文）、body_only（移除已知邮件头，保留正文）、subject_body（标题+正文）、subject_body_author（作者+标题+正文）。"
   "固定规则逐行处理，各版本样本顺序与标签一致；未知头字段、引用和签名可能残留。作者可提供来源线索，也可能使跨作者泛化变差。\n\n"
   "使用TfidfVectorizer转小写、分词，min_df=2、max_df=0.95、max_features=20000。TF表示词频，IDF降低常见词权重；L2归一化减轻文档长度影响。"
   "词表与IDF仅在训练集拟合，再transform其他子集，避免统计量泄露。最终矩阵有20000维，约99.37%元素为零。")
md("## 4. 模型原理与训练设置\n\n"
   "LinearSVC学习最大间隔线性边界，用L2正则与平方hinge损失，按一对其余方式完成10分类，适合高维稀疏特征。"
   "MLP使用ReLU隐藏层、softmax输出及交叉熵，学习非线性关系；两者使用相同TF-IDF输入。\n\n" +
   markdown_table(["模型", "候选参数与训练设置"], [
       ["Linear SVM", "C∈{0.1,1,10}；max_iter=5000；seed=42"],
       ["MLP", "h∈{(50,),(100,),(100,50)}；α∈{0.0001,0.001}"],
   ]) + "\n\nMLP使用Adam、学习率0.001、batch_size=200、seed=42，训练300轮；关闭早停，tol=0、n_iter_no_change=301保持轮数一致。"
   "选定SVM约20万参数，MLP(50,)约100万参数，后者容量更大。")
md("## 5. 对照实验、调参与结果\n\n"
   "三种划分×四种文本×(3种SVM+6种MLP配置)，共108组；分别固定模型、划分或文本，只改变研究因素。"
   "SVM的C按等比网格搜索，MLP考察结构与正则强度。各模型按验证指标选定的配置如下，均为B划分、作者+标题+正文；F1取宏平均，测试分数不参与本次排序。")
code("display(Markdown(results_table(ctx)))")
md("## 6. 并列图表与结果分析", page=2)
code("figure = make_comparison(ctx)\ndisplay(Image(filename=str(figure)))", page=2)
md("图1：(a)固定MLP原文、h=(100,50)、α=0.001，改变划分；(b)固定B及相同MLP参数，改变文本；"
   "(c)固定B、作者+标题+正文、α=0.001，改变结构；(d)固定B，比较SVM的C与文本。(a)-(c)为300轮真实训练loss，采用对数纵轴；(d)为验证Accuracy。", page=2)
md("**划分与文本。** 图(a)(b)中训练loss随迭代降低，但不能代替验证结果。固定SVM(C=1)、作者+标题+正文，A/B/C验证Accuracy为94.98%/96.48%/94.85%，"
   "增加训练比例没有单调改善。A/B测试样本相同，C不同，验证样本也随划分变化，因此不是固定验证集的纯训练规模实验。固定B和C=1，"
   "正文/标题+正文/原文/作者+标题+正文为94.31%/96.21%/96.21%/96.48%：标题有帮助，作者仅多判对1条，不足以确认稳定收益。\n\n"
   "**结构与正则。** 图(c)中各结构loss都下降；对应h=(50,)/(100,)/(100,50)的验证Accuracy为96.21%/95.93%/94.04%，扩大模型未改善泛化。"
   "双层MLP将α从0.0001增至0.001后，验证分数从91.60%升至94.04%，说明正则强度值得调节；缺少逐轮验证loss，不能定位过拟合轮次。"
   "图(d)中最终文本的C=0.1/1/10验证分数为94.04%/96.48%/96.48%；C=1的宏F1略高于C=10（96.51%与96.49%），故按预定规则选C=1。\n\n"
   "**分类性能与错误。** SVM比MLP验证Accuracy高0.27个百分点，仅多判对1条；其参数更少，体现性能与复杂度的权衡。SVM本地测试737条中错44条，"
   "类别2召回率最低（64/74=86.49%），类别1和7各有5条被判为2。错误集中提示需检查这些类别的词汇重叠，但数字标签不能直接说明语义原因。", page=2)
md("## 7. 最终预测、结论与复现边界", page=2)
code('''display(Markdown(
    "最终选择B、subject_body_author、Linear SVM(C=1)，仅用6262条训练记录重新拟合。"
    f"本地测试Accuracy={100*scores['test_acc']:.2f}%，宏F1={100*scores['test_f1']:.2f}%，"
    f"宏Precision/Recall={100*scores['test_precision']:.2f}%/{100*scores['test_recall']:.2f}%。"
    "对2457条无标签文本生成reports/predictions.csv：无表头，每行一个0-9标签，保持原始输入顺序。"))''', page=2)
md("**结论与局限。** 高维稀疏TF-IDF下，线性SVM已具有较好表现；标题保留优于只留正文，增加网络容量未改善结果。"
   "当前仅一个随机种子，不能据微小差异宣称显著优势；后续应重复种子、按作者分组验证，并与词袋、Word2Vec或BERT表示对照。"
   "本次分析沿用既有日志，旧脚本曾对每个候选计算测试分数，因此不满足严格测试封存规范，需用预先封存的独立数据重做最终评价。\n\n"
   "**复现。** Windows CPU；Python 3.13.2、sklearn 1.8.0、NumPy 2.4.6、pandas 2.3.0，seed=42。README提供数据处理、训练与报告运行命令；"
   "Notebook保留核验、画图和预测代码，evidence_manifest.json记录输入摘要、依赖与配置。已重新拟合核验SVM，MLP指标及loss来自历史日志，未重训全部网格。", page=2)
code("write_manifest(ctx)", page=2, hidden=True)
nb.cells = cells
path = ROOT / "reports" / "实验一报告.ipynb"
nbf.write(nb, path)
print(path)
