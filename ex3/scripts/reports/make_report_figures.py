"""Create compact report figures from stored loss/validation records only."""
from pathlib import Path
import runpy
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False})

hidden_plot = runpy.run_path(str(ROOT / "scripts/plots/plot_mlp_hidden_size_split_B.py"))
curves = hidden_plot["load_curves"]()
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.35))
fig.subplots_adjust(left=.06, right=.99, bottom=.24, top=.80, wspace=.35)
for c, color in zip([.1, 1., 10.], ["#1675E8", "#FF8500", "#D92780"]):
    loss = np.load(ROOT / "results/loss" / f"svm_split_B_subject_body_author_C{c}.npy")
    axes[0].plot(np.arange(1, len(loss)+1), loss, lw=.8, color=color, label=f"C={c:g}")
for ax, alpha in zip(axes[1:], hidden_plot["ALPHAS"]):
    for row in [r for r in curves if r["alpha"]==alpha]:
        h=row["hidden"]
        ax.plot(np.arange(1,301),row["loss"],lw=1.2,color=hidden_plot["COLORS"][h],
                linestyle=hidden_plot["STYLES"][h],label=f'h={hidden_plot["LABELS"][h]}')
for ax, title, ylabel in zip(axes, ["SGD SVM: varying C", "MLP: alpha=0.0001", "MLP: alpha=0.001"], ["Margin hinge (log)", "Cross-entropy + L2 (log)", "Cross-entropy + L2 (log)"]):
    ax.set_yscale("log")
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel("Recorded training round (1-300)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1,300)
    ax.grid(alpha=.20, which="both", lw=.5)
    ax.legend(loc="upper right", frameon=False)
for ax in axes[1:]:
    ax.set_ylim(.0002,3.5)
fig.savefig(OUT / "report_loss_comparison.png", dpi=240)
plt.close(fig)

best = pd.read_csv(ROOT / "results/linear_svm_split_B/best_by_model_version.csv")
versions = ["raw", "body_only", "subject_body", "subject_body_author"]
models = ["svm", "mlp", "linear_svm"]
values = best.pivot(index="version",columns="model",values="accuracy").reindex(index=versions,columns=models)
fig, ax = plt.subplots(figsize=(9.5,2.25))
fig.subplots_adjust(left=.065, right=.99, bottom=.25, top=.82)
for i, (model, label, color) in enumerate(zip(models,["SGD SVM (3 configs)","MLP (6 configs)","LinearSVC (3 configs)"],["#2878B5","#E88A36","#3B9970"])):
    bars = ax.bar(np.arange(4)+(i-1)*.24,values[model],width=.24,color=color,label=label,zorder=3)
    ax.bar_label(bars,labels=[f"{x:.2%}" for x in values[model]],padding=3,fontsize=7)
ax.set_ylim(.88,1.)
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.yaxis.set_major_locator(MultipleLocator(.02))
ax.set_ylabel("Validation accuracy")
ax.set_xticks(np.arange(4))
ax.set_xticklabels(["Raw email","Body only","Subject + body","Author + subject + body"])
ax.set_axisbelow(True)
ax.grid(axis="y",alpha=.25)
fig.legend(loc="upper center",bbox_to_anchor=(.53,1.02),ncol=3,frameon=False)
fig.savefig(OUT / "report_accuracy_comparison.png",dpi=240)
plt.close(fig)
print("Saved report loss and accuracy figures from existing results.")

# Full-size loss combination: the original MLP hidden-size data plus SGD SVM.
svm_rows = []
for c in [.1,1.,10.]:
    name=f"svm_split_B_subject_body_author_C{c}"
    loss_path=ROOT/"results/loss"/f"{name}.npy"
    metric_path=ROOT/"results/baseline"/f"{name}_metrics.csv"
    metrics=pd.read_csv(metric_path,float_precision="round_trip").set_index("eval_set")
    svm_rows.append({"config_id":name,"model":"SGD SVM","setting":f"C={c:g}",
                     "loss":np.load(loss_path),"loss_path":loss_path,"metrics_path":metric_path,
                     "train_accuracy":float(metrics.loc['train','accuracy']),
                     "val_accuracy":float(metrics.loc['val','accuracy'])})
all_loss_rows=svm_rows+[{**r,"model":"MLP","setting":f"a={r['alpha']:g}, h={hidden_plot['LABELS'][r['hidden']]}"} for r in curves]
fig=plt.figure(figsize=(15,9.6))
full_axes=[fig.add_axes([.055+i*.325,.49,.285,.32]) for i in range(3)]
fig.suptitle("split_B: SGD SVM and MLP training-loss comparison",x=.055,y=.97,ha="left",fontsize=21,fontweight="bold")
fig.text(.055,.91,"Fixed text: Author + Subject + Body | train=6,262, validation=369 | seed=42 | 300 recorded rounds | no smoothing",fontsize=11)
for row,color in zip(svm_rows,["#1675E8","#FF8500","#D92780"]):
    full_axes[0].plot(np.arange(1,301),row['loss'],color=color,lw=1.4,label=row['setting'])
for ax,alpha in zip(full_axes[1:],hidden_plot['ALPHAS']):
    for row in [r for r in curves if r['alpha']==alpha]:
        h=row['hidden']
        ax.plot(np.arange(1,301),row['loss'],color=hidden_plot['COLORS'][h],linestyle=hidden_plot['STYLES'][h],
                lw=2,marker=hidden_plot['MARKERS'][h],markevery=[0,49,99,199,299],markersize=4,label=f"h={hidden_plot['LABELS'][h]}")
    ax.set_ylim(.0002,3.5)
for ax,title,ylabel in zip(full_axes,["SGD SVM: varying C","MLP: alpha=0.0001","MLP: alpha=0.001"],
                           ["Multiclass margin hinge","Cross-entropy + L2","Cross-entropy + L2"]):
    ax.set_title(title,fontsize=12,fontweight="bold",pad=12);ax.set_yscale('log');ax.set_xlim(1,300)
    ax.set_ylabel(ylabel+" (log)",fontsize=10);ax.set_xlabel("Recorded round (1-300)",fontsize=10)
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=9,loc='upper right')
table_ax=fig.add_axes([.055,.13,.92,.25]);table_ax.axis('off')
columns=['Model','Configuration','Loss @1','Loss @50','Loss @100','Loss @200','Loss @300','Train Acc','Val Acc']
table_rows=[[r['model'],r['setting'],*[f"{r['loss'][i-1]:.6f}" for i in [1,50,100,200,300]],
             f"{r['train_accuracy']:.2%}",f"{r['val_accuracy']:.2%}"] for r in all_loss_rows]
table=table_ax.table(cellText=table_rows,colLabels=columns,cellLoc='center',loc='center',bbox=[0,0,1,1],colWidths=[.08,.20,.10,.10,.10,.10,.10,.10,.10])
table.auto_set_font_size(False);table.set_fontsize(9)
for (i,j),cell in table.get_celld().items():
    cell.set_edgecolor('#D7E0E9');cell.set_facecolor('#E8F0F9' if i==0 else '#FFFFFF' if i%2 else '#F6F9FC')
    if i==0:cell.set_text_props(fontweight='bold')
fig.text(.055,.405,"Stored values at checkpoints (6-decimal display; original values retained in NPY/CSV)",fontsize=12,fontweight='bold')
fig.text(.055,.055,"SGD loss is a multiclass margin diagnostic; MLP loss is the epoch mean of mini-batch cross-entropy + L2 objectives.",fontsize=10)
fig.text(.055,.025,"Compare trends within each training objective and fixed alpha. Smaller training loss does not imply higher validation accuracy.",fontsize=10)
combined_loss_path=ROOT/'results/evaluation/figures/svm_mlp_split_B_loss_combined.png'
fig.savefig(combined_loss_path,dpi=200,bbox_inches='tight');plt.close(fig)
loss_summary=[]
for r in all_loss_rows:
    loss_summary.append({'config_id':r['config_id'],'model':r['model'],'setting':r['setting'],
                         **{f'loss_iter_{i}':float(r['loss'][i-1]) for i in [1,50,100,200,300]},
                         'train_accuracy':r['train_accuracy'],'val_accuracy':r['val_accuracy']})
pd.DataFrame(loss_summary).to_csv(ROOT/'results/evaluation/svm_mlp_split_B_loss_summary.csv',index=False,float_format='%.17g')

# Merge the SVM and MLP version-effect comparisons from their underlying data.
# The original plotting function now reindexes versions explicitly (raw/body).
frames=[pd.read_csv(p,float_precision='round_trip') for p in sorted((ROOT/'results/baseline').glob('*_metrics.csv'))]
all_metrics=pd.concat(frames,ignore_index=True)
validation=all_metrics[all_metrics.eval_set=='val']
version_summary=validation.groupby(['model','split','version']).agg(
    mean_val_accuracy=('accuracy','mean'),mean_val_macro_f1=('macro_f1','mean'),n_configs=('accuracy','size')).reset_index()
version_summary.to_csv(ROOT/'results/evaluation/version_effect_svm_mlp_summary.csv',index=False,float_format='%.17g')
def version_figure(compact=False):
    fig,axes=plt.subplots(2,2,figsize=(9.5,3.0) if compact else (13.5,7),sharey=True)
    fig.subplots_adjust(left=.06,right=.99,bottom=.15,top=.86,hspace=.68 if compact else .5,wspace=.14)
    for col,model in enumerate(['svm','mlp']):
        for row,metric in enumerate(['mean_val_accuracy','mean_val_macro_f1']):
            ax=axes[row,col]
            for split,color,marker in zip(['split_A','split_B','split_C'],['#1675E8','#FF8500','#22A447'],['o','s','^']):
                block=version_summary[(version_summary.model==model)&(version_summary.split==split)].set_index('version').reindex(versions)
                assert not block[metric].isna().any()
                ax.plot(np.arange(4),block[metric],color=color,marker=marker,lw=1.3 if compact else 2.2,
                        markersize=3 if compact else 6,label=split)
            ax.set_title(f"{'SGD SVM' if model=='svm' else 'MLP'}: {'Accuracy' if row==0 else 'Macro F1'}",fontsize=8 if compact else 12,loc='left',pad=3)
            ax.set_ylim(.88,.97);ax.set_xticks(np.arange(4))
            labels=['Raw','Body','Subject+body','Author+\nsubject+body'] if compact else ['Raw','Body','Subject+body','Author+subject+body']
            ax.set_xticklabels(labels,fontsize=6.4 if compact else 9)
            if compact:ax.set_xlim(-.2,3.5)
            ax.tick_params(axis='y',labelsize=6.5 if compact else 9)
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_locator(MultipleLocator(.04 if compact else .02))
            ax.set_axisbelow(True);ax.grid(alpha=.22)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,1.01 if compact else .985),ncol=3,frameon=False,fontsize=7 if compact else 11)
    if not compact:
        fig.suptitle('Text-version effects: SGD SVM and MLP',x=.06,y=1.035,ha='left',fontsize=19,fontweight='bold')
        fig.text(.06,.035,'Validation means across hyperparameters: SGD SVM=3, MLP=6 per split/version. Same version order in every panel.',fontsize=10)
    return fig
fig=version_figure();version_path=ROOT/'results/evaluation/figures/version_effect_svm_mlp_combined.png'
fig.savefig(version_path,dpi=200,bbox_inches='tight');plt.close(fig)
fig=version_figure(compact=True);fig.savefig(OUT/'report_version_effect_comparison.png',dpi=260);plt.close(fig)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'version_order':versions,'version_statistic':'validation mean across hyperparameters, not best configuration',
          'original_axis_error_corrected':'raw/body_only data were alphabetical while the original labels assumed raw first',
          'version_source_images':{name:sha(ROOT/'results/evaluation/figures'/name) for name in ['version_effect_svm.png','version_effect_mlp.png']},
          'combined_loss_image_sha256':sha(combined_loss_path),'combined_version_image_sha256':sha(version_path),
          'hidden_size_source_image_sha256':sha(ROOT/'results/evaluation/figures/mlp_split_B_hidden_size_comparison.png'),
          'loss_sources':{r['loss_path'].relative_to(ROOT).as_posix():sha(r['loss_path']) for r in all_loss_rows},'refitted':False}
(ROOT/'results/evaluation/report_sections_4_5_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved combined loss and version-effect figures, report assets and source summaries.')
