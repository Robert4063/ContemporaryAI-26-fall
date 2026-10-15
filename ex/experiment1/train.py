# -*- coding: utf-8 -*-
"""
训练脚本：读取 data_processed/ 中已划分、已处理好的数据。

流程：先对每个模型配置计时 1 个 epoch 预估总耗时，然后直接开始训练。
每个 (划分, 预处理) 作业独立保存结果，已完成的作业会被跳过，可中断后续跑，
也可以用 --splits / --methods 在多个终端里分开跑。

模型选择只依据验证集准确率；测试集指标仅对验证集最优 epoch 的模型报告。
"""
import argparse
import copy
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data_processed')
OUT_DIR = os.path.join(ROOT, 'outputs')
RESULT_DIR = os.path.join(OUT_DIR, 'results')
MODEL_DIR = os.path.join(OUT_DIR, 'models')

SEED = 42
SPLITS = ['80_10_10', '85_5_10', '90_5_5']
METHODS = ['raw', 'body_only', 'subject_body', 'subject_body_author']
TFIDF_PARAMS = dict(min_df=2, max_df=0.95, max_features=20000)

SVM_CONFIGS = [{'C': c} for c in (0.1, 1, 10)]
MLP_CONFIGS = [{'hidden': h, 'alpha': a}
               for h in ((50,), (100,), (100, 50)) for a in (1e-4, 1e-3)]
MAX_EPOCHS = {'svm': 100, 'mlp': 100}
PATIENCE = 8  # 验证集准确率连续 PATIENCE 个 epoch 没有提升就停止


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def cfg_name(kind, cfg):
    if kind == 'svm':
        return f"SVM_C{cfg['C']}"
    return f"MLP_h{'_'.join(map(str, cfg['hidden']))}_a{cfg['alpha']}"


def load_job(split, method):
    """读取已划分好的数据；TF-IDF 的词表和 IDF 只在训练集上拟合。"""
    d = os.path.join(DATA_DIR, split, method)
    # keep_default_na=False：body_only 中有空文本，不能被读成 NaN
    rd = lambda k: pd.read_csv(os.path.join(d, f'{k}.csv'), keep_default_na=False)
    tr, va, te = rd('train'), rd('val'), rd('test')
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    Xtr = vec.fit_transform(tr.text)
    return {'train': (Xtr, tr.target.values),
            'val': (vec.transform(va.text), va.target.values),
            'test': (vec.transform(te.text), te.target.values),
            'vectorizer': vec}


def make_model(kind, cfg, n_train):
    if kind == 'svm':
        # alpha = 1/(C*n)：与 LinearSVC 的 C 含义对应；n_jobs=1 便于多终端并行
        return SGDClassifier(loss='hinge', penalty='l2', alpha=1.0 / (cfg['C'] * n_train),
                             learning_rate='optimal', random_state=SEED, n_jobs=1)
    return MLPClassifier(hidden_layer_sizes=cfg['hidden'], alpha=cfg['alpha'],
                         solver='adam', learning_rate_init=1e-3, random_state=SEED)


def train_loss(kind, model, X, y, classes):
    """每个 epoch 后的训练 loss。SVM：one-vs-rest 平均 hinge loss；MLP：sklearn 自带 loss。"""
    if kind == 'mlp':
        return float(model.loss_curve_[-1])
    Y = -np.ones((len(y), len(classes)))
    Y[np.arange(len(y)), np.searchsorted(classes, y)] = 1.0
    return float(np.maximum(0.0, 1.0 - Y * model.decision_function(X)).mean())


def all_configs():
    return [('svm', c) for c in SVM_CONFIGS] + [('mlp', c) for c in MLP_CONFIGS]


def train_one(kind, cfg, data, tag):
    """逐 epoch 训练；按验证集准确率保留最优 epoch 的模型，连续 PATIENCE 个 epoch 无提升则早停。"""
    Xtr, ytr = data['train']
    Xva, yva = data['val']
    Xte, yte = data['test']
    classes = np.unique(ytr)
    model = make_model(kind, cfg, Xtr.shape[0])
    best = {'val_acc': -1.0, 'epoch': 0, 'model': None}
    losses, val_accs, bad = [], [], 0
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS[kind] + 1):
        model.partial_fit(Xtr, ytr, classes=classes)
        losses.append(train_loss(kind, model, Xtr, ytr, classes))
        val_accs.append(float(accuracy_score(yva, model.predict(Xva))))
        if val_accs[-1] > best['val_acc']:
            best.update(val_acc=val_accs[-1], epoch=ep, model=copy.deepcopy(model))
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            log(f"  {tag} epoch {ep:3d} loss={losses[-1]:.4f} val_acc={val_accs[-1]:.4f}")
        if bad >= PATIENCE:
            break
    m = best['model']
    pv, pt = m.predict(Xva), m.predict(Xte)
    return {'name': cfg_name(kind, cfg), 'kind': kind, 'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
            'best_epoch': best['epoch'], 'epochs_run': len(losses),
            'val_acc': float(accuracy_score(yva, pv)), 'val_f1': float(f1_score(yva, pv, average='macro')),
            'test_acc': float(accuracy_score(yte, pt)), 'test_f1': float(f1_score(yte, pt, average='macro')),
            'train_time': time.time() - t0, 'loss_curve': losses, 'val_acc_curve': val_accs}, m


def result_path(split, method):
    return os.path.join(RESULT_DIR, f'{split}__{method}.json')


def run_job(split, method):
    """一个 (划分, 预处理) 作业：训练全部配置，保存结果和 SVM/MLP 各自的最优模型。"""
    log(f"=== job {split} / {method}: loading + TF-IDF")
    data = load_job(split, method)
    log(f"  train matrix {data['train'][0].shape}")
    results, best = [], {}
    for i, (kind, cfg) in enumerate(all_configs(), 1):
        tag = f"[{i}/{len(all_configs())}] {split}/{method}/{cfg_name(kind, cfg)}"
        log(tag)
        res, model = train_one(kind, cfg, data, tag)
        res.update(split=split, preprocess=method)
        results.append(res)
        log(f"  done: best_epoch={res['best_epoch']} val_acc={res['val_acc']:.4f} "
            f"test_acc={res['test_acc']:.4f} time={res['train_time']:.1f}s")
        if kind not in best or res['val_acc'] > best[kind][0]['val_acc']:
            best[kind] = (res, model)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for kind, (res, model) in best.items():
        with open(os.path.join(MODEL_DIR, f'{split}__{method}__best_{kind}.pkl'), 'wb') as f:
            pickle.dump({'model': model, 'vectorizer': data['vectorizer'], 'result': res}, f)
    with open(result_path(split, method), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    log(f"=== job {split} / {method} finished")


def estimate(jobs):
    """对每个配置计时 1 个 epoch，外推总耗时（上界：不早停，跑满 MAX_EPOCHS）。"""
    split, method = SPLITS[0], METHODS[0]
    log(f"estimating on {split}/{method} (1 epoch per config) ...")
    data = load_job(split, method)
    Xtr, ytr = data['train']
    classes = np.unique(ytr)
    per_kind = {'svm': 0.0, 'mlp': 0.0}
    for kind, cfg in all_configs():
        model = make_model(kind, cfg, Xtr.shape[0])
        t = time.time()
        model.partial_fit(Xtr, ytr, classes=classes)
        train_loss(kind, model, Xtr, ytr, classes)
        dt = time.time() - t
        per_kind[kind] += dt
        log(f"  1 epoch {cfg_name(kind, cfg)}: {dt:.2f}s")
    one_job_max = sum(per_kind[k] * MAX_EPOCHS[k] for k in per_kind)
    log(f"estimated upper bound per job: {one_job_max/60:.1f} min; "
        f"{len(jobs)} job(s) serial: {one_job_max*len(jobs)/60:.1f} min "
        f"(early stopping, patience={PATIENCE}, will usually be shorter)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=SPLITS, choices=SPLITS)
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--no-estimate', action='store_true', help='跳过耗时预估')
    ap.add_argument('--force', action='store_true', help='重跑已完成的作业')
    args = ap.parse_args()

    jobs = [(s, m) for s in args.splits for m in args.methods]
    todo = [j for j in jobs if args.force or not os.path.exists(result_path(*j))]
    log(f"jobs: {len(jobs)} total, {len(todo)} to run, {len(jobs) - len(todo)} already done")
    os.makedirs(RESULT_DIR, exist_ok=True)
    if todo and not args.no_estimate:
        estimate(todo)
    for s, m in todo:
        run_job(s, m)
    log("all requested jobs finished")


if __name__ == '__main__':
    main()
