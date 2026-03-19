"""
全样本训练 + 超参数优化 — 寻找最佳模型
==========================================
使用XGBoost GPU对全部18956样本进行:
  1. 网格搜索超参数优化 (5折CV)
  2. 用最优参数在全样本上训练最终模型
  3. 保存最优模型、特征重要性、预测结果

两个预测目标:
  - gap_days: 两次住院间隔天数
  - next_los: 下次住院时长

用法:
  python readmission_full_train.py
"""

import os
import sys
import io
import json
import time
import warnings
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


OUTPUT_DIR = r'D:\LDH_cancer\files\healthline\readmission_output'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'best_model')


# ============================================================
# 超参数搜索空间
# ============================================================

PARAM_GRID = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    'min_child_weight': [1, 3, 5, 10],
    'gamma': [0, 0.1, 0.3, 0.5],
}

# 分阶段搜索：先粗调核心参数，再微调其余参数
STAGE1_GRID = {
    'n_estimators': [800, 1200, 2000],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
}

STAGE2_GRID = {
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 10],
}

STAGE3_GRID = {
    'reg_alpha': [0, 0.05, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    'gamma': [0, 0.1, 0.3, 0.5],
}


# ============================================================
# 工具函数
# ============================================================

def load_data():
    """加载所有数据"""
    X_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'structured_features.csv'))
    targets = np.load(os.path.join(OUTPUT_DIR, 'targets.npz'))
    y_gap = targets['y_gap']
    y_los = targets['y_los']
    
    feature_names = list(X_df.columns)
    
    # 填充NaN为中位数
    for col in X_df.columns:
        median_val = X_df[col].median()
        X_df[col] = X_df[col].fillna(median_val if pd.notna(median_val) else 0)
    
    X = X_df.values.astype(np.float32)
    
    # 加载文本统计特征
    texts_path = os.path.join(OUTPUT_DIR, 'llm_texts.txt')
    text_feats = None
    if os.path.exists(texts_path):
        with open(texts_path, 'r', encoding='utf-8') as f:
            content = f.read()
        texts = []
        for block in content.split('===RECORD_')[1:]:
            lines = block.split('\n', 1)
            texts.append(lines[1].strip() if len(lines) > 1 else '')
        
        if len(texts) == len(X):
            text_feats = extract_text_features(texts)
    
    return X, y_gap, y_los, feature_names, text_feats


def extract_text_features(texts):
    """提取文本统计特征"""
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ''
        feat = [
            len(text),
            text.count('\n'),
            sum(c.isdigit() for c in text),
            sum('\u4e00' <= c <= '\u9fff' for c in text),
            int('诊断' in text),
            int('检验' in text),
            int('检查' in text),
            int('费用' in text),
            int('医嘱' in text),
            text.count('='),
            text.count(',') + text.count('，'),
        ]
        features.append(feat)
    return np.array(features, dtype=np.float32)


def cv_evaluate(X, y, params, n_splits=5, random_state=42):
    """
    5折CV评估给定参数组合
    
    Returns: dict with mae_mean, rmse_mean, r2_mean
    """
    valid_mask = ~np.isnan(y)
    X_v = X[valid_mask]
    y_v = y[valid_mask]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    maes, rmses, r2s = [], [], []
    
    for tr_idx, va_idx in kf.split(X_v):
        X_tr, X_va = X_v[tr_idx], X_v[va_idx]
        y_tr, y_va = y_v[tr_idx], y_v[va_idx]
        
        model = xgb.XGBRegressor(
            **params,
            verbosity=0,
            early_stopping_rounds=50,
            random_state=random_state,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        y_pred = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_va, y_pred)))
        r2s.append(r2_score(y_va, y_pred))
    
    return {
        'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
    }


def grid_search_stage(X, y, base_params, search_grid, target_name, 
                       n_splits=5, optimize_metric='rmse'):
    """
    对给定搜索网格进行穷举搜索
    
    Parameters
    ----------
    base_params : dict
        固定的基础参数
    search_grid : dict
        待搜索的参数 {param_name: [values]}
    optimize_metric : str
        'rmse' or 'mae' or 'r2'
    
    Returns
    -------
    best_params : dict
        最优参数组合
    all_results : list[dict]
        所有组合的结果
    """
    param_names = list(search_grid.keys())
    param_values = list(search_grid.values())
    combinations = list(itertools.product(*param_values))
    
    total = len(combinations)
    print(f'\n  [{target_name}] 搜索 {total} 种参数组合: {param_names}')
    
    all_results = []
    best_score = float('inf') if optimize_metric != 'r2' else float('-inf')
    best_params = {}
    
    for i, combo in enumerate(combinations, 1):
        params = dict(base_params)
        for k, v in zip(param_names, combo):
            params[k] = v
        
        metrics = cv_evaluate(X, y, params, n_splits=n_splits)
        
        result = {**{k: v for k, v in zip(param_names, combo)}, **metrics}
        all_results.append(result)
        
        if optimize_metric == 'r2':
            score = metrics['r2_mean']
            is_better = score > best_score
        else:
            score = metrics[f'{optimize_metric}_mean']
            is_better = score < best_score
        
        if is_better:
            best_score = score
            best_params = {k: v for k, v in zip(param_names, combo)}
            marker = ' *** BEST ***'
        else:
            marker = ''
        
        if i <= 5 or i % 10 == 0 or marker:
            combo_str = ', '.join(f'{k}={v}' for k, v in zip(param_names, combo))
            print(f'    [{i:3d}/{total}] {combo_str}  '
                  f'MAE={metrics["mae_mean"]:.2f} RMSE={metrics["rmse_mean"]:.2f} '
                  f'R²={metrics["r2_mean"]:.4f}{marker}')
    
    return best_params, all_results


def staged_hyperparameter_search(X, y, target_name, device='cuda', n_splits=5):
    """
    分阶段超参数搜索
    
    Stage 1: 核心参数 (n_estimators, max_depth, learning_rate)
    Stage 2: 采样参数 (subsample, colsample_bytree, min_child_weight)
    Stage 3: 正则化参数 (reg_alpha, reg_lambda, gamma)
    
    Returns
    -------
    best_params : dict
        最优参数组合
    search_history : dict
        每阶段的搜索历史
    """
    print(f'\n{"="*60}')
    print(f'超参数优化: {target_name}')
    print(f'{"="*60}')
    
    base_params = {
        'device': device,
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0,
    }
    
    search_history = {}
    
    # --- Stage 1: 核心参数 ---
    print('\n--- Stage 1: 核心参数优化 (n_estimators, max_depth, learning_rate) ---')
    t1 = time.time()
    best_s1, results_s1 = grid_search_stage(
        X, y, base_params, STAGE1_GRID, target_name, n_splits)
    base_params.update(best_s1)
    search_history['stage1'] = {
        'params_searched': list(STAGE1_GRID.keys()),
        'best': best_s1,
        'n_combos': len(results_s1),
        'time_sec': time.time() - t1,
    }
    print(f'  Stage 1 最优: {best_s1} (耗时 {time.time()-t1:.1f}s)')
    
    # --- Stage 2: 采样参数 ---
    print('\n--- Stage 2: 采样参数优化 (subsample, colsample_bytree, min_child_weight) ---')
    t2 = time.time()
    best_s2, results_s2 = grid_search_stage(
        X, y, base_params, STAGE2_GRID, target_name, n_splits)
    base_params.update(best_s2)
    search_history['stage2'] = {
        'params_searched': list(STAGE2_GRID.keys()),
        'best': best_s2,
        'n_combos': len(results_s2),
        'time_sec': time.time() - t2,
    }
    print(f'  Stage 2 最优: {best_s2} (耗时 {time.time()-t2:.1f}s)')
    
    # --- Stage 3: 正则化参数 ---
    print('\n--- Stage 3: 正则化参数优化 (reg_alpha, reg_lambda, gamma) ---')
    t3 = time.time()
    best_s3, results_s3 = grid_search_stage(
        X, y, base_params, STAGE3_GRID, target_name, n_splits)
    base_params.update(best_s3)
    search_history['stage3'] = {
        'params_searched': list(STAGE3_GRID.keys()),
        'best': best_s3,
        'n_combos': len(results_s3),
        'time_sec': time.time() - t3,
    }
    print(f'  Stage 3 最优: {best_s3} (耗时 {time.time()-t3:.1f}s)')
    
    # --- 最终验证 ---
    print(f'\n--- 最终参数验证 ---')
    final_metrics = cv_evaluate(X, y, base_params, n_splits=n_splits)
    print(f'  最优参数: {json.dumps({k:v for k,v in base_params.items() if k!="device"}, indent=2)}')
    print(f'  最终CV结果:')
    print(f'    MAE  = {final_metrics["mae_mean"]:.2f} ± {final_metrics["mae_std"]:.2f}')
    print(f'    RMSE = {final_metrics["rmse_mean"]:.2f} ± {final_metrics["rmse_std"]:.2f}')
    print(f'    R²   = {final_metrics["r2_mean"]:.4f} ± {final_metrics["r2_std"]:.4f}')
    
    search_history['final_metrics'] = final_metrics
    search_history['best_params'] = {k: v for k, v in base_params.items()}
    
    return base_params, search_history


def train_final_model(X, y, params, target_name, save_dir):
    """
    用全量数据训练最终模型 (带CV验证)
    
    Returns
    -------
    model : XGBRegressor
    cv_metrics : dict
    """
    valid_mask = ~np.isnan(y)
    X_v = X[valid_mask]
    y_v = y[valid_mask]
    
    print(f'\n[最终训练] {target_name}: {len(y_v)} 样本')
    
    # 先用CV获取最终指标和最优迭代数
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_iters = []
    maes, rmses, r2s = [], [], []
    all_preds = np.full(len(y_v), np.nan)
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_v), 1):
        X_tr, X_va = X_v[tr_idx], X_v[va_idx]
        y_tr, y_va = y_v[tr_idx], y_v[va_idx]
        
        model = xgb.XGBRegressor(
            **params,
            verbosity=0,
            early_stopping_rounds=50,
            random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        y_pred = model.predict(X_va)
        all_preds[va_idx] = y_pred
        best_iters.append(model.best_iteration)
        
        mae = mean_absolute_error(y_va, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va, y_pred))
        r2 = r2_score(y_va, y_pred)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        
        print(f'  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f} (iter={model.best_iteration})')
    
    cv_metrics = {
        'target': target_name,
        'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'best_iterations': best_iters,
        'avg_best_iteration': int(np.mean(best_iters)),
    }
    
    print(f'\n  CV汇总: MAE={cv_metrics["mae_mean"]:.2f}±{cv_metrics["mae_std"]:.2f}, '
          f'RMSE={cv_metrics["rmse_mean"]:.2f}±{cv_metrics["rmse_std"]:.2f}, '
          f'R²={cv_metrics["r2_mean"]:.4f}±{cv_metrics["r2_std"]:.4f}')
    
    # 用全量数据训练最终模型
    final_n_estimators = int(np.mean(best_iters) * 1.1)  # 略多于CV平均最优
    final_params = dict(params)
    final_params['n_estimators'] = final_n_estimators
    if 'early_stopping_rounds' in final_params:
        del final_params['early_stopping_rounds']
    
    print(f'\n  全样本训练: n_estimators={final_n_estimators}')
    final_model = xgb.XGBRegressor(**final_params, verbosity=0, random_state=42)
    final_model.fit(X_v, y_v)
    
    # 全样本预测（训练集拟合情况）
    y_pred_full = final_model.predict(X_v)
    train_mae = mean_absolute_error(y_v, y_pred_full)
    train_r2 = r2_score(y_v, y_pred_full)
    print(f'  全样本拟合: MAE={train_mae:.2f}, R²={train_r2:.4f}')
    
    # 保存模型
    model_path = os.path.join(save_dir, f'best_model_{target_name}.json')
    final_model.save_model(model_path)
    print(f'  模型已保存: {model_path}')
    
    # 保存特征重要性
    imp = final_model.feature_importances_
    
    # 保存CV预测
    cv_pred_df = pd.DataFrame({
        'y_true': y_v,
        'y_pred_cv': all_preds,
        'residual': y_v - all_preds,
    })
    cv_pred_df.to_csv(os.path.join(save_dir, f'cv_predictions_{target_name}.csv'),
                       index=False, encoding='utf-8-sig')
    
    return final_model, cv_metrics, imp


def generate_report(gap_metrics, los_metrics, gap_params, los_params,
                    gap_imp, los_imp, feature_names, save_dir,
                    baseline_gap=None, baseline_los=None):
    """生成完整分析报告"""
    
    report = []
    report.append('=' * 80)
    report.append('全样本训练 + 超参数优化 — 最终报告')
    report.append('=' * 80)
    report.append(f'日期: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    report.append(f'样本数: {gap_metrics.get("n_samples", "N/A")}')
    report.append('')
    
    # 参数对比
    report.append('\n' + '=' * 60)
    report.append('最优超参数')
    report.append('=' * 60)
    
    for name, params in [('Gap Days', gap_params), ('Next LOS', los_params)]:
        report.append(f'\n--- {name} ---')
        for k, v in sorted(params.items()):
            if k != 'device':
                report.append(f'  {k:25s} = {v}')
    
    # 结果对比
    report.append('\n\n' + '=' * 60)
    report.append('模型性能对比')
    report.append('=' * 60)
    
    report.append(f'\n{"目标":<20s} {"指标":<8s} {"基线(默认参数)":<20s} {"优化后":<20s} {"提升":<10s}')
    report.append('-' * 80)
    
    if baseline_gap:
        for metric in ['mae', 'rmse', 'r2']:
            base_val = baseline_gap.get(f'{metric}_mean', 0)
            opt_val = gap_metrics.get(f'{metric}_mean', 0)
            if metric == 'r2':
                improvement = opt_val - base_val
                imp_str = f'+{improvement:.4f}' if improvement > 0 else f'{improvement:.4f}'
            else:
                improvement = base_val - opt_val
                imp_str = f'-{improvement:.2f}' if improvement > 0 else f'+{abs(improvement):.2f}'
            report.append(f'{"gap_days":<20s} {metric.upper():<8s} '
                         f'{base_val:<20.4f} {opt_val:<20.4f} {imp_str:<10s}')
    
    if baseline_los:
        for metric in ['mae', 'rmse', 'r2']:
            base_val = baseline_los.get(f'{metric}_mean', 0)
            opt_val = los_metrics.get(f'{metric}_mean', 0)
            if metric == 'r2':
                improvement = opt_val - base_val
                imp_str = f'+{improvement:.4f}' if improvement > 0 else f'{improvement:.4f}'
            else:
                improvement = base_val - opt_val
                imp_str = f'-{improvement:.2f}' if improvement > 0 else f'+{abs(improvement):.2f}'
            report.append(f'{"next_los":<20s} {metric.upper():<8s} '
                         f'{base_val:<20.4f} {opt_val:<20.4f} {imp_str:<10s}')
    
    # 特征重要性Top 20
    report.append('\n\n' + '=' * 60)
    report.append('特征重要性 Top 20')
    report.append('=' * 60)
    
    for name, imp in [('Gap Days', gap_imp), ('Next LOS', los_imp)]:
        imp_total = imp.sum()
        imp_norm = imp / imp_total if imp_total > 0 else imp
        sorted_idx = np.argsort(imp_norm)[::-1]
        
        report.append(f'\n--- {name} ---')
        report.append(f'{"排名":<4s} {"特征":<45s} {"重要性":>10s} {"累积":>8s}')
        report.append('-' * 70)
        
        cumsum = 0
        for rank, idx in enumerate(sorted_idx[:20], 1):
            cumsum += imp_norm[idx]
            report.append(f'{rank:<4d} {feature_names[idx]:<45s} '
                         f'{imp_norm[idx]:>10.4f} {cumsum:>8.4f}')
    
    report_text = '\n'.join(report)
    
    report_path = os.path.join(save_dir, 'final_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Tee输出到日志
    log_path = os.path.join(RESULTS_DIR, 'training_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    
    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, s):
            for w in self.writers:
                try: w.write(s); w.flush()
                except: pass
        def flush(self):
            for w in self.writers:
                try: w.flush()
                except: pass
    
    sys.stdout = TeeWriter(original_stdout, log_file)
    
    t_start = time.time()
    
    print('=' * 80)
    print('全样本训练 + 超参数优化')
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)
    
    # 检测GPU
    try:
        _m = xgb.XGBRegressor(device='cuda', n_estimators=2, verbosity=0)
        _m.fit(np.random.randn(10, 3).astype(np.float32),
               np.random.randn(10).astype(np.float32))
        device = 'cuda'
        print('[GPU] XGBoost CUDA 加速已启用')
    except Exception:
        device = 'cpu'
        print('[CPU] XGBoost 使用CPU模式')
    
    # 加载数据
    print('\n[1] 加载数据...')
    X, y_gap, y_los, feature_names, text_feats = load_data()
    
    print(f'  结构化特征: {X.shape}')
    print(f'  Gap目标有效: {(~np.isnan(y_gap)).sum()}')
    print(f'  LOS目标有效: {(~np.isnan(y_los)).sum()}')
    
    # 合并文本统计特征
    if text_feats is not None:
        X_full = np.hstack([X, text_feats]).astype(np.float32)
        text_feat_names = ['txt_len', 'txt_lines', 'txt_digits', 'txt_chinese',
                          'txt_diag', 'txt_lab', 'txt_exam', 'txt_cost',
                          'txt_order', 'txt_equals', 'txt_commas']
        feature_names_full = feature_names + text_feat_names
        print(f'  合并文本特征后: {X_full.shape}')
    else:
        X_full = X
        feature_names_full = feature_names
    
    # 加载基线结果（用于对比）
    baseline_gap = None
    baseline_los = None
    baseline_path = os.path.join(OUTPUT_DIR, 'model_comparison', 'comparison_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baselines = json.load(f)
        for b in baselines:
            if 'XGBoost-GPU(全特征)' in b.get('model', '') and b['target'] == 'gap_days':
                baseline_gap = b
            if 'XGBoost-GPU(全特征)' in b.get('model', '') and b['target'] == 'next_los':
                baseline_los = b
    
    # ===== 超参数优化: Gap Days =====
    print('\n\n' + '#' * 80)
    print('# 目标 1: 住院间隔天数 (gap_days)')
    print('#' * 80)
    
    gap_params, gap_history = staged_hyperparameter_search(
        X_full, y_gap, 'gap_days', device=device, n_splits=5
    )
    
    # ===== 超参数优化: Next LOS =====
    print('\n\n' + '#' * 80)
    print('# 目标 2: 下次住院时长 (next_los)')
    print('#' * 80)
    
    los_params, los_history = staged_hyperparameter_search(
        X_full, y_los, 'next_los', device=device, n_splits=5
    )
    
    # ===== 最终模型训练 =====
    print('\n\n' + '#' * 80)
    print('# 最终模型训练 (全样本)')
    print('#' * 80)
    
    gap_model, gap_cv_metrics, gap_imp = train_final_model(
        X_full, y_gap, gap_params, 'gap_days', RESULTS_DIR
    )
    
    los_model, los_cv_metrics, los_imp = train_final_model(
        X_full, y_los, los_params, 'next_los', RESULTS_DIR
    )
    
    # ===== 保存搜索历史 =====
    search_results = {
        'gap_days': {
            'best_params': {k: v for k, v in gap_params.items() if k != 'device'},
            'cv_metrics': gap_cv_metrics,
            'search_history': gap_history,
        },
        'next_los': {
            'best_params': {k: v for k, v in los_params.items() if k != 'device'},
            'cv_metrics': los_cv_metrics,
            'search_history': los_history,
        },
        'device': device,
        'n_features': X_full.shape[1],
        'feature_names': feature_names_full,
    }
    
    with open(os.path.join(RESULTS_DIR, 'optimization_results.json'), 'w', encoding='utf-8') as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存特征重要性
    imp_df = pd.DataFrame({
        'feature': feature_names_full,
        'gap_importance': gap_imp / gap_imp.sum(),
        'los_importance': los_imp / los_imp.sum(),
    })
    imp_df['avg_importance'] = (imp_df['gap_importance'] + imp_df['los_importance']) / 2
    imp_df = imp_df.sort_values('avg_importance', ascending=False)
    imp_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_final.csv'),
                   index=False, encoding='utf-8-sig')
    
    # ===== 生成最终报告 =====
    gap_cv_metrics['n_samples'] = X_full.shape[0]
    generate_report(
        gap_cv_metrics, los_cv_metrics,
        gap_params, los_params,
        gap_imp, los_imp, feature_names_full,
        RESULTS_DIR,
        baseline_gap, baseline_los,
    )
    
    elapsed = time.time() - t_start
    print(f'\n\n{"=" * 80}')
    print(f'全部完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print(f'结果保存: {RESULTS_DIR}')
    print(f'{"=" * 80}')
    
    sys.stdout = original_stdout
    log_file.close()


if __name__ == '__main__':
    main()
