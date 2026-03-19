"""
Step 3a: XGBoost GPU 特征工程 — 特征重要性筛选 & 降维
=====================================================
使用 XGBoost (GPU加速) 对74个结构化特征进行:
  1. 特征重要性排序 (基于gain)
  2. 自动特征筛选 (累积重要性阈值 / top-K)
  3. 输出降维后的特征矩阵

用法:
    python readmission_step3a_xgb_feature_select.py
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# 核心功能
# ============================================================

def xgb_feature_importance(X, y, target_name='target', device='cuda',
                           n_splits=5, random_state=42):
    """
    使用 XGBoost 5折交叉验证计算特征重要性 (gain)
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    target_name : str
    device : str, 'cuda' or 'cpu'
    
    Returns
    -------
    importance_df : pd.DataFrame
        特征重要性排名 (列: feature, importance_mean, importance_std, rank)
    cv_metrics : dict
        交叉验证指标
    """
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    if len(y_valid) < 100:
        print(f'  [WARNING] {target_name} 有效样本过少: {len(y_valid)}')
    
    print(f'  [{target_name}] 有效样本: {len(y_valid)}, 特征数: {X_valid.shape[1]}')
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_importances = []
    fold_metrics = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_valid), 1):
        X_tr, X_va = X_valid[tr_idx], X_valid[va_idx]
        y_tr, y_va = y_valid[tr_idx], y_valid[va_idx]
        
        model = xgb.XGBRegressor(
            device=device,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state + fold,
            verbosity=0,
            early_stopping_rounds=30,
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        
        # 预测
        y_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va, y_pred))
        r2 = r2_score(y_va, y_pred)
        fold_metrics.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})
        
        # 特征重要性 (gain)
        imp = model.feature_importances_
        all_importances.append(imp)
        
        print(f'    Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f} '
              f'(best_iter={model.best_iteration})')
    
    # 汇总特征重要性
    imp_matrix = np.array(all_importances)  # (n_splits, n_features)
    imp_mean = imp_matrix.mean(axis=0)
    imp_std = imp_matrix.std(axis=0)
    
    # 归一化
    imp_total = imp_mean.sum()
    if imp_total > 0:
        imp_mean_norm = imp_mean / imp_total
    else:
        imp_mean_norm = imp_mean
    
    importance_df = pd.DataFrame({
        'feature_idx': range(len(imp_mean)),
        'importance_mean': imp_mean_norm,
        'importance_std': imp_std / imp_total if imp_total > 0 else imp_std,
        'importance_raw': imp_mean,
    })
    importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['cumulative_importance'] = importance_df['importance_mean'].cumsum()
    
    # 汇总CV指标
    metrics_df = pd.DataFrame(fold_metrics)
    cv_metrics = {
        'target': target_name,
        'mae_mean': metrics_df['mae'].mean(),
        'mae_std': metrics_df['mae'].std(),
        'rmse_mean': metrics_df['rmse'].mean(),
        'rmse_std': metrics_df['rmse'].std(),
        'r2_mean': metrics_df['r2'].mean(),
        'r2_std': metrics_df['r2'].std(),
    }
    print(f'  [{target_name}] CV汇总: MAE={cv_metrics["mae_mean"]:.2f}±{cv_metrics["mae_std"]:.2f}, '
          f'R²={cv_metrics["r2_mean"]:.4f}±{cv_metrics["r2_std"]:.4f}')
    
    return importance_df, cv_metrics


def select_features(importance_gap, importance_los, feature_names,
                    cumulative_threshold=0.95, min_features=15):
    """
    基于两个目标的特征重要性联合筛选
    
    策略:
      1. 对每个目标，按累积重要性阈值筛选特征
      2. 取两个目标的并集
      3. 确保至少保留 min_features 个特征
    
    Returns
    -------
    selected_indices : list[int]
        被选中特征的原始索引
    selection_report : dict
        筛选详情
    """
    def get_top_features(imp_df, threshold):
        """按累积重要性阈值选取特征"""
        mask = imp_df['cumulative_importance'] <= threshold
        # 至少保留达到阈值时的那一个特征
        n_selected = max(mask.sum(), 1)
        # 额外加一个（超过阈值的第一个）
        n_selected = min(n_selected + 1, len(imp_df))
        return set(imp_df.iloc[:n_selected]['feature_idx'].tolist())
    
    gap_selected = get_top_features(importance_gap, cumulative_threshold)
    los_selected = get_top_features(importance_los, cumulative_threshold)
    
    # 并集
    selected = sorted(gap_selected | los_selected)
    
    # 确保最少特征数
    if len(selected) < min_features:
        # 补充两个目标中排名靠前但未被选中的特征
        all_ranked = []
        for idx in importance_gap['feature_idx'].tolist():
            if idx not in selected:
                all_ranked.append(idx)
        for idx in importance_los['feature_idx'].tolist():
            if idx not in selected and idx not in all_ranked:
                all_ranked.append(idx)
        
        while len(selected) < min_features and all_ranked:
            selected.append(all_ranked.pop(0))
        selected = sorted(selected)
    
    # 生成报告
    selected_names = [feature_names[i] for i in selected]
    
    report = {
        'total_features': len(feature_names),
        'selected_features': len(selected),
        'reduction_ratio': 1 - len(selected) / len(feature_names),
        'gap_only_count': len(gap_selected - los_selected),
        'los_only_count': len(los_selected - gap_selected),
        'shared_count': len(gap_selected & los_selected),
        'cumulative_threshold': cumulative_threshold,
        'selected_indices': selected,
        'selected_names': selected_names,
    }
    
    return selected, report


def run_feature_selection(output_dir, device='cuda',
                          cumulative_threshold=0.95, min_features=15):
    """
    执行完整的XGBoost特征筛选流程
    
    Parameters
    ----------
    output_dir : str
        readmission_output 目录路径
    device : str
        'cuda' or 'cpu'
    
    Returns
    -------
    selected_indices : list[int]
        被选中特征的索引
    X_selected : np.ndarray
        降维后的特征矩阵
    feature_names_selected : list[str]
        降维后的特征名
    """
    print('=' * 80)
    print('Step 3a: XGBoost GPU 特征工程')
    print(f'  设备: {device}')
    print('=' * 80)
    
    t0 = time.time()
    
    # 加载数据
    X_df = pd.read_csv(os.path.join(output_dir, 'structured_features.csv'))
    targets = np.load(os.path.join(output_dir, 'targets.npz'))
    y_gap = targets['y_gap']
    y_los = targets['y_los']
    
    feature_names = list(X_df.columns)
    
    # 填充NaN为中位数
    X_filled = X_df.copy()
    for col in X_filled.columns:
        median_val = X_filled[col].median()
        X_filled[col] = X_filled[col].fillna(median_val if pd.notna(median_val) else 0)
    
    X = X_filled.values.astype(np.float32)
    
    print(f'\n数据维度: {X.shape[0]} samples × {X.shape[1]} features')
    print(f'目标1 (gap_days): {(~np.isnan(y_gap)).sum()} 有效')
    print(f'目标2 (next_los): {(~np.isnan(y_los)).sum()} 有效')
    
    # --- 目标1: 住院间隔 ---
    print('\n--- 目标1: 住院间隔天数 (gap_days) ---')
    imp_gap, metrics_gap = xgb_feature_importance(
        X, y_gap, target_name='gap_days', device=device)
    
    # --- 目标2: 下次住院时长 ---
    print('\n--- 目标2: 下次住院时长 (next_los) ---')
    imp_los, metrics_los = xgb_feature_importance(
        X, y_los, target_name='next_los', device=device)
    
    # 添加特征名
    imp_gap['feature_name'] = imp_gap['feature_idx'].apply(lambda i: feature_names[i])
    imp_los['feature_name'] = imp_los['feature_idx'].apply(lambda i: feature_names[i])
    
    # --- 联合特征筛选 ---
    print(f'\n--- 联合特征筛选 (阈值={cumulative_threshold}) ---')
    selected_indices, report = select_features(
        imp_gap, imp_los, feature_names,
        cumulative_threshold=cumulative_threshold,
        min_features=min_features,
    )
    
    print(f'  原始特征数: {report["total_features"]}')
    print(f'  筛选后特征数: {report["selected_features"]}')
    print(f'  降维比例: {report["reduction_ratio"]:.1%}')
    print(f'  仅gap相关: {report["gap_only_count"]}')
    print(f'  仅los相关: {report["los_only_count"]}')
    print(f'  两个目标共享: {report["shared_count"]}')
    
    print(f'\n  被选特征:')
    for i, (idx, name) in enumerate(zip(selected_indices, report['selected_names'])):
        gap_rank = imp_gap[imp_gap['feature_idx'] == idx]['rank'].values
        los_rank = imp_los[imp_los['feature_idx'] == idx]['rank'].values
        gr = gap_rank[0] if len(gap_rank) > 0 else '?'
        lr = los_rank[0] if len(los_rank) > 0 else '?'
        print(f'    {i+1:3d}. {name:40s}  (gap排名={gr}, los排名={lr})')
    
    # 降维
    X_selected = X[:, selected_indices]
    feature_names_selected = report['selected_names']
    
    # --- 保存结果 ---
    save_dir = os.path.join(output_dir, 'feature_selection')
    os.makedirs(save_dir, exist_ok=True)
    
    # 特征重要性表
    imp_gap.to_csv(os.path.join(save_dir, 'importance_gap.csv'),
                    index=False, encoding='utf-8-sig')
    imp_los.to_csv(os.path.join(save_dir, 'importance_los.csv'),
                    index=False, encoding='utf-8-sig')
    
    # 筛选结果
    with open(os.path.join(save_dir, 'selection_report.json'), 'w', encoding='utf-8') as f:
        json.dump({**report, **metrics_gap, **metrics_los}, f,
                  ensure_ascii=False, indent=2, default=str)
    
    # 降维后的特征矩阵
    X_sel_df = pd.DataFrame(X_selected, columns=feature_names_selected)
    X_sel_df.to_csv(os.path.join(save_dir, 'selected_features.csv'),
                     index=False, encoding='utf-8-sig')
    
    elapsed = time.time() - t0
    print(f'\n[Step3a] 完成 (耗时 {elapsed:.1f}s)')
    print(f'  结果保存: {save_dir}')
    
    return selected_indices, X_selected, feature_names_selected


# ============================================================
# 独立运行
# ============================================================

if __name__ == '__main__':
    OUTPUT_DIR = r'D:\LDH_cancer\files\healthline\readmission_output'
    
    # 检测GPU
    try:
        test_model = xgb.XGBRegressor(device='cuda', n_estimators=2, verbosity=0)
        _X = np.random.randn(10, 3).astype(np.float32)
        _y = np.random.randn(10).astype(np.float32)
        test_model.fit(_X, _y)
        device = 'cuda'
        print('[GPU] XGBoost CUDA 可用')
    except Exception:
        device = 'cpu'
        print('[CPU] XGBoost 使用CPU')
    
    selected_indices, X_selected, feature_names_selected = run_feature_selection(
        OUTPUT_DIR, device=device, cumulative_threshold=0.95, min_features=15
    )
