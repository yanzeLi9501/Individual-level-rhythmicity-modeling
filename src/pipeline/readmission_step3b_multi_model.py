"""
Step 3b: 多模型对比实验
========================
模型列表:
  1. XGBoost-GPU (全特征)
  2. XGBoost-GPU (筛选特征)
  3. GradientBoosting (CPU基线)
  4. RandomForest (CPU基线)
  5. Transformer文本编码器 + 结构化特征融合 (CPU)
  6. XGBoost + 文本统计特征 (混合)

评估指标: MAE, RMSE, R² — 5折交叉验证
输出: 对比表格 + 可视化

用法:
    python readmission_step3b_multi_model.py
"""

import os
import sys
import json
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')


# ============================================================
# 模型包装器
# ============================================================

class ModelWrapper:
    """统一模型接口"""
    
    def __init__(self, name, model_cls, params, use_text=False):
        self.name = name
        self.model_cls = model_cls
        self.params = params
        self.use_text = use_text  # 是否使用文本特征
    
    def create_model(self):
        return self.model_cls(**self.params)


class TransformerModelWrapper:
    """Transformer文本+结构化特征融合模型包装器"""
    
    def __init__(self, name='Transformer融合模型', max_text_length=256,
                 epochs=20, batch_size=32, lr=1e-3):
        self.name = name
        self.use_text = True
        self.max_text_length = max_text_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
    
    def train_and_predict(self, X_train, y_train, X_val, y_val,
                          texts_train=None, texts_val=None):
        """训练并预测"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        device = 'cpu'  # PyTorch只有CPU
        
        # 标准化结构化特征
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)
        
        struct_dim = X_tr.shape[1]
        
        # 文本特征：提取统计特征（长度、关键词计数等）
        text_dim = 0
        if texts_train is not None:
            text_feats_tr = self._extract_text_features(texts_train)
            text_feats_va = self._extract_text_features(texts_val)
            
            text_scaler = StandardScaler()
            text_feats_tr = text_scaler.fit_transform(text_feats_tr)
            text_feats_va = text_scaler.transform(text_feats_va)
            
            X_tr = np.hstack([X_tr, text_feats_tr])
            X_va = np.hstack([X_va, text_feats_va])
            text_dim = text_feats_tr.shape[1]
        
        total_dim = struct_dim + text_dim
        
        # 构建简单深度学习模型
        model = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss(delta=10.0)
        
        # 转换数据
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        
        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 训练
        best_loss = float('inf')
        best_state = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va_t)
                val_loss = criterion(val_pred, 
                                   torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # 加载最佳模型预测
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions = model(X_va_t).squeeze().numpy()
        
        return predictions
    
    def _extract_text_features(self, texts):
        """从文本中提取统计特征"""
        features = []
        for text in texts:
            if not isinstance(text, str):
                text = ''
            feat = {
                'text_len': len(text),
                'n_lines': text.count('\n'),
                'n_digits': sum(c.isdigit() for c in text),
                'n_chinese': sum('\u4e00' <= c <= '\u9fff' for c in text),
                'has_diag': int('诊断' in text),
                'has_lab': int('检验' in text),
                'has_exam': int('检查' in text),
                'has_cost': int('费用' in text),
                'has_order': int('医嘱' in text),
                'n_equal_signs': text.count('='),
                'n_commas': text.count(',') + text.count('，'),
            }
            features.append(feat)
        return pd.DataFrame(features).values.astype(np.float32)


# ============================================================
# 核心评估函数
# ============================================================

def evaluate_sklearn_model(model_wrapper, X, y, target_name,
                           n_splits=5, random_state=42,
                           texts=None, early_stop_xgb=True):
    """
    5折交叉验证评估单个sklearn/xgboost模型
    
    Returns
    -------
    results : dict
        包含MAE/RMSE/R²的均值和标准差
    fold_predictions : list of (y_true, y_pred) per fold
    """
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    if texts is not None:
        texts_valid = [texts[i] for i in range(len(texts)) if valid_mask[i]]
    else:
        texts_valid = None
    
    if len(y_valid) < 50:
        return None, None
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_predictions = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_valid), 1):
        X_tr, X_va = X_valid[tr_idx], X_valid[va_idx]
        y_tr, y_va = y_valid[tr_idx], y_valid[va_idx]
        
        if isinstance(model_wrapper, TransformerModelWrapper):
            texts_tr = [texts_valid[i] for i in tr_idx] if texts_valid else None
            texts_va = [texts_valid[i] for i in va_idx] if texts_valid else None
            y_pred = model_wrapper.train_and_predict(
                X_tr, y_tr, X_va, y_va, texts_tr, texts_va)
        else:
            model = model_wrapper.create_model()
            
            # XGBoost支持early stopping
            is_xgb = isinstance(model, xgb.XGBRegressor)
            if is_xgb and early_stop_xgb:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_va)
        
        mae = mean_absolute_error(y_va, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va, y_pred))
        r2 = r2_score(y_va, y_pred)
        
        fold_results.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})
        fold_predictions.append((y_va, y_pred))
    
    results_df = pd.DataFrame(fold_results)
    results = {
        'model': model_wrapper.name,
        'target': target_name,
        'mae_mean': results_df['mae'].mean(),
        'mae_std': results_df['mae'].std(),
        'rmse_mean': results_df['rmse'].mean(),
        'rmse_std': results_df['rmse'].std(),
        'r2_mean': results_df['r2'].mean(),
        'r2_std': results_df['r2'].std(),
    }
    
    return results, fold_predictions


def run_multi_model_comparison(output_dir, device='cuda', n_splits=5):
    """
    运行完整的多模型对比实验
    
    Parameters
    ----------
    output_dir : str
        readmission_output 目录路径
    device : str
        XGBoost设备 ('cuda' or 'cpu')
    n_splits : int
        交叉验证折数
    
    Returns
    -------
    comparison_df : pd.DataFrame
        所有模型所有目标的对比结果表
    """
    print('=' * 80)
    print('Step 3b: 多模型对比实验')
    print('=' * 80)
    
    t0 = time.time()
    
    # ===== 加载数据 =====
    X_all_df = pd.read_csv(os.path.join(output_dir, 'structured_features.csv'))
    targets = np.load(os.path.join(output_dir, 'targets.npz'))
    y_gap = targets['y_gap']
    y_los = targets['y_los']
    
    feature_names = list(X_all_df.columns)
    
    # 填充NaN
    X_all = X_all_df.copy()
    for col in X_all.columns:
        median_val = X_all[col].median()
        X_all[col] = X_all[col].fillna(median_val if pd.notna(median_val) else 0)
    X_all_np = X_all.values.astype(np.float32)
    
    # 标准化（用于非树模型）
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_np)
    
    # 加载LLM文本
    texts = None
    texts_path = os.path.join(output_dir, 'llm_texts.txt')
    if os.path.exists(texts_path):
        with open(texts_path, 'r', encoding='utf-8') as f:
            content = f.read()
        texts = []
        for block in content.split('===RECORD_')[1:]:
            lines = block.split('\n', 1)
            if len(lines) > 1:
                texts.append(lines[1].strip())
            else:
                texts.append('')
        if len(texts) != len(X_all_np):
            print(f'  [WARNING] 文本数({len(texts)}) != 样本数({len(X_all_np)}), 禁用文本特征')
            texts = None
    
    # 加载筛选后的特征（如果存在）
    sel_path = os.path.join(output_dir, 'feature_selection', 'selected_features.csv')
    X_selected = None
    if os.path.exists(sel_path):
        X_sel_df = pd.read_csv(sel_path)
        X_selected = X_sel_df.values.astype(np.float32)
        print(f'  已加载筛选特征: {X_selected.shape[1]} 列 (原始 {X_all_np.shape[1]} 列)')
    
    print(f'\n数据概况:')
    print(f'  全部特征: {X_all_np.shape} (samples × features)')
    if X_selected is not None:
        print(f'  筛选特征: {X_selected.shape}')
    print(f'  Gap目标有效: {(~np.isnan(y_gap)).sum()}')
    print(f'  LOS目标有效: {(~np.isnan(y_los)).sum()}')
    if texts:
        print(f'  文本样本: {len(texts)}')
    
    # ===== 定义模型集合 =====
    models = []
    
    # (1) XGBoost-GPU 全特征
    models.append(('all', ModelWrapper(
        name='XGBoost-GPU(全特征)',
        model_cls=xgb.XGBRegressor,
        params=dict(
            device=device, n_estimators=800, max_depth=6,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
            early_stopping_rounds=30, random_state=42,
        ),
    )))
    
    # (2) XGBoost-GPU 筛选特征
    if X_selected is not None:
        models.append(('selected', ModelWrapper(
            name='XGBoost-GPU(筛选特征)',
            model_cls=xgb.XGBRegressor,
            params=dict(
                device=device, n_estimators=800, max_depth=6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
                early_stopping_rounds=30, random_state=42,
            ),
        )))
    
    # (3) GradientBoosting (CPU基线)
    models.append(('all', ModelWrapper(
        name='GradientBoosting(CPU)',
        model_cls=GradientBoostingRegressor,
        params=dict(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    )))
    
    # (4) RandomForest (CPU基线)
    models.append(('all', ModelWrapper(
        name='RandomForest(CPU)',
        model_cls=RandomForestRegressor,
        params=dict(
            n_estimators=300, max_depth=10, n_jobs=-1,
            random_state=42,
        ),
    )))
    
    # (5) XGBoost + 文本统计特征(混合)
    if texts is not None:
        models.append(('text_hybrid', ModelWrapper(
            name='XGBoost-GPU(结构化+文本特征)',
            model_cls=xgb.XGBRegressor,
            params=dict(
                device=device, n_estimators=800, max_depth=7,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
                early_stopping_rounds=30, random_state=42,
            ),
        )))
    
    # (6) Deep Learning Transformer融合
    try:
        import torch
        models.append(('all_text', TransformerModelWrapper(
            name='深度学习融合模型(CPU)',
            max_text_length=256, epochs=20, batch_size=64, lr=1e-3,
        )))
    except ImportError:
        print('  [SKIP] PyTorch未安装，跳过深度学习模型')
    
    # ===== 构建文本统计特征矩阵（用于混合模型） =====
    X_text_hybrid = None
    if texts is not None:
        tw = TransformerModelWrapper()
        text_feats = tw._extract_text_features(texts)
        X_text_hybrid = np.hstack([X_all_np, text_feats]).astype(np.float32)
        print(f'  混合特征维度: {X_text_hybrid.shape[1]} (结构化{X_all_np.shape[1]} + 文本{text_feats.shape[1]})')
    
    # ===== 执行评估 =====
    all_results = []
    target_configs = [
        ('gap_days', y_gap, '住院间隔天数'),
        ('next_los', y_los, '下次住院时长'),
    ]
    
    for target_key, y_target, target_label in target_configs:
        print(f'\n{"="*60}')
        print(f'目标: {target_label} ({target_key})')
        print(f'{"="*60}')
        
        for feat_type, mw in models:
            # 选择特征矩阵
            if feat_type == 'selected':
                X_use = X_selected
            elif feat_type == 'text_hybrid':
                X_use = X_text_hybrid
            elif feat_type == 'all_text':
                X_use = X_all_scaled  # DL需要标准化
            else:
                X_use = X_all_np
            
            if X_use is None:
                continue
            
            t_texts = texts if isinstance(mw, TransformerModelWrapper) else None
            
            print(f'\n  >> {mw.name} (特征维度: {X_use.shape[1]})')
            t1 = time.time()
            
            results, fold_preds = evaluate_sklearn_model(
                mw, X_use, y_target, target_key,
                n_splits=n_splits, texts=t_texts,
            )
            
            elapsed = time.time() - t1
            
            if results:
                results['time_sec'] = elapsed
                all_results.append(results)
                print(f'     MAE: {results["mae_mean"]:.2f} ± {results["mae_std"]:.2f}')
                print(f'     RMSE: {results["rmse_mean"]:.2f} ± {results["rmse_std"]:.2f}')
                print(f'     R²: {results["r2_mean"]:.4f} ± {results["r2_std"]:.4f}')
                print(f'     耗时: {elapsed:.1f}s')
            else:
                print(f'     [SKIP] 有效样本不足')
    
    # ===== 生成对比表格 =====
    comparison_df = pd.DataFrame(all_results)
    
    # 打印对比表格
    print('\n\n' + '=' * 100)
    print('多模型对比结果汇总')
    print('=' * 100)
    
    for target_key, _, target_label in target_configs:
        target_df = comparison_df[comparison_df['target'] == target_key].copy()
        if target_df.empty:
            continue
        
        target_df = target_df.sort_values('mae_mean')
        
        print(f'\n--- {target_label} ({target_key}) ---')
        print(f'{"模型":<35s} {"MAE":>12s} {"RMSE":>12s} {"R²":>14s} {"耗时":>8s}')
        print('-' * 85)
        
        for _, row in target_df.iterrows():
            mae_str = f'{row["mae_mean"]:.2f}±{row["mae_std"]:.2f}'
            rmse_str = f'{row["rmse_mean"]:.2f}±{row["rmse_std"]:.2f}'
            r2_str = f'{row["r2_mean"]:.4f}±{row["r2_std"]:.4f}'
            time_str = f'{row.get("time_sec", 0):.1f}s'
            print(f'{row["model"]:<35s} {mae_str:>12s} {rmse_str:>12s} {r2_str:>14s} {time_str:>8s}')
    
    # ===== 保存结果 =====
    save_dir = os.path.join(output_dir, 'model_comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_df.to_csv(os.path.join(save_dir, 'comparison_results.csv'),
                          index=False, encoding='utf-8-sig')
    
    # 保存为格式化文本报告
    report_lines = []
    report_lines.append('多模型对比实验报告')
    report_lines.append('=' * 80)
    report_lines.append(f'日期: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    report_lines.append(f'样本数: {len(X_all_np)}')
    report_lines.append(f'全部特征数: {X_all_np.shape[1]}')
    if X_selected is not None:
        report_lines.append(f'筛选特征数: {X_selected.shape[1]}')
    report_lines.append(f'交叉验证折数: {n_splits}')
    report_lines.append(f'XGBoost设备: {device}')
    report_lines.append('')
    
    for target_key, _, target_label in target_configs:
        target_df = comparison_df[comparison_df['target'] == target_key].sort_values('mae_mean')
        if target_df.empty:
            continue
        
        report_lines.append(f'\n目标: {target_label} ({target_key})')
        report_lines.append('-' * 80)
        report_lines.append(f'{"排名":<4s} {"模型":<35s} {"MAE":>12s} {"RMSE":>12s} {"R²":>14s}')
        report_lines.append('-' * 80)
        
        for rank, (_, row) in enumerate(target_df.iterrows(), 1):
            mae_str = f'{row["mae_mean"]:.2f}±{row["mae_std"]:.2f}'
            rmse_str = f'{row["rmse_mean"]:.2f}±{row["rmse_std"]:.2f}'
            r2_str = f'{row["r2_mean"]:.4f}±{row["r2_std"]:.4f}'
            report_lines.append(f'{rank:<4d} {row["model"]:<35s} {mae_str:>12s} {rmse_str:>12s} {r2_str:>14s}')
    
    with open(os.path.join(save_dir, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 保存JSON
    with open(os.path.join(save_dir, 'comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    elapsed_total = time.time() - t0
    print(f'\n[Step3b] 完成 (总耗时 {elapsed_total:.1f}s)')
    print(f'  结果保存: {save_dir}')
    
    return comparison_df


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
    
    comparison_df = run_multi_model_comparison(
        OUTPUT_DIR, device=device, n_splits=5
    )
