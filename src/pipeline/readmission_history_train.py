"""
基于患者完整历史住院序列的预测模型
=======================================
核心思路: 利用患者每一次历史住院记录（间隔天数、住院时长、化验变化、费用趋势等）
         来预测下一次住院的相关特征。

特征分类:
  A. 原始单次住院特征 (人口学、检验、费用等) — 约74个
  B. 历史序列特征 (lag/rolling/trend) — 约45个新增
  C. 文本统计特征 — 11个

目标:
  - gap_days: R² > 0.85, MAE < 24
  - next_los: R² > 0.5, MAE < 2

用法:
  python readmission_history_train.py
"""

import os, sys, io, json, time, warnings, itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = r'D:\LDH_cancer\files\healthline\readmission_output'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'history_model')
CITY_DB_PATH = (r'D:\肺癌npj\评估肺癌背景患者不同放化疗治疗策略的效果2-20250822-1317'
                r'\评估肺癌背景患者不同放化疗治疗策略的效果2-20250822-1317'
                r'\研究组1\io_pipeline_v2\io_v2\city_database.xlsx')


# ============================================================
# 1. 加载数据
# ============================================================

def load_train_data():
    """加载已有的训练数据"""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_data.csv'), low_memory=False)
    df['入院日期'] = pd.to_datetime(df['入院日期'], errors='coerce')
    df['出院日期'] = pd.to_datetime(df['出院日期'], errors='coerce')
    df['实际住院天数'] = pd.to_numeric(df['实际住院天数'], errors='coerce')
    print(f'  原始训练数据: {df.shape}')
    return df


# ============================================================
# 2. 历史序列特征工程 (核心创新)
# ============================================================

def build_history_features(df):
    """
    对每位患者的每次住院记录，基于其之前所有历史住院构建特征。
    
    新增特征类别:
      A. 入院间隔Lag特征 (incoming_gap, prev_gap_1/2/3, 统计量, 趋势)
      B. 住院时长Lag特征 (prev_los_1/2/3, 统计量, 趋势)
      C. 费用变化特征
      D. 化验差值特征
      E. 时间模式特征
    """
    print('[特征工程] 构建患者历史序列特征...')
    
    df = df.sort_values(['病案号', 'visit_order']).copy()
    n = len(df)
    
    # ---------- 预计算: 每个患者组的索引和日期 ----------
    # 创建空列
    new_cols = {}
    
    # A) 入院间隔 (incoming gap) 特征
    # incoming_gap = 从上次出院到本次入院的天数
    new_cols['incoming_gap'] = np.full(n, np.nan)
    new_cols['log_incoming_gap'] = np.full(n, np.nan)
    new_cols['prev_gap_2'] = np.full(n, np.nan)
    new_cols['prev_gap_3'] = np.full(n, np.nan)
    new_cols['gap_mean_prev'] = np.full(n, np.nan)
    new_cols['gap_std_prev'] = np.full(n, np.nan)
    new_cols['gap_median_prev'] = np.full(n, np.nan)
    new_cols['gap_min_prev'] = np.full(n, np.nan)
    new_cols['gap_max_prev'] = np.full(n, np.nan)
    new_cols['gap_cv_prev'] = np.full(n, np.nan)
    new_cols['gap_trend'] = np.full(n, np.nan)
    new_cols['gap_ema'] = np.full(n, np.nan)
    new_cols['gap_ratio_last_mean'] = np.full(n, np.nan)
    new_cols['gap_acceleration'] = np.full(n, np.nan)
    
    # B) 住院时长Lag特征
    new_cols['prev_los_1'] = np.full(n, np.nan)
    new_cols['prev_los_2'] = np.full(n, np.nan)
    new_cols['prev_los_3'] = np.full(n, np.nan)
    new_cols['los_mean_prev'] = np.full(n, np.nan)
    new_cols['los_std_prev'] = np.full(n, np.nan)
    new_cols['los_median_prev'] = np.full(n, np.nan)
    new_cols['los_min_prev'] = np.full(n, np.nan)
    new_cols['los_max_prev'] = np.full(n, np.nan)
    new_cols['los_trend'] = np.full(n, np.nan)
    new_cols['los_ema'] = np.full(n, np.nan)
    new_cols['los_ratio_curr_mean'] = np.full(n, np.nan)
    
    # C) 费用变化
    new_cols['prev_total_cost'] = np.full(n, np.nan)
    new_cols['cost_change'] = np.full(n, np.nan)
    new_cols['cost_change_ratio'] = np.full(n, np.nan)
    
    # D) 化验差值
    key_labs = ['lab_WBC', 'lab_HGB', 'lab_ALB', 'lab_CREA', 'lab_LDH',
                'lab_CRP', 'lab_BUN', 'lab_ALT', 'lab_AST', 'lab_PLT']
    for lab in key_labs:
        new_cols[f'delta_{lab}'] = np.full(n, np.nan)
    
    # E) 时间模式
    new_cols['days_since_first'] = np.full(n, np.nan)
    new_cols['admission_frequency'] = np.full(n, np.nan)
    new_cols['recent_visits_90'] = np.full(n, np.nan)
    new_cols['recent_visits_180'] = np.full(n, np.nan)
    new_cols['recent_visits_365'] = np.full(n, np.nan)
    new_cols['same_department'] = np.full(n, np.nan)
    
    # 预先解析 费用数据
    cost_cols = [c for c in df.columns if c.startswith('费用_')]
    df['_total_cost'] = 0.0
    for c in cost_cols:
        df['_total_cost'] += pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 预先解析 lab 数据
    for lab in key_labs:
        if lab in df.columns:
            df[lab] = pd.to_numeric(df[lab], errors='coerce')
    
    # ---------- 逐患者组构建历史特征 ----------
    idx_array = df.index.values
    pid_col = df['病案号'].values
    admit_dates = df['入院日期'].values
    discharge_dates = df['出院日期'].values
    los_vals = df['实际住院天数'].values
    total_costs = df['_total_cost'].values
    dept_vals = df['入院科别'].values if '入院科别' in df.columns else None
    
    lab_arrays = {}
    for lab in key_labs:
        if lab in df.columns:
            lab_arrays[lab] = df[lab].values
    
    patients = df.groupby('病案号', sort=False)
    count = 0
    
    for pid, grp in patients:
        idxs = grp.index.tolist()
        k = len(idxs)
        if k < 2:
            continue
        
        # 获取该患者所有住院数据
        admits = [df.at[i, '入院日期'] for i in idxs]
        discharges = [df.at[i, '出院日期'] for i in idxs]
        los_list = [df.at[i, '实际住院天数'] for i in idxs]
        cost_list = [total_costs[df.index.get_loc(i)] for i in idxs]
        
        # 计算所有incoming gaps
        gaps = []
        for j in range(1, k):
            if pd.notna(discharges[j-1]) and pd.notna(admits[j]):
                g = (admits[j] - discharges[j-1]).days
                gaps.append(g if g >= 0 else np.nan)
            else:
                gaps.append(np.nan)
        
        for j in range(k):
            row_idx = idxs[j]
            loc = df.index.get_loc(row_idx)
            
            # --- A) Gap features ---
            if j >= 1 and j-1 < len(gaps):
                g = gaps[j-1]
                new_cols['incoming_gap'][loc] = g
                if pd.notna(g) and g >= 0:
                    new_cols['log_incoming_gap'][loc] = np.log1p(g)
            
            if j >= 2 and j-2 < len(gaps):
                new_cols['prev_gap_2'][loc] = gaps[j-2]
            if j >= 3 and j-3 < len(gaps):
                new_cols['prev_gap_3'][loc] = gaps[j-3]
            
            # 所有历史gaps统计
            hist_gaps = [g for g in gaps[:j] if pd.notna(g)]
            if len(hist_gaps) >= 1:
                arr = np.array(hist_gaps)
                new_cols['gap_mean_prev'][loc] = np.mean(arr)
                new_cols['gap_median_prev'][loc] = np.median(arr)
                new_cols['gap_min_prev'][loc] = np.min(arr)
                new_cols['gap_max_prev'][loc] = np.max(arr)
                
                # EMA (alpha=0.4, more weight on recent)
                ema = arr[0]
                for v in arr[1:]:
                    ema = 0.4 * v + 0.6 * ema
                new_cols['gap_ema'][loc] = ema
                
                if len(hist_gaps) >= 2:
                    new_cols['gap_std_prev'][loc] = np.std(arr)
                    mean_g = np.mean(arr)
                    if mean_g > 0:
                        new_cols['gap_cv_prev'][loc] = np.std(arr) / mean_g
                    
                    # trend: slope of linear fit
                    x = np.arange(len(arr))
                    slope = np.polyfit(x, arr, 1)[0]
                    new_cols['gap_trend'][loc] = slope
                    
                    # acceleration
                    if len(arr) >= 2:
                        new_cols['gap_acceleration'][loc] = arr[-1] - arr[-2]
                
                # ratio
                if j >= 1 and pd.notna(gaps[j-1]):
                    mean_g = np.mean(arr)
                    if mean_g > 0:
                        new_cols['gap_ratio_last_mean'][loc] = gaps[j-1] / mean_g
            
            # --- B) LOS lag features ---
            if j >= 1 and pd.notna(los_list[j-1]):
                new_cols['prev_los_1'][loc] = los_list[j-1]
            if j >= 2 and pd.notna(los_list[j-2]):
                new_cols['prev_los_2'][loc] = los_list[j-2]
            if j >= 3 and pd.notna(los_list[j-3]):
                new_cols['prev_los_3'][loc] = los_list[j-3]
            
            hist_los = [l for l in los_list[:j] if pd.notna(l)]
            if len(hist_los) >= 1:
                arr_l = np.array(hist_los)
                new_cols['los_mean_prev'][loc] = np.mean(arr_l)
                new_cols['los_median_prev'][loc] = np.median(arr_l)
                new_cols['los_min_prev'][loc] = np.min(arr_l)
                new_cols['los_max_prev'][loc] = np.max(arr_l)
                
                ema_l = arr_l[0]
                for v in arr_l[1:]:
                    ema_l = 0.4 * v + 0.6 * ema_l
                new_cols['los_ema'][loc] = ema_l
                
                curr_los = los_list[j]
                if pd.notna(curr_los) and np.mean(arr_l) > 0:
                    new_cols['los_ratio_curr_mean'][loc] = curr_los / np.mean(arr_l)
                
                if len(hist_los) >= 2:
                    new_cols['los_std_prev'][loc] = np.std(arr_l)
                    x = np.arange(len(arr_l))
                    slope = np.polyfit(x, arr_l, 1)[0]
                    new_cols['los_trend'][loc] = slope
            
            # --- C) Cost change ---
            if j >= 1:
                new_cols['prev_total_cost'][loc] = cost_list[j-1]
                if pd.notna(cost_list[j]) and pd.notna(cost_list[j-1]):
                    new_cols['cost_change'][loc] = cost_list[j] - cost_list[j-1]
                    if cost_list[j-1] > 0:
                        new_cols['cost_change_ratio'][loc] = cost_list[j] / cost_list[j-1]
            
            # --- D) Lab deltas ---
            if j >= 1:
                for lab in key_labs:
                    if lab in lab_arrays:
                        prev_idx = idxs[j-1]
                        prev_loc = df.index.get_loc(prev_idx)
                        curr_val = lab_arrays[lab][loc]
                        prev_val = lab_arrays[lab][prev_loc]
                        if pd.notna(curr_val) and pd.notna(prev_val):
                            new_cols[f'delta_{lab}'][loc] = curr_val - prev_val
            
            # --- E) Temporal pattern ---
            if j >= 1 and pd.notna(admits[0]) and pd.notna(admits[j]):
                days_span = (admits[j] - admits[0]).days
                new_cols['days_since_first'][loc] = days_span
                if days_span > 0:
                    new_cols['admission_frequency'][loc] = (j + 1) / days_span * 365  # visits/year
            
            if pd.notna(admits[j]):
                curr_admit = admits[j]
                for window, col in [(90, 'recent_visits_90'), 
                                     (180, 'recent_visits_180'),
                                     (365, 'recent_visits_365')]:
                    cnt = 0
                    for prev_j in range(j):
                        if pd.notna(admits[prev_j]):
                            diff = (curr_admit - admits[prev_j]).days
                            if 0 < diff <= window:
                                cnt += 1
                    new_cols[col][loc] = cnt
            
            # Same department
            if j >= 1 and dept_vals is not None:
                prev_dept = dept_vals[df.index.get_loc(idxs[j-1])]
                curr_dept = dept_vals[loc]
                new_cols['same_department'][loc] = int(
                    pd.notna(prev_dept) and pd.notna(curr_dept) and prev_dept == curr_dept
                )
        
        count += 1
        if count % 1000 == 0:
            print(f'    已处理 {count} 位患者...')
    
    # 添加到DataFrame
    for col_name, values in new_cols.items():
        df[col_name] = values
    
    df.drop('_total_cost', axis=1, inplace=True)
    
    total_new = len(new_cols)
    print(f'  新增 {total_new} 个历史特征')
    print(f'  有历史数据的记录数 (visit_order>=2): {(~np.isnan(new_cols["incoming_gap"])).sum()}')
    
    return df


# ============================================================
# 3. 构建完整特征矩阵
# ============================================================

def build_full_feature_matrix(df):
    """
    合并原始特征 + 历史序列特征 → 完整特征矩阵
    """
    print('\n[特征矩阵] 构建完整特征矩阵...')
    
    features = pd.DataFrame(index=df.index)
    
    # --- 原始特征 (同step2) ---
    features['age'] = pd.to_numeric(df['年龄'], errors='coerce')
    features['sex'] = (df['性别'].str.contains('男', na=False)).astype(int)
    marriage_map = {'已婚': 1, '未婚': 0, '离异': 2, '丧偶': 3}
    features['marriage'] = df['婚姻'].apply(
        lambda x: next((v for k, v in marriage_map.items()
                        if isinstance(x, str) and k in x), -1))
    features['pay_type'] = LabelEncoder().fit_transform(
        df['医疗支付方式'].fillna('未知').astype(str))
    features['admit_route'] = LabelEncoder().fit_transform(
        df['入院途径'].fillna('未知').astype(str))
    features['los_days'] = pd.to_numeric(df['实际住院天数'], errors='coerce')
    features['visit_order'] = df['visit_order']
    features['total_visits'] = df['total_visits']
    
    if '入院科别' in df.columns:
        dept_counts = df['入院科别'].value_counts()
        top_depts = dept_counts.head(20).index.tolist()
        dept_mapped = df['入院科别'].apply(lambda x: x if x in top_depts else '其他')
        features['department'] = LabelEncoder().fit_transform(dept_mapped.fillna('未知'))
    
    lab_cols = [c for c in df.columns if c.startswith('lab_')]
    for col in lab_cols:
        features[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'lab_NEU' in df.columns and 'lab_LYM' in df.columns:
        neu = pd.to_numeric(df['lab_NEU'], errors='coerce')
        lym = pd.to_numeric(df['lab_LYM'], errors='coerce')
        features['NLR'] = neu / lym.replace(0, np.nan)
        features['PLR'] = pd.to_numeric(df.get('lab_PLT', pd.Series(dtype=float)),
                                         errors='coerce') / lym.replace(0, np.nan)
    
    features['n_orders'] = pd.to_numeric(df['医嘱数量'], errors='coerce')
    features['n_exams'] = pd.to_numeric(df['检查数量'], errors='coerce')
    features['n_labs'] = pd.to_numeric(df['检验项目数'], errors='coerce')
    
    cost_cols_feat = [c for c in df.columns if c.startswith('费用_') and c != '费用_姓名']
    for col in cost_cols_feat:
        features[col] = pd.to_numeric(df[col], errors='coerce')
    if cost_cols_feat:
        features['total_cost'] = features[cost_cols_feat].sum(axis=1)
    
    features['diag_text_len'] = df['诊断文本'].fillna('').str.len()
    features['n_diagnoses'] = df['诊断文本'].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0)
    
    features['admit_month'] = df['入院日期'].dt.month
    features['admit_dayofweek'] = df['入院日期'].dt.dayofweek
    features['admit_year'] = df['入院日期'].dt.year
    
    # --- 新增: 历史序列特征 ---
    history_cols = [
        # Gap lag
        'incoming_gap', 'log_incoming_gap', 'prev_gap_2', 'prev_gap_3',
        'gap_mean_prev', 'gap_std_prev', 'gap_median_prev',
        'gap_min_prev', 'gap_max_prev', 'gap_cv_prev',
        'gap_trend', 'gap_ema', 'gap_ratio_last_mean', 'gap_acceleration',
        # LOS lag
        'prev_los_1', 'prev_los_2', 'prev_los_3',
        'los_mean_prev', 'los_std_prev', 'los_median_prev',
        'los_min_prev', 'los_max_prev',
        'los_trend', 'los_ema', 'los_ratio_curr_mean',
        # Cost change
        'prev_total_cost', 'cost_change', 'cost_change_ratio',
        # Lab deltas
        'delta_lab_WBC', 'delta_lab_HGB', 'delta_lab_ALB', 'delta_lab_CREA',
        'delta_lab_LDH', 'delta_lab_CRP', 'delta_lab_BUN', 'delta_lab_ALT',
        'delta_lab_AST', 'delta_lab_PLT',
        # Time pattern
        'days_since_first', 'admission_frequency',
        'recent_visits_90', 'recent_visits_180', 'recent_visits_365',
        'same_department',
    ]
    
    for col in history_cols:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- 社会经济特征 (从已有的structured_features.csv中获取) ---
    try:
        existing = pd.read_csv(os.path.join(OUTPUT_DIR, 'structured_features.csv'))
        socio_cols = [c for c in existing.columns if c.startswith('socio_')]
        if socio_cols and len(existing) == len(features):
            for col in socio_cols:
                features[col] = existing[col].values
            print(f'  合并 {len(socio_cols)} 个社会经济特征')
    except Exception:
        pass
    
    # --- 衍生交互特征 ---
    # gap * LOS相关
    features['gap_los_interaction'] = features.get('incoming_gap', 0) * features.get('los_days', 0)
    features['log_los_days'] = np.log1p(features['los_days'].fillna(0))
    
    # 填充NaN (XGBoost可以处理，但保险起见对非数值列填充)
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = pd.to_numeric(features[col], errors='coerce')
    
    feature_names = list(features.columns)
    print(f'  总特征数: {len(feature_names)}')
    
    return features, feature_names


# ============================================================
# 4. XGBoost训练与评估
# ============================================================

def cv_evaluate(X, y, params, n_splits=5, log_target=False, random_state=42):
    """5折CV评估"""
    valid_mask = ~np.isnan(y)
    X_v = X[valid_mask]
    y_v = y[valid_mask]
    
    if log_target:
        y_train_target = np.log1p(y_v)
    else:
        y_train_target = y_v
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes, rmses, r2s = [], [], []
    
    for tr_idx, va_idx in kf.split(X_v):
        X_tr, X_va = X_v[tr_idx], X_v[va_idx]
        y_tr = y_train_target[tr_idx]
        y_va_true = y_v[va_idx]  # 原始空间的真实值
        
        model = xgb.XGBRegressor(
            **params, verbosity=0, early_stopping_rounds=50, random_state=random_state)
        
        if log_target:
            y_va_train = y_train_target[va_idx]
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va_train)], verbose=False)
            y_pred_log = model.predict(X_va)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)  # 确保非负
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va_true)], verbose=False)
            y_pred = model.predict(X_va)
        
        maes.append(mean_absolute_error(y_va_true, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_va_true, y_pred)))
        r2s.append(r2_score(y_va_true, y_pred))
    
    return {
        'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
    }


def staged_search(X, y, target_name, device='cuda', log_target=False):
    """3阶段超参数搜索"""
    
    base_params = {
        'device': device,
        'n_estimators': 1200,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0,
    }
    
    lt_str = '(log-target)' if log_target else '(raw-target)'
    
    # Stage 1: 核心参数
    print(f'\n  Stage1: 核心参数 {lt_str}')
    s1_grid = {
        'n_estimators': [800, 1200, 2000, 3000],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
    }
    best_s1, _, _ = _grid_search(X, y, base_params, s1_grid, target_name, log_target)
    base_params.update(best_s1)
    print(f'    最优: {best_s1}')
    
    # Stage 2: 采样参数
    print(f'  Stage2: 采样参数')
    s2_grid = {
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5, 10],
    }
    best_s2, _, _ = _grid_search(X, y, base_params, s2_grid, target_name, log_target)
    base_params.update(best_s2)
    print(f'    最优: {best_s2}')
    
    # Stage 3: 正则化
    print(f'  Stage3: 正则化参数')
    s3_grid = {
        'reg_alpha': [0, 0.05, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
        'gamma': [0, 0.1, 0.3],
    }
    best_s3, _, _ = _grid_search(X, y, base_params, s3_grid, target_name, log_target)
    base_params.update(best_s3)
    print(f'    最优: {best_s3}')
    
    # 最终验证
    final = cv_evaluate(X, y, base_params, log_target=log_target)
    print(f'\n  最终结果 {lt_str}: MAE={final["mae_mean"]:.2f}±{final["mae_std"]:.2f}, '
          f'R²={final["r2_mean"]:.4f}±{final["r2_std"]:.4f}')
    
    return base_params, final


def _grid_search(X, y, base_params, grid, target_name, log_target):
    """穷举搜索"""
    param_names = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    
    best_score = float('inf')
    best_params = {}
    best_metrics = None
    
    for i, combo in enumerate(combos, 1):
        params = dict(base_params)
        for k, v in zip(param_names, combo):
            params[k] = v
        
        metrics = cv_evaluate(X, y, params, log_target=log_target)
        score = metrics['rmse_mean']  # optimize RMSE
        
        if score < best_score:
            best_score = score
            best_params = {k: v for k, v in zip(param_names, combo)}
            best_metrics = metrics
            
            combo_str = ', '.join(f'{k}={v}' for k, v in zip(param_names, combo))
            print(f'    [{i:3d}/{len(combos)}] {combo_str}  '
                  f'MAE={metrics["mae_mean"]:.2f} R²={metrics["r2_mean"]:.4f} ***')
    
    return best_params, best_metrics, None


def train_final_model(X, y, params, target_name, save_dir, 
                      feature_names=None, log_target=False):
    """全量训练 + CV评估"""
    valid_mask = ~np.isnan(y)
    X_v = X[valid_mask]
    y_v = y[valid_mask]
    
    if log_target:
        y_target = np.log1p(y_v)
    else:
        y_target = y_v
    
    print(f'\n[最终训练] {target_name}: {len(y_v)} 样本 (log={log_target})')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_iters = []
    maes, rmses, r2s = [], [], []
    all_preds = np.full(len(y_v), np.nan)
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_v), 1):
        X_tr, X_va = X_v[tr_idx], X_v[va_idx]
        y_tr = y_target[tr_idx]
        
        model = xgb.XGBRegressor(
            **params, verbosity=0, early_stopping_rounds=50, random_state=42)
        
        if log_target:
            y_va_t = y_target[va_idx]
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va_t)], verbose=False)
            y_pred_log = model.predict(X_va)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_v[va_idx])], verbose=False)
            y_pred = model.predict(X_va)
        
        all_preds[va_idx] = y_pred
        best_iters.append(model.best_iteration)
        
        y_va_true = y_v[va_idx]
        mae = mean_absolute_error(y_va_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_va_true, y_pred))
        r2 = r2_score(y_va_true, y_pred)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        
        print(f'  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f} (iter={model.best_iteration})')
    
    cv_metrics = {
        'target': target_name,
        'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
        'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'avg_best_iteration': int(np.mean(best_iters)),
        'log_target': log_target,
    }
    
    print(f'\n  CV汇总: MAE={cv_metrics["mae_mean"]:.2f}±{cv_metrics["mae_std"]:.2f}, '
          f'RMSE={cv_metrics["rmse_mean"]:.2f}±{cv_metrics["rmse_std"]:.2f}, '
          f'R²={cv_metrics["r2_mean"]:.4f}±{cv_metrics["r2_std"]:.4f}')
    
    # 全量训练
    n_est = int(np.mean(best_iters) * 1.1)
    final_params = dict(params)
    final_params['n_estimators'] = n_est
    
    final_model = xgb.XGBRegressor(**final_params, verbosity=0, random_state=42)
    final_model.fit(X_v, y_target)
    
    model_path = os.path.join(save_dir, f'best_model_{target_name}.json')
    final_model.save_model(model_path)
    print(f'  模型已保存: {model_path}')
    
    # 特征重要性
    imp = final_model.feature_importances_
    
    if feature_names:
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': imp / imp.sum(),
        }).sort_values('importance', ascending=False)
        imp_df.to_csv(os.path.join(save_dir, f'feature_importance_{target_name}.csv'),
                       index=False, encoding='utf-8-sig')
        
        print(f'\n  Top 15 特征:')
        for i, row in imp_df.head(15).iterrows():
            print(f'    {row["feature"]:45s} {row["importance"]:.4f}')
    
    # 保存CV预测
    pd.DataFrame({
        'y_true': y_v,
        'y_pred_cv': all_preds,
        'residual': y_v - all_preds,
    }).to_csv(os.path.join(save_dir, f'cv_predictions_{target_name}.csv'),
              index=False, encoding='utf-8-sig')
    
    return final_model, cv_metrics, imp


# ============================================================
# 5. 分层分析
# ============================================================

def analyze_by_visit_order(X, y, visit_orders, params, target_name, log_target=False):
    """按visit_order分组分析模型性能"""
    print(f'\n[分层分析] {target_name} 按visit_order分析:')
    
    valid_mask = ~np.isnan(y)
    X_v = X[valid_mask]
    y_v = y[valid_mask]
    vo = visit_orders[valid_mask]
    
    # 全量CV预测
    if log_target:
        y_target = np.log1p(y_v)
    else:
        y_target = y_v
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.full(len(y_v), np.nan)
    
    for tr_idx, va_idx in kf.split(X_v):
        model = xgb.XGBRegressor(
            **params, verbosity=0, early_stopping_rounds=50, random_state=42)
        if log_target:
            model.fit(X_v[tr_idx], y_target[tr_idx],
                     eval_set=[(X_v[va_idx], y_target[va_idx])], verbose=False)
            pred = np.expm1(model.predict(X_v[va_idx]))
            pred = np.maximum(pred, 0)
        else:
            model.fit(X_v[tr_idx], y_target[tr_idx],
                     eval_set=[(X_v[va_idx], y_v[va_idx])], verbose=False)
            pred = model.predict(X_v[va_idx])
        all_preds[va_idx] = pred
    
    # 分组报告
    print(f'  {"visit_order":<15s} {"N":>6s} {"MAE":>10s} {"RMSE":>10s} {"R²":>10s}')
    print('  ' + '-' * 55)
    
    groups = [('=1', vo == 1), ('=2', vo == 2), ('=3', vo == 3),
              ('4-5', (vo >= 4) & (vo <= 5)),
              ('6-10', (vo >= 6) & (vo <= 10)),
              ('>10', vo > 10)]
    
    results = {}
    for label, mask_g in groups:
        if mask_g.sum() < 10:
            continue
        y_t = y_v[mask_g]
        y_p = all_preds[mask_g]
        valid = ~np.isnan(y_p)
        if valid.sum() < 10:
            continue
        mae = mean_absolute_error(y_t[valid], y_p[valid])
        rmse = np.sqrt(mean_squared_error(y_t[valid], y_p[valid]))
        r2 = r2_score(y_t[valid], y_p[valid])
        print(f'  {label:<15s} {int(valid.sum()):>6d} {mae:>10.2f} {rmse:>10.2f} {r2:>10.4f}')
        results[label] = {'n': int(valid.sum()), 'mae': mae, 'rmse': rmse, 'r2': r2}
    
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    log_path = os.path.join(RESULTS_DIR, 'training_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    
    class TeeWriter:
        def __init__(self, *w):
            self.writers = w
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
    print('患者历史序列 + 超参数优化 预测模型')
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)
    
    # GPU检测
    try:
        _m = xgb.XGBRegressor(device='cuda', n_estimators=2, verbosity=0)
        _m.fit(np.random.randn(10, 3).astype(np.float32),
               np.random.randn(10).astype(np.float32))
        device = 'cuda'
        print('[GPU] XGBoost CUDA 已启用')
    except Exception:
        device = 'cpu'
        print('[CPU] XGBoost CPU 模式')
    
    # 1. 加载数据
    print('\n[Step 1] 加载数据')
    df = load_train_data()
    
    # 2. 构建历史特征
    print('\n[Step 2] 构建历史序列特征')
    df = build_history_features(df)
    
    # 3. 特征矩阵
    print('\n[Step 3] 构建完整特征矩阵')
    features_df, feature_names = build_full_feature_matrix(df)
    
    X = features_df.values.astype(np.float32)
    y_gap = df['target_gap_days'].values.astype(np.float64)
    y_los = df['target_next_los'].values.astype(np.float64)
    visit_orders = df['visit_order'].values
    
    print(f'\n  特征矩阵: {X.shape}')
    print(f'  gap_days 有效: {(~np.isnan(y_gap)).sum()}')
    print(f'  next_los 有效: {(~np.isnan(y_los)).sum()}')
    
    # 保存特征数据
    features_df.to_csv(os.path.join(RESULTS_DIR, 'history_features.csv'),
                        index=False, encoding='utf-8-sig')
    
    # ==========================================
    # 4. Gap Days 优化
    # ==========================================
    print('\n' + '#' * 80)
    print('# 目标 1: 住院间隔 (gap_days)')
    print('#' * 80)
    
    # 先测试 raw vs log-target
    print('\n--- 快速对比: raw-target vs log-target ---')
    quick_params = {
        'device': device, 'n_estimators': 1200, 'max_depth': 6,
        'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0,
    }
    m_raw = cv_evaluate(X, y_gap, quick_params, log_target=False)
    m_log = cv_evaluate(X, y_gap, quick_params, log_target=True)
    print(f'  Raw-target:  MAE={m_raw["mae_mean"]:.2f}, R²={m_raw["r2_mean"]:.4f}')
    print(f'  Log-target:  MAE={m_log["mae_mean"]:.2f}, R²={m_log["r2_mean"]:.4f}')
    
    use_log_gap = m_log['r2_mean'] > m_raw['r2_mean']
    print(f'  → 选择: {"log-target" if use_log_gap else "raw-target"}')
    
    # 超参数搜索
    print('\n--- 超参数搜索 ---')
    gap_params, gap_final = staged_search(X, y_gap, 'gap_days', device, log_target=use_log_gap)
    
    # 最终训练
    gap_model, gap_metrics, gap_imp = train_final_model(
        X, y_gap, gap_params, 'gap_days', RESULTS_DIR,
        feature_names, log_target=use_log_gap)
    
    # 分层分析
    gap_strata = analyze_by_visit_order(
        X, y_gap, visit_orders, gap_params, 'gap_days', log_target=use_log_gap)
    
    # ==========================================
    # 5. Next LOS 优化
    # ==========================================
    print('\n\n' + '#' * 80)
    print('# 目标 2: 下次住院时长 (next_los)')
    print('#' * 80)
    
    print('\n--- 快速对比: raw-target vs log-target ---')
    m_raw_l = cv_evaluate(X, y_los, quick_params, log_target=False)
    m_log_l = cv_evaluate(X, y_los, quick_params, log_target=True)
    print(f'  Raw-target:  MAE={m_raw_l["mae_mean"]:.2f}, R²={m_raw_l["r2_mean"]:.4f}')
    print(f'  Log-target:  MAE={m_log_l["mae_mean"]:.2f}, R²={m_log_l["r2_mean"]:.4f}')
    
    use_log_los = m_log_l['r2_mean'] > m_raw_l['r2_mean']
    print(f'  → 选择: {"log-target" if use_log_los else "raw-target"}')
    
    print('\n--- 超参数搜索 ---')
    los_params, los_final = staged_search(X, y_los, 'next_los', device, log_target=use_log_los)
    
    los_model, los_metrics, los_imp = train_final_model(
        X, y_los, los_params, 'next_los', RESULTS_DIR,
        feature_names, log_target=use_log_los)
    
    los_strata = analyze_by_visit_order(
        X, y_los, visit_orders, los_params, 'next_los', log_target=use_log_los)
    
    # ==========================================
    # 6. 报告
    # ==========================================
    print('\n\n' + '=' * 80)
    print('最终结果汇总')
    print('=' * 80)
    
    print(f'\n{"":15s} {"MAE":>12s} {"RMSE":>12s} {"R²":>12s} {"目标MAE":>10s} {"目标R²":>10s} {"达标":>6s}')
    print('-' * 80)
    
    gap_pass_mae = gap_metrics['mae_mean'] < 24
    gap_pass_r2 = gap_metrics['r2_mean'] > 0.85
    gap_pass = '✓' if (gap_pass_mae and gap_pass_r2) else '✗'
    print(f'{"gap_days":15s} {gap_metrics["mae_mean"]:>10.2f}±{gap_metrics["mae_std"]:.1f} '
          f'{gap_metrics["rmse_mean"]:>10.2f}±{gap_metrics["rmse_std"]:.1f} '
          f'{gap_metrics["r2_mean"]:>10.4f}±{gap_metrics["r2_std"]:.3f} '
          f'{"<24":>10s} {">0.85":>10s} {gap_pass:>6s}')
    
    los_pass_mae = los_metrics['mae_mean'] < 2
    los_pass_r2 = los_metrics['r2_mean'] > 0.5
    los_pass = '✓' if (los_pass_mae and los_pass_r2) else '✗'
    print(f'{"next_los":15s} {los_metrics["mae_mean"]:>10.2f}±{los_metrics["mae_std"]:.1f} '
          f'{los_metrics["rmse_mean"]:>10.2f}±{los_metrics["rmse_std"]:.1f} '
          f'{los_metrics["r2_mean"]:>10.4f}±{los_metrics["r2_std"]:.3f} '
          f'{"<2":>10s} {">0.5":>10s} {los_pass:>6s}')
    
    # 保存所有结果
    all_results = {
        'gap_days': {
            'params': {k: v for k, v in gap_params.items() if k != 'device'},
            'metrics': gap_metrics,
            'log_target': use_log_gap,
            'strata': gap_strata,
        },
        'next_los': {
            'params': {k: v for k, v in los_params.items() if k != 'device'},
            'metrics': los_metrics,
            'log_target': use_log_los,
            'strata': los_strata,
        },
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }
    
    with open(os.path.join(RESULTS_DIR, 'history_model_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    elapsed = time.time() - t_start
    print(f'\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print(f'结果保存: {RESULTS_DIR}')
    
    sys.stdout = original_stdout
    log_file.close()


if __name__ == '__main__':
    main()
