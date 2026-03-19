"""
Step 2: 特征工程 + 社会经济数据合并 + 预测目标构建
================================================
1. 按患者分组，构建住院序列
2. 计算预测目标: 两次住院间隔(天) + 下次住院时长(天)
3. 合并 city_database.xlsx 社会经济特征
4. 构建结构化特征矩阵
5. 生成LLM输入文本
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# 城市名称匹配
# ============================================================

def normalize_city(name):
    """标准化城市名称，去除后缀"""
    if not name or not isinstance(name, str):
        return ''
    name = name.strip()
    for suffix in ['市', '地区', '自治州', '盟', '林区']:
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[:-len(suffix)]
    return name


def load_city_database(xlsx_path):
    """
    加载城市社会经济数据库
    选取关键社会经济指标（降维到核心特征）
    """
    df = pd.read_excel(xlsx_path)
    
    # 标准化地区名
    df['地区_std'] = df['地区'].apply(normalize_city)
    
    # 选取核心社会经济特征（避免189列全部导入）
    CORE_COLS = [
        '年份', '行政区划代码', '地区', '地区_std',
        '地区生产总值(万元)',
        '人均地区生产总值(元)',
        '户籍人口(万人)',
        '常住人口()',
        '城镇常住人口(万人)',
        '第一产业增加值占GDP比重(%)',
        '第二产业增加值占GDP比重(%)',
        '第三产业增加值占GDP比重(%)',
        '职工平均工资(元)',
        '年末城镇登记失业人员数(人)',
        '行政区域土地面积(平方公里)',
        '人口密度(人／平方公里)',
        '地区生产总值增长率(%)',
        '卫生机构数(个)',
        '医院、卫生院数(个)',
        '医院、卫生院床位数(张)',
        '医生数(人)',
        '普通高等学校学校数(所)',
        '普通中学学校数(所)',
        '城镇职工基本养老保险参保人数(人)',
        '城镇基本医疗保险参保人数(人)',
        '失业保险参保人数(人)',
        '社会消费品零售总额(万元)',
        '年末金融机构存款余额(万元)',
        '城乡居民储蓄年末余额(万元)',
        '可吸入细颗粒物年平均浓度(微克/立方米)',
        '生活污水处理率(%)',
    ]
    
    # 只保留存在的列
    available = [c for c in CORE_COLS if c in df.columns]
    df_core = df[available].copy()
    
    # 数值列转数值
    for col in df_core.columns:
        if col not in ['年份', '行政区划代码', '地区', '地区_std']:
            df_core[col] = pd.to_numeric(df_core[col], errors='coerce')
    
    return df_core


def match_city(patient_row, city_db, year=None):
    """
    根据患者地理信息匹配城市数据库
    优先级: 工作市 > 出生地市 > 籍贯市
    """
    city_fields = ['工作市', '出生地市', '籍贯市']
    
    for field in city_fields:
        city_name = patient_row.get(field, '')
        if not city_name or not isinstance(city_name, str) or not city_name.strip():
            continue
        
        city_std = normalize_city(city_name)
        if not city_std:
            continue
        
        matches = city_db[city_db['地区_std'] == city_std]
        
        if len(matches) == 0:
            # 尝试模糊匹配
            matches = city_db[city_db['地区_std'].str.contains(city_std, na=False)]
        
        if len(matches) == 0:
            continue
        
        # 根据年份选择最接近的记录
        if year and '年份' in matches.columns:
            year_diff = (matches['年份'] - year).abs()
            best_idx = year_diff.idxmin()
            return matches.loc[best_idx]
        else:
            return matches.iloc[-1]  # 最新年份
    
    return None


# ============================================================
# 特征工程
# ============================================================

def build_patient_sequences(df):
    """
    按患者分组，构建住院序列并计算预测目标
    
    Returns
    -------
    df_seq : pd.DataFrame
        每行 = 一次住院记录，增加了预测目标列:
        - target_gap_days: 与下次住院的间隔天数 (出院→下次入院)
        - target_next_los: 下次住院时长(天)
        - visit_order: 该患者第几次住院
        - total_visits: 该患者总住院次数
    """
    df = df.copy()
    
    # 必须有关键字段
    required = ['病案号', '入院日期', '出院日期']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    # 只保留有效日期的记录
    mask = df['入院日期'].notna() & df['出院日期'].notna() & (df['病案号'] != '')
    df = df[mask].copy()
    
    # 按患者和入院日期排序
    df = df.sort_values(['病案号', '入院日期']).reset_index(drop=True)
    
    # 为每位患者编号住院序列
    df['visit_order'] = df.groupby('病案号').cumcount() + 1
    df['total_visits'] = df.groupby('病案号')['visit_order'].transform('max')
    
    # 计算预测目标
    df['target_gap_days'] = np.nan
    df['target_next_los'] = np.nan
    
    for pid, group in df.groupby('病案号'):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        for i in range(len(indices) - 1):
            curr_idx = indices[i]
            next_idx = indices[i + 1]
            
            curr_discharge = df.loc[curr_idx, '出院日期']
            next_admit = df.loc[next_idx, '入院日期']
            next_los = df.loc[next_idx, '实际住院天数']
            
            if pd.notna(curr_discharge) and pd.notna(next_admit):
                gap = (next_admit - curr_discharge).days
                if gap >= 0:  # 排除异常负数
                    df.loc[curr_idx, 'target_gap_days'] = gap
            
            if pd.notna(next_los):
                df.loc[curr_idx, 'target_next_los'] = next_los
    
    return df


def build_structured_features(df, city_db=None):
    """
    构建结构化数值特征矩阵
    
    Parameters
    ----------
    df : pd.DataFrame
        带有预测目标的住院记录表
    city_db : pd.DataFrame, optional
        城市社会经济数据
    
    Returns
    -------
    X_struct : pd.DataFrame
        结构化特征矩阵
    feature_names : list
        特征名列表
    """
    features = pd.DataFrame(index=df.index)
    
    # ----- 1. 人口学特征 -----
    features['age'] = pd.to_numeric(df['年龄'], errors='coerce')
    features['sex'] = (df['性别'].str.contains('男', na=False)).astype(int)
    
    # 婚姻状态编码
    marriage_map = {'已婚': 1, '未婚': 0, '离异': 2, '丧偶': 3}
    features['marriage'] = df['婚姻'].apply(
        lambda x: next((v for k, v in marriage_map.items() 
                       if isinstance(x, str) and k in x), -1)
    )
    
    # 医疗支付方式编码
    features['pay_type'] = LabelEncoder().fit_transform(
        df['医疗支付方式'].fillna('未知').astype(str)
    )
    
    # 入院途径编码
    features['admit_route'] = LabelEncoder().fit_transform(
        df['入院途径'].fillna('未知').astype(str)
    )
    
    # ----- 2. 住院特征 -----
    features['los_days'] = pd.to_numeric(df['实际住院天数'], errors='coerce')
    features['visit_order'] = df['visit_order']
    features['total_visits'] = df['total_visits']
    
    # 入院科别编码（取top频次，其余归为"其他"）
    if '入院科别' in df.columns:
        dept_counts = df['入院科别'].value_counts()
        top_depts = dept_counts.head(20).index.tolist()
        dept_mapped = df['入院科别'].apply(lambda x: x if x in top_depts else '其他')
        features['department'] = LabelEncoder().fit_transform(dept_mapped.fillna('未知'))
    
    # ----- 3. 检验指标 -----
    lab_cols = [c for c in df.columns if c.startswith('lab_')]
    for col in lab_cols:
        features[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 衍生指标
    if 'lab_NEU' in df.columns and 'lab_LYM' in df.columns:
        neu = pd.to_numeric(df['lab_NEU'], errors='coerce')
        lym = pd.to_numeric(df['lab_LYM'], errors='coerce')
        features['NLR'] = neu / lym.replace(0, np.nan)
        features['PLR'] = pd.to_numeric(df.get('lab_PLT', pd.Series(dtype=float)), errors='coerce') / lym.replace(0, np.nan)
    
    # ----- 4. 医嘱和检查数量 -----
    features['n_orders'] = pd.to_numeric(df['医嘱数量'], errors='coerce')
    features['n_exams'] = pd.to_numeric(df['检查数量'], errors='coerce')
    features['n_labs'] = pd.to_numeric(df['检验项目数'], errors='coerce')
    
    # ----- 5. 费用特征 -----
    cost_cols = [c for c in df.columns if c.startswith('费用_')]
    for col in cost_cols:
        features[col] = pd.to_numeric(df[col], errors='coerce')
    
    if cost_cols:
        features['total_cost'] = features[cost_cols].sum(axis=1)
    
    # ----- 6. 诊断数量/长度 -----
    features['diag_text_len'] = df['诊断文本'].fillna('').str.len()
    features['n_diagnoses'] = df['诊断文本'].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0
    )
    
    # ----- 7. 时间特征 -----
    if '入院日期' in df.columns:
        features['admit_month'] = df['入院日期'].dt.month
        features['admit_dayofweek'] = df['入院日期'].dt.dayofweek
        features['admit_year'] = df['入院日期'].dt.year
    
    # ----- 8. 社会经济特征 -----
    if city_db is not None:
        print('[Step2] 合并城市社会经济数据...')
        socio_features = []
        for idx, row in df.iterrows():
            year = row['入院日期'].year if pd.notna(row.get('入院日期')) else None
            match = match_city(row, city_db, year=year)
            if match is not None:
                socio_dict = {}
                for col in match.index:
                    if col not in ['年份', '行政区划代码', '地区', '地区_std']:
                        socio_dict[f'socio_{col}'] = match[col]
                socio_features.append(socio_dict)
            else:
                socio_features.append({})
        
        socio_df = pd.DataFrame(socio_features, index=df.index)
        features = pd.concat([features, socio_df], axis=1)
        print(f'  匹配成功率: {(socio_df.notna().any(axis=1).sum())/len(df)*100:.1f}%')
    
    return features


def build_llm_text(row):
    """
    将单条住院记录序列化为LLM输入文本
    
    包含: 基本信息 + 住院信息 + 诊断 + 检验 + 检查 + 费用 + EMR文本
    """
    parts = []
    
    # 基本信息
    parts.append(f"患者信息: {row.get('性别','')} {row.get('年龄','')}岁 "
                 f"婚姻{row.get('婚姻','')} 民族{row.get('民族','')}")
    
    # 地理信息
    geo = f"出生地{row.get('出生地省','')}{row.get('出生地市','')} "
    geo += f"工作地{row.get('工作省','')}{row.get('工作市','')}"
    parts.append(geo)
    
    # 住院信息
    parts.append(f"入院: {row.get('入院日期','')} 科室{row.get('入院科别','')} "
                 f"途径{row.get('入院途径','')} "
                 f"住院{row.get('实际住院天数','')}天 "
                 f"第{row.get('visit_order','')}次住院(共{row.get('total_visits','')}次)")
    
    # 支付方式
    parts.append(f"支付方式: {row.get('医疗支付方式','')}")
    
    # 诊断
    diag = row.get('诊断文本', '')
    if diag and isinstance(diag, str):
        parts.append(f"诊断: {diag[:300]}")
    
    # 检验结果
    lab_parts = []
    for col in row.index:
        if col.startswith('lab_') and pd.notna(row[col]):
            lab_name = col.replace('lab_', '')
            lab_parts.append(f"{lab_name}={row[col]}")
    if lab_parts:
        parts.append(f"检验: {', '.join(lab_parts)}")
    
    # 检查
    exam_info = row.get('检查项目', '')
    if exam_info and isinstance(exam_info, str):
        parts.append(f"检查项目: {exam_info[:200]}")
    exam_concl = row.get('检查结论', '')
    if exam_concl and isinstance(exam_concl, str):
        parts.append(f"检查结论: {exam_concl[:300]}")
    
    # 费用
    cost_parts = []
    for col in row.index:
        if col.startswith('费用_') and pd.notna(row[col]):
            cost_parts.append(f"{col.replace('费用_','')}={row[col]:.0f}")
    if cost_parts:
        parts.append(f"费用: {', '.join(cost_parts[:10])}")
    
    # EMR文本
    for col in row.index:
        if col.startswith('EMR_') and pd.notna(row[col]) and str(row[col]).strip():
            field = col.replace('EMR_', '')
            text = str(row[col])[:200]
            parts.append(f"{field}: {text}")
    
    # 医嘱
    orders = row.get('药品医嘱', '')
    if orders and isinstance(orders, str):
        parts.append(f"医嘱: {orders[:300]}")
    
    return '\n'.join(parts)


def build_features_and_targets(df, city_db=None, output_dir=None):
    """
    完整的特征工程流程
    
    Returns
    -------
    df_train : pd.DataFrame
        可用于训练的样本（有预测目标的记录）
    X_struct : pd.DataFrame
        结构化特征矩阵
    texts : list[str]
        LLM输入文本列表
    y_gap : np.ndarray
        目标1: 住院间隔天数
    y_los : np.ndarray
        目标2: 下次住院时长
    """
    print('[Step2] 构建患者住院序列...')
    df_seq = build_patient_sequences(df)
    
    print(f'  总记录: {len(df_seq)}')
    has_gap = df_seq['target_gap_days'].notna().sum()
    has_los = df_seq['target_next_los'].notna().sum()
    print(f'  有住院间隔目标: {has_gap}')
    print(f'  有下次住院时长目标: {has_los}')
    
    # 筛选有预测目标的样本（至少有一个目标）
    mask = df_seq['target_gap_days'].notna() | df_seq['target_next_los'].notna()
    df_train = df_seq[mask].copy().reset_index(drop=True)
    print(f'  可训练样本数: {len(df_train)}')
    
    if len(df_train) == 0:
        raise ValueError("没有可训练的样本（没有多次住院患者）")
    
    # 构建结构化特征
    print('[Step2] 构建结构化特征...')
    X_struct = build_structured_features(df_train, city_db=city_db)
    
    # 构建LLM文本输入
    print('[Step2] 生成LLM文本输入...')
    texts = df_train.apply(build_llm_text, axis=1).tolist()
    
    y_gap = df_train['target_gap_days'].values
    y_los = df_train['target_next_los'].values
    
    # 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_train.to_csv(os.path.join(output_dir, 'train_data.csv'),
                        index=False, encoding='utf-8-sig')
        X_struct.to_csv(os.path.join(output_dir, 'structured_features.csv'),
                        index=False, encoding='utf-8-sig')
        # 保存文本
        with open(os.path.join(output_dir, 'llm_texts.txt'), 'w', encoding='utf-8') as f:
            for i, t in enumerate(texts):
                f.write(f'===RECORD_{i}===\n{t}\n\n')
        # 保存目标
        np.savez(os.path.join(output_dir, 'targets.npz'),
                 y_gap=y_gap, y_los=y_los)
        print(f'[Step2] 已保存到 {output_dir}')
    
    return df_train, X_struct, texts, y_gap, y_los


# ============================================================
# 独立运行
# ============================================================

if __name__ == '__main__':
    # 加载Step1的输出
    DATA_DIR = r'D:\LDH_cancer\files\healthline\readmission_output'
    csv_path = os.path.join(DATA_DIR, 'all_admissions.csv')
    
    print(f'加载数据: {csv_path}')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['入院日期'] = pd.to_datetime(df['入院日期'], errors='coerce')
    df['出院日期'] = pd.to_datetime(df['出院日期'], errors='coerce')
    
    # 加载城市数据库
    CITY_DB_PATH = r'D:\肺癌npj\评估肺癌背景患者不同放化疗治疗策略的效果2-20250822-1317\评估肺癌背景患者不同放化疗治疗策略的效果2-20250822-1317\研究组1\io_pipeline_v2\io_v2\city_database.xlsx'
    city_db = load_city_database(CITY_DB_PATH)
    print(f'城市数据库: {len(city_db)} 条记录, {city_db["地区_std"].nunique()} 个城市')
    
    df_train, X_struct, texts, y_gap, y_los = build_features_and_targets(
        df, city_db=city_db, output_dir=DATA_DIR
    )
    
    print(f'\n特征矩阵维度: {X_struct.shape}')
    print(f'LLM文本示例 (第1条):\n{texts[0][:500]}')
    print(f'\n目标统计:')
    print(f'  住院间隔: mean={np.nanmean(y_gap):.1f} median={np.nanmedian(y_gap):.1f} days')
    print(f'  下次住院时长: mean={np.nanmean(y_los):.1f} median={np.nanmedian(y_los):.1f} days')
