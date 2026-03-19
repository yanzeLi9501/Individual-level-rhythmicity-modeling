"""
Step 1: 从 healthline JSON 文件中提取患者住院记录
========================================
解析所有JSON文件，提取:
- 患者基本信息 (demographics)
- 住院基本信息 (admission info)
- 诊断信息 (diagnoses)
- 检验结果 (lab results)
- 医嘱信息 (orders)
- 检查信息 (examinations)
- 费用信息 (costs)
- 电子病历文本 (EMR text fields)
"""

import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 辅助函数
# ============================================================

def parse_date(s):
    """解析各种日期格式"""
    if not s or not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    # "2018年9月4日 17:13:47" or "2018年9月4日"
    m = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # "2018-09-04 17:13:47" or "2018-09-04"
    try:
        return pd.Timestamp(s[:10])
    except Exception:
        return pd.NaT


def parse_age(s):
    """解析年龄字符串 → 数值"""
    if not s or not isinstance(s, str):
        return np.nan
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else np.nan


def safe_float(v):
    """安全转换为浮点数"""
    if v is None or v == '':
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def get_emr_section(emr_list, key):
    """从电子病历菜单列表中提取指定section的内容"""
    if not isinstance(emr_list, list):
        return None
    for item in emr_list:
        if isinstance(item, dict) and key in item:
            return item[key]
    return None


def flatten_emr_section(section):
    """将EMR section（可能是list of dicts或dict）展平为dict"""
    result = {}
    if isinstance(section, dict):
        result.update(section)
    elif isinstance(section, list):
        for item in section:
            if isinstance(item, dict):
                result.update(item)
    return result


# ============================================================
# 核心提取函数
# ============================================================

def extract_patient_info(emr_list):
    """提取个人基本信息"""
    section = get_emr_section(emr_list, '个人基本信息')
    info = flatten_emr_section(section)
    return {
        '病案号': info.get('病案号', ''),
        '姓名': info.get('姓名', ''),
        '身份证号': info.get('身份证号', ''),
        '就诊号': info.get('就诊号', ''),
        '性别': info.get('性别', ''),
        '年龄': parse_age(info.get('年龄', '')),
        '出生日期': info.get('出生日期', ''),
        '民族': info.get('民族', ''),
        '婚姻': info.get('婚姻', ''),
        '国籍': info.get('国籍', ''),
        '医疗支付方式': info.get('医疗支付方式', ''),
        '住院次数': info.get('住院次数', ''),
        # 地理信息
        '出生地省': info.get('出生地省', ''),
        '出生地市': info.get('出生地市', ''),
        '出生地县': info.get('出生地县', info.get('出生地县I', '')),
        '籍贯省': info.get('籍贯省', ''),
        '籍贯市': info.get('籍贯市', ''),
        '工作省': info.get('工作省', ''),
        '工作市': info.get('工作市', ''),
        '工作县': info.get('工作县', ''),
        '户口地址': info.get('户口地址', ''),
        # 其他
        '入院时日常生活能力评定量表得分': info.get('入院时日常生活能力评定量表得分', ''),
        '出院时日常生活能力评定量表': info.get('出院时日常生活能力评定量表', ''),
    }


def extract_admission_info(emr_list):
    """提取住院基本信息"""
    section = get_emr_section(emr_list, '住院基本信息')
    info = flatten_emr_section(section)
    return {
        '入院途径': info.get('入院途径', ''),
        '入院日期': parse_date(info.get('入院日期', '')),
        '入院科别': info.get('入院科别', ''),
        '入院病区': info.get('入院病区', ''),
        '转科科别': info.get('转科科别', ''),
        '出院日期': parse_date(info.get('出院日期', '')),
        '出院科别': info.get('出院科别', ''),
        '出院病区': info.get('出院病区', ''),
        '实际住院天数': safe_float(info.get('实际住院天数', '')),
    }


def extract_diagnoses(record):
    """提取诊断信息，及EMR中的诊断"""
    diag_list = record.get('诊断', [])
    emr_list = record.get('电子病历菜单及详情', [])
    
    diagnoses = []
    # 从顶层诊断字段
    if isinstance(diag_list, list):
        for d in diag_list:
            if isinstance(d, dict):
                diagnoses.append(d)
    
    # 从EMR疾病诊断信息
    section = get_emr_section(emr_list, '疾病诊断信息及病理诊断信息')
    if section:
        info = flatten_emr_section(section)
        for key in ['主要诊断ICD编码', '主要诊断入院病情', '其他诊断1']:
            if key in info and info[key]:
                diagnoses.append({'来源': 'EMR', key: info[key]})
    
    # 合并为文本
    diag_texts = []
    for d in diagnoses:
        if isinstance(d, dict):
            name = d.get('诊断名称', d.get('主要诊断ICD编码', ''))
            if name:
                diag_texts.append(str(name))
    return '; '.join(diag_texts) if diag_texts else ''


def extract_lab_summary(record):
    """提取检验结果摘要（关键指标）"""
    labs = record.get('检验列表', [])
    if not isinstance(labs, list):
        return {}
    
    # 关键检验项目及其匹配关键词
    KEY_LABS = {
        'WBC': ['白细胞计数', '白细胞', 'WBC'],
        'RBC': ['红细胞计数', '红细胞', 'RBC'],
        'HGB': ['血红蛋白', 'HGB', 'Hb'],
        'PLT': ['血小板计数', '血小板', 'PLT'],
        'NEU': ['中性粒细胞计数', '中性粒细胞绝对值', '中性粒细胞#'],
        'LYM': ['淋巴细胞计数', '淋巴细胞绝对值', '淋巴细胞#'],
        'ALB': ['白蛋白', '清蛋白', 'ALB'],
        'ALT': ['谷丙转氨酶', '丙氨酸氨基转移酶', 'ALT'],
        'AST': ['谷草转氨酶', '天门冬氨酸氨基转移酶', 'AST'],
        'CREA': ['肌酐', '血肌酐', 'Cr', 'CREA'],
        'BUN': ['尿素氮', '尿素', 'BUN'],
        'GLU': ['葡萄糖', '血糖', '空腹血糖', 'GLU'],
        'CRP': ['C反应蛋白', 'CRP', 'C-反应蛋白'],
        'PCT': ['降钙素原', 'PCT'],
        'LDH': ['乳酸脱氢酶', 'LDH'],
        'TB': ['总胆红素', 'TBIL'],
        'K': ['钾', 'K+'],
        'Na': ['钠', 'Na+'],
        'Ca': ['钙', 'Ca'],
    }
    
    results = {}
    for lab_entry in labs:
        if not isinstance(lab_entry, dict):
            continue
        details = lab_entry.get('检验详情', {})
        if not isinstance(details, dict):
            continue
        sub_items = details.get('报告子项', [])
        if not isinstance(sub_items, list):
            continue
        for item in sub_items:
            if not isinstance(item, dict):
                continue
            item_name = item.get('项目名称', '')
            value_str = item.get('检查结果', '')
            if not item_name or not value_str:
                continue
            for std_name, keywords in KEY_LABS.items():
                if std_name in results:
                    continue
                for kw in keywords:
                    if kw in item_name:
                        val = safe_float(value_str)
                        if not np.isnan(val):
                            results[std_name] = val
                        break
    return results


def extract_orders_summary(record):
    """提取医嘱摘要"""
    orders = record.get('医嘱列表', [])
    if not isinstance(orders, list):
        return {'医嘱数量': 0, '药品医嘱': ''}
    
    drug_names = []
    for o in orders:
        if isinstance(o, dict):
            name = o.get('医嘱名称', '')
            if name:
                drug_names.append(name)
    
    return {
        '医嘱数量': len(orders),
        '药品医嘱': '; '.join(drug_names[:20]),  # 最多保留20条
    }


def extract_exam_summary(record):
    """提取检查结果摘要"""
    exams = record.get('检查列表', [])
    if not isinstance(exams, list):
        return {'检查数量': 0, '检查项目': '', '检查结论': ''}
    
    exam_names = []
    conclusions = []
    for e in exams:
        if isinstance(e, dict):
            name = e.get('检查名称', '')
            if name:
                exam_names.append(name)
            detail = e.get('检查详情', {})
            if isinstance(detail, dict):
                concl = detail.get('诊断或提示', '')
                if concl:
                    conclusions.append(str(concl)[:200])
    
    return {
        '检查数量': len(exams),
        '检查项目': '; '.join(exam_names[:10]),
        '检查结论': ' | '.join(conclusions[:5]),
    }


def extract_costs(emr_list):
    """提取费用信息"""
    section = get_emr_section(emr_list, '费用信息')
    costs = flatten_emr_section(section)
    
    result = {}
    for k, v in costs.items():
        if isinstance(v, str) and '元' in v:
            val = safe_float(v.replace('元', '').strip())
            result[f'费用_{k}'] = val
        elif isinstance(v, (int, float)):
            result[f'费用_{k}'] = float(v)
    return result


def extract_emr_texts(emr_list):
    """提取电子病历文本字段"""
    text_keys = ['主诉', '现病史', '既往史', '个人史', '初步诊断',
                 '诊疗计划', '辅助检查', '专科情况', '病例摘要']
    texts = {}
    for key in text_keys:
        section = get_emr_section(emr_list, key)
        if section:
            if isinstance(section, dict):
                val = section.get(key, '')
            elif isinstance(section, list):
                parts = []
                for item in section:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str) and v.strip():
                                parts.append(v.strip())
                val = ' '.join(parts)
            elif isinstance(section, str):
                val = section
            else:
                val = ''
            texts[f'EMR_{key}'] = str(val)[:500]  # 截断以控制长度
    
    # 出院记录
    discharge = get_emr_section(emr_list, 'EMR120001 出院记录')
    if discharge:
        info = flatten_emr_section(discharge)
        texts['EMR_出院记录'] = str(info)[:500]
    
    return texts


# ============================================================
# 主提取流程
# ============================================================

def process_single_json(filepath):
    """处理单个JSON文件，返回记录列表"""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return records
    
    if not isinstance(data, list):
        return records
    
    for elem in data:
        if not isinstance(elem, dict) or not elem:
            continue
        
        emr_list = elem.get('电子病历菜单及详情', [])
        
        # 基本字段
        rec = {
            '就诊ID': elem.get('就诊ID', ''),
            '住院流水号': elem.get('住院流水号', ''),
            '科室': elem.get('科室', ''),
            '类型': elem.get('类型', ''),
            '日期': elem.get('日期', ''),
        }
        
        # 患者基本信息
        rec.update(extract_patient_info(emr_list))
        
        # 住院信息
        rec.update(extract_admission_info(emr_list))
        
        # 诊断
        rec['诊断文本'] = extract_diagnoses(elem)
        
        # 检验摘要
        lab_summary = extract_lab_summary(elem)
        for k, v in lab_summary.items():
            rec[f'lab_{k}'] = v
        rec['检验项目数'] = len(elem.get('检验列表', []) or [])
        
        # 医嘱摘要
        order_summary = extract_orders_summary(elem)
        rec.update(order_summary)
        
        # 检查摘要
        exam_summary = extract_exam_summary(elem)
        rec.update(exam_summary)
        
        # 费用
        cost_info = extract_costs(emr_list)
        rec.update(cost_info)
        
        # EMR文本
        emr_texts = extract_emr_texts(emr_list)
        rec.update(emr_texts)
        
        records.append(rec)
    
    return records


def extract_all_data(json_dir, output_path=None):
    """
    解析所有JSON文件，构建完整的患者住院记录表
    
    Parameters
    ----------
    json_dir : str
        JSON文件所在目录
    output_path : str, optional
        输出CSV路径
    
    Returns
    -------
    pd.DataFrame
        所有患者住院记录
    """
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob('*.json'))
    print(f'[Step1] 发现 {len(json_files)} 个JSON文件')
    
    all_records = []
    errors = 0
    
    for i, fp in enumerate(json_files):
        if (i + 1) % 1000 == 0:
            print(f'  处理进度: {i+1}/{len(json_files)} ({errors} errors)', file=sys.stderr)
        try:
            records = process_single_json(fp)
            all_records.extend(records)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  Error processing {fp.name}: {e}', file=sys.stderr)
    
    print(f'[Step1] 解析完成: {len(all_records)} 条住院记录, {errors} 个文件出错')
    
    df = pd.DataFrame(all_records)
    
    # 确保日期列正确
    for col in ['入院日期', '出院日期']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 排序
    if '病案号' in df.columns and '入院日期' in df.columns:
        df = df.sort_values(['病案号', '入院日期']).reset_index(drop=True)
    
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'[Step1] 已保存到 {output_path}')
    
    return df


# ============================================================
# 独立运行
# ============================================================

if __name__ == '__main__':
    JSON_DIR = r'D:\LDH_cancer\files\healthline'
    OUTPUT_DIR = os.path.join(JSON_DIR, 'readmission_output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = extract_all_data(
        json_dir=JSON_DIR,
        output_path=os.path.join(OUTPUT_DIR, 'all_admissions.csv')
    )
    
    print(f'\n数据概览:')
    print(f'  总记录数: {len(df)}')
    print(f'  唯一患者数 (病案号): {df["病案号"].nunique()}')
    print(f'  列数: {len(df.columns)}')
    print(f'  日期范围: {df["入院日期"].min()} ~ {df["入院日期"].max()}')
    
    # 统计多次住院患者
    visit_counts = df.groupby('病案号').size()
    multi_visit = (visit_counts >= 2).sum()
    print(f'  多次住院患者数 (≥2次): {multi_visit}')
    print(f'  最大住院次数: {visit_counts.max()}')
