import os
import glob
import json
import re
import argparse

import pandas as pd

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLER_AVAILABLE = True
except Exception:
    IMBLER_AVAILABLE = False


def load_json_dir(data_dir):
    files = glob.glob(os.path.join(data_dir, '*.json'))
    rows = []
    for p in files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            obj['_source_file'] = os.path.basename(p)
            rows.append(obj)
        except Exception:
            continue
    return pd.DataFrame(rows)


def detect_lab_features(df, extra_patterns=None):
    patterns = [r'检验', r'化验', r'lab', r'Lab', r'test', r'Test', r'血', r'尿', r'生化', r'CRP', r'WBC', r'RBC']
    if extra_patterns:
        patterns = list(patterns) + list(extra_patterns)
    regex = re.compile('|'.join(patterns))
    return [c for c in df.columns if regex.search(c)]


def merge_socio(df, socio_csv, on=['province', 'city']):
    socio = pd.read_csv(socio_csv, dtype=str)
    merged = df.merge(socio, how='left', left_on=on, right_on=on)
    return merged


def balance_train(X, y, method='oversample', random_state=42):
    if method is None or method == 'none':
        return X, y
    if IMBLER_AVAILABLE:
        if method == 'oversample':
            ros = RandomOverSampler(random_state=random_state)
            X_res, y_res = ros.fit_resample(X, y)
            return X_res, y_res
        if method == 'undersample':
            rus = RandomUnderSampler(random_state=random_state)
            X_res, y_res = rus.fit_resample(X, y)
    # fallback simple resample
    from sklearn.utils import resample
    df = X.copy()
    df['_label_'] = y
    counts = df['_label_'].value_counts()
    if method == 'oversample':
        max_n = counts.max()
        parts = []
        for label, group in df.groupby('_label_'):
            if len(group) < max_n:
                parts.append(resample(group, replace=True, n_samples=max_n, random_state=random_state))
            else:
                parts.append(group)
        res = pd.concat(parts)
    else:  # undersample
        min_n = counts.min()
        parts = [resample(g, replace=False, n_samples=min_n, random_state=random_state) for _, g in df.groupby('_label_')]
        res = pd.concat(parts)
    y_res = res['_label_']
    X_res = res.drop(columns=['_label_'])
    return X_res, y_res


def smart_preprocess(df, exclude_lab_list=None, detect_extra=None):
    drop_cols = []
    possible_id_cols = ['id', 'patient_id', '住院号', '门诊号']
    for c in possible_id_cols:
        if c in df.columns:
            drop_cols.append(c)

    lab_cols = []
    if exclude_lab_list:
        lab_cols = [c for c in df.columns if c in set(exclude_lab_list)]
    else:
        lab_cols = detect_lab_features(df, extra_patterns=detect_extra)

    keep_cols = [c for c in df.columns if c not in set(lab_cols + drop_cols)]
    return df[keep_cols], lab_cols, drop_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory with patient JSON files')
    parser.add_argument('--socio_csv', required=False, help='CSV with socioeconomic data (province,city and metrics)')
    parser.add_argument('--label_col', required=True, help='Name of label column for prediction')
    parser.add_argument('--exclude_lab_file', required=False, help='Optional text file listing lab feature names (one per line)')
    parser.add_argument('--balance_method', choices=['oversample', 'undersample', 'none'], default='oversample')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--output_dir', default='processed')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_json_dir(args.data_dir)
    if df.empty:
        print('No JSON files loaded from', args.data_dir)
        return

    exclude_lab_list = None
    if args.exclude_lab_file and os.path.exists(args.exclude_lab_file):
        with open(args.exclude_lab_file, 'r', encoding='utf-8') as f:
            exclude_lab_list = [l.strip() for l in f if l.strip()]

    df_proc, lab_cols, id_cols = smart_preprocess(df, exclude_lab_list=exclude_lab_list)

    if args.socio_csv:
        df_proc = merge_socio(df_proc, args.socio_csv)

    if args.label_col not in df_proc.columns:
        raise KeyError(f'label column {args.label_col} not found after preprocessing')

    df_proc = df_proc.dropna(axis=1, how='all')
    df_proc = df_proc.fillna(-999)

    X = df_proc.drop(columns=[args.label_col])
    y = df_proc[args.label_col]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    X_train_bal, y_train_bal = balance_train(X_train, y_train, method=args.balance_method, random_state=args.random_state)

    X_train_bal.to_csv(os.path.join(args.output_dir, 'train_X.csv'), index=False)
    y_train_bal.to_csv(os.path.join(args.output_dir, 'train_y.csv'), index=False)
    X_test.to_csv(os.path.join(args.output_dir, 'test_X.csv'), index=False)
    y_test.to_csv(os.path.join(args.output_dir, 'test_y.csv'), index=False)

    meta = {
        'removed_lab_cols': lab_cols,
        'removed_id_cols': id_cols,
        'balance_method': args.balance_method,
        'imblearn_available': IMBLER_AVAILABLE,
    }
    with open(os.path.join(args.output_dir, 'processing_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('Processing complete. Outputs in', args.output_dir)


if __name__ == '__main__':
    main()
