"""
Shared configuration for all figure generation modules.
Spring/Summer color palette, Nature journal style, English-only.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Paths ──
OUTPUT_DIR = r'D:\LDH_cancer\files\healthline\readmission_output'
HISTORY_DIR = os.path.join(OUTPUT_DIR, 'history_model')
V5_DIR = os.path.join(OUTPUT_DIR, 'final_v5')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures_v2', 'merge')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Find English font (avoid CJK/SimHei) ──
def _find_english_font():
    """Find a clean sans-serif font that won't render Chinese."""
    for name in ['Arial', 'Helvetica', 'DejaVu Sans', 'Calibri', 'Segoe UI']:
        fonts = fm.findSystemFonts()
        for fp in fonts:
            try:
                prop = fm.FontProperties(fname=fp)
                if name.lower() in prop.get_name().lower():
                    return prop.get_name()
            except Exception:
                continue
    return 'DejaVu Sans'

FONT_NAME = _find_english_font()

# ── Style: Nature journal + spring/summer palette ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [FONT_NAME, 'Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'pdf.fonttype': 42,       # TrueType (Nature requirement)
    'ps.fonttype': 42,
    # Ensure all text/ticks/labels are black
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
})

# Spring/Summer color palette
COLORS = {
    'primary':   '#4A90D9',   # Sky blue
    'secondary': '#E8636E',   # Coral rose
    'accent1':   '#5BBD8C',   # Fresh green
    'accent2':   '#F5A623',   # Warm amber
    'accent3':   '#9B7ED8',   # Lavender
    'accent4':   '#F48FB1',   # Cherry blossom
    'accent5':   '#4DD0E1',   # Aqua
    'accent6':   '#FFD54F',   # Sunflower
    'light_bg':  '#FAFBFC',
    'grid':      '#E8ECF0',
    'text':      '#2C3E50',
}

# 10-color palette for multi-category plots
PAL = [
    '#4A90D9',  # Sky blue
    '#E8636E',  # Coral rose
    '#5BBD8C',  # Fresh green
    '#F5A623',  # Warm amber
    '#9B7ED8',  # Lavender
    '#F48FB1',  # Cherry blossom
    '#4DD0E1',  # Aqua
    '#FFD54F',  # Sunflower
    '#78909C',  # Cool grey
    '#AED581',  # Lime
]

# Gradient colormap for heatmaps (spring/summer)
import matplotlib.colors as mcolors
SPRING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'spring_summer',
    ['#E3F2FD', '#90CAF9', '#42A5F5', '#F5A623', '#E8636E'],
    N=256
)


def save_figure(fig, name, close=True):
    """Save figure in PNG, PDF, and TIF formats."""
    png_path = os.path.join(FIG_DIR, f'{name}.png')
    pdf_path = os.path.join(FIG_DIR, f'{name}.pdf')
    tif_path = os.path.join(FIG_DIR, f'{name}.tif')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tif_path, dpi=300, bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    
    print(f"  Saved: {name}.png / .pdf / .tif")
    if close:
        plt.close(fig)


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add Nature-style panel label (A, B, C...)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            color='black')


# ── Shared readable feature name mapping ──
FEATURE_NAME_MAP = {
    'incoming_gap': 'Incoming Gap',
    'gap_mean_prev': 'Mean Previous Gap',
    'gap_cv_prev': 'Gap CV',
    'gap_std_prev': 'Gap Std Dev',
    'gap_min_prev': 'Min Previous Gap',
    'gap_max_prev': 'Max Previous Gap',
    'gap_median_prev': 'Median Previous Gap',
    'gap_ema': 'Gap EMA',
    'prev_gap_2': 'Previous Gap (t-2)',
    'prev_gap_3': 'Previous Gap (t-3)',
    'gap_trend': 'Gap Trend',
    'los_days': 'Current LOS',
    'los_mean_prev': 'Mean Previous LOS',
    'los_ema': 'LOS EMA',
    'prev_los_1': 'Previous LOS (t-1)',
    'prev_los_2': 'Previous LOS (t-2)',
    'visit_order': 'Visit Order',
    'admission_frequency': 'Admission Frequency',
    'gap_regularity': 'Gap Regularity',
    'gap_deviation': 'Gap Deviation',
    'gap_shortening': 'Gap Shortening',
    'gap_last_diff': 'Gap Last Diff',
    'gap_last_ratio': 'Gap Last Ratio',
    'gap_accel': 'Gap Acceleration',
    'gap_ema_ratio': 'Gap EMA Ratio',
    'log_incoming_gap': 'Log Incoming Gap',
    'log_gap_mean': 'Log Gap Mean',
    'gap_range': 'Gap Range',
    'gap_iqr_proxy': 'Gap IQR Proxy',
    'los_deviation': 'LOS Deviation',
    'los_ema_ratio': 'LOS EMA Ratio',
    'log_los_days': 'Log LOS',
    'los_days_sq': 'LOS Squared',
    'los_days_sqrt': 'LOS Square Root',
    'los_wavg_2': 'LOS Weighted Avg (2)',
    'los_wavg_3': 'LOS Weighted Avg (3)',
    'gap_x_los': 'Gap x LOS',
    'gap_x_freq': 'Gap x Frequency',
    'los_x_freq': 'LOS x Frequency',
    'age': 'Age',
    'sex': 'Sex',
    'total_cost': 'Total Cost',
    'cost_change': 'Cost Change',
    'cost_change_ratio': 'Cost Change Ratio',
    'recent_visits_90': 'Recent Visits (90d)',
    'recent_visits_180': 'Recent Visits (180d)',
    'recent_visits_365': 'Recent Visits (365d)',
    # Cost category features (Chinese -> English)
    '费用_床位费': 'Bed Fee',
    '费用_西药费': 'Western Medicine Fee',
    '费用_中成费': 'Chinese Patent Medicine Fee',
    '费用_中草药': 'Herbal Medicine Fee',
    '费用_检查费': 'Examination Fee',
    '费用_治疗费': 'Treatment Fee',
    '费用_诊查费': 'Consultation Fee',
    '费用_手术费': 'Surgery Fee',
    '费用_化验费': 'Lab Test Fee',
    '费用_护理费': 'Nursing Fee',
    '费用_输血费': 'Transfusion Fee',
    # Socioeconomic features (Chinese -> English)
    'socio_地区生产总值(万元)': 'Regional GDP',
    'socio_人均地区生产总值(元)': 'GDP per Capita',
    'socio_户籍人口(万人)': 'Registered Population',
    'socio_常住人口()': 'Resident Population',
    'socio_城镇常住人口(万人)': 'Urban Population',
    'socio_第一产业增加值占GDP比重(%)': 'Primary Industry Share',
    'socio_第二产业增加值占GDP比重(%)': 'Secondary Industry Share',
    'socio_第三产业增加值占GDP比重(%)': 'Tertiary Industry Share',
    'socio_职工平均工资(元)': 'Average Wage',
    'socio_年末城镇登记失业人员数(人)': 'Registered Unemployed',
    'socio_行政区域土地面积(平方公里)': 'Land Area',
    'socio_人口密度(人／平方公里)': 'Population Density',
    'socio_地区生产总值增长率(%)': 'GDP Growth Rate',
    'socio_卫生机构数(个)': 'Healthcare Institutions',
    'socio_医院、卫生院数(个)': 'Hospitals Count',
    'socio_医院、卫生院床位数(张)': 'Hospital Beds',
    'socio_医生数(人)': 'Physicians Count',
    'socio_普通高等学校学校数(所)': 'Universities Count',
    'socio_普通中学学校数(所)': 'Secondary Schools',
    'socio_城镇职工基本养老保险参保人数(人)': 'Pension Insurance',
    'socio_城镇基本医疗保险参保人数(人)': 'Medical Insurance',
    'socio_失业保险参保人数(人)': 'Unemployment Insurance',
    'socio_社会消费品零售总额(万元)': 'Retail Sales',
    'socio_年末金融机构存款余额(万元)': 'Bank Deposits',
    'socio_城乡居民储蓄年末余额(万元)': 'Resident Savings',
    'socio_可吸入细颗粒物年平均浓度(微克/立方米)': 'PM2.5 Concentration',
    'socio_生活污水处理率(%)': 'Wastewater Treatment Rate',
    # Other unmapped features
    'marriage': 'Marriage Status',
    'pay_type': 'Payment Type',
    'admit_route': 'Admission Route',
    'total_visits': 'Total Visits',
    'department': 'Department',
    'lab_WBC': 'Lab WBC',
    'lab_PLT': 'Lab PLT',
    'lab_RBC': 'Lab RBC',
    'lab_HGB': 'Lab HGB',
    'lab_Na': 'Lab Na',
    'lab_AST': 'Lab AST',
    'lab_LDH': 'Lab LDH',
    'lab_Ca': 'Lab Ca',
    'lab_CRP': 'Lab CRP',
    'lab_TB': 'Lab TB',
    'lab_K': 'Lab K',
    'lab_GLU': 'Lab GLU',
    'lab_CREA': 'Lab CREA',
    'lab_BUN': 'Lab BUN',
    'lab_ALT': 'Lab ALT',
    'lab_ALB': 'Lab ALB',
    'lab_PCT': 'Lab PCT',
    'n_orders': 'Order Count',
    'n_exams': 'Exam Count',
    'n_labs': 'Lab Test Count',
    'diag_text_len': 'Diagnosis Text Length',
    'n_diagnoses': 'Diagnosis Count',
    'admit_month': 'Admission Month',
    'admit_dayofweek': 'Admission Day of Week',
    'admit_year': 'Admission Year',
    'gap_ratio_last_mean': 'Gap Ratio (Last/Mean)',
    'gap_acceleration': 'Gap Acceleration',
    'prev_los_3': 'Previous LOS (t-3)',
    'los_std_prev': 'LOS Std Dev',
    'los_median_prev': 'Median Previous LOS',
    'los_min_prev': 'Min Previous LOS',
    'los_max_prev': 'Max Previous LOS',
    'los_trend': 'LOS Trend',
    'los_ratio_curr_mean': 'LOS Ratio (Curr/Mean)',
    'prev_total_cost': 'Previous Total Cost',
    'delta_lab_WBC': 'Delta Lab WBC',
    'delta_lab_HGB': 'Delta Lab HGB',
    'delta_lab_ALB': 'Delta Lab ALB',
    'delta_lab_CREA': 'Delta Lab CREA',
    'delta_lab_LDH': 'Delta Lab LDH',
    'delta_lab_CRP': 'Delta Lab CRP',
    'delta_lab_BUN': 'Delta Lab BUN',
    'delta_lab_ALT': 'Delta Lab ALT',
    'delta_lab_AST': 'Delta Lab AST',
    'delta_lab_PLT': 'Delta Lab PLT',
    'days_since_first': 'Days Since First Visit',
    'same_department': 'Same Department',
    'gap_los_interaction': 'Gap-LOS Interaction',
    'log_los_days': 'Log LOS',
    'gap_iqr_proxy': 'Gap IQR Proxy',
}


def get_readable_name(name):
    """Convert internal feature name to readable English label."""
    return FEATURE_NAME_MAP.get(name, name.replace('_', ' ').title())
