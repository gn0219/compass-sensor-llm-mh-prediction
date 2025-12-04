import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # (NEW) Y축 눈금(ticks) 설정을 위해 numpy 임포트
import os # (NEW) 폰트 파일 경로를 확인하기 위해 os 임포트
import matplotlib.font_manager as fm # (NEW) 폰트 매니저 임포트
from matplotlib.container import BarContainer
from pathlib import Path
from typing import Tuple

# Seaborn 스타일 설정 (논문용)
sns.set_theme(style="whitegrid", context="paper")

# (NEW) 폰트 설정 (Linux Biolinum)
# LinBiolinum_R.ttf 파일을 업로드했는지 확인하고 적용합니다.
font_path = 'LinBiolinum_R.ttf'
if os.path.exists(font_path):
    try:
        # 폰트 매니저에 폰트 파일 추가
        fm.fontManager.addfont(font_path)
        # 폰트 속성에서 실제 폰트 이름 가져오기
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name() # 'Linux Biolinum'
        
        # 기본 폰트로 설정
        plt.rcParams['font.family'] = font_name
        print(f"Success: Custom font '{font_name}' loaded from {font_path}")
    except Exception as e:
        print(f"Warning: Found font file, but failed to load '{font_path}'. Error: {e}")
        print("Using default font.")
else:
    print(f"Warning: Font file '{font_path}' not found.")
    print("Please upload 'LinBiolinum_R.ttf' to use the custom font.")
    print("Using default font.")

# ---------------------------------------------------------------------------
# (NEW) Global Chart Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    'y_limit': (0.0, 0.8),
    'chart_height': 3,
    'chart_aspect_ratio_wide': 2.4,  # For charts with more x-axis items
    'chart_aspect_ratio_narrow': 2.4, # For charts with fewer x-axis items
    'image_dpi': 300,
    
    # (NEW) X축 레이블 회전 각도 (0 = 수평)
    'x_label_rotation': 0, 
    
    # (NEW) True로 설정하면 막대 위에 수치(소수점 2자리)를 표시합니다.
    'show_values_on_bars': True,
    'enable_chart1': False,
    
    # --- 여기서 차트에 표시할 모델을 수정하세요 ---
    # 'models_to_show': ['GPT-5', 'Claude Sonnet 4.5', 'Gemini 2.5 Pro', 'gpt-oss-20b', 'Llama-3.1 8b'],
    'models_to_show': ['GPT-5', 'Claude Sonnet 4.5', 'Gemini 2.5 Pro', 'gpt-oss-20b'],
    'dataset_filter': None,  # 특정 데이터셋만 사용 (None이면 전체 사용)
    
    # (NEW) --- 여기서 모델 이름을 그래프에 표시할 이름으로 수정하세요 ---
    'model_rename_map': {
        'GPT-5': 'GPT-5',
        'Claude Sonnet 4.5': 'Claude Sonnet 4.5',
        'Gemini 2.5 Pro': 'Gemini 2.5 Pro',
        'gpt-oss-20b': 'gpt-oss-20b',
        'Llama-3.1 8b': 'Llama-3.1 8B',
        'Mistral 7B': 'Mistral 7B'
    },

    # --- (NEW) 여기서 차트 색상을 수정하세요 ---
    'custom_palette_reasoning': {
        'Direct': '#1f77b4', # Muted Blue
        'CoT': '#ff7f0e'     # Safety Orange
    },
    'custom_palette_strategy': {
        'Zero-shot': '#A1B1BA',        # Gray (사용자님이 'Base: CoT'에서 변경)
        'Cross Random': '#C58AF9',     # Muted Blue
        'Cross Retrieval': '#FFE51E',  # Safety Orange (사용자님이 변경)
        'Personal Recent': '#F538A0',  # Cooked Asparagus Green (사용자님이 변경)
        'Hybrid': '#00BFFF'      # Brick Red
    },
    
    # --- 여기서 X축, 범례 등의 이름을 수정하세요 ---
    'shot_rename_map': {
        'zeroshot': '0-shot',
        '4shot': '4-shot'
    },
    'reasoning_rename_map': {
        'direct': 'Direct',
        'cot': 'CoT'
    },
    'strategy_rename_map': {
        # (NEW) 'none'을 Base: CoT로 매핑
        'none': 'Zero-shot', # (사용자님이 'Base: CoT'에서 변경)
        'zeroshot': 'Zero-shot',
        'crossrandom': 'Cross Random',
        'crossretrieval': 'Cross Retrieval',
        'personalrecent': 'Personal Recent',
        'hybridblend': 'Hybrid',
        'hybrid': 'Hybrid'
    }
}
# --- End of Global Configuration ---

METRICS = [
    {
        'name': 'Depression',
        'value_col': 'depression_f1_macro',
        'ci_col': 'depression_f1_macro_ci',
    },
    {
        'name': 'Anxiety',
        'value_col': 'anxiety_f1_macro',
        'ci_col': 'anxiety_f1_macro_ci',
    },
    {
        'name': 'Stress',
        'value_col': 'stress_f1_macro',
        'ci_col': 'stress_f1_macro_ci',
    },
]


def normalize_key(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def slugify_label(value: str) -> str:
    if not isinstance(value, str):
        return "unknown"
    return "".join(ch.lower() for ch in value if ch.isalnum())


def parse_ci_bounds(value) -> Tuple[float, float]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        parts = [part.strip() for part in cleaned.split(",")]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return (np.nan, np.nan)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return (np.nan, np.nan)
    return (np.nan, np.nan)


def add_confidence_intervals(ax, data, x_col, hue_col, y_col, lower_col, upper_col, x_order, hue_order):
    if lower_col not in data.columns or upper_col not in data.columns:
        return

    bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
    if not bar_containers:
        return

    x_categories = [x for x in x_order if x in data[x_col].unique()]
    for hue_val, container in zip(hue_order, bar_containers):
        patches = list(container.patches)
        for idx, patch in enumerate(patches):
            if idx >= len(x_categories):
                continue
            x_val = x_categories[idx]
            row = data[(data[x_col] == x_val) & (data[hue_col] == hue_val)]
            if row.empty:
                continue
            y = row.iloc[0][y_col]
            lower = row.iloc[0][lower_col]
            upper = row.iloc[0][upper_col]
            if any(pd.isna(val) for val in (y, lower, upper)):
                continue
            lower_err = max(0.0, y - lower)
            upper_err = max(0.0, upper - y)
            if lower_err == 0 and upper_err == 0:
                continue
            center = patch.get_x() + patch.get_width() / 2.0
            ax.errorbar(
                center,
                y,
                yerr=[[lower_err], [upper_err]],
                fmt='none',
                ecolor='#303030',
                elinewidth=1.1,
                capsize=4,
                capthick=1.1,
                zorder=5,
            )


# ---------------------------------------------------------------------------
# (MODIFIED) 함수 1: Reasoning (Direct vs. CoT) 0-shot 비교
# ---------------------------------------------------------------------------
def generate_reasoning_comparison_chart(df, metric, dataset_name, config):
    """
    차트 (1) 생성: Reasoning (Direct vs. CoT) 비교
    - (MODIFIED) '0-shot'으로 고정
    - X축: Model
    - Hue: Reasoning (Direct, CoT)
    - (REMOVED) Col: Shot
    """
    metric_name = metric['name']
    y_column = metric['value_col']
    ci_lower_col = metric.get('ci_lower_col')
    ci_upper_col = metric.get('ci_upper_col')

    print(f"  Generating Chart 1 (Reasoning Comparison, 0-shot) for {metric_name}...")
    
    # (MODIFIED) 0-shot의 'direct' 및 'cot' 데이터만 필터링
    df_chart1 = df[
        (df['Reasoning'].isin(['direct', 'cot'])) &
        (df['Shot'] == 'zeroshot') & # 0-shot으로 고정
        (df['Model'].isin(config['models_to_show']))
    ].copy()

    if df_chart1.empty:
        print("  Warning: No data available for Chart 1 after filtering.")
        return

    # (NEW) 모델 이름 변경 적용
    df_chart1['Model'] = df_chart1['Model'].replace(config.get('model_rename_map', {}))
    # 이름 변경 적용
    df_chart1['Reasoning'] = df_chart1['Reasoning'].replace(config['reasoning_rename_map'])
    
    # (MODIFIED) catplot이 아닌 barplot을 사용 (단일 차트)
    plt.figure(figsize=(config['chart_height'] * config['chart_aspect_ratio_wide'], config['chart_height']))
    
    # (NEW) 색상 문제를 해결하기 위해, hue_order에 맞는 색상 리스트를 생성
    reasoning_order = ['Direct', 'CoT']
    # (NEW) 리스트에 없는 항목이 있어도 오류가 나지 않도록 .get() 사용
    palette_list = [config['custom_palette_reasoning'].get(r, '#4C72B0') for r in reasoning_order]
    model_order_display = [
        config.get('model_rename_map', {}).get(m, m)
        for m in config['models_to_show']
        if config.get('model_rename_map', {}).get(m, m) in df_chart1['Model'].unique()
    ]
    if not model_order_display:
        model_order_display = sorted(df_chart1['Model'].unique())
    ax = sns.barplot(
        data=df_chart1,
        x='Model',
        y=y_column,
        hue='Reasoning',
        hue_order=reasoning_order, # (NEW) hue 순서 고정
        palette=palette_list, # (MODIFIED) 리스트 팔레트 적용
        order=model_order_display,
        errorbar=None,
    )
    add_confidence_intervals(
        ax,
        df_chart1,
        'Model',
        'Reasoning',
        y_column,
        ci_lower_col,
        ci_upper_col,
        model_order_display,
        reasoning_order,
    )
    
    y_axis_label = f"Macro F1 ({metric_name})"
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel("") # X축 레이블 숨기기
    ax.set_ylim(config['y_limit'])
    
    # (NEW) Y축 눈금을 0.1 단위로 설정 (사용자님이 0.9로 수정)
    ax.set_yticks(np.arange(0.0, 0.9, 0.1))
    # (MODIFIED) Y축 그리드(가로선)를 실선으로 변경
    ax.grid(axis='y', linestyle='-', alpha=0.7)
    
    # X축 레이블 회전 (MODIFIED: config에서 값 가져오기)
    ax.tick_params(axis='x', labelrotation=config['x_label_rotation'])
    
    # (NEW) 막대 위에 값 표시 (config['show_values_on_bars']가 True일 때)
    # Temporarily disable numeric labels on bars to avoid overlapping with error bars.
    # if config.get('show_values_on_bars', False): # .get()으로 안전하게 접근
    #     for p in ax.patches:
    #         if p.get_height() == 0:
    #             continue
    #         ax.annotate(
    #             f'{p.get_height():.2f}',
    #             (p.get_x() + p.get_width() / 2., p.get_height()),
    #             ha='center',
    #             va='bottom',
    #             xytext=(0, 3),
    #             textcoords='offset points',
    #             fontsize=8,
    #             color='black',
    #         )
    
    # 범례 설정 (MODIFIED: 범례를 차트 하단으로 이동)
    plt.legend(title='Reasoning', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    sns.despine(left=True) # 왼쪽 테두리 제거
    
    title = f'{dataset_name} · Reasoning Comparison (Direct vs. CoT) - {metric_name} (0-shot)' # (사용자님이 (1) 제거)
    plt.title(title, y=1.03)

    dataset_slug = slugify_label(dataset_name)
    filename = f'chart_1_reasoning_comp_{dataset_slug}_{metric_name.lower()}.png'
    # (MODIFIED) bbox_inches='tight'로 범례가 잘리지 않게 저장
    plt.savefig(filename, dpi=config['image_dpi'], bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# (MODIFIED) 함수 2: ICL Strategy 비교 차트 (4-shot)
# ---------------------------------------------------------------------------
def generate_icl_strategy_chart(df, metric, dataset_name, config):
    """
    차트 (2) 생성: 0-shot CoT (Baseline) vs 4-shot ICL Strategies
    - X축: Model
    - Hue: ICL Strategy (Base: CoT 포함 5가지)
    """
    metric_name = metric['name']
    y_column = metric['value_col']
    ci_lower_col = metric.get('ci_lower_col')
    ci_upper_col = metric.get('ci_upper_col')
    print(f"  Generating Chart 2 (ICL Strategy) for {metric_name}...")
    
    # (MODIFIED)
    # (Reasoning == 'cot')인 모든 데이터를 가져옵니다.
    # 이렇게 하면 'zeroshot' (Base: CoT)과 '4shot' (Strategies)이 모두 포함됩니다.
    # (사용자님 데이터 구조 상 4shot + 'none'은 없다고 가정)
    df_chart2 = df[
        (df['Reasoning'] == 'cot') & 
        # (df['Shot'] == '4shot') & # (MODIFIED) 0-shot(Base)를 포함하기 위해 주석 처리 (사용자님 요청)
        (df['Model'].isin(config['models_to_show']))
    ].copy()

    if df_chart2.empty:
        print("  Warning: No data available for Chart 2 after filtering.")
        return

    # (NEW) 모델 이름 변경 적용
    df_chart2['Model'] = df_chart2['Model'].replace(config.get('model_rename_map', {}))
    # 이름 변경 적용
    df_chart2['ICL Strategy'] = df_chart2['ICL Strategy'].replace(config['strategy_rename_map'])
    
    # ICL Strategy 순서 (데이터에 있는 것만)
    strategy_order = [
        'Zero-shot', # (사용자님이 'Base: CoT'에서 변경)
        'Cross Random', 
        'Cross Retrieval', 
        'Personal Recent', 
        'Hybrid'
    ]
    
    # (MODIFIED) catplot이 아닌 barplot을 사용 (단일 차트)
    plt.figure(figsize=(config['chart_height'] * config['chart_aspect_ratio_wide'], config['chart_height']))

    # (NEW) 색상 문제를 해결하기 위해, hue_order에 맞는 색상 리스트를 생성
    # (NEW) 리스트에 없는 항목이 있어도 오류가 나지 않도록 .get() 사용
    strategy_order = [strategy for strategy in strategy_order if strategy in df_chart2['ICL Strategy'].unique()]
    if not strategy_order:
        print("  Warning: No strategies to plot for Chart 2.")
        return
    palette_list = [config['custom_palette_strategy'].get(strategy, '#4C72B0') for strategy in strategy_order]
    model_order_display = [
        config.get('model_rename_map', {}).get(m, m)
        for m in config['models_to_show']
        if config.get('model_rename_map', {}).get(m, m) in df_chart2['Model'].unique()
    ]
    if not model_order_display:
        model_order_display = sorted(df_chart2['Model'].unique())
    ax = sns.barplot(
        data=df_chart2,
        x='Model',
        y=y_column,
        hue='ICL Strategy',
        hue_order=strategy_order,
        palette=palette_list, # (MODIFIED) 리스트 팔레트 적용
        order=model_order_display,
        errorbar=None,
    )
    add_confidence_intervals(
        ax,
        df_chart2,
        'Model',
        'ICL Strategy',
        y_column,
        ci_lower_col,
        ci_upper_col,
        model_order_display,
        strategy_order,
    )

    y_axis_label = f"F1-macro"
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel("") # X축 레이블 숨기기
    ax.set_ylim(config['y_limit'])
    
    # (NEW) Y축 눈금을 0.1 단위로 설정 (사용자님이 0.9로 수정)
    ax.set_yticks(np.arange(0.0, 0.8, 0.1))
    # (MODIFIED) Y축 그리드(가로선)를 실선으로 변경
    ax.grid(axis='y', linestyle='-', alpha=0.7)
    
    # X축 레이블 회전 (MODIFIED: config에서 값 가져오기)
    ax.tick_params(axis='x', labelrotation=config['x_label_rotation'])
    
    # (NEW) 막대 위에 값 표시 (config['show_values_on_bars']가 True일 때)
    # Temporarily disable numeric labels on bars to avoid overlapping with error bars.
    # if config.get('show_values_on_bars', False):
    #     for p in ax.patches:
    #         if p.get_height() == 0:
    #             continue
    #         ax.annotate(
    #             f'{p.get_height():.2f}',
    #             (p.get_x() + p.get_width() / 2., p.get_height()),
    #             ha='center',
    #             va='bottom',
    #             xytext=(0, 3),
    #             textcoords='offset points',
    #             fontsize=7,
    #             color='black',
    #         )
    
    # 범례 설정 (MODIFIED: 범례를 차트 하단으로 이동, 3열로)
    # ncol=3으로 설정하면 5개 항목이 3 + 2 레이아웃으로 배치됩니다.
    plt.legend(title='In-Context Learning Strategy (0-shot Base vs 4-shot)', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    sns.despine(left=True)
    
    title = f'{dataset_name} · {metric_name}'
    plt.title(title, y=1.03)
    
    dataset_slug = slugify_label(dataset_name)
    filename = f'chart_2_icl_strategy_{dataset_slug}_{metric_name.lower()}.png'
    # (MODIFIED) bbox_inches='tight'로 범례가 잘리지 않게 저장
    plt.savefig(filename, dpi=config['image_dpi'], bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# (NEW) 메인 실행 로직
# ---------------------------------------------------------------------------
def main():
    dataset_names = []
    try:
        file_path = Path('data') / 'Table_COMPass - BootstrapLLM.csv'
        df = pd.read_csv(file_path)

        rename_map = {
            'dataset': 'Dataset',
            'N_shot': 'Shot',
            'ICL_strategy': 'ICL Strategy',
            'reasoning': 'Reasoning',
            'model': 'Model',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if 'Model' in df.columns:
            df['Model'] = df['Model'].astype(str).str.strip()

        if 'Shot' in df.columns:
            df['Shot'] = (
                df['Shot']
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace('-', '', regex=False)
            )
            df['Shot'] = df['Shot'].replace({'0shot': 'zeroshot', '4shot': '4shot'})

        if 'Reasoning' in df.columns:
            df['Reasoning'] = df['Reasoning'].astype(str).str.strip().str.lower()

        if 'ICL Strategy' in df.columns:
            df['ICL Strategy'] = df['ICL Strategy'].fillna('Zero-shot').astype(str).str.strip()
            df['ICL Strategy'] = df['ICL Strategy'].apply(
                lambda val: CONFIG['strategy_rename_map'].get(normalize_key(val), val)
            )

        if 'Dataset' in df.columns:
            dataset_map = {
                'ces': 'CES',
                'mental-iot': 'Mental IoT',
                'globem': 'GLOBEM',
            }
            normalized = df['Dataset'].astype(str).str.strip().str.lower()
            df['Dataset'] = normalized.map(dataset_map).fillna(df['Dataset'])

        for metric in METRICS:
            value_col = metric['value_col']
            ci_col = metric['ci_col']
            ci_lower_col = f"{value_col}_ci_lower"
            ci_upper_col = f"{value_col}_ci_upper"
            metric['ci_lower_col'] = ci_lower_col
            metric['ci_upper_col'] = ci_upper_col
            if ci_col not in df.columns:
                continue
            bounds = df[ci_col].apply(parse_ci_bounds).tolist()
            if not bounds:
                continue
            lower_vals, upper_vals = zip(*bounds)
            df[ci_lower_col] = lower_vals
            df[ci_upper_col] = upper_vals

        primary_metric_col = METRICS[0]['value_col']
        df = df.dropna(subset=['Model', 'Dataset', primary_metric_col])

        dataset_filter = CONFIG.get('dataset_filter')
        if dataset_filter:
            dataset_names = [dataset_filter]
        else:
            dataset_names = sorted(df['Dataset'].dropna().unique())
        if not dataset_names:
            print("Warning: No datasets available after cleaning.")
            return

        print("Data loaded and cleaned successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred during data prep: {e}")
        return

    for dataset_name in dataset_names:
        df_dataset = df[df['Dataset'] == dataset_name].copy()
        if df_dataset.empty:
            print(f"\n=== Dataset: {dataset_name} ===")
            print("  Warning: No data available after filtering. Skipping dataset.")
            continue

        print(f"\n=== Dataset: {dataset_name} ===")
        for metric in METRICS:
            metric_name = metric['name']
            value_col = metric['value_col']
            ci_col = metric['ci_col']
            ci_lower_col = metric.get('ci_lower_col')
            ci_upper_col = metric.get('ci_upper_col')
            print(f"--- Processing charts for: {metric_name} ---")

            if value_col not in df_dataset.columns:
                print(f"  Warning: Column '{value_col}' not found. Skipping {metric_name}.")
                continue

            if ci_col not in df_dataset.columns or ci_lower_col not in df_dataset.columns or ci_upper_col not in df_dataset.columns:
                print(f"  Warning: CI data for '{metric_name}' not found. Skipping.")
                continue

            if not df_dataset[value_col].notna().any():
                print(f"  Warning: No values present for '{metric_name}' in dataset '{dataset_name}'. Skipping.")
                continue

            if CONFIG.get('enable_chart1', True):
                generate_reasoning_comparison_chart(df_dataset, metric, dataset_name, CONFIG)
            generate_icl_strategy_chart(df_dataset, metric, dataset_name, CONFIG)

    print("\n--- All chart PNG files generated successfully. ---")

# 스크립트 실행 시 main 함수 호출
if __name__ == "__main__":
    main()

