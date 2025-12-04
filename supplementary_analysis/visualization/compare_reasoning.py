import pandas as pd
import altair as alt

# 1. 데이터 로드
file_path = './data/Table_COMPass - reasoning_performance.csv' # no file
df = pd.read_csv(file_path)

# 2. 데이터 전처리 (결측치 처리)
# CSV의 빈 문자열을 NaN으로 변환
df.replace('', pd.NA, inplace=True)

f1_cols = ['Depression F1', 'Anxiety F1', 'Stress F1']
for col in f1_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. 시각화를 위한 'Method' 컬럼 생성
def assign_method(row):
    if row['Shot'] == 'Zero shot':
        # Baseline
        return 'Baseline (Zero-shot DP)'
    elif row['Shot'] == '4 shot':
        # 4-shot 비교군
        if row['Reasoning'] == 'Direct Prediction':
            return '4-shot Direct Prediction'
        elif row['Reasoning'] == 'Chain-of-Though':
            return '4-shot Chain-of-Though'
        elif row['Reasoning'] == 'Self Refine':
            return '4-shot Self Refine'
    return 'Other' # 혹시 모를 다른 케이스

df['Method'] = df.apply(assign_method, axis=1)

# 4. 데이터 'Melt' (Long-form. Altair에 적합)
df_melted = df.melt(
    id_vars=['DATA', 'Method'], 
    value_vars=f1_cols, 
    var_name='Metric', 
    value_name='F1 Score'
)

# 5. Altair 차트 생성
# Method 순서 (Baseline이 가장 먼저 오도록)
method_order = [
    'Baseline (Zero-shot DP)', 
    '4-shot Direct Prediction', 
    '4-shot Chain-of-Though', 
    '4-shot Self Refine'
]

# 툴팁에 소수점 3자리까지 표시
tooltip = [
    alt.Tooltip('DATA', title='Dataset'),
    alt.Tooltip('Metric', title='Performance Metric'),
    alt.Tooltip('Method'),
    alt.Tooltip('F1 Score', format='.3f') # 소수점 3자리
]

chart = alt.Chart(df_melted).mark_bar().encode(
    # X축: Method (순서 지정, 축 라벨은 숨김)
    x=alt.X('Method', sort=method_order, axis=None),
    
    # Y축: F1 Score (Scale을 0.4에서 0.9로 조정하여 차이 부각)
    y=alt.Y('F1 Score', scale=alt.Scale(domain=[0.4, 0.9])),
    
    # Color: Method별로 색상 지정 (범례 포함)
    color=alt.Color('Method', sort=method_order, legend=alt.Legend(title="Reasoning Method")),
    
    # Column (열 분할): Dataset
    column=alt.Column('DATA', header=alt.Header(titleOrient="bottom", labelOrient="bottom", title="Dataset")),
    
    # Row (행 분할): Metric
    row=alt.Row('Metric', header=alt.Header(titleOrient="left", labelOrient="left", title="Performance Metric")),
    
    # 툴팁 (마우스 오버 시 정보 표시)
    tooltip=tooltip
).properties(
    title='Reasoning Performance by Dataset and Metric'
).interactive() # 차트 확대/축소 가능

# 6. 차트 저장
chart_path = 'reasoning_performance_comparison.json'
chart.save(chart_path)

print(f"Visualization generated and saved to '{chart_path}'")