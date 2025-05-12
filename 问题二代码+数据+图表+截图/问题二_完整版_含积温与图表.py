import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib as mpl

# 设置中文显示
mpl.rcParams['font.family'] = 'SimHei'  # 设置全局字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示
mpl.rcParams['pdf.fonttype'] = 42  # 输出PDF时保留文字
mpl.rcParams['ps.fonttype'] = 42   # 输出PS时保留文字

class MLBloomModel:
    def __init__(self):
        self.model_first = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_full = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_duration = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = [
            'year', 'temp_jan_avg', 'temp_feb_avg', 'temp_mar_avg',
            'rainfall_jan_total', 'rainfall_feb_total', 'rainfall_mar_total',
            'daylight_jan_avg', 'daylight_feb_avg', 'daylight_mar_avg', 'gdd'
        ]
        self.target_first = 'first_bloom_doy'
        self.target_full = 'full_bloom_doy'
        self.target_duration = 'bloom_duration_days'

    def fit(self, df):
        df = df.dropna(subset=self.features + [self.target_first, self.target_full, self.target_duration])
        X = self.scaler.fit_transform(df[self.features])
        self.model_first.fit(X, df[self.target_first])
        self.model_full.fit(X, df[self.target_full])
        self.model_duration.fit(X, df[self.target_duration])
        return self

    def predict(self, df):
        df = df.copy()
        for feat in self.features:
            if feat not in df.columns:
                df[feat] = 0
        X = self.scaler.transform(df[self.features].fillna(0))
        first = self.model_first.predict(X)
        full = self.model_full.predict(X)
        dur = self.model_duration.predict(X)
        return [{
            'first_bloom_doy': int(round(f)),
            'full_bloom_doy': int(round(max(fu, f + 1))),
            'bloom_duration_days': int(round(max(1, d))),
            'end_bloom_doy': int(round(f + d))
        } for f, fu, d in zip(first, full, dur)][0]

    def plot_feature_importance(self):
        """新增：绘制特征重要性图"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        for i, (model, title) in enumerate(zip(
                [self.model_first, self.model_full, self.model_duration],
                ['始花期特征重要性', '盛花期特征重要性', '花期持续时间特征重要性']
        )):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            sns.barplot(x=importances[indices], y=np.array(self.features)[indices], ax=axes[i])
            axes[i].set_title(title)
        plt.tight_layout()
        return fig

def load_data(filepath):
    """加载并验证数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件 {filepath} 不存在")
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # 自动计算gdd（如果不存在）
    if 'gdd' not in df.columns and all(c in df.columns for c in ['temp_feb_avg', 'temp_mar_avg']):
        df['gdd'] = max(0, df['temp_feb_avg'] - 5) * 28 + max(0, df['temp_mar_avg'] - 5) * 31

    required_cols = ['year', 'temp_feb_avg', 'temp_mar_avg', 'first_bloom_doy']
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"缺少必要列: {missing}")
    return df

def plot_bloom_history(df):
    """绘制历史花期趋势图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x='year', y='first_bloom_doy', data=df, marker='o', label='始花期', ax=ax)
    sns.lineplot(x='year', y='full_bloom_doy', data=df, marker='s', label='盛花期', ax=ax)
    sns.lineplot(x='year', y='end_bloom_doy', data=df, marker='^', label='末花期', ax=ax)

    # 添加花期区间阴影
    for _, row in df.iterrows():
        ax.fill_between([row['year']-0.2, row['year']+0.2],
                        row['first_bloom_doy'],
                        row['end_bloom_doy'],
                        alpha=0.1, color='green')

    ax.set_title('历史花期变化趋势', fontsize=16)
    ax.set_xlabel('年份')
    ax.set_ylabel('年积日(DOY)')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_bloom_prediction(prediction, year):
    """绘制预测花期强度图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # 生成花期强度曲线
    start, peak, end = prediction['first_bloom_doy'], prediction['full_bloom_doy'], prediction['end_bloom_doy']
    x = np.linspace(start-10, end+10, 200)
    y = np.zeros_like(x)
    y[(x >= start) & (x <= end)] = 0.2 + 0.8 * np.sin(
        np.pi * (x[(x >= start) & (x <= end)] - start) / (end - start)
    )

    ax.plot(x, y, color='red', lw=2, label='花期强度')
    ax.axvline(start, color='green', ls='--', label=f'始花期 ({start} DOY)')
    ax.axvline(peak, color='blue', ls='--', label=f'盛花期 ({peak} DOY)')
    ax.axvline(end, color='purple', ls='--', label=f'末花期 ({end} DOY)')

    ax.set_title(f'{year}年油菜花花期预测', fontsize=16)
    ax.set_xlabel('年积日(DOY)')
    ax.set_ylabel('开花强度')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def main():
    # 配置参数
    data_file = r"D:\学习资料\数模\2025数维杯\问题二代码+数据+图表+截图\新版\杏花年度历史数据_含积温.csv"
    target_year = 2026
    output_dir = "杏花"
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("正在加载数据...")
    try:
        df = load_data(data_file)
        print(f"数据加载成功，共 {len(df)} 条记录")
        print(df[['year', 'temp_feb_avg', 'first_bloom_doy']].head())
    except Exception as e:
        print(f"错误: {e}")
        return

    # 训练模型
    print("\n训练模型中...")
    model = MLBloomModel().fit(df)

    # 保存特征重要性图
    feat_imp_fig = model.plot_feature_importance()
    feat_imp_fig.savefig(f"{output_dir}/特征重要性.png", dpi=300, bbox_inches='tight')
    plt.close(feat_imp_fig)

    # 准备预测数据（使用最近一年数据+小幅变化）
    last = df.iloc[-1]
    predict_data = {
        'year': target_year,
        'temp_jan_avg': last['temp_jan_avg'] + 0.1,
        'temp_feb_avg': last['temp_feb_avg'] + 0.15,
        'temp_mar_avg': last['temp_mar_avg'] + 0.2,
        'rainfall_jan_total': last['rainfall_jan_total'],
        'rainfall_feb_total': last['rainfall_feb_total'],
        'rainfall_mar_total': last['rainfall_mar_total'],
        'daylight_jan_avg': last['daylight_jan_avg'],
        'daylight_feb_avg': last['daylight_feb_avg'],
        'daylight_mar_avg': last['daylight_mar_avg'],
        'gdd': max(0, last['temp_feb_avg'] + 0.15 - 5) * 28 + max(0, last['temp_mar_avg'] + 0.2 - 5) * 31
    }

    # 执行预测
    print("\n进行预测...")
    prediction = model.predict(pd.DataFrame([predict_data]))

    # 转换日期
    base_date = datetime(target_year, 1, 1)
    start_date = base_date + timedelta(days=prediction['first_bloom_doy']-1)
    peak_date = base_date + timedelta(days=prediction['full_bloom_doy']-1)
    end_date = base_date + timedelta(days=prediction['end_bloom_doy']-1)

    # 输出结果
    print("\n=== 预测结果 ===")
    print(f"始花期: {start_date.strftime('%Y-%m-%d')} (DOY: {prediction['first_bloom_doy']})")
    print(f"盛花期: {peak_date.strftime('%Y-%m-%d')} (DOY: {prediction['full_bloom_doy']})")
    print(f"末花期: {end_date.strftime('%Y-%m-%d')} (DOY: {prediction['end_bloom_doy']})")
    print(f"持续天数: {prediction['bloom_duration_days']}天")

    # 保存可视化结果
    print("\n生成可视化图表...")
    history_fig = plot_bloom_history(df)
    history_fig.savefig(f"{output_dir}/历史趋势.png", dpi=300, bbox_inches='tight')

    pred_fig = plot_bloom_prediction(prediction, target_year)
    pred_fig.savefig(f"{output_dir}/花期预测.png", dpi=300, bbox_inches='tight')

    plt.close('all')
    print(f"所有结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()