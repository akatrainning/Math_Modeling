import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import logging
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（需系统已安装对应字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑支持更全的符号
plt.rcParams['axes.unicode_minus'] = False
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='apricot_bloom_analysis.log'
)

# 1. 数据加载和预处理
def load_data(file_path):
    """加载CSV数据文件"""
    try:
        logging.info(f"加载数据: {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"成功加载数据，包含 {data.shape[0]} 行和 {data.shape[1]} 列")
        return data
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        raise

# 2. 数据清理：填补缺失值和异常值处理
def clean_data(data):
    """清理数据，包括处理缺失值和异常值"""
    logging.info("开始数据清理")

    # 记录清理前的缺失值情况
    missing_before = data.isnull().sum()
    if missing_before.sum() > 0:
        logging.info(f"清理前的缺失值情况:\n{missing_before[missing_before > 0]}")

    # 填补缺失值
    data.fillna(method='ffill', inplace=True)
    # 如果前向填充后仍有缺失值，使用后向填充
    data.fillna(method='bfill', inplace=True)

    # 记录清理后的缺失值情况
    missing_after = data.isnull().sum()
    if missing_after.sum() > 0:
        logging.warning(f"清理后仍有缺失值:\n{missing_after[missing_after > 0]}")

    # 检查并处理异常值 (使用IQR方法)
    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].nunique() > 10:  # 只对连续型数值变量进行异常值检查
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > 0:
                logging.info(f"列 {col} 中检测到 {outliers} 个异常值")
                # 将异常值替换为边界值
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound

    logging.info("数据清理完成")
    return data

# 3. 特征工程：提取花期延长的特征
def feature_engineering(data):
    """创建有用的特征"""
    logging.info("开始特征工程")

    # 检查关键列是否存在
    required_columns = ['first_bloom_doy', 'full_bloom_doy',
                        'temp_jan_avg', 'temp_feb_avg', 'temp_mar_avg']

    for col in required_columns:
        if col not in data.columns:
            logging.error(f"缺少必要的列: {col}")
            raise ValueError(f"数据中缺少必要的列: {col}")

    # 计算花期长度
    data['flowering_period_length'] = data['full_bloom_doy'] - data['first_bloom_doy']

    # 验证计算结果合理性
    if (data['flowering_period_length'] < 0).any():
        logging.warning("检测到部分花期长度为负值，这可能表示数据错误")
        # 将负值设为0
        data.loc[data['flowering_period_length'] < 0, 'flowering_period_length'] = 0

    # 计算平均温度
    data['avg_temperature'] = (data['temp_jan_avg'] + data['temp_feb_avg'] + data['temp_mar_avg']) / 3

    # 可能的额外特征
    if 'rainfall_jan_total' in data.columns and 'rainfall_feb_total' in data.columns and 'rainfall_mar_total' in data.columns:
        data['total_spring_rainfall'] = data['rainfall_jan_total'] + data['rainfall_feb_total'] + data['rainfall_mar_total']

    # 如果数据包含年份，可以添加年份特征
    if 'year' in data.columns:
        data['is_recent'] = (data['year'] >= 2020).astype(int)

    logging.info(f"特征工程完成，新增特征: {list(set(data.columns) - set(required_columns))}")
    return data

# 4. 直接收益模型：计算游客和农产品收入
def direct_revenue(data, base_visitors=100, base_product_output=500, product_price=10000, visitor_price=300):
    """计算直接经济收益"""
    logging.info("计算直接经济收益")

    data['visitor_growth'] = 1.4  # 游客量增长40%
    data['product_growth'] = 2.0  # 农产品加工增值系数为2

    # 计算直接收益
    data['visitor_revenue'] = base_visitors * data['visitor_growth'] * visitor_price
    data['product_revenue'] = base_product_output * data['product_growth'] * product_price
    data['direct_revenue'] = data['visitor_revenue'] + data['product_revenue']

    logging.info(f"直接收益计算完成，平均直接收益: {data['direct_revenue'].mean():.2f}")
    return data

# 5. 产业链收益模型：计算文旅产业带动效应
def industry_revenue(data, industry_multiplier=2.5):
    """计算产业链总收益"""
    logging.info("计算产业链总收益")

    if 'direct_revenue' not in data.columns:
        logging.error("缺少直接收益列，无法计算产业链总收益")
        raise ValueError("必须先计算直接收益")

    data['industry_growth_factor'] = industry_multiplier
    data['total_revenue'] = data['direct_revenue'] * data['industry_growth_factor']

    logging.info(f"产业链总收益计算完成，平均总收益: {data['total_revenue'].mean():.2f}")
    return data

# 6. 敏感性分析：游客量变化对总收益的影响
def sensitivity_analysis(data, growth_factors=[0.05, 0.1, 0.15, 0.2]):
    """进行敏感性分析，测试不同游客增长率的影响"""
    logging.info("进行敏感性分析")

    if 'direct_revenue' not in data.columns:
        logging.error("缺少直接收益列，无法进行敏感性分析")
        raise ValueError("必须先计算直接收益")

    base_revenue = data['direct_revenue'].copy()

    for factor in growth_factors:
        column_name = f'revenue_with_{int(factor*100)}pct_increase'
        data[column_name] = base_revenue * (1 + factor)
        logging.info(f"游客增长 {factor*100}% 时的平均收益: {data[column_name].mean():.2f}")

    return data

# 7. 模拟花期延长对经济效益的影响
def flower_period_extension_impact(data, extension_factors=[0.1, 0.2, 0.3], base_impact=100000):
    """计算花期延长对经济效益的影响"""
    logging.info("计算花期延长的经济效益影响")

    if 'flowering_period_length' not in data.columns:
        logging.error("缺少花期长度列，无法计算花期延长影响")
        raise ValueError("必须先计算花期长度")

    # 为多个延长系数计算经济效益
    for factor in extension_factors:
        extension_col = f'extension_{int(factor*100)}pct'
        impact_col = f'economic_impact_{int(factor*100)}pct'

        data[extension_col] = data['flowering_period_length'] * (1 + factor)
        data[impact_col] = data[extension_col] * base_impact

        logging.info(f"花期延长 {factor*100}% 的平均经济效益: {data[impact_col].mean():.2f}")

    # 添加标准经济效益列用于后续分析
    data['extended_flowering_period'] = data['flowering_period_length'] * 1.1  # 默认使用10%延长
    data['economic_impact_extension'] = data['extended_flowering_period'] * base_impact

    return data

# 8. 基于回归模型预测花期延长的经济效益
def build_prediction_model(data):
    """建立预测模型，分析影响经济效益的因素"""
    logging.info("建立花期-经济效益预测模型")

    # 检查必要的特征
    required_features = ['flowering_period_length', 'avg_temperature', 'economic_impact_extension']
    if not all(col in data.columns for col in required_features):
        missing = [col for col in required_features if col not in data.columns]
        logging.error(f"缺少建模所需特征: {missing}")
        raise ValueError(f"数据中缺少建模所需特征: {missing}")

    # 准备特征和目标变量
    X = data[['flowering_period_length', 'avg_temperature']]
    y = data['economic_impact_extension']

    # 建立线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 评估模型
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mse = np.mean((y - y_pred) ** 2)

    logging.info(f"模型结果 - R²: {r2:.4f}, MSE: {mse:.2f}")
    logging.info(f"系数: 花期长度 = {model.coef_[0]:.2f}, 平均温度 = {model.coef_[1]:.2f}, 截距 = {model.intercept_:.2f}")

    # 将预测结果添加到数据中
    data['predicted_economic_impact'] = y_pred

    return data, model

# 9. 可视化分析：创建图表目录
def ensure_output_dir():
    """确保输出目录存在"""
    output_dir = 'visualization_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"创建输出目录: {output_dir}")
    return output_dir

# 9.1 花期延长与经济效益的散点图
def plot_flowering_vs_economic_impact(data, output_dir=None):
    """绘制花期长度与经济效益的关系图"""
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    plt.scatter(data['flowering_period_length'], data['economic_impact_extension'],
                color='blue', alpha=0.7, label='实际数据')

    # 添加趋势线
    z = np.polyfit(data['flowering_period_length'], data['economic_impact_extension'], 1)
    p = np.poly1d(z)
    plt.plot(data['flowering_period_length'], p(data['flowering_period_length']),
             "r--", label=f'趋势线: y={z[0]:.0f}x{z[1]:+.0f}')

    # 设置图表属性
    plt.title('花期长度与经济效益关系', fontsize=14)
    plt.xlabel('花期长度 (天)', fontsize=12)
    plt.ylabel('经济效益 (元)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图表
    if output_dir:
        plt.tight_layout()
        plt.savefig(f'{output_dir}/flowering_vs_economic_impact.png', dpi=300)
        logging.info(f"图表已保存: {output_dir}/flowering_vs_economic_impact.png")

    plt.close()

# 9.2 温度与花期长度的关系
def plot_temperature_vs_flowering(data, output_dir=None):
    """绘制温度与花期长度的关系图"""
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    scatter = plt.scatter(data['avg_temperature'], data['flowering_period_length'],
                          c=data['economic_impact_extension'], cmap='viridis',
                          s=100, alpha=0.7)

    # 添加趋势线
    z = np.polyfit(data['avg_temperature'], data['flowering_period_length'], 1)
    p = np.poly1d(z)
    plt.plot(data['avg_temperature'], p(data['avg_temperature']),
             "r--", label=f'趋势线: y={z[0]:.2f}x{z[1]:+.2f}')

    # 添加颜色条，表示经济效益
    cbar = plt.colorbar(scatter)
    cbar.set_label('经济效益 (元)', rotation=270, labelpad=15)

    # 设置图表属性
    plt.title('平均温度与花期长度关系', fontsize=14)
    plt.xlabel('平均温度 (°C)', fontsize=12)
    plt.ylabel('花期长度 (天)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 保存图表
    if output_dir:
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temperature_vs_flowering.png', dpi=300)
        logging.info(f"图表已保存: {output_dir}/temperature_vs_flowering.png")

    plt.close()

# 9.3 绘制城市比较图
def plot_city_comparison(data, output_dir=None):
    """比较不同城市的经济效益"""
    if 'city' in data.columns:
        plt.figure(figsize=(12, 8))

        # 使用更有吸引力的调色板
        sns.set_palette("viridis", n_colors=len(data['city'].unique()))

        # 绘制箱线图
        ax = sns.boxplot(x='city', y='economic_impact_extension', data=data)

        # 添加数据点
        sns.stripplot(x='city', y='economic_impact_extension', data=data,
                      size=4, color=".3", linewidth=0, alpha=0.6)

        # 设置图表属性
        plt.title('各城市杏花经济效益比较', fontsize=14)
        plt.xlabel('城市', fontsize=12)
        plt.ylabel('经济效益 (元)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        # 在每个箱线图上添加均值标签
        for i, city in enumerate(data['city'].unique()):
            city_mean = data[data['city'] == city]['economic_impact_extension'].mean()
            ax.text(i, city_mean, f'{city_mean:.0f}',
                    horizontalalignment='center', size='small', color='black', weight='semibold')

        # 保存图表
        if output_dir:
            plt.tight_layout()
            plt.savefig(f'{output_dir}/city_comparison.png', dpi=300)
            logging.info(f"图表已保存: {output_dir}/city_comparison.png")

        plt.close()
    else:
        logging.info("数据中没有城市列，跳过城市比较图")

# 9.4 绘制降雨与花期的关系
def plot_rainfall_vs_bloom(data, output_dir=None):
    """绘制降雨量与花期的关系"""
    if 'rainfall_mar_total' in data.columns and 'flowering_period_length' in data.columns:
        plt.figure(figsize=(10, 6))

        # 绘制散点图
        sns.regplot(x='rainfall_mar_total', y='flowering_period_length', data=data,
                    scatter_kws={'alpha':0.6, 's':80}, line_kws={'color':'red'})

        # 设置图表属性
        plt.title('3月降雨量与花期长度关系', fontsize=14)
        plt.xlabel('3月总降雨量 (mm)', fontsize=12)
        plt.ylabel('花期长度 (天)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图表
        if output_dir:
            plt.tight_layout()
            plt.savefig(f'{output_dir}/rainfall_vs_bloom.png', dpi=300)
            logging.info(f"图表已保存: {output_dir}/rainfall_vs_bloom.png")

        plt.close()
    else:
        logging.info("数据中缺少降雨量或花期长度列，跳过降雨与花期关系图")

# 9.5 经济效益预测模型可视化
def plot_prediction_model(data, model, output_dir=None):
    """可视化经济效益预测模型"""
    if 'predicted_economic_impact' in data.columns and 'economic_impact_extension' in data.columns:
        plt.figure(figsize=(10, 6))

        # 绘制实际值与预测值对比
        plt.scatter(data['economic_impact_extension'], data['predicted_economic_impact'],
                    alpha=0.7, s=80, color='blue')

        # 添加45度线
        max_val = max(data['economic_impact_extension'].max(), data['predicted_economic_impact'].max())
        min_val = min(data['economic_impact_extension'].min(), data['predicted_economic_impact'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')

        # 设置图表属性
        plt.title('经济效益预测模型评估', fontsize=14)
        plt.xlabel('实际经济效益 (元)', fontsize=12)
        plt.ylabel('预测经济效益 (元)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 添加R²值
        r2 = model.score(data[['flowering_period_length', 'avg_temperature']], data['economic_impact_extension'])
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # 保存图表
        if output_dir:
            plt.tight_layout()
            plt.savefig(f'{output_dir}/prediction_model.png', dpi=300)
            logging.info(f"图表已保存: {output_dir}/prediction_model.png")

        plt.close()
    else:
        logging.info("数据中缺少预测值列，跳过预测模型可视化")

# 9.6 敏感性分析可视化
def plot_sensitivity_analysis(data, output_dir=None):
    """可视化敏感性分析结果"""
    # 查找所有敏感性分析结果列
    sensitivity_cols = [col for col in data.columns if col.startswith('revenue_with_')]

    if sensitivity_cols:
        plt.figure(figsize=(12, 8))

        # 准备数据
        mean_values = [data[col].mean() for col in sensitivity_cols]
        labels = [col.replace('revenue_with_', '').replace('pct_increase', '%') for col in sensitivity_cols]

        # 绘制条形图
        bars = plt.bar(labels, mean_values, color='skyblue', width=0.6)

        # 在条形上添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5000,
                     f'{height:.0f}',
                     ha='center', va='bottom', rotation=0, fontsize=10)

        # 设置图表属性
        plt.title('游客增长对收益的敏感性分析', fontsize=14)
        plt.xlabel('游客增长比例', fontsize=12)
        plt.ylabel('平均收益 (元)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.ylim(0, max(mean_values) * 1.15)  # 为标签留出空间

        # 保存图表
        if output_dir:
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sensitivity_analysis.png', dpi=300)
            logging.info(f"图表已保存: {output_dir}/sensitivity_analysis.png")

        plt.close()
    else:
        logging.info("数据中没有敏感性分析结果列，跳过敏感性分析可视化")

# 10. 综合分析与汇总报告
def generate_summary_report(data, model=None):
    """生成分析报告"""
    report = {
        "数据概览": {
            "记录数": len(data),
            "特征数": data.shape[1]
        },
        "花期统计": {
            "平均花期长度": data['flowering_period_length'].mean(),
            "最短花期": data['flowering_period_length'].min(),
            "最长花期": data['flowering_period_length'].max()
        },
        "经济效益": {
            "平均经济效益": data['economic_impact_extension'].mean(),
            "最高经济效益": data['economic_impact_extension'].max()
        }
    }

    # 添加模型评估结果
    if model is not None:
        report["模型评估"] = {
            "R²": model.score(data[['flowering_period_length', 'avg_temperature']], data['economic_impact_extension']),
            "花期长度系数": model.coef_[0],
            "温度系数": model.coef_[1]
        }

    # 记录报告
    logging.info("生成分析报告:")
    for section, details in report.items():
        logging.info(f"== {section} ==")
        for key, value in details.items():
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.2f}")
            else:
                logging.info(f"  {key}: {value}")

    return report

# 主函数：整合所有步骤
def main(file_path):
    """主函数，执行完整的分析流程"""
    try:
        logging.info(f"开始分析 {file_path}")

        # 1. 加载数据
        data = load_data(file_path)

        # 2. 数据清理
        data = clean_data(data)

        # 3. 特征工程
        data = feature_engineering(data)

        # 4. 计算直接收益
        data = direct_revenue(data)

        # 5. 计算产业链总收益
        data = industry_revenue(data)

        # 6. 敏感性分析
        data = sensitivity_analysis(data)

        # 7. 计算花期延长的经济效益
        data = flower_period_extension_impact(data)

        # 8. 建立预测模型
        data, model = build_prediction_model(data)

        # 9. 创建可视化输出目录
        output_dir = ensure_output_dir()

        # 10. 进行可视化分析
        plot_flowering_vs_economic_impact(data, output_dir)
        plot_temperature_vs_flowering(data, output_dir)
        plot_city_comparison(data, output_dir)
        plot_rainfall_vs_bloom(data, output_dir)
        plot_prediction_model(data, model, output_dir)
        plot_sensitivity_analysis(data, output_dir)

        # 11. 生成分析报告
        report = generate_summary_report(data, model)

        logging.info("分析完成")

        # 返回处理后的数据和报告
        return {
            'data': data,
            'model': model,
            'report': report
        }

    except Exception as e:
        logging.error(f"分析过程中出错: {str(e)}", exc_info=True)
        raise

# 执行分析
if __name__ == "__main__":
    try:
        # 设置文件路径
        file_path = './油菜花花年度历史数据_含积温.csv'

        # 执行主函数
        result = main(file_path)

        # 输出基本结果
        print("分析完成！")
        print(f"数据记录数: {len(result['data'])}")
        print(f"经济效益模型 R²: {result['report']['模型评估']['R²']:.4f}")
        print(f"可视化结果已保存到 visualization_output 目录")

    except Exception as e:
        print(f"错误: {str(e)}")