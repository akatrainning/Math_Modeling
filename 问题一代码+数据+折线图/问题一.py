# -----------------------------------------------------------
#   predict_qingming_rain_2025_validation.py
# -----------------------------------------------------------
#   功能：判定清明假期(4‑4~4‑6)是否"雨纷纷"，使用逻辑回归模型
#   ‑ 将2025年数据分离用于模型验证
#   ‑ 基于2025年之前的数据训练模型
#   ‑ 预测2025年概率并与实际结果对比
#   ‑ 预测2026年概率
# -----------------------------------------------------------
#   运行示例:
#       python predict_qingming_rain_2025_validation.py
# -----------------------------------------------------------

from pathlib import Path
from datetime import datetime, date
import sys
import re
import pandas as pd
import numpy as np

# Import necessary components from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ========= [1]  路径与字段配置 ===============================================
# !! 注意: 请将此路径修改为您实际存储数据文件的目录 !!
DATA_DIR   = Path("D:\\学习资料\\数模\\2025数维杯\\2025数维杯C题解题思路+完整代码！\\2025数维杯C题解题思路+完整代码！\\S6 2025数维杯C题1-4问可运行代码+参考答案！\\西安天气")
FILE_REGEX = re.compile(r".*\.(xlsx|csv)$", re.I)

# 根据您的表头图片配置列名
DATE_COL   = "时间"
PRECIP_COL = "降水量(mm)"
TEMP_COL = '气温(℃)'
HUMIDITY_COL = '相对湿度(%)'
PRES_COL = '气压(hPa)'
DEW_PT_COL = '露点温度(℃)'
WIND_SPD_COL = '风速(m/s)'
AVG_WIND_SPD_3H_COL = None
MIN_TEMP_COL_DAILY = '最低温度(℃)'
MAX_TEMP_COL_DAILY = '最高温度(℃)'
WIND_DIR_INST_COL = None
MAX_WIND_SPD_COL_INST = None
PRECIP_PHEN_COL = None

# ========= [2]  "雨纷纷"判定规则 =============================================
RAIN_MM_RANGE = (1.0, 25.0)      # 雨纷纷降雨量范围 (mm)
MIN_RAIN_PERIODS_24H = 3         # 最小降雨时段数（24小时内）

# ========= [3]  读取数据 ======================================================
def read_one_file(fp: Path) -> pd.DataFrame:
    """读取单个 Excel/CSV 文件到 DataFrame"""
    print(f"\n读取文件: {fp.name}")
    try:
        if fp.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(fp, engine="openpyxl")
        else:
            try:
                df = pd.read_csv(fp, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(fp, encoding="gbk")
                except Exception as e:
                    print(f"❌ 无法使用 utf-8 或 gbk 编码读取 CSV 文件 {fp.name}: {e}")
                    return pd.DataFrame()
    except FileNotFoundError:
        print(f"❌ 文件未找到: {fp.name}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ 读取文件 {fp.name} 时出错: {e}")
        return pd.DataFrame()
    return df

def load_dataset(root: Path) -> pd.DataFrame:
    """加载指定目录/文件的全部数据并合并"""
    dfs = []
    if root.is_file():
        dfs.append(read_one_file(root))
    elif root.is_dir():
        files = [f for f in root.iterdir() if FILE_REGEX.match(f.name)]
        if not files:
            sys.exit(f"⚠️ 在目录 {root} 中未找到任何 .xlsx / .csv 数据文件")
        dfs = [read_one_file(f) for f in files]
    else:
        sys.exit(f"⚠️ 指定的路径 {root} 不是有效的文件或目录")

    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        sys.exit("⚠️ 未成功读取任何数据文件。")

    df_all = pd.concat(dfs, ignore_index=True)

    global DATE_COL
    date_column_candidates = ["时间", "日期", "日/时", "日/月", "time", "date"]
    if DATE_COL not in df_all.columns:
        print(f"警告: 手动设置的日期列名 {DATE_COL!r} 不在表格列中，尝试自动匹配...")
        for col in date_column_candidates:
            if col in df_all.columns:
                DATE_COL = col
                print(f"找到日期列: {DATE_COL!r}")
                break
        else:
            print(f"❌ 找不到合适的日期列")

    return df_all

# Helper function to calculate yearly aggregated features for a specific period
def calculate_yearly_feature_for_period(df_input, years_to_include, col_name, month_filter, day_range_filter, agg_func, feature_name):
    """Calculates a yearly aggregated feature for a specified month/day range from an input DataFrame."""
    # Filter the input DataFrame by year first
    df_filtered_years = df_input[df_input['Year'].isin(years_to_include)].copy()

    if col_name not in df_filtered_years.columns:
        print(f"警告: 数据中缺少列 {col_name!r}，跳过特征 {feature_name} 的计算。")
        return pd.Series(index=years_to_include, name=feature_name)

    # Filter data for the specified month and day range within the selected years
    if day_range_filter:
        df_period = df_filtered_years[(df_filtered_years['Month'] == month_filter) & (df_filtered_years['Day'].between(day_range_filter[0], day_range_filter[1]))].copy()
    else: # Filter by month only
        df_period = df_filtered_years[df_filtered_years['Month'] == month_filter].copy()

    if df_period.empty:
        # If no data for the period for any of the included years
        print(f"警告: 指定年份 ({sorted(years_to_include)}) 的 {month_filter}月{''.join(f'-{d}' for d in day_range_filter) if day_range_filter else ''} 没有数据，无法计算特征 {feature_name}")
        return pd.Series(index=years_to_include, name=feature_name) # Return Series with NaNs for all years


    # Ensure column is numeric, coercing errors to NaN
    df_period[col_name] = pd.to_numeric(df_period[col_name], errors='coerce')

    # Group by year and apply aggregation function
    yearly_agg = df_period.groupby('Year')[col_name].agg(agg_func)
    yearly_agg.name = feature_name

    # Reindex to ensure all years in years_to_include are present (with NaN if no data for that year/period)
    yearly_agg = yearly_agg.reindex(years_to_include)

    return yearly_agg

# Helper function for count-based features (rainy days, rain periods)
def calculate_yearly_count_feature(df_input, years_to_include, col_name, month_filter, day_range_filter, count_type):
    """Calculates yearly count features (rainy days or rain periods)."""
    df_filtered_years = df_input[df_input['Year'].isin(years_to_include)].copy()

    if col_name not in df_filtered_years.columns:
        print(f"警告: 数据中缺少列 {col_name!r}，跳过特征 {count_type} 的计算。")
        return pd.Series(index=years_to_include, name=f'March_{count_type}' if month_filter==3 else f'EarlyApril_{count_type}')

    # Filter data for the specified month and day range
    if day_range_filter:
        df_period = df_filtered_years[(df_filtered_years['Month'] == month_filter) & (df_filtered_years['Day'].between(day_range_filter[0], day_range_filter[1]))].copy()
    else: # Filter by month only
        df_period = df_filtered_years[df_filtered_years['Month'] == month_filter].copy()

    if df_period.empty:
        print(f"警告: 指定年份 ({sorted(years_to_include)}) 的 {month_filter}月{''.join(f'-{d}' for d in day_range_filter) if day_range_filter else ''} 没有数据，无法计算特征 {count_type}")
        series_name = f'March_{count_type}' if month_filter==3 else f'EarlyApril_{count_type}'
        return pd.Series(index=years_to_include, name=series_name, data=0) # Assume 0 count if no data? Or NaN? Let's return 0 count if no data for the period

    try:
        df_period[col_name] = pd.to_numeric(df_period[col_name], errors='coerce').fillna(0) # Fill precip NaNs with 0

        if count_type == 'Rainy_Days':
            # Group by year and day, sum precip per day
            daily_precip = df_period.groupby(['Year', 'Month', 'Day'])[col_name].sum()
            # Count days where sum is > 0, grouped by year
            count_series = daily_precip[daily_precip > 0].groupby('Year').size()
        elif count_type == 'Rain_Period_Count':
            # Filter for records with precipitation > 0
            df_period_rain_periods = df_period[df_period[col_name] > 0].copy()
            # Group by year and count the number of records (periods)
            count_series = df_period_rain_periods.groupby('Year').size()
        else:
            raise ValueError("Invalid count_type. Use 'Rainy_Days' or 'Rain_Period_Count'.")

        series_name = f'March_{count_type}' if month_filter==3 else f'EarlyApril_{count_type}'
        count_series.name = series_name

        # Reindex to include all specified years, fill missing years (no counts) with 0
        count_series = count_series.reindex(years_to_include, fill_value=0)

        return count_series

    except Exception as e:
        print(f"  - 计算 {count_type} 时出错: {e}")
        series_name = f'March_{count_type}' if month_filter==3 else f'EarlyApril_{count_type}'
        return pd.Series(index=years_to_include, name=series_name, data=np.nan) # Return NaNs on calculation error


# ========= [4]  主流程 ========================================================
def main():
    print(f"开始加载数据集，从目录/文件: {DATA_DIR}")
    df = load_dataset(DATA_DIR)

    if df.empty:
        sys.exit("⚠️ 未加载到任何有效数据，程序终止。")

    # Ensure Date column is datetime type and extract Year, Month, Day
    if DATE_COL not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        print(f"错误: 日期列 {DATE_COL!r} 不可用或不是日期时间类型，尝试重新转换...")
        try:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
            df.dropna(subset=[DATE_COL], inplace=True)
            if df.empty:
                sys.exit("❌ 日期列转换失败且没有有效日期，程序终止。")
            print("日期列转换成功。")
        except Exception as e:
            sys.exit(f"❌ 日期列最终无法转换为日期时间类型: {e}")

    df['Year'] = df[DATE_COL].dt.year
    df['Month'] = df[DATE_COL].dt.month
    df['Day'] = df[DATE_COL].dt.day

    # Ensure essential numeric columns exist and convert them
    numeric_cols_to_check = [PRECIP_COL, TEMP_COL, HUMIDITY_COL, PRES_COL, DEW_PT_COL, WIND_SPD_COL, AVG_WIND_SPD_3H_COL, MIN_TEMP_COL_DAILY, MAX_TEMP_COL_DAILY, MAX_WIND_SPD_COL_INST]
    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == PRECIP_COL:
                df[col] = df[col].fillna(0) # Fill NaN precip with 0
        else:
            print(f"警告: 数据中缺少预期列 {col!r}。")


    # View overall data years
    all_years = df['Year'].dropna().unique()
    if len(all_years) > 0:
        print(f"\n数据集包含的年份: {sorted(all_years)}")
        print(f"总年份数: {len(all_years)}")
    else:
        print("\n警告: 数据集中没有识别到有效的年份信息。")


    # Filter for Qingming holiday dates (Apr 4-6) across all available years
    df_qm_all_years = df[df[DATE_COL].dt.strftime("%m-%d").isin(["04-04", "04-05", "04-06"])].copy()


    # === Calculate the Target Variable (雨纷纷 marker) for ALL historical years ===
    # This is needed to get the actual outcome for 2025
    frequent_rain_dates_all_qm = []

    if not df_qm_all_years.empty and PRECIP_COL in df_qm_all_years.columns:
        df_qm_all_valid_rain = df_qm_all_years[
            (~pd.isna(df_qm_all_years[PRECIP_COL])) &
            (df_qm_all_years[PRECIP_COL] > 0) &
            (df_qm_all_years[PRECIP_COL] >= RAIN_MM_RANGE[0]) &
            (df_qm_all_years[PRECIP_COL] <= RAIN_MM_RANGE[1])
            ].copy()

        if not df_qm_all_valid_rain.empty:
            rain_periods_per_day_all_qm = df_qm_all_valid_rain.groupby(df_qm_all_valid_rain[DATE_COL].dt.date).size()
            frequent_rain_dates_all_qm = rain_periods_per_day_all_qm[rain_periods_per_day_all_qm >= MIN_RAIN_PERIODS_24H].index.tolist()

            # Print frequent rain dates for all years for context
            print(f"\n满足频繁降水条件的清明日期 (所有年份, ≥ {MIN_RAIN_PERIODS_24H} 个满足降雨量范围的时段):")
            if frequent_rain_dates_all_qm:
                for d in sorted(frequent_rain_dates_all_qm):
                    count = rain_periods_per_day_all_qm.get(d, 0)
                    print(f"- {d.strftime('%Y-%m-%d')} ({count} 个时段)")
            else:
                print("无")
        else:
            print("\n所有年份的清明期间都没有满足降雨量范围的记录。")


    # Determine if each historical year had "雨纷纷"
    yearly_flag_all = {}
    years_to_check_all = sorted(df['Year'].dropna().unique())
    for year in years_to_check_all:
        is_fenfen_this_year = False
        for day_offset in range(3): # Apr 4, 5, 6
            try:
                qm_date = date(year, 4, 4 + day_offset)
                if qm_date in frequent_rain_dates_all_qm:
                    is_fenfen_this_year = True
                    break
            except ValueError:
                continue
        yearly_flag_all[year] = is_fenfen_this_year

    yearly_flag_series_all = pd.Series(yearly_flag_all, name="qingming_has_rain_fenfen").sort_index()
    yearly_flag_series_all.index.name = "Year"

    # === Separate Data for Training (Before 2025) and Validation (2025) ===
    train_years = [year for year in yearly_flag_series_all.index if year < 2025]
    year_2025 = 2025
    years_for_prediction = [2026] # We will also predict 2026

    if year_2025 not in yearly_flag_series_all.index:
        sys.exit(f"❌ 数据中不包含年份 {year_2025} 的记录，无法进行 2025 年的验证。请确保数据包含 {year_2025} 全年的记录。")

    # Get the actual outcome for 2025
    y_2025_actual = yearly_flag_series_all.loc[year_2025]
    print(f"\n2025 年清明假期实际是否为『雨纷纷』: {'是' if y_2025_actual else '否'}")


    # === Calculate Features (X) for Training and Validation ===
    print("\n============== 计算模型特征 ==============")

    # --- Calculate Features for all years (including 2025) first ---
    # Use all years available in the data to calculate features
    years_for_feature_calc = sorted(df['Year'].dropna().unique())
    X_all = pd.DataFrame(index=years_for_feature_calc)
    X_all['Year_Feature'] = X_all.index

    print("\n--- 计算3月特征 (所有年份) ---")
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, TEMP_COL, 3, None, 'mean', 'March_Avg_Temp'))
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, HUMIDITY_COL, 3, None, 'mean', 'March_Avg_Humidity'))
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, PRES_COL, 3, None, 'mean', 'March_Avg_Pressure'))
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, PRECIP_COL, 3, None, 'sum', 'March_Total_Precip'))
    X_all = X_all.join(calculate_yearly_count_feature(df, years_for_feature_calc, PRECIP_COL, 3, None, 'Rainy_Days'))
    X_all = X_all.join(calculate_yearly_count_feature(df, years_for_feature_calc, PRECIP_COL, 3, None, 'Rain_Period_Count'))
    # Add more March features...

    print("\n--- 计算4月1-3日特征 (所有年份) ---")
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, TEMP_COL, 4, (1, 3), 'mean', 'EarlyApril_Avg_Temp'))
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, HUMIDITY_COL, 4, (1, 3), 'mean', 'EarlyApril_Avg_Humidity'))
    X_all = X_all.join(calculate_yearly_feature_for_period(df, years_for_feature_calc, PRECIP_COL, 4, (1, 3), 'sum', 'EarlyApril_Total_Precip'))
    X_all = X_all.join(calculate_yearly_count_feature(df, years_for_feature_calc, PRECIP_COL, 4, (1, 3), 'Rainy_Days'))
    X_all = X_all.join(calculate_yearly_count_feature(df, years_for_feature_calc, PRECIP_COL, 4, (1, 3), 'Rain_Period_Count'))
    # Add more Early April features...


    # --- Separate X and y into Training (pre-2025) and 2025 ---
    X_train_full = X_all.loc[train_years].copy()
    y_train_full = yearly_flag_series_all.loc[train_years].copy()

    # Get 2025 features
    X_2025 = X_all.loc[[year_2025]].copy()

    # Drop any rows (years) from training data where features are all NaN (e.g., no data for that year/period)
    # Or handle NaNs using imputation (recommended)
    initial_train_years = X_train_full.shape[0]
    # X_train_full = X_train_full.dropna(how='all') # Drop if ALL features are NaN for a year
    # y_train_full = y_train_full.loc[X_train_full.index] # Align y
    # if initial_train_years > X_train_full.shape[0]:
    #     print(f"\n警告: 训练集中 {initial_train_years - X_train_full.shape[0]} 年因特征缺失被丢弃。")


    # Check if 2025 features have NaNs (this is expected if data for March/EarlyApril 2025 is missing)
    if X_2025.isna().any().any():
        print(f"\n警告: 2025 年的特征数据包含缺失值:\n{X_2025.isna().sum()[X_2025.isna().sum() > 0]}")
        print("这些缺失值将在填充步骤中处理。")


    # Ensure we have enough data for training
    if X_train_full.shape[0] > 5 and y_train_full.nunique() > 1:
        print(f"\n用于训练模型的历史年份数 (截止到 2024): {X_train_full.shape[0]}")
        print(f"训练集目标变量分布:\n{y_train_full.value_counts()}")

        # === Train and Use Logistic Regression Model ===
        print("\n============== 训练逻辑回归模型 ==============")

        try:
            # 1. 划分训练集和验证集 (从预2025年的数据中分)
            # test_size here is for internal model validation, not the 2025 external validation
            X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full)
            print(f"内部训练集年份数: {X_train.shape[0]}, 内部测试集年份数: {X_test.shape[0]}")

            # 2. 应用 Imputation (填充缺失值) 和 Scaling (标准化)
            # Fit imputer and scaler ONLY on the internal training data (X_train)
            print("拟合 Imputer 和 Scaler (基于内部训练集)...")
            imputer = SimpleImputer(strategy='mean') # Or 'median', etc.
            scaler = StandardScaler() # Or None if no scaling is desired

            X_train_processed = imputer.fit_transform(X_train)
            if scaler:
                X_train_processed = scaler.fit_transform(X_train_processed)
            X_train_processed_df = pd.DataFrame(X_train_processed, columns=X_train.columns, index=X_train.index) # Convert back to DF

            # Transform internal test set, 2025 data, and 2026 data using the *fitted* imputer and scaler
            print("转换内部测试集、2025年数据...")
            X_test_processed = imputer.transform(X_test)
            if scaler:
                X_test_processed = scaler.transform(X_test_processed)
            X_test_processed_df = pd.DataFrame(X_test_processed, columns=X_test.columns, index=X_test.index)

            X_2025_processed = imputer.transform(X_2025)
            if scaler:
                X_2025_processed = scaler.transform(X_2025_processed)
            X_2025_processed_df = pd.DataFrame(X_2025_processed, columns=X_2025.columns, index=X_2025.index)


            # 3. 创建并训练逻辑回归模型 (使用处理后的内部训练集)
            model = LogisticRegression(random_state=42, solver='liblinear')
            model.fit(X_train_processed_df, y_train)
            print("逻辑回归模型训练完成。")

            # 4. 评估模型 (在内部测试集上)
            print("\n模型在内部测试集上的评估:")
            y_pred_test = model.predict(X_test_processed_df)
            y_pred_proba_test = model.predict_proba(X_test_processed_df)[:, 1]

            print(f"内部测试集 Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
            if y_test.nunique() > 1:
                print(f"内部测试集 Precision: {precision_score(y_test, y_pred_test):.4f}")
                print(f"内部测试集 Recall: {recall_score(y_test, y_pred_test):.4f}")
                print(f"内部测试集 F1 Score: {f1_score(y_test, y_pred_test):.4f}")
                print(f"内部测试集 AUC-ROC: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
            else:
                print("内部测试集只包含单一类别，跳过 Precision, Recall, F1, AUC 计算。")


            # === 预测 2025 年并进行验证 ===
            print("\n============== 预测 2025 年并进行验证 ==============")
            prob_fenfen_2025_pred = model.predict_proba(X_2025_processed_df)[:, 1][0]
            print(f"2025 年清明假期实际是否为『雨纷纷』: {'是' if y_2025_actual else '否'}")
            print(f"模型预测 2025 年清明假期出现『雨纷纷』的概率 ≈ {prob_fenfen_2025_pred:.1%}")

            # 你可以根据需要添加验证逻辑，例如：
            # 如果 prob_fenfen_2025_pred > 0.5 (或你的阈值)，预测为“是”，然后与 y_2025_actual 对比
            # Compare predicted class vs actual class
            # predicted_class_2025 = (prob_fenfen_2025_pred > 0.5).astype(int)
            # print(f"基于阈值0.5的预测结果: {'是' if predicted_class_2025 else '否'} (实际结果: {'是' if y_2025_actual else '否'})")
            # print(f"预测是否正确: {predicted_class_2025 == y_2025_actual}")


            # === 预测 2026 年 ===
            print("\n============== 预测 2026 年 ==============")
            # Prepare 2026 data - You NEED to replace NaN with your estimated values for 2026 features!
            # This DataFrame MUST have the EXACT SAME COLUMNS as X_train_full
            X_2026 = pd.DataFrame([[2026] + [np.nan] * (len(X_train_full.columns) - 1)], columns=X_train_full.columns)
            # IMPORTANT: Replace np.nan below with your estimated values for 2026 for each feature!
            # X_2026['March_Avg_Temp'] = estimated_2026_march_avg_temp
            # ... fill all columns

            # Transform 2026 data using the SAME fitted imputer and scaler
            if X_2026.isna().any().any():
                print("警告: 2026 年的特征数据包含缺失值。将使用训练集均值填充。") # Imputer handles this, but you should provide real estimates
            X_2026_processed = imputer.transform(X_2026)
            if scaler:
                X_2026_processed = scaler.transform(X_2026_processed)
            X_2026_processed_df = pd.DataFrame(X_2026_processed, columns=X_2026.columns, index=X_2026.index)


            prob_fenfen_2026_pred = model.predict_proba(X_2026_processed_df)[:, 1][0]
            print(f"【预测】2026 年清明假期西安出现『雨纷纷』的概率 (基于逻辑回归模型) ≈ {prob_fenfen_2026_pred:.1%}")


        except Exception as e:
            print(f"\n❌ 模型训练或预测出错: {e}")
            print("请检查数据是否包含足够年份，以及特征提取和处理是否正确。")
            model = None

    else:
        print("\n用于训练模型的历史年份数据不足 或 目标变量类别数不足，无法训练逻辑回归模型。")
        model = None


    # === Save Historical Results ===
    # Save the historical rain fenfen flags (all years)
    out_hist_flag = DATA_DIR / "qingming_rain_fenfen_historical_flags_all_years.csv"
    try:
        out_hist_flag.parent.mkdir(parents=True, exist_ok=True)
        yearly_flag_series_all.to_csv(out_hist_flag, header=True, encoding="utf-8-sig")
        print(f"\n📄 所有年份的历史雨纷纷标记已保存: {out_hist_flag}")
    except Exception as e:
        print(f"\n❌ 保存历史标记失败: {e}")

    # Optional: Save the features and target variable used for modeling training (pre-2025)
    # if 'X_train_full' in locals() and not X_train_full.empty and 'y_train_full' in locals() and not y_train_full.empty:
    #     df_train_model_data = X_train_full.copy()
    #     df_train_model_data['qingming_has_rain_fenfen'] = y_train_full
    #     model_train_data_out = DATA_DIR / "qingming_model_train_data_pre2025.csv"
    #     df_train_model_data.to_csv(model_train_data_out, encoding="utf-8-sig")
    #     print(f"📄 模型训练数据 (特征与标记, 预2025年) 已保存: {model_train_data_out}")


if __name__ == "__main__":
    main()