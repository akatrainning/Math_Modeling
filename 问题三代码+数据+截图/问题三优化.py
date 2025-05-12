import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from matplotlib.font_manager import FontProperties
import os
from itertools import permutations

# 如果在非中文环境中运行代码，尝试设置中文字体
try:
    # 尝试查找系统中的中文字体
    font_path = None
    if os.name == 'nt':  # Windows
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    elif os.name == 'posix':  # Linux/macOS
        if os.path.exists('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
            font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        elif os.path.exists('/System/Library/Fonts/PingFang.ttc'):  # macOS
            font_path = '/System/Library/Fonts/PingFang.ttc'
    if font_path and os.path.exists(font_path):
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    else:
        print("警告：未找到中文字体文件，图表中的中文可能无法正确显示")
        chinese_font = None
except:
    print("警告：设置中文字体失败，图表中的中文可能无法正确显示")
    chinese_font = None

# 1. 读取下雨状态数据
rain_data = pd.read_excel('问题三下雨状态.xlsx')
full_rain_data = rain_data

# 2. 花卉的开放时间和花期
flower_opening = {
'xinghua': {  # 杏花（花期集中在3月中下旬，北方略晚）
    'East_China': {'start': '03.20', 'duration': 16},       # 花期到4.05 ✔（涵盖清明）
    'Central_China': {'start': '03.15', 'duration': 14},    # 花期到3.29 ✘（略早，保留不动）
    'Southwest_China': {'start': '02.25', 'duration': 20},  # 花期到3.17 ✘（偏早，略调整）
    'Northwest_China': {'start': '03.25', 'duration': 18},  # 花期到4.12 ✔
    'North_China': {'start': '03.28', 'duration': 18}       # 花期到4.14 ✔
},

'youcaohua': {  # 油菜花（主力赏花季节，需覆盖清明）
    'East_China': {'start': '03.10', 'duration': 30},       # 花期到4.09 ✔
    'Central_China': {'start': '03.05', 'duration': 25},    # 花期到3.30 ✘（略早，合理保留）
    'Southwest_China': {'start': '02.20', 'duration': 35},  # 花期到3.26 ✘（合理，因气温高）
    'Northwest_China': {'start': '03.25', 'duration': 20},  # 花期到4.14 ✔
    'North_China': {'start': '03.28', 'duration': 18}       # 花期到4.14 ✔
}



    }


# 花卉的中文名称
flower_chinese_names = {
    'xinghua': '杏花',
    'youcaohua': '油菜花',

}

# 3. 定义城市信息
cities_info = [
    {'name': 'XiAn', 'coords': [108.951777, 34.257678], 'province': '陕西省', 'region': 'Northwest_China',
     'chinese_name': '西安'},
    {'name': 'Hangzhou', 'coords': [120.138874, 30.259739], 'province': '浙江省', 'region': 'East_China',
     'chinese_name': '杭州'},
    {'name': 'Wuhan', 'coords': [114.311156, 30.571845], 'province': '湖北省', 'region': 'Central_China',
     'chinese_name': '武汉'},
    {'name': 'Bijie', 'coords': [105.328528, 27.008451], 'province': '贵州省', 'region': 'Southwest_China',
     'chinese_name': '毕节'},
    {'name': 'Turpan', 'coords': [89.50739, 42.983802], 'province': '新疆维吾尔自治区', 'region': 'Northwest_China',
     'chinese_name': '吐鲁番'},
    {'name': 'Luoyang', 'coords': [113.648723, 34.756182], 'province': '河南省', 'region': 'North_China',
     'chinese_name': '洛阳'},
    {'name': 'Wuyuan', 'coords': [117.52, 29.2], 'province': '江西省', 'region': 'Central_China',
     'chinese_name': '婺源'}
]

# 4. 解析开始日期并计算开放结束日期
flower_open_dates = {}

for flower, regions in flower_opening.items():
    flower_open_dates[flower] = {}
    for city_info in cities_info:
        region = city_info['region']
        if region in regions:
            start_date_str = regions[region]['start']
            # 解析日期字符串为datetime对象
            month, day = map(int, start_date_str.split('.'))
            start_date = datetime(2026, month, day)
            # 计算结束日期
            duration = regions[region]['duration']
            end_date = start_date + timedelta(days=duration - 1)

            flower_open_dates[flower][city_info['name']] = {'start': start_date, 'end': end_date}

# 5. 清明节假期日期（2026年4月4日至6日）
holiday_start = datetime(2026, 4, 4)
holiday_end = datetime(2026, 4, 6)
holiday_dates = [holiday_start + timedelta(days=i) for i in range((holiday_end - holiday_start).days + 1)]

# 6. 处理下雨数据
rain_status = {}
for city_info in cities_info:
    city_name = city_info['name']

    # 获取对应城市的下雨状态数据
    city_rain_data = full_rain_data[city_name].values

    # 计算假期期间的下雨情况
    rain_status[city_name] = city_rain_data

    # 计算下雨小时数和百分比
    rain_hours = np.sum(city_rain_data)
    rain_percentage = rain_hours / 72 * 100

    # 输出城市的下雨状态概述
    print(f"{city_info['chinese_name']}（{city_name}）下雨状态：共{rain_hours}小时下雨，占总时间的{rain_percentage:.1f}%")

# 7. 计算城市间的距离矩阵
num_cities = len(cities_info)
distance_matrix = np.zeros((num_cities, num_cities))


# Haversine公式计算两点间的球面距离
def haversine_distance(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球半径，单位：公里
    return c * r

for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            # 使用Haversine公式计算距离
            lat1, lon1 = cities_info[i]['coords'][1], cities_info[i]['coords'][0]
            lat2, lon2 = cities_info[j]['coords'][1], cities_info[j]['coords'][0]
            distance_matrix[i, j] = haversine_distance(lat1, lon1, lat2, lon2)

# 显示距离矩阵
print("\n城市间距离矩阵（单位：公里）:")
city_names_chinese = [city['chinese_name'] for city in cities_info]
distance_df = pd.DataFrame(distance_matrix, index=city_names_chinese, columns=city_names_chinese)
print(distance_df)

# 8. 计算旅行时间矩阵（单位：小时）
train_speed = 250  # 高铁速度，单位：公里/小时
boarding_time = 0.5  # 上下车时间，单位：小时
station_to_spot_time = 1  # 从车站到景点的时间，单位：小时
min_visit_time = 4  # 参观景点的最少时间，单位：小时

# 城市间旅行时间 = 距离/速度 + 上下车时间
travel_time_matrix = distance_matrix / train_speed + boarding_time * 2

# 显示旅行时间矩阵
print("\n城市间旅行时间矩阵（单位：小时）:")
travel_time_df = pd.DataFrame(travel_time_matrix, index=city_names_chinese, columns=city_names_chinese)
print(travel_time_df)

# 9. 计算每个城市的花卉观赏评分
# 总可用时间（3天 = 72小时）
total_time = 72

# 创建一个字典来存储每个城市的花卉观赏评分
flower_scores = {}
for i, city_info in enumerate(cities_info):
    city_name = city_info['name']
    city_chinese_name = city_info['chinese_name']
    flower_scores[city_name] = 0

    # 检查哪些花卉在假期期间开放
    blooming_flowers = []
    for flower_name, cities_data in flower_open_dates.items():
        if city_name in cities_data:
            bloom_start = cities_data[city_name]['start']
            bloom_end = cities_data[city_name]['end']

            # 检查假期是否与开花期重叠
            if (holiday_dates[0] <= bloom_end and holiday_dates[-1] >= bloom_start):
                # 计算假期中花卉开放的天数
                overlap_start = max(holiday_dates[0], bloom_start)
                overlap_end = min(holiday_dates[-1], bloom_end)
                overlap_days = (overlap_end - overlap_start).days + 1

                # 根据开花质量添加评分（假设开花期中间是最佳状态）
                bloom_duration = (bloom_end - bloom_start).days + 1
                bloom_peak = bloom_start + timedelta(days=bloom_duration // 2)

                # 找出距离花期高峰最近的假期日期
                distance_from_peak = min([abs((date - bloom_peak).days) for date in holiday_dates])
                peak_score = max(0, 1 - distance_from_peak / (bloom_duration / 2))

                # 添加到城市的花卉评分
                flower_scores[city_name] += overlap_days * peak_score

                # 添加到开花列表
                blooming_flowers.append(flower_chinese_names[flower_name])

    # 根据降雨概率调整评分
    rain_hours = np.sum(rain_status[city_name])
    rain_percentage = rain_hours / 72
    weather_factor = 1 - rain_percentage * 0.8  # 下雨降低评分（100%下雨时降低80%）

    flower_scores[city_name] *= weather_factor

    # 输出城市的花卉评分和开花情况
    blooming_str = '、'.join(blooming_flowers) if blooming_flowers else '无'

    print(
        f"{city_chinese_name}（{city_name}）花卉评分: {flower_scores[city_name]:.2f}，开放花卉: {blooming_str}，降雨百分比: {rain_percentage * 100:.1f}%")

# 10. 使用贪心算法优化旅行计划
# 从评分最高的城市开始，然后如果时间允许，移动到下一个最佳城市

# 将评分转换为数组以进行排序
city_scores = [(i, flower_scores[city_info['name']]) for i, city_info in enumerate(cities_info)]
# 按评分降序排序城市（评分越高越好）
sorted_city_scores = sorted(city_scores, key=lambda x: x[1], reverse=True)
sorted_cities_idx = [idx for idx, _ in sorted_city_scores]
sorted_scores = [score for _, score in sorted_city_scores]

# 展示排序后的城市评分
print("\n城市花卉观赏评分排名:")
for i, (idx, score) in enumerate(sorted_city_scores):
    print(f"{i + 1}. {cities_info[idx]['chinese_name']}（{cities_info[idx]['name']}）: {score:.2f}")

# 初始化行程规划
# 尝试从前2个评分最高的城市作为起点，选择最优的路线
best_itinerary = []
best_score = 0
best_time_used = 0

for start_idx_pos in range(min(2, num_cities)):
    # 初始化行程
    itinerary = []
    current_city_idx = sorted_cities_idx[start_idx_pos]
    remaining_time = total_time
    total_score = sorted_scores[start_idx_pos]
    time_used = 0
    visited = [False] * num_cities

    # 添加第一个城市到行程
    visit_time = min_visit_time + 2 * station_to_spot_time  # 参观 + 往返车站
    itinerary.append({
        'type': 'visit',
        'city_idx': current_city_idx,
        'city': cities_info[current_city_idx]['name'],
        'chinese_name': cities_info[current_city_idx]['chinese_name'],
        'time_spent': visit_time
    })
    visited[current_city_idx] = True
    remaining_time -= visit_time
    time_used += visit_time

    # 尝试访问其他城市，按评分排名依次尝试
    while remaining_time > 0:
        best_next_city = -1
        best_next_score_per_time = 0
        best_travel_time = 0

        # 寻找下一个最佳城市
        for i in range(num_cities):
            if not visited[i]:
                next_city_idx = i

                # 计算所需时间
                travel_time = travel_time_matrix[current_city_idx, next_city_idx]
                visit_time = min_visit_time + 2 * station_to_spot_time
                total_needed = travel_time + visit_time

                # 如果时间足够，且评分/时间比率更好
                if total_needed <= remaining_time:
                    score_per_time = flower_scores[cities_info[next_city_idx]['name']] / total_needed
                    if score_per_time > best_next_score_per_time:
                        best_next_city = next_city_idx
                        best_next_score_per_time = score_per_time
                        best_travel_time = travel_time

        # 如果找到了下一个城市
        if best_next_city >= 0:
            # 添加旅行到行程
            itinerary.append({
                'type': 'travel',
                'from': current_city_idx,
                'to': best_next_city,
                'from_city': cities_info[current_city_idx]['name'],
                'to_city': cities_info[best_next_city]['name'],
                'from_chinese': cities_info[current_city_idx]['chinese_name'],
                'to_chinese': cities_info[best_next_city]['chinese_name'],
                'time': best_travel_time
            })

            # 添加访问到行程
            visit_time = min_visit_time + 2 * station_to_spot_time
            itinerary.append({
                'type': 'visit',
                'city_idx': best_next_city,
                'city': cities_info[best_next_city]['name'],
                'chinese_name': cities_info[best_next_city]['chinese_name'],
                'time_spent': visit_time
            })

            # 更新状态
            current_city_idx = best_next_city
            visited[best_next_city] = True
            remaining_time -= (best_travel_time + visit_time)
            time_used += best_travel_time + visit_time
            total_score += flower_scores[cities_info[best_next_city]['name']]
        else:
            # 没有更多城市可以访问
            break

    # 检查这个行程是否比当前最佳行程更好
    if total_score > best_score:
        best_itinerary = itinerary
        best_score = total_score
        best_time_used = time_used

# 11. 输出最终行程
print("\n最优化的清明节行程（2026年4月4日-6日）:")
print(f"总时长: 3天 (72小时)\n")
print("\n 旅行贴士")
print("1. 交通安排：高铁是本次旅行的主要交通工具，请提前30分钟到达车站。")
print("2. 赏花时机：清明时节花期易受气温影响，建议出发前再次确认花期情况。")
print("3. 防雨准备：清明时节，请务必携带雨具。")
print("4. 摄影建议：阴天下的花卉色彩更加饱满，是拍摄的好时机。")
print("5. 行程灵活：此行程仅供参考，可根据实际情况适当调整。")

time_used = 0
current_day = 1
current_hour = 8  # 从早上8点开始

for item in best_itinerary:
    if item['type'] == 'visit':
        # 这是城市参观
        hours = int(current_hour)
        minutes = round((current_hour - hours) * 60)
        time_str = f"{hours:02d}:{minutes:02d}"

        print(f"第{current_day}天, {time_str} - 参观{item['chinese_name']}")

        # 更新时间
        current_hour += item['time_spent']
        time_used += item['time_spent']

        # 显示正在开放的花卉
        blooming_flowers = []
        for flower_name, cities_data in flower_open_dates.items():
            if item['city'] in cities_data:
                bloom_start = cities_data[item['city']]['start']
                bloom_end = cities_data[item['city']]['end']

                if (holiday_dates[0] <= bloom_end and holiday_dates[-1] >= bloom_start):
                    blooming_flowers.append(flower_chinese_names[flower_name])

        if not blooming_flowers:
            print("  开放的花卉: 无")
        else:
            print(f"  开放的花卉: {', '.join(blooming_flowers)}")

        # 雨天状况
        rain_hours = np.sum(rain_status[item['city']])
        rain_percentage = rain_hours / 72 * 100
        print(f"  天气状况: 假期期间有{rain_percentage:.1f}%的时间可能下雨")

    else:
        # 这是城市间旅行
        hours = int(current_hour)
        minutes = round((current_hour - hours) * 60)
        time_str = f"{hours:02d}:{minutes:02d}"

        print(
            f"第{current_day}天, {time_str} - 从{item['from_chinese']}前往{item['to_chinese']}（{item['time']:.1f}小时）")

        # 更新时间
        current_hour += item['time']
        time_used += item['time']

    # 检查是否需要进入下一天
    if current_hour >= 22:  # 晚上10点结束当天行程
        current_day += 1
        current_hour = 8  # 新的一天从早上8点开始
        print("")

print(f"\n总计用时: {time_used:.1f}小时（共72小时）")
print(f"剩余时间: {total_time - time_used:.1f}小时")

# 12. 生成行程可视化
plt.figure(figsize=(16, 8), num='清明节赏花旅行路线')

# 创建中国地图（简化版）
plt.subplot(1, 2, 1)

# 绘制城市点
for city_info in cities_info:
    plt.plot(city_info['coords'][0], city_info['coords'][1], 'o', markersize=8, markerfacecolor='blue')
    plt.text(city_info['coords'][0], city_info['coords'][1] + 0.5, city_info['chinese_name'],
             fontsize=10, horizontalalignment='center', fontproperties=chinese_font if chinese_font else None)

# 绘制旅行路线
for item in best_itinerary:
    if item['type'] == 'travel':
        from_coords = cities_info[item['from']]['coords']
        to_coords = cities_info[item['to']]['coords']
        plt.plot([from_coords[0], to_coords[0]], [from_coords[1], to_coords[1]], '->', linewidth=2)

plt.title('清明节赏花旅行路线', fontproperties=chinese_font if chinese_font else None)
plt.xlabel('经度', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('纬度', fontproperties=chinese_font if chinese_font else None)
plt.grid(True)

# 绘制行程甘特图
plt.subplot(1, 2, 2)

# 设置颜色
visit_color = [0.3, 0.6, 0.9]
travel_color = [0.9, 0.6, 0.3]

# 绘制甘特图
for i, item in enumerate(best_itinerary):
    if item['type'] == 'visit':
        plt.bar(i, item['time_spent'], 0.5, color=visit_color)
        plt.text(i, item['time_spent'] + 0.2, f"{item['chinese_name']} ({item['time_spent']}h)",
                 horizontalalignment='center', verticalalignment='bottom', rotation=90,
                 fontproperties=chinese_font if chinese_font else None)
    else:
        plt.bar(i, item['time'], 0.5, color=travel_color)
        plt.text(i, item['time'] + 0.2, f"前往{item['to_chinese']} ({item['time']:.1f}h)",
                 horizontalalignment='center', verticalalignment='bottom', rotation=90,
                 fontproperties=chinese_font if chinese_font else None)

plt.title('清明节旅行时间安排', fontproperties=chinese_font if chinese_font else None)
plt.ylabel('时间（小时）', fontproperties=chinese_font if chinese_font else None)
plt.grid(True)

# 添加图例
plt.legend(['参观景点', '城市间旅行'], loc='upper right', prop=chinese_font if chinese_font else None)

# 设置坐标轴
plt.xlim(0, len(best_itinerary) + 1)
plt.xticks([])

plt.tight_layout()
plt.savefig('清明节赏花旅行规划.png')
plt.show()


# 调用函数并输出结果
route = generate_routes(cities_info, rain_status, travel_time_matrix, holiday_dates)

