import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
result_dir = 'analysis_results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 创建摘要文件
summary_file = os.path.join(result_dir, 'analysis_summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("===== 数据分析结果摘要 =====\n\n")

# 函数：将统计结果添加到摘要文件
def add_to_summary(content):
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(content + "\n")

# 函数：保存图表并添加描述到摘要
def save_figure(filename, description):
    plt.savefig(os.path.join(result_dir, filename), dpi=300, bbox_inches='tight')
    add_to_summary(f"图表: {filename} - {description}")
    plt.close()  # 关闭图表释放内存

# ===== 开始数据分析 =====

# 加载数据
train_data = pd.read_csv('D:\\Python\project\\bookRecommendation\\data\\train_dataset.csv')
test_users = pd.read_csv('D:\\Python\project\\bookRecommendation\\data\\test_dataset.csv')

# 基本信息查看
print("训练集信息:")
print(train_data.info())
print("\n前5行数据:")
print(train_data.head())

# 检查是否有缺失值
print("\n缺失值统计:")
print(train_data.isnull().sum())

# 统计基本信息
n_users = train_data['user_id'].nunique()
n_items = train_data['item_id'].nunique()
n_interactions = len(train_data)

# 计算稀疏度
sparsity = 1 - (n_interactions / (n_users * n_items))
print(f"用户数: {n_users}, 物品数: {n_items}, 交互数: {n_interactions}")
print(f"交互矩阵稀疏度: {sparsity:.4f}")

# 记录基础数据统计信息
basic_info = f"""
基础数据统计:
- 用户数: {n_users}
- 物品数: {n_items}
- 交互数: {n_interactions}
- 交互矩阵稀疏度: {sparsity:.6f} ({sparsity*100:.2f}%)
"""
add_to_summary(basic_info)

# 检查测试集中的用户是否在训练集中出现过
test_users_set = set(test_users['user_id'])
train_users_set = set(train_data['user_id'])
cold_start_users = test_users_set - train_users_set
print(f"测试集中的冷启动用户数量: {len(cold_start_users)}")

# 用户行为分布
user_freq = train_data['user_id'].value_counts()
user_freq_stats = f"""
用户交互次数统计:
- 平均每用户交互次数: {user_freq.mean():.2f}
- 用户交互次数中位数: {user_freq.median()}
- 最活跃用户交互次数: {user_freq.max()}
- 最不活跃用户交互次数: {user_freq.min()}
"""
add_to_summary(user_freq_stats)

# 绘制并保存用户交互分布图
plt.figure(figsize=(10, 5))
sns.histplot(user_freq, bins=50)
plt.title('用户交互次数分布')
plt.xlabel('交互次数')
plt.ylabel('用户数')
save_figure('user_frequency_distribution.png', '用户交互次数的分布情况')

# 物品流行度分布
item_freq = train_data['item_id'].value_counts()
item_freq_stats = f"""
物品流行度统计:
- 平均每物品被交互次数: {item_freq.mean():.2f}
- 物品被交互次数中位数: {item_freq.median()}
- 最流行物品交互次数: {item_freq.max()}
- 最不流行物品交互次数: {item_freq.min()}
"""
add_to_summary(item_freq_stats)

# 绘制并保存物品流行度分布图
plt.figure(figsize=(10, 5))
sns.histplot(item_freq, bins=50)
plt.title('物品被交互次数分布')
plt.xlabel('被交互次数')
plt.ylabel('物品数')
save_figure('item_frequency_distribution.png', '物品被交互次数的分布情况')

# 再添加其他分析图表和统计...
# 例如交互矩阵可视化、热门物品分析等

print(f"所有分析结果已保存到 {result_dir} 目录")