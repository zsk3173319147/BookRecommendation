import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

class RecommenderEvaluator:
    def __init__(self, train_data_path, test_users_path=None, validation_ratio=0.2, 
                 random_state=42, force_split=False):
        """
        初始化推荐系统评估器
        
        参数:
            train_data_path: 训练数据路径
            test_users_path: 测试用户路径
            validation_ratio: 验证集占比
            random_state: 随机种子(确保可重复性)
            force_split: 是否强制重新划分数据，即使已存在划分好的数据
        """
        self.train_data_path = train_data_path
        self.test_users_path = test_users_path
        self.validation_ratio = validation_ratio
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
        # 设置数据文件路径
        self.train_split_path = 'train_for_validation.csv'
        self.validation_path = 'validation_set.csv'
        
        # 加载并划分数据
        self.load_and_split_data(force_split)
        
    def load_and_split_data(self, force_split=False):
        """加载数据并划分训练/验证集(仅在需要时)"""
        # 检查是否已存在划分好的数据
        if not force_split and os.path.exists(self.train_split_path) and os.path.exists(self.validation_path):
            print("加载已划分的数据集...")
            self.train_set = pd.read_csv(self.train_split_path)
            self.validation_set = pd.read_csv(self.validation_path)
            print(f"训练集: {len(self.train_set)} 条交互，验证集: {len(self.validation_set)} 条交互")
            return
        
        # 需要重新划分数据
        print("加载原始训练数据...")
        self.train_data = pd.read_csv(self.train_data_path)
        
        # 测试用户(如果提供)
        if self.test_users_path:
            test_users_df = pd.read_csv(self.test_users_path)
            self.test_users = test_users_df['user_id'].unique()
            print(f"测试用户: {len(self.test_users)} 个")
        else:
            # 如果未提供测试用户，随机选择一部分
            self.test_users = np.random.choice(
                self.train_data['user_id'].unique(), 
                size=min(1000, self.train_data['user_id'].nunique()),
                replace=False
            )
        
        # 创建验证集
        self.create_validation_set()
        
    def create_validation_set(self):
        """从训练数据创建验证集 - 使用固定随机种子"""
        print("创建验证集...")
        
        # 初始化结果列表
        train_interactions = []
        valid_interactions = []
        
        # 对测试用户进行特殊处理
        test_user_data = self.train_data[self.train_data['user_id'].isin(self.test_users)]
        for  group in test_user_data.groupby('user_id'):
            # 如果用户交互太少，则全部作为训练数据
            if len(group) <= 2:
                train_interactions.append(group)
                continue
                
            # 使用固定随机种子划分
            user_train, user_valid = train_test_split(
                group, test_size=self.validation_ratio, 
                random_state=self.random_state
            )
            
            train_interactions.append(user_train)
            valid_interactions.append(user_valid)
        
        # 非测试用户数据全部用于训练
        non_test_data = self.train_data[~self.train_data['user_id'].isin(self.test_users)]
        train_interactions.append(non_test_data)
        
        # 合并结果
        self.train_set = pd.concat(train_interactions, ignore_index=True)
        self.validation_set = pd.concat(valid_interactions, ignore_index=True) if valid_interactions else pd.DataFrame(columns=self.train_data.columns)
        
        print(f"验证集: {len(self.validation_set)} 条交互")
        print(f"训练集: {len(self.train_set)} 条交互")
        
        # 保存划分结果供所有模型使用
        self.train_set.to_csv(self.train_split_path, index=False)
        self.validation_set.to_csv(self.validation_path, index=False)
        print(f"数据集已保存到 {self.train_split_path} 和 {self.validation_path}")
        
    def add_model(self, model_name, recommend_func):
        """添加一个要评估的模型"""
        self.models[model_name] = recommend_func
        print(f"添加模型: {model_name}")
    
    def evaluate(self, top_n=10):
        """评估所有模型的性能"""
        if not self.models:
            print("没有模型可评估。请先使用add_model()添加模型。")
            return
            
        print(f"开始评估模型 (Top-{top_n})...")
        
        # 创建用户已交互物品映射
        train_user_items = {}
        for _, row in self.train_set.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            if user_id not in train_user_items:
                train_user_items[user_id] = set()
            train_user_items[user_id].add(item_id)
        
        # 验证集映射
        valid_user_items = {}
        for _, row in self.validation_set.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            if user_id not in valid_user_items:
                valid_user_items[user_id] = set()
            valid_user_items[user_id].add(item_id)
        
        # 评估每个模型
        for model_name, recommend_func in self.models.items():
            print(f"评估模型: {model_name}")
            start_time = time.time()
            
            # 初始化指标
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0
            ndcg_sum = 0
            hit_sum = 0
            user_count = 0
            
            # 对验证集中的每个用户进行评估
            valid_users = list(valid_user_items.keys())
            for i, user_id in enumerate(valid_users):
                # 获取用户已交互的物品(训练集)
                user_train_items = train_user_items.get(user_id, set())
                
                # 获取用户在验证集中的物品(作为评估标准)
                user_valid_items = valid_user_items[user_id]
                
                # 获取推荐结果
                try:
                    recommendations = recommend_func(user_id, top_n)
                    
                    # 过滤掉训练集中已交互的物品(这是重要的步骤!)
                    filtered_recs = [item for item in recommendations if item not in user_train_items]
                    
                    # 如果过滤后没有推荐，跳过此用户
                    if not filtered_recs:
                        continue
                    
                    # 计算命中的物品
                    hits = set(filtered_recs) & user_valid_items
                    
                    # 计算各项指标
                    precision = len(hits) / len(filtered_recs)
                    recall = len(hits) / len(user_valid_items) if user_valid_items else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                    
                    # 计算NDCG(考虑排序质量)
                    dcg = 0
                    for j, item in enumerate(filtered_recs):
                        if item in user_valid_items:
                            dcg += 1 / np.log2(j + 2)  # j+2因为j从0开始
                    
                    # 理想DCG
                    idcg = sum(1 / np.log2(j + 2) for j in range(min(len(user_valid_items), len(filtered_recs))))
                    ndcg = dcg / idcg if idcg > 0 else 0
                    
                    # 累加指标
                    precision_sum += precision
                    recall_sum += recall
                    f1_sum += f1
                    ndcg_sum += ndcg
                    hit_sum += 1 if hits else 0
                    user_count += 1
                    
                except Exception as e:
                    print(f"用户 {user_id} 评估出错: {str(e)}")
                
            
            # 计算平均指标
            avg_precision = precision_sum / user_count if user_count > 0 else 0
            avg_recall = recall_sum / user_count if user_count > 0 else 0
            avg_f1 = f1_sum / user_count if user_count > 0 else 0
            avg_ndcg = ndcg_sum / user_count if user_count > 0 else 0
            hit_rate = hit_sum / user_count if user_count > 0 else 0
            
            # 保存结果
            self.results[model_name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'ndcg': avg_ndcg,
                'hit_rate': hit_rate,
                'evaluated_users': user_count,
                'total_valid_users': len(valid_users),
                'evaluation_time': time.time() - start_time
            }
            
            print(f"{model_name} 评估完成，有效评估用户: {user_count}/{len(valid_users)}")
            print(f"Precision@{top_n}: {avg_precision:.4f}")
            print(f"Recall@{top_n}: {avg_recall:.4f}")
            print(f"F1@{top_n}: {avg_f1:.4f}")
            print(f"NDCG@{top_n}: {avg_ndcg:.4f}")
            print(f"Hit Rate@{top_n}: {hit_rate:.4f}")
            print("----------------------------------")
    
    def visualize_results(self, save_path='evaluation_results'):
        """可视化评估结果 - 便于比较不同模型"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        models = list(self.results.keys())
        metrics = ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']
        
        # 创建比较图表
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            values = [self.results[model][metric] for model in models]
            sns.barplot(x=models, y=values)
            plt.title(f'{metric.title()}')
            plt.ylabel(f'{metric.title()} Score')
            plt.ylim(0, max(values) * 1.2)
            
            # 添加数值标签
            for j, v in enumerate(values):
                plt.text(j, v + 0.02, f"{v:.4f}", ha='center')
                
        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300)
        
        # 保存结果表
        results_df = pd.DataFrame({model: {metric: self.results[model][metric] for metric in metrics} 
                                  for model in models}).T
        results_df.to_csv(f'{save_path}/results_summary.csv')
        
        print(f"评估结果可视化已保存到 {save_path} 目录")


if __name__ == "__main__":
    # 初始化评估器
    evaluator = RecommenderEvaluator(
        train_data_path='D:\Python\project\\bookRecommendation\BookRecommendation\data\\train_dataset.csv',
        test_users_path='D:\Python\project\\bookRecommendation\BookRecommendation\data\\test_dataset.csv'
    )
    
    # 添加要评估的模型
    # from CF import CollaborativeFiltering
    # from MF import MatrixFactorization
    from NCF import NCF

    # 初始化并训练模型
    # cf = CollaborativeFiltering()
    # cf.load_data('train_for_validation.csv')  # 使用划分后的训练集
    # cf.preprocess()
    # cf.compute_item_similarity()
    
    # mf = MatrixFactorization()
    # mf.load_data('train_for_validation.csv')
    # mf.preprocess()
    # mf.train()

    ncf = NCF(
        embedding_dim=16, 
        layers=[64, 32], 
        learning_rate=0.005, 
        epochs=5,  # 减少训练轮数以加快评估
        batch_size=1024,
        neg_ratio=2,
        early_stopping_patience=2,
        sample_ratio=0.3  # 使用30%数据进行快速评估
    )
    ncf.load_data(r'D:\\Python\\project\bookRecommendation\BookRecommendation\data\\train_dataset.csv')
    ncf.preprocess()
    ncf.train()
      # 加载测试用户
    test_users = pd.read_csv(
        r'D:\\Python\\project\bookRecommendation\BookRecommendation\data\\test_dataset.csv'
    )['user_id'].tolist()

    ncf_recommendations = ncf.generate_recommendations(test_users, top_n=10)
    
    # 创建包装函数将推荐结果字典转换为可调用函数
    def ncf_recommender(user_id, top_n=10):
        """包装NCF推荐结果为函数"""
        if user_id in ncf_recommendations:
            return ncf_recommendations[user_id][:top_n]
        return []  # 用户不存在时返回空列表

    # 正确添加模型到评估器
    evaluator.add_model('NCF', ncf_recommender)


    # 评估模型
    evaluator.evaluate(top_n=10)
    
    # 可视化结果
    evaluator.visualize_results()