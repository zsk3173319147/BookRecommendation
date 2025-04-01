import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time
from implicit.als import AlternatingLeastSquares

class MatrixFactorization:
    def __init__(self, factors=100, regularization=0.01, iterations=15, alpha=40):
        """
        初始化矩阵分解模型
        
        参数:
            factors: 潜在因子数量
            regularization: 正则化参数
            iterations: 迭代次数
            alpha: 置信权重参数，控制正样本的权重
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        
    def load_data(self, filepath):
        """加载训练数据"""
        print("加载数据...")
        self.train_data = pd.read_csv(filepath)
        print(f"数据加载完成，共 {len(self.train_data)} 条交互记录")
        
    def preprocess(self):
        """预处理数据并创建用户-物品矩阵"""
        print("预处理数据...")
        # 创建用户和物品的映射字典
        unique_users = self.train_data['user_id'].unique()
        unique_items = self.train_data['item_id'].unique()
        
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        # 将原始ID转换为矩阵索引
        user_indices = [self.user_map[user] for user in self.train_data['user_id']]
        item_indices = [self.item_map[item] for item in self.train_data['item_id']]
        
        # 创建值为1的交互矩阵
        values = np.ones(len(self.train_data))
        user_item_matrix = csr_matrix((values, (user_indices, item_indices)), 
                                      shape=(len(unique_users), len(unique_items)))
        
        # 应用置信权重
        self.user_item_matrix = user_item_matrix * self.alpha
        print(f"预处理完成，用户数: {len(unique_users)}, 物品数: {len(unique_items)}")
        
    def train(self):
        """训练ALS矩阵分解模型"""
        print("开始训练ALS模型...")
        start_time = time.time()
        
        # 初始化ALS模型
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            calculate_training_loss=True
        )
        
        # 训练模型
        self.model.fit(self.user_item_matrix)
        
        print(f"模型训练完成，耗时 {time.time() - start_time:.2f} 秒")
        
    def recommend(self, user_id, top_n=10):
        """为指定用户生成推荐"""
        if user_id not in self.user_map:
            # 用户不在训练集中，使用热门推荐
            return self.recommend_popular_items(top_n)
        
        user_idx = self.user_map[user_id]
        # 获取该用户的交互矩阵行
        user_items = self.user_item_matrix[user_idx]
        
        # 使用模型生成推荐
        recommended = self.model.recommend(user_idx, user_items, N=top_n)
        
        # 处理推荐结果
        try:
            # 从元组中获取物品ID数组
            item_ids = recommended[0]
            # 将numpy数组中的每个ID转换为原始物品ID
            return [self.reverse_item_map[int(item_id)] for item_id in item_ids]
        except Exception as e:
            # 出错时返回热门推荐
            return self.recommend_popular_items(top_n)
    
    def recommend_popular_items(self, top_n=10):
        """推荐热门物品（用于冷启动问题）"""
        # 计算物品流行度（交互次数）
        item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        
        # 获取最流行的物品索引
        top_item_indices = np.argsort(item_popularity)[-top_n:][::-1]
        return [self.reverse_item_map[idx] for idx in top_item_indices]
        
    def generate_recommendations(self, test_users, top_n=10):
        """为测试集用户生成推荐"""
        print(f"为测试用户生成推荐...")
        
        recommendations = {}
        
        for  user_id in enumerate(test_users):
            user_recs = self.recommend(user_id, top_n)
            recommendations[user_id] = user_recs
            
        return recommendations
    
    def save_recommendations(self, recommendations, output_file):
        """保存推荐结果到文件"""
        results = []
        for user_id, items in recommendations.items():
            for item_id in items:
                results.append([user_id, item_id])
                
        output_df = pd.DataFrame(results, columns=['user_id', 'item_id'])
        output_df.to_csv(output_file, index=False)
        print(f"推荐结果已保存到 {output_file}")


if __name__ == "__main__":
    # 初始化模型
    mf = MatrixFactorization(factors=100, regularization=0.01, iterations=15, alpha=40)
    
    # 加载数据
    mf.load_data('D:\Python\project\\bookRecommendation\BookRecommendation\data\\train_dataset.csv')
    mf.preprocess()
    
    # 训练模型
    mf.train()
    
    # 加载测试用户
    test_users = pd.read_csv('D:\Python\project\\bookRecommendation\BookRecommendation\data\\test_dataset.csv')['user_id'].tolist()
    
    # 生成推荐
    recommendations = mf.generate_recommendations(test_users, top_n=1)
    
    # 保存结果
    mf.save_recommendations(recommendations, 'als_recommendations.csv')