import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import time

class CollaborativeFiltering:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None
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
        
        # 创建稀疏矩阵
        values = np.ones(len(self.train_data))
        self.user_item_matrix = csr_matrix((values, (user_indices, item_indices)), 
                                          shape=(len(unique_users), len(unique_items)))
        
        print(f"预处理完成，用户数: {len(unique_users)}, 物品数: {len(unique_items)}")
        
    def compute_item_similarity(self):
        """计算物品间的相似度（基于物品的协同过滤）"""
        print("计算物品相似度...")
        start_time = time.time()
        
        # 转置矩阵以使用cosine_similarity计算物品相似度
        item_sparse = self.user_item_matrix.T.tocsr()
        
        # 由于大型数据集，分批计算相似度
        batch_size = 1000  # 调整批次大小以适应内存
        n_items = item_sparse.shape[0]
        self.item_similarity = np.zeros((n_items, n_items))
        
        for i in range(0, n_items, batch_size):
            end = min(i + batch_size, n_items)
            batch = item_sparse[i:end]
            
            # 计算当前批次与所有物品的相似度
            similarities = cosine_similarity(batch, item_sparse)
            self.item_similarity[i:end] = similarities
            
            if (i + batch_size) % 2000 == 0 or end == n_items:
                print(f"已计算 {end}/{n_items} 个物品的相似度")
        
        print(f"物品相似度计算完成，耗时 {time.time() - start_time:.2f} 秒")
        
    def compute_user_similarity(self):
        """计算用户间的相似度（基于用户的协同过滤）"""
        print("计算用户相似度...")
        start_time = time.time()
        
        # 使用稀疏矩阵计算用户相似度
        user_sparse = self.user_item_matrix.tocsr()
        
        # 分批计算以节省内存
        batch_size = 1000
        n_users = user_sparse.shape[0]
        self.user_similarity = np.zeros((n_users, n_users))
        
        for i in range(0, n_users, batch_size):
            end = min(i + batch_size, n_users)
            batch = user_sparse[i:end]
            
            # 计算当前批次与所有用户的相似度
            similarities = cosine_similarity(batch, user_sparse)
            self.user_similarity[i:end] = similarities
            
            if (i + batch_size) % 2000 == 0 or end == n_users:
                print(f"已计算 {end}/{n_users} 个用户的相似度")
        
        print(f"用户相似度计算完成，耗时 {time.time() - start_time:.2f} 秒")
        
    def recommend_item_based(self, user_id, top_n=10):
        """基于物品的协同过滤推荐"""
        if user_id not in self.user_map:
            print(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_map[user_id]
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        interacted_items = np.where(user_interactions > 0)[0]
        
        # 没有交互记录的用户
        if len(interacted_items) == 0:
            return []
            
        # 计算推荐得分
        scores = np.zeros(self.user_item_matrix.shape[1])
        for item_idx in interacted_items:
            scores += self.item_similarity[item_idx]
            
        # 过滤掉已交互物品
        scores[interacted_items] = -1
        
        # 获取得分最高的N个物品
        top_item_indices = np.argsort(scores)[-top_n:][::-1]
        recommendations = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return recommendations
    
    def recommend_user_based(self, user_id, top_n=10):
        """基于用户的协同过滤推荐"""
        if user_id not in self.user_map:
            print(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_map[user_id]
        user_similarity_scores = self.user_similarity[user_idx]
        
        # 获取目标用户的交互记录
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        interacted_items = set(np.where(user_interactions > 0)[0])
        
        # 计算推荐得分
        scores = np.zeros(self.user_item_matrix.shape[1])
        total_similarity = np.zeros(self.user_item_matrix.shape[1])
        
        for other_user_idx in range(len(self.user_similarity)):
            # 跳过目标用户自己
            if other_user_idx == user_idx:
                continue
                
            similarity = user_similarity_scores[other_user_idx]
            
            # 忽略相似度极低的用户
            if similarity <= 0:
                continue
                   
            other_interactions = self.user_item_matrix[other_user_idx].toarray().flatten()
            for item_idx in np.where(other_interactions > 0)[0]:
                scores[item_idx] += similarity
                total_similarity[item_idx] += similarity
                
        # 避免除零错误
        total_similarity[total_similarity == 0] = 1
        scores = scores / total_similarity
        
        # 过滤掉已交互物品
        for item_idx in interacted_items:
            scores[item_idx] = -1
            
        # 获取得分最高的N个物品
        top_item_indices = np.argsort(scores)[-top_n:][::-1]
        recommendations = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return recommendations
        
    def generate_recommendations(self, test_users, method='item', top_n=10):
        """为测试集用户生成推荐"""
        print(f"使用{method}方法为测试用户生成推荐...")
        
        recommendations = {}
        recommend_func = self.recommend_item_based if method == 'item' else self.recommend_user_based
        
        for user_id in test_users:
            user_recs = recommend_func(user_id, top_n)
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
   # 初始化推荐系统
    cf = CollaborativeFiltering()
    
    # 加载数据
    cf.load_data('D:\Python\project\\bookRecommendation\BookRecommendation\data\\train_dataset.csv')
    cf.preprocess()
    
    # 计算相似度
    cf.compute_item_similarity()  # 基于物品的CF
    
    # 加载测试用户
    test_users = pd.read_csv('D:\Python\project\\bookRecommendation\BookRecommendation\data\\test_dataset.csv')['user_id'].tolist()
    
    # 生成物品推荐
    item_recommendations = cf.generate_recommendations(test_users, method='item', top_n=1)
    
    # # 如果想使用基于用户的推荐，需要先计算用户相似度
    # cf.compute_user_similarity()  # 添加这行
    # user_recommendations = cf.generate_recommendations(test_users, method='user', top_n=1)
    
    
    # 保存结果
    cf.save_recommendations(item_recommendations, 'item_based_recommendations.csv')