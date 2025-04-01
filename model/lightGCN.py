import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import time

class LightGCNModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, device):
        super(LightGCNModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 初始化权重
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, adj_matrix):
        # 获取初始嵌入
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        # 存储所有层的嵌入
        embs = [all_emb]
        
        # 消息传递
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        
        # 将所有层的嵌入拼接并求平均
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        # 分离用户和物品的最终嵌入
        users_emb_final, items_emb_final = torch.split(light_out, [self.n_users, self.n_items])
        
        return users_emb_final, items_emb_final
    
    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        # 获取用户、正样本和负样本的嵌入
        user_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # 计算正样本和负样本的预测分数
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        # BPR损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2正则化
        regularizer = (torch.norm(user_emb) ** 2 + 
                       torch.norm(pos_emb) ** 2 + 
                       torch.norm(neg_emb) ** 2) / len(users)
        
        return loss, regularizer

class BPRDataset(Dataset):
    def __init__(self, user_list, pos_item_list, neg_item_list):
        self.user_list = user_list
        self.pos_item_list = pos_item_list
        self.neg_item_list = neg_item_list
        
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, idx):
        return self.user_list[idx], self.pos_item_list[idx], self.neg_item_list[idx]

class LightGCN:
    def __init__(self, embedding_dim=64, n_layers=3, learning_rate=0.001, 
                 epochs=10, batch_size=1024, weight_decay=1e-4, 
                 device=None, neg_ratio=1, early_stopping_patience=2,
                 sample_ratio=1.0):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neg_ratio = neg_ratio
        self.early_stopping_patience = early_stopping_patience
        self.sample_ratio = sample_ratio
        self.train_losses = []
        
    def load_data(self, filepath):
        """加载数据"""
        print("加载数据...")
        self.train_data = pd.read_csv(filepath)
        print(f"数据加载完成，共 {len(self.train_data)} 条交互记录")
    
    def preprocess(self):
        """预处理数据"""
        print("预处理数据...")
        
        # 创建user和item的映射
        self.user_ids = sorted(self.train_data['user_id'].unique())
        self.item_ids = sorted(self.train_data['item_id'].unique())
        
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)
        
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(self.user_ids)}
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(self.item_ids)}
        self.new_to_old_user_id = {v: k for k, v in self.user_id_map.items()}
        self.new_to_old_item_id = {v: k for k, v in self.item_id_map.items()}
        
        # 将原始id映射为连续的id
        self.train_data['user_idx'] = self.train_data['user_id'].map(self.user_id_map)
        self.train_data['item_idx'] = self.train_data['item_id'].map(self.item_id_map)
        
        # 构建用户-物品交互矩阵 (稀疏矩阵)
        self.user_item_matrix = sp.coo_matrix(
            (np.ones(len(self.train_data)), 
             (self.train_data['user_idx'].values, self.train_data['item_idx'].values)),
            shape=(self.n_users, self.n_items)
        ).tocsr()
        
        # 为每个用户创建物品交互集合
        self.user_items_dict = {}
        for user, item in zip(self.train_data['user_idx'], self.train_data['item_idx']):
            if user not in self.user_items_dict:
                self.user_items_dict[user] = set()
            self.user_items_dict[user].add(item)
        
        # 数据抽样
        if self.sample_ratio < 1.0:
            sample_size = int(len(self.train_data) * self.sample_ratio)
            self.train_data = self.train_data.sample(n=sample_size, random_state=42)
            print(f"数据抽样: {sample_size} 条记录 ({self.sample_ratio * 100:.1f}%)")
        
        print(f"预处理完成，用户数: {self.n_users}, 物品数: {self.n_items}")
        
        # 创建邻接矩阵
        self._create_adj_matrix()
    
    def _create_adj_matrix(self):
        """创建归一化的邻接矩阵"""
        # 构建邻接矩阵 [[0, R], [R^T, 0]]
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        
        # 填充用户-物品交互
        for user, items in self.user_items_dict.items():
            for item in items:
                # 用户到物品
                adj_mat[user, self.n_users + item] = 1
                # 物品到用户
                adj_mat[self.n_users + item, user] = 1
        
        # 转换为COO格式以便于计算度矩阵
        adj_mat = adj_mat.tocoo()
        
        # 计算度矩阵的平方根的逆
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # 归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)
        self.norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
        
        # 转换为PyTorch稀疏张量
        indices = torch.LongTensor([self.norm_adj.row, self.norm_adj.col])
        values = torch.FloatTensor(self.norm_adj.data)
        shape = torch.Size(self.norm_adj.shape)
        
        self.norm_adj_tensor = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
    
    def _negative_sampling(self):
        """为每个用户-物品对采样负样本"""
        user_indices, pos_item_indices, neg_item_indices = [], [], []
        
        for user, pos_items in self.user_items_dict.items():
            pos_items = list(pos_items)
            for pos_item in pos_items:
                user_indices.append(user)
                pos_item_indices.append(pos_item)
                
                # 采样负样本
                neg_item = np.random.randint(0, self.n_items)
                while neg_item in pos_items:
                    neg_item = np.random.randint(0, self.n_items)
                neg_item_indices.append(neg_item)
        
        return user_indices, pos_item_indices, neg_item_indices
    
    def build_model(self):
        """构建LightGCN模型"""
        self.model = LightGCNModel(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            n_layers=self.n_layers,
            device=self.device
        ).to(self.device)
        
        print(self.model)
    
    def train(self):
        """训练模型"""
        print("开始训练LightGCN模型...")
        
        # 构建模型
        self.build_model()
        
        # 定义优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True
        )
        
        # 早停变量
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 采样负样本
            user_indices, pos_item_indices, neg_item_indices = self._negative_sampling()
            
            # 创建数据集和加载器
            dataset = BPRDataset(user_indices, pos_item_indices, neg_item_indices)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # 训练过程
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            start_time = time.time()
            
            for batch_idx, (users, pos_items, neg_items) in enumerate(dataloader):
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)
                
                # 前向传播
                user_emb, item_emb = self.model(self.norm_adj_tensor)
                
                # 计算BPR损失
                loss, reg_loss = self.model.bpr_loss(users, pos_items, neg_items, user_emb, item_emb)
                total_loss = loss + self.weight_decay * reg_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss.item():.4f}")
            
            # 计算平均损失
            avg_loss = epoch_loss / batch_count
            self.train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs}, 平均损失: {avg_loss:.4f}, 耗时: {time.time() - start_time:.2f}秒")
            
            # 学习率调度
            scheduler.step(avg_loss)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_lightgcn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"早停：{self.early_stopping_patience} 个epochs内没有改进")
                    # 加载最佳模型
                    self.model.load_state_dict(torch.load('best_lightgcn_model.pth'))
                    break
        
        # 保存最终模型
        torch.save(self.model.state_dict(), 'lightgcn_model.pth')
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', linewidth=2)
        plt.title('LightGCN 训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.grid(True)
        plt.savefig('lightgcn_training_loss.png')
        plt.close()
    
    def recommend(self, user_id, top_n=10):
        """为指定用户推荐物品"""
        # 检查用户是否在训练集中
        if user_id not in self.user_id_map:
            return self.recommend_popular_items(top_n)
        
        # 获取模型编码下的用户ID
        user_idx = self.user_id_map[user_id]
        
        # 设置为评估模式
        self.model.eval()
        with torch.no_grad():
            # 获取用户和物品嵌入
            user_emb, item_emb = self.model(self.norm_adj_tensor)
            
            # 获取目标用户的嵌入
            u_emb = user_emb[user_idx].unsqueeze(0)
            
            # 计算所有物品的评分
            scores = torch.matmul(u_emb, item_emb.t()).squeeze()
            
            # 过滤已交互的物品
            user_items = self.user_items_dict.get(user_idx, set())
            scores[list(user_items)] = -float('inf')
            
            # 获取前N个物品
            _, indices = torch.topk(scores, top_n)
            recommended_items = indices.cpu().numpy()
            
        # 将内部物品ID映射回原始ID
        return [self.new_to_old_item_id[item_idx] for item_idx in recommended_items]
    
    def recommend_popular_items(self, top_n=10):
        """推荐最流行的物品"""
        # 计算每个物品的流行度（交互次数）
        item_counts = self.train_data['item_idx'].value_counts()
        popular_items = item_counts.index[:top_n].tolist()
        
        # 将内部物品ID映射回原始ID
        return [self.new_to_old_item_id[item_idx] for item_idx in popular_items]
    
    def generate_recommendations(self, test_users, top_n=10):
        """为测试集用户生成推荐"""
        print(f"为测试用户生成推荐...")
        
        recommendations = {}
        
        for i, user_id in enumerate(test_users):
            user_recs = self.recommend(user_id, top_n)
            recommendations[user_id] = user_recs
            
            if (i + 1) % 1000 == 0:
                print(f"已为 {i+1}/{len(test_users)} 位用户生成推荐")
                
        return recommendations
    
    def save_recommendations(self, recommendations, output_file):
        """保存推荐结果到文件"""
        results = []
        for user_id, items in recommendations.items():
            for rank, item_id in enumerate(items):
                results.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rank': rank + 1
                })
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存结果
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"推荐结果已保存到 {output_file}")

# 主函数
def main():
    # 初始化模型
    lightgcn = LightGCN(
        embedding_dim=64,
        n_layers=3,
        learning_rate=0.001,
        epochs=5,
        batch_size=1024,
        weight_decay=1e-4,
        neg_ratio=1,
        early_stopping_patience=2,
        sample_ratio=0.3
    )
    
    # 加载数据
    lightgcn.load_data(r'D:\Python\project\bookRecommendation\BookRecommendation\data\train_dataset.csv')
    lightgcn.preprocess()
    
    # 训练模型
    lightgcn.train()
    
    # 绘制训练历史
    lightgcn.plot_training_history()
    
    # 加载测试用户
    test_users = pd.read_csv(r'D:\Python\project\bookRecommendation\BookRecommendation\data\test_dataset.csv')['user_id'].tolist()
    
    # 生成推荐
    recommendations = lightgcn.generate_recommendations(test_users, top_n=10)
    
    # 保存结果
    lightgcn.save_recommendations(recommendations, 'results/lightgcn_recommendations.csv')

if __name__ == "__main__":
    main()