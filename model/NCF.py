import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os

class NCFDataset(Dataset):
    """用于NCF模型的PyTorch数据集"""
    def __init__(self, user_indices, item_indices, labels):
        self.user_indices = torch.LongTensor(user_indices)
        self.item_indices = torch.LongTensor(item_indices)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.user_indices[idx], self.item_indices[idx], self.labels[idx])

class NCFModel(nn.Module):
    """Neural Collaborative Filtering模型"""
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16]):
        super(NCFModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.layers = layers
        
        # GMF部分的嵌入
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP部分的嵌入
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP层
        self.mlp_layers = nn.ModuleList()
        input_size = embedding_dim * 2
        for i, layer_size in enumerate(layers):
            # 修改这里，确保传入的是整数而不是元组
            self.mlp_layers.append(nn.Linear(int(input_size), int(layer_size)))
            input_size = layer_size
            
        # 预测层 - 同样确保是整数
        self.prediction = nn.Linear(int(embedding_dim + layers[-1]), 1)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, user_indices, item_indices):
        # GMF部分
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = user_gmf * item_gmf
        
        # MLP部分
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=1)
        
        # MLP层前向传播
        for layer in self.mlp_layers:
            mlp_vector = nn.ReLU()(layer(mlp_vector))
            
        # 组合GMF和MLP
        vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        
        # 输出预测
        predict = self.sigmoid(self.prediction(vector))
        return predict.view(-1)

class NeuralCollaborativeFiltering:
    def __init__(self, embedding_dim=32, layers=[64, 32, 16], learning_rate=0.001, 
                 epochs=20, batch_size=256, device=None):
        """
        初始化NCF模型
        
        参数:
            embedding_dim: 嵌入向量维度
            layers: MLP层的神经元数量列表
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批量大小
            device: 计算设备 (CPU/GPU)
        """
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # 设置计算设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.model = None
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self, filepath):
        """加载训练数据"""
        print("加载数据...")
        self.train_data = pd.read_csv(filepath)
        print(f"数据加载完成，共 {len(self.train_data)} 条交互记录")
        
    def preprocess(self):
        """预处理数据并创建训练样本"""
        print("预处理数据...")
        # 创建用户和物品的映射字典
        self.unique_users = self.train_data['user_id'].unique()
        self.unique_items = self.train_data['item_id'].unique()
        
        self.user_map = {user: idx for idx, user in enumerate(self.unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(self.unique_items)}
        
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        # 转换数据
        user_indices = np.array([self.user_map[user] for user in self.train_data['user_id']])
        item_indices = np.array([self.item_map[item] for item in self.train_data['item_id']])
        
        # 创建用户-物品交互矩阵（用于负采样）
        self.interaction_matrix = np.zeros((len(self.unique_users), len(self.unique_items)), dtype=bool)
        for u, i in zip(user_indices, item_indices):
            self.interaction_matrix[u, i] = True
        
        # 创建正样本
        self.train_user_indices = user_indices
        self.train_item_indices = item_indices
        self.train_labels = np.ones(len(self.train_data))
        
        # 负采样：为每个正样本生成4个负样本
        self._negative_sampling(negative_ratio=4)
        
        print(f"预处理完成，用户数: {len(self.unique_users)}, 物品数: {len(self.unique_items)}")
        print(f"训练样本: {len(self.train_labels)} (正样本: {len(self.train_data)}, 负样本: {len(self.train_labels) - len(self.train_data)})")
        
    def _negative_sampling(self, negative_ratio=4):
        """生成负样本"""
        print(f"生成负样本，比例 1:{negative_ratio}...")
        neg_user_indices = []
        neg_item_indices = []
        neg_labels = []
        
        # 为每个正样本生成negative_ratio个负样本
        for user_idx in range(len(self.unique_users)):
            # 找出用户未交互的物品
            interacted_items = self.interaction_matrix[user_idx]
            non_interacted_items = np.where(~interacted_items)[0]
            
            # 如果该用户几乎与所有物品都有交互，跳过
            if len(non_interacted_items) < negative_ratio:
                continue
                
            # 获取用户的正样本数量
            pos_samples = np.sum(self.train_user_indices == user_idx)
            
            # 随机选择负样本
            neg_samples_to_generate = min(negative_ratio * pos_samples, len(non_interacted_items))
            if neg_samples_to_generate <= 0:
                continue
                
            sampled_negatives = np.random.choice(non_interacted_items, size=neg_samples_to_generate, replace=False)
            
            # 添加到训练数据
            for item_idx in sampled_negatives:
                neg_user_indices.append(user_idx)
                neg_item_indices.append(item_idx)
                neg_labels.append(0)  # 负样本标签为0
        
        # 合并正负样本
        self.train_user_indices = np.concatenate([self.train_user_indices, np.array(neg_user_indices)])
        self.train_item_indices = np.concatenate([self.train_item_indices, np.array(neg_item_indices)])
        self.train_labels = np.concatenate([self.train_labels, np.array(neg_labels)])
        
        # 打乱数据
        indices = np.arange(len(self.train_labels))
        np.random.shuffle(indices)
        self.train_user_indices = self.train_user_indices[indices]
        self.train_item_indices = self.train_item_indices[indices]
        self.train_labels = self.train_labels[indices]
    
    def build_model(self):
        """构建NCF模型架构"""
        self.model = NCFModel(
            num_users=len(self.unique_users),
            num_items=len(self.unique_items),
            embedding_dim=self.embedding_dim,
            layers=self.layers
        ).to(self.device)
        
        # 打印模型架构
        print(self.model)
        
    def train(self):
        """训练NCF模型"""
        print("开始训练NCF模型...")
        start_time = time.time()
        
        # 构建模型
        self.build_model()
        
        # 创建保存模型的目录
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # 创建数据集和数据加载器
        # 划分训练集和验证集
        num_samples = len(self.train_labels)
        num_val = int(num_samples * 0.1)
        
        # 创建训练集
        train_dataset = NCFDataset(
            self.train_user_indices[num_val:], 
            self.train_item_indices[num_val:], 
            self.train_labels[num_val:]
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建验证集
        val_dataset = NCFDataset(
            self.train_user_indices[:num_val], 
            self.train_item_indices[:num_val], 
            self.train_labels[:num_val]
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 初始化最佳验证损失和早停计数器
        best_val_loss = float('inf')
        early_stop_count = 0
        patience = 3
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            for batch_idx, (user_indices, item_indices, labels) in enumerate(train_loader):
                # 将数据移到设备上
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                predictions = self.model(user_indices, item_indices)
                loss = criterion(predictions, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 打印批次进度
                if (batch_idx+1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # 计算平均训练损失
            avg_train_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for user_indices, item_indices, labels in val_loader:
                    user_indices = user_indices.to(self.device)
                    item_indices = item_indices.to(self.device)
                    labels = labels.to(self.device)
                    
                    predictions = self.model(user_indices, item_indices)
                    loss = criterion(predictions, labels)
                    val_loss += loss.item()
            
            # 计算平均验证损失
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'models/ncf_best_model.pth')
                early_stop_count = 0
                print("保存最佳模型!")
            else:
                early_stop_count += 1
                
            # 早停
            if early_stop_count >= patience:
                print(f"验证损失连续{patience}轮未下降，提前停止训练")
                break
        
        # 训练完成，记录总时间
        training_time = time.time() - start_time
        print(f"模型训练完成，耗时: {training_time:.2f} 秒")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('models/ncf_best_model.pth'))
        
        # 绘制训练历史
        self.plot_training_history()
        
    def plot_training_history(self):
        """绘制训练历史曲线"""
        plt.figure(figsize=(10, 4))
        
        # 创建保存目录
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 绘制损失曲线
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.title('模型训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/ncf_training_history.png')
        plt.close()
        
    def recommend(self, user_id, top_n=10):
        """为指定用户生成推荐"""
        if user_id not in self.user_map:
            return self.recommend_popular_items(top_n)
            
        user_idx = self.user_map[user_id]
        
        # 获取用户已交互的物品
        interacted_items = self.interaction_matrix[user_idx]
        
        # 生成所有未交互物品的预测分数
        user_input = np.full(len(self.unique_items), user_idx)
        item_input = np.arange(len(self.unique_items))
        
        # 过滤掉已交互的物品
        mask = ~interacted_items
        user_input = user_input[mask]
        item_input = item_input[mask]
        
        # 如果用户与所有物品都交互过，返回热门推荐
        if len(user_input) == 0:
            return self.recommend_popular_items(top_n)
        
        # 为减少内存使用，分批次预测
        self.model.eval()
        batch_size = 1024
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(user_input), batch_size):
                batch_users = torch.LongTensor(user_input[i:i+batch_size]).to(self.device)
                batch_items = torch.LongTensor(item_input[i:i+batch_size]).to(self.device)
                batch_predictions = self.model(batch_users, batch_items).cpu().numpy()
                predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        
        # 获取得分最高的物品
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        top_item_indices = item_input[top_indices]
        
        # 转换回原始物品ID
        return [self.reverse_item_map[idx] for idx in top_item_indices]
        
    def recommend_popular_items(self, top_n=10):
        """推荐热门物品（用于冷启动问题）"""
        # 计算物品流行度
        item_counts = self.train_data['item_id'].value_counts()
        popular_items = item_counts.index[:top_n].tolist()
        return popular_items
    
    def generate_recommendations(self, test_users, top_n=10):
        """为测试集用户生成推荐"""
        print(f"为测试用户生成推荐...")
        
        recommendations = {}
        
        for i, user_id in enumerate(test_users):
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

# 主函数
def main():
    # 初始化模型
    ncf = NeuralCollaborativeFiltering(
        embedding_dim=32, 
        layers=[128, 64, 32], 
        learning_rate=0.001, 
        epochs=10,
        batch_size=256
    )
    
    # 加载数据
    ncf.load_data('D:\\Python\project\\bookRecommendation\\data\\train_dataset.csv')  # 使用评估划分的训练集
    ncf.preprocess()
    
    # 训练模型
    ncf.train()
    
    # 加载测试用户
    test_users = pd.read_csv('D:\\Python\project\\bookRecommendation\\data\\test_dataset.csv')['user_id'].tolist()
    
    # 生成推荐
    recommendations = ncf.generate_recommendations(test_users, top_n=10)
    
    # 保存结果
    ncf.save_recommendations(recommendations, 'results/ncf_recommendations.csv')

if __name__ == "__main__":
    main()