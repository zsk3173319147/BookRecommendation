相似度矩阵的数学原理
在协同过滤中，相似度矩阵是通过余弦相似度(Cosine Similarity)计算得到的。这是一个测量两个向量方向相似性的指标。

余弦相似度的数学原理
余弦相似度的公式为： $\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||}$   

其中：   

$\mathbf{A} \cdot \mathbf{B}$ 是向量的点积
$||\mathbf{A}||$ 和 $||\mathbf{B}||$ 是向量的欧几里得范数(长度)
在协同过滤中的应用
1. 基于物品的相似度
每个物品表示为用户交互向量：

物品A的向量：[用户1对A的交互, 用户2对A的交互, ...]
物品B的向量：[用户1对B的交互, 用户2对B的交互, ...]
两个物品的相似度计算：

2. 基于用户的相似度
每个用户表示为物品交互向量：

用户X的向量：[X对物品1的交互, X对物品2的交互, ...]
用户Y的向量：[Y对物品1的交互, Y对物品2的交互, ...]
两个用户的相似度计算：

在代码中的实现
在你的代码中，相似度矩阵通过cosine_similarity函数计算：

余弦相似度的特点
值域：[-1, 1]，1表示完全相似，0表示无关，-1表示完全相反
只考虑方向：余弦相似度只关注向量的方向，忽略向量的大小
适用于稀疏数据：在推荐系统中非常有效，因为大多数用户只与极少数物品交互
在二元交互数据(如你的情况，只有交互/未交互)中，余弦相似度能有效捕捉用户行为模式或物品特性的相似性，而不受交互总数的影响。





 



矩阵分解原理与数学形式
基本思想
矩阵分解的核心思想是：将用户-物品交互矩阵 $R$ (维度为 $m \times n$，$m$ 是用户数，$n$ 是物品数)分解为两个低维矩阵的乘积：

用户潜在因子矩阵 $U$ (维度为 $m \times k$)
物品潜在因子矩阵 $V$ (维度为 $n \times k$)
其中 $k$ 是潜在因子数量，通常远小于用户数和物品数。

$R \approx U \cdot V^T$

ALS算法数学形式
ALS (交替最小二乘法) 特别适合处理隐式反馈数据。其目标函数为：

$$\min_{U,V} \sum_{(i,j) \in \mathcal{K}} c_{ij}(p_{ij} - u_i^T v_j)^2 + \lambda(\sum_i ||u_i||^2 + \sum_j ||v_j||^2)$$

其中：

$\mathcal{K}$ 是所有用户-物品对的集合
$p_{ij}$ 是用户 $i$ 与物品 $j$ 的交互情况(通常为0或1)
$c_{ij}$ 是置信权重，通常定义为 $c_{ij} = 1 + \alpha \cdot p_{ij}$，$\alpha$ 是置信参数
$u_i$ 是用户 $i$ 的潜在因子向量
$v_j$ 是物品 $j$ 的潜在因子向量
$\lambda$ 是正则化参数
求解过程
ALS通过交替固定一个矩阵来优化另一个矩阵：

固定 $V$，求解 $U$： 对每个用户 $i$，求解： $u_i = (V^T C_i V + \lambda I)^{-1} V^T C_i p_i$ 

固定 $U$，求解 $V$： 对每个物品 $j$，求解： $v_j = (U^T C_j U + \lambda I)^{-1} U^T C_j p_j$

其中 $C_i$ 和 $C_j$ 是对角权重矩阵。   




NCF原理详解
Neural Collaborative Filtering是一种深度学习推荐模型，结合了传统矩阵分解和神经网络的优势。

1. 数学基础
NCF主要解决的问题是预测用户-物品交互的可能性: $\hat{y}_{ui} = f(u, i | \Theta)$

其中:

$\hat{y}_{ui}$ 是用户$u$与物品$i$交互的预测概率
$f$ 是模型函数
$\Theta$ 是模型参数
2. NCF的三个关键组件
2.1 通用矩阵分解 (GMF)
GMF将传统的矩阵分解表示为神经网络形式:

$\hat{y}_{ui}^{GMF} = \sigma(\mathbf{h}^T(\mathbf{p}_u^G \odot \mathbf{q}_i^G))$

其中:

$\mathbf{p}_u^G$ 是用户$u$的GMF嵌入向量
$\mathbf{q}_i^G$ 是物品$i$的GMF嵌入向量
$\odot$ 表示元素级乘法
$\sigma$ 是sigmoid激活函数
2.2 多层感知器 (MLP)
MLP通过神经网络捕获用户-物品交互的非线性特征:

$\mathbf{z}_1 = \begin{bmatrix} \mathbf{p}_u^M \ \mathbf{q}_i^M \end{bmatrix}$ $\mathbf{z}2 = \sigma_1(\mathbf{W}1\mathbf{z}1 + \mathbf{b}1)$ $\cdots$ $\mathbf{z}L = \sigma{L-1}(\mathbf{W}{L-1}\mathbf{z}{L-1} + \mathbf{b}{L-1})$ $\hat{y}{ui}^{MLP} = \sigma(\mathbf{h}^T\mathbf{z}_L)$

其中:

$\mathbf{p}_u^M$ 是用户$u$的MLP嵌入向量
$\mathbf{q}_i^M$ 是物品$i$的MLP嵌入向量
$\mathbf{W}_x, \mathbf{b}_x$ 是第$x$层的权重和偏置
$\sigma_x$ 是激活函数(通常是ReLU)
2.3 融合模型 (NeuMF)
NeuMF将GMF和MLP结合起来:

$\hat{y}_{ui} = \sigma(\mathbf{h}^T[\mathbf{p}_u^G \odot \mathbf{q}_i^G, \mathbf{z}_L])$

3. 训练过程
负采样：由于隐式反馈只有正样本，需要生成负样本
损失函数：使用二元交叉熵 $\mathcal{L} = -\sum_{(u,i,y) \in \mathcal{D}} y \log(\hat{y}{ui}) + (1-y) \log(1-\hat{y}{ui})$
优化：通过反向传播和Adam优化器更新参数
4. 优势
发现非线性关系：能捕捉传统矩阵分解无法捕捉的复杂交互模式
灵活的架构：可以轻松调整网络结构和深度
端到端训练：从原始数据到最终推荐的完整过程









LightGCN 推荐模型详解
LightGCN 是一种轻量级的图卷积网络，由何向南等人在2020年提出，专为推荐系统设计。它基于以下观察：传统GCN中的特征变换和非线性激活在推荐场景中并不必要，反而可能导致过拟合。

1. 核心思想
LightGCN核心设计原则：

去除特征变换矩阵
去除非线性激活函数
仅保留最核心的邻居聚合操作
使用层聚合机制整合不同层的信息
2. 数学原理
图构建
用户-物品交互构建成二分图 G = (U  I, E)：

U: 用户集合
I: 物品集合
E: 交互边集合
邻接矩阵
$A = \begin{bmatrix} 0 & R \ R^T & 0 \end{bmatrix}$

其中R是用户-物品交互矩阵。

图卷积传播
第k+1层嵌入通过以下方式计算： $e^{(k+1)} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} e^{(k)}$

对于用户u和物品i： $e_u^{(k+1)} = \sum_{i \in N_u} \frac{1}{\sqrt{|N_u|} \cdot \sqrt{|N_i|}} e_i^{(k)}$

$e_i^{(k+1)} = \sum_{u \in N_i} \frac{1}{\sqrt{|N_i|} \cdot \sqrt{|N_u|}} e_u^{(k)}$

层聚合
最终嵌入是所有层嵌入的加权和： $e_u = \sum_{k=0}^K \alpha_k e_u^{(k)}, \quad e_i = \sum_{k=0}^K \alpha_k e_i^{(k)}$

通常设置为平均值：$\alpha_k = \frac{1}{K+1}$

预测和损失函数
用户u对物品i的预测分数： $\hat{y}_{ui} = e_u^T e_i$

使用BPR损失优化： $L_{BPR} = \sum_{(u,i,j)} -\ln \sigma(\hat{y}{ui} - \hat{y}{uj}) + \lambda||E||^2$

3. 算法流程
初始化用户和物品嵌入
构建并归一化邻接矩阵
执行K层图卷积操作
聚合各层表示得到最终嵌入
使用BPR损失进行优化
4. LightGCN优势
高效性: 参数量显著减少，计算速度更快
性能提升: 去除不必要组件反而提高了推荐精度
捕捉高阶关系: 能够自然建模用户-物品间的高阶连接
可解释性: 模型结构简单，易于理解
扩展性: 适用于大规模推荐场景
相比传统的CF和MF，LightGCN能够更好地捕捉用户-物品交互网络中的结构信息，特别是能够通过多层传播来建模高阶连接关系，从而产生更准确的推荐