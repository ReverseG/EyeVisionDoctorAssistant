# 眼底 AI 系统 - 算法关联分析与症状抽取设计文档

> **辅助诊断参考** - 不替代医生最终诊断  
> **版本**: v1.0  
> **创建日期**: 2026-04-01

---

## 一、多模态关联分析算法设计

### 1.1 核心问题定义

**目标**：建立眼底图像表征（Image Features）与病人临床症状（Clinical Symptoms）之间的可解释性关联。

**输入**：
- 眼底图像 AI 分析结果（DR 分级、病变检测、OCT 分层等）
- 患者电子病历（主诉、现病史、既往史、实验室检查）
- 医生诊断报告文本

**输出**：
- 图像 - 症状关联图谱
- 关联强度评分（0-1）
- 可解释性证据链

---

### 1.2 整体技术架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        多模态关联分析引擎                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  图像特征层  │    │  文本特征层  │    │  结构化数据层 │          │
│  │              │    │              │    │              │          │
│  │ • DR 分级    │    │ • 主诉       │    │ • 血糖       │          │
│  │ • 病变类型   │    │ • 现病史     │    │ • 血压       │          │
│  │ • 病变数量   │    │ • 既往史     │    │ • HbA1c      │          │
│  │ • 热力图     │    │ • 用药史     │    │ • 肾功能     │          │
│  │ • OCT 参数   │    │ • 家族史     │    │ • 血脂       │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                    │
│         └───────────────────┼───────────────────┘                    │
│                             ↓                                        │
│                  ┌─────────────────────┐                             │
│                  │   特征对齐与融合层   │                             │
│                  │                     │                             │
│                  │ • 时间窗口对齐      │                             │
│                  │ • 患者 ID 关联       │                             │
│                  │ • 特征标准化        │                             │
│                  │ • 缺失值处理        │                             │
│                  └──────────┬──────────┘                             │
│                             ↓                                        │
│                  ┌─────────────────────┐                             │
│                  │   关联分析算法层     │                             │
│                  │                     │                             │
│                  │ • 典型相关分析 CCA  │                             │
│                  │ • 多核学习 MKL      │                             │
│                  │ • 图神经网络 GNN    │                             │
│                  │ • 注意力机制        │                             │
│                  └──────────┬──────────┘                             │
│                             ↓                                        │
│                  ┌─────────────────────┐                             │
│                  │   可解释性输出层     │                             │
│                  │                     │                             │
│                  │ • 关联强度评分      │                             │
│                  │ • 证据链可视化      │                             │
│                  │ • 置信区间          │                             │
│                  └─────────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 关联分析算法详解

#### 算法 1：典型相关分析 (Canonical Correlation Analysis, CCA)

**用途**：发现图像特征组与临床症状组之间的最大相关性。

**数学形式**：
```
给定：
  X = [x₁, x₂, ..., xₙ] ∈ ℝ^(p×n)  # 图像特征矩阵
  Y = [y₁, y₂, ..., yₙ] ∈ ℝ^(q×n)  # 症状特征矩阵

目标：
  找到投影向量 a ∈ ℝ^p, b ∈ ℝ^q
  使得 ρ = corr(Xᵀa, Yᵀb) 最大化

解：
  max ρ = (aᵀΣₓᵧb) / √(aᵀΣₓₓa · bᵀΣᵧᵧb)
  
  其中 Σₓₓ, Σᵧᵧ 为协方差矩阵，Σₓᵧ 为互协方差矩阵
```

**Python 实现**：
```python
from sklearn.cross_decomposition import CCA
import numpy as np

class ImageSymptomCCA:
    def __init__(self, n_components=5):
        self.cca = CCA(n_components=n_components)
        self.correlations = None
        
    def fit(self, image_features, symptom_features):
        """
        image_features: [n_samples, n_image_features]
        symptom_features: [n_samples, n_symptom_features]
        """
        X_transformed, Y_transformed = self.cca.fit_transform(
            image_features, symptom_features
        )
        
        # 计算典型相关系数
        self.correlations = [
            np.corrcoef(X_transformed[:, i], Y_transformed[:, i])[0, 1]
            for i in range(X_transformed.shape[1])
        ]
        
        return self
    
    def get_feature_weights(self):
        """获取各特征对关联的贡献权重"""
        x_weights = self.cca.x_weights_
        y_weights = self.cca.y_weights_
        return x_weights, y_weights
    
    def explain(self, patient_id):
        """生成可解释性报告"""
        report = {
            'patient_id': patient_id,
            'canonical_correlations': self.correlations,
            'top_image_features': self._get_top_features('image'),
            'top_symptom_features': self._get_top_features('symptom'),
            'interpretation': self._generate_interpretation()
        }
        return report
```

**输出示例**：
```json
{
  "patient_id": "P00001",
  "canonical_correlations": [0.87, 0.72, 0.65, 0.51, 0.43],
  "top_image_features": [
    {"feature": "微动脉瘤数量", "weight": 0.82, "p_value": 0.001},
    {"feature": "出血点密度", "weight": 0.76, "p_value": 0.003},
    {"feature": "DR 分级", "weight": 0.71, "p_value": 0.008}
  ],
  "top_symptom_features": [
    {"feature": "糖尿病病程", "weight": 0.79, "p_value": 0.002},
    {"feature": "HbA1c", "weight": 0.74, "p_value": 0.005},
    {"feature": "收缩压", "weight": 0.68, "p_value": 0.012}
  ],
  "interpretation": "微动脉瘤数量与糖尿病病程呈显著正相关 (r=0.87, p<0.001)，提示病程越长视网膜微血管损伤越严重"
}
```

---

#### 算法 2：多核学习 (Multiple Kernel Learning, MKL)

**用途**：融合异构特征（图像 + 文本 + 结构化数据），学习最优核组合。

**数学形式**：
```
给定 M 个核函数 K₁, K₂, ..., Kₘ

目标：
  学习最优核组合 K* = Σᵢ ηᵢKᵢ
  使得分类/回归性能最大化

约束：
  ηᵢ ≥ 0, Σᵢ ηᵢ = 1

优化问题：
  min_(f,η) L(f, K*) + λ||f||²_K* + μΩ(η)
```

**Python 实现**：
```python
from sklearn.svm import SVR
import numpy as np
from typing import List, Dict

class MultiModalMKL:
    def __init__(self, kernel_types: List[str] = ['rbf', 'linear', 'poly']):
        self.kernel_types = kernel_types
        self.kernel_weights = None
        self.models = []
        
    def _compute_kernels(self, X: np.ndarray) -> List[np.ndarray]:
        """计算多种核矩阵"""
        kernels = []
        n_samples = X.shape[0]
        
        # RBF 核
        if 'rbf' in self.kernel_types:
            gamma = 1.0 / X.shape[1]
            sq_dist = np.sum(X**2, axis=1).reshape(-1, 1) + \
                      np.sum(X**2, axis=1).reshape(1, -1) - \
                      2 * X @ X.T
            k_rbf = np.exp(-gamma * sq_dist)
            kernels.append(('rbf', k_rbf))
        
        # 线性核
        if 'linear' in self.kernel_types:
            k_linear = X @ X.T
            kernels.append(('linear', k_linear))
        
        # 多项式核
        if 'poly' in self.kernel_types:
            k_poly = (X @ X.T + 1) ** 3
            kernels.append(('poly', k_poly))
        
        return kernels
    
    def fit(self, image_features: np.ndarray, 
            symptom_features: np.ndarray,
            labels: np.ndarray):
        """
        训练 MKL 模型
        """
        # 计算各模态的核矩阵
        image_kernels = self._compute_kernels(image_features)
        symptom_kernels = self._compute_kernels(symptom_features)
        
        # 简单平均融合（可优化为学习权重）
        combined_kernel = (
            np.mean([k[1] for k in image_kernels], axis=0) +
            np.mean([k[1] for k in symptom_kernels], axis=0)
        ) / 2
        
        # 训练 SVR
        self.model = SVR(kernel='precomputed')
        self.model.fit(combined_kernel, labels)
        
        # 计算核权重（基于对齐度）
        self.kernel_weights = self._compute_kernel_weights(
            image_kernels, symptom_kernels, labels
        )
        
        return self
    
    def predict_correlation(self, image_feat: np.ndarray, 
                           symptom_feat: np.ndarray) -> float:
        """预测关联强度"""
        # 计算测试样本与训练样本的核矩阵
        # ... 实现略
        pass
    
    def get_modality_importance(self) -> Dict[str, float]:
        """获取各模态的重要性"""
        return {
            'image_features': self.kernel_weights.get('image', 0.0),
            'symptom_features': self.kernel_weights.get('symptom', 0.0),
            'lab_features': self.kernel_weights.get('lab', 0.0)
        }
```

---

#### 算法 3：图神经网络关联建模 (Graph Neural Network)

**用途**：构建患者 - 图像 - 症状异构图，学习高阶关联。

**图结构定义**：
```
节点类型：
  - 患者节点 (Patient Node)
  - 图像特征节点 (Image Feature Node)
  - 症状节点 (Symptom Node)
  - 诊断节点 (Diagnosis Node)

边类型：
  - 患者-图像：has_image (权重=图像质量评分)
  - 患者-症状：has_symptom (权重=症状严重程度)
  - 图像-症状：correlated_with (权重=关联强度)
  - 患者-诊断：has_diagnosis
```

**PyTorch Geometric 实现**：
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from torch_geometric.data import HeteroData

class EyeVisionGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64, num_layers=3):
        super().__init__()
        
        # 异构图卷积层
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('patient', 'has_image', 'image'): GATConv((-1, -1), hidden_channels),
                ('patient', 'has_symptom', 'symptom'): GATConv((-1, -1), hidden_channels),
                ('image', 'correlated_with', 'symptom'): GATConv((-1, -1), hidden_channels),
                ('patient', 'has_diagnosis', 'diagnosis'): GATConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        
        # 关联预测头
        self.correlation_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, data: HeteroData):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # 图卷积
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # 预测图像 - 症状关联
        image_emb = x_dict['image']
        symptom_emb = x_dict['symptom']
        
        # 计算所有图像 - 症状对的关联分数
        correlations = []
        for i in range(len(image_emb)):
            for j in range(len(symptom_emb)):
                pair_emb = torch.cat([image_emb[i], symptom_emb[j]], dim=-1)
                corr_score = self.correlation_head(pair_emb)
                correlations.append(corr_score)
        
        return torch.stack(correlations)
    
    def explain_path(self, patient_id: str, 
                     image_feature: str, 
                     symptom: str) -> List[Dict]:
        """提取解释路径"""
        # 使用 GNNExplainer 或提取注意力权重
        pass
```

---

### 1.4 特征工程详细设计

#### 图像特征提取

| 特征类别 | 具体特征 | 提取方法 | 维度 |
|----------|----------|----------|------|
| **全局特征** | DR 分级 (0-4) | EfficientNet-B4 | 1 |
| | 置信度分数 | Softmax 输出 | 1 |
| | 图像质量评分 | CNN 回归模型 | 1 |
| **病变特征** | 微动脉瘤数量 | YOLOv8 检测 | 1 |
| | 出血点数量 | YOLOv8 检测 | 1 |
| | 硬性渗出数量 | YOLOv8 检测 | 1 |
| | 软性渗出数量 | YOLOv8 检测 | 1 |
| | 新生血管有无 | 二分类模型 | 1 |
| | IRMA 有无 | 二分类模型 | 1 |
| **空间特征** | 病变分布象限 | 四分位统计 | 4 |
| | 黄斑中心距离 | 几何计算 | 1 |
| | 视盘面积比 | 分割模型 | 1 |
| **纹理特征** | LBP 直方图 | 局部二值模式 | 256 |
| | GLCM 特征 | 灰度共生矩阵 | 14 |
| | HOG 特征 | 方向梯度直方图 | 144 |
| **OCT 特征** | 视网膜总厚度 | U-Net 分割 | 1 |
| | 各分层厚度 | U-Net 分割 | 9 |
| | 黄斑水肿体积 | 3D 分割 | 1 |
| | 杯盘比 CDR | DeepLabV3+ | 1 |

**总计**: ~450 维图像特征

#### 症状特征提取

| 特征类别 | 具体特征 | 数据来源 | 类型 |
|----------|----------|----------|------|
| **人口学** | 年龄 | 患者档案 | 数值 |
| | 性别 | 患者档案 | 类别 |
| | BMI | 身高体重计算 | 数值 |
| **病史** | 糖尿病病程 (年) | 既往史 | 数值 |
| | 高血压病程 (年) | 既往史 | 数值 |
| | 吸烟史 (包年) | 既往史 | 数值 |
| | 饮酒史 (频率) | 既往史 | 类别 |
| | 家族史 (DR) | 家族史 | 布尔 |
| **症状** | 视力下降程度 | 主诉 | 有序类别 |
| | 视物模糊 | 主诉 | 布尔 |
| | 飞蚊症 | 主诉 | 布尔 |
| | 视野缺损 | 主诉 | 布尔 |
| | 眼痛/头痛 | 主诉 | 布尔 |
| **实验室** | 空腹血糖 | 检验结果 | 数值 |
| | 餐后 2h 血糖 | 检验结果 | 数值 |
| | HbA1c | 检验结果 | 数值 |
| | 收缩压/舒张压 | 体征 | 数值 |
| | 总胆固醇 | 检验结果 | 数值 |
| | LDL/HDL | 检验结果 | 数值 |
| | 甘油三酯 | 检验结果 | 数值 |
| | 肌酐 | 检验结果 | 数值 |
| | eGFR | 计算 | 数值 |
| | 尿微量白蛋白 | 检验结果 | 数值 |
| **用药** | 降糖药种类数 | 用药史 | 数值 |
| | 胰岛素使用 | 用药史 | 布尔 |
| | 降压药使用 | 用药史 | 布尔 |
| | 他汀使用 | 用药史 | 布尔 |

**总计**: ~35 维症状特征

---

### 1.5 关联强度评分系统

**评分公式**：
```
Correlation_Score = α·CCA_Score + β·MKL_Score + γ·GNN_Score

其中：
  CCA_Score = 典型相关系数 × 特征权重乘积
  MKL_Score = 核对齐度 × 预测置信度
  GNN_Score = 图注意力权重 × 路径可信度
  
  α + β + γ = 1 (默认 α=0.4, β=0.3, γ=0.3)
```

**评分等级**：
| 分数范围 | 等级 | 颜色 | 解释 |
|----------|------|------|------|
| 0.8 - 1.0 | 极强关联 | 🔴 | 高度可信，临床意义明确 |
| 0.6 - 0.8 | 强关联 | 🟠 | 可信，需结合临床判断 |
| 0.4 - 0.6 | 中等关联 | 🟡 | 参考性关联，需进一步验证 |
| 0.2 - 0.4 | 弱关联 | 🔵 | 弱相关，谨慎解读 |
| 0.0 - 0.2 | 无关联 | ⚪ | 无统计学意义 |

---

## 二、医生报告症状抽取模块

### 2.1 任务定义

**目标**：从医生书写的诊断报告中自动抽取结构化症状信息。

**输入**：医生诊断报告文本（自由文本）

**输出**：结构化症状列表（JSON）

**示例**：
```
输入文本:
"患者双眼视力渐进性下降 3 年，近 1 月加重。
眼底检查：双眼视盘边界清，C/D=0.3，视网膜静脉迂曲扩张，
可见散在微动脉瘤及点片状出血，黄斑区可见硬性渗出。
诊断：双眼糖尿病视网膜病变（左眼 2 期，右眼 1 期）"

输出 JSON:
{
  "symptoms": [
    {"entity": "视力渐进性下降", "type": "视觉症状", "duration": "3 年", "severity": "加重"},
    {"entity": "视网膜静脉迂曲扩张", "type": "眼底体征", "location": "双眼"},
    {"entity": "微动脉瘤", "type": "病变", "distribution": "散在"},
    {"entity": "点片状出血", "type": "病变", "distribution": "散在"},
    {"entity": "硬性渗出", "type": "病变", "location": "黄斑区"}
  ],
  "diagnosis": [
    {"entity": "糖尿病视网膜病变", "stage_left": "2 期", "stage_right": "1 期"}
  ]
}
```

---

### 2.2 技术架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      医学报告症状抽取引擎                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  输入层                                                      │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │ 医生诊断报告文本（PDF/图片/文本）                       │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  预处理层                                                    │   │
│  │  • OCR 识别（如为图片/PDF）                                   │   │
│  │  • 文本清洗（去除特殊字符、标准化）                           │   │
│  │  • 句子分割                                                  │   │
│  │  • 医学缩写扩展（OD→右眼，OS→左眼）                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  命名实体识别 (NER)                                          │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │  BioBERT / Chinese-BERT-wwm + CRF                     │  │   │
│  │  │                                                       │  │   │
│  │  │  实体类型：                                             │  │   │
│  │  │  • 症状 (Symptom)                                      │  │   │
│  │  │  • 体征 (Sign)                                         │  │   │
│  │  │  • 病变 (Lesion)                                       │  │   │
│  │  │  • 诊断 (Diagnosis)                                    │  │   │
│  │  │  • 检查 (Examination)                                  │  │   │
│  │  │  • 治疗 (Treatment)                                    │  │   │
│  │  │  • 时间 (Time)                                         │  │   │
│  │  │  • 部位 (Location)                                     │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  关系抽取 (Relation Extraction)                              │   │
│  │  • 症状 - 部位关系（视力下降→双眼）                           │   │
│  │  • 症状 - 时间关系（下降→3 年）                                │   │
│  │  • 症状 - 程度关系（加重/轻度/中度/重度）                     │   │
│  │  • 病变 - 分布关系（微动脉瘤→散在）                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  结构化输出                                                  │   │
│  │  • JSON 格式化                                                │   │
│  │  • 置信度评分                                                │   │
│  │  • 来源句子标注                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.3 命名实体识别模型

#### 模型架构

**Base Model**: Chinese-BERT-wwm-ext (哈工大)

**原因**：
- 中文医学文本预训练
- 支持全词 Mask，更适合中文
- 在中文医疗 NER 任务上 SOTA

**架构**：
```
[Input Text] 
    ↓
[Chinese-BERT-wwm]  # 768 维上下文表示
    ↓
[BiLSTM]  # 捕捉长距离依赖
    ↓
[CRF]  # 序列标注约束
    ↓
[BIOES Tags]  # B-Symptom, I-Symptom, E-Symptom, S-Symptom, O
```

#### 训练数据标注规范

**实体类型定义**：

| 类型 | 标签 | 示例 |
|------|------|------|
| 症状 | SYMPTOM | 视力下降、视物模糊、飞蚊症 |
| 体征 | SIGN | 视盘边界清、C/D=0.3、静脉迂曲 |
| 病变 | LESION | 微动脉瘤、出血、渗出物 |
| 诊断 | DIAGNOSIS | 糖尿病视网膜病变、黄斑水肿 |
| 检查 | EXAM | 眼底检查、OCT、FFA |
| 治疗 | TREATMENT | 激光光凝、抗 VEGF 注射 |
| 时间 | TIME | 3 年、近 1 月、渐进性 |
| 部位 | LOCATION | 双眼、左眼、右眼、黄斑区 |
| 程度 | DEGREE | 轻度、中度、重度、散在、密集 |

**标注示例**：
```
文本：患者双眼视力渐进性下降 3 年，近 1 月加重。

标注：
患 O 者 O 双 B-LOCATION 眼 E-LOCATION 视 B-SYMPTOM 力 I-SYMPTOM 下 I-SYMPTOM 降 I-SYMPTOM 渐 B-TIME 进 I-TIME 性 I-TIME 下 I-TIME 降 I-TIME 3 I-TIME 年 E-TIME，近 B-TIME 1 I-TIME 月 E-TIME 加 B-DEGREE 重 E-DEGREE。
```

#### 模型训练代码

```python
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torchcrf import CRF

class MedicalNERModel(nn.Module):
    def __init__(self, num_tags: int, dropout: float = 0.3):
        super().__init__()
        
        # BERT 编码器
        self.bert = BertModel.from_pretrained(
            'hfl/chinese-bert-wwm-ext',
            output_hidden_states=False
        )
        
        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # 输出层
        self.classifier = nn.Linear(512, num_tags)
        
        # CRF 层
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_output = outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # BiLSTM
        lstm_output, _ = self.lstm(bert_output)  # [batch, seq_len, 512]
        
        # 分类
        emissions = self.classifier(lstm_output)  # [batch, seq_len, num_tags]
        
        # CRF 解码
        if labels is not None:
            # 训练模式：返回负对数似然损失
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            # 推理模式：返回最优路径
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction

# 训练配置
model = MedicalNERModel(num_tags=49)  # 9 种实体 × 5 种标签 (B/I/E/S/O) + 特殊标签
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

---

### 2.4 关系抽取模块

#### 关系类型定义

| 关系类型 | 头实体 | 尾实体 | 示例 |
|----------|--------|--------|------|
| 症状 - 部位 | SYMPTOM | LOCATION | 视力下降→双眼 |
| 症状 - 时间 | SYMPTOM | TIME | 下降→3 年 |
| 症状 - 程度 | SYMPTOM | DEGREE | 下降→渐进性 |
| 病变 - 部位 | LESION | LOCATION | 出血→黄斑区 |
| 病变 - 分布 | LESION | DEGREE | 微动脉瘤→散在 |
| 诊断 - 分期 | DIAGNOSIS | DEGREE | DR→2 期 |
| 诊断 - 部位 | DIAGNOSIS | LOCATION | DR→左眼 |

#### 关系抽取模型

**方法**：基于 BERT 的关系分类

```python
class RelationExtractor(nn.Module):
    def __init__(self, num_relations: int):
        super().__init__()
        
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        
        # 关系分类头
        self.classifier = nn.Sequential(
            nn.Linear(768 * 3, 256),  # [CLS] + 头实体 + 尾实体
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_relations)
        )
        
    def forward(self, input_ids, attention_mask, 
                head_positions, tail_positions):
        """
        head_positions: [batch, 2] 头实体起止位置
        tail_positions: [batch, 2] 尾实体起止位置
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq, 768]
        
        # 提取 [CLS] 表示
        cls_repr = sequence_output[:, 0, :]  # [batch, 768]
        
        # 提取头实体表示（平均池化）
        head_reprs = []
        for i in range(input_ids.shape[0]):
            start, end = head_positions[i]
            head_repr = sequence_output[i, start:end+1, :].mean(dim=0)
            head_reprs.append(head_repr)
        head_repr = torch.stack(head_reprs)  # [batch, 768]
        
        # 提取尾实体表示
        tail_reprs = []
        for i in range(input_ids.shape[0]):
            start, end = tail_positions[i]
            tail_repr = sequence_output[i, start:end+1, :].mean(dim=0)
            tail_reprs.append(tail_repr)
        tail_repr = torch.stack(tail_reprs)  # [batch, 768]
        
        # 拼接表示
        combined = torch.cat([cls_repr, head_repr, tail_repr], dim=-1)
        
        # 分类
        logits = self.classifier(combined)
        
        return logits
```

---

### 2.5 完整抽取流程

```python
from typing import List, Dict
import re

class SymptomExtractor:
    def __init__(self, ner_model_path: str, re_model_path: str):
        # 加载 NER 模型
        self.ner_model = MedicalNERModel.load_from_checkpoint(ner_model_path)
        self.ner_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        
        # 加载关系抽取模型
        self.re_model = RelationExtractor.load_from_checkpoint(re_model_path)
        self.re_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        
        # 医学缩写映射
        self.abbreviations = {
            'OD': '右眼', 'OS': '左眼', 'OU': '双眼',
            'DR': '糖尿病视网膜病变', 'DME': '糖尿病性黄斑水肿',
            'PDR': '增殖期糖尿病视网膜病变', 'NPDR': '非增殖期糖尿病视网膜病变'
        }
        
    def preprocess(self, text: str) -> str:
        """文本预处理"""
        # 扩展医学缩写
        for abbr, full in self.abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
        
        # 标准化空格和标点
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict]:
        """命名实体识别"""
        inputs = self.ner_tokenizer(
            text, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            predictions = self.ner_model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
        
        # 解码 BIOES 标签
        entities = self._decode_bioes(
            text, 
            predictions[0], 
            self.ner_tokenizer
        )
        
        return entities
    
    def extract_relations(self, text: str, 
                         entities: List[Dict]) -> List[Dict]:
        """关系抽取"""
        relations = []
        
        # 枚举实体对
        for i, head_ent in enumerate(entities):
            for j, tail_ent in enumerate(entities):
                if i == j:
                    continue
                
                # 判断关系类型
                relation_type = self._predict_relation(
                    text, head_ent, tail_ent
                )
                
                if relation_type != 'None':
                    relations.append({
                        'head': head_ent,
                        'tail': tail_ent,
                        'relation': relation_type,
                        'confidence': self._get_confidence(relation_type),
                        'source_sentence': self._get_source_sentence(text, head_ent, tail_ent)
                    })
        
        return relations
    
    def _get_confidence(self, relation_type: str) -> float:
        """获取关系置信度（简化版本）"""
        confidence_map = {
            '症状 - 部位': 0.92,
            '症状 - 时间': 0.89,
            '症状 - 程度': 0.87,
            '病变 - 部位': 0.91,
            '病变 - 分布': 0.88,
            '诊断 - 分期': 0.95,
            '诊断 - 部位': 0.93,
            'None': 0.0
        }
        return confidence_map.get(relation_type, 0.5)
    
    def _get_source_sentence(self, text: str, head_ent: Dict, tail_ent: Dict) -> str:
        """提取包含两个实体的句子"""
        sentences = re.split(r'[。！？.!?]', text)
        for sent in sentences:
            if head_ent['text'] in sent and tail_ent['text'] in sent:
                return sent.strip() + '。'
        return text[:50] + '...'
    
    def _decode_bioes(self, text: str, tags: List[int], tokenizer) -> List[Dict]:
        """解码 BIOES 标签为实体列表"""
        # 简化实现
        entities = []
        current_entity = None
        
        id2label = {
            0: 'O', 1: 'B-SYMPTOM', 2: 'I-SYMPTOM', 3: 'E-SYMPTOM', 4: 'S-SYMPTOM',
            5: 'B-SIGN', 6: 'I-SIGN', 7: 'E-SIGN', 8: 'S-SIGN',
            9: 'B-LESION', 10: 'I-LESION', 11: 'E-LESION', 12: 'S-LESION',
            13: 'B-DIAGNOSIS', 14: 'I-DIAGNOSIS', 15: 'E-DIAGNOSIS', 16: 'S-DIAGNOSIS',
            17: 'B-LOCATION', 18: 'I-LOCATION', 19: 'E-LOCATION', 20: 'S-LOCATION',
            21: 'B-TIME', 22: 'I-TIME', 23: 'E-TIME', 24: 'S-TIME',
            25: 'B-DEGREE', 26: 'I-DEGREE', 27: 'E-DEGREE', 28: 'S-DEGREE'
        }
        
        tokens = tokenizer.convert_ids_to_tokens(tags)
        
        for i, tag_id in enumerate(tags):
            tag = id2label.get(tag_id, 'O')
            
            if tag.startswith('B-') or tag.startswith('S-'):
                if current_entity:
                    entities.append(current_entity)
                entity_type = tag.split('-')[1]
                current_entity = {
                    'type': entity_type,
                    'text': tokens[i],
                    'start': i,
                    'end': i
                }
                if tag.startswith('S-'):
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith('I-') or tag.startswith('E-'):
                if current_entity:
                    current_entity['text'] += tokens[i]
                    current_entity['end'] = i
                    if tag.startswith('E-'):
                        entities.append(current_entity)
                        current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def extract(self, text: str) -> Dict:
        """完整抽取流程"""
        # 预处理
        text = self.preprocess(text)
        
        # 实体抽取
        entities = self.extract_entities(text)
        
        # 关系抽取
        relations = self.extract_relations(text, entities)
        
        # 结构化输出
        result = {
            'original_text': text,
            'entities': entities,
            'relations': relations,
            'symptoms': [e for e in entities if e['type'] == 'SYMPTOM'],
            'signs': [e for e in entities if e['type'] == 'SIGN'],
            'lesions': [e for e in entities if e['type'] == 'LESION'],
            'diagnoses': [e for e in entities if e['type'] == 'DIAGNOSIS']
        }
        
        return result


# 使用示例
if __name__ == '__main__':
    extractor = SymptomExtractor(
        ner_model_path='models/ner_model.pt',
        re_model_path='models/re_model.pt'
    )
    
    report_text = """
    患者双眼视力渐进性下降 3 年，近 1 月加重。
    眼底检查：双眼视盘边界清，C/D=0.3，视网膜静脉迂曲扩张，
    可见散在微动脉瘤及点片状出血，黄斑区可见硬性渗出。
    诊断：双眼糖尿病视网膜病变（左眼 2 期，右眼 1 期）
    """
    
    result = extractor.extract(report_text)
    
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 三、关联分析结果可视化

### 3.1 关联网络图

**用途**：展示图像特征与症状之间的关联网络。

**技术实现**：
- **前端**: D3.js / ECharts Graph
- **节点**: 图像特征（圆形）、症状（方形）
- **边**: 关联强度（粗细）、关联方向（箭头）
- **交互**: 悬停显示详情、点击过滤、拖拽重排

**示例配置**：
```javascript
const correlationGraph = {
  nodes: [
    { id: 'img_ma', label: '微动脉瘤数量', type: 'image', size: 20 },
    { id: 'img_bleed', label: '出血点数量', type: 'image', size: 18 },
    { id: 'sym_dm_duration', label: '糖尿病病程', type: 'symptom', size: 22 },
    { id: 'sym_hba1c', label: 'HbA1c', type: 'symptom', size: 19 }
  ],
  links: [
    { source: 'img_ma', target: 'sym_dm_duration', value: 0.87, p_value: 0.001 },
    { source: 'img_ma', target: 'sym_hba1c', value: 0.74, p_value: 0.005 },
    { source: 'img_bleed', target: 'sym_dm_duration', value: 0.71, p_value: 0.008 }
  ]
};
```

---

### 3.2 热力图矩阵

**用途**：展示所有图像 - 症状对的关联强度矩阵。

**技术实现**：
- **前端**: ECharts Heatmap / Plotly
- **X 轴**: 症状特征
- **Y 轴**: 图像特征
- **颜色**: 关联强度（红→强，蓝→弱）
- **交互**: 悬停显示数值、点击钻取详情

**示例配置**：
```javascript
const heatmapData = [
  [0, 0, 0.87], [0, 1, 0.74], [0, 2, 0.65],  // 微动脉瘤 vs 各症状
  [1, 0, 0.71], [1, 1, 0.68], [1, 2, 0.59],  // 出血点 vs 各症状
  [2, 0, 0.82], [2, 1, 0.76], [2, 2, 0.70]   // DR 分级 vs 各症状
];

const heatmapOption = {
  tooltip: {
    formatter: function (param) {
      return `图像特征：${imageFeatures[param.value[0]]}<br/>
              症状：${symptomFeatures[param.value[1]]}<br/>
              关联强度：${param.value[2]}`;
    }
  },
  visualMap: {
    min: 0,
    max: 1,
    calculable: true,
    orient: 'horizontal',
    left: 'center',
    bottom: '10%',
    inRange: { color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'] }
  }
};
```

---

### 3.3 证据链展示

**用途**：展示关联分析的依据和推理过程。

**展示内容**：
```
┌─────────────────────────────────────────────────────────────────┐
│  关联证据链：微动脉瘤数量 ↔ 糖尿病病程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 统计证据                                                    │
│  • 相关系数：r = 0.87 (p < 0.001)                              │
│  • 样本量：n = 1,247 例患者                                    │
│  • 置信区间：[0.83, 0.90] (95% CI)                             │
│                                                                 │
│  📚 文献支持                                                    │
│  • Early Treatment Diabetic Retinopathy Study (ETDRS):         │
│    "微动脉瘤是 DR 最早出现的体征，与病程显著相关"                │
│  • UKPDS 研究：病程每增加 5 年，DR 风险增加 1.8 倍                 │
│                                                                 │
│  🏥 本中心数据                                                  │
│  • 2024-2025 年数据：病程>10 年患者微动脉瘤检出率 87%            │
│  • 病程<5 年患者微动脉瘤检出率 34%                              │
│                                                                 │
│  🤖 模型注意力                                                  │
│  • CCA 权重：图像侧 0.82，症状侧 0.79                            │
│  • GNN 注意力：路径可信度 0.91                                  │
│                                                                 │
│  ⚠️ 注意事项                                                    │
│  • 相关性≠因果性                                                │
│  • 需排除其他混杂因素（血糖控制、血压等）                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、API 接口设计

### 4.1 关联分析 API

```yaml
POST /api/v1/analysis/correlation

请求体:
  patient_id: string          # 患者 ID
  image_features: object      # 图像特征
    dr_grade: number
    microaneurysm_count: number
    hemorrhage_count: number
    # ...
  symptom_features: object    # 症状特征
    dm_duration_years: number
    hba1c: number
    sbp: number
    # ...

响应:
  correlation_score: number   # 综合关联评分 (0-1)
  correlation_level: string   # 极强/强/中等/弱/无
  feature_correlations: array
    - image_feature: string
      symptom_feature: string
      score: number
      p_value: number
      ci_95: [number, number]
  evidence_chain: object
    statistical_evidence: string
    literature_evidence: array
    local_data_evidence: string
  visualization_urls: object
    network_graph: string
    heatmap: string
```

---

### 4.2 症状抽取 API

```yaml
POST /api/v1/nlp/symptom-extract

请求体:
  report_text: string         # 医生报告文本
  report_type: string         # 报告类型 (初诊/复诊/手术)

响应:
  entities: array
    - text: string
      type: string            # SYMPTOM/SIGN/LESION/DIAGNOSIS
      start_pos: number
      end_pos: number
      confidence: number
  relations: array
    - head_entity: object
      tail_entity: object
      relation_type: string
      confidence: number
  structured_output: object
    symptoms: array
    signs: array
    lesions: array
    diagnoses: array
  source_annotations: object  # 原文标注（用于人工审核）
    highlighted_html: string
```

---

## 五、评估指标

### 5.1 关联分析评估

| 指标 | 计算方法 | 目标值 |
|------|----------|--------|
| 关联强度准确性 | 与专家标注的相关系数对比 | r > 0.8 |
| 统计显著性 | 正确识别 p<0.05 的关联 | >90% |
| 临床一致性 | 与临床指南的一致性 | >85% |
| 可解释性评分 | 医生对证据链的满意度 | >4/5 分 |

### 5.2 症状抽取评估

| 指标 | 计算方法 | 目标值 |
|------|----------|--------|
| NER - Precision | TP/(TP+FP) | >0.90 |
| NER - Recall | TP/(TP+FN) | >0.88 |
| NER - F1 | 2·P·R/(P+R) | >0.89 |
| RE - Accuracy | 正确关系数/总关系数 | >0.85 |
| 端到端 F1 | 完整抽取正确的样本比例 | >0.82 |

---

## 六、实施计划

### 阶段 1：数据准备（4 周）
- [ ] 收集 1000+ 例标注报告
- [ ] 构建图像 - 症状配对数据集
- [ ] 制定标注规范
- [ ] 培训标注人员

### 阶段 2：模型开发（8 周）
- [ ] NER 模型训练与调优
- [ ] 关系抽取模型训练
- [ ] CCA/MKL/GNN 关联算法实现
- [ ] 模型集成与融合

### 阶段 3：系统集成（4 周）
- [ ] API 接口开发
- [ ] 前端可视化组件
- [ ] 与现有系统对接
- [ ] 性能优化

### 阶段 4：临床验证（4 周）
- [ ] 回顾性验证（500 例）
- [ ] 前瞻性验证（100 例）
- [ ] 医生反馈收集
- [ ] 模型迭代优化

---

## 七、风险与应对

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|----------|
| 标注数据不足 | 高 | 中 | 数据增强、迁移学习、主动学习 |
| 模型泛化能力差 | 高 | 中 | 多中心数据、域适应、正则化 |
| 医生接受度低 | 中 | 中 | 可解释性增强、用户培训、渐进式部署 |
| 计算资源不足 | 中 | 低 | 模型蒸馏、量化、云端部署 |
| 隐私合规风险 | 高 | 低 | 数据脱敏、本地部署、合规审查 |

---

*文档版本：v1.0 | 最后更新：2026-04-01 | 作者：AI 辅助设计*