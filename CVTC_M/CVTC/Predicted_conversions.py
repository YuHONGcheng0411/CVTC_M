import torch
from torch import nn, einsum
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# Lambda 层，用于执行自定义操作（如转置）
class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# 多尺度卷积嵌入
class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=2):
        super(CrossEmbedLayer, self).__init__()
        kernel_size = sorted(kernel_size)
        num_scales = len(kernel_size)
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.conv = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_size, dim_scales):
            self.conv.append(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=kernel, stride=stride, padding=(kernel - stride) // 2)
            )

    def forward(self, x):
        f = tuple(map(lambda conv: conv(x), self.conv))
        return torch.cat(f, dim=1)

# 动态位置偏置
def DynamicPositionBias(dim):
    return nn.Sequential(
        nn.Linear(2, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, 1),
        nn.Flatten(start_dim=0)
    )

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return ((x - mean) / (var + self.eps).sqrt()) * self.g + self.b

# 前馈传播
def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )

# 注意力机制
class Attention(nn.Module):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.):
        super(Attention, self).__init__()
        assert attn_type in {'short', 'long'}, '注意力类型必须是long或short'
        heads = dim // dim_head
        assert dim >= dim_head, 'dim 必须大于等于 dim_head'
        if heads == 0:
            raise ValueError('头数不能为零，请确保 dim >= dim_head')
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.attn_type = attn_type
        self.window_size = window_size
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dpb = DynamicPositionBias(dim // 4)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        _, w1, w2 = grid.size()
        grid = grid.view(-1, w1 * w2).permute(1, 0).contiguous()
        rel_pos = grid.view(w1 * w2, 1, 2) - grid.view(1, w1 * w2, 2)
        rel_pos = rel_pos + window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        b, dim, h, w, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device
        x = self.norm(x)
        if self.attn_type == 'short':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)
        elif self.attn_type == 'long':
            x = x.view(b, dim, h // wsz, wsz, w // wsz, wsz)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, dim, wsz, wsz)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda x: x.view(-1, self.heads, wsz * wsz, self.dim_head), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        _, size1, size2 = rel_pos.size()
        rel_pos = rel_pos.permute(1, 2, 0).view(size1 * size2, 2)
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]
        sim = sim + rel_pos_bias
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(-1, self.heads * self.dim_head, wsz, wsz)
        out = self.to_out(out)
        if self.attn_type == 'short':
            b, d, h, w = b, dim, h // wsz, w // wsz
            out = out.view(b, h, w, d, wsz, wsz)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(b, d, h * wsz, w * wsz)
        elif self.attn_type == 'long':
            b, d, l1, l2 = b, dim, h // wsz, w // wsz
            out = out.view(b, l1, l2, d, wsz, wsz)
            out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = out.view(b, d, l1 * wsz, l2 * wsz)
        return out

# Transformer模块
class Transformer(nn.Module):
    def __init__(self, dim, local_window_size, global_window_size, depth=4, dim_head=32, attn_dropout=0., ff_dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attn_type='short', window_size=local_window_size, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout),
                Attention(dim=dim, attn_type='long', window_size=global_window_size, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout)
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x
        return x

# 交叉注意力融合模块
class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.2):
        super(CrossAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.scale = (feature_dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(feature_dim, feature_dim * 3, bias=False)
        self.to_out = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, image_features, protein_features, importance_features, apoe_features):
        features = torch.stack([image_features, protein_features, importance_features, apoe_features], dim=1)  # [B, 4, feature_dim]
        B, N, D = features.shape
        features = self.norm(features)
        qkv = self.to_qkv(features).view(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.to_out(out)
        fused_features = out.mean(dim=1)  # [B, feature_dim]
        return fused_features

# 多模态诊断网络
class MultimodalDiagnosisNetwork(nn.Module):
    def __init__(
        self,
        dim=(32, 64, 128, 256),
        depth=(2, 2, 2, 2),
        global_window_size=(8, 4, 2, 1),
        local_window_size=16,
        cross_embed_kernel_sizes=((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        num_classes=3,
        attn_dropout=0.2,
        ff_dropout=0.2,
        channels=3,
        protein_dim=5,
        importance_dim=(87, 24),  # SNP 数据形状
        apoe_dim=7,
        feature_dim=256
    ):
        super(MultimodalDiagnosisNetwork, self).__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # 图像处理分支
        self.image_layers = nn.ModuleList([])
        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(
            dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides
        ):
            self.image_layers.append(nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                           attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]))

        # 蛋白数据处理分支
        self.protein_fc = nn.Sequential(
            nn.Linear(protein_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(attn_dropout)
        )

        # 重要性数据处理分支（SNP）
        self.importance_fc = nn.Sequential(
            nn.Linear(importance_dim[1], feature_dim),  # [B, 87, 24] -> [B, 87, 256]
            nn.LayerNorm(feature_dim),  # 归一化 [B, 87, 256]
            nn.ReLU(),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=4,
                    dim_feedforward=feature_dim * 4,
                    dropout=attn_dropout,
                    activation='relu'
                ),
                num_layers=2
            ),  # [B, 87, 256] -> [B, 87, 256]
            Lambda(lambda x: x.permute(0, 2, 1)),  # [B, 87, 256] -> [B, 256, 87]
            nn.AdaptiveAvgPool1d(1),  # [B, 256, 87] -> [B, 256, 1]
            nn.Flatten(),  # [B, 256, 1] -> [B, 256]
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(attn_dropout)
        )

        # APOE表达数据处理分支
        self.apoe_fc = nn.Sequential(
            nn.Linear(apoe_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(attn_dropout)
        )

        # 交叉注意力融合层
        self.fusion_layer = CrossAttentionFusion(feature_dim=feature_dim, num_heads=4, dropout=attn_dropout)

        # 分类层
        self.to_logits = nn.Linear(feature_dim, num_classes)

    def forward(self, image, protein, importance, apoe):
        # 图像处理
        x = image
        for cel, transformer in self.image_layers:
            x = cel(x)
            x = transformer(x)
        image_features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])  # [B, last_dim]

        # 蛋白数据处理
        protein_features = self.protein_fc(protein)  # [B, feature_dim]

        # 重要性数据处理（SNP）
        importance_features = self.importance_fc(importance)  # [B, 87, 24] -> [B, 256]

        # APOE表达数据处理
        apoe_features = self.apoe_fc(apoe)  # [B, feature_dim]

        # 交叉注意力融合
        fused_features = self.fusion_layer(image_features, protein_features, importance_features, apoe_features)  # [B, feature_dim]

        # 分类输出
        logits = self.to_logits(fused_features)  # [B, num_classes]
        return logits

    def extract_features(self, image, protein, importance, apoe):
        x = image
        for cel, transformer in self.image_layers:
            x = cel(x)
            x = transformer(x)
        image_features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        protein_features = self.protein_fc(protein)
        importance_features = self.importance_fc(importance)
        apoe_features = self.apoe_fc(apoe)
        fused_features = self.fusion_layer(image_features, protein_features, importance_features, apoe_features)
        return fused_features

# 多模态诊断数据集
class MultimodalDiagnosisDataset(Dataset):
    def __init__(self, image_root_dir, protein_csv, snp_root_dir, apoe_csv, snp_augment_dir=None, augment_prob=0.3, transform=None, min_time_points=2):
        """
        参数：
        - image_root_dir: 图像数据的大文件夹路径
        - protein_csv: 蛋白数据CSV文件路径
        - snp_root_dir: SNP数据的大文件夹路径（包含以ID命名的CSV文件）
        - apoe_csv: APOE表达数据CSV文件路径
        - snp_augment_dir: SNP增强数据的大文件夹路径（包含以类别命名的子文件夹）
        - augment_prob: SNP增强替换的概率
        - transform: 图像预处理变换
        - min_time_points: 最小时间点数（过滤少于此数量的个体）
        """
        self.image_root_dir = image_root_dir
        self.snp_root_dir = snp_root_dir
        self.snp_augment_dir = snp_augment_dir
        self.augment_prob = augment_prob
        self.transform = transform
        self.min_time_points = min_time_points

        # 读取CSV文件
        self.protein_df = pd.read_csv(protein_csv)
        self.apoe_df = pd.read_csv(apoe_csv)

        # 打印APOE CSV的列信息以便调试
        print(f"APOE CSV 列数: {len(self.apoe_df.columns)}")
        print(f"APOE CSV 列名: {list(self.apoe_df.columns)}")

        # 删除蛋白CSV中标签为'Unknown'的行
        self.protein_df = self.protein_df[self.protein_df.iloc[:, -2] != 'Unknown'].reset_index(drop=True)
        print(f"删除'Unknown'类别后，蛋白CSV剩余 {len(self.protein_df)} 行")

        # 获取标签的唯一值并创建映射
        self.label_map = {}
        unique_labels = self.protein_df.iloc[:, -2].dropna().unique()
        print(f"标签唯一值: {unique_labels}")
        for idx, label in enumerate(sorted(unique_labels)):
            self.label_map[label] = idx
        print(f"标签映射: {self.label_map}")

        # 检查标签数量
        if len(self.label_map) != 3:
            print(f"警告：标签类别数为 {len(self.label_map)}，与模型的num_classes=3不匹配")

        # 填补蛋白CSV中的缺失值
        protein_cols = [8, 9, 10, 13, 14]
        for col in protein_cols:
            if col < len(self.protein_df.columns):
                col_name = self.protein_df.columns[col]
                median_by_class = self.protein_df.groupby(self.protein_df.iloc[:, -2])[col_name].median()
                for label in median_by_class.index:
                    self.protein_df.loc[
                        (self.protein_df.iloc[:, -2] == label) & (self.protein_df[col_name].isnull()),
                        col_name
                    ] = median_by_class[label]
            else:
                print(f"警告：蛋白CSV列索引 {col} 越界，最大索引为 {len(self.protein_df.columns)-1}")

        # 处理APOE CSV中的GENOTYPE列，进行独热编码
        self.apoe_df['label'] = self.apoe_df.iloc[:, 1].map(
            lambda x: self.protein_df[
                self.protein_df.iloc[:, 1].str.startswith(x + '_')
            ].iloc[:, -2].iloc[0] if any(self.protein_df.iloc[:, 1].str.startswith(x + '_')) else None
        )
        self.apoe_df = self.apoe_df[self.apoe_df['label'].notnull()].reset_index(drop=True)

        # 检查GENOTYPE列（索引4）是否存在并进行独热编码
        genotype_cols = []
        if 4 < len(self.apoe_df.columns):
            genotype_values = self.apoe_df.iloc[:, 4].dropna().unique()
            print(f"GENOTYPE唯一值: {genotype_values}")
            if len(genotype_values) == 0:
                print("警告：GENOTYPE列所有值均为NaN，跳过独热编码")
            else:
                for genotype in genotype_values:
                    self.apoe_df[f'is_{genotype.replace("/", "_")}'] = (self.apoe_df.iloc[:, 4] == genotype).astype(int)
                genotype_cols = [col for col in self.apoe_df.columns if col.startswith('is_')]
        else:
            print("警告：GENOTYPE列索引4越界，跳过独热编码")

        # 原始APOE数值列，固定为索引6到12
        apoe_cols = list(range(6, 13))
        print(f"APOE原始数值列索引: {apoe_cols}")
        print(f"GENOTYPE独热编码列: {genotype_cols}")

        # 填补APOE CSV中的缺失值（包括原始列和独热编码列）
        all_cols = apoe_cols + [self.apoe_df.columns.get_loc(col) for col in genotype_cols]
        for col in all_cols:
            if col < len(self.apoe_df.columns):
                col_name = self.apoe_df.columns[col]
                median_by_class = self.apoe_df.groupby('label')[col_name].median()
                for label in median_by_class.index:
                    self.apoe_df.loc[
                        (self.apoe_df['label'] == label) & (self.apoe_df[col_name].isnull()),
                        col_name
                    ] = median_by_class[label]
            else:
                print(f"警告：APOE CSV列索引 {col} 越界，最大索引为 {len(self.apoe_df.columns)-1}")
        self.apoe_df = self.apoe_df.drop(columns=['label'])

        # 存储GENOTYPE独热编码列的名称
        self.genotype_cols = genotype_cols

        # 初始化SNP增强文件列表
        self.augment_snp_files = {}
        if self.snp_augment_dir:
            for label in self.label_map:
                label_dir = os.path.join(self.snp_augment_dir, label)
                if os.path.isdir(label_dir):
                    self.augment_snp_files[label] = [
                        os.path.join(label_dir, f) for f in os.listdir(label_dir)
                        if f.lower().endswith('.csv')
                    ]
                    print(f"类别 {label} 的增强SNP文件数: {len(self.augment_snp_files[label])}")
                else:
                    print(f"警告：增强数据文件夹 {label_dir} 不存在")
                    self.augment_snp_files[label] = []

        # 按 short_id 组织时间序列，记录每个 short_id 的 img_id 和时间点
        self.time_series = {}
        self.image_paths = []
        self.image_ids = []
        self.short_ids = []
        self.snp_files = []
        for folder in os.listdir(image_root_dir):
            folder_path = os.path.join(image_root_dir, folder)
            if os.path.isdir(folder_path):
                if folder in self.protein_df.iloc[:, 1].values:
                    short_id = '_'.join(folder.split('_')[:3])  # 如 941_S_1202
                    snp_file = os.path.join(snp_root_dir, f"{short_id}.csv")
                    if (short_id in self.apoe_df.iloc[:, 1].values) and os.path.exists(snp_file):
                        if short_id not in self.time_series:
                            self.time_series[short_id] = []
                        # 添加该文件夹（img_id）对应的图像
                        for img_name in os.listdir(folder_path):
                            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                                self.time_series[short_id].append({
                                    'img_id': folder,  # 完整 img_id，如 941_S_1202_20070815
                                    'img_path': os.path.join(folder_path, img_name),
                                    'timestamp': folder.split('_')[-1]  # 提取时间戳，如 20070815
                                })

        # 过滤掉时间点少于 min_time_points 的个体，基于唯一时间戳计数
        valid_short_ids = []
        for short_id in self.time_series:
            # 获取该 short_id 的唯一时间戳
            unique_timestamps = set(item['timestamp'] for item in self.time_series[short_id])
            if len(unique_timestamps) >= min_time_points:
                valid_short_ids.append(short_id)
        # 展开有效 short_id 的所有时间点样本
        for short_id in valid_short_ids:
            snp_file = os.path.join(snp_root_dir, f"{short_id}.csv")
            for time_point in self.time_series[short_id]:
                self.image_ids.append(time_point['img_id'])
                self.image_paths.append(time_point['img_path'])
                self.short_ids.append(short_id)
                self.snp_files.append(snp_file)
        print(f"过滤后，保留 {len(valid_short_ids)} 个个体（至少 {min_time_points} 个时间点）")
        print(f"总样本数（单时间点）: {len(self.image_paths)}")
        print(f"找到 {len(self.image_paths)} 个有效图像样本，匹配 {len(set(self.snp_files))} 个SNP文件")

        # 数据完整性检查
        if self.protein_df.iloc[:, protein_cols].isnull().any().any():
            print("警告：蛋白CSV填补后仍存在缺失值")
        if apoe_cols and max(apoe_cols) < len(self.apoe_df.columns):
            if self.apoe_df.iloc[:, apoe_cols].isnull().any().any():
                print("警告：APOE原始数值列填补后仍存在缺失值")
        if genotype_cols and max(self.apoe_df.columns.get_loc(col) for col in genotype_cols) < len(self.apoe_df.columns):
            if self.apoe_df[genotype_cols].isnull().any().any():
                print("警告：GENOTYPE独热编码列填补后仍存在缺失值")
        if self.protein_df.iloc[:, -2].isnull().any():
            print("警告：标签列中存在缺失值")

        # 检查SNP文件的列一致性
        snp_columns = None
        for snp_file in set(self.snp_files):
            snp_df = pd.read_csv(snp_file)
            if snp_columns is None:
                snp_columns = set(snp_df.columns)
            elif set(snp_df.columns) != snp_columns:
                print(f"警告：SNP文件 {snp_file} 的列与预期不一致")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        short_id = self.short_ids[idx]
        img_id = self.image_ids[idx]
        img_path = self.image_paths[idx]
        snp_file = self.snp_files[idx]

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        except Exception as e:
            print(f"加载图像 {img_path} 失败: {e}")
            raise

        # 获取蛋白数据
        protein_row = self.protein_df[self.protein_df.iloc[:, 1] == img_id]
        try:
            protein_data = protein_row.iloc[:, [8, 9, 10, 13, 14]].values.astype(float)
            protein_data = torch.tensor(protein_data, dtype=torch.float32).squeeze()
        except Exception as e:
            print(f"处理蛋白数据 {img_id} 失败: {e}")
            raise

        # 获取标签（使用该个体的最后一个时间点的标签）
        try:
            time_points = sorted(
                [(tp['img_id'], tp['img_path']) for tp in self.time_series[short_id]],
                key=lambda x: x[0].split('_')[-1]
            )
            last_img_id = time_points[-1][0]
            label_str = self.protein_df[self.protein_df.iloc[:, 1] == last_img_id].iloc[:, -2].values[0]
            if label_str not in self.label_map:
                raise ValueError(f"未知标签 {label_str} 在ID {last_img_id}")
            label = self.label_map[label_str]
            label = torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"处理标签 {short_id} 失败: {e}")
            raise

        # 获取SNP数据并填补NaN/无穷大值
        try:
            # 根据增强概率和标签决定是否替换SNP文件
            if (self.snp_augment_dir and
                label_str in self.augment_snp_files and
                self.augment_snp_files[label_str] and
                random.random() < self.augment_prob):
                snp_file = random.choice(self.augment_snp_files[label_str])
                # print(f"使用增强SNP文件: {snp_file} (类别: {label_str})")
            snp_df = pd.read_csv(snp_file)
            snp_data = snp_df.values.astype(float)  # [num_snps, num_features]
            if np.any(np.isnan(snp_data)) or np.any(np.isinf(snp_data)):
                col_medians = np.nanmedian(snp_data, axis=0)
                nan_mask = np.isnan(snp_data)
                snp_data[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
                inf_mask = np.isinf(snp_data)
                snp_data[inf_mask] = np.take(col_medians, np.where(inf_mask)[1])
            if np.any(np.isnan(snp_data)) or np.any(np.isinf(snp_data)):
                raise ValueError(f"SNP数据 {snp_file} 填补后仍包含NaN或无穷大值")
            snp_data = torch.tensor(snp_data, dtype=torch.float32)  # [num_snps, num_features]
        except Exception as e:
            print(f"处理SNP数据 {snp_file} 失败: {e}")
            raise

        # 获取APOE表达数据（原始数值列 + 独热编码GENOTYPE列）
        apoe_row = self.apoe_df[self.apoe_df.iloc[:, 1] == short_id]
        try:
            # 获取原始APOE数值列
            apoe_cols = list(range(6, 13))
            apoe_numeric_data = apoe_row.iloc[:, apoe_cols].values.astype(float)
            # 获取GENOTYPE独热编码列
            genotype_data = apoe_row[self.genotype_cols].values.astype(float) if self.genotype_cols else np.array([])
            # 合并数据
            apoe_data = np.concatenate([apoe_numeric_data, genotype_data], axis=1) if genotype_data.size else apoe_numeric_data
            apoe_data = torch.tensor(apoe_data, dtype=torch.float32).squeeze()
        except Exception as e:
            print(f"处理APOE数据 {short_id} 失败: {e}")
            raise

        return {
            'image': image,
            'protein': protein_data,
            'snp': snp_data,
            'apoe': apoe_data,
            'label': label
        }

# 自定义collate_fn
def custom_collate_fn(batch):
    """
    自定义批处理函数，将数据集的批次数据整理为张量。

    参数：
    - batch: 从 MultimodalDiagnosisDataset 返回的样本列表

    返回：
    - 字典，包含以下键：
        - images: 图像张量 [batch_size, 3, H, W]
        - proteins:  protein_data张量 [batch_size, 5]
        - snps: SNP 数据张量 [batch_size, num_snps, num_features]
        - apoes: APOE 数据张量 [batch_size, 7]
        - labels: 标签张量 [batch_size]
    """
    images = torch.stack([item['image'] for item in batch])
    proteins = torch.stack([item['protein'] for item in batch])
    snps = torch.stack([item['snp'] for item in batch])  # 堆叠为 [batch_size, num_snps, num_features]
    apoes = torch.stack([item['apoe'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        'images': images,
        'proteins': proteins,
        'snps': snps,
        'apoes': apoes,
        'labels': labels
    }

# 数据集划分函数
def split_dataset(dataset, train_ratio=0.9, random_seed=42):
    # 获取所有短ID
    short_ids = dataset.short_ids
    unique_short_ids = list(set(short_ids))

    # 按短ID划分训练和测试集
    train_ids, test_ids = train_test_split(
        unique_short_ids, train_size=train_ratio, random_state=random_seed
    )

    # 根据短ID获取训练和测试集的索引
    train_indices = [i for i, sid in enumerate(short_ids) if sid in train_ids]
    test_indices = [i for i, sid in enumerate(short_ids) if sid in test_ids]

    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")

    return train_indices, test_indices

# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_acc = 0.0
    best_model_path = "best_time_model.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for batch in train_bar:
            images = batch['images'].to(device)
            proteins = batch['proteins'].to(device)
            proteins=torch.randn_like(proteins)

            snp = batch['snps'].to(device)
            apoes = batch['apoes'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(images, proteins, snp, apoes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                images = batch['images'].to(device)
                proteins = batch['proteins'].to(device)
                snp = batch['snps'].to(device)
                apoes = batch['apoes'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(images, proteins, snp, apoes)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        val_loss = val_loss / len(test_loader.dataset)
        val_acc = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")
        torch.save(model.state_dict(), '多模态-time2.pth')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型，验证准确率: {best_acc:.2f}%")


# 主函数
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = MultimodalDiagnosisDataset(
            image_root_dir=r"F:\dataset\多种模态+SNP",
            protein_csv=r"E:\dataset-CVTC\csv\蛋白2.csv",
            snp_root_dir=r'F:\SNP-处理',
            apoe_csv=r"E:\dataset-CVTC\ApoE （简体中文）基因分型 - 结果 [ADNI1，GO，2,3,4].csv",
            snp_augment_dir=r'F:\dataset\筛选SNP',
            augment_prob=0.5,
            transform=transform,
        )

    train_indices, test_indices = split_dataset(dataset, train_ratio=0.8, random_seed=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(
        MultimodalDiagnosisDataset(
            image_root_dir=r"F:\dataset\多种模态+SNP",
            protein_csv=r"E:\dataset-CVTC\csv\蛋白2.csv",
            snp_root_dir=r'F:\SNP-处理',
            apoe_csv=r"E:\dataset-CVTC\ApoE （简体中文）基因分型 - 结果 [ADNI1，GO，2,3,4].csv",
            transform=transform,
        ),
        test_indices
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

    model = MultimodalDiagnosisNetwork(
        dim=(32, 64, 128, 256),
        depth=(2, 2, 2, 2),
        global_window_size=(8, 4, 2, 1),
        local_window_size=16,
        cross_embed_kernel_sizes=((2, 4, 6, 8), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        num_classes=3,
        attn_dropout=0.2,
        ff_dropout=0.2,
        channels=3,
        protein_dim=5,
        importance_dim=(87,24),
        apoe_dim=13,
        feature_dim=256
    ).to(device)

    # 加载预训练模型（可选）
    model.load_state_dict(torch.load(r"多模态-time2.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, device=device)