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
from datetime import datetime
import random
from sklearn.metrics import mean_squared_error, r2_score

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
                nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=kernel, stride=stride,
                          padding=(kernel - stride) // 2)
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
    def __init__(self, dim, local_window_size, global_window_size, depth=4, dim_head=32, attn_dropout=0.,
                 ff_dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attn_type='short', window_size=local_window_size, dim_head=dim_head,
                          dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout),
                Attention(dim=dim, attn_type='long', window_size=global_window_size, dim_head=dim_head,
                          dropout=attn_dropout),
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
        features = torch.stack([image_features, protein_features, importance_features, apoe_features], dim=1)
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
        fused_features = out.mean(dim=1)
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
                dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes,
                cross_embed_strides
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

        # 分类层（预测当前标签）
        self.to_logits = nn.Linear(feature_dim, num_classes)

        # 回归层（预测转化时间）
        self.to_conversion_time = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(attn_dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.ReLU()  # 确保时间非负
        )

    def forward(self, images, proteins, importances, apoes):
        x = images
        for cel, transformer in self.image_layers:
            x = cel(x)
            x = transformer(x)
        image_features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        protein_features = self.protein_fc(proteins)
        importance_features = self.importance_fc(importances)
        apoe_features = self.apoe_fc(apoes)
        fused_features = self.fusion_layer(image_features, protein_features, importance_features, apoe_features)
        logits = self.to_logits(fused_features)
        conversion_time = self.to_conversion_time(fused_features)
        return logits, conversion_time

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
    def __init__(self, image_root_dir, protein_csv, snp_root_dir, apoe_csv, snp_augment_dir=None, augment_prob=0.3, transform=None, min_time_points=2, mode='label'):
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
        - mode: 数据集模式，'label'（所有时间点，用于标签预测）或 'time'（仅转化前数据，用于转化时间预测）
        """
        self.image_root_dir = image_root_dir
        self.snp_root_dir = snp_root_dir
        self.snp_augment_dir = snp_augment_dir
        self.augment_prob = augment_prob
        self.transform = transform
        self.min_time_points = min_time_points
        self.mode = mode

        if mode not in ['label', 'time']:
            raise ValueError("mode 必须是 'label' 或 'time'")

        # 读取CSV文件
        self.protein_df = pd.read_csv(protein_csv)
        self.apoe_df = pd.read_csv(apoe_csv)

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
            col_name = self.protein_df.columns[col]
            median_by_class = self.protein_df.groupby(self.protein_df.iloc[:, -2])[col_name].median()
            for label in median_by_class.index:
                self.protein_df.loc[
                    (self.protein_df.iloc[:, -2] == label) & (self.protein_df[col_name].isnull()),
                    col_name
                ] = median_by_class[label]

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

        # 按 short_id 组织时间序列，记录每个 short_id 的 img_id、时间点和标签
        self.time_series = {}
        self.image_paths = []
        self.image_ids = []
        self.short_ids = []
        self.conversion_times = []
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
                        for img_name in os.listdir(folder_path):
                            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                                self.time_series[short_id].append({
                                    'img_id': folder,
                                    'img_path': os.path.join(folder_path, img_name),
                                    'timestamp': folder.split('_')[-1],
                                    'label': self.protein_df[self.protein_df.iloc[:, 1] == folder].iloc[:, -2].values[0]
                                })
                                self.image_ids.append(folder)
                                self.short_ids.append(short_id)
                                self.image_paths.append(os.path.join(folder_path, img_name))
                                self.snp_files.append(snp_file)

        # 过滤掉时间点少于 min_time_points 的个体，并计算转化时间
        valid_short_ids = []
        filtered_image_paths = []
        filtered_image_ids = []
        filtered_short_ids = []
        filtered_conversion_times = []
        filtered_snp_files = []
        for short_id in self.time_series:
            unique_timestamps = set(item['timestamp'] for item in self.time_series[short_id])
            if len(unique_timestamps) >= min_time_points:
                time_points = sorted(self.time_series[short_id], key=lambda x: x['timestamp'])
                labels = [item['label'] for item in time_points]
                conversion_timestamp = None
                for i in range(1, len(labels)):
                    if labels[i] != labels[i - 1]:
                        try:
                            conversion_timestamp = time_points[i]['timestamp']
                            break
                        except ValueError as e:
                            print(f"时间戳解析错误 {short_id}: {e}")
                            continue
                for idx, tp in enumerate(time_points):
                    try:
                        current_time = datetime.strptime(tp['timestamp'], '%Y%m%d')
                        if self.mode == 'time' and conversion_timestamp is not None:
                            conv_time = datetime.strptime(conversion_timestamp, '%Y%m%d')
                            if current_time < conv_time:
                                time_to_conversion = (conv_time - current_time).days / 365.25
                                filtered_image_ids.append(tp['img_id'])
                                filtered_image_paths.append(tp['img_path'])
                                filtered_short_ids.append(short_id)
                                filtered_conversion_times.append(time_to_conversion)
                                filtered_snp_files.append(self.snp_files[self.image_paths.index(tp['img_path'])])
                                valid_short_ids.append(short_id)
                        else:
                            if conversion_timestamp is None:
                                t1 = datetime.strptime(time_points[0]['timestamp'], '%Y%m%d')
                                t2 = datetime.strptime(time_points[-1]['timestamp'], '%Y%m%d')
                                time_to_conversion = (t2 - t1).days / 365.25
                            else:
                                conv_time = datetime.strptime(conversion_timestamp, '%Y%m%d')
                                time_to_conversion = (
                                    conv_time - current_time).days / 365.25 if current_time <= conv_time else 0.0
                            filtered_image_ids.append(tp['img_id'])
                            filtered_image_paths.append(tp['img_path'])
                            filtered_short_ids.append(short_id)
                            filtered_conversion_times.append(time_to_conversion)
                            filtered_snp_files.append(self.snp_files[self.image_paths.index(tp['img_path'])])
                            valid_short_ids.append(short_id)
                    except ValueError as e:
                        print(f"时间戳处理错误 {short_id}: {e}")
                        continue
        self.image_ids = filtered_image_ids
        self.image_paths = filtered_image_paths
        self.short_ids = filtered_short_ids
        self.conversion_times = filtered_conversion_times
        self.snp_files = filtered_snp_files
        valid_short_ids = list(set(valid_short_ids))
        print(f"模式: {self.mode}")
        print(f"过滤后，保留 {len(valid_short_ids)} 个个体（至少 {min_time_points} 个时间点）")
        print(f"总样本数（单时间点）: {len(self.image_paths)}")

        # 数据完整性检查
        if self.protein_df.iloc[:, protein_cols].isnull().any().any():
            print("警告：蛋白CSV填补后仍存在缺失值")
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
        conversion_time = self.conversion_times[idx]

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

        # 获取标签（用于SNP增强）
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
            snp_data = snp_df.values.astype(float)
            if np.any(np.isnan(snp_data)) or np.any(np.isinf(snp_data)):
                col_medians = np.nanmedian(snp_data, axis=0)
                nan_mask = np.isnan(snp_data)
                snp_data[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
                inf_mask = np.isinf(snp_data)
                snp_data[inf_mask] = np.take(col_medians, np.where(inf_mask)[1])
            if np.any(np.isnan(snp_data)) or np.any(np.isinf(snp_data)):
                raise ValueError(f"SNP数据 {snp_file} 填补后仍包含NaN或无穷大值")
            snp_data = torch.tensor(snp_data, dtype=torch.float32)
        except Exception as e:
            print(f"处理SNP数据 {snp_file} 失败: {e}")
            raise

        # 获取APOE表达数据（原始数值列 + 独热编码GENOTYPE列）
        apoe_row = self.apoe_df[self.apoe_df.iloc[:, 1] == short_id]
        try:
            # 获取原始APOE数值列
            apoe_numeric_data = apoe_row.iloc[:, 6:13].values.astype(float)
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
            'label': label,
            'conversion_time': torch.tensor(conversion_time, dtype=torch.float32)
        }


# 自定义collate_fn
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    proteins = torch.stack([item['protein'] for item in batch])
    snps = torch.stack([item['snp'] for item in batch])  # Shape: [B, num_snps, num_features]
    apoes = torch.stack([item['apoe'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    conversion_times = torch.stack([item['conversion_time'] for item in batch])
    return {
        'images': images,
        'proteins': proteins,
        'snps': snps,
        'apoes': apoes,
        'labels': labels,
        'conversion_times': conversion_times
    }


# 数据集划分函数（基于短ID）
def split_dataset(dataset, train_ratio=0.9, random_seed=42):
    """
    参数：
    - dataset: MultimodalDiagnosisDataset 实例
    - train_ratio: 训练集比例（默认0.9）
    - random_seed: 随机种子（默认42）
    返回：
    - train_indices: 训练集索引列表
    - test_indices: 测试集索引列表
    """
    # 获取数据集总长度
    total_indices = list(range(len(dataset)))

    # 直接对索引进行随机划分
    train_indices, test_indices = train_test_split(
        total_indices, train_size=train_ratio, random_state=random_seed
    )

    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")

    return train_indices, test_indices


def train_model(model, train_loader, test_loader, criterion_reg, optimizer, num_epochs, device):
    """
    参数：
    - model: 模型
    - train_loader, test_loader: 数据加载器
    - criterion_reg: 回归损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - device: 设备
    """
    best_mae = float('inf')
    best_model_path = "best_time_model-转化.pth"

    for epoch in range(num_epochs):
        model.train()
        running_reg_loss = 0.0
        mae_sum = 0.0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for batch in train_bar:
            images = batch['images'].to(device)
            proteins = batch['proteins'].to(device)
            snp = batch['snps'].to(device)
            apoes = batch['apoes'].to(device)
            conversion_times = batch['conversion_times'].to(device)

            # 对真实时间进行 log 变换
            log_conversion_times = torch.log1p(conversion_times)

            optimizer.zero_grad()
            logits, pred_times = model(images, proteins, snp, apoes)
            reg_loss = criterion_reg(pred_times.squeeze(), log_conversion_times)
            reg_loss.backward()
            optimizer.step()

            running_reg_loss += reg_loss.item() * images.size(0)
            total += images.size(0)
            # 反 log 变换计算 MAE
            pred_times_orig = torch.expm1(pred_times.squeeze())
            mae_sum += torch.abs(pred_times_orig - conversion_times).sum().item()

            train_bar.set_postfix(reg_loss=reg_loss.item(), mae=mae_sum / total)

        epoch_reg_loss = running_reg_loss / len(train_loader.dataset)
        epoch_mae = mae_sum / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Reg Loss (log scale): {epoch_reg_loss:.4f}, Train MAE (years): {epoch_mae:.2f}")

        model.eval()
        val_reg_loss = 0.0
        mae_sum = 0.0
        total = 0

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                images = batch['images'].to(device)
                proteins = batch['proteins'].to(device)
                snp = batch['snps'].to(device)
                apoes = batch['apoes'].to(device)
                conversion_times = batch['conversion_times'].to(device)

                # 对真实时间进行 log 变换
                log_conversion_times = torch.log1p(conversion_times)

                logits, pred_times = model(images, proteins, snp, apoes)
                reg_loss = criterion_reg(pred_times.squeeze(), log_conversion_times)

                val_reg_loss += reg_loss.item() * images.size(0)
                total += images.size(0)
                # 反 log 变换计算 MAE
                pred_times_orig = torch.expm1(pred_times.squeeze())
                mae_sum += torch.abs(pred_times_orig - conversion_times).sum().item()

                val_bar.set_postfix(reg_loss=reg_loss.item(), mae=mae_sum / total)

        val_reg_loss = val_reg_loss / len(test_loader.dataset)
        val_mae = mae_sum / total
        print(
            f"Validation Reg Loss (log scale): {val_reg_loss:.4f}, Validation MAE (years): {val_mae:.2f}")
        torch.save(model.state_dict(), '多模态-time-转化时间-2.pth')

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型，验证MAE: {best_mae:.2f} years")

def eval_model(model, train_loader, test_loader, criterion_reg, optimizer, num_epochs, device):
    """
    参数：
    - model: 模型
    - train_loader, test_loader: 数据加载器
    - criterion_reg: 回归损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - device: 设备
    """

    for epoch in range(num_epochs):
        model.eval()
        val_reg_loss = 0.0
        mae_sum = 0.0
        total = 0
        all_pred_times = []
        all_true_times = []

        val_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                images = batch['images'].to(device)
                proteins = batch['proteins'].to(device)
                snp = batch['snps'].to(device)
                apoes = batch['apoes'].to(device)
                conversion_times = batch['conversion_times'].to(device)

                # 对真实时间进行 log 变换
                log_conversion_times = torch.log1p(conversion_times)

                # 前向传播
                logits, pred_times = model(images, proteins, snp, apoes)
                reg_loss = criterion_reg(pred_times.squeeze(), log_conversion_times)

                # 计算损失
                val_reg_loss += reg_loss.item() * images.size(0)
                total += images.size(0)

                # 反 log 变换计算原始时间尺度的指标（保持为 Tensor）
                pred_times_orig = torch.expm1(pred_times.squeeze())

                # 计算 MAE（在 Tensor 上操作）
                mae_sum += torch.abs(pred_times_orig - conversion_times).sum().item()

                # 收集预测和真实值用于后续指标计算（转换为 NumPy）
                all_pred_times.append(pred_times_orig.cpu().numpy())
                all_true_times.append(conversion_times.cpu().numpy())

                val_bar.set_postfix(reg_loss=reg_loss.item(), mae=mae_sum / total)

        # 计算平均验证损失和 MAE
        val_reg_loss = val_reg_loss / len(test_loader.dataset)
        val_mae = mae_sum / total

        # 合并所有批次的预测和真实值
        all_pred_times = np.concatenate(all_pred_times, axis=0)
        all_true_times = np.concatenate(all_true_times, axis=0)

        # 计算 MSE、RMSE 和 R² 分数
        mse = mean_squared_error(all_true_times, all_pred_times)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_true_times, all_pred_times)

        print(
            f"Validation Reg Loss (log scale): {val_reg_loss:.4f}, "
            f"Validation MAE (years): {val_mae:.2f}, "
            f"Validation MSE (years²): {mse:.4f}, "
            f"Validation RMSE (years): {rmse:.4f}, "
            f"Validation R² Score: {r2:.4f}"
        )


# 主函数（仅更新 train_model 调用部分）
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

    # 转化时间预测数据集
    time_dataset = MultimodalDiagnosisDataset(
        image_root_dir=r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/多种模态+SNP",
        protein_csv=r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/蛋白2.csv",
        snp_root_dir=r'/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/SNP-处理',
        apoe_csv=r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/ApoE （简体中文）基因分型 - 结果 [ADNI1，GO，2,3,4].csv",
        transform=transform,
        min_time_points=2,
        mode='time',
        snp_augment_dir=r'/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/筛选SNP',
        augment_prob=0.5,
    )

    train_indices, test_indices = split_dataset(time_dataset, train_ratio=0.8, random_seed=42)
    train_time_dataset = Subset(time_dataset, train_indices)
    test_time_dataset = Subset(
        MultimodalDiagnosisDataset(
            image_root_dir=r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/多种模态+SNP",
            protein_csv=r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/蛋白2.csv",
            snp_root_dir=r'/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/SNP-处理',
            apoe_csv=r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/ApoE （简体中文）基因分型 - 结果 [ADNI1，GO，2,3,4].csv",
            transform=test_transform,
            min_time_points=2,
            mode='time'
        ),
        test_indices
    )

    train_time_loader = DataLoader(train_time_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    test_time_loader = DataLoader(test_time_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

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

    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.load_state_dict(torch.load(r'多模态-time-转化时间-2.pth'))
    # 训练转化时间预测
    print("训练转化时间预测模型...")
    # train_model(model, train_time_loader, test_time_loader, criterion_reg, optimizer, num_epochs=30,
    #             device=device)
    eval_model(model, test_time_loader, train_time_loader, criterion_reg, optimizer, num_epochs=30,
                device=device)
    # 调试：检查数据集输出
    print("转化时间预测数据集样本：")
    for i in range(min(5, len(time_dataset))):
        sample = time_dataset[i]
        print(
            f"样本 {i}: 图像ID={time_dataset.image_ids[i]}, 标签={sample['label']}, 转化时间={sample['conversion_time']} 年")