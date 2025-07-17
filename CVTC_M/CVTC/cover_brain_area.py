import torch
from torch import nn, einsum
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
import random
import json
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)

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
        return ((x - mean) / (var + self.eps)) * self.g + self.b

# 前馈网络
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
        assert dim >= dim_head, 'dim必须大于等于dim_head'
        if heads == 0:
            raise ValueError('heads不能为零，请确保dim >= dim_head')
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

# Transformer
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

# 基因特征提取器
class GeneFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super(GeneFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(embed_dim)
            ]) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, 128)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        x = self.input_proj(x)
        pos_embed = self.pos_embed.expand(b, -1)
        x = x + pos_embed
        x_attn = x.unsqueeze(1)
        for attn, norm1, ff, norm2 in self.transformer:
            attn_output, _ = attn(x_attn, x_attn, x_attn)
            x_attn = norm1(x_attn + attn_output)
            ff_output = ff(x_attn)
            x_attn = norm2(x_attn + ff_output)
        x_attn = x_attn.squeeze(1)
        gene_features = self.output_proj(x_attn)
        return gene_features

# 跨模态注意力融合模块（仅图像和基因）
class MultimodalFusion(nn.Module):
    def __init__(self, image_dim, gene_dim, fusion_dim=512, num_heads=8, dropout=0.1):
        super(MultimodalFusion, self).__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.scale = (fusion_dim // num_heads) ** -0.5
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.gene_proj = nn.Linear(gene_dim, fusion_dim)
        self.image_weight = torch.tensor(5.0, device=device)
        self.gene_weight = 0.01
        self.to_q = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.to_kv = nn.Linear(fusion_dim, fusion_dim * 2, bias=False)
        self.to_out = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(fusion_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            nn.init.constant_(m, 1.0)

    def forward(self, image_features, gene_features):
        image_features = F.normalize(image_features, dim=-1)
        gene_features = F.normalize(gene_features, dim=-1)
        image_f = self.image_proj(image_features) * self.image_weight
        gene_f = self.gene_proj(gene_features) * self.gene_weight
        features = image_f.unsqueeze(1)
        B, N, D = features.shape
        q = self.to_q(image_f).view(B, self.num_heads, 1, D // self.num_heads).permute(0, 1, 2, 3)
        kv = self.to_kv(features).view(B, 1, 2, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, 1, D)
        out = self.to_out(out.squeeze(1))
        out = out + 0.01 * gene_f
        return out

# 多模态 CrossFormer
class MultimodalCrossFormer(nn.Module):
    def __init__(
            self,
            dim=(64, 128, 256, 512),
            depth=(2, 2, 8, 2),
            global_window_size=(8, 4, 2, 1),
            local_window_size=16,
            cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
            cross_embed_strides=(4, 2, 2, 2),
            num_classes=10,
            attn_dropout=0.,
            ff_dropout=0.,
            channels=3,
            gene_input_dim=40000,
            feature_dim=20
    ):
        super(MultimodalCrossFormer, self).__init__()
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
        self.image_layers = nn.ModuleList([])
        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(
                dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes,
                cross_embed_strides):
            self.image_layers.append(nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]))
        self.gene_extractor = GeneFeatureExtractor(input_dim=gene_input_dim, embed_dim=512, num_heads=8, num_layers=2)
        self.fusion = MultimodalFusion(image_dim=last_dim, gene_dim=128, fusion_dim=512)
        self.to_logits = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        self.feature_reducer = nn.Linear(512, feature_dim)

    def forward(self, image, gene_data):
        x = image
        for cel, transformer in self.image_layers:
            x = cel(x)
            x = transformer(x)
        image_features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        gene_features = self.gene_extractor(gene_data)
        fused_features = self.fusion(image_features, gene_features)
        logits = self.to_logits(fused_features)
        return logits

    def extract_features(self, image, gene_data):
        x = image
        for cel, transformer in self.image_layers:
            x = cel(x)
            x = transformer(x)
        image_features = torch.einsum('b c h w -> b c', x) / (x.shape[2] * x.shape[3])
        gene_features = self.gene_extractor(gene_data)
        fused_features = self.fusion(image_features, gene_features)
        features = self.feature_reducer(fused_features)
        return features

# MRI图像遮盖函数
def process_mri_image(image, selected_region=None):
    transform_to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform_to_tensor(image)
    img_np = img_tensor.permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    img_rotated = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    gray_img = cv2.cvtColor(img_rotated, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    if not valid_contours:
        mask = np.ones((img_rotated.shape[0], img_rotated.shape[1]), dtype=np.uint8) * 255
        x_min, y_min, w_mask, h_mask = 0, 0, img_rotated.shape[1], img_rotated.shape[0]
    else:
        max_contour = max(valid_contours, key=cv2.contourArea)
        mask = np.zeros((img_rotated.shape[0], img_rotated.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        x_min, y_min, w_mask, h_mask = cv2.boundingRect(max_contour)
    h, w = img_rotated.shape[:2]
    regions = ["top", "bottom", "left", "right", "center"]
    selected_region = random.choice(regions) if selected_region is None else selected_region
    cover_mask = np.zeros((h, w), dtype=np.uint8)
    if selected_region == "top":
        x_start, x_end = x_min, x_min + w_mask
        y_start, y_end = y_min, y_min + int(h_mask * 0.4)
    elif selected_region == "bottom":
        x_start, x_end = x_min, x_min + w_mask
        y_start, y_end = y_min + int(h_mask * 0.6), y_min + h_mask
    elif selected_region == "left":
        x_start, x_end = x_min, x_min + int(w_mask * 0.4)
        y_start, y_end = y_min, y_min + h_mask
    elif selected_region == "right":
        x_start, x_end = x_min + int(w_mask * 0.6), x_min + w_mask
        y_start, y_end = y_min, y_min + h_mask
    else:
        x_start, x_end = x_min + int(w_mask * 0.3), x_min + int(w_mask * 0.7)
        y_start, y_end = y_min + int(h_mask * 0.3), y_min + int(h_mask * 0.7)
    x_start = max(0, x_start)
    x_end = min(w, x_end)
    y_start = max(0, y_start)
    y_end = min(h, y_end)
    cv2.rectangle(cover_mask, (x_start, y_start), (x_end, y_end), 255, thickness=cv2.FILLED)
    cover_mask = cv2.bitwise_and(cover_mask, mask)
    img_tensor_rotated = torch.tensor(img_rotated.astype(np.float32) / 255.0).permute(2, 0, 1)
    img_tensor_np = img_tensor_rotated.permute(1, 2, 0).numpy()
    img_tensor_np[cover_mask == 255] = 0
    img_tensor = torch.tensor(img_tensor_np).permute(2, 0, 1)
    transform_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img_tensor = transform_normalize(img_tensor)
    return img_tensor, selected_region

class CustomNeuroDataset(Dataset):
    def __init__(self, image_root_dir, gene_csv_path, transform=None, apply_mask=True, fixed_region=None, fixed_gene_indices=None):
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.apply_mask = apply_mask
        self.fixed_region = fixed_region
        self.fixed_gene_indices = fixed_gene_indices
        self.regions = ["top", "bottom", "left", "right", "center"]
        self.gene_df = pd.read_csv(gene_csv_path, skiprows=3)
        self.gene_data = self.gene_df.iloc[:, [0] + list(range(2, self.gene_df.shape[1]))]
        gene_names_df = pd.read_csv(gene_csv_path, nrows=3, header=None)
        # 原始基因名称（未去重）
        self.raw_gene_names = gene_names_df.iloc[2, 1:].values
        # 去重后的基因名称
        self.gene_names = [str(name) for name in set(self.raw_gene_names) if pd.notna(name) and str(name).strip().lower() != 'symbol']
        self.valid_gene_indices = [i for i, name in enumerate(self.gene_names) if pd.notna(name)]
        # 构建基因名称到原始列索引的映射
        self.gene_name_to_indices = {}
        for idx, name in enumerate(self.raw_gene_names):
            if pd.notna(name) and str(name).strip().lower() != 'symbol':
                name = str(name)
                if name in self.gene_name_to_indices:
                    self.gene_name_to_indices[name].append(idx)
                else:
                    self.gene_name_to_indices[name] = [idx]
        print(f"Sample gene names: {self.gene_names[:5]}")
        print(f"Number of valid gene indices: {len(self.valid_gene_indices)}")
        self.gene_data.columns = ['subject_id'] + [f'gene_{i}' for i in range(self.gene_data.shape[1] - 1)]
        scaler = StandardScaler()
        gene_values = self.gene_data.iloc[:, 1:].values
        gene_values = scaler.fit_transform(gene_values)
        self.gene_data.iloc[:, 1:] = gene_values
        self.min_gene_value = gene_values.min()
        gene_subject_ids = set(self.gene_data['subject_id'].dropna())
        self.image_paths = []
        self.labels = []
        self.subject_ids = []
        image_subject_ids = set()
        unmatched_images = {'no_gene': set(), 'invalid_format': set()}
        self.categories = sorted(os.listdir(image_root_dir))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        for category in self.categories:
            category_path = os.path.join(image_root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for img_name in os.listdir(category_path):
                if not img_name.endswith('.png'):
                    continue
                parts = img_name.split('_')
                if len(parts) < 3:
                    unmatched_images['invalid_format'].add(img_name)
                    continue
                subject_id = f"{parts[0]}_S_{parts[2]}"
                if len(subject_id.split('_')) != 3:
                    unmatched_images['invalid_format'].add(img_name)
                    continue
                image_subject_ids.add(subject_id)
                if subject_id not in gene_subject_ids:
                    unmatched_images['no_gene'].add(subject_id)
                    continue
                img_path = os.path.join(category_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.category_to_idx[category])
                self.subject_ids.append(subject_id)
        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Image subject IDs count: {len(image_subject_ids)}")
        print(f"Sample image subject IDs: {list(image_subject_ids)[:5]}")
        print(f"Matching subject IDs with gene data: {len(image_subject_ids.intersection(gene_subject_ids))}")
        print(f"Unmatched subject IDs - No gene data: {list(unmatched_images['no_gene'])[:5]}")
        print(f"Unmatched images - Invalid format: {list(unmatched_images['invalid_format'])[:5]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subject_id = self.subject_ids[idx]
        gene_row = self.gene_data[self.gene_data['subject_id'] == subject_id]
        if gene_row.empty:
            print(f"No gene data found for subject_id: {subject_id}")
            return None
        gene_data = gene_row.iloc[:, 1:].values[0].copy()
        region_name = ""
        gene_names = []
        if self.apply_mask:
            region_name = self.fixed_region if self.fixed_region else random.choice(self.regions)
            image, region_name = process_mri_image(image, region_name)
            # 选择去重后的基因索引
            gene_indices = self.fixed_gene_indices if self.fixed_gene_indices else random.sample(self.valid_gene_indices, min(2000, len(self.valid_gene_indices)))
            # 获取对应的基因名称
            gene_names = [str(self.gene_names[idx]) for idx in gene_indices]
            # 遮盖所有重复的基因列
            for gene_name in gene_names:
                if gene_name in self.gene_name_to_indices:
                    for raw_idx in self.gene_name_to_indices[gene_name]:
                        gene_data[raw_idx] = self.min_gene_value
        elif self.transform:
            image = self.transform(image)
            gene_names = []
        gene_data = torch.tensor(gene_data, dtype=torch.float32)
        return {
            'image': image,
            'label': label,
            'gene_data': gene_data,
            'subject_id': subject_id,
            'region_name': region_name,
            'gene_name': gene_names
        }

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b['image'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    gene_data = torch.stack([b['gene_data'] for b in batch])
    subject_ids = [b['subject_id'] for b in batch]
    region_names = [b['region_name'] for b in batch]
    gene_names = [b['gene_name'] for b in batch]
    return {
        'image': images,
        'label': labels,
        'gene_data': gene_data,
        'subject_id': subject_ids,
        'region_name': region_names,
        'gene_name': gene_names
    }

def evaluate_model(model, data_loader, criterion, l1_lambda):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    results = []
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating')
        for batch in progress_bar:
            if batch is None:
                continue
            images = batch['image'].to(device)
            gene_data = batch['gene_data'].to(device)
            labels = batch['label'].to(device)
            region_names = batch['region_name']
            gene_names = batch['gene_name']
            logits = model(images, gene_data)
            loss = criterion(logits, labels)
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.norm(param, p=1)
            loss += l1_lambda * l1_loss
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                region_name = region_names[i] if isinstance(region_names[i], str) else str(region_names[i])
                gene_name = gene_names[i] if isinstance(gene_names[i], list) else list(gene_names[i])
                if region_name and gene_name:
                    results.append({
                        'label': labels[i].item(),
                        'region': region_name,
                        'genes': gene_name,
                        'correct': predicted[i].item() == labels[i].item()
                    })
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0
    return accuracy, avg_loss, results

def evaluate_with_masking(model, val_loader, criterion, l1_lambda, baseline_acc_by_class, categories, num_runs=10):
    all_combination_impacts = {cat: [] for cat in categories}
    regions = ["top", "bottom", "left", "right", "center"]
    dataset = val_loader.dataset
    valid_gene_indices = dataset.valid_gene_indices
    num_genes_to_mask = min(2000, len(valid_gene_indices))
    combinations = []
    for region in regions:
        gene_combinations = set()
        while len(gene_combinations) < num_runs:
            gene_indices = tuple(sorted(random.sample(valid_gene_indices, num_genes_to_mask)))
            gene_combinations.add(gene_indices)
        for gene_indices in gene_combinations:
            combinations.append((region, gene_indices))
    random.shuffle(combinations)
    combinations = combinations[:num_runs]
    for run, (region, gene_indices) in enumerate(combinations, 1):
        print(f"\nRun {run}/{len(combinations)}: Region={region}, Genes={len(gene_indices)}")
        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)
        dataset.fixed_region = region
        dataset.fixed_gene_indices = list(gene_indices)
        combination_impacts = {cat: [] for cat in categories}
        val_acc, val_loss, results = evaluate_model(model, val_loader, criterion, l1_lambda)
        print(f'Run {run} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
        for result in results:
            label_name = categories[result['label']]
            acc_drop = baseline_acc_by_class.get(result['label'], 100) - (100 if result['correct'] else 0)
            combination_impacts[label_name].append({
                'region': result['region'],
                'genes': result['genes'],
                'acc_drop': acc_drop
            })
        for category in combination_impacts:
            all_combination_impacts[category].extend(combination_impacts[category])
    top_combinations = {}
    for category in all_combination_impacts:
        impact_dict = {}
        for item in all_combination_impacts[category]:
            key = (item['region'], tuple(item['genes']))
            if key not in impact_dict:
                impact_dict[key] = []
            impact_dict[key].append(item['acc_drop'])
        avg_impacts = [(key, sum(impacts) / len(impacts)) for key, impacts in impact_dict.items() if len(impacts) >= 1]
        avg_impacts.sort(key=lambda x: x[1], reverse=True)
        top_combinations[category] = avg_impacts[:500]
        category_idx = categories.index(category)
        baseline_acc = baseline_acc_by_class.get(category_idx, 100)
        df = pd.DataFrame([
            {
                'category': category,
                'acc_drop_percentage': (acc_drop / baseline_acc * 100) if baseline_acc != 0 else 0,
                'region': region,
                'genes': ', '.join(genes),
                'avg_acc_drop': acc_drop
            }
            for (region, genes), acc_drop in top_combinations[category]
        ])
        if not df.empty:
            df = df[['category', 'acc_drop_percentage', 'region', 'genes', 'avg_acc_drop']]
            df.to_csv(f'top_combinations_{category}_multi_run.csv', index=False)
            print(f"Saved top combinations for {category} to top_combinations_{category}_multi_run.csv")
        else:
            print(f"No combinations saved for {category} - no valid data found.")
    all_dfs = []
    for category in top_combinations:
        category_idx = categories.index(category)
        baseline_acc = baseline_acc_by_class.get(category_idx, 100)
        df = pd.DataFrame([
            {
                'category': category,
                'acc_drop_percentage': (acc_drop / baseline_acc * 100) if baseline_acc != 0 else 0,
                'region': region,
                'genes': ', '.join(genes),
                'avg_acc_drop': acc_drop
            }
            for (region, genes), acc_drop in top_combinations[category]
        ])
        if not df.empty:
            df = df[['category', 'acc_drop_percentage', 'region', 'genes', 'avg_acc_drop']]
            all_dfs.append(df)
        else:
            print(f"Warning: No data for category {category} - skipping.")
    if not all_dfs:
        print("Error: No valid data frames to concatenate. Saving empty final CSV.")
        pd.DataFrame(columns=['category', 'acc_drop_percentage', 'region', 'genes', 'avg_acc_drop']).to_csv('final_top_combinations_multi_run.csv', index=False)
        return top_combinations
    final_df = pd.concat(all_dfs, ignore_index=True)
    if final_df.empty:
        print("Error: Concatenated DataFrame is empty. Saving empty final CSV.")
        pd.DataFrame(columns=['category', 'acc_drop_percentage', 'region', 'genes', 'avg_acc_drop']).to_csv('final_top_combinations_multi_run.csv', index=False)
        return top_combinations
    if 'category' in final_df.columns and 'avg_acc_drop' in final_df.columns:
        final_df = final_df.sort_values(by=['category', 'avg_acc_drop'], ascending=[True, False])
        final_df = final_df[['category', 'acc_drop_percentage', 'region', 'genes', 'avg_acc_drop']]
        final_df.to_csv('final_top_combinations_multi_run.csv', index=False)
        print("Saved final top combinations to final_top_combinations_multi_run.csv")
    else:
        print("Error: Required columns missing in final_df. Saving as is.")
        final_df.to_csv('final_top_combinations_multi_run.csv', index=False)
    return top_combinations

def visualize_masked_image(image_tensor, subject_id, region_name, save_dir, max_images=100):
    global saved_image_count
    if not hasattr(visualize_masked_image, 'saved_image_count'):
        visualize_masked_image.saved_image_count = 0
    if visualize_masked_image.saved_image_count < max_images:
        os.makedirs(save_dir, exist_ok=True)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 0.5 + 0.5).clip(0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        save_path = os.path.join(save_dir, f"{subject_id}_masked_{region_name}.png")
        image_pil.save(save_path)
        visualize_masked_image.saved_image_count += 1
        print(f"Saved masked image to {save_path}")

if __name__ == "__main__":
    image_root_dir = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/图像 (2)/图像"
    gene_csv_path = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/dataset/ADNI_Gene_Expression_Profile_wash.csv"
    baseline_acc_file = "baseline_acc_by_class.json"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    baseline_dataset = CustomNeuroDataset(
        image_root_dir=image_root_dir,
        gene_csv_path=gene_csv_path,
        transform=transform,
        apply_mask=False
    )
    baseline_loader = DataLoader(baseline_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    model = torch.load('gene.pth', map_location=device, weights_only=False)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    l1_lambda = 1e-6
    baseline_acc_by_class = {}
    if os.path.exists(baseline_acc_file):
        print(f"Loading baseline accuracies from {baseline_acc_file}...")
        with open(baseline_acc_file, 'r') as f:
            baseline_acc_by_class = json.load(f)
            baseline_acc_by_class = {int(k): v for k, v in baseline_acc_by_class.items()}
    else:
        print("Evaluating baseline model...")
        baseline_acc, _, _ = evaluate_model(model, baseline_loader, criterion, l1_lambda)
        class_correct = {i: 0 for i in range(len(baseline_dataset.categories))}
        class_total = {i: 0 for i in range(len(baseline_dataset.categories))}
        with torch.no_grad():
            progress_bar = tqdm(baseline_loader, desc='Baseline Evaluation')
            for batch in progress_bar:
                if batch is None:
                    continue
                images = batch['image'].to(device)
                gene_data = batch['gene_data'].to(device)
                labels = batch['label'].to(device)
                logits = model(images, gene_data)
                _, predicted = torch.max(logits, 1)
                for label, pred in zip(labels, predicted):
                    class_total[label.item()] += 1
                    if pred.item() == label.item():
                        class_correct[label.item()] += 1
                progress_bar.set_postfix({
                    'acc': f'{100 * sum(class_correct.values()) / sum(class_total.values()):.2f}%' if sum(
                        class_total.values()) > 0 else 'N/A'})
        for i in class_correct:
            baseline_acc_by_class[i] = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print("Baseline accuracies by class:",
              {baseline_dataset.categories[i]: acc for i, acc in baseline_acc_by_class.items()})
        with open(baseline_acc_file, 'w') as f:
            json.dump(baseline_acc_by_class, f, indent=4)
        print(f"Baseline accuracies saved to {baseline_acc_file}")
    dataset = CustomNeuroDataset(
        image_root_dir=image_root_dir,
        gene_csv_path=gene_csv_path,
        transform=transform,
        apply_mask=True
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    vis_save_dir = r"/media/lenovo/A06B2FA1620B6FCB/CT生成HE/CVTC-基因/masked_images"
    vis_max_images = 10
    print("Visualizing masked images from DataLoader...")
    saved_image_count = 0
    for batch in loader:
        if batch is None:
            continue
        images = batch['image']
        subject_ids = batch['subject_id']
        region_names = batch['region_name']
        for i, (image, subject_id, region_name) in enumerate(zip(images, subject_ids, region_names)):
            if region_name and saved_image_count < vis_max_images:
                visualize_masked_image(image, subject_id, region_name, vis_save_dir, vis_max_images)
                saved_image_count += 1
        if saved_image_count >= vis_max_images:
            break
    print(f"Visualized {saved_image_count} masked images, saved to {vis_save_dir}")
    num_runs = 100
    top_combinations = evaluate_with_masking(model, loader, criterion, l1_lambda, baseline_acc_by_class,
                                            dataset.categories, num_runs=num_runs)
    print("Evaluation completed!")
    print("Top combinations saved to CSV files for each category and final summary.")