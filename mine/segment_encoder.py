# segment_encoder.py
import torch
import torch.nn as nn
from traj_encoder import PatchTokenizer, CrossAttentionReprogrammer


class SegmentEncoder(nn.Module):
    def __init__(self, d_feat=11, d_model=256, n_global=20,
                 patch_len=2, stride=1, n_heads=8, n_prototypes=256,
                 dropout=0.1):
        super().__init__()
        self.patch_tok = PatchTokenizer(patch_len, stride, d_feat, d_model)
        self.reprogrammer = CrossAttentionReprogrammer(
            d_model=d_model, n_heads=n_heads, n_prototypes=n_prototypes
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.global_proj = nn.Linear(n_global, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, seg_feats, global_feats):
        patches = self.patch_tok(seg_feats)
        h = self.reprogrammer(patches)
        q = self.pool_query.expand(h.size(0), -1, -1)
        pooled, _ = self.pool_attn(q, h, h)
        pooled = pooled.squeeze(1)
        g = self.global_proj(global_feats)
        z = self.dropout(pooled + g)
        return self.norm(z)


class MultiTaskSegmentClassifier(nn.Module):
    """片段编码器 + 三个分类头 + 一个回归头（5 个物理量）。"""
    def __init__(self, encoder, n_comprehensive, n_speed, n_height, n_regress=5):
        super().__init__()
        self.encoder = encoder
        d = encoder.output_dim
        self.head_comp    = nn.Linear(d, n_comprehensive)
        self.head_speed   = nn.Linear(d, n_speed)
        self.head_height  = nn.Linear(d, n_height)
        self.head_regress = nn.Linear(d, n_regress)

    def forward(self, seg_feats, global_feats):
        z = self.encoder(seg_feats, global_feats)
        return {
            "comprehensive": self.head_comp(z),
            "speed":         self.head_speed(z),
            "height":        self.head_height(z),
            "regress":       self.head_regress(z),   # (B, 5)
            "embedding":     z,
        }