# traj_encoder.py
import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    def __init__(self, patch_len=8, stride=4, d_feat=11, d_model=1024):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * d_feat, d_model)

    def forward(self, x):
        # x: (B, T, d_feat)
        B, T, D = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)   # (B, N, D, patch_len)
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, self.patch_len * D)
        return self.proj(patches)                            # (B, N, d_model)


class CrossAttentionReprogrammer(nn.Module):
    """Time-LLM / LLM4AirTrack 风格：航迹 patch 为 Query，文本原型为 Key/Value。"""
    def __init__(self, d_model=1024, n_heads=8, n_prototypes=256, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.02)
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, patch_embeds):
        B = patch_embeds.size(0)
        proto = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        x = patch_embeds
        for attn, norm in zip(self.attns, self.norms):
            out, _ = attn(x, proto, proto)
            x = norm(x + out)
        return x


class TrajectoryEncoder(nn.Module):
    def __init__(self, d_feat=11, d_model=1024,
                 patch_len=8, stride=4, n_heads=8, n_prototypes=256):
        super().__init__()
        self.patch_tok = PatchTokenizer(patch_len, stride, d_feat, d_model)
        self.reprogrammer = CrossAttentionReprogrammer(
            d_model=d_model, n_heads=n_heads, n_prototypes=n_prototypes
        )
        self.output_dim = d_model

    def forward(self, traj_feats):
        # traj_feats: (B, T, d_feat)
        patches = self.patch_tok(traj_feats)        # (B, N, d_model)
        return self.reprogrammer(patches)           # (B, N, d_model)

class PhysicsHead(nn.Module):
    """从航迹隐藏表示预测位置、速度、加速度（各3维）。"""
    def __init__(self, d_model, d_state=3):
        super().__init__()
        self.pos_head = nn.Linear(d_model, d_state)
        self.vel_head = nn.Linear(d_model, d_state)
        self.acc_head = nn.Linear(d_model, d_state)
        nn.init.normal_(self.pos_head.weight, std=0.02)
        nn.init.normal_(self.vel_head.weight, std=0.02)
        nn.init.normal_(self.acc_head.weight, std=0.02)

    def forward(self, hidden):
        # hidden: (B, N, D)
        return {
            "pos": self.pos_head(hidden),   # (B, N, 3)
            "vel": self.vel_head(hidden),
            "acc": self.acc_head(hidden),
        }


def physical_consistency_loss(pred, dt=1.0):
    """
    pred: dict，含 pos / vel / acc，各 (B, N, 3)
    约束：vel ≈ Δpos/Δt，acc ≈ Δvel/Δt（中心差分近似）
    """
    pos, vel, acc = pred["pos"], pred["vel"], pred["acc"]

    # 中心差分：p[i+1] - p[i-1] ≈ 2*Δt * v[i]
    d_pos = (pos[:, 2:, :] - pos[:, :-2, :]) / (2 * dt)   # (B, N-2, 3)
    v_mid = vel[:, 1:-1, :]

    d_vel = (vel[:, 2:, :] - vel[:, :-2, :]) / (2 * dt)
    a_mid = acc[:, 1:-1, :]

    loss_pos = F.mse_loss(d_pos, v_mid)
    loss_vel = F.mse_loss(d_vel, a_mid)
    return loss_pos + loss_vel


def motion_smoothness_loss(pred):
    """
    惩罚加速度变化（jerk）和曲率变化，鼓励轨迹平滑。
    """
    acc = pred["acc"]                       # (B, N, 3)
    jerk = acc[:, 1:, :] - acc[:, :-1, :]   # (B, N-1, 3)
    loss_jerk = torch.mean(jerk ** 2)

    pos = pred["pos"]                       # (B, N, 3)
    # 粗略曲率：相邻位移向量的夹角变化（水平面内）
    d1 = pos[:, 1:, :2] - pos[:, :-1, :2]
    d2 = pos[:, 2:, :2] - pos[:, 1:-1, :2]
    cos_angle = F.cosine_similarity(d1, d2, dim=-1)
    curvature_change = torch.mean((cos_angle - 1.0) ** 2)
    return loss_jerk + curvature_change