# traj_features.py
import numpy as np
import torch
from datetime import datetime


def _parse_timestamp(v):
    """兼容数字时间戳和 ISO 8601 字符串时间戳（如 '2022-03-15T03:57:56Z'）。"""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    # 先尝试当纯数字字符串处理
    try:
        return float(s)
    except ValueError:
        pass
    # 再尝试 ISO 8601
    s2 = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s2).timestamp()


def build_trajectory_features(points, max_len=128):
    """
    points: list[dict]，每个 dict 含 时间戳/经度/纬度/高度/速度/航向角
            时间戳可以是数字（秒）或 ISO 8601 字符串
    返回: (max_len, 11) float32 tensor
    """
    n = len(points)

    # 1) 时间戳：统一解析为浮点，再减去首点，得到“相对首点的秒数”
    ts_raw = np.array([_parse_timestamp(p["时间戳"]) for p in points], dtype=np.float64)
    ts = ts_raw - ts_raw[0]

    # 2) 其余字段：用 float() 包裹，兼容字符串数字
    lon = np.array([float(p["经度"]) for p in points], dtype=np.float64)
    lat = np.array([float(p["纬度"]) for p in points], dtype=np.float64)
    alt = np.array([float(p["高度"]) for p in points], dtype=np.float64)
    spd = np.array([float(p["速度"]) for p in points], dtype=np.float64)
    hdg = np.deg2rad(np.array([float(p["航向角"]) for p in points], dtype=np.float64))

    # 3) 局部 ENU（以首点为原点）
    lat0 = np.deg2rad(lat[0])
    x = (lon - lon[0]) * 111320.0 * np.cos(lat0)
    y = (lat - lat[0]) * 110540.0
    z = alt - alt[0]

    # 4) 相邻点时间差
    dt = np.diff(ts, prepend=ts[0])

    # 5) 极坐标 & 方向
    ux, uy = np.cos(hdg), np.sin(hdg)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    # 6) 归一化时间
    t_span = max(ts[-1], 1e-6)          # ts 已经减过首点
    t_norm = ts / t_span

    # 7) 拼接 11 维特征
    feats = np.stack([
        x / 1000.0,                     # km
        y / 1000.0,
        z / 1000.0,
        ux, uy,
        r / 1000.0,
        np.sin(theta), np.cos(theta),
        t_norm,
        spd / 250.0,
        np.log1p(np.abs(dt)) / 10.0,    # 用 abs 防止时间回退导致 NaN
    ], axis=-1).astype(np.float32)      # (n, 11)

    # 8) 重采样到 max_len
    if n > max_len:
        idx = np.linspace(0, n - 1, max_len).round().astype(int)
        feats = feats[idx]
    elif n < max_len:
        pad = np.zeros((max_len - n, feats.shape[1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)

    return torch.from_numpy(feats)