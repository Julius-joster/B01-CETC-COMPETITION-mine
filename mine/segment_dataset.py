# segment_dataset.py
import csv
import re
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from traj_features import build_trajectory_features, _parse_timestamp


# ============ 速度层/高度层枚举（新数据可能没有这两列，保留以兼容老数据） ============
SPEED_TYPES = [
    "加速", "速度保持", "减速", "速度周期波动",
    "阶梯型减速", "阶梯型加速",
]
HEIGHT_TYPES = [
    "爬升", "巡航保持", "下降", "高度周期波动",
    "阶梯型下降", "阶梯型上升",
]
SPEED2ID = {t: i for i, t in enumerate(SPEED_TYPES)}
HEIGHT2ID = {t: i for i, t in enumerate(HEIGHT_TYPES)}


# ============ 5 个回归目标的归一化常数 ============
DURATION_SCALE = 600.0      # 持续时间基准：10 分钟
START_SCALE    = 3600.0     # 起始时间基准：1 小时
DIST_SCALE     = 20000.0    # 飞行距离基准：20 km
HEIGHT_SCALE   = 5000.0     # 高度差距基准：5 km


# ============ 飞行方向字符串解析 ============
def parse_flight_direction(s):
    """
    '北偏东66.5度' → bearing 66.5（从北顺时针） → (66.5-180)/180 = -0.63
    返回归一化到 [-1, 1] 的标量。
    """
    s = str(s).strip()
    if not s:
        return 0.0
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if not m:
        return 0.0
    angle = float(m.group(1))

    if "北偏东" in s:
        bearing = angle
    elif "北偏西" in s:
        bearing = 360.0 - angle
    elif "南偏东" in s:
        bearing = 180.0 - angle
    elif "南偏西" in s:
        bearing = 180.0 + angle
    elif "正北" in s or s == "北":
        bearing = 0.0
    elif "正东" in s or s == "东":
        bearing = 90.0
    elif "正南" in s or s == "南":
        bearing = 180.0
    elif "正西" in s or "西" in s:
        bearing = 270.0
    else:
        bearing = angle  # 兜底：当作从北顺时针的度数

    return (bearing - 180.0) / 180.0


# ============ 编码器输入的 20 维全局特征（不变） ============
def _global_features(points):
    t = np.array([_parse_timestamp(p["时间戳"]) for p in points], dtype=np.float64)
    h = np.array([float(p["高度"]) for p in points], dtype=np.float64)
    v = np.array([float(p["速度"]) for p in points], dtype=np.float64)
    hdg = np.array([float(p["航向角"]) for p in points], dtype=np.float64)
    lon = np.array([float(p["经度"]) for p in points], dtype=np.float64)
    lat = np.array([float(p["纬度"]) for p in points], dtype=np.float64)

    dt = np.diff(t)
    dt = np.where(dt <= 0, 1.0, dt)
    dh_dt = np.diff(h) / dt
    dv_dt = np.diff(v) / dt
    d_hdg = (np.diff(hdg) + 180) % 360 - 180
    turn_rate = d_hdg / dt
    total_turn = np.sum(np.abs(d_hdg))

    return np.array([
        len(points) / 100.0,
        max(t[-1] - t[0], 0) / 1000.0,
        h.mean() / 10000.0,
        h.std() / 1000.0,
        (h.max() - h.min()) / 1000.0,
        v.mean() / 250.0,
        v.std() / 50.0,
        (v.max() - v.min()) / 50.0,
        total_turn / 360.0,
        (lon.max() - lon.min()) * 100,
        (lat.max() - lat.min()) * 100,
        np.mean(dh_dt) / 10.0,
        np.std(dh_dt) / 10.0,
        np.mean(dv_dt) / 2.0,
        np.std(dv_dt) / 2.0,
        np.mean(np.abs(turn_rate)) / 5.0,
        np.std(turn_rate) / 5.0,
        (h[-1] - h[0]) / 1000.0,
        (v[-1] - v[0]) / 50.0,
        hdg.std() / 180.0,
    ], dtype=np.float32)


# ============ 从 CSV 行直接读取 5 个回归目标 ============
def _regress_targets_from_row(row):
    """
    直接读 CSV 的 5 列：持续时间、起始时间、飞行方向、飞行距离、高度差距
    返回归一化到与全局特征相近量级的 5 维数组。
    """
    try:
        duration = float(row.get("持续时间", 0.0)) / DURATION_SCALE
    except (ValueError, TypeError):
        duration = 0.0
    try:
        start_time = float(row.get("起始时间", 0.0)) / START_SCALE
    except (ValueError, TypeError):
        start_time = 0.0
    direction = parse_flight_direction(row.get("飞行方向", ""))
    try:
        dist = float(row.get("飞行距离", 0.0)) / DIST_SCALE
    except (ValueError, TypeError):
        dist = 0.0
    try:
        dh = float(row.get("高度差距", 0.0)) / HEIGHT_SCALE
    except (ValueError, TypeError):
        dh = 0.0

    return np.array([duration, start_time, dist, direction, dh], dtype=np.float32)


# ============ 数据增强 ============
def _augment_points(points, rng):
    pts = [dict(p) for p in points]
    h_scale = rng.uniform(0.95, 1.05)
    v_scale = rng.uniform(0.95, 1.05)
    for p in pts:
        p["高度"] = float(p["高度"]) * h_scale + rng.normal(0, 2.0)
        p["速度"] = float(p["速度"]) * v_scale + rng.normal(0, 0.3)
        p["航向角"] = (float(p["航向角"]) + rng.normal(0, 1.0)) % 360
    return pts


class SegmentCSVDataset(Dataset):
    """
    从宽表 CSV 读取，每个 (航迹编号, 阶段序号) 组成一个片段。
    主分类目标：行为标签（自动发现所有唯一值）。
    速度层/高度层：如果 CSV 有这两列就用，没有就设为 -1（ignore_index）。
    5 个回归目标：直接从 CSV 列读取（持续时间、起始时间、飞行方向、飞行距离、高度差距）。
    """
    def __init__(self, csv_path, seg_len=32, min_points=2,
                 augment=False, seed=42, behavior_types=None):
        self.seg_len = seg_len
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = []

        groups = defaultdict(list)
        labels = {}
        regress_map = {}

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    traj_id = row["航迹编号"].strip()
                    stage = int(row["阶段序号"])
                except (KeyError, ValueError, TypeError):
                    continue

                t_val = _parse_timestamp(row["时间戳"])
                point = {
                    "时间戳": t_val,
                    "经度":   float(row["经度"]),
                    "纬度":   float(row["纬度"]),
                    "高度":   float(row["高度"]),
                    "速度":   float(row["速度"]),
                    "航向角": float(row["航向角"]),
                }
                key = (traj_id, stage)
                groups[key].append(point)

                if key not in labels:
                    labels[key] = {
                        "behavior": row.get("行为标签", "").strip(),
                        "speed":    row.get("速度层", "").strip(),
                        "height":   row.get("高度层", "").strip(),
                    }
                    regress_map[key] = _regress_targets_from_row(row)

        # 自动发现所有行为标签
        if behavior_types is None:
            all_b = set(lab["behavior"] for lab in labels.values() if lab["behavior"])
            behavior_types = sorted(all_b)
        self.behavior_types = behavior_types
        self.behavior2id = {t: i for i, t in enumerate(behavior_types)}

        n_skip_label = n_skip_len = 0
        for key, pts in groups.items():
            lab = labels[key]
            if lab["behavior"] not in self.behavior2id:
                n_skip_label += 1
                continue
            if len(pts) < min_points:
                n_skip_len += 1
                continue

            self.samples.append({
                "traj_id":      key[0],
                "stage":        key[1],
                "points":       pts,
                "behavior_id":  self.behavior2id[lab["behavior"]],
                "speed_id":     SPEED2ID.get(lab["speed"], -1),
                "height_id":    HEIGHT2ID.get(lab["height"], -1),
                "targets":      regress_map[key],
            })

        print(f"[SegmentCSVDataset] loaded {len(self.samples)} segments "
              f"(augment={augment})")
        print(f"  跳过 {n_skip_label} 个（行为标签无效）")
        print(f"  跳过 {n_skip_len} 个（点数 < {min_points}）")
        print(f"  行为标签类别数：{len(self.behavior_types)} → {self.behavior_types}")
        has_speed  = any(lab["speed"] for lab in labels.values())
        has_height = any(lab["height"] for lab in labels.values())
        print(f"  速度层存在：{has_speed}；高度层存在：{has_height}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.augment:
            pts = _augment_points(s["points"], self.rng)
        else:
            pts = s["points"]

        feats = build_trajectory_features(pts, self.seg_len)
        g = _global_features(pts)
        return {
            "seg_feats":    feats,
            "global_feats": torch.from_numpy(g),
            "regress_targets": torch.from_numpy(s["targets"]),      # (5,)
            "behavior_id":  torch.tensor(s["behavior_id"], dtype=torch.long),
            "speed_id":     torch.tensor(s["speed_id"],    dtype=torch.long),
            "height_id":    torch.tensor(s["height_id"],   dtype=torch.long),
        }