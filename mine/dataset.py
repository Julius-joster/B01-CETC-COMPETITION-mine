# dataset.py
import csv
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from traj_features import build_trajectory_features, _parse_timestamp
from segment_dataset import _global_features


SYSTEM_PROMPT = (
    "你是一个专业的航迹分析助手。请根据提供的ADS-B航迹判断飞行阶段，"
    "并回答选择题。回答时先在<think></think>中写出推理过程，再给出答案。"
)


# ============ 时间戳类型判断 ============
def _is_absolute_time(t):
    if isinstance(t, (int, float)):
        return False
    s = str(t).strip()
    if not s:
        return False
    try:
        float(s)
        return False
    except ValueError:
        return True


# ============ 选项解析（支持 A/B/C/... 任意数量） ============
def _parse_options(instruction_text):
    options = {}
    pattern = re.compile(r'^([A-Z])[.、:：\s]\s*(.+)$', re.MULTILINE)
    for m in pattern.finditer(instruction_text):
        letter = m.group(1)
        content = m.group(2).strip()
        if letter not in options:
            options[letter] = content
    return options


# ============ 从答案文本解析阶段序列 ============
def _parse_stages_from_answer(answer_text):
    """'A. 高度周期波动 -> 爬升' → (['高度周期波动', '爬升'], 'A')"""
    if not answer_text:
        return [], ""
    m = re.match(r'^([A-Z])[.、:：\s]\s*(.+)$', answer_text.strip())
    if m:
        letter = m.group(1)
        content = m.group(2).strip()
    else:
        letter = ""
        content = answer_text.strip()
    parts = re.split(r'\s*(?:->|→|—>|＞)\s*', content)
    parts = [p.strip() for p in parts if p.strip()]
    return parts, letter


# ============ 运动学变化检测：找分段边界 ============
def _find_split_points(points, n_stages):
    """在运动学变化最显著的位置切分。返回 [(i0, i1), ...]。"""
    T = len(points)
    if n_stages <= 1:
        return [(0, T)]
    if T < n_stages * 5:
        bounds = [int(i * T / n_stages) for i in range(n_stages + 1)]
        return [(bounds[i], bounds[i + 1]) for i in range(n_stages)]

    h = np.array([float(p["高度"]) for p in points])
    v = np.array([float(p["速度"]) for p in points])

    win = max(T // 20, 3)
    kernel = np.ones(win) / win
    h_smooth = np.convolve(h, kernel, mode='same')
    v_smooth = np.convolve(v, kernel, mode='same')

    dh = np.abs(np.gradient(h_smooth))
    dv = np.abs(np.gradient(v_smooth))
    dh = dh / (dh.max() + 1e-6)
    dv = dv / (dv.max() + 1e-6)
    score = dh + dv

    bounds = []
    prev = 0
    for i in range(1, n_stages):
        expected = int(i * T / n_stages)
        window = max(T // (n_stages * 4), 5)
        lo = max(prev + 3, expected - window)
        hi = min(T - (n_stages - i) * 3, expected + window)
        if hi <= lo:
            idx = expected
        else:
            idx = lo + int(np.argmax(score[lo:hi]))
        bounds.append(idx)
        prev = idx

    all_bounds = [0] + bounds + [T]
    return [(all_bounds[i], all_bounds[i + 1]) for i in range(n_stages)]


def _split_trajectory(points, stages):
    """按 stages 的阶段数切分航迹，每段对应一个阶段标签。"""
    bounds = _find_split_points(points, len(stages))
    segments = []
    for (i0, i1), label in zip(bounds, stages):
        chunk = points[i0:i1]
        if not chunk:
            chunk = [points[min(i0, len(points) - 1)]]
        segments.append({"points": chunk, "behavior": label})
    return segments


# ============ 时间范围格式化 ============
def _format_time_range(pts, is_absolute):
    if not pts:
        return ""
    t0 = pts[0]["时间戳"]
    t1 = pts[-1]["时间戳"]
    if is_absolute:
        return f"{t0} 至 {t1}"
    try:
        return f"{float(t0):.1f}s 至 {float(t1):.1f}s"
    except (ValueError, TypeError):
        return f"{t0} 至 {t1}"


# ============ CoT 目标构造 ============
def build_cot_target(instruction_text, answer_text, segments, is_absolute):
    stages, letter = _parse_stages_from_answer(answer_text)
    stem = instruction_text.split("\n")[0].strip()
    options = _parse_options(instruction_text)

    lines = ["<think>"]
    lines.append(f"本题要求：{stem}")
    lines.append(f"航迹时间戳类型：{'绝对时间' if is_absolute else '相对时间'}。")
    lines.append(f"将航迹按运动学变化分为 {len(segments)} 段，逐段分析：")

    for i, seg in enumerate(segments):
        pts = seg["points"]
        if not pts:
            continue
        time_range = _format_time_range(pts, is_absolute)
        h = np.array([float(p["高度"]) for p in pts])
        v = np.array([float(p["速度"]) for p in pts])
        hdg = np.array([float(p["航向角"]) for p in pts])

        h_change = h[-1] - h[0]
        h_range = h.max() - h.min()
        v_change = v[-1] - v[0]
        hdg_change = hdg[-1] - hdg[0]

        if abs(h_change) < 50 and h_range < 150:
            h_desc = "高度基本稳定"
        elif h_change > 200:
            h_desc = "持续爬升"
        elif h_change < -200:
            h_desc = "持续下降"
        else:
            h_desc = "高度周期波动"

        lines.append(
            f"段{i + 1}（{time_range}，共 {len(pts)} 个点）："
            f"高度 {h[0]:.0f}→{h[-1]:.0f}m（范围 {h_range:.0f}m，{h_desc}），"
            f"速度 {v[0]:.0f}→{v[-1]:.0f}m/s（变化 {v_change:+.0f}），"
            f"航向 {hdg[0]:.0f}°→{hdg[-1]:.0f}°（变化 {hdg_change:+.0f}°）。"
            f"综合判断阶段为「{seg['behavior']}」。"
        )

    if options:
        lines.append("各选项内容：")
        for k in sorted(options.keys()):
            lines.append(f"  {k}. {options[k]}")

    lines.append(
        f"航迹实际阶段序列为：{' → '.join(stages)}。"
        f"与选项比对，只有选项 {letter}「{answer_text}」与实际序列一致。"
    )
    lines.append("</think>")
    lines.append(f"答案是：{answer_text}")
    return "\n".join(lines)


class ADSBQADataset(Dataset):
    """
    输入：
        jsonl_path:       每行 {"样本编号","航迹数据":{"航迹编号","起始时间","点位序列"},"文本指令"}
        answer_csv_path:  每行 序号,样本编号,任务描述,答案
    """
    def __init__(self, jsonl_path, answer_csv_path, tokenizer,
                 max_n_seg=8, max_seg_len=32, max_text_len=1024):
        self.tokenizer = tokenizer
        self.max_n_seg = max_n_seg
        self.max_seg_len = max_seg_len
        self.max_text_len = max_text_len

        # ---------- 1) 读答案 CSV ----------
        answers = {}
        with open(answer_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                try:
                    int(row[0])
                except (ValueError, TypeError):
                    continue
                sid = row[1].strip()
                if sid:
                    answers[sid] = row[3].strip()
        print(f"[answers] 解析到 {len(answers)} 条答案")

        # ---------- 2) 读 jsonl ----------
        self.samples = []
        missing_ans = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("样本编号")
                if not sid:
                    continue
                if sid not in answers:
                    missing_ans += 1
                    continue
                traj = obj.get("航迹数据", {}) or {}
                points = traj.get("点位序列", [])
                instruction = obj.get("文本指令", "")
                if not points or not instruction:
                    continue

                is_absolute = _is_absolute_time(points[0].get("时间戳"))
                self.samples.append({
                    "sample_id":   sid,
                    "points":      points,
                    "start_iso":   traj.get("起始时间", ""),
                    "is_absolute": is_absolute,
                    "instruction": instruction,
                    "answer":      answers[sid],
                })

        if missing_ans:
            print(f"[warn] {missing_ans} 条 jsonl 样本无答案，已跳过")
        n_abs = sum(1 for s in self.samples if s["is_absolute"])
        print(f"[ADSBQADataset] loaded {len(self.samples)} samples "
              f"(绝对时间 {n_abs} / 相对时间 {len(self.samples)-n_abs})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---------- 从答案解析阶段数，切分航迹 ----------
        stages, _ = _parse_stages_from_answer(s["answer"])
        if not stages:
            stages = ["未知", "未知"]
        if len(stages) > self.max_n_seg:
            stages = stages[: self.max_n_seg]

        segments = _split_trajectory(s["points"], stages)

        # ---------- 编码每段 ----------
        seg_feats_list, glob_feats_list = [], []
        for seg in segments:
            feats = build_trajectory_features(seg["points"], self.max_seg_len)
            g = _global_features(seg["points"])
            seg_feats_list.append(feats)
            glob_feats_list.append(torch.from_numpy(g))

        n_valid = len(seg_feats_list)
        while len(seg_feats_list) < self.max_n_seg:
            seg_feats_list.append(torch.zeros(self.max_seg_len, 11))
            glob_feats_list.append(torch.zeros(20))

        seg_feats    = torch.stack(seg_feats_list, dim=0)     # (max_n_seg, seg_len, 11)
        global_feats = torch.stack(glob_feats_list, dim=0)    # (max_n_seg, 20)
        seg_mask = torch.zeros(self.max_n_seg, dtype=torch.long)
        seg_mask[:n_valid] = 1

        # ---------- Prompt ----------
        user_content = s["instruction"]
        if s["start_iso"]:
            user_content = f"该航迹的起始时间为 {s['start_iso']}。\n{user_content}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        )

        # ---------- CoT 目标 ----------
        target_text = build_cot_target(
            s["instruction"], s["answer"], segments, s["is_absolute"],
        )
        answer_ids = self.tokenizer(
            target_text, add_special_tokens=False
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        # ---------- 拼接 ----------
        input_ids = prompt_ids + answer_ids
        labels    = [-100] * len(prompt_ids) + answer_ids

        if len(input_ids) > self.max_text_len:
            input_ids = input_ids[: self.max_text_len]
            labels    = labels[: self.max_text_len]
        else:
            pad = self.max_text_len - len(input_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_id] * pad
            labels    = labels + [-100] * pad

        attention_mask = [
            1 if tok != (self.tokenizer.pad_token_id or 0) else 0
            for tok in input_ids
        ]

        return {
            "seg_feats":      seg_feats,
            "global_feats":   global_feats,
            "seg_mask":       seg_mask,
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }