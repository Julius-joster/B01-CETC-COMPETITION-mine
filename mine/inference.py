# inference.py
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from dataset import (
    ADSBQADataset, SYSTEM_PROMPT,
    _is_absolute_time, _find_split_points, _split_trajectory,
)
from traj_features import build_trajectory_features
from segment_dataset import _global_features
from segment_encoder import SegmentEncoder
from modeling_qwen_traj import QwenTrajForCausalLM


LOCAL_MODEL_PATH = "E:/models/Qwen3.5-0.8B"
LORA_DIR         = "./ckpt_qwen_traj/epoch0_lora"
SEG_CKPT         = "./ckpt_qwen_traj/epoch0_seg.pt"
JSONL_PATH       = "./data/01_segment_questions.jsonl"
TARGET_SAMPLE_ID = "seg_syn_v5_height_000001_concat"

MAX_N_SEG   = 8
MAX_SEG_LEN = 32
SEG_D_MODEL = 256


def load_sample(jsonl_path, sample_id):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("样本编号") == sample_id:
                return obj
    raise ValueError(f"找不到样本 {sample_id}")


def _infer_stage_count(points):
    """推理时不知道真实阶段数，按点数动态估计。"""
    T = len(points)
    if T < 40:
        return 2
    if T < 200:
        return 3
    return 4


@torch.no_grad()
def answer_one(model, tokenizer, sample, device, max_new_tokens=512):
    traj = sample["航迹数据"]
    points = traj["点位序列"]
    start_iso = traj.get("起始时间", "")
    is_absolute = _is_absolute_time(points[0].get("时间戳"))

    # 推理时用动态阶段数切分
    n_stages = min(_infer_stage_count(points), MAX_N_SEG)
    stages_placeholder = [f"阶段{i+1}" for i in range(n_stages)]
    segments = _split_trajectory(points, stages_placeholder)

    seg_feats_list, glob_list = [], []
    for seg in segments:
        seg_feats_list.append(build_trajectory_features(seg["points"], MAX_SEG_LEN))
        glob_list.append(torch.from_numpy(_global_features(seg["points"])))
    n_valid = len(seg_feats_list)
    while len(seg_feats_list) < MAX_N_SEG:
        seg_feats_list.append(torch.zeros(MAX_SEG_LEN, 11))
        glob_list.append(torch.zeros(20))
    seg_feats    = torch.stack(seg_feats_list, 0).unsqueeze(0).to(device, dtype=torch.bfloat16)
    global_feats = torch.stack(glob_list, 0).unsqueeze(0).to(device, dtype=torch.bfloat16)
    seg_mask = torch.zeros(1, MAX_N_SEG, dtype=torch.long, device=device)
    seg_mask[0, :n_valid] = 1

    user_content = sample["文本指令"]
    if start_iso:
        user_content = f"该航迹的起始时间为 {start_iso}。\n{user_content}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    domain_embeds = model._domain_embeds().expand(1, -1, -1)
    flat_seg  = seg_feats.view(MAX_N_SEG, MAX_SEG_LEN, 11)
    flat_glob = global_feats.view(MAX_N_SEG, -1)
    seg_hidden = model.traj_encoder(flat_seg, flat_glob).unsqueeze(0)
    traj_embeds = model.traj_proj(seg_hidden)
    text_embeds = model.qwen.get_input_embeddings()(prompt_ids)
    inputs_embeds = torch.cat([domain_embeds, traj_embeds, text_embeds], dim=1)

    domain_mask = torch.ones(1, model.domain_prompt_len, dtype=torch.long, device=device)
    full_mask = torch.cat([domain_mask, seg_mask, torch.ones_like(prompt_ids)], dim=1)

    gen = model.qwen.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
        local_files_only=True, use_safetensors=True,
    )
    qwen = PeftModel.from_pretrained(qwen, LORA_DIR)

    hidden = qwen.config.hidden_size
    traj_encoder = SegmentEncoder(
        d_feat=11, d_model=SEG_D_MODEL, n_global=20,
        patch_len=2, stride=1, dropout=0.1,
    ).to(dtype=torch.bfloat16, device=qwen.device)
    model = QwenTrajForCausalLM(
        qwen, traj_encoder, hidden, tokenizer=tokenizer,
    ).to(qwen.device)

    ckpt = torch.load(SEG_CKPT, map_location=qwen.device)
    model.traj_encoder.load_state_dict(ckpt["segment_encoder"])
    model.traj_proj.load_state_dict(ckpt["traj_proj"])
    model.eval()

    sample = load_sample(JSONL_PATH, TARGET_SAMPLE_ID)
    pred = answer_one(model, tokenizer, sample, qwen.device)
    print(f"[{TARGET_SAMPLE_ID}] 输出：\n{pred}")


if __name__ == "__main__":
    main()