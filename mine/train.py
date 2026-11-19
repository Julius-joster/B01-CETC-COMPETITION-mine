# train.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from dataset import ADSBQADataset
from segment_encoder import SegmentEncoder
from modeling_qwen_traj import QwenTrajForCausalLM


LOCAL_MODEL_PATH     = "E:/models/Qwen3.5-0.8B"
JSONL_PATH           = "E:/desktop/比赛/last/B01-CETC-COMPETITION-main/数据集与预处理/生成数据集/task1/task1/01_segment_questions.jsonl"
ANSWER_CSV_PATH      = "E:/desktop/比赛/last/B01-CETC-COMPETITION-main/数据集与预处理/生成数据集/task1/task1_label/answer.csv"
SEGMENT_ENCODER_CKPT = "E:/desktop/比赛/last/ckpt_segment_encoder/encoder.pt"
OUTPUT_DIR           = "./ckpt_qwen_traj"

MAX_N_SEG    = 8
MAX_SEG_LEN  = 32
MAX_TEXT_LEN = 1024
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 2e-4
EPOCHS       = 3
SEG_D_MODEL  = 256


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
    qwen.config.use_cache = False
    for p in qwen.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    qwen = get_peft_model(qwen, lora_cfg)
    qwen.gradient_checkpointing_enable()
    qwen.enable_input_require_grads()

    # 加载阶段1片段编码器
    traj_encoder = SegmentEncoder(
        d_feat=11, d_model=SEG_D_MODEL, n_global=20,
        patch_len=2, stride=1, dropout=0.1,
    ).to(dtype=torch.bfloat16, device=qwen.device)
    ckpt = torch.load(SEGMENT_ENCODER_CKPT, map_location=qwen.device)
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        traj_encoder.load_state_dict(ckpt["encoder"])
        print(f"[loaded] segment encoder, "
              f"behavior_types={ckpt.get('behavior_types', [])}")
    else:
        traj_encoder.load_state_dict(ckpt)
        print("[loaded] segment encoder (legacy format)")

    hidden_size = qwen.config.hidden_size
    model = QwenTrajForCausalLM(
        qwen, traj_encoder, hidden_size, tokenizer=tokenizer,
    ).to(qwen.device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all   = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_all:,} = {n_train/n_all:.2%}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)

    dataset = ADSBQADataset(
        jsonl_path=JSONL_PATH,
        answer_csv_path=ANSWER_CSV_PATH,
        tokenizer=tokenizer,
        max_n_seg=MAX_N_SEG,
        max_seg_len=MAX_SEG_LEN,
        max_text_len=MAX_TEXT_LEN,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    global_step = 0
    for epoch in range(EPOCHS):
        for step, batch in enumerate(loader):
            seg_feats      = batch["seg_feats"].to(qwen.device, dtype=torch.bfloat16)
            global_feats   = batch["global_feats"].to(qwen.device, dtype=torch.bfloat16)
            seg_mask       = batch["seg_mask"].to(qwen.device)
            input_ids      = batch["input_ids"].to(qwen.device)
            attention_mask = batch["attention_mask"].to(qwen.device)
            labels         = batch["labels"].to(qwen.device)

            out = model(
                seg_feats=seg_feats,
                global_feats=global_feats,
                seg_mask=seg_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if step % 20 == 0:
                print(f"epoch={epoch} step={step} "
                      f"loss={out.loss.item():.4f} global={global_step}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.qwen.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch{epoch}_lora"))
        torch.save(
            {"segment_encoder": model.traj_encoder.state_dict(),
             "traj_proj":       model.traj_proj.state_dict()},
            os.path.join(OUTPUT_DIR, f"epoch{epoch}_seg.pt"),
        )
        print(f"[saved] epoch={epoch}")


if __name__ == "__main__":
    main()