# pretrain_segment.py
import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from segment_encoder import SegmentEncoder, MultiTaskSegmentClassifier
from segment_dataset import SegmentCSVDataset, SPEED_TYPES, HEIGHT_TYPES


SEG_CSV     = "E:/desktop/比赛/last/B01-CETC-COMPETITION-main/数据集与预处理/生成数据集/task1/mine/point_labels_with_layers_and_phase200_add.csv"
OUTPUT_PATH = "./ckpt_segment_encoder/encoder.pt"

SEG_LEN       = 32
D_MODEL       = 256
BATCH_SIZE    = 32
LR            = 1e-3
EPOCHS        = 40
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
WARMUP_EPOCHS = 2
VAL_RATIO     = 0.2
SEED          = 42

W_COMP    = 1.0
W_SPEED   = 0.3
W_HEIGHT  = 0.3
W_REGRESS = 0.5

REGRESS_NAMES = ["持续时间", "起始位置", "飞行距离", "飞行方向", "高度差距"]
# 打印时换算回物理单位
REGRESS_SCALE = [600.0, 3600.0, 20000.0, 180.0, 5000.0]


def split_indices(n, val_ratio=0.2, seed=42):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(int(n * val_ratio), 1)
    return indices[n_val:], indices[:n_val]


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device, ce):
    model.eval()
    total_loss = 0.0
    correct_comp = correct_speed = correct_height = 0
    regress_abs_err = torch.zeros(5, device=device)
    total = 0
    with torch.no_grad():
        for batch in loader:
            seg_feats    = batch["seg_feats"].to(device)
            global_feats = batch["global_feats"].to(device)
            y_comp       = batch["behavior_id"].to(device)
            y_speed      = batch["speed_id"].to(device)
            y_height     = batch["height_id"].to(device)
            y_reg        = batch["regress_targets"].to(device)

            out = model(seg_feats, global_feats)
            loss = (W_COMP    * ce(out["comprehensive"], y_comp)
                  + W_SPEED   * ce(out["speed"],         y_speed)
                  + W_HEIGHT  * ce(out["height"],        y_height)
                  + W_REGRESS * F.smooth_l1_loss(out["regress"], y_reg))

            bs = y_comp.size(0)
            total_loss += loss.item() * bs
            correct_comp   += (out["comprehensive"].argmax(-1) == y_comp).sum().item()
            correct_speed  += ((out["speed"].argmax(-1)    == y_speed)  & (y_speed  >= 0)).sum().item()
            correct_height += ((out["height"].argmax(-1)   == y_height) & (y_height >= 0)).sum().item()
            regress_abs_err += (out["regress"] - y_reg).abs().sum(dim=0)
            total += bs

    return (total_loss / total, correct_comp / total,
            correct_speed / total, correct_height / total,
            (regress_abs_err / total).cpu())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 先加载数据集（自动发现行为标签）
    ds_train = SegmentCSVDataset(SEG_CSV, seg_len=SEG_LEN, augment=True, seed=SEED)
    ds_val   = SegmentCSVDataset(
        SEG_CSV, seg_len=SEG_LEN, augment=False,
        behavior_types=ds_train.behavior_types,   # 验证集用同一套标签
    )

    # 2) 再构造模型（类别数从数据集获得）
    n_behavior = len(ds_train.behavior_types)
    encoder = SegmentEncoder(
        d_feat=11, d_model=D_MODEL, n_global=20,
        patch_len=2, stride=1, dropout=0.1,
    ).to(device)
    model = MultiTaskSegmentClassifier(
        encoder,
        n_comprehensive=n_behavior,
        n_speed=len(SPEED_TYPES),
        n_height=len(HEIGHT_TYPES),
        n_regress=5,
    ).to(device)

    train_idx, val_idx = split_indices(len(ds_train), VAL_RATIO, SEED)
    print(f"[split] train={len(train_idx)}  val={len(val_idx)}")

    train_loader = DataLoader(Subset(ds_train, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Subset(ds_val, val_idx),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    ce = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=LABEL_SMOOTH)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_comp = correct_speed = correct_height = 0
        total = 0

        for batch in train_loader:
            seg_feats    = batch["seg_feats"].to(device)
            global_feats = batch["global_feats"].to(device)
            y_comp       = batch["behavior_id"].to(device)
            y_speed      = batch["speed_id"].to(device)
            y_height     = batch["height_id"].to(device)
            y_reg        = batch["regress_targets"].to(device)

            out = model(seg_feats, global_feats)
            loss = (W_COMP    * ce(out["comprehensive"], y_comp)
                  + W_SPEED   * ce(out["speed"],         y_speed)
                  + W_HEIGHT  * ce(out["height"],        y_height)
                  + W_REGRESS * F.smooth_l1_loss(out["regress"], y_reg))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bs = y_comp.size(0)
            total_loss += loss.item() * bs
            correct_comp   += (out["comprehensive"].argmax(-1) == y_comp).sum().item()
            correct_speed  += ((out["speed"].argmax(-1)    == y_speed)  & (y_speed  >= 0)).sum().item()
            correct_height += ((out["height"].argmax(-1)   == y_height) & (y_height >= 0)).sum().item()
            total += bs

        train_loss = total_loss / total
        train_acc  = correct_comp / total

        val_loss, val_comp, val_speed, val_height, regress_err = evaluate(
            model, val_loader, device, ce
        )

        regress_str = " ".join(
            f"{name}={err*scale:.2f}" for name, err, scale in
            zip(REGRESS_NAMES, regress_err.tolist(), REGRESS_SCALE)
        )

        print(f"epoch={epoch} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc_comp={val_comp:.4f} "
              f"val_acc_speed={val_speed:.4f} val_acc_height={val_height:.4f}")
        print(f"  val_regress_MAE: {regress_str}")

        if val_comp > best_val_acc:
            best_val_acc = val_comp
            best_state = copy.deepcopy(model.state_dict())
            print(f"  [best] val_acc_comp={best_val_acc:.4f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    # 保存 encoder 权重 + 行为标签列表（阶段二需要知道类别数和标签顺序）
    torch.save({
        "encoder":         model.encoder.state_dict(),
        "behavior_types":  ds_train.behavior_types,
    }, OUTPUT_PATH)
    print(f"[saved] {OUTPUT_PATH}  (best val_acc_comp={best_val_acc:.4f})")


if __name__ == "__main__":
    main()