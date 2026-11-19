# modeling_qwen_traj.py
import torch
import torch.nn as nn


DEFAULT_DOMAIN_PROMPT = (
    "分析ADS-B航迹。航迹已被划分为若干阶段片段，片段类型包括："
    "跑道巡逻/直线飞行/巡航/盘旋绕飞/拓展盘旋/起飞/滑行/进近等。"
    "按片段的行为标签序列判断飞行阶段顺序，"
    "先在<think></think>中推理，再给答案。"
)


class QwenTrajForCausalLM(nn.Module):
    def __init__(self, qwen_model, traj_encoder, qwen_hidden_size, tokenizer,
                 domain_prompt=DEFAULT_DOMAIN_PROMPT, dtype=torch.bfloat16,
                 lam_cons=0.0, lam_smooth=0.0):
        super().__init__()
        self.qwen = qwen_model
        self.traj_encoder = traj_encoder
        self.traj_proj = nn.Linear(
            traj_encoder.output_dim, qwen_hidden_size, dtype=dtype
        )
        nn.init.normal_(self.traj_proj.weight, std=0.02)
        nn.init.zeros_(self.traj_proj.bias)

        # 阶段二默认关闭物理头（阶段一的多任务训练已提供运动学判别力）
        self.lam_cons = lam_cons
        self.lam_smooth = lam_smooth

        with torch.no_grad():
            ids = tokenizer(
                domain_prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
        self.register_buffer("domain_prompt_ids", ids, persistent=False)
        self.domain_prompt_len = ids.size(1)

    def _domain_embeds(self):
        emb = self.qwen.get_input_embeddings()
        return emb(self.domain_prompt_ids.to(self.traj_proj.weight.device))

    def forward(self, seg_feats, global_feats, seg_mask,
                input_ids, attention_mask, labels=None):
        # seg_feats:    (B, n_seg, seg_len, 11)
        # global_feats: (B, n_seg, 20)
        # seg_mask:     (B, n_seg)
        B, n_seg, seg_len, d_feat = seg_feats.shape

        # 1) 扁平化逐片段编码
        flat_seg  = seg_feats.view(B * n_seg, seg_len, d_feat)
        flat_glob = global_feats.view(B * n_seg, -1)
        seg_hidden = self.traj_encoder(flat_seg, flat_glob)    # (B*n_seg, D)
        seg_hidden = seg_hidden.view(B, n_seg, -1)             # (B, n_seg, D)

        # 2) 片段软提示
        traj_embeds = self.traj_proj(seg_hidden)               # (B, n_seg, H)

        # 3) 领域提示
        domain_embeds = self._domain_embeds().expand(B, -1, -1)

        # 4) 文本嵌入
        text_embeds = self.qwen.get_input_embeddings()(input_ids)

        # 5) 拼接：[领域提示] + [片段软提示] + [问题文本]
        inputs_embeds = torch.cat([domain_embeds, traj_embeds, text_embeds], dim=1)

        # 6) mask
        dev = attention_mask.device
        domain_mask = torch.ones(B, self.domain_prompt_len,
                                 dtype=attention_mask.dtype, device=dev)
        seg_mask_ = seg_mask.to(dtype=attention_mask.dtype, device=dev)
        full_mask = torch.cat([domain_mask, seg_mask_, attention_mask], dim=1)

        # 7) labels 前缀补 -100
        if labels is not None:
            prefix_len = self.domain_prompt_len + n_seg
            prefix_labels = torch.full(
                (labels.size(0), prefix_len), -100,
                dtype=labels.dtype, device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs