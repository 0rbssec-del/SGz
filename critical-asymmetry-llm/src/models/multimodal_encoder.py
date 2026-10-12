"""Encodeur multi-modal vision + langage."""
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel


class MultiModalEncoder(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14",
                 text_model_name="meta-llama/Llama-3-8B",
                 fusion_dim=4096, n_heads=8, dtype=torch.bfloat16):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(vision_model_name, torch_dtype=dtype)
        self.text = AutoModel.from_pretrained(text_model_name, torch_dtype=dtype)
        self.vision_proj = nn.Linear(self.vision.config.hidden_size, fusion_dim)
        self.text_proj = nn.Linear(self.text.config.hidden_size, fusion_dim)
        self.cross_attn_tv = nn.MultiheadAttention(fusion_dim, n_heads, batch_first=True)
        self.cross_attn_vt = nn.MultiheadAttention(fusion_dim, n_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(fusion_dim)
        self.norm_v = nn.LayerNorm(fusion_dim)
        self.fusion_dim = fusion_dim

    def forward(self, input_ids, attention_mask, pixel_values, lambda_v=0.5):
        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        vision_out = self.vision(pixel_values=pixel_values).last_hidden_state
        t = self.text_proj(text_out)
        v = self.vision_proj(vision_out)
        t_attn, attn_tv = self.cross_attn_tv(query=t, key=v, value=v, need_weights=True)
        v_attn, attn_vt = self.cross_attn_vt(query=v, key=t, value=t, need_weights=True)
        t_fused = self.norm_t(t + lambda_v * t_attn)
        v_fused = self.norm_v(v + lambda_v * v_attn)
        fused = torch.cat([t_fused, v_fused], dim=1)
        return {"fused": fused, "text": t_fused, "vision": v_fused,
                "attn_tv": attn_tv, "attn_vt": attn_vt}
