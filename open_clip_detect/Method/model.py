from typing import Optional

import torch
from torch import nn
from functools import partial

from open_clip_detect.Config.clip_vision_cfg import CLIPVisionCfg
from open_clip_detect.Config.clip_text_cfg import CLIPTextCfg
from open_clip_detect.Model.Layer.hf_model import HFTextEncoder
from open_clip_detect.Model.Layer.modified_resnet import ModifiedResNet
from open_clip_detect.Model.Layer.timm_model import TimmModel
from open_clip_detect.Model.Layer.transformer import LayerNormFp32, LayerNorm, QuickGELU, VisionTransformer, TextTransformer


def build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text
