from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torchvision.transforms import functional as TF


LANCE_CODE_COMMIT = "4baeee086648996f6ab12e673cbe461b0b149997"
LANCE_MODEL_REVISION = "7395315758865e6f56ab87ad06a88c7ac172f056"
LANCE_HIDDEN_SIZE = 2048
LANCE_LAYERS = 36
LANCE_VIT_IMAGE_SIZE = 448
LANCE_VIT_PATCH_SIZE = 14
LANCE_VIT_MERGE_SIZE = 2
LANCE_IMAGE_TOKENS = 256
WAN_IMAGE_SIZE = 512
WAN_LATENT_CHANNELS = 48
WAN_LATENT_SIDE = 32

QWEN_VIT_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_VIT_STD = (0.26862954, 0.26130258, 0.27577711)


class DeviceAwareMaskList(list[torch.Tensor]):
    """Preserve Lance's per-sample mask list while satisfying decoder `.to()`."""

    def to(self, *args: Any, **kwargs: Any) -> "DeviceAwareMaskList":
        return DeviceAwareMaskList(mask.to(*args, **kwargs) for mask in self)


class LanceCompat(nn.Module):
    """Expose the narrow packed-language contract used by the G-token trainer."""

    def __init__(self, lance: nn.Module, tokenizer: Any, new_token_ids: dict[str, int]) -> None:
        super().__init__()
        self.lance = lance
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

    @property
    def vit_model(self) -> nn.Module:
        return self.lance.vit_model

    def _get_base_language_model(self) -> nn.Module:
        return self.lance.language_model

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lance.language_model.model.embed_tokens(token_ids)

    def _lm_forward_train(
        self,
        *,
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask: list[torch.Tensor],
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.Tensor,
        packed_gen_token_indexes: torch.Tensor,
    ) -> torch.Tensor:
        return self.lance.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=DeviceAwareMaskList(attention_mask),
            packed_position_ids=packed_position_ids,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )


def _assert_runtime_pin(runtime: Path) -> None:
    marker = json.loads((runtime / "RUNTIME_PIN.json").read_text(encoding="utf-8"))
    if marker.get("commit") != LANCE_CODE_COMMIT:
        raise RuntimeError(f"Lance runtime commit drift: {marker}")


def _assert_model_pin(model_root: Path) -> None:
    marker = json.loads((model_root / "DOWNLOAD_COMPLETE.json").read_text(encoding="utf-8"))
    if marker.get("revision") != LANCE_MODEL_REVISION:
        raise RuntimeError(f"Lance model revision drift: {marker.get('revision')}")


def load_lance_native(runtime: str, model_root: str, device: torch.device) -> LanceCompat:
    runtime_path = Path(runtime).resolve()
    model_path = Path(model_root).resolve()
    _assert_runtime_pin(runtime_path)
    _assert_model_pin(model_path)

    from data.data_utils import add_special_tokens
    from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.qwen2.modeling_qwen2 import Qwen2Config
    from modeling.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    llm_dir = model_path / "Lance_3B"
    vit_dir = model_path / "Qwen2.5-VL-ViT"
    llm_config = Qwen2Config.from_json_file(str(llm_dir / "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.qk_norm_und = True
    llm_config.qk_norm_gen = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    llm_config.apply_qwen_2_5_vl_pos_emb = False
    language_model = Qwen2ForCausalLM(llm_config)

    vit_config = Qwen2_5_VLVisionConfig.from_pretrained(str(vit_dir))
    vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
    vit_model.load_state_dict(load_file(str(vit_dir / "vit.safetensors")), strict=True)

    config = LanceConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        latent_patch_size=(1, 1, 1),
        max_num_frames=1,
        max_latent_size=64,
        vit_max_num_patch_per_side=64,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=4.0,
    )
    lance = Lance(
        language_model=language_model,
        vit_model=vit_model,
        vit_type="qwen_2_5_vl_original",
        config=config,
        training_args=SimpleNamespace(apply_qwen_2_5_vl_pos_emb=False),
    )
    state = load_file(str(llm_dir / "model.safetensors"), device="cpu")
    state.pop("latent_pos_embed.pos_embed", None)
    report = lance.load_state_dict(state, strict=False)
    real_missing = [name for name in report.missing_keys if not name.startswith("vit_model.")]
    allowed_unexpected_prefixes = (
        "time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.",
    )
    real_unexpected = [
        name for name in report.unexpected_keys
        if not name.startswith(allowed_unexpected_prefixes)
    ]
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"Lance checkpoint mismatch: missing={real_missing[:8]} "
            f"unexpected={real_unexpected[:8]}"
        )
    del state

    tokenizer = Qwen2Tokenizer.from_pretrained(str(llm_dir))
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens:
        raise RuntimeError(f"official Lance tokenizer unexpectedly needed {num_new_tokens} tokens")
    lance.update_tokenizer(tokenizer)
    lance.to(device=device, dtype=torch.bfloat16)
    return LanceCompat(lance, tokenizer, new_token_ids)


def load_qwen_vit_teacher(runtime: str, model_root: str, device: torch.device) -> nn.Module:
    """Load only Lance's understanding tower for Stage1 relation supervision."""
    runtime_path = Path(runtime).resolve()
    model_path = Path(model_root).resolve()
    _assert_runtime_pin(runtime_path)
    _assert_model_pin(model_path)

    from modeling.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

    vit_dir = model_path / "Qwen2.5-VL-ViT"
    config = Qwen2_5_VLVisionConfig.from_pretrained(str(vit_dir))
    model = Qwen2_5_VisionTransformerPretrainedModel(config)
    model.load_state_dict(load_file(str(vit_dir / "vit.safetensors")), strict=True)
    return model.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)


def load_native_vae_projection(
    model_root: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Lance's frozen 48->2048 VAE projection used by the native loss."""
    model_path = Path(model_root).resolve()
    _assert_model_pin(model_path)
    with safe_open(
        str(model_path / "Lance_3B" / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        weight = handle.get_tensor("vae2llm.weight").to(device=device, dtype=torch.float32)
        bias = handle.get_tensor("vae2llm.bias").to(device=device, dtype=torch.float32)
    if weight.shape != (LANCE_HIDDEN_SIZE, WAN_LATENT_CHANNELS):
        raise RuntimeError(f"unexpected vae2llm weight shape: {tuple(weight.shape)}")
    if bias.shape != (LANCE_HIDDEN_SIZE,):
        raise RuntimeError(f"unexpected vae2llm bias shape: {tuple(bias.shape)}")
    return weight, bias


def preprocess_qwen_vit(images: list[Image.Image]) -> tuple[list[torch.Tensor], torch.Tensor]:
    from data.data_utils import patchify_video_with_merge

    patches: list[torch.Tensor] = []
    grids = []
    for image in images:
        resized = image.resize((LANCE_VIT_IMAGE_SIZE, LANCE_VIT_IMAGE_SIZE), Image.Resampling.BICUBIC)
        tensor = TF.pil_to_tensor(resized).float().div_(255.0)
        tensor = TF.normalize(tensor, QWEN_VIT_MEAN, QWEN_VIT_STD)
        video = torch.stack((tensor, tensor), dim=1)
        patches.append(patchify_video_with_merge(
            video,
            LANCE_VIT_PATCH_SIZE,
            temporal_patch_size=2,
            merge_size=LANCE_VIT_MERGE_SIZE,
        ))
        grids.append((1, LANCE_VIT_IMAGE_SIZE // LANCE_VIT_PATCH_SIZE,
                      LANCE_VIT_IMAGE_SIZE // LANCE_VIT_PATCH_SIZE))
    return patches, torch.tensor(grids, dtype=torch.long)


@torch.no_grad()
def extract_qwen_u(
    model: LanceCompat | nn.Module,
    patches: list[torch.Tensor],
    grids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    packed = torch.cat(patches).to(device=device, dtype=torch.bfloat16)
    vit_model = model.vit_model if isinstance(model, LanceCompat) else model
    hidden = vit_model(hidden_states=packed, grid_thw=grids.to(device))
    expected = len(patches) * LANCE_IMAGE_TOKENS
    if hidden.shape != (expected, LANCE_HIDDEN_SIZE):
        raise RuntimeError(f"unexpected Lance U shape: {tuple(hidden.shape)}")
    return hidden.reshape(len(patches), LANCE_IMAGE_TOKENS, LANCE_HIDDEN_SIZE)


class FrozenWanEncoder:
    def __init__(self, runtime: str, vae_path: str, device: torch.device) -> None:
        _assert_runtime_pin(Path(runtime).resolve())
        from modeling.vae.wan.vae2_2 import Wan2_2_VAE

        self.device = device
        self.vae = Wan2_2_VAE(vae_pth=vae_path, device=device, dtype=torch.bfloat16)
        self.vae.model.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        latents = []
        for image in images:
            resized = image.resize((WAN_IMAGE_SIZE, WAN_IMAGE_SIZE), Image.Resampling.BICUBIC)
            tensor = TF.pil_to_tensor(resized).float().div_(127.5).sub_(1.0)
            sample = tensor.unsqueeze(0).unsqueeze(2).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mean, _log_var = self.vae.encode(sample)
            if mean.shape != (1, WAN_LATENT_CHANNELS, 1, WAN_LATENT_SIDE, WAN_LATENT_SIDE):
                raise RuntimeError(f"unexpected Wan image latent shape: {tuple(mean.shape)}")
            latents.append(mean[0, :, 0].float())
        return torch.stack(latents)


def flatten_wan_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or latents.shape[1:] != (
        WAN_LATENT_CHANNELS, WAN_LATENT_SIDE, WAN_LATENT_SIDE,
    ):
        raise ValueError(f"unexpected Wan latent batch: {tuple(latents.shape)}")
    return rearrange(latents, "b c h w -> b (h w) c")


def fixed_g_position_embedding(device: torch.device) -> torch.Tensor:
    from modeling.lance.modeling_utils import get_2d_sincos_pos_embed

    values = get_2d_sincos_pos_embed(LANCE_HIDDEN_SIZE, 16)
    return torch.from_numpy(np.asarray(values)).to(device=device, dtype=torch.float32).unsqueeze(0)
