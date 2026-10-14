from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn


MODEL_REVISION = "960ab33191f61342a4c60ae74d8dc356a39fafcb"
CODE_REVISION = "1daa72fa409002d40931bd7b36a9280362469ead"
CODE_TREE_SHA256 = "bb9398a4d14ddf35bc27f01f65578b816fc55f8c4e4489f9dde0204efb360fe7"
MODEL_WEIGHT_SHA256 = "ea7cf164cbed272be2a9999bc4c314da6a6f23ef51871ddef3afc2c0c430cc3f"
MODEL_CONFIG_SHA256 = "f139271ec1933cee5bd9242387da884dc4afd0ddd34ecc9987c0db60e5ce0443"
WIDTH = 2048
LAYERS = 24
U_TOKENS = 576
VQ_SIDE = 24
VQ_CHANNELS = 8
CODEBOOK_SIZE = 16384


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(code_root: str, model_root: str, *, verify_large_weight: bool = False) -> None:
    code, model = Path(code_root).resolve(), Path(model_root).resolve()
    code_marker = json.loads((code / "RUNTIME_PIN.json").read_text(encoding="utf-8"))
    model_marker = json.loads((model / "DOWNLOAD_COMPLETE.json").read_text(encoding="utf-8"))
    if code_marker.get("commit") != CODE_REVISION:
        raise RuntimeError(f"Janus code revision drift: {code_marker}")
    if code_marker.get("tree_sha256") != CODE_TREE_SHA256:
        raise RuntimeError(f"Janus code tree contract drift: {code_marker}")
    if model_marker.get("revision") != MODEL_REVISION:
        raise RuntimeError(f"Janus model revision drift: {model_marker}")
    if model_marker.get("weight_sha256") != MODEL_WEIGHT_SHA256:
        raise RuntimeError("Janus model marker lacks the verified weight SHA256")
    if sha256(model / "config.json") != MODEL_CONFIG_SHA256:
        raise RuntimeError("Janus config SHA256 drift")
    weight = model / "pytorch_model.bin"
    if weight.stat().st_size != 4_178_890_389:
        raise RuntimeError(f"Janus weight byte-size drift: {weight.stat().st_size}")
    if verify_large_weight and sha256(weight) != MODEL_WEIGHT_SHA256:
        raise RuntimeError("Janus weight SHA256 drift")


def load_janus(code_root: str, model_root: str, device: torch.device) -> tuple[nn.Module, Any]:
    verify_assets(code_root, model_root)
    resolved_code = str(Path(code_root).resolve())
    if resolved_code not in sys.path:
        sys.path.insert(0, resolved_code)
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: F401
    from transformers import AutoModelForCausalLM

    processor = VLChatProcessor.from_pretrained(model_root)
    model = AutoModelForCausalLM.from_pretrained(
        model_root, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    if len(model.language_model.model.layers) != LAYERS:
        raise RuntimeError("Janus Llama layer count drift")
    return model, processor


def preprocess_images(processor: Any, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    pixel = processor.image_processor(images, return_tensors="pt").pixel_values
    pixel = pixel.to(device=device, dtype=torch.bfloat16)
    if pixel.shape != (len(images), 3, 384, 384):
        raise RuntimeError(f"Janus pixel contract drift: {tuple(pixel.shape)}")
    return pixel


@torch.no_grad()
def extract_u(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model.vision_model(pixel_values)
        aligned = model.aligner(raw)
    if raw.shape != (pixel_values.shape[0], U_TOKENS, 1024):
        raise RuntimeError(f"Janus raw-U shape drift: {tuple(raw.shape)}")
    if aligned.shape != (pixel_values.shape[0], U_TOKENS, WIDTH):
        raise RuntimeError(f"Janus aligned-U shape drift: {tuple(aligned.shape)}")
    return aligned


@torch.no_grad()
def extract_vq(model: nn.Module, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prequant = model.gen_vision_model.quant_conv(model.gen_vision_model.encoder(pixel_values))
        quantized, _loss, info = model.gen_vision_model.quantize(prequant)
        indices = info[2]
        codebook = model.gen_vision_model.quantize.get_codebook_entry(
            indices, shape=list(quantized.shape), channel_first=True,
        )
    expected = (pixel_values.shape[0], VQ_CHANNELS, VQ_SIDE, VQ_SIDE)
    if codebook.shape != expected or indices.numel() != pixel_values.shape[0] * VQ_SIDE * VQ_SIDE:
        raise RuntimeError(f"Janus VQ shape drift: codebook={tuple(codebook.shape)} indices={tuple(indices.shape)}")
    indices = indices.reshape(pixel_values.shape[0], VQ_SIDE, VQ_SIDE)
    if int(indices.min()) < 0 or int(indices.max()) >= CODEBOOK_SIZE:
        raise RuntimeError("Janus VQ index is outside codebook")
    return codebook, indices


def native_generation_embedding(model: nn.Module, vq_tokens: torch.Tensor) -> torch.Tensor:
    if vq_tokens.shape[1:] != (VQ_SIDE * VQ_SIDE, VQ_CHANNELS):
        raise ValueError(f"native generation input shape drift: {tuple(vq_tokens.shape)}")
    return model.gen_aligner(vq_tokens)


@dataclass
class JanusLanguageBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.Tensor
    images_emb_mask: torch.Tensor
    answer_predictor_mask: torch.Tensor
    answer_labels: torch.Tensor
    answer_sample_ids: torch.Tensor

    def to(self, device: torch.device) -> "JanusLanguageBatch":
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name).to(device))
        return self


def build_language_batch(processor: Any, prompts: list[str], answers: list[str]) -> JanusLanguageBatch:
    if len(prompts) != len(answers):
        raise ValueError("prompt/answer batch mismatch")
    prepared = []
    answer_ranges: list[tuple[int, int]] = []
    blank = Image.new("RGB", (384, 384))
    for prompt, answer in zip(prompts, answers, strict=True):
        conversations = [
            {"role": "User", "content": f"{processor.image_token}\n{prompt}"},
            {"role": "Assistant", "content": answer},
        ]
        item = processor.process_one(conversations=conversations, images=[blank])
        empty = processor.process_one(
            conversations=[conversations[0], {"role": "Assistant", "content": ""}],
            images=[blank],
        )
        full_ids, empty_ids = item.input_ids.tolist(), empty.input_ids.tolist()
        start = 0
        while start < min(len(full_ids), len(empty_ids)) and full_ids[start] == empty_ids[start]:
            start += 1
        # Keep the complete answer-bearing suffix, including the official
        # template terminator, as causal supervision.
        if start >= len(full_ids):
            raise RuntimeError("Janus full/empty assistant templates have no answer suffix")
        answer_ranges.append((start, len(full_ids)))
        prepared.append(item)
    batch = processor.batchify(prepared)
    predictor_mask = torch.zeros_like(batch.attention_mask, dtype=torch.bool)
    labels: list[int] = []
    sample_ids: list[int] = []
    max_length = batch.input_ids.shape[1]
    for sample, (item, (start, end)) in enumerate(zip(prepared, answer_ranges, strict=True)):
        left = max_length - len(item.input_ids)
        for token_position in range(start, end):
            predictor = left + token_position - 1
            if predictor < left:
                raise RuntimeError("answer starts before a causal predictor position")
            predictor_mask[sample, predictor] = True
            labels.append(int(item.input_ids[token_position]))
            sample_ids.append(sample)
    return JanusLanguageBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        images_seq_mask=batch.images_seq_mask,
        images_emb_mask=batch.images_emb_mask,
        answer_predictor_mask=predictor_mask,
        answer_labels=torch.tensor(labels, dtype=torch.long),
        answer_sample_ids=torch.tensor(sample_ids, dtype=torch.long),
    )


def prepare_inputs_embeds(model: nn.Module, batch: JanusLanguageBatch, aligned_u: torch.Tensor) -> torch.Tensor:
    ids = batch.input_ids.clone()
    ids[ids < 0] = 0
    embeds = model.language_model.get_input_embeddings()(ids)
    selected_u = aligned_u.reshape(-1, aligned_u.shape[-1])
    if int(batch.images_seq_mask.sum()) != selected_u.shape[0]:
        raise RuntimeError("Janus image placeholder/U-token count mismatch")
    embeds[batch.images_seq_mask] = selected_u.to(embeds.dtype)
    return embeds


def fixed_g_position_embedding(device: torch.device, width: int = WIDTH, side: int = 16) -> torch.Tensor:
    if width % 4:
        raise ValueError("2D sinusoidal width must be divisible by four")
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    coords = torch.stack((y.reshape(-1), x.reshape(-1)), dim=1).float()
    quarter = width // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter).float() / quarter))
    values = torch.cat(
        (torch.sin(coords[:, :1] * omega), torch.cos(coords[:, :1] * omega),
         torch.sin(coords[:, 1:] * omega), torch.cos(coords[:, 1:] * omega)), dim=1,
    )
    return values.to(device=device, dtype=torch.float32).unsqueeze(0)


__all__ = [
    "CODE_REVISION", "LAYERS", "MODEL_REVISION", "U_TOKENS", "WIDTH", "JanusLanguageBatch",
    "build_language_batch", "extract_u", "extract_vq", "fixed_g_position_embedding", "load_janus",
    "native_generation_embedding", "prepare_inputs_embeds", "preprocess_images", "verify_assets",
]
