#!/usr/bin/env python3
"""Config-driven paper-main launcher for BAGEL, Lance, and Janus-Pro."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
MODEL_SCRIPTS = {
    "bagel": ROOT / "stage2" / "bagel" / "train_stage2.py",
    "lance": ROOT / "stage2" / "lance" / "train_stage2.py",
    "janus_pro": ROOT / "stage2" / "janus_pro" / "train_stage2.py",
}


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    model = payload.get("model")
    if model not in MODEL_SCRIPTS:
        raise ValueError(f"model must be one of {sorted(MODEL_SCRIPTS)}: {model!r}")
    runtime = payload.get("runtime", {})
    assets = payload.get("assets", {})
    stage1 = payload.get("stage1", {})
    stage2 = payload.get("stage2", {})
    stage2_data = stage2.get("data", {}) if isinstance(stage2, dict) else {}
    stage2_input = stage2.get("input", {}) if isinstance(stage2, dict) else {}
    stage1_training = stage1.get("training", {}) if isinstance(stage1, dict) else {}
    stage2_training = stage2.get("training", {}) if isinstance(stage2, dict) else {}
    if not all(isinstance(item, dict) for item in (runtime, assets, stage1, stage2, stage2_data, stage2_input, stage1_training, stage2_training)):
        raise ValueError("config.runtime, assets, stage1, stage2, stage2.data, stage2.input, stage1.training, and stage2.training must be mappings")
    payload["python"] = runtime.get("python", sys.executable)
    payload["gpus"] = str(runtime.get("gpus", "0"))
    checkpoint = stage2_input.get("stage1_checkpoint")
    if not checkpoint:
        stage1_output = stage1.get("output_dir")
        stage1_steps = stage1_training.get("max_steps", 8000)
        if not stage1_output:
            raise ValueError("stage1.output_dir is required when stage2.input.stage1_checkpoint is omitted")
        checkpoint = str(Path(str(stage1_output)) / f"checkpoint-{int(stage1_steps)}" / "flexae.safetensors")
    payload["paths"] = {"manifest": stage2_data.get("train"), "val_manifest": stage2_data.get("validation"), "flexae_checkpoint": checkpoint, "output_dir": stage2.get("output_dir"), "assets_root": assets.get("root", "../assets")}
    payload["training"] = stage2_training
    if not all(payload["paths"].get(key) for key in ("manifest", "val_manifest", "output_dir")):
        raise ValueError("stage2.data.train, stage2.data.validation, and stage2.output_dir are required")
    return payload

def resolve(value: Any, config_dir: Path) -> str:
    text = str(value)
    if text.startswith("./") or text.startswith("../"):
        return str((config_dir / text).resolve())
    return text


def required_paths(config: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Resolve the small public path surface into legacy trainer arguments.

    Users configure only ``paths.data_root``, ``paths.assets_root`` and
    ``paths.output_dir``.  The model-specific arguments are derived here so
    the train scripts remain self-contained and stable.
    """
    raw = config["paths"]
    def val(name: str, default: str | None = None) -> str:
        value = raw.get(name, default)
        if not value:
            raise ValueError(f"missing config.paths key: {name}")
        return resolve(value, config_dir)
    model = config["model"]
    assets = val("assets_root", "../assets")
    model_root = val("model_root", str(Path(assets) / "models" / model)) if raw.get("model_root") else str(Path(assets) / "models" / model)
    upstream = val("upstream_root", str(Path(assets) / "upstream" / model)) if raw.get("upstream_root") else str(Path(assets) / "upstream" / model)
    out = val("output_dir")
    result = {"manifest": val("manifest"), "val_manifest": val("val_manifest"),
              "flexae_checkpoint": val("flexae_checkpoint"), "output_dir": out,
              "flexae_root": val("flexae_root", str(ROOT / "stage1" / model)) if model in {"bagel", "lance"} else ""}
    if model == "bagel":
        result.update({"bagel_root": val("bagel_root", upstream), "bagel_model": val("bagel_model", model_root),
                       "bagel_plugins": val("bagel_plugins", str(Path(upstream) / "stage2" / "plugins")),
                       "legacy_scripts": val("legacy_scripts", str(Path(upstream) / "stage2" / "legacy"))})
    elif model == "lance":
        result.update({"lance_runtime": val("lance_runtime", upstream), "lance_model_root": val("lance_model_root", model_root)})
    else:
        result.update({"janus_code": val("janus_code", upstream), "janus_model": val("janus_model", model_root)})
    return result


def csv_value(training: dict[str, Any], key: str) -> str:
    value = training.get(key, "")
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_order(config: dict[str, Any], paths: dict[str, str], *, world_size: int) -> str:
    """Generate the paper shuffle order at launch time from the configured seed."""
    manifest_path = Path(paths["manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = sum(int(item["rows"]) for item in payload.get("files", []) if "text_only" not in str(item["path"]))
    if rows < 1:
        raise ValueError(f"manifest has no image rows: {manifest_path}")
    training = config["training"]
    seed = int(training.get("shuffle_seed", training.get("seed", 20261002)))
    micro = int(training.get("micro_batch_size", 1))
    accumulation = int(training.get("gradient_accumulation", 1))
    max_steps = int(training.get("max_steps", 5000))
    count = max_steps * world_size * micro * accumulation
    multiplier = (2 * seed + 1) % rows or 1
    while __import__("math").gcd(multiplier, rows) != 1:
        multiplier = (multiplier + 2) % rows or 1
    offset = seed % rows
    prefix = Path(paths["output_dir"]) / "generated_order"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    binary = prefix.with_suffix(".bin")
    values = array("I", ((multiplier * index + offset) % rows for index in range(count)))
    with binary.open("wb") as handle:
        values.tofile(handle)
    metadata = {
        "schema": "gafu_launch_shuffle_order_v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "rows": rows,
        "world_size": world_size,
        "micro_batch_size": micro,
        "gradient_accumulation": accumulation,
        "global_batch": world_size * micro * accumulation,
        "mix_seed": seed,
        "max_steps": max_steps,
        "resume_step": 0,
        "order_entries": count,
        "order_sha256": _sha256(binary),
        "generation": "deterministic_affine_shuffle_at_launch",
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(prefix)


def add_common(args: list[str], config: dict[str, Any], paths: dict[str, str], *, max_steps: int) -> None:
    training = config["training"]
    args.extend([
        "--manifest", paths["manifest"], "--val-manifest", paths["val_manifest"],
        "--flexae-checkpoint", paths["flexae_checkpoint"], "--output-dir", paths["output_dir"],
        "--max-steps", str(max_steps),
        "--micro-batch-size", str(training.get("micro_batch_size", 1)),
        "--gradient-accumulation", str(training.get("gradient_accumulation", 1)),
        "--real-ce-loss-weight", str(training.get("lambda_ce", 5.0)),
        "--warmup-steps", str(training.get("warmup_steps", 500)),
        "--min-lr-ratio", str(training.get("min_lr_ratio", 0.1)),
        "--max-grad-norm", str(training.get("max_grad_norm", 1.0)),
        "--max-answer-words", str(training.get("max_answer_words", 256)),
        "--max-prompt-tokens", str(training.get("max_prompt_tokens", 512)),
        "--max-microbatch-words", str(training.get("max_microbatch_words", 300)),
        "--val-limit", str(training.get("val_limit", 8)),
        "--val-batch-size", str(training.get("val_batch_size", 2)),
        "--save-steps", str(training.get("save_steps", 250)),
        "--seed", str(training.get("seed", 20261002)),
    ])
    if config["model"] != "janus_pro":
        args.extend([
            "--projector-lr", str(training.get("projector_lr", 2e-5)),
            "--sampler-lr", str(training.get("sampler_lr", 2e-5)),
            "--lora-lr", str(training.get("lora_lr", 1e-4)),
            "--lora-rank", str(training.get("lora_rank", 16)),
            "--lora-alpha", str(training.get("lora_alpha", 16)),
        ])
        if training.get("activation_checkpoint", False):
            args.append("--activation-checkpoint")


def add_schedule(args: list[str], training: dict[str, Any], *, model: str) -> None:
    if model != "janus_pro":
        args.extend(["--mask-u-prob", csv_value(training, "mask_u_schedule").split(",")[0]])
    args.extend([
        "--mask-u-schedule", csv_value(training, "mask_u_schedule"),
        "--mask-u-schedule-boundaries" if model in {"bagel", "lance"} else "--mask-u-boundaries",
        csv_value(training, "mask_u_schedule_boundaries"),
    ])
    if model == "janus_pro":
        args.extend([
            "--shuffle-kl-steps", csv_value(training, "shuffle_kl_schedule_steps"),
            "--shuffle-kl-weights", csv_value(training, "shuffle_kl_schedule_weights"),
        ])
    else:
        args.extend([
            "--shuffle-kl-weight", str(training.get("shuffle_kl_weight", 1.0)),
            "--shuffle-kl-schedule-steps", csv_value(training, "shuffle_kl_schedule_steps"),
            "--shuffle-kl-schedule-weights", csv_value(training, "shuffle_kl_schedule_weights"),
            "--mask-u-schedule-horizon", str(training.get("max_steps", 5000)),
            "--schedule-max-steps", str(training.get("max_steps", 5000)),
        ])


def dpo_args(args: list[str], training: dict[str, Any], *, model: str) -> None:
    args.extend([
        "--answer-use-loss-weight" if model != "janus_pro" else "--answer-use-weight",
        str(training.get("lambda_dpo", 1.0)),
        "--answer-use-beta" if model != "janus_pro" else "--answer-use-beta",
        str(training.get("dpo_beta", 0.1)),
        "--drop-kl-weight", str(training.get("drop_kl_weight", 0.0)),
    ])
    if model in {"bagel", "lance", "janus_pro"}:
        args.extend([
            "--dpo-real-weight", str(training.get("dpo_real_weight", 4.0)),
            "--dpo-negative-weight", str(training.get("dpo_negative_weight", 1.0)),
            "--dpo-negative-grad-cap", str(training.get("dpo_negative_grad_cap", 0.25)),
        ])


def build_commands(config: dict[str, Any], config_path: Path, mode: str, *, generated_order: str | None = None) -> list[list[str]]:
    model = str(config["model"])
    paths = required_paths(config, config_path.parent)
    training = config["training"]
    python = str(config.get("python") or sys.executable)
    max_steps = int(training.get("max_steps", 5000))
    if max_steps != 5000:
        raise ValueError("paper-main schedule is defined for max_steps=5000")
    if int(training.get("encoder_unfreeze_step", 2501)) != 2501:
        raise ValueError("paper-main Encoder unfreezes at global step 2501")
    count = len([value for value in str(config.get("gpus", "0")).split(",") if value.strip()])
    effective_batch = count * int(training.get("micro_batch_size", 1)) * int(training.get("gradient_accumulation", 1))
    if effective_batch != int(training.get("global_batch", 32)):
        raise ValueError(f"global batch mismatch: {effective_batch} != {training.get('global_batch', 32)}")
    stop = int(training.get("smoke_steps", 2)) if mode == "smoke" else max_steps
    output = paths["output_dir"]
    stage2_output = str(Path(output) / "stage2")
    order_arg = ["--balanced-order", generated_order] if generated_order else []
    if model == "janus_pro":
        command = [python, str(MODEL_SCRIPTS[model]),
                   "--janus-code", paths["janus_code"], "--janus-model", paths["janus_model"]]
        command.extend(order_arg); add_common(command, config, paths, max_steps=max_steps)
        add_schedule(command, training, model=model); dpo_args(command, training, model=model)
        command.extend([
            "--extractor-lr", str(training.get("extractor_lr", 1e-4)),
            "--projector-lr", str(training.get("projector_lr", 2e-5)),
            "--sampler-lr", str(training.get("sampler_lr", 2e-5)),
            "--lora-lr", str(training.get("lora_lr", 1e-4)),
            "--lora-rank", str(training.get("lora_rank", 16)),
            "--lora-alpha", str(training.get("lora_alpha", 16)),
            "--encoder-unfreeze-step", str(training.get("encoder_unfreeze_step", 2501)),
            "--schedule-max-steps", str(max_steps),
        ])
        if mode == "smoke": command.extend(["--stop-after-step", str(stop), "--allow-nonformal-smoke"])
        if mode == "train" or training.get("resume", False): command.append("--resume")
        return [command]

    if model == "bagel":
        base = [python, str(MODEL_SCRIPTS[model]),
                "--flexae-root", paths["flexae_root"],
                "--bagel-root", paths["bagel_root"], "--bagel-model", paths["bagel_model"],
                "--bagel-plugins", paths["bagel_plugins"], "--legacy-scripts", paths["legacy_scripts"]]
        stage2 = list(base); stage2.extend(order_arg); add_common(stage2, config, paths, max_steps=max_steps)
        stage2[stage2.index("--output-dir") + 1] = stage2_output
        add_schedule(stage2, training, model=model); dpo_args(stage2, training, model=model)
        stage2.extend([
            "--stage", "stage2", "--keep-latest", "2",
            "--extractor-lr", str(training.get("extractor_lr", 1e-4)),
            "--encoder-unfreeze-step", str(training.get("encoder_unfreeze_step", 2501)),
        ])
        if mode == "smoke":
            stage2[stage2.index("--max-steps") + 1] = str(stop)
            stage2[stage2.index("--save-steps") + 1] = str(max(1, stop))
            return [stage2]
        if mode == "train": stage2.append("--resume")
        return [stage2]

    base = [python, str(MODEL_SCRIPTS[model]),
            "--flexae-root", paths["flexae_root"],
            "--lance-runtime", paths["lance_runtime"], "--lance-model-root", paths["lance_model_root"]]
    stage2 = list(base); stage2.extend(order_arg); add_common(stage2, config, paths, max_steps=max_steps)
    stage2[stage2.index("--output-dir") + 1] = stage2_output
    add_schedule(stage2, training, model=model); dpo_args(stage2, training, model=model)
    stage2.extend([
        "--stage", "stage2", "--keep-latest", "2",
        "--extractor-lr", str(training.get("extractor_lr", 1e-4)),
        "--encoder-unfreeze-step", str(training.get("encoder_unfreeze_step", 2501)),
    ])
    if mode == "smoke":
        stage2[stage2.index("--max-steps") + 1] = str(stop)
        stage2[stage2.index("--save-steps") + 1] = str(max(1, stop))
        return [stage2]
    if mode == "train": stage2.append("--resume")
    return [stage2]


def distributed_command(command: list[str], gpus: str) -> list[str]:
    count = len([value for value in gpus.split(",") if value.strip()])
    if count < 1:
        raise ValueError("gpus must contain at least one device")
    return [
        command[0], "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", str(count), *command[1:],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("contract", "smoke", "train"), default="contract")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.mode != "contract":
        paths = required_paths(config, args.config.resolve().parent)
        placeholders = sorted(name for name, value in paths.items() if value.startswith("/path/to/"))
        if placeholders:
            raise ValueError(f"replace placeholder paths before {args.mode}: {', '.join(placeholders)}")
    gpus = str(config.get("gpus", "0"))
    paths = required_paths(config, args.config.resolve().parent)
    generated_order = None
    if args.mode != "contract":
        generated_order = generate_order(config, paths, world_size=len([value for value in gpus.split(",") if value.strip()]))
    commands = build_commands(config, args.config.resolve(), args.mode, generated_order=generated_order)
    commands = [distributed_command(command, gpus) for command in commands]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    if args.mode == "contract":
        print(json.dumps({"model": config["model"], "mode": args.mode, "commands": commands}, indent=2))
        return 0
    for command in commands:
        print("$ " + shlex.join(command), flush=True)
        if args.print_only:
            continue
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

