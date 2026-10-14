#!/usr/bin/env python3
"""Config-driven Stage 1 launcher for the three model-specific GAFU paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
TRAINERS = {
    "bagel": ROOT / "stage1" / "bagel" / "train_stage1.py",
    "lance": ROOT / "stage1" / "lance" / "train_stage1.py",
    "janus_pro": ROOT / "stage1" / "janus_pro" / "train_stage1.py",
}


def load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("model") not in TRAINERS:
        raise ValueError(f"config.model must be one of {sorted(TRAINERS)}")
    runtime = data.get("runtime", {})
    assets = data.get("assets", {})
    stage = data.get("stage1", {})
    stage_data = stage.get("data", {}) if isinstance(stage, dict) else {}
    stage_training = stage.get("training", {}) if isinstance(stage, dict) else {}
    if not all(isinstance(item, dict) for item in (runtime, assets, stage, stage_data, stage_training)):
        raise ValueError("config.runtime, assets, stage1, stage1.data, and stage1.training must be mappings")
    data["python"] = runtime.get("python", sys.executable)
    data["gpus"] = str(runtime.get("gpus", "0"))
    data["paths"] = {"assets_root": assets.get("root", "../assets")}
    data["stage1"] = {**stage_training, "train_manifest": stage_data.get("train"), "val_manifest": stage_data.get("validation"), "output_dir": stage.get("output_dir")}
    if not all(data["stage1"].get(key) for key in ("train_manifest", "val_manifest", "output_dir")):
        raise ValueError("stage1.data.train, stage1.data.validation, and stage1.output_dir are required")
    return data

def resolve(value: Any, config_dir: Path) -> str:
    text = str(value)
    if text.startswith("./") or text.startswith("../"):
        return str((config_dir / text).resolve())
    return text


def required(section: dict[str, Any], key: str, config_dir: Path, data: dict[str, Any] | None = None) -> str:
    value = section.get(key)
    if not value and data:
        paths = data.get("paths", {})
        model = data.get("model")
        assets = resolve(paths.get("assets_root", "../assets"), config_dir)
        upstream = resolve(paths.get("upstream_root", str(Path(assets) / "upstream" / model)), config_dir)
        model_root = resolve(paths.get("model_root", str(Path(assets) / "models" / model)), config_dir)
        aliases = {"bagel_root": upstream, "bagel_model": model_root, "lance_runtime": upstream,
                   "lance_model_root": model_root, "janus_code": upstream, "janus_model": model_root,
                   "flexae_root": str(ROOT / "stage1" / model)}
        value = aliases.get(key)
    if not value:
        raise ValueError(f"missing stage1.{key}")
    return resolve(value, config_dir)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sum(int(item["rows"]) for item in payload.get("files", [])
               if "text_only" not in str(item.get("path", "")))
    if rows < 1:
        raise ValueError(f"manifest has no embedded-image rows: {path}")
    return rows


def generate_unique_order(
    section: dict[str, Any],
    config_dir: Path,
    data: dict[str, Any],
    *,
    world_size: int,
    max_steps: int | None = None,
) -> str:
    """Create the Stage 1 image order in the launch output directory."""
    manifest = Path(required(section, "train_manifest", config_dir, data))
    rows = manifest_rows(manifest)
    micro = int(section.get("micro_batch_size", 1))
    accumulation = int(section.get("gradient_accumulation", 1))
    max_steps = int(section.get("max_steps", 1) if max_steps is None else max_steps)
    seed = int(section.get("shuffle_seed", section.get("seed", 20261002)))
    count = max_steps * world_size * micro * accumulation
    multiplier = (2 * seed + 1) % rows or 1
    while math.gcd(multiplier, rows) != 1:
        multiplier = (multiplier + 2) % rows or 1
    offset = seed % rows
    prefix = Path(required(section, "output_dir", config_dir, data)) / "generated_stage1_order"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    binary = prefix.with_suffix(".bin")
    values = array("I", ((multiplier * index + offset) % rows for index in range(count)))
    with binary.open("wb") as handle:
        values.tofile(handle)
    metadata = {
        "schema": "finevision_unique_image_order_v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "unique_images": rows,
        "rows": rows,
        "world_size": world_size,
        "micro_batch_size": micro,
        "gradient_accumulation": accumulation,
        "seed": seed,
        "max_steps": max_steps,
        "order_entries": count,
        "order_sha256": sha256(binary),
        "generation": "deterministic_affine_shuffle_at_launch",
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(prefix)


def common_stage1_args(section: dict[str, Any], config_dir: Path, data: dict[str, Any], *, bagel: bool = False) -> list[str]:
    args = [
        "--manifest", required(section, "train_manifest", config_dir, data),
        "--val-manifest", required(section, "val_manifest", config_dir, data),
        "--output-dir", required(section, "output_dir", config_dir, data),
        "--micro-batch-size", str(section.get("micro_batch_size", 1)),
        "--learning-rate", str(section.get("learning_rate", 1e-4)),
        "--warmup-steps", str(section.get("warmup_steps", 200)),
        "--weight-decay", str(section.get("weight_decay", 0.01)),
        "--max-grad-norm", str(section.get("max_grad_norm", 1.0)),
        "--native-loss-weight", str(section.get("native_loss_weight", 0.25)),
        "--binding-loss-weight", str(section.get("binding_loss_weight", 1.0)),
        "--global-relation-weight", str(section.get("global_relation_weight", 0.137)),
        "--local-relation-weight", str(section.get("local_relation_weight", 1.34)),
        "--relation-ramp-steps", str(section.get("relation_ramp_steps", 100)),
        "--relation-initial-scale", str(section.get("relation_initial_scale", 0.2)),
        "--save-steps", str(section.get("save_steps", 250)),
        "--logging-steps", str(section.get("logging_steps", 10)),
        "--val-limit", str(section.get("val_limit", 64)),
        "--seed", str(section.get("seed", 20261002)),
    ]
    if bagel:
        args.extend(["--min-learning-rate", str(section.get("min_learning_rate", 1e-5)),
                     "--eval-steps", str(section.get("eval_steps", section.get("save_steps", 250)))])
    else:
        args.append("--min-lr-ratio")
        args.append(str(section.get("min_lr_ratio", 0.1)))
    return args


def command(config: Path, data: dict[str, Any], mode: str, order: str | None) -> list[str]:
    model = str(data["model"])
    section = data["stage1"]
    config_dir = config.parent
    python = str(data.get("python") or sys.executable)
    gpus = str(data.get("gpus", "0"))
    world_size = len([item for item in gpus.split(",") if item.strip()])
    smoke_steps = int(section.get("smoke_steps", 2))
    max_steps = smoke_steps if mode == "smoke" else int(section.get("max_steps", 8000))
    args = [python, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(world_size), str(TRAINERS[model])]

    if model == "bagel":
        args.extend([
            "--flexae-root", required(section, "flexae_root", config_dir, data),
            "--bagel-root", required(section, "bagel_root", config_dir, data),
            "--bagel-model", required(section, "bagel_model", config_dir, data),
            "--max-steps", str(max_steps),
        ])
        args.extend(common_stage1_args(section, config_dir, data, bagel=True))
        args.extend(["--local-relation-scales", "4,8,16"])
    else:
        args.extend(["--unique-order", order or str(Path(required(section, "output_dir", config_dir, data)) / "generated_stage1_order")])
        if model == "lance":
            args.extend(["--lance-runtime", required(section, "lance_runtime", config_dir, data),
                         "--lance-model-root", required(section, "lance_model_root", config_dir, data)])
        else:
            args.extend(["--janus-code", required(section, "janus_code", config_dir, data),
                         "--janus-model", required(section, "janus_model", config_dir, data),
                         "--gradient-accumulation", str(section.get("gradient_accumulation", 1)),
                         "--max-steps", str(max_steps)])
        args.extend(common_stage1_args(section, config_dir, data))
        if model == "janus_pro":
            if mode == "smoke":
                args.extend(["--stop-after-step", str(smoke_steps), "--allow-nonformal-smoke"])
            if section.get("keep_latest_checkpoints") is not None:
                args.extend(["--keep-latest-checkpoints", str(section["keep_latest_checkpoints"])])
    if section.get("resume", mode == "train"):
        args.append("--resume")
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("contract", "smoke", "train"), default="contract")
    parser.add_argument("--print-only", action="store_true")
    cli = parser.parse_args()
    config = cli.config.resolve()
    data = load(config)
    section = data["stage1"]
    gpus = str(data.get("gpus", "0"))
    world_size = len([item for item in gpus.split(",") if item.strip()])
    order = None
    if data["model"] != "bagel" and cli.mode != "contract" and not cli.print_only:
        order = generate_unique_order(
            section, config.parent, data, world_size=world_size,
            max_steps=int(section.get("smoke_steps", 2)) if cli.mode == "smoke" else None,
        )
    cmd = command(config, data, cli.mode, order)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    print("$ " + shlex.join(cmd), flush=True)
    if cli.mode != "contract" and not cli.print_only:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

