# GAFU: Improving Fine-Grained Understanding in Unified Multimodal Models

<p align="center">
  <strong>A post-training framework for using generation-side visual representations to improve fine-grained visual understanding.</strong>
</p>

<!-- Replace PAPER_URL with the public paper link before release. -->
<p align="center">
  <a href="PAPER_URL">📄 Paper</a>
</p>

<p align="center">
  <img src="assets/framework.png" alt="GAFU framework overview" width="95%">
</p>

## News

- Code and training configurations are released.

## Table of Contents

- [GAFU: Improving Fine-Grained Understanding in Unified Multimodal Models](#gafu-improving-fine-grained-understanding-in-unified-multimodal-models)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
  - [Data](#data)
  - [Train](#train)
      - [Stage 1 — Compression and Alignment](#stage-1--compression-and-alignment)
      - [Stage 2 — Task-Conditioned Injection and Fine-Tuning](#stage-2--task-conditioned-injection-and-fine-tuning)
  - [Results](#results)
  - [Citation](#citation)

## Repository Structure

```text
configs/                 Backbone-specific YAML configurations
requirements/            Environment dependencies by backbone
scripts/                 Installation and model download scripts
src/                     Self-contained GAFU code
requirements.md          Dependency and upstream setup notes
```

## Installation

```bash
# Clone the repository
git clone --recursive https://github.com/LRuca/GAFU.git
cd GAFU

# Install the shared environment and upstream source trees
bash scripts/install.sh

# Download the backbone weights needed for your experiment
bash scripts/download_models.sh bagel   # or: lance, janus, all
```


> **Note:** Each backbone uses its own environment because the upstream projects have different runtime and CUDA dependencies. Install only the environment and model(s) required for your experiment. See [`requirements.md`](requirements.md) for details.

## Data

The paper uses Cambrian, FineVision, VSR-Bench, and PathVQA. The training scripts read these datasets in their official formats; the datasets are not included in this repository.

For Compression and Alignment, provide image records containing an image path or image payload.

<!-- Compression and Alignment 阶段需要包含图像路径或图像内容的图像记录。 -->

```json
{
  "image_path": "/data/images/example.jpg"
}
```

For Task-Conditioned Injection and Fine-Tuning, provide image-question-answer records.

<!-- Task-Conditioned Injection and Fine-Tuning 阶段需要图像问答数据。 -->

```json
{
  "image": "/data/images/example.jpg",
  "question": "Question",
  "answer": "Answer"
}
```

Set the dataset paths in the corresponding YAML configuration.

## Train

GAFU supports BAGEL, Lance, and Janus-Pro. Select the corresponding configuration under `configs/`.



#### Stage 1 — Compression and Alignment

```bash
conda run -n gafu-bagel \
  python src/train_stage1.py \
  --config configs/bagel.yaml \
  --mode train
```

#### Stage 2 — Task-Conditioned Injection and Fine-Tuning

Stage 2 automatically resolves the Stage 1 checkpoint from `stage1.output_dir` and `stage1.training.max_steps` in the same configuration.

```bash
conda run -n gafu-bagel \
  python src/train_stage2.py \
  --config configs/bagel.yaml \
  --mode train
```

Replace `gafu-bagel` and `configs/bagel.yaml` with the corresponding Lance or Janus-Pro environment and configuration.

## Results

<!-- GAFU 在相同下游数据训练的 SFT 基线上取得稳定提升。 -->
GAFU consistently outperforms SFT baselines trained on the same downstream data.

| Backbone | Method | VStar-Bench | OmniSpatial | MMVP | PathVQA | VSR-Bench | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|
| BAGEL | SFT | 39.27 | 31.90 | 74.00 | 51.42 | 71.39 | 53.60 |
| BAGEL | **GAFU** | **47.12** | **34.51** | **79.67** | **59.07** | **77.54** | **59.58** |
| Janus-Pro | SFT | 33.51 | 27.14 | 54.67 | 51.75 | 57.54 | 44.92 |
| Janus-Pro | **GAFU** | **38.74** | **31.57** | **59.67** | **52.87** | **59.54** | **48.48** |
| Lance | SFT | 36.13 | 24.72 | 50.00 | 43.04 | 53.26 | 41.43 |
| Lance | **GAFU** | **42.41** | **29.16** | **53.67** | **46.91** | **54.35** | **45.30** |

## Citation

```bibtex
@inproceedings{gafu2027,
  title     = {What Generation Knows: Unlocking Fine-Grained Understanding in Unified Multimodal Models},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```
