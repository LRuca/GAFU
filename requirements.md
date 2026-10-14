# GAFU 环境要求

本文档描述 GAFU launcher 和训练脚本的 Python 依赖。BAGEL、Lance、Janus-Pro 的模型代码和权重由各自上游项目提供，不随本目录分发。

这里的“不同环境”指三个上游模型 runtime：BAGEL、Lance、Janus-Pro 的 Stage 1 和 Stage 2 都加载不同的模型代码、视觉组件、latent/VAE 组件和 checkpoint。它们的基础 PyTorch 版本可以相同，但上游 Python 包、CUDA 扩展和导入路径不一定相同。

## 推荐环境

- Linux，Python 3.10
- NVIDIA GPU；论文配置使用 8 x A100 40GB，通过 `torchrun` 启动
- CUDA 12.4 对应的 PyTorch wheel，或与本机驱动兼容的 PyTorch CUDA 版本
- 磁盘空间足够保存上游模型权重、Stage 1 checkpoint 和 FineVision Parquet shards

本次 A100 验证使用的核心版本为：

```text
torch       2.5.1+cu124
torchvision 0.20.1+cu124
PyYAML      6.0.2
Pillow      11.3.0
safetensors 0.4.5
pyarrow     11.0.0
numpy       1.24.4
einops      0.8.1
transformers 4.49.0  # Janus-Pro backend
```

## 共享依赖

在与 CUDA 12.4 兼容的 Python 3.10 环境中安装：

```bash
python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  PyYAML==6.0.2 Pillow==11.3.0 safetensors==0.4.5 \
  pyarrow==11.0.0 numpy==1.24.4 einops==0.8.1
```

Janus-Pro 还需要：

```bash
python -m pip install transformers==4.49.0
```

如果使用不同 CUDA/PyTorch 版本，应保持 `torch` 和 `torchvision` 的版本、CUDA 后缀匹配；不要把多个模型上游环境的依赖混装后再复用同一个环境。

## 推荐环境拆分

建议使用下面的环境布局：

| 环境 | 用途 | 运行入口 |
| --- | --- | --- |
| `gafu-bagel` | BAGEL Stage 1 和 Stage 2 | `configs/bagel.yaml` |
| `gafu-lance` | Lance Stage 1 和 Stage 2 | `configs/lance.yaml` |
| `gafu-janus` | Janus-Pro Stage 1 和 Stage 2 | `configs/janus_pro.yaml` |

三个环境可以安装同一组共享依赖，但分别安装对应上游项目的 requirements。只运行某个模型时，安装对应环境即可；运行 Lance 或 Janus-Pro 时不需要把它们的 runtime 安装进 BAGEL 环境。

## 上游运行时和权重

每个模型还需要把上游代码路径填入对应 YAML：

- BAGEL：Stage 1 需要 `stage1.bagel_root`、`stage1.bagel_model`；Stage 2 需要 `paths.bagel_root`、`paths.bagel_model`、`paths.bagel_plugins`、`paths.legacy_scripts`。
- Lance：Stage 1 需要 `stage1.lance_runtime`、`stage1.lance_model_root`；Stage 2 需要 `paths.lance_runtime`、`paths.lance_model_root`。模型目录还必须包含上游 Lance 需要的 `Wan2.2_VAE.pth`。
- Janus-Pro：Stage 1 和 Stage 2 分别需要 `stage1.janus_code`/`stage1.janus_model` 与 `paths.janus_code`/`paths.janus_model`，模型目录应是可被上游 `transformers`/Janus loader 读取的 checkpoint。

这里的 runtime 是上游模型的代码目录，权重是模型 checkpoint；它们不是通过 `pip install` 安装的普通依赖。三个上游项目和权重可能有各自的许可证、下载方式和访问限制，因此本仓库不自动下载、不重新打包，也不提交模型权重。用户只需把本地路径填入对应 YAML。

## 快速检查

```bash
python -c "import torch, yaml, pyarrow, PIL, safetensors; print(torch.__version__, torch.cuda.is_available())"
python src/train_stage1.py --config configs/bagel.yaml --mode contract
python src/train_stage2.py --config configs/bagel.yaml --mode contract
```

`contract` 只检查配置并打印最终命令，不加载模型，不需要 GPU 显存；路径仍需在实际 `smoke` 或 `train` 前替换为有效路径。


