# Multimodal Paper Notes

这是一个持续维护的多模态论文读书站点。目标不是简单堆 Markdown，而是把每篇论文整理成一个可以复习、横向比较、持续扩展的研究知识库。

## 当前收录

| Paper | Topic | Status |
|---|---|---|
| [HiDream-O1-Image](papers/hidream-o1-image/index.md) | Pixel-level Unified Transformer, Qwen3-VL initialized raw-pixel generation | 第一版完整笔记 |
| [SenseNova-U1](papers/sensenova-u1/index.md) | Native unified understanding + generation, NEO-unify, MoT | 第一版完整笔记 |
| [TUNA-2](papers/tuna-2/index.md) | Pixel embeddings, VAE-free / encoder-free unified model | 已迁入 |
| [FD-loss](papers/class-to-image/fd-loss/index.md) | Class-to-image post-training, representation Fréchet loss | 第一版完整笔记 |
| [DINOv3 + JiT Generation Project](projects/dinov3-jit-generation/index.md) | Time-conditioned DINOv3 encoder + JiT denoising + pixel-space generation finetune | 项目方法快照 |

## 推荐阅读方式

如果是第一次读一篇论文，建议按这个顺序：

```text
Overview
  -> Model Architecture
  -> Data Construction
  -> Training Recipe
  -> Paper-Code Crosscheck
  -> Reproducibility Gaps
```

如果是在面试或复习前快速回顾：

```text
Paper index
  -> TL;DR
  -> Key Architecture Diagram
  -> Training Timeline
  -> FAQ / Crosscheck
```

## 知识库结构

```text
papers/
  multimodal / text-to-image 论文
  class-to-image 论文

concepts/
  跨论文复用的概念，比如 MoT、Native RoPE、Flow Matching

comparisons/
  论文之间的横向对比
```

## 当前主线：Native Unified Multimodal Models

```mermaid
flowchart LR
  A["Separate systems"] --> B["Unified tokenizer / latent space"]
  B --> C["Native pixel-word modeling"]
  C --> D["Unified understanding + generation"]

  A1["VE for understanding"] --> A
  A2["VAE / diffusion for generation"] --> A

  C1["TUNA-2"] --> C
  C2["SenseNova-U1"] --> C
  C3["HiDream-O1-Image"] --> C
```

## 新增主线：Class-to-Image Generation

```mermaid
flowchart LR
  A["Pretrained class-conditioned generator"] --> B["Post-training objective"]
  B --> C["Representation FD-loss"]
  C --> D["Improved one-step ImageNet generator"]

  C1["FID / FD as metric"] --> C
  C2["Queue or EMA population stats"] --> C
  C3["Frozen representation models"] --> C
```

## 进行中项目：DINOv3 + JiT Generation

```mermaid
flowchart LR
  A["DINOv3 SSL encoder"] --> B["time-conditioned pretraining"]
  B --> C["JiT denoising auxiliary"]
  C --> D["generation-friendly encoder tokens"]
  D --> E["skip-connected denoising decoder"]
  E --> F["ImageNet class-to-image generation"]

  T["time tokens"] --> D
  Y["class tokens / CLS+y"] --> D
  S["storage tokens removed"] --> E
```

这条线不是论文对比，而是当前自己的研究项目记录。重点是维护当前方法语义、代码实现、已验证经验和后续可扩展方向。

## 后续维护模板

每加入一篇新论文，建议新增：

```text
docs/papers/<paper-name>/
  index.md
  00_overview.md
  01_model_architecture.md
  02_data_construction.md
  03_training_recipe.md
  04_paper_code_crosscheck.md
  05_reproducibility_gaps.md
```

如果这篇论文带来重要新概念，再同步加入：

```text
docs/concepts/<concept-name>.md
```

如果它和已有论文强相关，再加入：

```text
docs/comparisons/<paper-a>-vs-<paper-b>.md
```
