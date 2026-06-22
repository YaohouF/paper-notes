# Reproducibility Gaps

FD-loss 的代码开源程度相对不错：训练、评估、reference stats、scripts 都有。但完整复现仍然有一些实际门槛。

## 已经公开且清楚的部分

| Part | Status |
|---|---|
| FD-loss 数学实现 | 代码清楚 |
| queue / EMA estimator | 代码清楚 |
| 多 representation training | 代码清楚 |
| ImageNet evaluation scripts | 代码清楚 |
| 表格复现实验脚本 | 有 shell scripts |
| released checkpoints | Hugging Face 提供 |
| reference stats | Hugging Face 提供，也能本地重算 |

## 缺口 1：算力门槛高

论文主实验使用：

```text
global batch size = 1024
bf16
50 / 100 epochs
multiple frozen representation models
50k evaluation samples
```

这不是单卡笔记本能完整复现的实验。即使代码完整，复现主表仍需要多 GPU。

## 缺口 2：ImageNet 访问

需要本地 ImageNet：

```text
DATA_ROOT=/path/to/imagenet
```

如果没有 ImageNet train/val，就只能评估 released stats/checkpoints，不能重新计算 reference statistics。

## 缺口 3：依赖多个大 representation models

FD-SIM 和 FDr6 依赖：

- Inception-v3
- SigLIP2
- MAE
- DINOv2
- ConvNeXt-v2
- CLIP

这些模型会通过 timm / torch-fidelity 加载。实际复现时，需要考虑下载、缓存、显存和 preprocessing 是否完全一致。

## 缺口 4：base generator checkpoints

训练从 released base checkpoints 开始。虽然 repo/Hugging Face 提供下载方式，但如果想换成自己的 generator，需要实现：

```python
sample_images_with_grad(z, y, sampling_args)
generate(...)
in_channels
input_size
```

也就是说，FD-loss 是通用思想，但代码接口仍围绕他们集成的 pMF/iMF/JiT。

## 缺口 5：metric optimization 风险

论文自己也承认 Goodhart 风险：一旦把 automatic metric 作为训练目标，模型可能 exploit representation blind spots。

他们在 appendix 中展示过 Inception Score / FID 被过度优化后的 artifact：

```text
IS very high, FID low,
but samples contain artifacts and FDr6 degrades.
```

所以 FD-loss 不是“最终完美目标”，而是一种强但需要谨慎使用的 distributional objective。

## 缺口 6：T2I 扩展不如 ImageNet 主线完整

论文展示了 SD3.5 Medium 的 T2I one-step repurposing，但公开代码主结构和 scripts 主要围绕 ImageNet class-conditioned experiments。T2I 部分更像 method scalability demo。

## 如果我们以后想小规模复现

建议不要一开始追主表。可以按这个顺序：

```text
1. 先只用 released checkpoint 做 evaluation smoke test
2. 用 NUM_IMAGES=1024 跑小规模 FID/FDr pipeline
3. 只用 Inception FD-loss 跑一个很短 post-training
4. 再换 EMA beta / queue size
5. 最后加入 SigLIP2 + MAE 做 FD-SIM
```

最小概念验证不需要完整 ImageNet 100 epochs；先验证 loss 能下降、samples 不崩、queue/EMA 正常即可。

