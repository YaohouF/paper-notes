# Data and Task Setup

FD-loss 的主要实验是 ImageNet class-conditioned generation。它不像多模态 T2I 论文那样复杂构造 prompt 数据；这里的数据重点是：

```text
real-image reference statistics
class labels
generated sample distribution
```

## 1. 主任务：ImageNet class-to-image

论文设置：

- Dataset: ImageNet-1k
- Resolution: 256×256 and 512×512
- Conditioning: class label
- Evaluation: 50,000 generated images
- Sampling protocol: 1000 classes uniformly sampled, 50 images per class

也就是说生成任务是：

```text
class id y + random noise z -> generated image x_hat
```

不是 text-to-image prompt，也不是图像编辑。

## 2. 真实图像 statistics

FD-loss 需要预先计算真实图像在不同 representation spaces 下的：

```text
mu_ref
sigma_ref
```

代码里用 ImageNet train set 计算 reference stats。脚本是：

```text
compute_repr_stats.py
scripts/compute_ref_stats.sh
```

每个 representation model 都有自己的 stats 文件，例如：

```text
guided_diffusion_stats.npz
convnext_in256_t224_stats.npz
vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz
vit_large_patch16_224_mae_in256_t224_stats.npz
vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz
```

这些 stats 是 FD-loss 训练的“目标分布”。

## 3. Representation models

论文的 FDr6 使用六个 frozen feature extractors：

| Representation | Architecture | Objective | Feature dim |
|---|---|---|---:|
| Inception-v3 | CNN | supervised | 2048 |
| ConvNeXt-v2 | CNN | self-supervised / fine-tuned | 1024 |
| DINOv2 | ViT | self-supervised / contrastive | 1024 |
| MAE | ViT | reconstructive | 1024 |
| SigLIP2 | ViT | vision-language | 1152 |
| CLIP | ViT | vision-language | 1024 |

训练默认的 FD-SIM 不是全部六个，而是：

```text
FD-SIM = SigLIP2 + Inception-v3 + MAE
```

CLIP 在主要 ablation 表里作为 held-out evaluator，不作为训练信号。

## 4. Base generators

论文测试了多个 ImageNet generator families：

| Family | Space | Nature |
|---|---|---|
| pMF | pixel-space | one-step generator |
| iMF | latent-space | one-step generator |
| JiT | pixel-space | originally multi-step, repurposed to one-step |

所有模型都从官方 released pretrained weights 初始化，FD-loss 只做 post-training。

这点很关键：FD-loss 本身不从零训练一个生成器。

## 5. Multi-step generator repurposing

对于 multi-step velocity model，论文用 terminal timestep 的一次前向：

```math
\hat{x}_0 = z - v_{\theta}(z, t=1)
```

对于 `x0` prediction model：

```math
\hat{x}_0 = x_{\theta}(z, t=1)
```

然后把这个“本来很差的一步生成器”用 FD-loss post-train。

这件事的含义是：FD-loss 不需要已有 one-step teacher。它可以直接把一个 multi-step denoiser 压成 one-step 行为。

## 6. Text-to-image 扩展不是主线

论文最后还把 SD3.5 Medium repurpose 成 1-NFE text-to-image generator。这说明方法可以超出 class-to-image。

但对我们的站点分类来说，这篇仍放在 class-to-image，因为：

```text
main experiments = ImageNet class-conditioned generation
main tables = ImageNet 256/512 class-to-image
```

T2I 部分更像额外展示 scalability。

## 7. 数据在这篇论文里的角色

这篇论文的数据重点不是“构造更好的训练集”，而是“构造更好的分布目标”：

```text
real ImageNet train images
  -> frozen representation features
  -> reference mean/cov
```

训练时并不需要真实图像逐样本参与 loss；它需要的是真实分布的 feature statistics。

