# Pretraining Recipe

预训练阶段的目标不是直接训练一个完整 generator，而是让 DINOv3-style encoder 学到更适合 generation finetune 的 noisy-image representation。

## Backbone

当前主线使用 DINOv3 ViT-B/16：

```yaml
student:
  arch: vit_base
  patch_size: 16
  n_storage_tokens: 4
  norm_layer: layernorm
```

输入图像在 patch embedding 后变成 patch tokens。对于 224 pretraining crop 和 patch size 16：

```text
224 x 224 image
  -> 14 x 14 patch grid
  -> 196 patch tokens
```

DINOv3 encoder 使用 RoPE 处理 patch token 的位置关系。conditioning tokens 被 prepend 到 sequence 前面，RoPE 只作用于 patch tokens。

## Time Conditioning

encoder 被扩展为 timestep-aware backbone。其内部 sequence layout 是：

```text
[time token, CLS token, storage tokens, patch tokens]
```

在 class conditioning 未启用的 pretraining 路径中，class token 不进入 encoder。time token 来自：

```text
t -> frequency embedding -> MLP -> time token
```

代码中对应 `TimestepEmbedder`。

## JiT-style Noising

当前 JiT pretraining config 使用 logit-normal timestep：

```yaml
diffusion:
  training_style: jit
  time_scheduler: logit_normal
  P_mean: 0.8
  P_std: 0.8
  t_eps: 5.0e-2
  noise_scale: 1.0
```

语义上：

```text
t = sigmoid(N(P_mean, P_std))
xt = (1 - t) * x0 + t * eps
```

这里 `t=0` 表示 clean image，`t=1` 表示 pure noise。这个方向和一些 EDM sigma notation 不同，所以看代码时要特别注意。

## SSL + Denoising

当前 pretraining 路线保留 DINOv3 SSL 训练，同时加入 JiT-style denoising auxiliary：

```yaml
diffusion:
  consistency_loss_weight: 0.5
  jit_denoising_aux_loss_weight: 0.1
```

这条 auxiliary loss 的目的不是让 encoder 直接具备完整生成能力，而是给 patch tokens 加入更强的 local reconstruction / denoising pressure。

## 图像归一化

当前 JiT 路线使用 JiT pixel convention：

```yaml
crops:
  rgb_mean: [0.5, 0.5, 0.5]
  rgb_std: [0.5, 0.5, 0.5]
```

这意味着图像被映射到 `[-1, 1]`。这和标准 ImageNet mean/std 不同。对 generation finetune 来说，pretrain 和 finetune 的 pixel convention 保持一致更重要。

## 当前可扩展点

预训练阶段还没有完全定型，当前值得继续 ablate 的变量包括：

| Component | Current direction | Open question |
|---|---|---|
| Time sampling | logit-normal JiT time | 是否需要更靠近 original JiT 或更多覆盖 low-noise region |
| Denoising aux weight | small weight, e.g. 0.1 | 太强会不会损害 DINO representation |
| Consistency loss | active with reduced weight | 和 denoising aux 的权重如何平衡 |
| Target granularity | patch-level denoising auxiliary | 是否需要更强 decoder-style head |
| Crop usage | global crop first | local crops 是否适合做 denoising auxiliary |

## 预训练阶段的保守原则

当前比较合理的原则是：

```text
The denoising auxiliary should guide representation,
not replace DINO-style representation learning.
```

所以 auxiliary loss weight 不宜过大。它更像是把 encoder 的 patch tokens 往 generation-friendly 方向拉一点，而不是把 encoder 改造成一个完整 image generator。
