# Overview

这个项目的核心问题是：

```text
Can a DINOv3-style SSL encoder be made useful for pixel-space generative modeling,
without throwing away its representation strength?
```

普通 DINO encoder 学到的是强语义表征，但它不一定保留生成任务需要的局部、高频、背景和噪声条件信息。纯 generation model 又通常从头训练一个 denoising transformer，缺少 SSL representation 的先验。这个项目尝试把两者接起来：

```text
DINOv3 representation learning
  + JiT-style denoising semantics
  -> a pretrained encoder usable by a downstream generation decoder
```

## 当前假设

当前工作假设是：

```text
If the encoder is pretrained with timestep-aware denoising pressure,
then its patch tokens should become more compatible with pixel-space generation.
```

这里的关键不是让 encoder 本身成为完整 generator，而是让 encoder 的 tokens 在下游 finetune 时更容易被 decoder 使用。

## 两阶段训练

### Stage 1: Time-conditioned DINOv3 pretraining

预训练阶段保留 DINOv3-style SSL 框架，同时加入 JiT-style denoising pressure。

从代码配置看，当前 JiT pretrain 线包括：

```yaml
diffusion:
  training_style: jit
  enable_time: true
  time_scheduler: logit_normal
  P_mean: 0.8
  P_std: 0.8
  t_eps: 5.0e-2
  consistency_loss_weight: 0.5
  jit_denoising_aux_loss_weight: 0.1
```

输入图像使用 JiT pixel convention：

```yaml
crops:
  rgb_mean: [0.5, 0.5, 0.5]
  rgb_std: [0.5, 0.5, 0.5]
```

也就是把图像归一化到 `[-1, 1]`。

### Stage 2: Generation finetuning

finetune 阶段使用 pretrained encoder 初始化 generation model。模型输入是 noisy image `x_t`，输出预测 clean image `x0`，loss 使用 JiT-style velocity loss：

```text
target_v = (x0 - xt) / t
pred_v   = (pred_x0 - xt) / t
loss     = mse(pred_v, target_v)
```

这里虽然模型 head 输出的是 `x0`，但优化目标等价于 JiT 的 velocity-style denoising loss。

## 为什么不是简单 encoder-decoder

如果只是：

```text
image -> DINO encoder -> decoder
```

很容易出现一个问题：DINO representation 过强地偏向语义，局部像素和背景细节可能不足。

所以当前方案强调三件事：

1. encoder 在 pretrain 阶段就看见 timestep/noisy image。
2. finetune 阶段保留 encoder token sequence，而不是只拿一个 pooled feature。
3. decoder 使用 skip connections，从 encoder intermediate states 取更低层、更局部的特征。

## 当前结论

目前比较确定的经验是：

```text
drop_path_rate must be 0.0 during generation finetuning.
```

原因是 finetune 时 encoder 不只是一个普通 backbone。它的输出 sequence 和 skip states 是 decoder 的条件输入。Drop path 会随机扰动这些条件路径，导致 decoder 学到不稳定的 reconstruction mapping，尤其容易表现为背景噪点和高频区域崩坏。

## 项目定位

这个项目目前更像一个可扩展 framework，而不是一个已经固定的 architecture：

```text
fixed:
  pretrain with DINOv3 + JiT denoising semantics
  downstream generation uses pretrained encoder + denoising decoder

still open:
  number and placement of time/class tokens
  decoder block design
  skip vs MLS feature fusion
  optimizer and schedule
  whether auxiliary pretraining losses should be stronger or weaker
```
