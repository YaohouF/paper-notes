# Overview

FD-loss 这篇论文很漂亮的一点是，它抓住了图像生成里一个长期有点“奇怪”的现象：

```text
FID 是大家最常用的评估指标，
但训练时几乎没人直接优化 FID。
```

原因不是大家没想到，而是直接优化 FID 很麻烦：FID 是一个分布级指标，不是单样本 loss。它需要足够大的生成样本集合来估计均值和协方差。

这篇论文的核心贡献就是让这件事变得 practical。

## 论文问题

标准 FID 做法是：

```text
real images -> Inception features -> real mean/cov
generated 50k images -> Inception features -> generated mean/cov
compare two Gaussians with Fréchet Distance
```

公式是：

```math
\mathrm{FD}_{\phi}(\mathcal{R}, \mathcal{G}) =
\|\mu_r - \mu_g\|_2^2
+ \mathrm{Tr}\left(
\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}
\right)
```

当 `phi` 是 Inception-v3 时，就是 FID。

问题在于训练时：

```text
evaluation wants 50k samples
training batch may be 64 / 256 / 1024
```

如果只用当前 batch 估计 FD，会很不稳定；如果每步都生成 50k 图像再反传，又不现实。

## 核心想法：decouple population and optimization

论文的关键想法是：

```text
FD statistics population size N
  can be large, e.g. 50k / 100k

gradient batch size B
  can remain normal, e.g. 1024
```

也就是：

```text
用大窗口估计生成分布统计量，
但只让当前 batch 带梯度。
```

这正是论文标题和图里强调的 decoupling。

## 两个 estimator

论文实现了两个版本。

### Queue-based estimator

维护一个 feature queue：

```text
queue size N = 50k / 100k
current batch B = 1024

each step:
  generate B images
  extract B features
  replace oldest B features in queue
  compute FD over queue
  backprop only through current B features
```

旧 features 被当成 constants，不带梯度。

### EMA-based estimator

不存整个 queue，只维护一阶和二阶矩：

```math
\mu_g^{(t)} = \beta \mu_g^{(t-1)} + (1-\beta)\mu_\text{batch}^{(t)}
```

```math
M_g^{(t)} = \beta M_g^{(t-1)}
+ (1-\beta)M_\text{batch}^{(t)}
```

再恢复 covariance：

```math
\Sigma_g^{(t)} = M_g^{(t)} - \mu_g^{(t)}\mu_g^{(t)\top}
```

论文最后默认使用 EMA，`beta=0.999`。

## 论文的三个主要发现

### 1. FD-loss 可以有效 post-train one-step generators

在 ImageNet 256×256 上，FD-loss 可以让已有 one-step generators 的 FID 和 FDr 变好。

比如 pMF-B/16 ablation 中：

```text
Base FID: 3.31
FD-Inception + EMA beta=0.999: FID 0.81
FD-SIM: FDr6 4.20
```

### 2. 多 representation 比单一 Inception 更可靠

单独优化 Inception 可以把 FID 压得很低，但未必视觉最好。论文指出 FID 已经有 saturation / metric hacking 风险。

所以他们提出多 representation 的 normalized FD ratio：

```text
FDr^6 = average normalized FD ratio over six feature spaces
```

六个 feature spaces 是：

- Inception-v3
- ConvNeXt-v2
- DINOv2
- MAE
- SigLIP2
- CLIP

训练默认强配置是 FD-SIM：

```text
FD-SIM = SigLIP2 + Inception + MAE
```

### 3. 它可以 repurpose multi-step generators into one-step generators

这点很有意思。对于原本 multi-step 的 JiT，他们直接把它当作 one-step generator：

```text
z at terminal timestep
  -> run model once
  -> interpret output as x0
```

一开始生成很差，但用 FD-loss post-train 后，能变成不错的 1-NFE generator。

论文强调这不需要：

```text
teacher distillation
adversarial training
per-sample regression target
```

## 这篇论文的定位

如果用一句话判断它的贡献类型：

```text
FD-loss is a post-training distribution-matching objective,
not a new image generator architecture.
```

所以读它时，不要主要问“它的 Transformer block 长什么样”；更应该问：

```text
How can a population metric become a stable differentiable loss?
Which representation space should define visual quality?
How much can we trust FID once it becomes optimized directly?
```

