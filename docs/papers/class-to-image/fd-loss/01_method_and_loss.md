# Method and Loss

FD-loss 的方法部分可以拆成四层：

1. Fréchet Distance 本身是什么。
2. 为什么直接用 batch FD 不稳定。
3. queue / EMA 如何把大 population statistics 和当前 batch gradient 解耦。
4. 多 representation loss 如何组合。

## 1. Fréchet Distance

给定一个 frozen feature extractor：

```text
phi(image) -> feature vector
```

真实图像和生成图像在 feature space 中分别被近似成高斯分布：

```math
\mu_r = \mathbb{E}[\phi(x)],\quad
\Sigma_r = \mathrm{Cov}[\phi(x)]
```

```math
\mu_g = \mathbb{E}[\phi(\hat{x})],\quad
\Sigma_g = \mathrm{Cov}[\phi(\hat{x})]
```

FD 定义为：

```math
\mathrm{FD}_{\phi}(\mathcal{R}, \mathcal{G}) =
\|\mu_r - \mu_g\|_2^2
+ \mathrm{Tr}\left(
\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}
\right)
```

其中：

- mean term 对齐 feature mean。
- covariance term 对齐 feature covariance。
- `phi = Inception-v3` 时就是 FID。

## 2. 为什么 FD 不像普通 loss

普通训练 loss 通常是 sample-wise：

```text
each generated image has its own loss
```

但 FD 是 distribution-wise：

```text
a batch / population of generated images together has one scalar FD
```

这带来两个难点：

1. 当前 batch 太小，协方差估计不稳定。
2. 大 population 估计可靠，但不能每一步对 50k 图像反传。

所以论文提出：

```text
estimate FD over a large effective population,
but backpropagate only through current batch.
```

## 3. Queue estimator

设：

```text
N = queue size, e.g. 50k
B = current batch size, e.g. 1024
```

每步训练：

```text
1. generator samples B images
2. frozen phi extracts B features
3. enqueue current B features
4. evict oldest B features
5. compute FD over N features
6. only current B features carry gradients
```

直观地说，queue 里的旧 features 提供稳定的 population statistics；当前 features 决定这一步模型怎么改。

代码里对应：

```python
all_feats = judge["queue"].build_feats_snapshot(new_feats)
fid = compute_frechet_distance_loss(..., all_feats=all_feats)
```

`build_feats_snapshot` 会 clone/detach 旧 queue，然后把当前 pointer 区域替换成带梯度的 `new_feats`。

## 4. EMA estimator

EMA 不存完整 feature queue，而是维护：

```text
mu_ema: first moment
m2_ema: second raw moment E[xx^T]
```

训练时：

```math
\mu_g = \beta\mu_\text{ema} + (1-\beta)\mu_\text{batch}
```

```math
M_g = \beta M_\text{ema} + (1-\beta)M_\text{batch}
```

```math
\Sigma_g = M_g - \mu_g\mu_g^\top
```

代码里对应：

```python
mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B
sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
```

注意 `.detach()`：历史统计量不反传，当前 batch 反传。

论文默认：

```text
EMA beta = 0.999
```

## 5. 多 representation FD-loss

不同 representation 的 FD 数值尺度差别很大。例如 MAE 和 Inception 的 raw FD 可能不是一个量级。

论文用 normalized loss：

```math
\mathcal{L}_{\phi_i}
=
\frac{\mathrm{FD}_{\phi_i}(\mathcal{R}, \mathcal{G})}
{\mathrm{sg}(\mathrm{FD}_{\phi_i}(\mathcal{R}, \mathcal{G})) + c}
```

总 loss：

```math
\mathcal{L}
=
\sum_i w_i \mathcal{L}_{\phi_i}
```

其中：

- `sg` 是 stop-gradient。
- `c = 0.01`，用于数值稳定。
- 论文中 `w_i = 1`。

这个归一化很巧：forward value 近似单位尺度，但 gradient 仍来自 FD。

代码里对应：

```python
fid_loss = fid / (fid.detach() + fid_norm_eps)
loss = loss + judge["weight"] * fid_loss
```

这就是论文公式的直接实现。

## 6. FDr：为什么他们又提出新指标

论文指出单一 FID 可能已经饱和：一些生成模型在 Inception FID 上甚至“超过”真实 validation images，但人看起来仍然不是完全真实。

因此他们提出 normalized FD ratio：

```math
\mathrm{FDr}_{\phi_i}(\mathcal{G})
=
\frac{
\mathrm{FD}_{\phi_i}(\mathcal{G}, \mathcal{T})
}{
\mathrm{FD}_{\phi_i}(\mathcal{V}, \mathcal{T})
}
```

其中：

- `T` 是 ImageNet training set。
- `V` 是 ImageNet validation set。
- validation images 的 FDr 被定义为 1.0。

如果：

```text
FDr = 2.0
```

意思是生成图像在该 representation space 中，距离训练集的分布差距是 validation images 的两倍。

再平均多个 representation：

```math
\mathrm{FDr}^{K}(\mathcal{G})
=
\frac{1}{K}\sum_i \mathrm{FDr}_{\phi_i}(\mathcal{G})
```

论文主要报告 `FDr^6`。

## 7. 一个容易误解的点：FD-loss 不是 per-image reward

FD-loss 并不会告诉某张图：

```text
你这张图哪里错了
```

它只告诉当前 batch：

```text
你们这批图像加入 population 后，
整体 feature distribution 应该往哪里移动。
```

所以它更像 distribution matching，而不是 caption loss、reconstruction loss 或 discriminator loss。

## 8. 方法的核心直觉

一句话版：

```text
If evaluation judges a generator by population feature statistics,
then post-training can directly move the generator's population statistics.
```

真正难的是让这个 population statistic 在训练时又大、又新、又可微。

