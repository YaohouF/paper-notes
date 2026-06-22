# Paper-Code Crosscheck

官方代码仓库：

```text
/Users/yoho/Desktop/my_paper/fd-loss/FD-loss-main
```

代码链接来自论文 abstract：

```text
https://github.com/Jiawei-Yang/FD-loss
```

## 1. FD 是否真的可微？

是。核心在 `frechet_distance/losses.py`：

```python
def compute_frechet_distance_loss(
    mu_ref,
    sigma_ref,
    all_feats=None,
    mu=None,
    sigma=None,
    sigma_ref_sqrt=None,
):
```

如果传入 `all_feats`，它会直接计算：

```python
mu = all_feats.mean(dim=0)
feats_c = all_feats - mu
sigma = (feats_c.T @ feats_c) / (n_samples - 1)
```

然后计算 mean term 和 covariance trace term。因为 `all_feats` 可以带 autograd graph，所以 FD 对当前 features 可微。

## 2. 多 GPU all-gather 是否保留梯度？

是。代码写了自定义 autograd function：

```python
class _DiffAllGather(torch.autograd.Function):
```

forward 做 distributed all-gather；backward 只把属于当前 rank 的 gradient chunk 返回。

这使得全局 batch features 可以参与 FD 统计，同时每个 GPU 只接收自己那部分梯度。

## 3. queue estimator 是否真的只对当前 batch 反传？

是。`FeatureQueue.build_feats_snapshot` 会 clone/detach 旧 queue：

```python
snap = buf.clone().detach()
snap[ptr : ptr + n] = new
```

旧 features 是 constants；当前 `new_feats` 被放进 snapshot 的 pointer 区域，保留梯度。

## 4. EMA estimator 是否与论文一致？

一致。代码里：

```python
mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B
sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
```

训练后更新 EMA：

```python
self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
```

这和论文的一阶/二阶矩 EMA 公式一致。

## 5. normalized FD loss 是否与公式一致？

是。论文公式：

```math
\mathcal{L}_{\phi_i}
=
\frac{\mathrm{FD}_{\phi_i}}
{\mathrm{sg}(\mathrm{FD}_{\phi_i}) + c}
```

代码：

```python
fid_loss = fid / (fid.detach() + fid_norm_eps)
loss = loss + judge["weight"] * fid_loss
```

默认：

```python
--fd_fid_norm_eps 0.01
```

## 6. matrix square root 如何实现？

论文说预计算：

```math
\Sigma_r^{1/2}
```

然后使用 symmetric product：

```math
\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}
```

代码里对应：

```python
sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
M = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
evals = torch.linalg.eigvalsh(M)
tr_covmean = torch.sum(torch.sqrt(evals))
```

这避免了每步显式计算矩阵平方根，和 appendix 描述一致。

## 7. training loop 是否是 post-training？

是。`main_fd.py` 里：

```python
model, ema_model = create_generation_model(args)
optimizer = create_optimizer(args, model, print_trainable_params=True)
extra = ckpt_resume(...)
```

训练脚本从 base checkpoints 加载模型，然后进行 FD fine-tuning。README 也明确说：

```text
Training starts from the released base checkpoints.
```

## 8. 每步是否真的不需要真实图像 batch？

是。训练 step 只采样：

```python
z = torch.randn(...)
y = torch.randint(...)
sampled = model_wo_ddp.sample_images_with_grad(z, y, ...)
```

当前 step 不加载真实图像。真实图像只通过预计算 reference stats 进入 loss。

这也是 FD-loss 和普通 supervised / reconstruction loss 很不同的地方。

## 9. representation models 是否 frozen？

是。`frechet_distance/repr_models.py` 中：

```python
self.model.to(device).eval().requires_grad_(False)
```

这些 representation models 是 judges，不被训练。

## 10. 代码支持哪些 representation？

代码支持：

- `inception`
- `convnext`
- 任意 timm model name，例如 DINOv2、MAE、CLIP、SigLIP2

`TimmReprModel` 返回：

```python
return cls_token, mean_token
```

`pool_type='cls'` 或 `'avg'` 控制用哪种 feature。

## 11. queue warm-start 是否实现？

是。`frechet_distance/judges.py`：

```python
fill_all_queues(judges, model_wo_ddp, args, tokenizer=tokenizer)
```

它用 base/current model 先生成图像，填满 queue 或初始化 EMA moments。论文说 warm-start 50k samples，代码逻辑一致。

## 12. 脚本是否可复现实验表？

README 和 `scripts/README.md` 提供：

```text
table_1a_queue_size.sh
table_1b_ema_beta.sh
table_1c_backbone_single.sh
table_1c_backbone_combo.sh
table_2_repurpose_jit_L.sh
table_3_pMF.sh
table_3_iMF.sh
table_3_JiT.sh
```

这比很多论文更完整。只是实际复现仍需要 ImageNet、base checkpoints、多 GPU、representation stats。

## 13. 一个重要代码版理解

FD-loss 的核心训练 step 可以压缩成：

```python
sampled = generator.sample_images_with_grad(z, y)
features = frozen_repr_model(sampled)

mu_g, sigma_g = queue_or_ema.build_stats(features)
fd = frechet_distance(mu_ref, sigma_ref, mu_g, sigma_g)
loss = fd / (fd.detach() + eps)

loss.backward()
optimizer.step()
queue_or_ema.enqueue(features.detach())
```

这就是整篇论文最核心的实现。

