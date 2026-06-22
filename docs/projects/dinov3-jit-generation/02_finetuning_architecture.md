# Finetuning Architecture

finetune 阶段把 pretrained time-conditioned DINOv3 encoder 接到一个 pixel-space denoising decoder 上。当前路线是 class-to-image generation，输入是 noisy image `x_t`、time `t` 和 class label `y`，输出预测 clean image `x0`。

## Forward Path

当前 baseline forward 可以写成：

```text
x_t, t, y
  -> label_embedder(y)
  -> encoder.forward_tokens_with_skips(x_t, t, cls_token_offset=y_embed)
  -> drop storage tokens
  -> encoder_to_decoder projection
  -> skip-connected AdaLN decoder
  -> final conditioned head
  -> pred_x0
```

## Encoder Token Layout

baseline path 的 encoder input layout 是：

```text
[time, CLS + y, storage, patch]
```

其中：

- `time` 来自 pretrained encoder 的 `t_embedder(t)`。
- `CLS + y` 是把 label embedding 加到已有 CLS token 上，而不是新插入一个 class token。
- `storage` 是 DINOv3 storage/register tokens。
- `patch` 是 noisy image `x_t` 的 patch tokens。

## Storage Tokens Are Removed

进入 decoder 前会删除 storage tokens。当前实现保留：

```text
[time/class tokens, CLS, patch tokens]
```

删除：

```text
storage tokens
```

这件事同时发生在：

```text
main decoder input sequence
skip connection features
MLS feature fusion path
```

对应逻辑是：

```python
n_keep_prefix = num_t_tokens + num_c_tokens + 1
tokens = concat(
    tokens[:, :n_keep_prefix],
    tokens[:, n_keep_prefix + n_storage:]
)
```

baseline 中 `num_t_tokens=1, num_c_tokens=0`，所以保留 `[time, CLS, patch]`。

## Decoder

decoder 是一个 AdaLN-conditioned transformer stack。每个 decoder block 接收：

```text
dec tokens
global AdaLN condition cond
optional skip features
```

当前 block 结构是：

```text
skip fusion:
  concat(dec, skip) -> linear -> optional norm

self-attention:
  norm -> AdaLN modulate -> attention -> gated residual

MLP:
  norm -> AdaLN modulate -> MLP -> gated residual
```

final head 是 conditioned pixel head：

```text
decoder patch tokens -> AdaLN final layer -> patch pixels -> unpatchify
```

## Skip Connections

skip connection 是当前 preferred route 的重要部分。因为 DINO-style high-level tokens 容易偏语义，skip features 可以给 decoder 提供更多局部结构和中间层细节。

当前路径：

```text
encoder block outputs
  -> reverse order
  -> remove storage tokens
  -> optional norm
  -> decoder block skip fusion
```

注意，skip path 里也保留 time/class/CLS prefix tokens。它们和 patch tokens 一起进入 skip fusion。

## Decoder Positional Embedding

代码支持一个可选的 decoder positional embedding：

```yaml
generation:
  use_decoder_pos_embed: false
```

当前 preferred route 是不使用 decoder positional embedding。原因是这个项目还没有证明它稳定带来收益，而且旧 checkpoint 与新 pos embed 的兼容性容易混淆。

因此当前实验应显式保持：

```text
use_decoder_pos_embed = false
```

## Loss

finetune model head 输出 `pred_x0`，但 JiT path 使用 velocity-style loss：

```text
target_v = (x0 - xt) / clamp(t, t_eps)
pred_v   = (pred_x0 - xt) / clamp(t, t_eps)
loss     = mse(pred_v, target_v)
```

这样可以保持 output interface 简单，同时和 JiT 的 denoising objective 对齐。

## Sampling

sampling 默认使用 Heun sampler：

```yaml
sample:
  sampler: heun
  sampling_step: 50
  cfg_scale: 3.2
```

当前 `t` 语义是 noise fraction。采样时模型反复预测 `x0`，再转换为 ODE derivative 更新 noisy image。
