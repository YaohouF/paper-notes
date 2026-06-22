# Ablation Log and Open Questions

这个页面记录当前已经比较明确的经验，以及仍然需要继续验证的模块。它不是最终实验表，而是项目维护用的 research log。

## Confirmed Findings

### Drop path should be disabled in generation finetuning

结论：

```text
generation finetune: drop_path_rate = 0.0
```

这是当前最重要的经验之一。实验已经确认 drop path 是训练失败和噪点问题的主要原因。

原因：

```text
In generation finetuning, the encoder output is not just a representation;
it is the decoder's condition sequence and skip feature source.
```

Drop path 会随机扰动 encoder block 的 residual path，使 decoder 每一步看到的 condition / skip features 不稳定。主体语义可能仍然存在，但背景和高频细节更容易崩。

### Time tokens should enter the decoder sequence

早期版本中，如果 time token 没有贯穿到 decoder sequence，容易出现背景噪点。当前路线保留 time token，并让它进入 decoder input sequence 和 skip features。

当前 baseline：

```text
encoder: [time, CLS+y, storage, patch]
decoder: [time, CLS+y, patch]
```

### Storage tokens should be removed before decoder

DINOv3 storage/register tokens保留在 encoder 内部，但进入 decoder 前会删除。

原因是这些 tokens 主要服务于 representation pretraining，不一定适合直接参与 pixel reconstruction decoder。

### Decoder time embedding should not be hard-shared

当前设计：

```text
decoder cond_embedder initialized from encoder t_embedder
but trained independently
```

这比硬共享更安全。encoder time token 和 decoder AdaLN condition 的角色不同，不应强行绑定。

### Decoder positional embedding is not part of the current preferred route

代码支持：

```yaml
use_decoder_pos_embed: true / false
```

但当前 preferred route 是：

```text
use_decoder_pos_embed = false
```

这可以减少额外变量，也避免旧 checkpoint evaluation 时引入随机新参数。

## Current Preferred Finetune Configuration

当前建议的稳定路线：

```yaml
student:
  drop_path_rate: 0.0

generation:
  training_style: jit
  use_skip_connections: true
  skip_pre_norm: false
  skip_post_norm: true
  num_t_tokens: 1        # baseline
  num_c_tokens: 0        # baseline
  use_decoder_pos_embed: false
  cfg_drop_rate: 0.1
  std: simple
```

扩展实验可以在 baseline 稳定后打开：

```yaml
generation:
  num_t_tokens: 4
  num_c_tokens: 8 or 16
```

## Open Questions

### How many time/class tokens are actually useful?

更多 time/class slots 提供更大 condition capacity，但也改变 prefix token distribution。需要判断收益来自：

```text
more conditioning capacity
or simply more trainable tokens
or unstable shortcut behavior
```

### Skip connection vs MLS

当前 preferred route 是 per-block skip connection。MLS 可以把多个 encoder layers 的 features 先融合，再交给 decoder。

开放问题：

```text
Can MLS reduce memory and improve stability
without changing denoising semantics too much?
```

### Decoder block design

当前 decoder 使用 AdaLN + attention + MLP，并已经尝试向 JiT-style conditional head 靠近。

还可以继续比较：

```text
standard attention vs QK-norm attention
MLP vs SwiGLU
RMSNorm vs LayerNorm
decoder depth / width
```

### Optimizer and schedule

当前仍需分开验证：

```text
encoder lr vs decoder lr
constant schedule vs warmup-cosine
beta2 = 0.95 vs 0.999
weight decay on decoder
grad clip strength
```

注意：这些变量容易和模型结构改动耦合，不适合一次全改。

### Pretraining objective balance

JiT denoising auxiliary 目前是小权重辅助项。后续需要继续判断：

```text
too weak:
  encoder tokens may remain too semantic

too strong:
  SSL representation may be damaged
```

## Suggested Experiment Order

推荐以后按这个顺序扩展：

1. 固定 stable baseline：`drop_path_rate=0.0`，`num_t_tokens=1`，`num_c_tokens=0`，`use_decoder_pos_embed=false`。
2. 单独验证新的 pretrained checkpoint，不同时改 finetune architecture。
3. 加多 time tokens，只改 `num_t_tokens`。
4. 加 class slots，再改 `num_c_tokens`。
5. 再考虑 decoder block、optimizer schedule、MLS 等更大变量。

## Current Project Status

这个项目的当前状态不是“final method”，而是：

```text
a modular DINOv3 + JiT generation framework
with a stable preferred route and several open ablation axes.
```

维护文档时应避免把某个暂时实验配置写成最终结论。更好的写法是明确区分：

```text
confirmed implementation
confirmed empirical finding
current preferred route
open research question
```
