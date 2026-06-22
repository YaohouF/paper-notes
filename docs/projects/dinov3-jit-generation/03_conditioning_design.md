# Conditioning Design

这个项目最容易出问题的部分就是 condition construction。因为 encoder 和 decoder 都看 time/class 信息，但它们的角色不一样：encoder 需要把 condition 编进 token interactions，decoder 需要用 condition 调制 denoising blocks。

## Baseline Condition Layout

baseline 设置：

```yaml
generation:
  num_t_tokens: 1
  num_c_tokens: 0
```

对应 encoder sequence：

```text
[time, CLS + y, storage, patch]
```

对应 decoder sequence：

```text
[time, CLS + y, patch]
```

对应 decoder AdaLN condition：

```text
cond = cond_embedder(t) + label_embedder(y)
```

## Decoder Time Embedder

当前实现中：

```python
cond_embedder.load_state_dict(encoder.t_embedder.state_dict())
```

这意味着：

```text
initially:
  decoder cond_embedder == encoder t_embedder

during finetune:
  decoder cond_embedder and encoder t_embedder are independent parameters
```

这是一个重要设计点。它避免了 decoder AdaLN condition 和 encoder time token 被硬共享，同时又让两者从相同 pretrained time embedding 起步。

## Multi Time Tokens and Class Tokens

当前代码支持 RAEv2-style 多 condition slots：

```yaml
generation:
  num_t_tokens: 4
  num_c_tokens: 16
```

multi-token path 中：

```text
t_base = encoder.t_embedder(t)
time_tokens = t_base + time_slots

y_embed = label_embedder(y)
class_tokens = y_embed + class_slots
```

encoder input sequence 变为：

```text
[time tokens, class tokens, CLS, storage, patch]
```

删除 storage 后进入 decoder：

```text
[time tokens, class tokens, CLS, patch]
```

decoder AdaLN condition 仍然是：

```text
cond = cond_embedder(t) + y_embed
```

也就是说，多 tokens 影响 encoder/decoder token sequence，但不替代 decoder block 的 global AdaLN condition。

## 为什么不硬共享 encoder 和 decoder time embedding

曾经考虑过让 decoder 直接使用 encoder 的 `t_embedder(t)` 作为 AdaLN condition。这个设计看起来更“对齐”，但有风险：

```text
encoder time token:
  participates in token interaction
  affects representation sequence

decoder time condition:
  modulates denoising residual blocks
  controls generation dynamics
```

两者虽然都来自同一个 scalar `t`，但功能不同。硬共享会让两条路径互相牵制，可能导致训练不稳定。当前更稳妥的做法是：

```text
same initialization, separate optimization
```

## Class Conditioning

baseline 中 class label 不作为独立 class token 插入 encoder，而是：

```text
CLS token = CLS token + label_embedder(y)
```

在 multi-token path 中，class slots 可以作为额外 prefix tokens：

```text
class_tokens = y_embed + learnable class_slots
```

同时 decoder AdaLN 仍然收到 `y_embed`。这意味着 class information 同时通过：

```text
encoder sequence
decoder AdaLN condition
```

进入模型。

## CFG Dropout

训练时使用 classifier-free guidance dropout：

```yaml
generation:
  cfg_drop_rate: 0.1
```

当 label 被 drop 时，`y` 会变成 null class，对 encoder `CLS+y` 和 decoder AdaLN `y_embed` 同时生效。

## 当前设计原则

当前 condition 设计可以总结为：

```text
Token sequence conditioning:
  used for encoder-decoder interaction and skip features

AdaLN conditioning:
  used for global denoising modulation

Time embedder:
  copied from pretrained encoder to decoder at initialization,
  but optimized separately

Storage tokens:
  removed before decoder
```

## 风险点

多 time/class tokens 仍然是开放实验变量。它可能提供更强 condition capacity，但也会改变 decoder sequence prefix distribution。

因此推荐实验顺序是：

```text
1. num_t_tokens=1, num_c_tokens=0
2. verify no noise / stable convergence
3. add more time tokens
4. add class tokens
5. keep drop_path_rate=0.0 throughout
```
