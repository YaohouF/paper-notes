# TUNA-2 vs SenseNova-U1 vs HiDream-O1-Image

这三篇都在往 native unified multimodal model 走，但它们解决的问题侧重点不一样。

## 高层对比

| Dimension | TUNA-2 | SenseNova-U1 | HiDream-O1-Image |
|---|---|---|---|
| Main goal | pixel embedding 替代 VE/VAE | unified understanding + generation with MoT | pixel-level unified generation foundation model |
| Backbone | Qwen2.5 decoder | NEO-unify / Qwen3-style Native MoT | Qwen3-VL initialized decoder-only UiT |
| Vision encoder | 移除 | 移除 | target generation 不用 VAE；condition path 使用 Qwen3-VL visual encoder |
| VAE | 移除 | 移除 | 移除 |
| Visual token | 16×16 pixel embedding | effective 32×32 token | 32×32 raw pixel patch |
| Stream decoupling | 弱 | 强，MoT | 不显式 MoT；靠 token type + hybrid attention |
| Time injection | time/noise embedding + generation head | noisy image path 注入 timestep/noise | specialized timestep token |
| Text output | LM head | LM head | 主要生成图像；继承 Qwen3-VL AR ability |
| Image output | flow head | pixel flow head / MLP | linear RGB patch head predicts clean patch |

## 三者最像的地方

它们都不满足于传统：

```text
vision encoder for understanding
VAE latent diffusion for generation
```

而是尝试把图像和文字放入更统一的 token/backbone 系统里。

## 三者最不同的地方

### TUNA-2：证明 pixel embedding 这条路能走

TUNA-2 的重点是：

```text
Can simple pixel embeddings replace pretrained vision encoders?
```

它更像一个强 baseline：把 visual input 变成 pixel patches，塞进 Qwen decoder，再用 LM head / flow head 输出。

### SenseNova-U1：解决理解和生成目标干扰

SenseNova-U1 的重点是：

```text
How to share context but decouple understanding/generation parameters?
```

它引入 Native MoT，使 clean understanding stream 和 noisy generation stream 在 Transformer 内部走不同参数路径。

### HiDream-O1-Image：把 pixel diffusion 做成 Qwen3-VL-style unified decoder

HiDream-O1-Image 的重点是：

```text
Can a Qwen3-VL initialized decoder directly generate raw pixel patches?
```

它不强调 MoT stream 参数解耦，而强调：

```text
raw pixel patches
special timestep token
hybrid causal/full attention
reasoning-driven prompt agent
```

## Attention / block 机制对比

| Model | Attention / block idea |
|---|---|
| TUNA-2 | 更接近把 visual tokens 接进 Qwen decoder，再接 generation head |
| SenseNova-U1 | Native MoT：understanding/generation stream 参数解耦，同时保持统一上下文 |
| HiDream-O1-Image | text/condition tokens causal；generation tokens full attention |

一个非常短的记忆方式：

```text
TUNA-2: unified by pixel embedding
SenseNova-U1: unified by MoT streams
HiDream-O1: unified by Qwen3-VL decoder + hybrid attention
```

## Time embedding 对比

| Model | Time injection |
|---|---|
| TUNA-2 | generation/noise side 注入 timestep，供 flow head 使用 |
| SenseNova-U1 | noisy image token 加 timestep/noise embedding |
| HiDream-O1-Image | timestep 被编码成 `<|tms_token|>` special token |

HiDream 这个设计很 LLM：它不是把 time 当作每层调制信号，而是把 time 放进 sequence。

## 输出预测对比

| Model | Image prediction |
|---|---|
| TUNA-2 | flow/noise-style target，结合 diffusion objective |
| SenseNova-U1 | pixel-space flow matching head |
| HiDream-O1-Image | predicts clean RGB patches `x0`，再换算成 velocity 给 scheduler |

## 如果只用一句话区分

```text
TUNA-2 asks whether pixels can replace encoders.
SenseNova-U1 asks how understanding and generation can coexist without hurting each other.
HiDream-O1 asks whether a Qwen3-VL-style decoder can become a raw-pixel diffusion foundation model.
```

