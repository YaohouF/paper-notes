# Overview

HiDream-O1-Image 这篇工作的核心问题是：能不能不再把图像生成拆成一堆外部模块，而是让一个统一 Transformer 同时处理文本、条件图像和目标图像像素？

传统 T2I 系统通常是：

```text
text -> external text encoder
image -> VAE latent
latent + text condition -> DiT / U-Net
latent -> VAE decoder -> pixels
```

HiDream-O1-Image 想改成：

```text
text tokens + condition tokens + noisy pixel patches
  -> one decoder-only Transformer
  -> clean pixel patches
```

这也是论文标题里 “Natively Unified” 和 “Pixel-level Unified Transformer” 的含义。

## 论文想解决什么

论文认为现有图像生成模型有三个瓶颈：

1. VAE latent 会带来压缩损失，尤其对文字、小结构、复杂布局不友好。
2. 独立 text encoder 和 diffusion backbone 之间存在 semantic alignment gap。
3. 许多 pixel-space DiT 虽然去掉 VAE，但仍然只解决 T2I，不是真正统一编辑、个性化、多模态理解和语言能力。

HiDream-O1-Image 的回应是：把 raw pixels、text、conditions 都转成同一个 token space，由统一 Transformer 上下文建模。

## 主要贡献

| 方向 | HiDream-O1-Image 的做法 |
|---|---|
| 模型 | Pixel-level Unified Transformer，直接建模 raw pixel patches |
| Backbone | 8B 版本初始化自 Qwen3-VL-8B-Instruct；另有 200B+ Pro 版本 |
| Tokenization | 文本 token、condition token、generation token 三类 token 统一拼接 |
| Attention | text/condition causal，generation token full attention |
| Prompt | Reasoning-Driven Prompt Agent 先把用户短 prompt 扩展成结构化英文 prompt |
| 训练 | progressive pre-training + SFT + GRPO/RLHF + distillation |

## 它在我们当前主线里的位置

我们现在读过三类相近工作：

```text
TUNA-2
  -> 证明 pixel embedding 可以替代 vision encoder / VAE 的一部分角色

SenseNova-U1
  -> 更强调 understanding / generation 两类目标的 stream-wise 参数解耦

HiDream-O1-Image
  -> 更强调 Qwen3-VL 初始化 + pixel-level unified Transformer + hybrid attention
```

我的理解是，HiDream-O1-Image 的“统一”更像是输入 token / attention 级别的统一；SenseNova-U1 的“统一”更进一步触及 Transformer block 内部的 stream-specific 参数组织。

## 一句话记忆

```text
HiDream-O1-Image = Qwen3-VL initialized decoder-only backbone
                 + raw pixel patch diffusion
                 + timestep special token
                 + AR/full hybrid attention
                 + prompt reasoning agent
```

