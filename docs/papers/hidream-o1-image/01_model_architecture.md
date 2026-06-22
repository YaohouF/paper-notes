# Model Architecture

HiDream-O1-Image 的架构可以拆成四件事来看：

1. 三类 token 如何进入同一个空间。
2. Qwen3-VL backbone 如何承载 pixel diffusion。
3. causal attention 和 full attention 如何共存。
4. 输出的 clean image patches 如何用于采样。

## 1. Unified multimodal tokenization

论文把输入分成三类 primitive tokens：

| Token type | 含义 | 作用 |
|---|---|---|
| Text tokens `y` | refined prompt | 提供语义、布局、风格、文字等指令 |
| Condition tokens `c` | source/reference images, task-specific conditions | 用于 editing / subject personalization 等视觉条件任务 |
| Generation tokens `x_t` | noisy target image | diffusion/flow 过程中要被还原的目标图像 token |

其中 generation token 来自线性插值：

```math
x_t = t x + (1 - t)\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
```

这里 `x` 是 clean target image，`epsilon` 是 Gaussian noise，`t` 是 diffusion/flow timestep。

## 2. Pixel patch embedding

官方代码里，target image 的 patch size 是 32：

```python
self.patch_size = 32
self.in_channels = 3
self.x_embedder = BottleneckPatchEmbed(
    patch_size=32,
    in_chans=3,
    pca_dim=hidden_size // 4,
    embed_dim=hidden_size,
)
```

也就是说，一个 visual generation token 覆盖：

```text
32 * 32 * 3 RGB values
```

`BottleneckPatchEmbed` 不是简单一层 linear，而是：

```text
flattened 32x32 RGB patch
  -> bottleneck dim hidden_size/4
  -> hidden_size
```

输出端对应一个 `FinalLayer`：

```text
hidden_size -> 32 * 32 * 3
```

所以它是很直接的 pixel patch input/output interface。

## 3. Backbone 是什么

论文说 8B 版本初始化自 Qwen3-VL-8B-Instruct。公开代码里也能看到模型主体是 Qwen3-VL 的结构变体：

```python
self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
self.language_model = Qwen3VLTextModel._from_config(config.text_config)
```

这点非常重要：HiDream-O1-Image 不是从零开始训练一个纯 DiT，而是在 Qwen3-VL 的 multimodal understanding backbone 上加入 pixel diffusion 所需的 patch embedding、timestep token 和 image patch prediction head。

所以它的 “no disjoint text encoder” 更准确地理解是：不像 FLUX / SD3 那样外接独立 T5/CLIP text encoder；文本能力在同一个 Qwen3-VL/decoder backbone 内部承载。

## 4. timestep 注入方式

HiDream 的 timestep 注入很像“把时间当成一个语言模型 token”，而不是传统 DiT 里每层 AdaLN 调制。

代码里有一个特殊 token id：

```python
self.tms_token_id = 151673
```

forward 时：

```python
timestep_embedding = self.t_embedder1(timestep)
tms_mask = input_ids == self.tms_token_id
inputs_embeds[tms_mask] = timestep_embedding
```

也就是 prompt sequence 里出现 `<|tms_token|>`，它原本的 token embedding 会被 timestep embedding 替换。

这件事的意义是：模型仍然像 decoder-only LLM 一样吃一个 token sequence；diffusion time 只是 sequence 里的一个专门 token。

## 5. Hybrid Unified Attention

论文中最关键的机制是 hybrid unified attention：

| Token | Attention behavior |
|---|---|
| Text / condition tokens | causal attention，只看前面的 tokens |
| Generation tokens | full attention，可以看所有 tokens |

这解决了一个结构冲突：

```text
language modeling wants causal attention
image diffusion wants bidirectional / full attention
```

公开代码有两条实现路径。

非 flash attention 路径大意是：

```python
causal_mask = upper_triangular_mask()
gen_positions = token_types[b].bool()
causal_mask[gen_positions, :] = 0
```

这表示 generation token 行的 mask 被清空，所以它们可以 attend to all tokens。

flash attention 路径采用 two-pass：

```text
1. run causal attention for AR tokens
2. run full attention for all tokens
3. put causal outputs back onto AR positions
```

所以 AR tokens 保持 autoregressive，generation tokens 得到 full context。

## 6. T2I 时 sequence 大概长什么样

公开代码的 T2I 构造可以简化成：

```text
[chat-formatted prompt tokens] + [boi token] + [timestep token]
  + [noisy target image patch tokens appended through vinputs]
```

需要注意：target image patch tokens 不是 tokenizer 产生的 discrete token，而是作为 `vinputs` 经过 `x_embedder` 后 append 到 embedding sequence。

## 7. Editing / personalization 时 condition 怎么进来

论文说 condition images 会被投影成 condition tokens。公开代码里，reference/source image 有两条相关路径：

1. 作为 Qwen3-VL 的 visual input，通过 vision encoder 替换文本中的 image placeholders。
2. 在部分任务里，reference patches 也会拼到 `vinputs` 后面，和 target noisy patches 一起作为 full-attention token 参与上下文。

所以实践里它不是“所有图像都用同一层 raw pixel patch embedding”。target generation image 是 direct pixel patches；condition/reference image 还会利用 Qwen3-VL 的视觉编码能力。

## 8. 输出：predict clean image patches

论文描述是 output head 把 token 映射回 clean image patch。代码里：

```python
x_pred = self.final_layer2(hidden_states)
```

pipeline 里取出 target image positions 的 `x_pred`，再换成 velocity：

```python
v = (x_pred - z) / sigma
model_output = -v
```

这里：

- `z` 是当前 noisy image patch state。
- `x_pred` 是模型预测的 clean image patch。
- `sigma` 是当前 noise level。
- scheduler 接收的是 `model_output`，也就是由 clean prediction 换算出来的 velocity/denoising direction。

## 9. 和普通 DiT / MMDiT 的区别

普通 DiT 通常是：

```text
VAE latent patches + timestep AdaLN + text cross-attention / joint attention
```

HiDream-O1-Image 是：

```text
raw pixel patches + timestep special token + decoder-only unified attention
```

它更像把 diffusion generation 塞进一个 LLM-style multimodal decoder，而不是把文本条件塞进一个 diffusion transformer。

## 10. 和 SenseNova-U1 的关键区别

SenseNova-U1 的 MoT block 重点是 stream-specific 参数：

```text
understanding stream params
generation stream params
```

HiDream-O1-Image 公开代码里没有看到这种显式双 Transformer/MoT 参数解耦。它的重点是：

```text
same decoder stack
different token types
different attention visibility
```

所以，如果你问“它是不是和 SenseNova 一样干净分成两个 stream？”我的答案是：不是。它是统一 sequence + hybrid attention，而不是 SenseNova 那种更强的 stream-wise Transformer 参数隔离。

