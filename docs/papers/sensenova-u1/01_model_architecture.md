# SenseNova-U1 Model Architecture

## 1. 整体架构

论文把 SenseNova-U1 描述为 NEO-unify 上的 native unified model：

```text
pixels + words
  -> near-lossless visual/text interface
  -> Native Mixture-of-Transformers backbone
  -> LM head for text
  -> pixel-space flow head for image
```

它去掉两类传统模块：

- pretrained vision encoder, e.g. CLIP/SigLIP/EVA；
- VAE encoder/decoder, e.g. latent diffusion 里的 VAE。

所以它不是：

```text
image -> SigLIP -> projector -> LLM
image generation -> VAE latent -> diffusion -> VAE decoder
```

而是更直接：

```text
image pixels -> patch encoder -> Transformer tokens
image generation -> predict pixel patches directly
```

## 2. Near-Lossless Visual Interface

### 2.1 论文说法

论文说：

```text
two convolutional layers + GELU + 2D sinusoidal positional encoding
strides = 16 and 2
each token corresponds to a 32 x 32 image patch
```

图片用 `<img>` 和 `</img>` 包住。文本仍然用底层 LLM 原始 tokenizer，不重新设计文字 tokenizer。

### 2.2 代码对应

公开 inference/runtime 代码中的视觉 embedding 在：

```text
sensenova/SenseNova-U1-main/src/sensenova_u1/models/neo_unify/modeling_neo_vit.py
```

核心是：

```python
self.patch_embedding = nn.Conv2d(
    in_channels=config.num_channels,
    out_channels=self.embed_dim,
    kernel_size=self.patch_size,
    stride=self.patch_size,
)

self.dense_embedding = nn.Conv2d(
    in_channels=self.embed_dim,
    out_channels=self.llm_embed_dim,
    kernel_size=self.downsample_factor,
    stride=self.downsample_factor,
)
```

训练配置里：

```python
patch_size = 16
down_sample_ratio = 0.5
downsample_factor = int(1 / downsample_ratio) = 2
```

所以有效 token 覆盖：

```text
16 x 16 low-level patch
then 2 x 2 merge
=> 32 x 32 final visual token
```

这和论文里的 32×32 compression/token 对齐。

### 2.3 输入图像的形状逻辑

代码里 `pixel_values` 已经是 patch-flattened 形式：

```python
pixel_values = pixel_values.view(-1, 3, patch_size, patch_size)
patch_embeds = GELU(Conv2d(pixel_values))
```

然后按每张图的 `grid_hw` 还原成二维 patch grid，经过第二个 `dense_embedding` 合并成最终 LLM hidden size。

可以理解为：

```text
RGB image
  -> split into 16x16 patches
  -> Conv2d maps each patch to hidden
  -> 2D RoPE on patch grid
  -> second Conv2d merges 2x2 patch tokens
  -> final 32x32 visual token
```

## 3. Text Path

文本没有新 tokenizer，仍使用 Qwen3 tokenizer。

代码配置：

```python
VOCAB_SIZE = 151936
TOKENIZER_PATH = "/path/to/qwen3/tokenizer"
```

文本路径：

```text
text
  -> Qwen3 tokenizer
  -> token ids
  -> language_model.tok_embeddings / embed_tokens
  -> Qwen3 MoT backbone
  -> lm_head
```

`embed_tokens` 和 `lm_head` 在 parameter breakdown 中被算作 shared，因为 generation pathway 里的 thinking/interleaved 也会生成文字。

## 4. Understanding Stream vs Generation Stream

SenseNova-U1 最关键的概念是 stream。

### Understanding stream

处理：

- clean image tokens；
- normal text tokens；
- VQA / OCR / reasoning / multimodal dialogue。

代码模块：

```text
vision_model
language_model normal branch
lm_head
```

### Generation stream

处理：

- noisy image tokens；
- T2I / editing / interleaved image generation。

代码模块：

```text
fm_modules.vision_model_mot_gen
language_model *_mot_gen branch
fm_modules.timestep_embedder
fm_modules.noise_scale_embedder
fm_modules.fm_head
```

训练代码里 `extract_feature` 同时跑两条视觉路径：

```python
outputs = self.vision_model(pixel_values=...)
outputs_image_gen = self.fm_modules["vision_model_mot_gen"](
    pixel_values=pixel_values_noised,
    ...
)

vit_embeds = outputs.last_hidden_state
vit_embeds_image_gen = outputs_image_gen.last_hidden_state
```

然后根据 `image_gen_indicators` 选择当前 image context token 应该放 clean visual embedding 还是 noisy generation embedding：

```python
mixed = torch.where(gen_flags[:, None], vit_embeds_image_gen, vit_embeds)
hidden_states[selected] = mixed
```

## 5. Native Mixture-of-Transformers

### 5.1 论文说法

论文说 MoT 在每层中对两条 stream 做 full parameter decoupling：

- separate projections；
- separate normalizations；
- separate feedforward blocks；
- dynamic routing by token type。

意思不是只有最后的 head 分开，而是 Transformer 内部就按 token 类型走不同参数。

### 5.2 代码中的 dense 8B MoT

训练代码主干：

```text
sensenova/SenseNova-U1-main/training/sensenovalm/model/modeling_qwen3_moe_mot.py
```

每层中有 generation-side norm/FFN：

```python
self.attention_norm_mot_gen
self.ffn_norm_mot_gen
self.feed_forward_mot_gen
```

forward 时先统计 generation tokens：

```python
num_image_gen_tokens = image_gen_indicators.sum().item()
```

然后把 sequence pack 成 generation tokens 在前，non-generation tokens 在后。FFN 计算时：

```python
hidden_states = torch.cat([
    self.feed_forward_mot_gen(hidden_states[:, :num_image_gen_tokens]),
    self.feed_forward(hidden_states[:, num_image_gen_tokens:])
], 1)
```

这就是 dense MoT：不是 MoE，但有两套 Transformer FFN/norm 分支。

### 5.3 代码中的 A3B MoE-MoT

MoE 情况下：

```python
hidden_states_gen, gate_logits_gen, moe_gen = self.feed_forward_mot_gen(
    hidden_states[:, :num_image_gen_tokens]
)

hidden_states_und, gate_logits_und, moe_und = self.feed_forward(
    hidden_states[:, num_image_gen_tokens:]
)
```

也就是说：

- generation tokens 走 generation MoE experts；
- understanding/text/clean image tokens 走 understanding MoE experts。

论文表格对应：

```text
A3B:
  understanding experts: 128 total, top-8 active
  generation experts: 32 total, top-8 active
```

## 6. Attention Mask

### 6.1 普通理解 mask

代码函数：

```python
create_flex_mask_padding(...)
```

规则：

```text
same document causal attention
OR
same image bidirectional attention
```

代码等价于：

```python
causal_mask = q_idx >= kv_idx
samedoc_mask = document_ids[q_idx] == document_ids[kv_idx]
sameimg_mask =
    is_image(q)
    and same_image(q, k)
    and same_doc(q, k)

mask = (causal and same_doc) OR same_image
```

含义：

- 文本仍是 causal；
- 同一张图内部 patch tokens 可以双向互看；
- 后续文本可以看前面的图像；
- 不同 document 不互看。

### 6.2 Generation mask

代码函数：

```python
create_flex_mask_padding_image_gen(...)
```

先沿用：

```text
same-document causal OR same-image bidirectional
```

再加入 generation token gating：

```python
kv_is_gen = image_gen_indicators[kv_idx]
q_is_gen = image_gen_indicators[q_idx]
same_img = same image and same doc

gate1 = (~kv_is_gen) | (q_is_gen & same_img)
gate2 = (~q_is_gen) | kv_is_gen | (token_pos[kv_idx] != token_pos[q_idx])
```

直觉解释：

- 普通 clean tokens 不能随便 attend 到 future/noisy generation tokens，避免信息泄漏；
- generation tokens 可以在同一 noisy image block 内部双向互动；
- generation tokens 可以看前面的 clean context；
- duplicated clean/noisy image 的边界还有额外 `dup_boundary` gating，避免不该泄漏的 paired tokens。

论文里说的：

```text
noise tokens within each image block attend bidirectionally,
with full access to clean inputs,
whereas clean tokens are prevented from attending to any noise tokens.
```

代码实现就是这个方向。

## 7. Native RoPE

论文提出 Native RoPE，把 token 位置分成：

```text
T / H / W
```

- text tokens: only temporal axis T, H = W = 0；
- image tokens: temporal + spatial H/W；
- 8B/A3B 表格里的 head size:
  - T: 64
  - H: 32
  - W: 32
  - total head dim = 128

配置中：

```python
rope_base = 5000000.0
rope_theta_hw = 10000.0
```

论文 Stage 1 attention-fusion 也特别提到：

```text
temporal rope theta = 5,000,000
spatial rope theta = 10,000
```

视觉 embedding 自身也在 `NEOVisionEmbeddings` 里使用 2D rotary position embedding，将 x/y 位置编码到 patch embeddings。

## 8. Pixel-Space Flow Matching

### 8.1 论文公式

给 clean image:

```text
x in R^{3 x H x W}
```

采样噪声：

```text
epsilon ~ N(0, I)
```

动态噪声尺度：

```text
sigma_R(H,W) = sigma_0 * sqrt(N(H,W) / N0)
N(H,W) = H*W / 32^2
```

构造 flow matching noisy sample：

```text
z_t = t * x + (1 - t) * sigma_R * epsilon
```

模型直接预测 clean signal：

```text
x_hat_theta = model(z_t, t, condition)
```

再转成 velocity：

```text
v_theta = (x_hat_theta - z_t) / (1 - t)
v_star  = (x - z_t) / (1 - t)
L_Gen = MSE(v_theta, v_star)
```

### 8.2 代码对应

在 `prepare_image_gen_targets` 中：

```python
cur_noise = torch.randn_like(cur_pixel_values) * noise_scale
t = sigmoid(normal(P_mean, P_std))
cur_image_gen_z = t * cur_pixel_values + (1 - t) * cur_noise
image_gen_v = (image_gen_x - image_gen_z) / (1 - image_gen_t)
```

在 loss 处：

```python
image_gen_pred_x = self.fm_modules["fm_head"](image_gen_hidden_states)
image_gen_pred_v = (image_gen_pred_x - image_gen_z) / (1 - image_gen_t)
image_gen_loss = mse(image_gen_pred_v, image_gen_v)
```

这和论文的 `x-pred + v-loss` 完全一致。

## 9. Dynamic Noise Scale

论文动机：不同分辨率下 generation token 数不同。如果所有分辨率都用 unit Gaussian noise，则相同 timestep 下 SNR 分布不一致。

定义：

```text
N(H,W) = H*W / 32^2
sigma_R = sigma_0 * sqrt(N / N0)
```

论文训练表：

```text
sigma_0 = 1
N0 = 64
sigma in [1, 8] for N in [64, 4096]
```

代码：

```python
image_seq_len = cur_image_token_num // (merge_size**2)
base = float(self.noise_scale_base_image_seq_len)
scale = math.sqrt(image_seq_len / base)
noise_scale = scale * float(self.noise_scale)
```

公开脚本：

```bash
noise_scale_mode="resolution"
noise_scale_base_image_seq_len=64
noise_scale_max_value=8
add_noise_scale_embedding=true
```

如果开启 `add_noise_scale_embedding`，会把归一化后的 scale 编成 embedding，加到 timestep embedding 上：

```python
noise_scale_embedding = self.fm_modules["noise_scale_embedder"](image_gen_noise_scale)
timestep_embeddings = timestep_embeddings + noise_scale_embedding
```

## 10. Classifier-Free Guidance

训练时做 condition drop：

论文：

```text
drop text condition: 10%
drop both text and image condition: additional 10%
```

公开训练脚本：

```bash
cfg_txt_uncond_drop_prob=0.1
cfg_img_uncond_drop_prob=0
cfg_txtimg_uncond_drop_prob=0.1
cfg_is_uncond_drop_independent=false
```

推理时使用 text guidance 和 image-context guidance：

```text
gamma controls text guidance
gamma_img controls image-context guidance
```

论文经验：

```text
gamma = 4
gamma_img = 1
```

代码的生成函数里有 `cfg_scale`、`cfg_img_scale`、`cfg_norm` 等参数，用于对 predicted velocity 进行 CFG 组合。

## 11. Output Heads

### Text

```text
hidden states -> lm_head -> vocabulary logits
```

对应 autoregressive CE。

### Image

训练配置里 `fm_head_layers=2`，所以默认是浅 MLP：

```python
fm_head = nn.Sequential(
    nn.Linear(llm_hidden_size, 4096),
    nn.GELU(),
    nn.Linear(4096, output_dim),
)
```

`output_dim`：

```python
merge_size = int(1 / downsample_ratio) = 2
output_dim = 3 * (patch_size * merge_size) ** 2
           = 3 * 32^2
           = 3072
```

也就是每个 generation token 直接预测一个 32×32 RGB patch。

这就是论文说的：

```text
directly predicts pixel patches via MLP head
bypassing deep diffusion heads and VAE decoders
```

