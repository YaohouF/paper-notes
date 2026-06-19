# SenseNova-U1 Training Recipe

## 1. 总体训练路线

论文训练路线：

```text
Stage 1: Understanding Warmup
Stage 2: Generation Pre-Training
Stage 3: Unified Mid-Training
Stage 4: Unified SFT
Stage 5: Post Training for T2I Generation
```

其中 Stage 2 又分三段：

```text
Phase I: low/mid resolution T2I
Phase II: high resolution T2I
Phase III: editing + reasoning + interleaved added
```

总体思想：

1. 先把 understanding backbone 处理好。
2. 冻结 understanding branch，单独训练 generation branch，让像素生成先稳定。
3. 再 joint training，把理解和生成统一起来。
4. 最后 SFT 和 RL/distillation 提升指令跟随、画质和速度。

## 2. Joint Objective

总 loss：

```text
L_total = lambda_1 * L_Und + lambda_2 * L_Gen
```

### Understanding loss

标准 next-token prediction：

```text
L_Und = - 1/N * sum_n log p_theta(x_n | x_<n, c)
```

也就是 VQA、OCR、caption、reasoning 等最后都是文本 CE。

### Generation loss

Pixel-space flow matching：

```text
z_t = t*x + (1-t)*sigma_R*epsilon

x_hat_theta = model(z_t, t, condition)

v_theta = (x_hat_theta - z_t) / (1-t)
v_star  = (x - z_t) / (1-t)

L_Gen = MSE(v_theta, v_star)
```

代码中对每个 32×32 patch 做 MSE，再按 token 平均。

## 3. Stage 1: Understanding Warmup

论文说从 pretrained NEO 初始化。

Stage 1 包含两个子阶段：

### 3.1 Attention-Fusion Phase

目标：简化 NEO 原始 QK projections / normalization。

论文说将 temporal/spatial axes 上的 QK projection 和 norm 统一成一套，QK 参数量减半，同时保留 Native RoPE 多轴结构。

配置：

```text
temporal rope theta = 5,000,000
spatial rope theta = 10,000
```

训练策略：

```text
freeze rest of network
train only attention layers:
  Q, K, V, O projections
  QK normalization
until model recovers pre-fusion accuracy
```

### 3.2 Full-Model Continuation Phase

然后 unfreeze 整个 understanding branch，在 updated mid-training corpus 上继续训练。

论文超参：

```text
peak LR = 2e-5
optimizer = AdamW(beta1=0.9, beta2=0.95, eps=1e-8)
weight decay = 0
grad clip = 1.0
steps = 120K
seq length = 32768
und resolution = 256^2 -> 4096^2
loss weight CE:MSE = 1:0
training tokens = 0.75T
data sampling: understanding only
```

## 4. Stage 2: Generation Pre-Training

Stage 2 冻结 understanding branch，只训练 generation branch。

核心原因：

```text
understanding branch already has semantic/reasoning ability
generation branch starts less stable
freeze understanding branch avoids damaging perception/reasoning
train generation branch to map noisy pixel tokens -> clean pixel patches
```

### 4.1 Phase I

目标：建立基本 T2I 生成能力。

论文超参：

```text
data: text-to-image
resolution: 256^2 -> 512^2
larger than 512 resized to 512, keeping aspect ratio
steps = 120K
LR = 2e-4 constant
warmup = 2000
EMA = 0.9999
seq length = 8192
loss weight CE:MSE = 0:1
training tokens = 0.25T
```

### 4.2 Phase II

目标：高分辨率生成。

论文超参：

```text
data: T2I samples >= 512^2
resolution: 512^2 -> 2048^2
larger than 2048 resized to 2048
steps = 60K
LR = 1e-4 constant
warmup = 2000
EMA = 0.9999
seq length = 16384
loss weight CE:MSE = 0:1
training tokens = 0.25T
```

### 4.3 Phase III

目标：扩展到 editing、reasoning、interleaved。

论文超参：

```text
data:
  generation: 0.56
  editing: 0.37
  interleave: 0.07
steps = 120K
LR = 1e-4 -> 2e-5 cosine decay
warmup = 2000
EMA = 0.9999
resolution: 512^2 -> 2048^2
seq length = 16384
loss weight CE:MSE = 0:1
training tokens = 0.88T
```

A3B 额外：

```text
generation branch MoE balance loss coefficient = 5e-3
```

## 5. Stage 3: Unified Mid-Training

Stage 3 开始 full model joint training。

数据比例：

```text
understanding : generation : editing : interleave
= 0.33 : 0.37 : 0.24 : 0.06
```

论文超参：

```text
steps = 84K
LR = 2e-5 constant
warmup = 2000
EMA = 0.999
seq length = 32768
und resolution = 256^2 -> 4096^2
gen resolution = 512^2 -> 2048^2
loss weight CE:MSE = 0.1:1
training tokens = 1.19T
```

A3B：

```text
MoE balance loss coefficient = 1e-3
applied to generation and understanding branches
```

为什么 CE 权重只有 0.1？

直觉上 generation loss 是高维 pixel patch MSE，scale 和 CE 不同。联合训练时如果 CE 权重过大，可能让模型更偏 language/understanding；如果 MSE 权重过大，可能损害语言能力。论文选择 `0.1:1` 来降低 objective interference。

## 6. Stage 4: Unified SFT

目标：instruction alignment 和 task-specific performance。

数据覆盖：

```text
multimodal dialogue
image generation
editing
interleaved data
```

使用和 Stage 3 相同数据大类比例：

```text
understanding : generation : editing : interleave
= 0.33 : 0.37 : 0.24 : 0.06
```

论文超参：

```text
steps = 9K
LR = 2e-5 -> 0 cosine decay
warmup = 100
EMA = 0.999
seq length = 32768
loss weight CE:MSE = 0.1:1
training tokens = 0.13T
```

## 7. Stage 5: T2I Post Training

论文说最终模型还经过 initial round of T2I RL training，并使用 distillation。

### 7.1 Flow-GRPO

用于提升生成质量。

整体思路和 POCA 那类 diffusion RL 很像：

```text
sample image
compute reward
optimize denoising/flow policy with GRPO-style objective
```

论文列出三个 reward：

1. Text Rendering Reward
   - 用 PaddleOCR 抽取生成图中的文字。
   - 与 ground truth text 做 Counter IoU。

公式：

```text
r_ocr = |C(T_hat) ∩ C(T*)| / |C(T_hat) ∪ C(T*)|
```

2. 其他 reward 论文后文还有，但这份第一版笔记先聚焦主训练 recipe。后续可继续单独展开 Stage 5。

### 7.2 Dynamic Resolution Warmup

候选 resolution 来自：

```text
aspect ratios: {1:1, 16:9, 9:16, 3:2, 2:3}
target areas: {1536^2, 2048^2}
```

每个 resolution 有 difficulty score `d_i`，随训练 epoch 逐步打开更难分辨率。

论文公式：

```text
p_hat_i = p_i * clamp((min(e/E_warm, 1) - d_i)/delta + 1, 0, 1)
delta = 0.3
```

含义：先采样容易分辨率，逐渐加入更大、更极端 aspect ratio 的图。

### 7.3 Distribution Matching Distillation

用于提升效率，让模型用更少 step 生成。

README 中有 8-step preview / LoRA / infographic 8-step 模型，这对应 post-training 后的高效推理路线。

## 8. Time Sampling

论文训练表：

```text
Time shift:
mu = -0.8
sigma = 0.8
in logit-normal t-sampler
```

代码：

```python
u = Normal(0,1) * P_std + P_mean
t = sigmoid(u)
t = self._apply_time_schedule(t, image_seq_len)
```

公开脚本：

```bash
P_mean=-0.8
P_std=0.8
time_schedule="standard"
time_shift_type="exponential"
base_shift=0.5
max_shift=1.15
```

## 9. Dynamic Noise Scale

论文：

```text
sigma_R = sigma_0 * sqrt(N / N0)
sigma_0 = 1
N0 = 64
sigma in [1, 8] for N in [64, 4096]
```

公开脚本：

```bash
noise_scale_mode="resolution"
noise_scale_base_image_seq_len=64
noise_scale_max_value=8
add_noise_scale_embedding=true
```

代码：

```python
image_seq_len = cur_image_token_num // merge_size**2
scale = sqrt(image_seq_len / base)
noise_scale = scale * self.noise_scale
```

## 10. 公开训练脚本默认值

8B smoke-test:

```bash
lr=2e-4
lr_scheduler_type=constant
total_steps=200000
init_steps=2000
seq_len=28672
max_pixels=2048*2048
max_pixels_gen=512*512
ce_loss_weight=0.1
enable_und_loss=true
ema_decay=0.9999
```

A3B smoke-test:

```bash
lr=2e-4
tp_size=2
balance_moe_loss_coef=0.005
max_pixels=512*512
max_pixels_gen=512*512
mot_random_init=false
ce_loss_weight=0.1
```

这些脚本是公开训练/微调模板，不等于论文完整训练 schedule。

## 11. 训练时哪些模块参与

### Understanding-only

```text
vision_model
language_model understanding branch
lm_head
CE loss
```

### Generation-only

```text
fm_modules.vision_model_mot_gen
language_model mot_gen branch
fm_modules.timestep_embedder
fm_modules.noise_scale_embedder
fm_modules.fm_head
MSE velocity loss
```

### Unified

两者同时存在：

```text
clean image/text tokens -> understanding branch
noisy generation tokens -> mot_gen branch
same sequence / native attention
CE + MSE weighted sum
```

## 12. 面试/答辩式总结

```text
SenseNova-U1 uses a progressive training pipeline.
It first warms up the understanding backbone from NEO, then freezes it and trains the pixel-space generation branch with flow matching. After the generation branch becomes stable, both streams are jointly trained on understanding, T2I, editing, and interleaved data with a CE:MSE ratio of 0.1:1. Finally, unified SFT improves instruction following, and T2I post-training with Flow-GRPO and distillation improves image quality and sampling efficiency.
```

