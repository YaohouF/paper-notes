# Training Recipe

HiDream-O1-Image 的训练路线可以分成三块：

1. Progressive generalist pre-training。
2. Post-training：SFT + RLHF/GRPO。
3. Distillation：Full model 到 Dev model。

## 1. Overall objective

论文说目标函数由 flow matching image prediction 和 perceptual constraints 组成：

```text
flow matching loss
  + LPIPS loss
  + perceptual DINO loss
```

核心生成过程使用：

```math
x_t = t x + (1 - t)\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
```

模型从 `x_t` 和条件上下文中预测 clean image patch / reconstruction。直觉上：

```text
input: noisy pixel patches + prompt/context
output: clean pixel patch estimate
```

perceptual losses 的目的，是缓解 pixel-space regression 在长程语义和感知质量上的弱点。

## 2. Progressive Generalist Pre-training

论文采用三阶段渐进训练。resolution 逐步升高，数据也从 coarse 到 fine。

### Stage I: Foundational Alignment, 512×512

任务包括：

- T2I：text-to-image generation
- LM：language modeling
- MMU：multimodal understanding

数据包括 image-text pairs 和 text-only corpora。

这一阶段的目的不是只学画图，而是：

```text
native pixel patches <-> linguistic concepts
```

同时保留语言能力。论文强调使用 512×512 和大 batch，使训练可以扩展到 billions of image-text pairs。

### Stage II: Generalist In-Context Learning, 1024×1024

第二阶段把分辨率提高到 1024×1024，并扩展任务：

- T2I
- LM
- MMU
- image editing
- subject-driven personalization
- other in-context generation tasks

这一阶段是 HiDream 从“会根据文本生成图像”走向“能在上下文里做编辑/个性化生成”的关键。

可以把它理解成：

```text
condition images + instruction + noisy target image
  -> target clean image
```

这里的 condition image 就是模型在 unified sequence 里做 in-context visual reasoning 的依据。

### Stage III: High-Fidelity Refinement, 2048×2048+

第三阶段只使用 ultra-high-resolution subset，图像分辨率超过 2048×2048。

目标更集中：

```text
fine-grained details
perceptual quality
ultra-high-res synthesis
```

这个阶段不像 Stage II 那么强调任务扩展，而是强调高分辨率细节和观感。

## 3. Post-training

pre-training 后，论文继续做两阶段 post-training。

### Stage I: SFT

SFT 数据是 several hundred thousand high-quality samples，强调：

- aesthetics
- photorealism
- compositional coherence
- lighting consistency
- stylistic fidelity
- prompt reasoning trajectories

这一步同时服务主生成模型和 Prompt Agent。论文还提到一个训练细节：SFT 时把 pre-training 的 Logit-Normal timestep sampling 换成 uniform sampling。

原因是 uniform sampling 能更均衡覆盖 timesteps，尤其加强 late-stage denoising steps：

```text
late denoising -> fine details / texture / final visual quality
```

### Stage II: RLHF / GRPO

论文使用 GRPO 进一步对齐人类偏好。reward signals 包括：

- OCR accuracy
- aesthetic assessment
- instruction-following fidelity
- reasoning quality

这些 reward 共同构成 composite advantage function。

改善目标包括：

```text
photorealism
aesthetic quality
text rendering accuracy
semantic consistency
logical reasoning
artifact suppression
```

## 4. Distillation

官方 repo 中可见两个主要 8B 版本：

| Version | Inference steps | 用途 |
|---|---:|---|
| Full | 50 | 更高质量 |
| Dev | 28 | 蒸馏后，更快 |

论文提到 distillation 使用：

- DMD objective
- standard diffusion loss
- adversarial loss
- discriminator guided by multi-level features from frozen teacher backbone

可以理解为：用 Full model 作为 teacher，把多步生成能力压缩到更少采样步数，同时通过 adversarial/perceptual 信号尽量保住细节和质感。

## 5. 推理 recipe 中值得注意的代码细节

README 里给出的默认推理设置：

| Model type | Steps | Guidance scale | Shift | Scheduler |
|---|---:|---:|---:|---|
| full | 50 | 5.0 | 3.0 | default |
| dev | 28 | 0.0 | 1.0 | flash scheduler |

editing 任务中，如果使用 Dev model，默认 scheduler 是 `flow_match`。这说明公开推理 recipe 对不同任务/模型版本做了不少经验性调参。

## 6. 训练 recipe 的学习重点

HiDream 的训练不是单一 diffusion pretraining，而是四条能力一起堆：

```text
language ability
multimodal understanding
pixel-space image synthesis
in-context visual conditioning
```

这也是为什么它选择 Qwen3-VL 初始化：如果完全从零开始，同时获得语言、多模态理解和高质量 pixel synthesis，训练成本会极高。

