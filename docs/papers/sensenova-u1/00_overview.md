# SenseNova-U1 论文与代码阅读总览

## 材料位置

- 论文 LaTeX：`sensenova/`
- 技术报告 PDF：`sensenova/SenseNova-U1-main/docs/pdf/SenseNOVA_U1.pdf`
- 发布代码：`sensenova/SenseNova-U1-main/`
- 论文标题：*SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture*

## 推荐阅读顺序

1. `00_overview.md`：整体问题、核心贡献、和 TUNA-2 的直觉对照。
2. `01_model_architecture.md`：模型构造，视觉接口，MoT，attention，Native RoPE，flow head。
3. `02_data_construction.md`：理解数据、生成数据、编辑数据、interleaved 数据如何组织。
4. `03_training_recipe.md`：Stage 1 到 Stage 5 的训练路线和超参。
5. `04_paper_code_crosscheck.md`：论文说法和公开代码逐项对照。
6. `05_reproducibility_gaps.md`：复现缺口、公开代码和论文训练之间的差异。

## 一句话讲清楚

SenseNova-U1 是一个 native unified multimodal model：它试图用一个端到端架构同时做视觉理解、文本生成、图像生成、编辑和图文交错生成。它不使用传统 pretrained vision encoder，也不使用 VAE latent decoder，而是把图片直接切成 pixel patch tokens，与文本 token 一起送入一个 Mixture-of-Transformers backbone。

更口语化：

```text
pixels + words
  -> native visual/text tokens
  -> unified MoT backbone
  -> LM head for text
  -> pixel flow head for images
```

## 和 TUNA-2 的最大相似点

两者都在做一件“很激进”的事：

```text
remove visual encoder
remove VAE
operate in pixel space
train understanding + generation together
```

也就是不再让图片先经过 CLIP/SigLIP 这种语义 encoder，也不再让图像生成发生在 VAE latent space，而是直接从 pixel patch 层面进入 unified model。

## 和 TUNA-2 的最大区别

TUNA-2 更像：

```text
pixel patch embedding
  -> Qwen decoder
  -> LM head / flow head
```

SenseNova-U1 更像：

```text
clean image/text pathway + noisy generation pathway
  -> Native Mixture-of-Transformers
  -> branch-routed norms / FFNs / MoE experts
  -> shared attention-style interaction
```

它的关键不是简单把图像 token 塞进 Qwen，而是把 understanding stream 和 generation stream 在每层 Transformer 里做参数解耦：

- clean understanding tokens 走 understanding projections/norms/FFN 或 MoE experts；
- noisy image generation tokens 走 generation-side projections/norms/FFN 或 MoE experts；
- attention 仍然在统一序列中建模上下文关系。

这就是论文强调的 Native MoT。

## 主模型变体

论文和 README 主要提到两个 open-source lite 规模：

### SenseNova-U1-8B-MoT

- Dense 版本。
- 论文表格：42 layers，hidden size 4096，Q/KV heads 32/8。
- 代码配置：`36 + EXTRA_NUM_LAYER(6) = 42` layers。
- 参数解释：不是总参数 8B，而是 understanding transformer 约 8B，generation transformer 约 8B。
- README 的 parameter breakdown 显示：
  - total params: 17.552B
  - understanding transformer: 8.121B
  - generation transformer: 8.186B
  - shared text embedding/lm_head: 1.245B

### SenseNova-U1-A3B-MoT

- MoE 版本。
- 论文表格：48 layers，hidden size 2048，Q/KV heads 32/4。
- understanding stream：128 experts，总约 30B，top-8 active。
- generation stream：32 experts，总约 8.2B，top-8 active。
- A3B 指 active parameters 约 3B，而不是总参数 3B。

## Unified Forward Path 概念图

```text
text:
  words
  -> Qwen3 tokenizer
  -> token IDs
  -> shared text embedding table

understanding image:
  RGB image patches
  -> NEOVisionModel / patch encoder
  -> clean visual tokens

generation image:
  clean image x, noise eps, timestep t
  -> z_t = t*x + (1-t)*sigma_R*eps
  -> NEOVisionModel_mot_gen / patch encoder
  -> noisy visual tokens + time/noise-scale embedding

unified sequence:
  text tokens + clean visual tokens + noisy visual tokens
  -> Qwen3 MoT backbone

outputs:
  text hidden states -> lm_head -> next-token CE
  generation hidden states -> fm_head -> clean pixel patches -> velocity MSE
```

## 训练路线总览

论文将训练分成：

1. Stage 1: Understanding Warmup
   - 从 NEO 初始化。
   - attention-fusion phase。
   - full-model continuation。

2. Stage 2: Generation Pre-Training
   - 冻结 understanding branch。
   - 训练 generation branch 做 T2I pixel-space flow matching。
   - 三个 phase：低分辨率、多分辨率高分辨率、加入 editing/reasoning/interleaved。

3. Stage 3: Unified Mid-Training
   - understanding + generation 两条 branch 联合训练。
   - 数据比例：understanding : generation : editing : interleave = `0.33 : 0.37 : 0.24 : 0.06`。
   - loss weight：CE : MSE = `0.1 : 1`。

4. Stage 4: Unified SFT
   - 继续用高质量 instruction data 联合微调。
   - 同样 CE : MSE = `0.1 : 1`。

5. Stage 5: Post Training for T2I
   - Flow-GRPO 强化学习提升生成质量。
   - DMD / distribution matching distillation 提升采样效率。

## 目前最重要的学习抓手

这篇论文不是只要记“用了 Qwen3”就行。真正值得吃透的是：

1. 视觉接口为什么叫 near-lossless。
2. 32×32 patch token 是怎么从代码里的 `patch_size=16` 和 `downsample_ratio=0.5` 得到的。
3. MoT 到底是怎样在每层里把 generation tokens 和 understanding tokens 路由到不同参数。
4. clean image tokens 和 noisy image tokens 的 attention 规则。
5. pixel-space flow matching 的 `x-pred + v-loss`。
6. dynamic noise scale 为什么依赖 resolution。
7. Stage 2 先冻结 understanding branch 的训练动机。
8. Stage 3/4 为什么 CE loss 权重只有 0.1，而 image MSE 是 1.0。

