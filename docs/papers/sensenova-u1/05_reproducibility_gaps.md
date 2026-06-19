# SenseNova-U1 Reproducibility Gaps

## 1. Full training data is not public

论文训练依赖大量 internal SenseNova V6.5 datasets 和 web-scale curated data。

公开代码提供数据 loader 和 sample/meta format，但没有完整数据。

影响：

```text
cannot reproduce reported performance from scratch
can fine-tune / smoke-test / inspect architecture
```

## 2. Public training scripts are not exact paper schedules

论文 Stage 3/4：

```text
LR = 2e-5
seq length = 32768
resolution up to 2048^2 for gen
```

公开 `training/shell/train_u1/8B.sh`：

```bash
lr=2e-4
seq_len=28672
max_pixels_gen=512*512
total_steps=200000
```

这更像示例 fine-tuning/smoke-test launcher。

## 3. Stage 1 attention-fusion 细节不完整

论文描述：

```text
freeze rest
train Q/K/V/O and QK norm
recover pre-fusion accuracy
```

但公开代码里没有单独提供完整 attention-fusion training config、数据 mixture、恢复标准。

## 4. A3B generation expert count需要从 checkpoint config确认

论文表格写：

```text
understanding experts = 128
generation experts = 32
top-k = 8
```

公开 config class 支持：

```python
gen_num_experts
gen_num_experts_per_tok
gen_moe_intermediate_size
```

但 `sensenovau1_a3b_mot_sft.py` 中显式写了:

```python
num_experts = 128
moe_split_size = 8
```

未在该文件中直接看到 `gen_num_experts=32`。可能在 released HF config/checkpoint 中包含，或由 loading path 注入。

需要进一步核查已下载 checkpoint 的 `config.json`。

## 5. Stage 5 post-training 只在论文中高层描述

论文提到：

```text
Flow-GRPO
Distribution Matching Distillation
Dynamic Resolution Warmup
reward design
```

但公开代码主要是 training/inference/evaluation，不一定包含完整 RL/DMD 训练实现。

因此 Stage 5 可学习方法思想，但完整复现需要额外代码或配置。

## 6. Inference code 和 training code 有两套模型实现

推理代码：

```text
src/sensenova_u1/models/neo_unify/
```

训练代码：

```text
training/sensenovavl/model/
training/sensenovalm/model/
```

两套实现语义一致，但类名、config 字段、forward 组织不完全一样。读代码时要避免把 inference-only convenience path 当作 training forward。

## 7. Attention mask 很复杂

论文一句话概括：

```text
clean tokens cannot attend to noise tokens
noise tokens can attend to clean context and same noisy image block
```

代码还有：

- sequence packing；
- ISP tensor parallel；
- duplicated image boundary；
- modality indicators；
- image_seq_lens alignment；
- document ids；
- padding to block size。

如果要完全复现 attention 行为，需要写小 toy batch 单元测试可视化 mask。

## 8. Dynamic resolution and native-resolution packing依赖数据预处理

模型 forward 依赖：

```text
grid_hw
image_seq_lens
image_flags
image_for_gen_flags
is_image_duplicated_for_und_flags
```

这些字段来自 dataset/collator。仅看模型代码不能完全理解样本是如何拼进 sequence 的。

后续若要深入，应该单独做：

```text
one T2I sample -> input_ids/image flags/grid_hw/type_ids visualization
one editing sample -> source image + target image flags
one interleaved sample -> multiple image spans
```

## 9. Loss weighting和token weighting细节可继续深挖

论文给 `CE:MSE = 0.1:1`，代码还有：

```python
ce_loss_weight
image_gen_loss_coef
type_ids
image_gen_loss_t2i/editing/interleave logs
global_all_reduce_loss
```

不同任务是否有额外 per-token/per-type weight，第一版笔记只确认了主干。

## 10. 最值得继续深挖的几个问题

1. `pack_two_branch_sequence` 如何把 generation tokens 放到 sequence 前面。
2. Qwen3 attention projection 是否也完全分 generation/understanding 两套，还是主要 norm/FFN/MoE 分离。
3. Released HF config 中 A3B 的 `gen_num_experts` 是否为 32。
4. Interleaved generation 的数据样本如何构造 labels 和 `image_for_gen_flags`。
5. Stage 5 Flow-GRPO 是否有独立代码发布。
6. 8-step distillation LoRA 的训练过程是否在代码中公开。

