# Paper-Code Crosscheck

这一页只看论文主张和公开代码是否对得上。代码目录：

```text
/Users/yoho/Desktop/my_paper/hidream/HiDream-O1-Image-main
```

## 1. 论文说 no VAE，代码是否一致？

基本一致。公开推理代码没有 VAE encode/decode 路径，target image 是 raw pixel patches。

关键代码：

```python
self.patch_size = 32
self.x_embedder = BottleneckPatchEmbed(...)
self.final_layer2 = FinalLayer(...)
```

这说明模型直接把 RGB patches 投到 hidden space，再从 hidden state 投回 RGB patch。

## 2. 论文说 no disjoint text encoder，怎么理解？

README 说：

```text
One end-to-end model on raw pixels, no VAE, no disjoint text encoder.
```

代码里确实没有 FLUX/SD3 那种外接 T5/CLIP text encoder。文本输入通过 Qwen3-VL tokenizer 和 Qwen3-VL text model/backbone 处理。

但要注意：8B 版本初始化自 Qwen3-VL-8B-Instruct，所以它不是没有 text backbone，而是没有“与生成 backbone 分离的外部 text encoder”。

## 3. condition token 和 target generation token 是否同一路径？

论文高层写法会让人感觉所有图像都统一 patchify 到同一空间。公开代码更细：

| 图像类型 | 代码路径 |
|---|---|
| target noisy image | `vinputs -> x_embedder -> append to sequence` |
| source/reference condition image | Qwen3-VL visual encoder + image placeholders；部分任务还把 reference patches 拼进 `vinputs` |

所以 target generation path 是 direct pixel patches；condition path 仍然利用 Qwen3-VL 的视觉编码能力。

这个点很重要，因为它说明 HiDream 的 pixel-level unification 在 target generation 上最彻底；condition/reference image 处理更混合。

## 4. timestep special token 是否真的存在？

存在。代码里：

```python
self.tms_token_id = 151673
timestep_embedding = self.t_embedder1(timestep)
tms_mask = input_ids == self.tms_token_id
inputs_embeds[tms_mask] = timestep_embedding
```

所以 timestep 不是简单加到 image tokens 上，而是替换 sequence 里的特殊 token embedding。

## 5. Hybrid Unified Attention 是否真的实现？

是的。代码里 `token_types` 控制 token 的 attention behavior：

```text
0 = AR token
>0 = generation/full-attention token
```

T2I pipeline 中：

```python
token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3
token_types_bin = token_types > 0
```

非 flash attention 路径中，generation rows 的 causal mask 被清空：

```python
gen_positions = token_types[b].bool()
causal_mask[gen_positions, :] = 0
```

flash attention 路径中，代码注释说明是 two-pass：

```text
causal attention for AR tokens
full attention for all tokens
replace AR positions with causal outputs
```

这与论文描述一致。

## 6. 输出预测的是 noise 吗？

不是按传统 DDPM 那样直接预测 noise。代码里 `x_pred` 更像 clean image prediction：

```python
x_pred = outputs.x_pred
v = (x_pred - z) / sigma
model_output = -v
```

也就是说：

```text
model predicts x0
pipeline converts x0 prediction to velocity-like update
scheduler updates current noisy state
```

这和论文中 “maps output token back to clean image patch” 是一致的。

## 7. 是否有完整训练代码？

没有看到完整训练 pipeline。公开 repo 主要是：

- inference
- pipeline
- app/demo
- prompt agent
- scheduler
- modified Qwen3-VL model

训练中的数据加载、loss 组合、SFT/RLHF/GRPO、distillation 细节没有完全开源。

## 8. 和论文可能存在的表达差异

| 论文表达 | 代码观察 | 解释 |
|---|---|---|
| condition images projected via visual encoder, e.g. SigLip-2 | 代码中使用 Qwen3-VL visual model | 公开实现以 Qwen3-VL 组件为主，论文表述可能是高层/训练实现描述 |
| unified token space | target pixels direct patch embedding；condition image 经 visual encoder/placeholder | 统一发生在 shared hidden sequence，但输入路径并非完全同构 |
| no disjoint text encoder | 使用 Qwen3-VL text backbone | 指没有外部独立 T5/CLIP encoder，而不是没有语言模型 |
| full training recipe | repo 主要开源推理 | 训练细节需要依赖论文描述，不能从代码完全复现 |

## 9. 最值得记住的代码版 forward

```text
input_ids -> Qwen3-VL text embeddings
pixel_values/ref images -> visual features replace image placeholders
timestep -> TimestepEmbedder -> replace <tms_token>
vinputs/noisy target patches -> BottleneckPatchEmbed
concat text/condition/time/image embeddings
run Qwen3-VL decoder with hybrid attention
FinalLayer -> predicted clean RGB patches
pipeline converts x0 to velocity and scheduler step
```

