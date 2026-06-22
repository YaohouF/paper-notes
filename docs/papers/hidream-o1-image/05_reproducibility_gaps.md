# Reproducibility Gaps

HiDream-O1-Image 的公开材料足够我们理解推理结构，但还不足以完整复现训练。

## 已经比较清楚的部分

| Part | Status |
|---|---|
| 8B model high-level architecture | 论文 + 代码可核对 |
| pixel patch size | 代码明确是 32 |
| timestep token injection | 代码明确 |
| hybrid attention mask | 代码明确 |
| sampling from predicted clean image patches | 代码明确 |
| Prompt Agent 推理 | 代码和 README 明确 |
| Full/Dev 推理 steps | README 明确 |

## 缺口 1：训练代码没有完整开源

公开 repo 没有提供完整 training scripts。以下内容无法只靠代码复现：

- progressive pre-training data mixture
- optimizer / learning rate / batch size
- T2I / LM / MMU task mixing ratio
- SFT 数据构造细节
- GRPO/RLHF reward model 训练与权重
- distillation teacher/student 训练细节

论文描述了路线，但缺少可执行 recipe。

## 缺口 2：数据集不可复现

论文数据来自 public web corpora 和 internal licensed data。很多关键数据不可直接获得：

- internal licensed images
- high-quality editing pairs
- personalization reference sets
- multi-panel / video-derived constructed data
- reward model 标注数据

此外，Qwen3-VL prompt construction 和 Top-IQ/filtering 的具体阈值也没有完全公开。

## 缺口 3：200B+ Pro 版本不可直接复现

论文报告了 200B+ Pro 模型，但公开 repo 主要围绕 8B Full/Dev 版本。即使架构思路一致，200B+ 版本的：

- model config
- training budget
- parallelism strategy
- checkpoint
- post-training details

都不是普通读者可复现的。

## 缺口 4：condition image 路径和论文表述有细节差异

论文中 condition tokens 被描述为视觉编码器投影到 shared token space。代码里 condition/reference image 使用 Qwen3-VL visual path 和 image placeholder 机制，同时 target image 是 direct pixel patches。

这不是矛盾，但会影响我们复现时的设计判断：

```text
Are all images raw-pixel patchified?
No. Target generation patches are raw pixel patches;
reference/condition images may use the pretrained multimodal visual path.
```

## 缺口 5：Prompt Agent 是系统能力的一部分

HiDream 的生成质量很大程度依赖 Prompt Agent。论文里说它基于 Gemma，并在 SFT 阶段包含 reasoning trajectories。

如果复现时只训练/使用主生成模型，而没有高质量 prompt refiner，实际效果可能明显下降，尤其是：

- complex layout
- physical relation
- dense text rendering
- editing instruction disambiguation

## 复现优先级建议

如果我们以后想做一个 mini reproduction，优先级可以是：

```text
1. 用 Qwen3-VL / Qwen2.5-VL 初始化 decoder backbone
2. 实现 32x32 raw pixel patch embedding + final RGB patch head
3. 实现 timestep special token
4. 实现 AR/full hybrid attention mask
5. 先只做 512x512 T2I flow matching
6. 再加入 editing/reference image condition
```

不要一开始追 200B+、RLHF 或 Prompt Agent。那些是系统工程后期放大器，不是最小可验证核心。

