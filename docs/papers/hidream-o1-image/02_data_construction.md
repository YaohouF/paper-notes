# Data Construction

HiDream-O1-Image 的数据构造服务于一个目标：让同一个模型能处理 T2I、editing、subject personalization、graphic/text rendering、多面板图像等任务。

## 1. 数据来源

论文中的数据来源包括：

- public web corpora
- internal licensed data
- image-text pairs
- text-only corpora
- graphic-layout images，比如 slides、posters、documents、long-text images
- editing before/after pairs
- subject/IP personalization reference sets
- video-derived frames
- multi-panel images

这不是单纯的 captioned image dataset，而是一个围绕“统一生成任务”重新组织的数据池。

## 2. 去重：SSCD + clustering + Faiss

论文使用 SSCD features 做图像去重。流程大致是：

```text
images
  -> SSCD feature extraction
  -> sample 2M subset
  -> k-means into 16,000 clusters
  -> Faiss nearest-neighbor search
  -> remove near duplicates
```

论文说这个过程移除了大约 20% 的重复样本。

直觉上，这一步很关键：大规模 T2I 训练中，如果重复样本太多，模型容易记忆常见图像/水印/模板，对 benchmark 和实际泛化都不健康。

## 3. 过滤：质量、安全和任务一致性

论文提到多种过滤器：

| Filter | 目的 |
|---|---|
| NSFW classifier | 安全过滤 |
| aesthetic scoring | 视觉质量 |
| watermark detector | 去水印/模板污染 |
| VLM task consistency | 检查 editing/IP 数据是否真的符合任务 |
| Top-IQ score | 高质量样本筛选 |
| bytes-per-pixel JPEG artifact filter | 压缩伪影过滤 |

其中 VLM task consistency 对 editing 和 personalization 尤其重要。因为这类数据不是“图像和 caption 对不对”这么简单，而是：

```text
source image + instruction -> target image
```

是否真的保留了该保留的主体、改变了该改变的属性，需要更复杂的视觉-语言判断。

## 4. Prompt Construction

HiDream-O1-Image 不只是拿原始 alt-text 做 prompt。论文使用 Qwen3-VL 生成/增强 prompts。

不同数据类型的 prompt 构造重点不同：

| Data type | Prompt emphasis |
|---|---|
| 普通图文对 | 物体、场景、风格、关系 |
| graphic-layout | OCR text、排版、元素位置、布局 |
| editing pairs | source/target 差异、编辑指令 |
| personalization | identity-preserving subject description |
| multi-panel | 全局关系、panel-level 差异 |

这和模型最终能力高度相关：HiDream 在文字渲染、布局、subject preservation 上的表现，不只是模型结构带来的，也依赖这种 task-aware prompt/data construction。

## 5. Reasoning-Driven Prompt Agent 数据

论文和代码都强调 Prompt Agent。它的作用不是训练主模型时的普通 captioner，而是在推理前把用户输入重写成更明确的 prompt。

例如用户可能说：

```text
一个小猫在书店里看书，左边有招牌
```

Prompt Agent 会显式展开：

```text
spatial layout, subject attributes, physical logic, contextual relationships,
text rendering details, lighting, composition...
```

论文 post-training 里还说 SFT 阶段包含 high-quality reasoning trajectories，用来 fine-tune Prompt Agent，使它更稳定地产生结构化、无歧义 prompt。

## 6. 数据构造和模型结构的配合

HiDream 的数据不是“多任务拼盘”，而是和 unified tokenization 对齐：

```text
T2I:
  text prompt + noisy target image

Editing:
  instruction + source image condition + noisy target image

Subject personalization:
  subject reference images + prompt + noisy target image

MMU / LM:
  text / multimodal understanding tokens
```

这些任务最终都被整理成：

```text
text tokens + condition tokens + generation tokens
```

这就是为什么论文把数据部分和 “in-context generation” 联系得很紧：不同任务不是换模型头，而是换上下文里的条件 token。

