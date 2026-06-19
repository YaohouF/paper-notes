# SenseNova-U1 Data Construction

## 1. 数据总览

SenseNova-U1 的数据组织按能力分为四大类：

```text
understanding data
text-to-image generation data
image editing data
interleaved image-text generation data
```

论文里给出的核心比例：

- Stage 3/4 unified training:
  ```text
  understanding : generation : editing : interleave
  = 0.33 : 0.37 : 0.24 : 0.06
  ```

这和 TUNA-2 类似，都是把理解、生成、编辑、交错生成混在一个 unified training 里；但 SenseNova-U1 的数据体系更强调 infographic、dense text rendering、spatial/agent/reasoning、interleaved narrative。

## 2. Understanding Data

### 2.1 Pre-training Stage

论文说 pre-training corpus 包含：

```text
large-scale web text
image-text pairs
interleaved multimodal documents
```

组织成四类：

```text
image-text pairs: 32%
captions: 17%
infographic understanding: 14%
pure text: 37%
```

清洗 pipeline：

1. cross-source deduplication；
2. content and safety filtering；
3. image quality filtering；
4. CLIP-ratio-balanced re-captioning。

这里的 CLIP-ratio-balanced re-captioning 可以理解为：不要只让数据里充满高 CLIP 相似但单一风格的 caption，也不要让低质量图文对污染训练，而是在图文对齐强度上做分布平衡。

### 2.2 Mid-training Stage

Mid-training 主要来自 internal SenseNova V6.5 datasets。

大类比例：

```text
General: 39.2%
Agent and Spatial: 22.3%
Knowledge Reasoning: 19.3%
Pure Text: 19.2%
```

General 再细分：

```text
general VQA: 26.6%
multi-turn dialogue: 26.4%
captioning: 20.3%
OCR: 18.6%
multi-image understanding: 8.2%
```

Knowledge Reasoning：

```text
knowledge-oriented: 12.0%
reasoning-oriented: 7.2%
```

Mid-training curation pipeline 三步：

### 2.2.1 Distribution-Balanced Sampling

目标：从大池子里抽出多样、不偏的样本。

步骤：

```text
CLIP visual embeddings
  -> K-means clusters
  -> uniform sampling across clusters
  -> attribute profiling
  -> stratified sampling by attribute tiers
```

直觉：

- K-means 保证视觉分布别被高频类别统治；
- attribute profiling 保证感知属性、语义属性的覆盖。

### 2.2.2 Prompt Augmentation

增强维度：

```text
semantic expression
format and structural constraints
role and scenario
task complexity
```

也就是不只是改写 prompt，还会提高 instruction complexity，比如：

- 要求特定输出格式；
- 加 persona/role；
- 加 compositional reasoning；
- 从简单识别变成多步推理。

增强后，答案统一 regenerate，以保持风格和质量一致。

### 2.2.3 Multi-Criteria Filtering

模型自动评分 QA pair：

```text
correctness verification
hallucination detection
instruction-following assessment
```

这对应：

- 答案是否和 ground truth 一致；
- 是否编造视觉上不存在的信息；
- 是否满足格式/角色/persona 等约束。

## 3. Understanding SFT Data

SFT corpus 按 capability-atomic dimensions 组织，也就是把能力拆得更细，便于控制配比。

论文给出的近似比例：

```text
spatial intelligence: ~15%
general multimodal understanding: ~13%
reasoning: ~12%
general NLP: ~11%
OCR and document analysis: ~11%
agentic function calling: ~10%
long-context conversation: ~8%
code: ~6%
multi-turn dialogue: ~4%
complex compositional understanding: ~4%
others: remaining
```

SFT 不是重新采集，而是从 mid-training candidate pool 精修：

### 3.1 Quality-oriented selection

按这些维度打分：

```text
visual fidelity
instruction clarity
response correctness
reasoning quality
safety
```

高分样本占比比 mid-training 更高。

### 3.2 Difficulty-oriented reconstruction

三种做法：

1. 将短样本合成长样本：
   ```text
   long-context
   multi-image
   multi-turn
   ```

2. 对 reasoning-intensive domains 做 rejection sampling，保留中等难度样本。

3. 重写 underspecified queries，加入明确约束：
   ```text
   output format
   stylistic attributes
   target granularity
   ```

## 4. Generation Data

生成数据包括 T2I、editing、interleaved。

统一过滤 pipeline：

```text
low-level filtering
deduplication
VLM-based captioning
quality-aware filtering
```

这套 pipeline 主要保证：

- 图像质量；
- 图文一致；
- caption 足够细；
- 数据不要重复；
- text-rich/infographic 样本不过滤掉。

## 5. Text-to-Image Data

T2I corpus 三大主域：

```text
Nature: ~40.5%
People: ~26.7%
Design: ~20.7%
```

同时强化：

```text
complex infographics
bilingual text rendering
posters
charts
cityscapes
```

这个数据选择和 SenseNova-U1 的定位强相关：它不只是追求普通 T2I 图像美学，还特别想做 information-dense visual communication：

- posters；
- slides；
- infographics；
- resumes；
- comics；
- charts；
- text-heavy images。

## 6. Image Editing Data

编辑数据主要来自 web-scale data。

内容分布：

```text
natural scenes: ~52.3%
human subjects: ~14.7%
remaining: infographic and synthetic edits
```

操作类型包括：

```text
subject addition/removal
background changes
color changes
identity transfer
motion manipulation
portrait editing
compositing
reasoning-driven transformations
```

特别重要的是 editing pair validation。

论文说会把 instruction 分解成：

```text
dynamic objectives: what should change
static constraints: what must remain unchanged
physical-consistency constraint: source image consistency
```

这比普通 edit dataset 更细，因为图像编辑容易出现两个问题：

- 该改的没改；
- 不该改的被破坏。

所以它要同时验证 dynamic change 和 static preservation。

## 7. Interleaved Data

目标：训练模型生成文本和图片交替出现的 coherent multimodal narratives。

四类：

```text
Video: ~19%
Lifestyle: ~44%
Infographics: ~29%
Reasoning: ~8%
```

Lifestyle 再细分：

```text
tutorials: 26%
daily-life scenarios: 14%
picture books: 4%
```

Interleaved 数据的检查维度：

```text
text quality
image quality
image-text consistency
trajectory-level correctness
```

Reasoning subset 包含 explicit chain-of-thought trace。这里的目的不是只生成单张图，而是生成一串“有先后关系”的图文，比如：

```text
step 1: text explanation
step 2: image
step 3: text continuation
step 4: image
...
```

## 8. 代码里的数据格式线索

公开训练数据相关目录：

```text
sensenova/SenseNova-U1-main/training/sensenovavl/data/
```

关键字段从 collator 和 model forward 可见：

```text
input_ids
labels
pixel_values
image_seq_lens
image_flags
image_con_flags
image_for_gen_flags
image_for_gen_loss_flags
is_image_duplicated_for_und_flags
grid_hw
```

这些字段说明每张图不只是“有没有图”，还要标注：

- 这张图是否用于 understanding；
- 是否作为 generation target；
- 是否作为 condition image；
- 是否参与 generation loss；
- 是否是为了 understanding duplicated 的 image。

## 9. CFG Condition Drop 数据处理

代码：

```text
sensenova/SenseNova-U1-main/training/sensenovavl/data/cfg_cond_drop_utils.py
sensenova/SenseNova-U1-main/training/sensenovavl/data/multimodal_dataset.py
```

T2I/editing/interleave 会按概率 drop text 或 text+image condition，用来训练 classifier-free guidance。

公开脚本默认：

```bash
cfg_txt_uncond_drop_prob=0.1
cfg_img_uncond_drop_prob=0
cfg_txtimg_uncond_drop_prob=0.1
cfg_is_uncond_drop_independent=false
```

这意味着：

- 一部分样本去掉文本条件；
- 一部分样本同时去掉文本和图像条件；
- 不单独做 image-only drop。

## 10. 数据构造和模型目标的对应关系

| 数据类型 | 输入 | 输出 | 训练 loss |
|---|---|---|---|
| pure text | text | next text token | CE |
| VQA / OCR / reasoning | image + text | text answer | CE |
| T2I | text condition + noisy target image tokens | target image patches | flow MSE |
| editing | source image + text instruction + noisy target image tokens | edited image patches | flow MSE |
| interleaved | previous text/images + noisy next image tokens | next text/image | CE + flow MSE |

## 11. 学习时的核心理解

SenseNova-U1 的数据不是简单“多模态大杂烩”。它的组织与模型结构是配套的：

- understanding data 让 clean stream 学视觉语义和推理；
- generation data 让 gen stream 学 pixel synthesis；
- editing data 让 clean image condition 和 noisy target image 发生关系；
- interleaved data 让文本生成和图片生成在同一上下文里交替发生；
- CFG drop 让同一个模型学会 conditional / image-only / unconditional generation。

