# Paper Notes Site Maintenance Guide

这个文件是给后续维护本读书笔记站点的 agent / collaborator 使用的快速上手说明。目标是：新开一个项目或新开一个 Codex session 时，可以马上理解站点结构、写作约定、加新论文流程、构建和部署方式。

## 1. 项目位置和线上地址

本地仓库：

```text
/Users/yoho/Desktop/my_paper/paper_notes_site
```

GitHub 仓库：

```text
YaohouF/paper-notes
```

GitHub Pages：

```text
https://YaohouF.github.io/paper-notes/
```

站点框架：

```text
MkDocs + Material for MkDocs
```

主要配置文件：

```text
mkdocs.yml
```

## 2. 当前站点结构

```text
docs/
  index.md

  papers/
    hidream-o1-image/
    sensenova-u1/
    tuna-2/

    class-to-image/
      fd-loss/

  concepts/
    index.md
    native-unified-model.md
    mixture-of-transformers.md
    pixel-space-flow-matching.md
    native-rope.md
    dynamic-noise-scale.md

  comparisons/
    index.md
    tuna-2-vs-sensenova-u1.md
    tuna-2-vs-sensenova-u1-vs-hidream.md

  projects/
    dinov3-jit-generation/
```

目前有两条主要论文线：

```text
1. Multimodal / Text-to-Image
   - TUNA-2
   - SenseNova-U1
   - HiDream-O1-Image

2. Class-to-Image
   - FD-loss

3. Ongoing Research Projects
   - DINOv3 + JiT Generation
```

## 3. 写作语言和风格

默认写作语言：

```text
中文解释为主，公式和代码块保持英文。
```

写作目标：

- 不是简单摘要论文，而是帮助用户“细致学习和理解”。
- 尽量把论文主张、代码实现、训练 recipe、可复现性拆开。
- 明确区分：
  - 论文明确写了什么；
  - 代码里实际实现了什么；
  - 我们根据论文/代码作出的推断；
  - 公开材料无法确认的缺口。

语气：

- 可以直接、清晰、有教学感。
- 少用空泛评价，多给结构化解释。
- 遇到容易混淆的概念，要主动写“常见误解”。

## 4. 每篇论文的标准页面结构

对于 multimodal / text-to-image 论文，默认使用：

```text
docs/papers/<paper-slug>/
  index.md
  00_overview.md
  01_model_architecture.md
  02_data_construction.md
  03_training_recipe.md
  04_paper_code_crosscheck.md
  05_reproducibility_gaps.md
```

对于 class-to-image 论文，放在：

```text
docs/papers/class-to-image/<paper-slug>/
```

页面名称可以根据论文类型调整。例如 FD-loss 是训练目标论文，不是模型结构论文，所以使用：

```text
index.md
00_overview.md
01_method_and_loss.md
02_data_and_task_setup.md
03_training_recipe.md
04_paper_code_crosscheck.md
05_reproducibility_gaps.md
```

原则：

```text
如果论文主要改模型，就重点写 architecture。
如果论文主要改训练方法，就重点写 method/loss/training recipe。
如果论文主要改数据，就重点写 data/task/evaluation setup。
```

不要机械套模板，要根据论文贡献调整章节标题。

## 5. 单篇论文分析流程

推荐流程：

```text
1. 定位论文材料
   - PDF / LaTeX source / markdown
   - 代码仓库
   - README / scripts / checkpoints

2. 先读论文主线
   - abstract
   - introduction
   - method
   - training / experiments
   - appendix implementation details

3. 再读代码
   - README
   - model definition
   - training loop
   - loss implementation
   - data/eval scripts
   - config / launch scripts

4. 做 paper-code crosscheck
   - 哪些论文主张在代码里能确认
   - 哪些只在论文中有，代码没有
   - 哪些代码实现比论文表述更细

5. 写入站点
   - 新增 docs/papers/... 页面
   - 更新 docs/index.md
   - 更新 mkdocs.yml nav
   - 必要时新增 concepts 或 comparisons

6. 构建验证
   - python3 -m mkdocs build --strict

7. commit / push
   - git status --short
   - git add ...
   - git commit -m "..."
   - git push
```

## 6. 代码核对时的重点

不同类型论文的核对重点不同。

### 模型结构论文

优先找：

```text
model.py / modeling_*.py
transformer block
attention mask
token embedding
positional embedding / RoPE
timestep embedding
output heads
forward path
```

要回答：

- backbone 是什么；
- 是否 from scratch；
- text/image/noise tokens 如何进入模型；
- attention 如何交互；
- 输出预测什么；
- inference sampling 如何从模型输出得到下一步。

### 训练方法论文

优先找：

```text
loss implementation
training loop
optimizer / scheduler
batch construction
EMA / queue / memory bank
distributed all-gather
configs and scripts
```

要回答：

- loss 是 scalar 还是 per-token/per-pixel；
- 梯度通过哪些路径回传；
- 哪些张量 detach；
- 是否需要 teacher/discriminator/reward model；
- 训练阶段是 pretraining、fine-tuning 还是 post-training。

### 数据/benchmark 论文

优先找：

```text
dataset construction scripts
filtering
annotation
prompt/caption generation
evaluation protocol
metrics
```

要回答：

- 数据来源；
- 样本如何筛选；
- label/prompt/condition 如何构造；
- evaluation 是否公平可复现。

## 7. MkDocs 导航维护

每次新增页面后，都要更新：

```text
mkdocs.yml
```

当前 nav 大致结构：

```yaml
nav:
  - Home: index.md
  - Papers:
      - Multimodal / Text-to-Image:
          - ...
      - Class-to-Image:
          - ...
  - Projects:
      - ...
  - Concepts:
      - ...
  - Comparisons:
      - ...
```

注意：

- 新增论文必须挂到 `nav`，否则页面虽然存在但导航不可见。
- 不要随便移动旧论文路径，避免已有 GitHub Pages 链接失效。
- 如果确实要重构路径，最好保留 redirect 或同步更新所有内部链接。

## 8. 首页维护

新增论文时，更新：

```text
docs/index.md
```

至少更新：

- 当前收录表格；
- 如果新增一条研究主线，补一个简短 section；
- 如果新增概念，可以在知识库结构或概念地图里体现。

## 9. Concepts 和 Comparisons 的使用规则

### Concepts

当一个概念会跨多篇论文复用时，才新增 concept 页面。

适合放入 `docs/concepts/` 的内容：

- Native Unified Models
- Mixture-of-Transformers
- Pixel-Space Flow Matching
- Native RoPE
- Dynamic Noise Scale

不适合放 concept 的内容：

- 只属于某篇论文的细节；
- 没有跨论文复用价值的实验结果。

### Comparisons

只有当论文之间确实强相关、用户会反复问“它们区别是什么”时，才新增 comparison。

当前已有：

```text
TUNA-2 vs SenseNova-U1
TUNA-2 vs SenseNova-U1 vs HiDream-O1-Image
```

Class-to-image 大类目前不强制互相比较。除非用户明确要求，或几篇论文形成清晰对照，否则不用为每篇新增 comparison。

### Projects

当内容不是外部论文，而是用户自己的 ongoing research project 时，放入：

```text
docs/projects/<project-slug>/
```

Projects 页面应该明确区分：

- 当前 confirmed implementation；
- 已验证 empirical findings；
- current preferred route；
- 仍开放的 ablation axes / research questions。

不要把未定型实验写成最终方法结论。

## 10. 构建和本地预览

严格构建：

```bash
python3 -m mkdocs build --strict
```

本地预览：

```bash
python3 -m mkdocs serve
```

如果报：

```text
No module named 'pymdownx'
```

说明当前 Python 环境缺少 MkDocs Material 依赖。需要安装：

```bash
python3 -m pip install mkdocs-material pymdown-extensions
```

当前项目之前已经解决过这个依赖问题。

构建时可能出现 Material for MkDocs 关于 MkDocs 2.0 的未来警告。只要 `Documentation built` 成功，这个警告不影响当前站点。

## 11. Git / GitHub Pages 部署

站点通过 GitHub Pages 自动部署。正常流程：

```bash
git status --short
git add <changed-files>
git commit -m "Add <paper> notes"
git push
```

推送后 GitHub Actions 会自动部署到：

```text
https://YaohouF.github.io/paper-notes/
```

如果本地 sandbox 内 `git push` 出现：

```text
Could not resolve hostname github.com
```

通常是 sandbox 网络限制。需要用 Codex 的 escalated command 重试 `git push`。

## 12. 文件编辑注意事项

在 Codex 环境中：

- 使用 `apply_patch` 创建/修改 Markdown 文件。
- 不要用 `cat > file` 这类 shell 写文件方式。
- 读文件优先用 `rg`、`sed`、`find`。
- 改动前后用 `git status --short` 检查工作区。
- 不要覆盖用户已有未提交改动。

## 13. 新论文材料的常见位置

用户通常把论文和代码放在：

```text
/Users/yoho/Desktop/my_paper/<paper-name>
```

常见形式：

```text
<paper-name>/
  main.tex
  sections/
  figures/
  README
  <code-repo>/
```

如果本地只有论文没有代码：

1. 先在论文中找 GitHub / project page / code URL。
2. 如果需要联网 clone，用 `git clone`。
3. clone 后放在论文目录下，例如：

```text
/Users/yoho/Desktop/my_paper/fd-loss/FD-loss-main
```

## 14. 当前已分析论文简表

| Paper | Category | Main focus |
|---|---|---|
| TUNA-2 | Multimodal / Text-to-Image | Pixel embeddings, VAE-free / encoder-free unified model |
| SenseNova-U1 | Multimodal / Text-to-Image | Native unified understanding + generation, NEO-unify, MoT |
| HiDream-O1-Image | Multimodal / Text-to-Image | Pixel-level Unified Transformer, Qwen3-VL initialized raw-pixel generation |
| FD-loss | Class-to-Image | Representation Fréchet loss as post-training objective |
| DINOv3 + JiT Generation | Project | Time-conditioned DINOv3 encoder + JiT denoising + pixel-space generation finetune |

## 15. Commit message convention

推荐：

```text
Add <PaperName> reading notes
Add <PaperName> class-to-image notes
Update paper notes maintenance guide
```

保持短、清楚、可搜索。

## 16. 最重要的维护原则

```text
不要只搬运论文摘要。
每篇笔记都要回答：
  1. 它到底解决什么问题？
  2. 核心机制怎么工作？
  3. 训练/数据 recipe 是什么？
  4. 代码里是否真的这么实现？
  5. 如果我要复现，缺什么？
```

这套站点的价值在于“论文理解 + 代码核对 + 复现判断”三者同时存在。
