# Native Unified Models

Native unified multimodal model 指的是：模型不再通过多个 modality-specific subsystem 拼接起来，而是尽量在一个统一架构里同时处理理解和生成。

传统路线：

```text
understanding:
  image -> pretrained vision encoder -> projector -> LLM

generation:
  text/image condition -> diffusion in VAE latent -> VAE decoder
```

Native unified 路线：

```text
pixels + words
  -> unified tokens
  -> shared / natively coupled backbone
  -> text and image outputs
```

## 为什么重要？

传统系统的问题：

- understanding 和 generation 表征空间不一致；
- VE 适合语义理解，但可能丢失像素级细节；
- VAE 适合压缩生成，但可能限制文字渲染和细粒度定位；
- 多模块训练流程复杂，connector alignment 成本高。

Native unified model 试图让模型直接从 pixel-word relation 中学习，不依赖外部 encoder/decoder 先验。

## 代表论文

| Paper | Native 程度 |
|---|---|
| TUNA-2 | 去掉 VE/VAE，pixel embeddings + Qwen decoder |
| SenseNova-U1 | 去掉 VE/VAE，near-lossless visual interface + Native MoT |

