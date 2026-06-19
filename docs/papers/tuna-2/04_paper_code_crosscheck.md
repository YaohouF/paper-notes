# TUNA-2 Paper-Code Cross-check

## Verified Claims

### Encoder-free means only patch embedding

**Paper:** TUNA-2 removes VAE and representation encoder.

**Code:** `SimplePatchEmbedding` is a single strided Conv2d plus RMSNorm. No hidden ViT/CLIP module exists in the TUNA-2 class.

Status: **verified**.

### Patch size is 16

**Paper:** patch embedding size 16.

**Code config:** `patch_size: 16`, `image_latent_dim: 3`.

Status: **verified**.

Note: Python constructor defaults sometimes use `patch_size=2`; Hydra YAML overrides them. The experiment configuration is authoritative.

### Full model is end-to-end trainable

**Paper:** full-model pretraining/SFT.

**Code:** only names matching `frozen_params` are frozen. Main TUNA-2 config lists `['vae']`, but no VAE exists, so Qwen, patch embedding, and flow head remain trainable.

Status: **verified**.

### Text and image share one sequence

**Paper:** pixel embeddings and text tokens are jointly processed by the LLM decoder.

**Code:** image placeholders in Qwen embeddings are replaced by pixel embeddings before one Qwen forward pass.

Status: **verified**.

### Separate text and image heads

**Paper:** LM head for autoregressive text and flow head for image generation.

**Code:** Qwen returns logits and hidden states; logits feed CE, hidden states feed 16 modulated attention blocks plus patch output layer.

Status: **verified**.

### Masking applies to understanding and generation

**Paper:** same masking mechanism, different objectives.

**Code:** mask replacement happens before task-specific losses, so it affects any image-bearing sample. Generation uses image MSE; MMU uses CE.

Status: **verified**.

## Paper Details Supplemented by Code

Code provides:

- patch Conv2d and RMSNorm implementation;
- hidden size 3584;
- 56 Q heads and 14 KV heads;
- 16 flow-head layers;
- timestep MLP architecture;
- 3D RoPE dimension split;
- exact omni-attention mask;
- image-span-only AdaLN;
- final output size `16*16*3`;
- logit-normal timestep distribution;
- noise scale 2.0;
- weighted x0 loss;
- public optimizer betas, weight decay, warmup, EMA, clipping;
- public multi-resolution buckets;
- exact token order for T2I/MMU/edit.

## Apparent Differences Explained

### Paper v-loss vs code x0 loss

Not a contradiction.

```text
v_theta-v = (x_theta-x_clean)/(1-t)
```

Therefore velocity MSE equals x0 MSE weighted by `1/(1-t)^2`.

### Config says prediction=velocity

The pixel wrapper constructs the older transport object for API compatibility, but actual training uses `JiTNoiseScheduler` and `jit_x0_prediction_loss`.

For TUNA-2, the operative implementation is x0 prediction with equivalent velocity weighting.

### Public masking range differs from paper

Paper production recipe:

```text
mask ratio 0%-50%, 50% of examples
```

Public restoration config:

```text
random ratio -70%-30%
```

Negative ratios create no mask. This approximates a mixture of unmasked/masked examples but is not identical to the paper recipe.

### Public data mixture differs from Stage 1

Paper Stage 1:

```text
70% T2I / 30% caption within image-text
20% text-only overall
```

Public fine-tuning:

```text
50% T2I / 20% edit / 20% MMU / 10% text
```

These serve different purposes and should not be conflated.

## Implementation Concerns

### RoPE builder uses a hard-coded patch size

In `Tuna2Pixel._prepare_embeds`, `build_rope(..., patch_size=16, attention_head_dim=64)` is hard-coded.

This matches the main 7B TUNA-2 config, but makes architectural overrides fragile.

### Dataset defaults assume patch size 16

`TIDataset` and `EditDataset` calculate image token count using a hard-coded patch size 16 unless explicitly overridden.

Again, correct for the paper model, but not generic.

### Public code is a cleaned/reconstructed release

Comments repeatedly mention adaptation from Show-o2, JiT, DeTok, and original internal Tuna configurations. It should be treated as an authoritative implementation guide for the released model family, but not necessarily byte-identical to the internal production training stack.

### Some comments/defaults are stale

Examples:

- class defaults often reflect older latent configurations;
- README batch-size table and current YAML are not perfectly synchronized;
- `image_latent_*` names remain even when tensors are raw RGB pixels.

Use instantiated Hydra configs and active forward code rather than comments alone.

## Revised Mental Model

TUNA-2 is best understood as:

```text
Qwen2.5 decoder as a shared multimodal trunk

input:
  text token embeddings
  + raw RGB patch embeddings

attention:
  causal across sequence
  + bidirectional within image spans

outputs:
  LM logits for response tokens
  shared hidden states for image spans

image head:
  timestep-conditioned DiT-like refinement
  -> direct clean RGB patch prediction
```

It is not a standard DiT with a text encoder, nor a standard VLM with a separate diffusion model. The Qwen decoder itself is the shared understanding/generation representation backbone.

