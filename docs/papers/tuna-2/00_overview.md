# TUNA-2 Paper and Code Reading Notes

Sources:

- Paper LaTeX: `tuna-2/`
- Released code: `tuna-2-main/`
- Paper title: *TUNA-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation*

## Source Labels

The notes use:

- **[Paper]** explicitly stated in the paper.
- **[Code]** verified from released implementation/configuration.
- **[Inference]** a reasoned interpretation, not explicitly guaranteed.

## Reading Map

- `01_model_architecture.md`: architecture variants, token sequence, Qwen backbone, flow head, attention, position/time conditioning.
- `02_data_construction.md`: Stage-1 and SFT data, task formatting, image preprocessing, multi-resolution handling.
- `03_training_recipe.md`: pretraining/SFT recipe, losses, masking, optimizer, sampling and infrastructure.
- `04_paper_code_crosscheck.md`: paper claims checked against code, discrepancies, implementation caveats.
- `05_reproducibility_gaps.md`: details still unavailable after reading both sources.

## One-sentence Summary

TUNA-2 is a native unified multimodal model that removes both the VAE and semantic vision encoder, converts RGB images directly into non-overlapping 16x16 pixel embeddings, processes image and text tokens with a Qwen2.5 decoder, predicts text through the normal LM head, and predicts clean image patches through a timestep-modulated flow head.

## Three Compared Architectures

```text
TUNA:
pixels -> WAN VAE -> latent tokens -> SigLIP-style encoder -> Qwen decoder
generation target: VAE latent

TUNA-R:
pixels -> SigLIP-2 -> projected visual tokens -> Qwen decoder
generation target: raw pixels

TUNA-2:
pixels -> Conv2d patch embedding -> Qwen decoder
generation target: raw pixels
```

TUNA-2 is the paper's main encoder-free model.

## TUNA-2 Unified Forward Path

```text
text:
  strings -> Qwen tokenizer -> token IDs -> Qwen embedding table

image:
  RGB [-1, 1]
  -> Conv2d(kernel=16, stride=16, out=3584)
  -> RMSNorm
  -> pixel tokens

unified sequence:
  replace <image_pad> slots with pixel tokens
  -> Qwen2.5-7B decoder

understanding:
  Qwen LM head -> next-token CE

generation:
  Qwen final hidden states
  -> 16 modulated attention blocks
  -> final AdaLN + linear patch predictor
  -> clean RGB patches
  -> weighted x0 MSE, equivalent to velocity loss
```

## Main Training Stages

1. **Stage 1: full-model pretraining**
   - 550M internal image-text pairs;
   - image captioning and T2I at 3:7;
   - Nemotron text-only data is 20% of total sampling;
   - 300K steps;
   - AdamW, LR `1e-4`;
   - final 40% uses masked visual feature learning.

2. **Stage 2: full-model SFT**
   - image instruction following;
   - image editing;
   - high-quality image generation;
   - 13M FineVision conversations;
   - about 2M OmniEdit examples;
   - 50K steps;
   - AdamW, LR `2e-5`.

TUNA-R additionally has a 3K-step connector alignment stage at LR `5e-4`.

## Important Repository Caveat

The released repository is not a complete dump of the production training system:

- internal production data is not released;
- full production weights are restricted;
- the public checkpoint may have some LLM and flow-head layers removed/reinitialized;
- the default public training YAML is a restoration/fine-tuning template, not the exact Stage-1 production configuration.

