# TUNA-2 Remaining Reproducibility Gaps

## Production Data

Still unknown:

- sources and licenses of 550M image-text pairs;
- filtering and deduplication;
- caption quality and prompt construction;
- aesthetic filters;
- multilingual distribution;
- resolution/aspect-ratio distribution;
- high-quality generation SFT corpus;
- exact SFT mixing ratios.

## Production Optimization

Still unknown:

- effective global batch size;
- GPUs per node and accelerator type;
- gradient accumulation;
- weight decay and AdamW betas used internally;
- warmup/decay used internally;
- EMA usage internally;
- exact Stage-1/Stage-2 resolution curriculum;
- checkpoint frequency;
- loss normalization across variable token counts;
- whether task batches are synchronized exactly as in public code.

## Model Details

Mostly recoverable from code, but unclear:

- whether production Qwen2.5 differs from released base checkpoint;
- exact number of Qwen layers modified/removed in public foundation checkpoint;
- whether production flow head uses every implementation option in public code;
- exact Qwen attention modifications relative to upstream Transformers;
- production use of optional dispersive loss;
- production use of MMU pixel-noise augmentation.

## Masking

Paper gives the high-level schedule, but not:

- whether mask sampling is uniform per image or batch;
- whether the same random mask is shared across frames;
- whether masked examples are balanced by task;
- exact learned mask-token initialization.

The public code answers some of these for the released implementation, but differs from the paper's numeric masking schedule.

## Inference

Public defaults are known, but paper does not confirm:

- benchmark CFG scales;
- benchmark sampling steps;
- negative prompts;
- prompt rewriting;
- seed policy;
- whether all reported results use Euler;
- whether higher-resolution examples use different schedules.

## Best Practical Reproduction Path

1. Use `configs/model/tuna_2_pixel_7b.yaml`.
2. Initialize from Qwen2.5-7B-Instruct.
3. Use direct RGB Conv2d patchification with patch size 16.
4. Retain omni-attention and 3D image RoPE.
5. Train Qwen, patchifier, and 16-layer flow head jointly.
6. Use:

```text
t = sigmoid(N(-0.8, 0.8^2))
noise scale = 2
weighted x0 loss = MSE / max(1-t,0.05)^2
```

7. Approximate Stage-1 sampling:

```text
56% T2I
24% caption/MMU
20% text-only
```

8. Introduce paper masking only in final 40%:

```text
50% examples
Uniform mask ratio [0,0.5]
```

9. SFT on image instruction, edit pairs, and high-quality T2I at LR `2e-5`.
10. Treat public optimizer/EMA/FSDP settings as a workable baseline, not proven production values.

