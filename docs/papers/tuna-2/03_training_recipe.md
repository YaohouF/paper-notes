# TUNA-2 Training Recipe

## 1. Production Recipe from the Paper

### Stage 1: Full-model Pretraining

```yaml
backbone: Qwen2.5-7B-Instruct
patch_size: 16
steps: 300_000
hardware: 64 nodes
optimizer: AdamW
learning_rate: 1.0e-4
image_text_pairs: 550M
multimodal_ratio:
  caption: 0.30
  t2i: 0.70
text_only_total_ratio: 0.20
sequence_padding_per_gpu: 16_000
trainable: full model
```

Masking:

```yaml
active_period: final 40% of pretraining
example_probability: 0.50
mask_ratio: Uniform(0.0, 0.5)
```

TUNA-2 requires no separate connector alignment stage.

### TUNA-R Alignment Stage

```yaml
steps: 3_000
optimizer: AdamW
learning_rate: 5.0e-4
trainable: connector only
data: caption + T2I
```

Then TUNA-R follows full-model pretraining.

### Stage 2: SFT

```yaml
steps: 50_000
optimizer: AdamW
learning_rate: 2.0e-5
trainable: full model
data:
  - 13M FineVision conversations
  - approximately 2M OmniEdit
  - undisclosed high-quality generation corpus
sequence_padding_per_gpu: 16_000
```

## 2. Unified Objective

**[Code]**

```text
L_total =
    flow_coeff * L_image
    + ntp_coeff * L_text
```

TUNA-2 public configuration:

```text
flow_coeff = 1.0
ntp_coeff = 1.0
```

Optional dispersive loss:

```text
L_total += 0.25 * L_disp
```

but it is disabled in the main configuration.

### Text loss

Standard shifted next-token cross entropy:

```text
CE(logits[:, :-1], labels[:, 1:])
ignore_index = -100
```

### Image loss

Paper velocity form:

```text
L_v = ||(x_theta-x_t)/(1-t) - (x_1-x_0)||^2
```

Code x0 form:

```text
L_x = ||x_theta-x_1||^2 / max(1-t,0.05)^2
```

These are algebraically equivalent.

Image loss is applied only where `image_mask` is true:

- all target pixels for T2I;
- target image only for editing;
- no image loss for MMU/text-only.

## 3. Noise and Time Sampling

**[Paper]**

```text
x_t = t*x_clean + (1-t)*noise
t in [0,1]
```

**[Code]**

```yaml
t_distribution:
  z: Normal(-0.8, 0.8^2)
  t: sigmoid(z)
noise_scale: 2.0
t_denominator_floor: 0.05
```

The logit-normal time distribution favors particular noise levels instead of uniform `t`.

For MMU, the default public code usually fixes `t=1`, meaning clean images, with optional small-noise augmentation.

For editing:

- source image is fixed at `t=1`;
- target image receives a sampled timestep.

## 4. Optimizer Details from Public Code

The paper states only AdamW and LR.

The public trainer adds:

```yaml
optimizer: AdamW
betas: [0.9, 0.95]
epsilon: 1.0e-8
weight_decay: 0.01
gradient_clip_norm: 1.0
precision: bf16
warmup_steps: 1000
lr_after_warmup: constant
ema_decay: 0.9999
gradient_checkpointing: true
```

These values describe the public fine-tuning/restoration template. They are plausible production choices but are not explicitly confirmed as the paper's 300K/50K recipe.

## 5. Attention and Sequence Length

The paper says every GPU pads inputs to:

```text
16K tokens
```

The public example datasets use shorter `max_text_length` values, such as 2K or 4K, to make local training practical.

Attention pattern:

```text
text: causal
each image span: bidirectional
```

All ranks synchronize task-stream choice in distributed training to avoid shape/schema mismatches and collective deadlocks.

## 6. Masking Implementation

Masking occurs after pixel patch embedding:

```text
raw RGB
-> Conv2d patch embeddings
-> randomly replace selected embeddings with shared learnable mask token
-> Qwen decoder
```

Generation:

- flow loss remains on every target patch;
- masked and visible patches are both reconstructed.

Understanding:

- no image reconstruction target;
- text response is predicted from masked image evidence.

The mask token is initialized:

```text
Normal-like random parameter scaled by hidden_size^-0.5
shape [1,1,3584]
```

## 7. Inference Recipe

**[Code]**

```yaml
sampler: Euler
steps: 50
CFG: 7.5
time_shifting_factor: 3.0
```

Sampling starts from Gaussian RGB noise and integrates from `t=0` to `t=1`.

At every step:

1. model predicts clean `x0`;
2. convert to velocity:

```text
v = (x0_pred - x_t) / max(1-t,0.05)
```

3. Euler update:

```text
x_next = x_t + delta_t * v
```

Classifier-free guidance:

```text
x0_guided = x0_uncond + s*(x0_cond-x0_uncond)
```

The wrapper then converts the guided clean prediction into velocity for the solver.

## 8. Distributed Training

**[Paper]**

```text
64 nodes
```

The number/type of accelerators per node is not stated.

**[Code]** Public training supports:

- FSDP;
- default `SHARD_GRAD_OP`;
- sharded state dictionaries;
- bf16 parameters and reductions;
- DDP fallback;
- EMA;
- TensorBoard;
- per-stream batch sizes;
- gradient accumulation.

The released config is designed for fine-tuning rather than reconstructing the original 64-node run exactly.

