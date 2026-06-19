# TUNA-2 Model Architecture

## 1. Architecture Evolution

### TUNA: latent-space baseline

**[Paper]** Existing TUNA uses a vision encoder and Qwen decoder with:

- language-modeling head for text;
- flow-matching head for image generation;
- a unified visual representation derived from VAE latents.

**[Code]** Released TUNA configuration uses:

```text
WAN 2.2 VAE
latent channels: 48
latent spatial size at 512p: 32 x 32
SigLIP patch projection: Conv2d(48 -> 1152, kernel=1, stride=1)
Qwen2.5-7B or 1.5B
```

### TUNA-R: representation encoder, no VAE

**[Paper]**

```text
raw RGB pixels
  -> pretrained representation encoder
  -> visual tokens
  -> Qwen decoder

image generation occurs directly in pixel space
```

**[Code]**

- SigLIP2-SO400M;
- hidden size 1152;
- 27 layers in source config, truncated to 26 using feature layer `-2`;
- position embeddings interpolated for 512 resolution;
- projection:

```text
RMSNorm(1152)
-> Linear(1152, 3584)
-> GELU
-> Linear(3584, 3584)
```

### TUNA-2: no visual encoder

**[Paper]** TUNA-2 removes both VAE and representation encoder.

**[Code]** The only visual tokenizer is:

```python
Conv2d(
    in_channels=3,
    out_channels=3584,
    kernel_size=16,
    stride=16,
)
RMSNorm(3584)
```

For an RGB image:

```text
[B, 3, H, W]
-> [B, 3584, H/16, W/16]
-> [B, (H/16)*(W/16), 3584]
```

At 512x512:

```text
32 x 32 = 1024 image tokens
```

At the alternative public buckets:

```text
448 x 576 -> 28 x 36 = 1008 tokens
576 x 448 -> 36 x 28 = 1008 tokens
384 x 672 -> 24 x 42 = 1008 tokens
672 x 384 -> 42 x 24 = 1008 tokens
```

The patchifier is learned end-to-end with the complete model.

## 2. Unified Text-Image Sequence

The model extends the Qwen tokenizer with visual sentinels:

```text
<|vision_start|>
<|image_pad|>
<|vision_end|>
```

The dataset creates placeholder spans. Before Qwen forward, the embedding vectors of `<|image_pad|>` positions are replaced by actual image embeddings.

### Text-to-image

**[Code]**

```text
[BOS]
[prompt text]
[BOI]
[image placeholders x N]
[EOI]
[EOS]
```

The prompt is conditioning only:

- no LM loss on prompt;
- no LM loss on image placeholders;
- image span receives image reconstruction/flow loss.

### Image understanding

**[Code]**

```text
[BOS]
[BOI]
[input image placeholders x N]
[EOI]
[question/prompt]
[assistant response]
[EOS]
```

The image placeholders are replaced by visual embeddings. Only assistant response tokens contribute to the next-token CE loss.

### Image editing

**[Code]**

```text
[BOS]
[BOI][source image x N][EOI]
[edit instruction]
[BOI][target/noisy image x N][EOI]
[EOS]
```

Only the second image span is an image-loss target. The source image is conditioning.

## 3. Qwen Backbone

**[Paper]**

```text
Qwen2.5-7B-Instruct
```

**[Code]**

Main 7B configuration:

```text
hidden size: 3584
Q heads: 56
KV heads: 14
GQA ratio: 4 query heads per KV head
attention backend: SDPA by default
```

The Qwen input embedding table handles text IDs. Image vectors bypass lookup and replace the placeholder embeddings before entering the same transformer layers.

The normal Qwen LM head predicts text logits.

## 4. Omni-attention

The same sequence must support:

- autoregressive text;
- bidirectional image-token interaction during denoising.

**[Code]** Omni-attention starts from a causal lower-triangular mask, then opens full bidirectional attention inside each image span:

```text
allowed(q, k) =
    q >= k
    OR
    q and k are in the same image span
```

Consequences:

- text remains causally autoregressive;
- all patches of one image can attend to one another;
- later text can attend to preceding image tokens;
- image tokens cannot generally see future text unless sequence placement permits it;
- source and target image spans are each internally bidirectional.

This mask is used in both Qwen and the flow head.

## 5. Position Encoding

**[Code]** Text and image positions use different RoPE treatment:

- text positions use Qwen's normal 1D RoPE;
- image positions use 3D RoPE over `(time, height, width)`.

The 3D head-dimension allocation is:

```text
time:   D_head / 4
height: 3 * D_head / 8
width:  3 * D_head / 8
```

For head dimension 64:

```text
time: 16
height: 24
width: 24
```

The implementation applies 3D RoPE only to actual image patches, excluding the leading timestep meta-token.

For an image:

```text
t = 0
h = row index
w = column index
```

For video, `t` varies by frame.

## 6. Timestep Input

Each image span reserves its first placeholder position for the timestep embedding:

```text
image span:
[time embedding][patch token 0][patch token 1]...
```

With aspect-ratio embeddings enabled, the format would be:

```text
[height embedding][width embedding][time embedding][patches...]
```

The main TUNA-2 config disables aspect-ratio embeddings, so only the timestep meta-token is used.

Timestep embedding:

```text
sinusoidal frequency embedding, dim 256
-> Linear(256, 3584)
-> SiLU
-> Linear(3584, 3584)
```

## 7. Flow Head

The Qwen final hidden sequence is passed to a separate flow head.

**[Code]** Main configuration:

```text
16 ModulatedAttentionBlock layers
hidden size: 3584
56 Q heads / 14 KV heads
Qwen intermediate size reused for MLP
```

Each block contains:

```text
RMSNorm
AdaLN modulation
self-attention with omni-attention mask
residual gate
RMSNorm
AdaLN modulation
MLP
residual gate
```

The timestep MLP produces six vectors:

```text
shift_msa
scale_msa
gate_msa
shift_mlp
scale_mlp
gate_mlp
```

Crucially, these are applied only to image spans. Text positions use:

```text
shift = 0
scale = 0
gate = 1
```

Therefore timestep conditioning affects visual denoising tokens without globally modulating text tokens.

Final image layer:

```text
RMSNorm
time-conditioned shift/scale
Linear(3584, 16*16*3 = 768)
```

Each image-token output predicts one clean RGB patch of shape `16x16x3`.

The final layer and AdaLN projections are zero-initialized, following DiT-style stable initialization.

## 8. Pixel-space Rectified Flow

**[Paper]**

```text
x_t = t*x_1 + (1-t)*x_0
x_0 ~ N(0, I)
x_1 = clean image
```

The model predicts the clean image:

```text
x_theta = pi_theta(x_t, condition, t)
```

Then:

```text
v_theta = (x_theta - x_t) / (1-t)
v = x_1 - x_0
L_flow = ||v_theta - v||^2
```

**[Code]** Noise timesteps use:

```text
z ~ Normal(P_mean=-0.8, P_std=0.8)
t = sigmoid(z)
noise scale = 2.0
x_t = t*x + (1-t)*noise
```

The implementation directly minimizes:

```text
L_x0 = ||x_theta - x||^2 / max(1-t, 0.05)^2
```

This is equivalent to velocity MSE:

```text
v_theta - v = (x_theta - x) / (1-t)
```

So the paper and code describe the same objective in different algebraic forms.

## 9. Masked Visual Feature Learning

**[Paper]**

- randomly replace selected patch embeddings with one learnable mask token;
- use on both understanding and generation;
- generation predicts all masked and unmasked patches;
- understanding predicts response text from partial visual evidence.

**[Code]**

```text
mask applied after Conv2d patch embedding
mask token shape: [1, 1, 3584]
masked positions sampled independently per sample
```

Production paper recipe:

```text
active in final 40% of pretraining
applied to 50% of examples
mask ratio uniformly sampled from 0% to 50%
```

Public restoration config:

```text
masked_image_ratio_max = 0.3
masked_image_ratio_min = -0.7
```

Negative sampled ratios result in zero masked patches, creating an implicit probability of no masking. This public config is not the exact paper recipe.

