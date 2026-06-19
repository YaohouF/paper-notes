# SenseNova-U1 Paper-Code Cross-check

## 1. Visual Interface

**Paper:** two convolutional layers, strides 16 and 2, final token corresponds to 32×32 image patch.

**Code:** `NEOVisionEmbeddings` uses:

```python
patch_embedding: Conv2d(kernel=patch_size, stride=patch_size)
dense_embedding: Conv2d(kernel=downsample_factor, stride=downsample_factor)
```

Training config:

```python
patch_size = 16
down_sample_ratio = 0.5
downsample_factor = 2
```

So:

```text
16 * 2 = 32
```

**Verdict:** matches.

## 2. No pretrained VE / VAE

**Paper:** removes separate pretrained vision encoders and VAE decoders.

**Code:** visual interface is local `NEOVisionModel`, not CLIP/SigLIP. Generation output is `fm_head` predicting RGB pixel patch vector:

```python
output_dim = 3 * (patch_size * merge_size) ** 2
```

With patch size 16 and merge size 2:

```text
output_dim = 3 * 32^2 = 3072
```

No VAE decoder appears in the core model forward.

**Verdict:** matches.

## 3. Dense 8B model config

**Paper table:**

```text
layers = 42
Q/KV heads = 32/8
hidden size = 4096
head size T/H/W = 64/32/32
```

**Code config:**

```python
HIDDEN_SIZE = 4096
HEAD_DIM = 128
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
EXTRA_NUM_LAYER = 6
NUM_LAYER = 36 + EXTRA_NUM_LAYER + EXTRA_NUM_LAYER_POST
```

Default `post_layer_num=0`, so `NUM_LAYER=42`.

**Verdict:** matches.

## 4. A3B model config

**Paper table:**

```text
layers = 48
Q/KV heads = 32/4
hidden size = 2048
understanding/generation experts = 128/32
top-k active = 8
```

**Code config:**

```python
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 4
NUM_LAYER = 48
num_experts = 128
moe_split_size = 8
```

`NEOMoELLMConfig` supports generation-side MoE knobs:

```python
gen_num_experts
gen_num_experts_per_tok
gen_moe_intermediate_size
```

**Verdict:** mostly matches. Generation expert count is supported in config class, but the public training config shown here does not explicitly set `gen_num_experts=32`; it may be loaded from released checkpoint/config or passed through model path.

## 5. MoT parameter decoupling

**Paper:** generation and understanding streams have separate projections, norms, and FFNs/MoE experts, routed by token type.

**Code:** `modeling_qwen3_moe_mot.py` has:

```python
attention_norm_mot_gen
ffn_norm_mot_gen
feed_forward_mot_gen
```

Forward separates generation tokens and non-generation tokens:

```python
num_image_gen_tokens = image_gen_indicators.sum().item()

gen = hidden_states[:, :num_image_gen_tokens]
und = hidden_states[:, num_image_gen_tokens:]
```

Dense:

```python
feed_forward_mot_gen(gen)
feed_forward(und)
```

MoE:

```python
feed_forward_mot_gen(gen)
feed_forward(und)
```

**Verdict:** matches.

## 6. Native attention

**Paper:** text causal, same image bidirectional; noise tokens can attend bidirectionally within image block and to clean inputs; clean tokens cannot attend to noise tokens.

**Code:** base mask:

```python
mask = same_doc_causal OR same_image
```

generation mask adds:

```python
kv_is_gen
q_is_gen
same_img
gate1 = (~kv_is_gen) | (q_is_gen & same_img)
```

This prevents non-generation/clean queries from attending to generation KV tokens except controlled same-image cases, while generation tokens retain same-image bidirectional access.

**Verdict:** matches directionally. Exact duplicated-image boundary handling is more complex in code than in paper.

## 7. Native RoPE

**Paper:** T/H/W axes with head dimensions 64/32/32. Temporal theta 5,000,000, spatial theta 10,000.

**Code config:**

```python
HEAD_DIM = 128
rope_base = 5000000.0
rope_theta_hw = 10000.0
```

Config class:

```python
NEOLLMConfig(rope_theta_hw=10000.0)
NEOMoELLMConfig(rope_theta_hw=10000.0)
```

**Verdict:** matches.

## 8. Dynamic noise scale

**Paper:**

```text
sigma_R = sigma_0 * sqrt(N / N0)
sigma_0 = 1
N0 = 64
sigma in [1,8]
```

**Code training target:**

```python
image_seq_len = cur_image_token_num // merge_size**2
scale = sqrt(image_seq_len / noise_scale_base_image_seq_len)
noise_scale = scale * noise_scale
```

Public script:

```bash
noise_scale_mode=resolution
noise_scale_base_image_seq_len=64
noise_scale_max_value=8
add_noise_scale_embedding=true
```

**Verdict:** matches. Note that training code checks `("resolution", "dynamic")`; inference code also has `dynamic_sqrt`.

## 9. Flow matching objective

**Paper:**

```text
z_t = t*x + (1-t)*sigma_R*epsilon
v_theta = (x_hat_theta - z_t)/(1-t)
v_star = (x - z_t)/(1-t)
L = MSE(v_theta, v_star)
```

**Code:**

```python
cur_image_gen_z = t * cur_pixel_values + (1 - t) * cur_noise
image_gen_v = (image_gen_x - image_gen_z) / (1 - image_gen_t)

image_gen_pred_x = fm_head(image_gen_hidden_states)
image_gen_pred_v = (image_gen_pred_x - image_gen_z) / (1 - image_gen_t)
image_gen_loss = mse(image_gen_pred_v, image_gen_v)
```

**Verdict:** matches.

## 10. CFG drop

**Paper:** text condition dropped 10%; both text and image dropped additional 10%.

**Public script:**

```bash
cfg_txt_uncond_drop_prob=0.1
cfg_img_uncond_drop_prob=0
cfg_txtimg_uncond_drop_prob=0.1
cfg_is_uncond_drop_independent=false
```

**Verdict:** matches the two modes described: text-only drop and text+image drop. Public script does not use separate image-only drop.

## 11. Training table vs public script

**Paper Stage 3/4:**

```text
seq length = 32768
LR = 2e-5
```

**Public smoke-test script:**

```bash
seq_len=28672
lr=2e-4
total_steps=200000
max_pixels_gen=512*512
```

**Verdict:** public script is not the exact paper recipe; it is a standalone training/fine-tuning launcher with placeholder paths and smoke-test-like defaults.

## 12. Parameter count naming

**README:** `8B-MoT` refers to ~8B understanding parameters and ~8B generation parameters.

**Parameter breakdown doc:**

```text
Total params: 17.552B
generation_transformer: 8.186B
understanding_transformer: 8.121B
shared: 1.245B
```

**Verdict:** important naming caveat. Do not say total model is 8B.

