# TUNA-2 Data Construction and Input Formatting

## 1. Stage-1 Pretraining Data

### Multimodal data

**[Paper]**

```text
550M internal image-text pairs
30% image captioning
70% text-to-image generation
```

These two uses may derive from related image-text corpora but are formatted with opposite modality order and different loss targets.

Image captioning:

```text
image -> textual description
```

T2I:

```text
caption/prompt -> image
```

### Text-only data

**[Paper]**

```text
Nemotron
20% of total pretraining sampling
```

If the 30:70 caption/T2I ratio applies to the multimodal 80%, the total effective mixture is:

```text
T2I:       0.8 * 0.7 = 56%
caption:   0.8 * 0.3 = 24%
text-only:             20%
```

This interpretation is consistent with the paper's reported 7:3 generation-to-understanding balance inside image-text data.

## 2. SFT Data

**[Paper]** Stage 2 includes:

- image instruction following;
- image editing;
- high-quality image generation.

Explicit datasets:

```text
FineVision: 13M conversational examples
OmniEdit: approximately 2M editing examples
```

The scale/source of the high-quality generation subset is not disclosed.

## 3. Public JSONL Formats

### T2I

```json
{"image": "imgs/image.png", "caption": "a photo of a cat"}
```

Unified sequence:

```text
[BOS][caption][BOI][image span][EOI][EOS]
```

### MMU / visual instruction following

```json
{
  "image": "imgs/image.png",
  "conversations": [
    {"from": "human", "value": "Describe this image."},
    {"from": "gpt", "value": "A cat sitting on a chair."}
  ]
}
```

Unified sequence concept:

```text
[BOS][BOI][image span][EOI][prompt][response][EOS]
```

Only assistant tokens contribute to CE in the conversation-aware formatter.

### Text-only

```json
{
  "conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

No image flow loss is computed.

### Editing

```json
{
  "raw_image": "imgs/source.png",
  "out_image": "imgs/target.png",
  "instruction": "make it sunset"
}
```

Unified sequence:

```text
[source image][instruction][target image]
```

Only the target image span is selected by the image loss mask.

## 4. Image Preprocessing

**[Code]**

1. convert to RGB;
2. resize with bicubic interpolation;
3. optionally center crop;
4. convert to `[0,1]` tensor;
5. normalize:

```text
x_normalized = (x - 0.5) / 0.5
```

giving RGB values in `[-1,1]`.

This normalized RGB tensor is directly patchified and is also the clean image target.

## 5. Multi-resolution Buckets

The public 512-class buckets are:

```text
512 x 512
448 x 576
576 x 448
384 x 672
672 x 384
```

For each image, the dataset selects the bucket closest to its native aspect ratio.

The number of visual tokens remains approximately fixed:

```text
1024 or 1008 tokens with patch size 16
```

**[Paper caveat]** The active paper text does not state the Stage-1/Stage-2 image resolutions. A commented-out earlier experiment paragraph mentions 256p pretraining and 512p SFT, while released inference/training configs use 512-class buckets. Commented LaTeX is not reliable evidence for the final production recipe.

## 6. Public Four-stream Fine-tuning Template

The released training guide uses:

```text
T2I:  50%
edit: 20%
MMU:  20%
text: 10%
```

This is a restoration/fine-tuning recipe for released checkpoints.

It should not be confused with paper Stage 1:

```text
T2I:caption = 70:30 within image-text data
text-only = 20% of total
no editing in Stage 1
```

## 7. Conditioning and Targets by Task

| Task | Image input | Text input | LM target | Image target |
|---|---|---|---|---|
| T2I | noisy target image | caption/prompt | none | clean target image |
| Caption/MMU | clean or lightly noised image | prompt | assistant response | none |
| Text-only | none | prompt/history | assistant response | none |
| Editing | source + noisy target | instruction | none | clean target image |

For MMU, public code optionally adds slight pixel noise:

```text
probability: 0.1
maximum noise level: 0.1
```

The paper does not mention this MMU-noise augmentation.

## 8. What Is Not Disclosed About Data

Neither source provides:

- internal image sources and licenses;
- caption generation/recaptioning pipeline;
- aesthetic and quality filters;
- deduplication method;
- NSFW filtering;
- exact 550M split construction;
- high-quality generation SFT data scale;
- exact production resolution distribution;
- language distribution;
- effective batch-size/token accounting across tasks.

