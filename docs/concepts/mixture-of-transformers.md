# Mixture-of-Transformers

Mixture-of-Transformers, MoT, 在 SenseNova-U1 中指的是：understanding stream 和 generation stream 在 Transformer 内部使用不同参数分支，而不是只在最后接不同 head。

## 直觉

理解任务和生成任务需要的表示不同：

```text
understanding:
  clean image + text -> semantic reasoning -> text answer

generation:
  noisy image tokens + condition -> denoising / pixel synthesis
```

如果完全共享 Transformer 参数，两个目标可能互相干扰。MoT 的做法是：

```text
same sequence
same native attention context
but different norms / FFNs / experts by token type
```

## SenseNova-U1 里的实现

训练代码中有 generation branch 参数：

```python
attention_norm_mot_gen
ffn_norm_mot_gen
feed_forward_mot_gen
```

forward 时按 `image_gen_indicators` 区分 token：

```python
num_image_gen_tokens = image_gen_indicators.sum()

gen_tokens = hidden_states[:, :num_image_gen_tokens]
und_tokens = hidden_states[:, num_image_gen_tokens:]
```

然后：

```python
gen_tokens -> feed_forward_mot_gen
und_tokens -> feed_forward
```

## MoT vs MoE

MoT 和 MoE 不是同一个概念：

| Term | Meaning |
|---|---|
| MoT | 不同任务/stream 走不同 Transformer 参数分支 |
| MoE | FFN 被拆成多个 experts，token 稀疏选择 top-k experts |

SenseNova-U1-A3B 是 MoT + MoE：

```text
understanding stream -> understanding MoE experts
generation stream -> generation MoE experts
```

