# Native RoPE

Native RoPE 是 SenseNova-U1 / NEO-unify 用来统一文本和图像位置编码的机制。

## 核心思想

文本 token 只有一维顺序：

```text
T axis
```

图像 token 既有序列位置，也有空间位置：

```text
T axis + H axis + W axis
```

SenseNova-U1 把每个 attention head 的维度分配到 T/H/W：

```text
T: 64
H: 32
W: 32
total head dim = 128
```

## 位置表示

```text
text token:
  (T, H=0, W=0)

image token:
  (T, H, W)
```

这样同一个 Transformer 可以处理：

- language causal order；
- image spatial structure；
- interleaved image-text sequence。

## 配置

SenseNova-U1 中：

```text
temporal rope theta = 5,000,000
spatial rope theta = 10,000
```

代码中对应：

```python
rope_base = 5000000.0
rope_theta_hw = 10000.0
```

