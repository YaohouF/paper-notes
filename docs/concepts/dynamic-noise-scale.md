# Dynamic Noise Scale

Dynamic noise scale 是 SenseNova-U1 为多分辨率 pixel-space flow matching 引入的噪声尺度调整。

## 问题

如果不同分辨率都用：

```text
epsilon ~ N(0, I)
```

那么高分辨率图像包含更多 generation tokens，同一个 timestep 下整体噪声能量和 signal-to-noise ratio 会和低分辨率不同。

这会让多分辨率训练不稳定。

## SenseNova-U1 的做法

定义 generation token 数：

```text
N(H,W) = H*W / 32^2
```

动态噪声尺度：

```text
sigma_R(H,W) = sigma_0 * sqrt(N(H,W) / N0)
```

论文配置：

```text
sigma_0 = 1
N0 = 64
sigma_R in [1, 8] for N in [64, 4096]
```

## 代码直觉

```python
image_seq_len = cur_image_token_num // merge_size**2
scale = sqrt(image_seq_len / base)
noise_scale = scale * self.noise_scale
```

如果开启 noise-scale embedding：

```python
timestep_embedding += noise_scale_embedding
```

## 一句话

Dynamic noise scale 让不同分辨率下的 flow matching 看到更一致的噪声强度和 SNR 分布。

