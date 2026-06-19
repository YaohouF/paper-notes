# Pixel-Space Flow Matching

Pixel-space flow matching 指图像生成不发生在 VAE latent space，而是直接在 pixel patch space 中建模。

## 基本公式

给 clean image patch \(x\)，噪声 \(\epsilon\)，timestep \(t\)：

```text
z_t = t*x + (1-t)*sigma_R*epsilon
```

模型预测 clean signal：

```text
x_hat_theta = model(z_t, t, condition)
```

再转成 velocity：

```text
v_theta = (x_hat_theta - z_t) / (1-t)
v_star  = (x - z_t) / (1-t)
```

训练目标：

```text
L = MSE(v_theta, v_star)
```

## 为什么不是直接预测噪声？

SenseNova-U1 和 TUNA-2 都采用类似 JiT 的 `x-pred + v-loss` 思路：

- 模型输出 clean patch；
- loss 在 velocity space 里算；
- 推理时用 flow ODE / Euler solver 更新图像。

## 和 VAE latent diffusion 的区别

| Dimension | Latent diffusion | Pixel-space flow matching |
|---|---|---|
| Generation space | VAE latent | pixel patch |
| Decoder | VAE decoder needed | direct patch output |
| Bottleneck | VAE compression | patch compression only |
| Fine text rendering | may suffer | potentially better |

