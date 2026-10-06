# Qwen-Image 2.1

Qwen-Image 2.1 encodes the prompt and any condition images together with a Qwen3-VL model, then denoises the target
image with a single-stream block-causal transformer. See
[`QwenImage21Transformer2DModel`](../models/qwenimage21_transformer2d) for block-causal attention, the attention
processors, and `causal_condition`.

The defaults are the values Qwen recommends: 40 steps and no guidance. Pass a `negative_prompt` together with
`true_cfg_scale > 1` to turn classifier-free guidance on, which doubles the work per step.

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16).to("cuda")
