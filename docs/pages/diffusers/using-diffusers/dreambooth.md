# DreamBooth

[DreamBooth](https://huggingface.co/papers/2208.12242) personalizes a pretrained model to a specific subject from a few images (for example, your cat) by fine-tuning the full weights and binding that subject to a unique identifier in the prompt (`sks cat`). You can then generate the subject in new settings, lighting, poses, and styles.

DreamBooth checkpoints are typically a few GBs because they contain the full model weights. Load them with [from_pretrained()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) and include the unique identifier in the prompt to trigger generation.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "sd-dreambooth-library/herge-style",
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
prompt = "A cute sks herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
pipeline(prompt).images[0]
```

    

To train your own checkpoint, see [Train DreamBooth](../training/dreambooth).

### Batch inference
https://huggingface.co/docs/diffusers/v0.41.0/using-diffusers/batched_inference.md
