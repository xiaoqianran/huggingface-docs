# DDIM

[Denoising Diffusion Implicit Models](https://huggingface.co/papers/2010.02502) (DDIM) by Jiaming Song, Chenlin Meng and Stefano Ermon.

The abstract from the paper is:

*Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a Markovian diffusion process. We construct a class of non-Markovian diffusion processes that lead to the same training objective, but whose reverse process can be much faster to sample from. We empirically demonstrate that DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, and can perform semantically meaningful image interpolation directly in the latent space.*

The original codebase can be found at [ermongroup/ddim](https://github.com/ermongroup/ddim).

## DDIMPipeline[[diffusers.DDIMPipeline]]
#### diffusers.DDIMPipeline[[diffusers.DDIMPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ddim/pipeline_ddim.py#L37)

Pipeline for image generation.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.DDIMPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ddim/pipeline_ddim.py#L62[{"name": "batch_size", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "use_clipped_model_output", "val": ": bool | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}]- **batch_size** (`int`, *optional*, defaults to 1) --
  The number of images to generate.
- **generator** (`torch.Generator`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
  applies to the [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers. A value of `0`
  corresponds to DDIM and `1` corresponds to DDPM.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **use_clipped_model_output** (`bool`, *optional*, defaults to `None`) --
  If `True` or `False`, see documentation for [DDIMScheduler.step()](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler.step). If `None`, nothing is passed
  downstream to the scheduler (use `None` for schedulers which don't support this argument).
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.0[ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images

The call function to the pipeline for generation.

Example:

```py
>>> from diffusers import DDIMPipeline
>>> import PIL.Image
>>> import numpy as np

>>> # load model and scheduler
>>> pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

>>> # run pipeline in inference (sample random noise and denoise)
>>> image = pipe(eta=0.0, num_inference_steps=50)

>>> # process image to PIL
>>> image_processed = image.cpu().permute(0, 2, 3, 1)
>>> image_processed = (image_processed + 1.0) * 127.5
>>> image_processed = image_processed.numpy().astype(np.uint8)
>>> image_pil = PIL.Image.fromarray(image_processed[0])

>>> # save image
>>> image_pil.save("test.png")
```

**Parameters:**

unet ([UNet2DModel](/docs/diffusers/v0.39.0/en/api/models/unet2d#diffusers.UNet2DModel)) : A `UNet2DModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of [DDPMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler), or [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler).

**Returns:**

`[ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple``

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images

## ImagePipelineOutput[[diffusers.ImagePipelineOutput]]
#### diffusers.ImagePipelineOutput[[diffusers.ImagePipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pipeline_utils.py#L122)

Output class for image pipelines.

**Parameters:**

images (`List[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width, num_channels)`.

### Cosmos3
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/cosmos3.md

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Cosmos 3

NVIDIA Cosmos 3 is a unified world foundation model (WFM) for Physical AI — a single omni-model that combines world generation, physical reasoning, and action generation. It replaces the separate Predict, Reason, and Transfer models from earlier Cosmos releases: whether you're building for robotics, autonomous vehicles, or smart spaces, Cosmos 3 gives you one foundation to simulate and understand the physical world.

What's shipping with this release:

- Models on the Hugging Face Hub with model cards and licensing
- Cosmos 3 Diffusers integration for generation pipelines (this page)
- Post-training scripts for fine-tuning Cosmos 3 on your own data
- Open synthetic data generation (SDG) datasets for Physical AI

## What's new in Cosmos 3

The biggest change from previous Cosmos releases is that Cosmos 3 is an *omni-model*, built on a Mixture-of-Transformers (MoT) architecture. Previously, developers worked with separate models for world generation (Predict), controlled generation (Transfer), scene understanding (Reason), and action-policy generation. Cosmos 3 unifies all of these in one model that reasons and generates across modalities in a single forward pass.

From one model you can:

- Generate physically plausible video worlds from text, images, or action inputs (image-to-video, text-to-video, action-conditioned video generation).
- Reason about physical properties like motion, causality, and spatial relationships.
- Predict future video and action sequences from the current state.
- Transfer scenes across viewpoints and conditions with structural control *(coming soon)*.

Under the hood, a single `Cosmos3OmniTransformer` runs a Qwen-style language model in parallel with a diffusion generation pathway: text tokens flow through a causal "understanding" stream while video and sound latents flow through a bi-directionally-attended "generation" stream, joined by a 3D multimodal RoPE. See the [Cosmos World Foundation Model Platform paper](https://huggingface.co/papers/2501.03575) for the architectural background.

## Available checkpoints

Two checkpoints are released on the Hub — [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) (smaller, faster) and [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (larger, higher quality). The same pipeline class supports text-to-image, text-to-video, image-to-video, and (with a sound-capable checkpoint) text+image-to-video-with-sound — pick a repo and use the per-model tab in each workflow below.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Prompt upsampling

Cosmos 3 was trained on long, highly descriptive captions. For optimal quality, short text prompts should be **upsampled into a specific JSON structure** before they are passed to the pipeline. The upsampler lives in the [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) package.

Start from a short, plain-text prompt and save it to `assets/prompt.txt`. For the text-to-video example below, the original prompt is *"A robotic arm is cleaning a plate in a kitchen"*:

```bash
mkdir -p assets
echo "A robotic arm is cleaning a plate in a kitchen" > assets/prompt.txt
```

Then install the framework and run the upsampler. The example below upsamples for text-to-video using Opus-4.6:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos-framework
pip install -e packages/cosmos-framework

export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-opus-4-6"
export PROMPT_UPSAMPLER_API_TOKEN="REDACTED"

python -m cosmos_framework.inference.prompt_upsampling \
    --input assets/prompt.txt \
    --output assets/example_t2v_prompt.json \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "16,9"
```

Switch `--mode` to match the workflow you are targeting (`text2image`, `text2video`, `image2video`). The command writes the upsampled prompt(s) to the `--output` file as a JSON array (one object per non-empty line in `--input`); pass a `.jsonl` path instead to get one JSON object per line. For `image2video`, you must also supply the conditioning image via `--image-url` (a URL or local path) or `--image-list` (one image per prompt).

A pre-upsampled positive prompt (`assets/example_t2v_prompt.json`) and negative prompt (`assets/negative_prompt.json`) are provided for convenience, and are used by the generation examples below. The examples load these JSON files and pass them to the pipeline as JSON strings via `json.dumps(...)`.

## Text-to-video

Multi-frame generation conditioned on text alone. Pick `num_frames` based on the target duration — the default `num_frames=189` produces ≈ 7.9 s at 24 FPS. The prompt and negative prompt are read from the JSON-upsampled files described in [Prompt upsampling](#prompt-upsampling).

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
