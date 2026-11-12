# DreamLite

DreamLite is a text-to-image and image-editing model from ByteDance. It pairs a custom 2D U-Net
(`DreamLiteUNetModel`) with the `Qwen3-VL` multimodal encoder as its prompt / image-instruction encoder,
and uses an `AutoencoderTiny` (TAESD-style) VAE for fast latent encode/decode.

Two pipelines are exposed:

| Pipeline | Modes | CFG | Use case |
|---|---|---|---|
| [DreamLitePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipeline) | text-to-image **and** image-editing (auto-selected by whether `image` is `None`) | 3-branch dual CFG (`guidance_scale` on text branch, `image_guidance_scale` on image branch, à la InstructPix2Pix) | Highest quality |
| [DreamLiteMobilePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLiteMobilePipeline) | text-to-image **and** image-editing (auto-selected by whether `image` is `None`) | None — distilled, single UNet forward per step | On-device / low-latency |

Official checkpoints:

* Base model: [carlofkl/DreamLite-base](https://huggingface.co/carlofkl/DreamLite-base)
* Distilled mobile model: [carlofkl/DreamLite-mobile](https://huggingface.co/carlofkl/DreamLite-mobile)

> [!TIP]
> Both pipelines auto-detect text-to-image vs. image-editing mode from whether the `image` argument is
> provided. There is no separate `Img2Img` class.

> [!TIP]
> When loading an input image for editing, prefer `diffusers.utils.load_image(...)` over raw `PIL.Image.open(...)`.
> `load_image` enforces an RGB conversion and applies EXIF orientation, both of which the pipeline assumes.
> A plain `Image.open` of an RGBA / palette / EXIF-rotated source will silently produce a different latent
> conditioning and degrade output quality.

## Text-to-image (Base)

```python
import torch
from diffusers import DreamLitePipeline

pipe = DreamLitePipeline.from_pretrained("carlofkl/DreamLite-base", revision="diffusers", dtype=torch.bfloat16)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

image = pipe(
    prompt="a dog running on the grass",
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_t2i.png")
```

## Image editing (Base)

Pass an `image` to enter edit mode. Both `guidance_scale` (text branch) and `image_guidance_scale`
(image branch) are active here.

```python
import torch
from diffusers import DreamLitePipeline
from diffusers.utils import load_image

pipe = DreamLitePipeline.from_pretrained("carlofkl/DreamLite-base", revision="diffusers", dtype=torch.bfloat16)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

source = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
    prompt="turn the cat into a corgi",
    image=source,
    height=1024,
    width=1024,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_edit.png")
```

## Text-to-image (Mobile)

The mobile pipeline is distilled and skips CFG entirely — a single UNet forward per step. It accepts the
same `prompt` / `height` / `width` / `num_inference_steps` arguments, but **ignores** `guidance_scale` and
`image_guidance_scale` if passed (a warning is logged).

```python
import torch
from diffusers import DreamLiteMobilePipeline

pipe = DreamLiteMobilePipeline.from_pretrained("carlofkl/DreamLite-mobile", revision="diffusers", dtype=torch.bfloat16)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

image = pipe(
    prompt="a dog running on the grass",
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_mobile_t2i.png")
```

## Image editing (Mobile)

```python
import torch
from diffusers import DreamLiteMobilePipeline
from diffusers.utils import load_image

pipe = DreamLiteMobilePipeline.from_pretrained("carlofkl/DreamLite-mobile", revision="diffusers", dtype=torch.bfloat16)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

source = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
    prompt="turn the cat into a corgi",
    image=source,
    height=1024,
    width=1024,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("dreamlite_mobile_edit.png")
```

## Notes and limitations

* Both pipelines force `batch_size = 1` internally; `num_images_per_prompt` controls how many samples
  are drawn from the same prompt rather than parallel batching.
* The prompt encoder is `Qwen3-VL`, which is a multimodal model. Loading the full pipeline therefore
  requires sufficient GPU memory for both the U-Net and the Qwen3-VL text encoder (~4 GB + ~0.7 GB
  in bf16 for the base release).
* The VAE is `AutoencoderTiny` and exposes `encoder_block_out_channels`; `vae_scale_factor` is derived
  from it at pipeline init time.

## DreamLitePipeline[[diffusers.DreamLitePipeline]]

#### diffusers.DreamLitePipeline[[diffusers.DreamLitePipeline]]

```python
diffusers.DreamLitePipeline(text_encoder: Qwen3VLForConditionalGeneration, tokenizer: AutoTokenizer, processor: Qwen3VLProcessor, vae: AutoencoderTiny, unet: DreamLiteUNetModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/dreamlite/pipeline_dreamlite.py#L155)

DreamLite pipeline for text-to-image and instruction-based image editing.

The same pipeline supports both modes; the operating mode is auto-detected from the inputs:

- `image is None` -> text-to-image (single CFG on text).
- `image is not None` -> image-to-image / instruction edit (dual CFG: text + image).

Components:
text_encoder ([*~transformers.Qwen3VLForConditionalGeneration*]):
Multimodal text/vision encoder used to produce conditioning embeddings.
tokenizer ([*~transformers.AutoTokenizer*]):
Tokenizer for text-only (generate) mode.
processor ([*~transformers.Qwen3VLProcessor*]):
Multimodal processor for edit mode (text + image template).
vae ([*~diffusers.AutoencoderTiny*]):
Mobile-friendly tiny VAE for latent encode/decode.
unet ([*~diffusers.DreamLiteUNetModel*]):
DreamLite UNet (GQA + qk_norm + depthwise-separable convs).
scheduler ([*~diffusers.FlowMatchEulerDiscreteScheduler*]):
Flow-matching Euler scheduler with dynamic shift.

Note:
`batch_size` is currently forced to `1`; `num_images_per_prompt` is supported.

#### __call__[[diffusers.DreamLitePipeline.__call__]]

```python
__call__(prompt: typing.Optional[str] = None, negative_prompt: typing.Optional[str] = None, image: typing.Optional[PIL.Image.Image] = None, height: typing.Optional[int] = None, width: typing.Optional[int] = None, guidance_scale: float = 3.5, image_guidance_scale: float = 1.5, num_inference_steps: int = 30, sigmas: typing.Optional[typing.List[float]] = None, num_images_per_prompt: typing.Optional[int] = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, output_type: typing.Optional[str] = 'pil', return_dict: bool = True, max_sequence_length: int = 200, text_pad_embedding: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/dreamlite/pipeline_dreamlite.py#L388)

**Parameters:**

prompt : Text prompt.

negative_prompt : Negative text prompt (defaults to empty string).

image : Optional input image. If provided, the pipeline runs in **edit / image-to-image** mode with dual classifier-free guidance; otherwise it runs in **text-to-image** mode.

height : Output resolution (height). Defaults to `default_sample_size * vae_scale_factor` (1024). The same default applies in both T2I and I2I; pass an explicit value to override.

width : Output resolution (width). Defaults to `default_sample_size * vae_scale_factor` (1024). The same default applies in both T2I and I2I; pass an explicit value to override.

guidance_scale : CFG scale on the text branch (both modes).

image_guidance_scale : Additional CFG scale on the image branch (edit mode only).

num_inference_steps : Number of denoising steps.

sigmas : Optional explicit FlowMatch sigmas; defaults to a uniform linspace.

num_images_per_prompt : Output images per prompt (note: `batch_size` is forced to 1).

generator : Random generator(s).

output_type : `"pil"`, `"np"`, `"pt"` or `"latent"`.

return_dict : If True, returns a [DreamLitePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipelineOutput); else a tuple `(images,)`.

max_sequence_length : Maximum number of user-prompt tokens kept after dropping the chat-template prefix. Only applies to `generate` mode (the `edit` mode uses the multimodal processor's native padding).

text_pad_embedding : Optional learned pad embedding for masked positions.

**Returns:**

[DreamLitePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipelineOutput) or `tuple`.

Run the DreamLite pipeline.

## DreamLiteMobilePipeline[[diffusers.DreamLiteMobilePipeline]]

#### diffusers.DreamLiteMobilePipeline[[diffusers.DreamLiteMobilePipeline]]

```python
diffusers.DreamLiteMobilePipeline(text_encoder: Qwen3VLForConditionalGeneration, tokenizer: AutoTokenizer, processor: Qwen3VLProcessor, vae: AutoencoderTiny, unet: DreamLiteUNetModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/dreamlite/pipeline_dreamlite_mobile.py#L156)

DreamLite **Mobile** pipeline: a distilled, classifier-free-guidance-free variant of
[DreamLitePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipeline) for fast few-step inference (default 4 steps).

The operating mode is auto-detected from inputs (same as the base pipeline):

- `image is None` -> text-to-image.
- `image is not None` -> image-to-image / instruction edit.

Because classifier-free guidance is **distilled away**, `guidance_scale` and `image_guidance_scale` are
accepted for API parity with [DreamLitePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipeline) but are ignored in the denoising loop. `negative_prompt`
is intentionally absent.

Components (identical to the base pipeline):
text_encoder ([*~transformers.Qwen3VLForConditionalGeneration*]):
Multimodal text/vision encoder.
tokenizer ([*~transformers.AutoTokenizer*]):
Tokenizer for text-only (generate) mode.
processor ([*~transformers.Qwen3VLProcessor*]):
Multimodal processor for edit mode.
vae ([*~diffusers.AutoencoderTiny*]):
Mobile-friendly tiny VAE.
unet ([*~diffusers.DreamLiteUNetModel*]):
DreamLite UNet.
scheduler ([*~diffusers.FlowMatchEulerDiscreteScheduler*]):
Flow-matching Euler scheduler with dynamic shift.

Note:
`batch_size` is currently forced to `1`; `num_images_per_prompt` is supported.

#### __call__[[diffusers.DreamLiteMobilePipeline.__call__]]

```python
__call__(prompt: typing.Union[str, typing.List[str]] = None, image: typing.Optional[PIL.Image.Image] = None, height: typing.Optional[int] = None, width: typing.Optional[int] = None, num_inference_steps: int = 4, guidance_scale: typing.Optional[float] = None, image_guidance_scale: typing.Optional[float] = None, sigmas: typing.Optional[typing.List[float]] = None, num_images_per_prompt: typing.Optional[int] = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, output_type: typing.Optional[str] = 'pil', return_dict: bool = True, max_sequence_length: int = 200, text_pad_embedding: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/dreamlite/pipeline_dreamlite_mobile.py#L384)

**Parameters:**

prompt : Text prompt.

image : Optional input image. If provided, runs in **edit / image-to-image** mode; otherwise runs in **text-to-image** mode.

height : Output resolution (height). Defaults to `default_sample_size * vae_scale_factor` (1024).

width : Output resolution (width). Defaults to `default_sample_size * vae_scale_factor` (1024).

num_inference_steps : Number of denoising steps. Defaults to **4** (distilled).

guidance_scale : Accepted for API parity with [DreamLitePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipeline); **ignored** because CFG was distilled away.

image_guidance_scale : Accepted for API parity with [DreamLitePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipeline); **ignored** because CFG was distilled away.

sigmas : Optional explicit FlowMatch sigmas; defaults to a uniform linspace.

num_images_per_prompt : Output images per prompt (note: `batch_size` is forced to 1).

generator : Random generator(s).

output_type : `"pil"`, `"np"`, `"pt"` or `"latent"`.

return_dict : If True, returns a [DreamLitePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipelineOutput); else `(images,)`.

max_sequence_length : Maximum number of user-prompt tokens kept after dropping the chat-template prefix. Only applies to `generate` mode (the `edit` mode uses the multimodal processor's native padding).

text_pad_embedding : Optional learned pad embedding for masked positions.

**Returns:**

[DreamLitePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/dreamlite#diffusers.DreamLitePipelineOutput) or `tuple`.

Run the distilled DreamLite Mobile pipeline.

## DreamLitePipelineOutput[[diffusers.DreamLitePipelineOutput]]

#### diffusers.DreamLitePipelineOutput[[diffusers.DreamLitePipelineOutput]]

```python
diffusers.DreamLitePipelineOutput(images: typing.Union[typing.List[PIL.Image.Image], numpy.ndarray])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/dreamlite/pipeline_output.py#L25)

**Parameters:**

images (`List[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width, num_channels)`. PIL images or NumPy array present the denoised images of the diffusion pipeline.

Output class for DreamLite pipelines.

### Visualcloze
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/visualcloze.md

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
# limitations under the License.
-->

# VisualCloze

[VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](https://huggingface.co/papers/2504.07960) is an innovative in-context learning based universal image generation framework that offers key capabilities:
1. Support for various in-domain tasks
2. Generalization to unseen tasks through in-context learning
3. Unify multiple tasks into one step and generate both target image and intermediate results
4. Support reverse-engineering conditions from target images

## Overview

The abstract from the paper is:

*Recent progress in diffusion models significantly advances various image generation tasks. However, the current mainstream approach remains focused on building task-specific models, which have limited efficiency when supporting a wide range of different needs. While universal models attempt to address this limitation, they face critical challenges, including generalizable task instruction, appropriate task distributions, and unified architectural design. To tackle these challenges, we propose VisualCloze, a universal image generation framework, which supports a wide range of in-domain tasks, generalization to unseen ones, unseen unification of multiple tasks, and reverse generation. Unlike existing methods that rely on language-based task instruction, leading to task ambiguity and weak generalization, we integrate visual in-context learning, allowing models to identify tasks from visual demonstrations. Meanwhile, the inherent sparsity of visual task distributions hampers the learning of transferable knowledge across tasks. To this end, we introduce Graph200K, a graph-structured dataset that establishes various interrelated tasks, enhancing task density and transferable knowledge. Furthermore, we uncover that our unified image generation formulation shared a consistent objective with image infilling, enabling us to leverage the strong generative priors of pre-trained infilling models without modifying the architectures. The codes, dataset, and models are available at https://visualcloze.github.io.*

## Inference

### Model loading

VisualCloze is a two-stage cascade pipeline, containing `VisualClozeGenerationPipeline` and `VisualClozeUpsamplingPipeline`.
- In `VisualClozeGenerationPipeline`, each image is downsampled before concatenating images into a grid layout, avoiding excessively high resolutions. VisualCloze releases two models suitable for diffusers, i.e., [VisualClozePipeline-384](https://huggingface.co/VisualCloze/VisualClozePipeline-384) and [VisualClozePipeline-512](https://huggingface.co/VisualCloze/VisualClozePipeline-384), which downsample images to resolutions of 384 and 512, respectively. 
- `VisualClozeUpsamplingPipeline` uses [SDEdit](https://huggingface.co/papers/2108.01073) to enable high-resolution image synthesis.

The `VisualClozePipeline` integrates both stages to support convenient end-to-end sampling, while also allowing users to utilize each pipeline independently as needed.

### Input Specifications

#### Task and Content Prompts
- Task prompt: Required to describe the generation task intention
- Content prompt: Optional description or caption of the target image
- When content prompt is not needed, pass `None`
- For batch inference, pass `List[str|None]`

#### Image Input Format
- Format: `List[List[Image|None]]`
- Structure:
  - All rows except the last represent in-context examples
  - Last row represents the current query (target image set to `None`)
- For batch inference, pass `List[List[List[Image|None]]]`

#### Resolution Control
- Default behavior:
  - Initial generation in the first stage: area of ${pipe.resolution}^2$
  - Upsampling in the second stage: 3x factor
- Custom resolution: Adjust using `upsampling_height` and `upsampling_width` parameters

### Examples

For comprehensive examples covering a wide range of tasks, please refer to the [Online Demo](https://huggingface.co/spaces/VisualCloze/VisualCloze) and [GitHub Repository](https://github.com/lzyhha/VisualCloze). Below are simple examples for three cases: mask-to-image conversion, edge detection, and subject-driven generation.

#### Example for mask2image

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
content_prompt = """Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. 
The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. 
Its plumage is a mix of dark brown and golden hues, with intricate feather details. 
The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. 
The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, 
soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, 
tranquil, majestic, wildlife photography."""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1344,
    upsampling_height=768,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Example for edge-detection

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-1_image.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-1_edge.jpg'),
    ],
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-2_image.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-2_edge.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_query_image.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = "Each row illustrates a pathway from [IMAGE1] a sharp and beautifully composed photograph to [IMAGE2] edge map with natural well-connected outlines using a clear logical task."
content_prompt = ""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=864,
    upsampling_height=1152,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Example for subject-driven generation

```python
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_depth.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_image.jpg'),
    ],
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_depth.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_image.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_query_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_query_depth.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = """Each row describes a process that begins with [IMAGE1] an image containing the key object, 
[IMAGE2] depth map revealing gray-toned spatial layers and results in 
[IMAGE3] an image with artistic qualitya high-quality image with exceptional detail."""
content_prompt = """A vintage porcelain collector's item. Beneath a blossoming cherry tree in early spring, 
this treasure is photographed up close, with soft pink petals drifting through the air and vibrant blossoms framing the scene."""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1024,
    upsampling_height=1024,
    upsampling_strength=0.2,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Utilize each pipeline independently 

```python
import torch
from diffusers import VisualClozeGenerationPipeline, FluxFillPipeline as VisualClozeUpsamplingPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = VisualClozeGenerationPipeline.from_pretrained(
    "VisualCloze/VisualClozePipeline-384", resolution=384, dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

image_paths = [
    # in-context examples
    [
        load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
        ),
        load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
        ),
    ],
    # query with the target image
    [
        load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
        ),
        None,  # No image needed for the target image
    ],
]
task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."

# Stage 1: Generate initial image
image = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0][0]

# Stage 2 (optional): Upsample the generated image
pipe_upsample = VisualClozeUpsamplingPipeline.from_pipe(pipe)
pipe_upsample.to("cuda")

mask_image = Image.new("RGB", image.size, (255, 255, 255))

image = pipe_upsample(
    image=image,
    mask_image=mask_image,
    prompt=content_prompt,
    width=1344,
    height=768,
    strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("visualcloze.png")
```

## VisualClozePipeline[[diffusers.VisualClozePipeline]]

#### diffusers.VisualClozePipeline[[diffusers.VisualClozePipeline]]

```python
diffusers.VisualClozePipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: T5Tokenizer, transformer: FluxTransformer2DModel, resolution: int = 384)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py#L89)

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

resolution (`int`, *optional*, defaults to 384) : The resolution of each image when concatenating images from the query and in-context examples.

The VisualCloze pipeline for image generation with visual context. Reference:
https://github.com/lzyhha/VisualCloze/tree/main. This pipeline is designed to generate images based on visual
in-context examples.

#### __call__[[diffusers.VisualClozePipeline.__call__]]

```python
__call__(task_prompt: str | list[str] = None, content_prompt: str | list[str] = None, image: typing.Optional[torch.FloatTensor] = None, upsampling_height: int | None = None, upsampling_width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float = 30.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512, upsampling_strength: float = 1.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_combined.py#L252)

**Parameters:**

task_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the task intention.

content_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the content or caption of the target image to be generated.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.

upsampling_height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image (i.e., output image) after upsampling via SDEdit. By default, the image is upsampled by a factor of three, and the base resolution is determined by the resolution parameter of the pipeline. When only one of `upsampling_height` or `upsampling_width` is specified, the other will be automatically set based on the aspect ratio.

upsampling_width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image (i.e., output image) after upsampling via SDEdit. By default, the image is upsampled by a factor of three, and the base resolution is determined by the resolution parameter of the pipeline. When only one of `upsampling_height` or `upsampling_width` is specified, the other will be automatically set based on the aspect ratio.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 30.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

joint_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

upsampling_strength (`float`, *optional*, defaults to 1.0) : Indicates extent to transform the reference `image` when upsampling the results. Must be between 0 and 1. The generated image is used as a starting point and more noise is added the higher the `upsampling_strength`. The number of denoising steps depends on the amount of noise initially added. When `upsampling_strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 0 skips the upsampling step and output the results at the resolution of `self.resolution`.

**Returns:** `~pipelines.flux.FluxPipelineOutput` or `tuple`

`~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.

Function invoked when calling the VisualCloze pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import VisualClozePipeline
>>> from diffusers.utils import load_image

>>> image_paths = [
...     # in-context examples
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
...         ),
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
...         ),
...     ],
...     # query with the target image
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
...         ),
...         None,  # No image needed for the target image
...     ],
... ]
>>> task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
>>> content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."
>>> pipe = VisualClozePipeline.from_pretrained(
...     "VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     task_prompt=task_prompt,
...     content_prompt=content_prompt,
...     image=image_paths,
...     upsampling_width=1344,
...     upsampling_height=768,
...     upsampling_strength=0.4,
...     guidance_scale=30,
...     num_inference_steps=30,
...     max_sequence_length=512,
...     generator=torch.Generator("cpu").manual_seed(0),
... ).images[0][0]
>>> image.save("visualcloze.png")
```

## VisualClozeGenerationPipeline[[diffusers.VisualClozeGenerationPipeline]]

#### diffusers.VisualClozeGenerationPipeline[[diffusers.VisualClozeGenerationPipeline]]

```python
diffusers.VisualClozeGenerationPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: T5Tokenizer, transformer: FluxTransformer2DModel, resolution: int = 384)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py#L118)

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

resolution (`int`, *optional*, defaults to 384) : The resolution of each image when concatenating images from the query and in-context examples.

The VisualCloze pipeline for image generation with visual context. Reference:
https://github.com/lzyhha/VisualCloze/tree/main This pipeline is designed to generate images based on visual
in-context examples.

#### __call__[[diffusers.VisualClozeGenerationPipeline.__call__]]

```python
__call__(task_prompt: str | list[str] = None, content_prompt: str | list[str] = None, image: typing.Optional[torch.FloatTensor] = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float = 30.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py#L659)

**Parameters:**

task_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the task intention.

content_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the content or caption of the target image to be generated.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 30.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between `"pil"` ([PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

joint_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.flux.FluxPipelineOutput` or `tuple`

`~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.

Function invoked when calling the VisualCloze pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import VisualClozeGenerationPipeline, FluxFillPipeline as VisualClozeUpsamplingPipeline
>>> from diffusers.utils import load_image
>>> from PIL import Image

>>> image_paths = [
...     # in-context examples
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
...         ),
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
...         ),
...     ],
...     # query with the target image
...     [
...         load_image(
...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
...         ),
...         None,  # No image needed for the target image
...     ],
... ]
>>> task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
>>> content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."
>>> pipe = VisualClozeGenerationPipeline.from_pretrained(
...     "VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     task_prompt=task_prompt,
...     content_prompt=content_prompt,
...     image=image_paths,
...     guidance_scale=30,
...     num_inference_steps=30,
...     max_sequence_length=512,
...     generator=torch.Generator("cpu").manual_seed(0),
... ).images[0][0]

>>> # optional, upsampling the generated image
>>> pipe_upsample = VisualClozeUpsamplingPipeline.from_pipe(pipe)
>>> pipe_upsample.to("cuda")

>>> mask_image = Image.new("RGB", image.size, (255, 255, 255))

>>> image = pipe_upsample(
...     image=image,
...     mask_image=mask_image,
...     prompt=content_prompt,
...     width=1344,
...     height=768,
...     strength=0.4,
...     guidance_scale=30,
...     num_inference_steps=30,
...     max_sequence_length=512,
...     generator=torch.Generator("cpu").manual_seed(0),
... ).images[0]

>>> image.save("visualcloze.png")
```

#### encode_prompt[[diffusers.VisualClozeGenerationPipeline.encode_prompt]]

```python
encode_prompt(layout_prompt: str | list[str], task_prompt: str | list[str], content_prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, max_sequence_length: int = 512, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/visualcloze/pipeline_visualcloze_generation.py#L292)

**Parameters:**

layout_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the number of in-context examples and the number of images involved in the task.

task_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the task intention.

content_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to define the content or caption of the target image to be generated.

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

lora_scale (`float`, *optional*) : A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.

### Hunyuan-DiT
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/hunyuandit.md
