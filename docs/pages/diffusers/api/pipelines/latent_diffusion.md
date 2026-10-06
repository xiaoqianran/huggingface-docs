# Latent Diffusion

Latent Diffusion was proposed in [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.

The abstract from the paper is:

*By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.*

The original codebase can be found at [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

## LDMTextToImagePipeline[[diffusers.LDMTextToImagePipeline]]

#### diffusers.LDMTextToImagePipeline[[diffusers.LDMTextToImagePipeline]]

```python
diffusers.LDMTextToImagePipeline(vqvae: diffusers.models.autoencoders.vq_model.VQModel | diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL, bert: PreTrainedModel, tokenizer: PythonBackend, unet: diffusers.models.unets.unet_2d.UNet2DModel | diffusers.models.unets.unet_2d_condition.UNet2DConditionModel, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_pndm.PNDMScheduler | diffusers.utils.dummy_torch_and_scipy_objects.LMSDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L39)

**Parameters:**

vqvae ([VQModel](/docs/diffusers/v0.41.0/en/api/models/vq#diffusers.VQModel)) : Vector-quantized (VQ) model to encode and decode images to and from latent representations.

bert (`LDMBertModel`) : Text-encoder model based on `BERT`.

tokenizer ([BertTokenizer](https://huggingface.co/docs/transformers/v5.18.0/en/model_doc/layoutlm#transformers.BertTokenizer)) : A `BertTokenizer` to tokenize text.

unet ([UNet2DConditionModel](/docs/diffusers/v0.41.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : A `UNet2DConditionModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.41.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

Pipeline for text-to-image generation using latent diffusion.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.LDMTextToImagePipeline.__call__]]

```python
__call__(prompt: str | list[str], height: int | None = None, width: int | None = None, num_inference_steps: int | None = 50, guidance_scale: float | None = 1.0, eta: float | None = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L74)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 1.0) : A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the [DDIMScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `"pil"` (`PIL.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

The call function to the pipeline for generation.

Example:

```py
>>> from diffusers import DiffusionPipeline

>>> # load model and scheduler
>>> ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

>>> # run pipeline in inference (sample random noise and denoise)
>>> prompt = "A painting of a squirrel eating a burger"
>>> images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

>>> # save images
>>> for idx, image in enumerate(images):
...     image.save(f"squirrel-{idx}.png")
```

## LDMSuperResolutionPipeline[[diffusers.LDMSuperResolutionPipeline]]

#### diffusers.LDMSuperResolutionPipeline[[diffusers.LDMSuperResolutionPipeline]]

```python
diffusers.LDMSuperResolutionPipeline(vqvae: VQModel, unet: UNet2DModel, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_pndm.PNDMScheduler | diffusers.utils.dummy_torch_and_scipy_objects.LMSDiscreteScheduler | diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler | diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler | diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py#L39)

**Parameters:**

vqvae ([VQModel](/docs/diffusers/v0.41.0/en/api/models/vq#diffusers.VQModel)) : Vector-quantized (VQ) model to encode and decode images to and from latent representations.

unet ([UNet2DModel](/docs/diffusers/v0.41.0/en/api/models/unet2d#diffusers.UNet2DModel)) : A `UNet2DModel` to denoise the encoded image.

scheduler ([SchedulerMixin](/docs/diffusers/v0.41.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of [DDIMScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), [EulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler), [EulerAncestralDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler), [DPMSolverMultistepScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler), or [PNDMScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

A pipeline for image super-resolution using latent diffusion.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.LDMSuperResolutionPipeline.__call__]]

```python
__call__(image: typing.Union[torch.Tensor, PIL.Image.Image] = None, batch_size: int | None = 1, num_inference_steps: int | None = 100, eta: float | None = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str | None = 'pil', return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py#L71)

**Parameters:**

image (`torch.Tensor` or `PIL.Image.Image`) : `Image` or tensor representing an image batch to be used as the starting point for the process.

batch_size (`int`, *optional*, defaults to 1) : Number of images to generate.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the [DDIMScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `"pil"` (`PIL.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images

The call function to the pipeline for generation.

Example:

```py
>>> import requests
>>> from PIL import Image
>>> from io import BytesIO
>>> from diffusers import LDMSuperResolutionPipeline
>>> import torch

>>> # load model and scheduler
>>> pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
>>> pipeline = pipeline.to("cuda")

>>> # let's download an  image
>>> url = (
...     "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
... )
>>> response = requests.get(url)
>>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
>>> low_res_img = low_res_img.resize((128, 128))

>>> # run pipeline in inference (sample random noise and denoise)
>>> upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
>>> # save image
>>> upscaled_image.save("ldm_generated_image.png")
```

## ImagePipelineOutput[[diffusers.ImagePipelineOutput]]

#### diffusers.ImagePipelineOutput[[diffusers.ImagePipelineOutput]]

```python
diffusers.ImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/pipeline_utils.py#L135)

**Parameters:**

images (`List[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width, num_channels)`.

Output class for image pipelines.

### Ltx2
https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/ltx2.md

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

# LTX-2

  

[LTX-2](https://hf.co/papers/2601.03233) is a DiT-based foundation model designed to generate synchronized video and audio within a single model. It brings together the core building blocks of modern video generation, with open weights and a focus on practical, local execution.

You can find all the original LTX-Video checkpoints under the [Lightricks](https://huggingface.co/Lightricks) organization.

The original codebase for LTX-2 can be found [here](https://github.com/Lightricks/LTX-2).

## Two-stages Generation

The shared `LTX2Pipeline` / `LTX2ImageToVideoPipeline` `__call__` defaults match the LTX-2.5 reference (`num_inference_steps=30`; `num_frames` is optional when a `duration_head` is present, otherwise it falls back to `121`). The examples below use those defaults for LTX-2.0/2.3 as well.

Recommended pipeline to achieve production quality generation, this pipeline is composed of two stages:

- Stage 1: Generate a video at the target resolution using diffusion sampling with classifier-free guidance (CFG). This stage produces a coherent low-noise video sequence that respects the text/image conditioning.
- Stage 2: Upsample the Stage 1 output by 2 and refine details using a distilled LoRA model to improve fidelity and visual quality. Stage 2 may apply lighter CFG to preserve the structure from Stage 1 while enhancing texture and sharpness.

Sample usage of text-to-video two stages pipeline

```py
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda:0"
width = 768
height = 512

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2", dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

# Stage 1 default (non-distilled) inference
frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    sigmas=None,
    guidance_scale=3.0,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/LTX-2",
    subfolder="latent_upsampler",
    dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

# Load Stage 2 distilled LoRA
pipe.load_lora_weights(
    "Lightricks/LTX-2", adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors"
)
pipe.set_adapters("stage_2_distilled", 1.0)
# VAE tiling is usually necessary to avoid OOM error when VAE decoding
pipe.vae.enable_tiling()
# Change scheduler to use Stage 2 distilled sigmas as is
new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)
pipe.scheduler = new_scheduler
# Stage 2 inference with distilled LoRA and sigmas
video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], # renoise with first sigma value https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py#L218
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_lora_distilled_sample.mp4",
)
```

## Distilled checkpoint generation
Fastest two-stages generation pipeline using a distilled checkpoint.

```py
import torch
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2Pipeline.from_pretrained(
    model_path, dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], # renoise with first sigma value https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/distilled.py#L178
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_distilled_sample.mp4",
)
```

## Condition Pipeline Generation

You can use `LTX2ConditionPipeline` to specify image and/or video conditions at arbitrary latent indices. For example, we can specify both a first-frame and last-frame condition to perform first-last-frame-to-video (FLF2V) generation:

```py
import torch
from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from diffusers.utils import load_image

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are "
    "delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright "
    "sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, "
    "low-angle perspective."
)

first_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png",
)
last_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png",
)
first_cond = LTX2VideoCondition(frames=first_image, index=0, strength=1.0)
last_cond = LTX2VideoCondition(frames=last_image, index=-1, strength=1.0)
conditions = [first_cond, last_cond]

frame_rate = 24.0
video_latent, audio_latent = pipe(
    conditions=conditions,
    prompt=prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    width=width * 2,
    height=height * 2,
    num_inference_steps=3,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_distilled_flf2v.mp4",
)
```

You can use both image and video conditions:

```py
import torch
from diffusers import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image, load_video

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is "
    "divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features "
    "dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered "
    "clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, "
    "with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The "
    "landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the "
    "solitude and beauty of a winter drive through a mountainous region."
)

cond_video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
)
cond_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
)
video_cond = LTX2VideoCondition(frames=cond_video, index=0, strength=1.0)
image_cond = LTX2VideoCondition(frames=cond_image, index=8, strength=1.0)
conditions = [video_cond, image_cond]

frame_rate = 24.0
video, audio = pipe(
    conditions=conditions,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_cond_video.mp4",
)
```

Because the conditioning is done via latent frames, the 8 data space frames corresponding to the specified latent frame for an image condition will tend to be static.

## Multimodal Guidance

LTX-2.X pipelines support multimodal guidance. It is composed of three terms, all using a CFG-style update rule:

1. Classifier-Free Guidance (CFG): standard [CFG](https://huggingface.co/papers/2207.12598) where the perturbed ("weaker") output is generated using the negative prompt.
2. Spatio-Temporal Guidance (STG): [STG](https://huggingface.co/papers/2411.18664) moves away from a perturbed output created from short-cutting self-attention operations and substitutes in the attention values instead. The idea is that this creates sharper videos and better spatiotemporal consistency.
3. Modality Isolation Guidance: moves away from a perturbed output created from disabling cross-modality (audio-to-video and video-to-audio) cross attention. This guidance is more specific to [LTX-2.X](https://huggingface.co/papers/2601.03233) models, with the idea that this produces better consistency between the generated audio and video.

These are controlled by the `guidance_scale`, `stg_scale`, and `modality_scale` arguments and can be set separately for video and audio. Additionally, for STG the transformer block indices where self-attention is skipped needs to be specified via the `spatio_temporal_guidance_blocks` argument. The LTX-2.X pipelines also support [guidance rescaling](https://huggingface.co/papers/2305.08891) to help reduce over-exposure, which can be a problem when the guidance scales are set to high values.

```py
import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "diffusers/LTX-2.3-Diffusers"

pipe = LTX2ImageToVideoPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    "breath-taking, movie-like shot."
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
)

video, audio = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,  # Recommended LTX-2.3 guidance parameters
    stg_scale=1.0,  # Note that 0.0 (not 1.0) means that STG is disabled (all other guidance is disabled at 1.0)
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,  # Note that a higher CFG guidance scale is recommended for audio
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_3_i2v_stage_1.mp4",
)
```

## Prompt Enhancement

The LTX-2.X models are sensitive to prompting style. Refer to the [official prompting guide](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) for recommendations on how to write a good prompt. Using prompt enhancement, where the supplied prompts are enhanced using the pipeline's text encoder (by default a [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) model) given a system prompt, can also improve sample quality. The optional `processor` pipeline component needs to be present to use prompt enhancement. Enable it with `enable_prompt_enhancement=True` and a `system_prompt` (opt-in, matching the Lightricks reference pipelines):

```py
import torch
from transformers import Gemma3Processor
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, T2V_DEFAULT_SYSTEM_PROMPT

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "diffusers/LTX-2.3-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()
if getattr(pipe, "processor", None) is None:
    processor = Gemma3Processor.from_pretrained("google/gemma-3-12b-it-qat-q4_0-unquantized")
    pipe.processor = processor

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    "breath-taking, movie-like shot."
)

video, audio = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    enable_prompt_enhancement=True,
    system_prompt=T2V_DEFAULT_SYSTEM_PROMPT,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_3_t2v_stage_1.mp4",
)
```

## LTX-2.5

LTX-2.5 reuses the same `LTX2Pipeline`/`LTX2VideoTransformer3DModel`/`AutoencoderKLLTX2Video`/etc. classes as LTX-2.3 — there is no separate pipeline class for it. The user-visible difference is the text encoder: LTX-2.5 is paired with a Gemma 4 (`gemma4_unified`) checkpoint instead of Gemma 3. This is loaded automatically when you call `from_pretrained` on a converted LTX-2.5 checkpoint (via the `transformers` `Auto*` classes), so no extra setup is needed at inference time — just point `from_pretrained` at an LTX-2.5 repo instead of an LTX-2.3 one.

[`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) ships both transformers: the **distilled** DiT in `transformer/`, which is what `model_index.json` points at, and the full/SFT DiT in `transformer_full/`, which has to be loaded explicitly (see [Full / SFT transformer](#full--sft-transformer)). The repo's `scheduler/` is configured for the distilled checkpoint (`use_dynamic_shifting=False`, `shift_terminal=None`) so that its sigma schedule is used exactly as given. Everything [two-stage generation](#two-stage-generation-for-ltx-25) needs is shipped there too: a `latent_upsampler/` subfolder and the stage 2 distilled LoRA, `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`, at the root of the repo.

Distilled inference is driven by an explicit sigma schedule rather than a step count, and runs unguided (`guidance_scale=1.0`, so `negative_prompt` is unused). Passing `num_inference_steps` instead would hand the model a generic linear schedule and quietly cost quality:

```py
import torch
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "Lightricks/LTX-2.5-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

video, audio = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v.mp4",
)
```

### Two-stage generation for LTX-2.5

LTX-2.5 supports both two-stage variants, and `DISTILLED_SIGMA_VALUES` / `STAGE_2_DISTILLED_SIGMA_VALUES` are its reference schedules:

- **Distilled checkpoint, both stages** — the reference recipe for the default `transformer/`, and the one shown below. No stage 2 LoRA is involved, since the transformer is already distilled; this is [Distilled checkpoint generation](#distilled-checkpoint-generation) with LTX-2.5 weights.
- **Full/SFT stage 1 + distilled LoRA stage 2** — [Two-stages Generation](#two-stages-generation) as described at the top of this page, using `transformer_full/` and the shipped LoRA. See [below](#stage-2-with-the-distilled-lora) for what changes.

Stage 1 runs at half the target resolution, the upsampler doubles it, and stage 2 refines at full resolution — video *and* audio, both reseeded from the stage 1 latents at `noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0]`. Height and width must be divisible by 64, since stage 1 halves each axis and still has to land on the VAE's spatial grid.

```py
import torch
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda"  # or "mps", "xpu", "cpu"
width = 1536
height = 1024
num_frames = 121
frame_rate = 24.0
model_path = "Lightricks/LTX-2.5-Diffusers"

# One generator for the whole call, threaded through both stages, so stage 2 continues the noise
# stream instead of repeating stage 1's draw.
generator = torch.Generator(device).manual_seed(42)

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

# Stage 1: half resolution, 8 distilled sigmas
video_latent, audio_latent = pipe(
    prompt=prompt,
    width=width // 2,
    height=height // 2,
    num_frames=num_frames,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
# `latents_normalized=False`: `output_type="latent"` already applied the latent statistics, and the
# upsampler is trained on denormalized latents. Stage 2 renormalizes them in `prepare_latents`.
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    latents_normalized=False,
    output_type="latent",
    return_dict=False,
)[0]

# Stage 2: full resolution, 3 sigmas, reseeded from stage 1. Pass `num_frames` explicitly here --
# omitting it would run the duration head a second time instead of using the stage 1 length.
pipe.vae.enable_tiling()
video, audio = pipe(
    prompt=prompt,
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    width=width,
    height=height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],  # renoise with the stage 2 entry sigma
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v_two_stages.mp4",
)
```

When the length comes from the [duration head](#automatic-duration-for-ltx-25) rather than an explicit `num_frames`, let stage 1 decide and recover the realized length from its latents (`[B, C, F, H, W]`) before stage 2 runs, instead of predicting a second time:

```py
num_frames = (video_latent.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1
```

#### Stage 2 with the distilled LoRA

To run [Two-stages Generation](#two-stages-generation) instead — full/SFT DiT for stage 1, distilled LoRA for stage 2 — build the pipeline as in [Full / SFT transformer](#full--sft-transformer) and generate stage 1 latents with that guidance stack. Two things then differ from LTX-2.0/2.3. The LoRA lives in the diffusers repo itself rather than alongside the original weights, and the scheduler flip goes the other way round: LTX-2.5 ships the *distilled* scheduler config, so stage 1 is what turned dynamic shifting on, and stage 2 turns it back off.

```py
pipe.load_lora_weights(
    "Lightricks/LTX-2.5-Diffusers",
    adapter_name="stage_2_distilled",
    weight_name="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
)
pipe.set_adapters("stage_2_distilled", 1.0)
pipe.vae.enable_tiling()

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)
```

The upsample step and the stage 2 call itself are unchanged from the distilled recipe above: same `sigmas=STAGE_2_DISTILLED_SIGMA_VALUES`, same `noise_scale`, and `guidance_scale=1.0`, since stage 2 is running a distilled model either way.

### Convolutional and diffusion decoding

LTX-2.5 ships two video decoders over the same latent space, so latents are interchangeable between them:

- `vae/` — the convolutional VAE ([AutoencoderKLLTX2Video](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_ltx_2#diffusers.AutoencoderKLLTX2Video)). It is what the pipelines decode with, so every snippet above already uses it, and it is the only one of the two that tiles (`pipe.vae.enable_tiling()`), which is usually what makes a high resolution fit.
- `diffusion_decoder/` — [LTX2VideoDiffusionDecoderModel](/docs/diffusers/v0.41.0/en/api/models/ltx2_diffusion_decoder#diffusers.LTX2VideoDiffusionDecoderModel). It is a diffusion model in its own right rather than a pipeline component, so it is not passed as a `vae`: run the pipeline with `output_type="latent"` and hand the latents to [LTX2VideoDiffusionDecodePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2VideoDiffusionDecodePipeline).

Encoding always goes through `vae/`, so image and video conditioning are unaffected by the choice.

Two things change when you decode with the diffusion decoder. `output_type="latent"` also skips the vocoder, so the audio comes back as latents and has to be finished by hand, and the NATTEN processor is effectively required at video resolutions:

```py
import torch
from diffusers import LTX2Pipeline, LTX2VideoDiffusionDecodePipeline, LTX2VideoDiffusionDecoderModel
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda"  # or "mps", "xpu", "cpu"
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(42)
model_path = "Lightricks/LTX-2.5-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

latents, audio_latents = pipe(
    prompt=prompt,
    width=960,
    height=544,
    num_frames=121,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

# `output_type="latent"` skips the vocoder, so finish the audio by hand. These latents are already
# denormalized, which is what `audio_vae.decode` expects.
mel = pipe.audio_vae.decode(audio_latents.to(pipe.audio_vae.dtype), return_dict=False)[0]
audio = pipe.vocoder(mel)

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    model_path, subfolder="diffusion_decoder", dtype=torch.bfloat16
).to(device)
# The decoder runs on the `flex` backend by default, and uncompiled `flex_attention` materializes the
# full score matrix -- tens of GB at video resolutions. NATTEN's kernels are what the original
# implementation uses; they are fetched from the Hub by `kernels` (`pip install kernels`), not from a
# local NATTEN build. Switching the attention *backend* instead raises: only `flex` takes the BlockMask.
decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
# Decode in overlapping tiles so peak memory scales with the tile size rather than the video size.
decoder.enable_tiling()

decode_pipe = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=pipe.scheduler)

# `denormalize=False`: `output_type="latent"` already applied the latent statistics, so applying them
# again would rescale every channel by its std a second time. The decoder draws the noise it denoises,
# so pass a generator to make decoding reproducible.
video = decode_pipe(
    latents, generator=generator, output_type="np", denormalize=False, return_dict=False
)[0]

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v_diffusion_decode.mp4",
)
```

To combine this with [two-stage generation](#two-stage-generation-for-ltx-25), ask *stage 2* for `output_type="latent"` and decode that.

`decoder.enable_tiling()` is what keeps a high resolution decode in memory, the same way `pipe.vae.enable_tiling()` does for the convolutional VAE. The memory-dominant part of the decode — the last upsampling stage and the diffusion stage — then runs on overlapping tiles that are blended back together, so peak memory is bounded by the tile size instead of the video size. Tiling only kicks in once the latent exceeds one tile, and the tile and overlap sizes can be tuned via the `tile_sample_min_*` / `tile_sample_stride_*` arguments (defaults match the reference implementation). Since the diffusion stage denoises each tile separately, a tiled decode does not reproduce the untiled result exactly.

On a single card it is also worth moving the pipeline out of the way before decoding (`pipe.to("cpu")` and `torch.cuda.empty_cache()`, after capturing `pipe.scheduler` and the vocoder's `output_sampling_rate`), since the decoder needs its own headroom. See [LTX2VideoDiffusionDecoderModel](/docs/diffusers/v0.41.0/en/api/models/ltx2_diffusion_decoder#diffusers.LTX2VideoDiffusionDecoderModel) for the attention backends, the tiling details, and the rest of the decoder's behaviour.

### Full / SFT transformer

`transformer_full/` is not referenced by `model_index.json`, so load it explicitly. It also needs a different scheduler and a real guidance stack: the shipped `scheduler/` is configured for the distilled checkpoint, and the guidance defaults are LTX-2.0-era generics that leave an LTX-2.5 SFT run visibly under-guided without raising anything. The [Multimodal Guidance](#multimodal-guidance) recommendations apply here unchanged, including STG on block `28`:

```py
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

device = "cuda"  # or "mps", "xpu", "cpu"
model_path = "Lightricks/LTX-2.5-Diffusers"

# Passing `transformer=` keeps `from_pretrained` from fetching the distilled folder as well.
transformer = LTX2VideoTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer_full", dtype=torch.bfloat16
)
pipe = LTX2Pipeline.from_pretrained(model_path, transformer=transformer, dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

# Re-enable dynamic shifting and the terminal shift, which the distilled configuration turns off.
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=True, shift_terminal=0.1
)

video, audio = pipe(
    prompt="A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees.",
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    generator=torch.Generator(device).manual_seed(42),
    output_type="np",
    return_dict=False,
)
```

Drop `sigmas` here — the full DiT takes its schedule from the scheduler.

### Prompt Enhancement for LTX-2.5

**Using prompt enhancement is strongly recommended for LTX-2.5; pass `enable_prompt_enhancement=True` to opt in** (same as the Lightricks reference pipelines). Unlike LTX-2.0/2.3, where the same text encoder checkpoint doubles as the enhancer (see [Prompt Enhancement](#prompt-enhancement) above), LTX-2.5's fine-tuned text encoder was not trained for enhancement. Instead, enhancement uses a separate, off-the-shelf `google/gemma-4-E2B-it` checkpoint. Load it into the pipeline's optional `prompt_enhancer`/`processor` components, then enable enhancement — the pipeline defaults to `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and the Gemma 4 recipe (`do_sample=False`, `no_repeat_ngram_size=5`, `max_new_tokens=600`). Pass an explicit `system_prompt=` to override:

```py
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

device = "cuda"  # or "mps", "xpu", "cpu"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "Lightricks/LTX-2.5-Diffusers"
enhancer_model_id = "google/gemma-4-E2B-it"

pipe = LTX2Pipeline.from_pretrained(model_path, dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()
if getattr(pipe, "prompt_enhancer", None) is None:
    pipe.prompt_enhancer = AutoModelForImageTextToText.from_pretrained(enhancer_model_id)
    pipe.processor = AutoProcessor.from_pretrained(enhancer_model_id)

prompt = "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."

video, audio = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    enable_prompt_enhancement=True,
    # No `system_prompt=` needed -- defaults to `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` when `prompt_enhancer` is set.
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_t2v_enhanced.mp4",
)
```

The same applies to image-to-video with `LTX2ImageToVideoPipeline`: set `pipe.prompt_enhancer`/`pipe.processor` the same way and pass `enable_prompt_enhancement=True` (using `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`, conditioning on both the reference image and the text prompt) — again, no `system_prompt=` needed unless you want to override it.

### Automatic duration for LTX-2.5

LTX-2.5 checkpoints ship a small `duration_head` that predicts how long the described shot should be, from the same text-connector output the transformer is conditioned on. When the loaded pipeline has one, **`num_frames` is auto-predicted by default** — omit it and the model chooses the length:

```py
video, audio = pipe(prompt=prompt, output_type="np", return_dict=False)
```

To set the length yourself, pass `num_frames` explicitly. An integer always wins over the head:

```py
video, audio = pipe(prompt=prompt, num_frames=121, output_type="np", return_dict=False)
```

Pipelines loaded from LTX-2.0 or LTX-2.3 checkpoints have no duration head and keep the previous default of 121 frames, so this changes nothing for them.

Pass `min_seconds` / `max_seconds` to constrain the prediction. The raw prediction is clamped into the range, then converted to frames:

```py
video, audio = pipe(
    prompt=prompt,
    min_seconds=2.0,
    max_seconds=10.0,
    frame_rate=frame_rate,
    output_type="np",
    return_dict=False,
)
```

Predicted frame counts are snapped to the VAE's causal temporal grid (`8k + 1`), so the realized duration is quantized — about 0.33s per step at 24 fps — and it shifts with `frame_rate`, since the head predicts seconds rather than frames. `min_seconds` must be strictly less than `max_seconds`. These bounds are ignored when `num_frames` is set explicitly.

Bounds narrower than one grid step may not be satisfiable exactly: at 24 fps `[1.0s, 1.02s]` converts to `[24, 24]` frames, and 24 is not `8k + 1`. The nearest grid point is used and a warning is logged, so the returned length can fall just outside bounds that tight.

To inspect a prediction without generating a video, call the head directly. Everything it needs is public:

```py
prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False)
video_tokens, audio_tokens, _ = pipe.connectors(prompt_embeds, prompt_attention_mask)

num_frames = pipe.duration_head.predict_num_frames(
    video_tokens,
    audio_tokens,
    frame_rate=24.0,
    temporal_compression_ratio=pipe.vae_temporal_compression_ratio,
)
seconds = pipe.duration_head(video_tokens, audio_tokens).item()  # raw, before clamping
print(f"predicted {seconds:.2f}s -> {num_frames} frames")
```

Converting a 2.5 checkpoint picks the head up automatically with `--full_pipeline`, or on its own with `--duration_head`. Checkpoints predating 2.5 have no such weights, and conversion skips the component rather than failing.

### LTX-2.5 Modular

LTX-2.5 is also available as a modular pipeline. The default blockset uses the diffusion decoder and predicts the video duration when `num_frames` is omitted. It applies guidance separately to video and audio through the `guider` and `audio_guider` components. See [LTX2Guidance](/docs/diffusers/v0.41.0/en/api/modular_diffusers/guiders#diffusers.LTX2Guidance) for the available guidance parameters. By default, the modular pipeline will download the prompt enhancer and processor from the [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) repo. Below is a T2V modular example:

```py
import torch
from diffusers import ModularPipeline, ComponentsManager
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import encode_video

device = "cuda"  # or "mps", "xpu", "cpu"
frame_rate = 24.0
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)

model_path = "Lightricks/LTX-2.5-Diffusers"

cm = ComponentsManager()
pipe = ModularPipeline.from_pretrained(model_path, components_manager=cm)
pipe.load_components(dtype=torch.bfloat16)
# Set memory_reserve_margin higher to more aggressively offload component models
cm.enable_auto_cpu_offload(device=device, memory_reserve_margin="20GB")
# The NATTEN processor works if `kernels` is available (`pip install kernels`)
# Otherwise omit the below line to use the Flex Attention processor
pipe.diffusion_decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
pipe.diffusion_decoder.enable_tiling()

prompt = (
    "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."
)

output_state = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=None,  # Set to an int (e.g. 121) to specify a fixed video length
    frame_rate=frame_rate,
    num_inference_steps=30,
    use_cross_timestep=True,
    enable_prompt_enhancement=True,
    generator=generator,
    output_type="np",
)
video = output_state.get("videos")
audio = output_state.get("audio")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_modular_t2v.mp4",
)
```

The modular pipeline will automatically switch workflows based on the supplied inputs. For example, if `image` is supplied, an I2V workflow will be used:

```py
import torch
from diffusers import ModularPipeline, ComponentsManager
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeNeighborhoodNattenProcessor
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import encode_video, load_image

device = "cuda"  # or "mps", "xpu", "cpu"
frame_rate = 24.0
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)

model_path = "Lightricks/LTX-2.5-Diffusers"

cm = ComponentsManager()
pipe = ModularPipeline.from_pretrained(model_path, components_manager=cm)
pipe.load_components(dtype=torch.bfloat16)
cm.enable_auto_cpu_offload(device=device, memory_reserve_margin="20GB")
pipe.diffusion_decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
pipe.diffusion_decoder.enable_tiling()

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion."
)
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
image = load_image(image_path)

output_state = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=768,
    height=512,
    num_frames=None,  # Set to an int (e.g. 121) to specify a fixed video length
    frame_rate=frame_rate,
    num_inference_steps=30,
    use_cross_timestep=True,
    enable_prompt_enhancement=True,
    generator=generator,
    output_type="np",
)
video = output_state.get("videos")
audio = output_state.get("audio")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_modular_i2v.mp4",
)
```

You can see the supported workflows in the docs for each blockset (e.g. [LTX2AutoBlocks](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2AutoBlocks), [LTX25AutoBlocks](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX25AutoBlocks)).

### Diffusion Fidelity Rendering (DFR) for LTX-2.5

`LTX2DFRPipeline` trades wall-clock time for detail fidelity. Each `__call__` is **one denoise pass** at `height` × `width`: it generates video plus extra single-pixel-frame **keyframe slots**, or re-denoises supplied latents seeded from those slots. Callers compose stages the same way as other LTX two-stage pipelines — this pipeline, [LTX2LatentUpsamplePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2LatentUpsamplePipeline), this pipeline again, then [LTX2DFRTemporalRefinePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRTemporalRefinePipeline) for each temporal round.

A slot costs a full latent frame of tokens to buy one pixel frame, which relaxes the effective temporal compression at that position — so the surrounding video is conditioned on genuinely new frames instead of interpolated ones. Slot positions come from a segment grid aligned to the VAE's temporal border (24 or 32 pixel frames, whichever pads the request less). The canvas is padded to a whole number of segments internally; `output_type="latent"` returns that padded grid so a slot on the pad is not dropped. Trim with `trim_canvas` before VAE decode.

This needs a transformer whose config sets `use_keyframes_abs_pos_embedding`, which marks single-pixel-frame latents with a learned embedding. LTX-2.5 checkpoints ship it; the pipeline raises on anything older rather than spending the token budget on tokens it cannot interpret.

Budget for the extra tokens: each slot adds one latent frame's worth, so stage 2 runs a longer sequence than the equivalent two-stage distilled pass — +31% at 1024x1536 / 121 frames (24576 -> 32256 tokens, 5 slots on a 24-frame segment grid). Peak activation memory scales with that, so a resolution that just fits the plain distilled recipe may need `enable_sequential_cpu_offload`, `vae.enable_tiling()`, or a smaller canvas under DFR.

Composition uses `return_dict=True` for `keyframes` and `keyframe_positions` (`return_dict=False` returns the same four fields as a tuple).

The full recipe below is the one worth starting from: 1088×1920 image-to-video, one x2 temporal refine round, and the x2 spatial detailing IC-LoRA on stage 2.

```py
import torch
from diffusers import (
    LTX2DFRPipeline,
    LTX2DFRTemporalRefinePipeline,
    LTX2LatentUpsamplePipeline,
    LTXEulerAncestralRFScheduler,
)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import trim_canvas
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video, load_image

pipe = LTX2DFRPipeline.from_pretrained("Lightricks/LTX-2.5-Diffusers", torch_dtype=torch.bfloat16)
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/LTX-2.5-Diffusers", subfolder="latent_upsampler", torch_dtype=torch.bfloat16
)
# The x2 temporal upsampler is not in the published `model_index.json` — convert it with
# `--temporal_latent_upsampler`.
temporal_latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "path/to/converted/temporal_latent_upsampler", torch_dtype=torch.bfloat16
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
temporal_pipe = LTX2DFRTemporalRefinePipeline(
    scheduler=LTXEulerAncestralRFScheduler(eta=0.5),
    vae=pipe.vae,
    audio_vae=pipe.audio_vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    connectors=pipe.connectors,
    transformer=pipe.transformer,
    vocoder=pipe.vocoder,
    temporal_latent_upsampler=temporal_latent_upsampler,
)

# All three pipelines share the same components, so place them together. Do not call
# `enable_model_cpu_offload()` on one of them: its hooks would own modules the other two also call,
# and `temporal_latent_upsampler` — held only by `temporal_pipe` — would never reach the device.
pipe.to("cuda")
upsample_pipe.to("cuda")
temporal_pipe.to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
prompt = "A tabby cat stretching in a sunlit window, dust motes drifting in the light"
conditions = [LTX2VideoCondition(frames=image, index=0, strength=1.0)]
height, width = 1088, 1920
frame_rate = 24.0
generator = torch.Generator(device="cuda").manual_seed(0)

num_frames = 121
out = pipe(
    prompt=prompt,
    conditions=conditions,
    height=height // 2,
    width=width // 2,
    num_frames=num_frames,
    frame_rate=frame_rate,
    generator=generator,
    output_type="latent",
)
up_video = upsample_pipe(latents=out.frames, output_type="latent", return_dict=False)[0]
up_keyframes = upsample_pipe(latents=out.keyframes, output_type="latent", return_dict=False)[0]

# Load after stage 1 so the adapter is never disabled. `set_adapters` does not re-enable a
# transformer that already had `disable_adapters()` called on it.
pipe.load_lora_weights("Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler", adapter_name="detailing")
pipe.set_adapters(["detailing"], adapter_weights=[0.5])
out2 = pipe(
    prompt=prompt,
    conditions=conditions,
    latents=up_video,
    audio_latents=out.audio,
    keyframes_latents=up_keyframes,
    keyframe_positions=out.keyframe_positions,
    reference_latents=out.frames,
    height=height,
    width=width,
    num_frames=num_frames,
    frame_rate=frame_rate,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    output_type="latent",
)
pipe.transformer.disable_adapters()

# `num_frames` here is the *padded* canvas the latents actually cover, which `resolve_canvas` may have
# grown past the 121 that were asked for. `condition_num_frames` stays the original request so a
# negative `condition.index` does not wrap onto the pad.
ratio = pipe.vae.temporal_compression_ratio
canvas_frames = (out2.frames.shape[2] - 1) * ratio + 1
out3 = temporal_pipe(
    latents=out2.frames,
    keyframes_latents=out2.keyframes,
    keyframe_positions=out2.keyframe_positions,
    audio_latents=out.audio,
    prompt=prompt,
    conditions=conditions,
    height=height,
    width=width,
    num_frames=canvas_frames,
    frame_rate=frame_rate,
    source_seconds=canvas_frames / frame_rate,
    condition_num_frames=num_frames,
    generator=generator,
    output_type="latent",
)

# Keep the padded canvas until decode so a slot on the pad is not dropped. `trim_canvas` counts *pixel*
# frames, and the round mapped `N -> 2 (N - 1) + 1`.
playback_fps = frame_rate * 2
requested_frames = (num_frames - 1) * 2 + 1
video_latents = trim_canvas(out3.frames, requested_frames, ratio)
timestep = None
if pipe.vae.config.timestep_conditioning:
    timestep = torch.zeros(video_latents.shape[0], device=video_latents.device, dtype=pipe.vae.dtype)
video = pipe.vae.decode(video_latents.to(pipe.vae.dtype), timestep, return_dict=False)[0]
video = pipe.video_processor.postprocess_video(video, output_type="np")

# Audio is stage 1's. Cut it to the video's duration so a muxed container does not outlast the picture.
audio = pipe.vocoder(pipe.audio_vae.decode(out.audio.to(pipe.audio_vae.dtype), return_dict=False)[0])
audio_samples = round(requested_frames / playback_fps * pipe.vocoder.config.output_sampling_rate)
audio = audio[..., : min(audio.shape[-1], audio_samples)]

encode_video(
    video[0],
    fps=playback_fps,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_5_dfr.mp4",
)
```

`height` and `width` are **this pass**, not the final output. Stage 1 runs at half the 1080p canvas (544×960); stage 2 at 1088×1920. Each must be divisible by the VAE's spatial compression ratio (32 on LTX-2.5 for a single pass; 64 when stage 1 is half of 1080p). This is why 1080p is **1920×1088** and 4K is **3840×2176**. Each pass runs a fixed distilled schedule (`sigmas`), so there is no `num_inference_steps`; the distilled schedules are trained without guidance, so there is no `negative_prompt` or `guidance_scale` either. The shipped audio is stage 1's — later passes still run an audio stream so the video branch has cross-modal attention; the waveform itself is not refined after stage 1.

**Spatial detailing.** Load the 2x spatial detailing IC-LoRA under a named adapter **after stage 1** and activate it for stage 2 only (`set_adapters(["detailing"], adapter_weights=[0.5])`). Stage 2 then attends to the stage-1 half-resolution latent as `reference_latents`. Stage 1 and the temporal rounds run with the adapter off. If you load the LoRA before stage 1, `transformer.disable_adapters()` turns it off, and `set_adapters` does **not** turn it back on — call `transformer.enable_adapters()` before stage 2, or load the weights after stage 1 as in the recipe. `reference_downscale_factor` (default `2`) scales the reference tokens' spatial coordinates into the target's coordinate space.

**Temporal refinement.** [LTX2DFRTemporalRefinePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRTemporalRefinePipeline) is one round: temporally upsample, tile on keyframe seams, ancestral-denoise with `LTXEulerAncestralRFScheduler` (`eta=0.5`), stitch by dropping the later tile's lead-in, and merge the carry-keyframe bag. Construct that scheduler yourself — the round is refused with anything else, since a deterministic step would run to completion and only return a softer canvas. Call the pipeline once per round; loop for 2x / 4x, passing `round_index`. After a round, `keyframe_positions` cannot be re-derived from the original `num_frames` and must be passed through. Each tile is handed the slice of the frozen stage-1 audio covering its own playback window. `source_seconds` is the *stage-1* duration and stays fixed across rounds, so later rounds must pass it explicitly rather than take the default.

Conditioning fps is 60 whenever playback is above 30, independently of muxing: RoPE time is `pixel_frame / fps`, so a 120 fps time base would halve every token's temporal span versus the trained distribution, and 48 fps would stretch it. Both lie that they are 60 and treat the decoded frames at the playback rate.

**A third spatial stage.** Compose it; there is no fourth pipeline. Spatially upsample the **video only**, rebuild carry keyframes in RGB (`decode` → Lanczos ×2 → `encode` via [rebuild_epilogue_keyframes()](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipeline.rebuild_epilogue_keyframes); never latent-upsample epilogue keyframes), then:

```py
from diffusers.pipelines.ltx2.dfr_layout import epilogue_tiles, pixel_to_latent_index

# One more doubling on top of the recipe above, so every stage below the output halves again:
# `epilogue_height` must be divisible by 128 (`4 * 32`), which is why 4K is 3840x2176.
epilogue_height, epilogue_width = height * 2, width * 2
refined_frames = (out3.frames.shape[2] - 1) * ratio + 1

up_video = upsample_pipe(latents=out3.frames, output_type="latent", return_dict=False)[0]
epilogue_keyframes = pipe.rebuild_epilogue_keyframes(
    out3.keyframes,
    decode_timestep=0.0,
    decode_noise_scale=0.0,
    seed=0,
    device=up_video.device,
    dtype=torch.float32,
)

# Temporal cuts land on the seams the *last* round stitched on -- the positions handed into it, doubled --
# not on every carry keyframe, since the slots that round invented sit mid-window.
tiles = epilogue_tiles(
    latent_shape=(
        (refined_frames - 1) // ratio + 1,
        epilogue_height // pipe.vae.spatial_compression_ratio,
        epilogue_width // pipe.vae.spatial_compression_ratio,
    ),
    frame_tiles=2,  # 2 ** number of temporal rounds
    frame_seams=[pixel_to_latent_index(2 * p, ratio) for p in out2.keyframe_positions],
)

pipe.transformer.enable_adapters()  # the epilogue is a detailing pass too
out4 = pipe(
    prompt=prompt,
    conditions=conditions,
    latents=up_video,
    audio_latents=out.audio,
    generate_slots=False,
    guidance_keyframe_latents=epilogue_keyframes,
    guidance_keyframe_positions=out3.keyframe_positions,
    reference_latents=out3.frames,
    height=epilogue_height,
    width=epilogue_width,
    num_frames=refined_frames,
    frame_rate=playback_fps,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    freeze_audio=True,
    video_tiles=tiles,
    generator=generator,
    output_type="latent",
)
pipe.transformer.disable_adapters()
```

The two axes are seamed differently. Neither side of a spatial border holds a known answer, so those overlaps are blended with trapezoidal weights. Temporal tiles are cut on the last refine round's keyframe seams.

**Decoding with the diffusion decoder.** For maximum detail fidelity, stay on `output_type="latent"` and hand the (already denormalized, possibly `trim_canvas`'d) latents to [LTX2VideoDiffusionDecodePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2VideoDiffusionDecodePipeline).

```py
from diffusers import LTX2VideoDiffusionDecodePipeline
from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoDiffusionDecoderModel

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    "Lightricks/LTX-2.5-Diffusers", subfolder="diffusion_decoder", dtype=torch.bfloat16
)
decode_pipe = LTX2VideoDiffusionDecodePipeline(
    diffusion_decoder=decoder, scheduler=pipe.scheduler, vae=pipe.vae
)
decode_pipe.enable_model_cpu_offload()
# `denormalize=False`: the `output_type="latent"` path already applied the latent statistics.
video = decode_pipe(latents=out3.frames, denormalize=False, output_type="np", return_dict=False)[0]
```

## LTX2Pipeline[[diffusers.LTX2Pipeline]]

#### diffusers.LTX2Pipeline[[diffusers.LTX2Pipeline]]

```python
diffusers.LTX2Pipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLLTX2Video, audio_vae: AutoencoderKLLTX2Audio, text_encoder: transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration | transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration, tokenizer: GemmaTokenizer, connectors: LTX2TextConnectors, transformer: LTX2VideoTransformer3DModel, vocoder: diffusers.pipelines.ltx2.vocoder.LTX2Vocoder | diffusers.pipelines.ltx2.vocoder.LTX2VocoderWithBWE, processor: transformers.processing_utils.ProcessorMixin | None = None, prompt_enhancer: transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration | None = None, duration_head: diffusers.pipelines.ltx2.duration_head.LTX2DurationHead | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L206)

**Parameters:**

transformer ([LTXVideoTransformer3DModel](/docs/diffusers/v0.41.0/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLLTXVideo](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

connectors (`LTX2TextConnectors`) : Text connector stack used to adapt text encoder hidden states for the video and audio branches.

Pipeline for text-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

#### __call__[[diffusers.LTX2Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] | None = None, height: int = 512, width: int = 768, num_frames: int | None = None, min_seconds: float = 1.0, max_seconds: float = 20.0, frame_rate: float = 24.0, num_inference_steps: int = 30, sigmas: list[float] | None = None, timesteps: list = None, guidance_scale: float = 3.0, stg_scale: float = 1.0, modality_scale: float = 3.0, guidance_rescale: float = 0.7, audio_guidance_scale: float | None = 7.0, audio_stg_scale: float | None = 1.0, audio_modality_scale: float | None = 3.0, audio_guidance_rescale: float | None = 0.7, spatio_temporal_guidance_blocks: list[int] | None = [28], noise_scale: float = 0.0, num_videos_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, decode_timestep: float | list[float] = 0.0, decode_noise_scale: float | list[float] | None = None, use_cross_timestep: bool = True, system_prompt: str | None = None, enable_prompt_enhancement: bool = False, prompt_max_new_tokens: int | None = None, prompt_enhancement_kwargs: dict[str, typing.Any] | None = None, prompt_enhancement_seed: int = 10, output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L926)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

height (`int`, *optional*, defaults to `512`) : The height in pixels of the generated image. This is set to 480 by default for the best results.

width (`int`, *optional*, defaults to `768`) : The width in pixels of the generated image. This is set to 848 by default for the best results.

num_frames (`int`, *optional*) : The number of video frames to generate. If not supplied, defaults to an auto-predicted duration when this pipeline has a `duration_head` component (LTX-2.5 checkpoints and later), and to `121` otherwise. Pass an integer to set the length explicitly. Auto-predicted counts are snapped to the VAE's causal temporal grid, so the realized duration is quantized (roughly 0.33s at 24 fps).

min_seconds (`float`, *optional*, defaults to `1.0`) : Lower bound on the auto-predicted duration when `num_frames` is omitted and a `duration_head` is present. Ignored when `num_frames` is set explicitly.

max_seconds (`float`, *optional*, defaults to `20.0`) : Upper bound on the auto-predicted duration when `num_frames` is omitted and a `duration_head` is present. Ignored when `num_frames` is set explicitly. Must be strictly greater than `min_seconds`.

frame_rate (`float`, *optional*, defaults to `24.0`) : The frames per second (FPS) of the generated video.

num_inference_steps (`int`, *optional*, defaults to 30) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`List[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to `4.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is a separate value `audio_guidance_scale` for the audio modality).

stg_scale (`float`, *optional*, defaults to `0.0`) : Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate where we move the sample away from a weak sample from a perturbed version of the denoising model. Enabling STG will result in an additional denoising model forward pass; the default value of `0.0` means that STG is disabled.

modality_scale (`float`, *optional*, defaults to `1.0`) : Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio) cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an additional denoising model forward pass; the default value of `1.0` means that modality guidance is disabled.

guidance_rescale (`float`, *optional*, defaults to 0.0) : Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when using zero terminal SNR. Used for the video modality.

audio_guidance_scale (`float`, *optional* defaults to `None`) : Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value `guidance_scale`.

audio_stg_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the video value `stg_scale`.

audio_modality_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and audio. If `None`, defaults to the video value `modality_scale`.

audio_guidance_rescale (`float`, *optional*, defaults to `None`) : A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value `guidance_rescale`.

spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`) : The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0 and `[28]` is recommended for LTX-2.3.

noise_scale (`float`, *optional*, defaults to `0.0`) : The interpolation factor between random noise and denoised latents at each timestep. Applying noise to the `latents` and `audio_latents` before continue denoising.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

audio_latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

use_cross_timestep (`bool` *optional*, defaults to `True`) : Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when calculating the cross attention modulation parameters. `True` is the LTX-2.3/2.5 behavior; `False` is the legacy LTX-2.0 behavior.

system_prompt (`str`, *optional*, defaults to `None`) : Optional system prompt to use for prompt enhancement. The system prompt will be used by the prompt enhancer (a Gemma conditional-generation model -- the dedicated `prompt_enhancer` component if one is configured, otherwise the main `text_encoder`) to generate an enhanced prompt from the original `prompt` to condition generation. If not supplied and a dedicated `prompt_enhancer` is configured (LTX-2.5), defaults to `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` (from `diffusers.pipelines.ltx2.utils`) -- see `enable_prompt_enhancement`.

enable_prompt_enhancement (`bool`, *optional*, defaults to `False`) : Whether to run prompt enhancement. Opt-in, matching the Lightricks reference pipelines. When `True` and `system_prompt` is omitted, LTX-2.5 uses `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` if a dedicated `prompt_enhancer` is configured; LTX-2.0/2.3 require an explicit `system_prompt`.

prompt_max_new_tokens (`int`, *optional*, defaults to `None`) : The maximum number of new tokens to generate when performing prompt enhancement. If not supplied, uses 600 for a dedicated Gemma 4 `prompt_enhancer` (LTX-2.5) or 512 for the Gemma 3 `text_encoder` fallback (LTX-2.0/2.3).

prompt_enhancement_kwargs (`dict[str, Any]`, *optional*, defaults to `None`) : Keyword arguments for the prompt enhancer's `.generate` call. If not supplied, always matches whichever model is doing the enhancing: `do_sample=False, no_repeat_ngram_size=5` (greedy) when using a dedicated `prompt_enhancer` (LTX-2.5), or `do_sample=True, temperature=0.7` for the `text_encoder` fallback (LTX-2.0/2.3). See https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate for more details.

prompt_enhancement_seed (`int`, *optional*, defaults to `10`) : Random seed for any random operations during prompt enhancement.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.ltx.LTX2PipelineOutput` or `tuple`

If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2Pipeline
>>> from diffusers.utils import encode_video

>>> pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> frame_rate = 24.0
>>> video, audio = pipe(
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=30,
...     guidance_scale=3.0,
...     output_type="np",
...     return_dict=False,
... )

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

#### encode_prompt[[diffusers.LTX2Pipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024, scale_factor: int = 8, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L364)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

#### enhance_prompt[[diffusers.LTX2Pipeline.enhance_prompt]]

```python
enhance_prompt(prompt: str, system_prompt: str, max_new_tokens: int | None = None, seed: int = 10, generator: typing.Optional[torch.Generator] = None, generation_kwargs: dict[str, typing.Any] | None = None, device: typing.Union[str, torch.device, NoneType] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L571)

Enhances the supplied `prompt` by generating a new prompt using the prompt enhancer (a Gemma
conditional-generation model) from it and a system prompt. When `image` is supplied, the enhancer is also
conditioned on that reference frame (I2V / keyframe-style enhancement). Uses the dedicated `prompt_enhancer`
component if one is configured (e.g. LTX-2.5, whose text encoder isn't trained for enhancement), otherwise
falls back to the main `text_encoder` (LTX-2.0/2.3, which double as their own enhancer).

Message templates, decoding kwargs, response cleaning, and image long-side prep match `ltx-core` /
`ltx-pipelines` (`enhance_t2v` / `enhance_i2v` / `generate_enhanced_prompt`).

## LTX2ImageToVideoPipeline[[diffusers.LTX2ImageToVideoPipeline]]

#### diffusers.LTX2ImageToVideoPipeline[[diffusers.LTX2ImageToVideoPipeline]]

```python
diffusers.LTX2ImageToVideoPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLLTX2Video, audio_vae: AutoencoderKLLTX2Audio, text_encoder: transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration | transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration, tokenizer: GemmaTokenizer, connectors: LTX2TextConnectors, transformer: LTX2VideoTransformer3DModel, vocoder: diffusers.pipelines.ltx2.vocoder.LTX2Vocoder | diffusers.pipelines.ltx2.vocoder.LTX2VocoderWithBWE, processor: transformers.processing_utils.ProcessorMixin | None = None, prompt_enhancer: transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration | None = None, duration_head: diffusers.pipelines.ltx2.duration_head.LTX2DurationHead | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L226)

Pipeline for image-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

TODO

#### __call__[[diffusers.LTX2ImageToVideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, prompt: str | list[str] = None, negative_prompt: str | list[str] | None = None, height: int = 512, width: int = 768, num_frames: int | None = None, min_seconds: float = 1.0, max_seconds: float = 20.0, frame_rate: float = 24.0, num_inference_steps: int = 30, sigmas: list[float] | None = None, timesteps: list[int] | None = None, guidance_scale: float = 3.0, stg_scale: float = 1.0, modality_scale: float = 3.0, guidance_rescale: float = 0.7, audio_guidance_scale: float | None = 7.0, audio_stg_scale: float | None = 1.0, audio_modality_scale: float | None = 3.0, audio_guidance_rescale: float | None = 0.7, spatio_temporal_guidance_blocks: list[int] | None = [28], noise_scale: float = 0.0, num_videos_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, decode_timestep: float | list[float] = 0.0, decode_noise_scale: float | list[float] | None = None, use_cross_timestep: bool = True, system_prompt: str | None = None, enable_prompt_enhancement: bool = False, prompt_max_new_tokens: int | None = None, prompt_enhancement_kwargs: dict[str, typing.Any] | None = None, prompt_enhancement_seed: int = 10, image_crf: int | None = None, output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L980)

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

height (`int`, *optional*, defaults to `512`) : The height in pixels of the generated image. This is set to 480 by default for the best results.

width (`int`, *optional*, defaults to `768`) : The width in pixels of the generated image. This is set to 848 by default for the best results.

num_frames (`int`, *optional*) : The number of video frames to generate. If not supplied, defaults to an auto-predicted duration when this pipeline has a `duration_head` component (LTX-2.5 checkpoints and later), and to `121` otherwise. Pass an integer to set the length explicitly. Auto-predicted counts are snapped to the VAE's causal temporal grid, so the realized duration is quantized (roughly 0.33s at 24 fps).

min_seconds (`float`, *optional*, defaults to `1.0`) : Lower bound on the auto-predicted duration when `num_frames` is omitted and a `duration_head` is present. Ignored when `num_frames` is set explicitly.

max_seconds (`float`, *optional*, defaults to `20.0`) : Upper bound on the auto-predicted duration when `num_frames` is omitted and a `duration_head` is present. Ignored when `num_frames` is set explicitly. Must be strictly greater than `min_seconds`.

frame_rate (`float`, *optional*, defaults to `24.0`) : The frames per second (FPS) of the generated video.

num_inference_steps (`int`, *optional*, defaults to 30) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`List[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to `4.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is a separate value `audio_guidance_scale` for the audio modality).

stg_scale (`float`, *optional*, defaults to `0.0`) : Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate where we move the sample away from a weak sample from a perturbed version of the denoising model. Enabling STG will result in an additional denoising model forward pass; the default value of `0.0` means that STG is disabled.

modality_scale (`float`, *optional*, defaults to `1.0`) : Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio) cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an additional denoising model forward pass; the default value of `1.0` means that modality guidance is disabled.

guidance_rescale (`float`, *optional*, defaults to 0.0) : Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when using zero terminal SNR. Used for the video modality.

audio_guidance_scale (`float`, *optional* defaults to `None`) : Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value `guidance_scale`.

audio_stg_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the video value `stg_scale`.

audio_modality_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and audio. If `None`, defaults to the video value `modality_scale`.

audio_guidance_rescale (`float`, *optional*, defaults to `None`) : A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value `guidance_rescale`.

spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`) : The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0 and `[28]` is recommended for LTX-2.3.

noise_scale (`float`, *optional*, defaults to `0.0`) : The interpolation factor between random noise and denoised latents at each timestep. Applying noise to the `latents` and `audio_latents` before continue denoising.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

audio_latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

use_cross_timestep (`bool` *optional*, defaults to `True`) : Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when calculating the cross attention modulation parameters. `True` is the LTX-2.3/2.5 behavior; `False` is the legacy LTX-2.0 behavior.

system_prompt (`str`, *optional*, defaults to `None`) : Optional system prompt to use for prompt enhancement. The system prompt will be used by the prompt enhancer (a Gemma conditional-generation model -- the dedicated `prompt_enhancer` component if one is configured, otherwise the main `text_encoder`) to generate an enhanced prompt from the original `prompt` and the first `image` to condition generation. If not supplied and a dedicated `prompt_enhancer` is configured (LTX-2.5), defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` (from `diffusers.pipelines.ltx2.utils`) -- see `enable_prompt_enhancement`.

enable_prompt_enhancement (`bool`, *optional*, defaults to `False`) : Whether to run prompt enhancement. Opt-in, matching the Lightricks reference pipelines. When `True` and `system_prompt` is omitted, LTX-2.5 uses `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` if a dedicated `prompt_enhancer` is configured; LTX-2.0/2.3 require an explicit `system_prompt`.

prompt_max_new_tokens (`int`, *optional*, defaults to `None`) : The maximum number of new tokens to generate when performing prompt enhancement. If not supplied, uses 600 for a dedicated Gemma 4 `prompt_enhancer` (LTX-2.5) or 512 for the Gemma 3 `text_encoder` fallback (LTX-2.0/2.3).

prompt_enhancement_kwargs (`dict[str, Any]`, *optional*, defaults to `None`) : Keyword arguments for the prompt enhancer's `.generate` call. If not supplied, always matches whichever model is doing the enhancing: `do_sample=False, no_repeat_ngram_size=3` (greedy) when using a dedicated `prompt_enhancer` (LTX-2.5), or `do_sample=True, temperature=0.7` for the `text_encoder` fallback (LTX-2.0/2.3). See https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate for more details.

prompt_enhancement_seed (`int`, *optional*, defaults to `10`) : Random seed for any random operations during prompt enhancement.

image_crf (`int`, *optional*, defaults to `None`) : H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the model was trained against. `None` means "use the model default" (33 through LTX-2.3, 18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when re-compression runs.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.ltx.LTX2PipelineOutput` or `tuple`

If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2ImageToVideoPipeline
>>> from diffusers.utils import encode_video
>>> from diffusers.utils import load_image

>>> pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> image = load_image(
...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
... )
>>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> frame_rate = 24.0
>>> video, audio = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=30,
...     guidance_scale=3.0,
...     output_type="np",
...     return_dict=False,
... )

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

#### encode_prompt[[diffusers.LTX2ImageToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024, scale_factor: int = 8, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L369)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

#### enhance_prompt[[diffusers.LTX2ImageToVideoPipeline.enhance_prompt]]

```python
enhance_prompt(prompt: str, system_prompt: str, max_new_tokens: int | None = None, seed: int = 10, generator: typing.Optional[torch.Generator] = None, generation_kwargs: dict[str, typing.Any] | None = None, device: typing.Union[str, torch.device, NoneType] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L577)

Enhances the supplied `prompt` by generating a new prompt using the prompt enhancer (a Gemma
conditional-generation model) from it and a system prompt. When `image` is supplied, the enhancer is also
conditioned on that reference frame (I2V / keyframe-style enhancement). Uses the dedicated `prompt_enhancer`
component if one is configured (e.g. LTX-2.5, whose text encoder isn't trained for enhancement), otherwise
falls back to the main `text_encoder` (LTX-2.0/2.3, which double as their own enhancer).

Message templates, decoding kwargs, response cleaning, and image long-side prep match `ltx-core` /
`ltx-pipelines` (`enhance_t2v` / `enhance_i2v` / `generate_enhanced_prompt`).

## LTX2ConditionPipeline[[diffusers.LTX2ConditionPipeline]]

#### diffusers.LTX2ConditionPipeline[[diffusers.LTX2ConditionPipeline]]

```python
diffusers.LTX2ConditionPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLLTX2Video, audio_vae: AutoencoderKLLTX2Audio, text_encoder: transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration | transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration, tokenizer: GemmaTokenizer, connectors: LTX2TextConnectors, transformer: LTX2VideoTransformer3DModel, vocoder: diffusers.pipelines.ltx2.vocoder.LTX2Vocoder | diffusers.pipelines.ltx2.vocoder.LTX2VocoderWithBWE, audio_scheduler: diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler | None = None, processor: transformers.processing_utils.ProcessorMixin | None = None, prompt_enhancer: transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration | None = None, duration_head: diffusers.pipelines.ltx2.duration_head.LTX2DurationHead | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L263)

Pipeline for video generation which allows image conditions to be inserted at arbitrary parts of the video.

Reference: https://github.com/Lightricks/LTX-Video

TODO

#### __call__[[diffusers.LTX2ConditionPipeline.__call__]]

```python
__call__(conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, prompt: str | list[str] = None, negative_prompt: str | list[str] | None = None, height: int = 512, width: int = 768, num_frames: int | None = None, min_seconds: float = 1.0, max_seconds: float = 20.0, frame_rate: float = 24.0, num_inference_steps: int = 30, sigmas: list[float] | None = None, timesteps: list[float] | None = None, guidance_scale: float = 3.0, stg_scale: float = 1.0, modality_scale: float = 3.0, guidance_rescale: float = 0.7, audio_guidance_scale: float | None = 7.0, audio_stg_scale: float | None = 1.0, audio_modality_scale: float | None = 3.0, audio_guidance_rescale: float | None = 0.7, spatio_temporal_guidance_blocks: list[int] | None = [28], noise_scale: float | None = None, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, decode_timestep: float | list[float] = 0.0, decode_noise_scale: float | list[float] | None = None, use_cross_timestep: bool = True, system_prompt: str | None = None, enable_prompt_enhancement: bool = False, prompt_max_new_tokens: int | None = None, prompt_enhancement_kwargs: dict[str, typing.Any] | None = None, prompt_enhancement_seed: int = 10, output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L1345)

**Parameters:**

conditions (`List[LTXVideoCondition], *optional*`) : The list of frame-conditioning items for the video generation.

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

height (`int`, *optional*, defaults to `512`) : The height in pixels of the generated image. This is set to 480 by default for the best results.

width (`int`, *optional*, defaults to `768`) : The width in pixels of the generated image. This is set to 848 by default for the best results.

num_frames (`int`, *optional*) : The number of video frames to generate. If not supplied, defaults to an auto-predicted duration when this pipeline has a `duration_head` component (LTX-2.5 checkpoints and later), and to `121` otherwise. Pass an integer to set the length explicitly. Auto-predicted counts are snapped to the VAE's causal temporal grid, so the realized duration is quantized (roughly 0.33s at 24 fps).

min_seconds (`float`, *optional*, defaults to `1.0`) : Lower bound on the auto-predicted duration when `num_frames` is omitted and a `duration_head` is present. Ignored when `num_frames` is set explicitly.

max_seconds (`float`, *optional*, defaults to `20.0`) : Upper bound on the auto-predicted duration when `num_frames` is omitted and a `duration_head` is present. Ignored when `num_frames` is set explicitly. Must be strictly greater than `min_seconds`.

frame_rate (`float`, *optional*, defaults to `24.0`) : The frames per second (FPS) of the generated video.

num_inference_steps (`int`, *optional*, defaults to 30) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`List[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to `4.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is a separate value `audio_guidance_scale` for the audio modality).

stg_scale (`float`, *optional*, defaults to `0.0`) : Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate where we move the sample away from a weak sample from a perturbed version of the denoising model. Enabling STG will result in an additional denoising model forward pass; the default value of `0.0` means that STG is disabled.

modality_scale (`float`, *optional*, defaults to `1.0`) : Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio) cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an additional denoising model forward pass; the default value of `1.0` means that modality guidance is disabled.

guidance_rescale (`float`, *optional*, defaults to 0.0) : Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when using zero terminal SNR. Used for the video modality.

audio_guidance_scale (`float`, *optional* defaults to `None`) : Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value `guidance_scale`.

audio_stg_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the video value `stg_scale`.

audio_modality_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and audio. If `None`, defaults to the video value `modality_scale`.

audio_guidance_rescale (`float`, *optional*, defaults to `None`) : A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value `guidance_rescale`.

spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`) : The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0 and `[28]` is recommended for LTX-2.3.

noise_scale (`float`, *optional*, defaults to `None`) : The interpolation factor between random noise and denoised latents at each timestep. Applying noise to the `latents` and `audio_latents` before continue denoising. If not set, will be inferred from the sigma schedule.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `List[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

audio_latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

use_cross_timestep (`bool` *optional*, defaults to `True`) : Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when calculating the cross attention modulation parameters. `True` is the LTX-2.3/2.5 behavior; `False` is the legacy LTX-2.0 behavior.

system_prompt (`str`, *optional*, defaults to `None`) : Optional system prompt to use for prompt enhancement. The system prompt will be used by the prompt enhancer (a Gemma conditional-generation model -- the dedicated `prompt_enhancer` component if one is configured, otherwise the main `text_encoder`) to generate an enhanced prompt from the original `prompt` (and a conditioning image when one is available) to condition generation. If not supplied and a dedicated `prompt_enhancer` is configured (LTX-2.5), defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` when a conditioning image is available, otherwise `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` -- see `enable_prompt_enhancement`.

enable_prompt_enhancement (`bool`, *optional*, defaults to `False`) : Whether to run prompt enhancement. Opt-in, matching the Lightricks reference pipelines. When `True` and `system_prompt` is omitted, LTX-2.5 picks `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` / `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` based on whether a conditioning image is available.

prompt_max_new_tokens (`int`, *optional*, defaults to `None`) : The maximum number of new tokens to generate when performing prompt enhancement. If not supplied, uses 600 for a dedicated Gemma 4 `prompt_enhancer` (LTX-2.5) or 512 for the Gemma 3 `text_encoder` fallback (LTX-2.0/2.3).

prompt_enhancement_kwargs (`dict[str, Any]`, *optional*, defaults to `None`) : Keyword arguments for the prompt enhancer's `.generate` call. If not supplied, always matches whichever model is doing the enhancing: `do_sample=False, no_repeat_ngram_size=5` (greedy) when using a dedicated `prompt_enhancer` (LTX-2.5), or `do_sample=True, temperature=0.7` for the `text_encoder` fallback (LTX-2.0/2.3). See https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate for more details.

prompt_enhancement_seed (`int`, *optional*, defaults to `10`) : Random seed for any random operations during prompt enhancement.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.ltx.LTX2PipelineOutput` or `tuple`

If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2ConditionPipeline
>>> from diffusers.utils import encode_video
>>> from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
>>> from diffusers.utils import load_image

>>> pipe = LTX2ConditionPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> first_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
... )
>>> last_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
... )
>>> first_cond = LTX2VideoCondition(frames=first_image, index=0, strength=1.0)
>>> last_cond = LTX2VideoCondition(frames=last_image, index=-1, strength=1.0)
>>> conditions = [first_cond, last_cond]
>>> prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, static"

>>> frame_rate = 24.0
>>> video = pipe(
...     conditions=conditions,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=30,
...     guidance_scale=3.0,
...     output_type="np",
...     return_dict=False,
... )
>>> video = (video * 255).round().astype("uint8")
>>> video = torch.from_numpy(video)

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

#### apply_first_frame_conditioning[[diffusers.LTX2ConditionPipeline.apply_first_frame_conditioning]]

```python
apply_first_frame_conditioning(latents: Tensor, conditioning_mask: Tensor, condition_latents: list, condition_strengths: list, condition_indices: list, latent_height: int, latent_width: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L961)

**Parameters:**

latents (`torch.Tensor`) : Initial packed (patchified) latents of shape [batch_size, patch_seq_len, hidden_dim].

conditioning_mask (`torch.Tensor`) : Initial packed (patchified) conditioning mask of shape [batch_size, patch_seq_len, 1] with values in [0, 1] where 0 means the denoising model output will be fully used and 1 means the condition will be fully used.

**Returns:** `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Returns a 3-tuple of tensors where:
1. The packed video latents with first-frame conditions applied.
2. The packed conditioning mask with first-frame strengths applied.
3. The clean conditioning latents at first-frame positions (zeros elsewhere).

Apply first-frame visual conditioning by overwriting tokens at the first-frame positions.

Only conditions with `latent_idx == 0` are applied here (matching `VideoConditionByLatentIndex` in the
reference implementation). Conditions at non-zero latent indices are appended as separate keyframe tokens via
`prepare_keyframe_extras` (matching `VideoConditionByKeyframeIndex`) and are skipped here.

#### encode_prompt[[diffusers.LTX2ConditionPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024, scale_factor: int = 8, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L416)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

#### enhance_prompt[[diffusers.LTX2ConditionPipeline.enhance_prompt]]

```python
enhance_prompt(prompt: str, system_prompt: str, max_new_tokens: int | None = None, seed: int = 10, generator: typing.Optional[torch.Generator] = None, generation_kwargs: dict[str, typing.Any] | None = None, device: typing.Union[str, torch.device, NoneType] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L624)

Enhances the supplied `prompt` by generating a new prompt using the prompt enhancer (a Gemma
conditional-generation model) from it and a system prompt. When `image` is supplied, the enhancer is also
conditioned on that reference frame (I2V / keyframe-style enhancement). Uses the dedicated `prompt_enhancer`
component if one is configured (e.g. LTX-2.5, whose text encoder isn't trained for enhancement), otherwise
falls back to the main `text_encoder` (LTX-2.0/2.3, which double as their own enhancer).

Message templates, decoding kwargs, response cleaning, and image long-side prep match `ltx-core` /
`ltx-pipelines` (`enhance_t2v` / `enhance_i2v` / `generate_enhanced_prompt`).

#### prepare_latents[[diffusers.LTX2ConditionPipeline.prepare_latents]]

```python
prepare_latents(conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, batch_size: int = 1, num_channels_latents: int = 128, height: int = 512, width: int = 768, num_frames: int = 121, frame_rate: float = 24.0, noise_scale: float = 1.0, dtype: typing.Optional[torch.dtype] = None, device: typing.Optional[torch.device] = None, generator: typing.Optional[torch.Generator] = None, latents: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L1068)

Prepare noisy video latents, applying frame conditions.

First-frame conditions (`latent_idx == 0`) are applied by overwriting tokens at the first-frame positions
(`VideoConditionByLatentIndex` semantics). Non-first-frame conditions (`latent_idx > 0`) are concatenated onto
the main latent sequence with per-token `conditioning_mask = strength` (`VideoConditionByKeyframeIndex`
semantics) — the denoising loop's existing timestep formula `t * (1 - conditioning_mask)` and post-process
blend `denoised * (1 - conditioning_mask) + clean * conditioning_mask` then drive them across steps.

Returns a 4-tuple:
- `latents`: packed noisy latents (base tokens + any keyframe tokens cat'd onto the sequence dim).
- `conditioning_mask`: packed conditioning mask with values in `[0, 1]` — `1` at first-frame positions,
  `strength` at keyframe positions, `0` elsewhere.
- `clean_latents`: clean condition values at conditioned positions (zeros elsewhere); same shape as
  `latents`.
- `keyframe_coords`: `[B, 3, num_keyframe_patches, 2]` positional coordinates to append to `video_coords`,
  or `None` if there are no non-first-frame conditions.

#### preprocess_conditions[[diffusers.LTX2ConditionPipeline.preprocess_conditions]]

```python
preprocess_conditions(conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, height: int = 512, width: int = 768, num_frames: int = 121, device: typing.Optional[torch.device] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L843)

**Parameters:**

conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*, defaults to `None`) : A list of image/video condition instances.

height (`int`, *optional*, defaults to `512`) : The desired height in pixels.

width (`int`, *optional*, defaults to `768`) : The desired width in pixels.

num_frames (`int`, *optional*, defaults to `121`) : The desired number of frames in the generated video.

device (`torch.device`, *optional*, defaults to `None`) : The device on which to put the preprocessed image/video tensors.

**Returns:** `Tuple[List[torch.Tensor], List[float], List[int], List[int]]`

Returns a 4-tuple of lists of length `len(conditions)` as follows:
1. The first list is a list of preprocessed video tensors of shape [batch_size=1, num_channels,
   num_frames, height, width].
2. The second list is a list of conditioning strengths.
3. The third list is a list of latent-space indices for each condition.
4. The fourth list is a list of (trimmed) pixel-space frame counts per condition. This is needed
   for keyframe coord semantics (single-pixel-frame keyframes have a clamped temporal extent).

Preprocesses the condition images/videos to torch tensors.

#### trim_conditioning_sequence[[diffusers.LTX2ConditionPipeline.trim_conditioning_sequence]]

```python
trim_conditioning_sequence(start_frame: int, sequence_num_frames: int, target_num_frames: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L826)

**Parameters:**

start_frame (int) : The target frame number of the first frame in the sequence.

sequence_num_frames (int) : The number of frames in the sequence.

target_num_frames (int) : The target number of frames in the generated video.

**Returns:** `int`

updated sequence length

Trim a conditioning sequence to the allowed number of frames.

## LTX2DFRPipeline[[diffusers.LTX2DFRPipeline]]

#### diffusers.LTX2DFRPipeline[[diffusers.LTX2DFRPipeline]]

```python
diffusers.LTX2DFRPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLLTX2Video, audio_vae: AutoencoderKLLTX2Audio, text_encoder: transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration | transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration, tokenizer: GemmaTokenizer, connectors: LTX2TextConnectors, transformer: LTX2VideoTransformer3DModel, vocoder: diffusers.pipelines.ltx2.vocoder.LTX2Vocoder | diffusers.pipelines.ltx2.vocoder.LTX2VocoderWithBWE, processor: transformers.processing_utils.ProcessorMixin | None = None, prompt_enhancer: transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration | None = None, duration_head: diffusers.pipelines.ltx2.duration_head.LTX2DurationHead | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L122)

**Parameters:**

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLLTX2Video](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_ltx_2#diffusers.AutoencoderKLLTX2Video)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

audio_vae ([AutoencoderKLLTX2Audio](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_audio_ltx_2#diffusers.AutoencoderKLLTX2Audio)) : Audio VAE to encode and decode audio spectrograms.

text_encoder (`Gemma3ForConditionalGeneration` or `Gemma4UnifiedForConditionalGeneration`) : Text encoder model.

tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`) : Tokenizer for the text encoder.

connectors (`LTX2TextConnectors`) : Text connector stack used to adapt text encoder hidden states for the video and audio branches.

transformer ([LTX2VideoTransformer3DModel](/docs/diffusers/v0.41.0/en/api/models/ltx2_video_transformer3d#diffusers.LTX2VideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

vocoder (`LTX2Vocoder` or `LTX2VocoderWithBWE`) : Vocoder to convert mel spectrograms to audio waveforms.

processor (`ProcessorMixin`, *optional*) : Processor used for prompt enhancement chat templating.

prompt_enhancer (`Gemma4ForConditionalGeneration`, *optional*) : Dedicated prompt enhancement model (LTX-2.5).

duration_head (`LTX2DurationHead`, *optional*) : Predicts `num_frames` from the prompt embeddings when `num_frames` is not supplied.

Pipeline for one Diffusion Fidelity Rendering (DFR) denoise pass with LTX-2.5.

A pass generates video *and* extra single-pixel-frame keyframe slots, or re-denoises supplied latents seeded from
those slots. Callers compose stages: this pipeline at half resolution, [LTX2LatentUpsamplePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2LatentUpsamplePipeline) spatially,
this pipeline again at full resolution with the upsampled slots and an IC-LoRA reference, then
[LTX2DFRTemporalRefinePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRTemporalRefinePipeline) for each temporal refine round. See the LTX-2 docs for the 1080p recipe.

Slot positions come from a segment grid aligned to the VAE's temporal border (`resolve_canvas`). The canvas is
padded to a whole number of segments; `output_type="latent"` returns that padded grid so a slot on the pad is not
dropped. Trim with `trim_canvas` before VAE decode.

Requires a transformer whose config sets `use_keyframes_abs_pos_embedding` (LTX-2.5 and later) when
`generate_slots=True`.

Reference: https://github.com/Lightricks/LTX-2

#### __call__[[diffusers.LTX2DFRPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, height: int = 704, width: int = 1216, num_frames: int | None = None, frame_rate: float = 24.0, min_seconds: float = 1.0, max_seconds: float = 20.0, latents: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.Tensor] = None, keyframes_latents: typing.Optional[torch.Tensor] = None, keyframe_positions: list[int] | None = None, reference_latents: typing.Optional[torch.Tensor] = None, reference_downscale_factor: int = 2, guidance_keyframe_latents: typing.Optional[torch.Tensor] = None, guidance_keyframe_positions: list[int] | None = None, guidance_keyframe_strength: float = 1.0, generate_slots: bool = True, sigmas: list = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875], noise_scale: float | None = None, freeze_audio: bool = False, video_tiles: list[diffusers.pipelines.ltx2.dfr_layout.LTX2DFREpilogueTile] | None = None, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, decode_timestep: float | list[float] = 0.0, decode_noise_scale: float | list[float] | None = None, use_cross_timestep: bool = True, system_prompt: str | None = None, enable_prompt_enhancement: bool = False, prompt_max_new_tokens: int | None = None, prompt_enhancement_kwargs: dict[str, typing.Any] | None = None, prompt_enhancement_seed: int = 10, output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L1522)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.

conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*) : Frame-level image or video conditions. `index` is a *latent* index on this pass's `num_frames`.

height (`int`, *optional*, defaults to `704`) : The height in pixels of **this pass**, not the final composed output.

width (`int`, *optional*, defaults to `1216`) : The width in pixels of this pass.

num_frames (`int`, *optional*) : Pixel frame count of this pass, before internal canvas padding. If not supplied, the duration is predicted from the prompt by the `duration_head`. Must satisfy `(num_frames - 1) % 8 == 0`.

frame_rate (`float`, *optional*, defaults to `24.0`) : Playback fps of this pass. RoPE time snaps to 60 whenever this is above 30.

min_seconds (`float`, *optional*, defaults to `1.0`) : Lower bound on the auto-predicted duration when `num_frames` is omitted.

max_seconds (`float`, *optional*, defaults to `20.0`) : Upper bound on the auto-predicted duration when `num_frames` is omitted.

latents (`torch.Tensor`, *optional*) : Raw `(batch_size, channels, frames, height, width)` video latents to re-denoise (stage 2 / epilogue).

audio_latents (`torch.Tensor`, *optional*) : Raw unpacked `(batch_size, channels, length, mel_bins)` audio latents. Stage 2 still runs a joint audio pass; the shipped waveform of a composed recipe is stage 1's, which the caller keeps.

keyframes_latents (`torch.Tensor`, *optional*) : Raw `(batch_size, channels, num_slots, height, width)` slot initials, used when `generate_slots=True`.

keyframe_positions (`list[int]`, *optional*) : Pixel-frame indices of the generated slots. Defaults to `resolve_canvas(num_frames)`.

reference_latents (`torch.Tensor`, *optional*) : Raw IC-LoRA reference video (typically stage 1's frames).

reference_downscale_factor (`int`, *optional*, defaults to `2`) : Ratio between this pass and the reference resolution, used to scale reference token coordinates.

guidance_keyframe_latents (`torch.Tensor`, *optional*) : Raw pinned guidance keyframes `(batch_size, channels, K, height, width)` — the epilogue path. Not generated slots.

guidance_keyframe_positions (`list[int]`, *optional*) : Pixel-frame indices for `guidance_keyframe_latents`.

guidance_keyframe_strength (`float`, *optional*, defaults to `1.0`) : Conditioning strength for pinned guidance keyframes.

generate_slots (`bool`, *optional*, defaults to `True`) : Append generated keyframe-slot tokens. The epilogue sets this to `False` and pins `guidance_keyframe_latents` instead.

sigmas (`list[float]`, *optional*) : Noise schedule for this pass, without the terminal `0.0`.

noise_scale (`float`, *optional*) : Noise level unconditioned tokens start at. Defaults to `sigmas[0]`.

freeze_audio (`bool`, *optional*, defaults to `False`) : Hold audio at sigma 0 (epilogue / when following a frozen stage-1 waveform).

video_tiles (`list[LTX2DFREpilogueTile]`, *optional*) : Epilogue tiling from `epilogue_tiles()`, so a resolution too large for one forward pass can still step a single canvas: each step runs the transformer once per tile and blends the predictions. Resolved into a token plan against this pass's own RoPE coordinates, which is why the layout is passed rather than the plan — the coordinates only exist once [prepare_latents()](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipeline.prepare_latents) has run.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Random generator(s) for reproducibility.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : Noise scale at decode time.

use_cross_timestep (`bool`, *optional*, defaults to `True`) : Whether to use cross-modality sigma for cross attention modulation. `True` for LTX-2.3+.

system_prompt (`str`, *optional*) : Optional system prompt for prompt enhancement. See `enable_prompt_enhancement`.

enable_prompt_enhancement (`bool`, *optional*, defaults to `False`) : Whether to run prompt enhancement.

prompt_max_new_tokens (`int`, *optional*) : The maximum number of new tokens to generate when performing prompt enhancement.

prompt_enhancement_kwargs (`dict[str, Any]`, *optional*) : Keyword arguments for the prompt enhancer's `.generate` call.

prompt_enhancement_seed (`int`, *optional*, defaults to `10`) : Random seed for any random operations during prompt enhancement.

output_type (`str`, *optional*, defaults to `"pil"`) : Output format. Choose `"pil"`, `"np"`, `"pt"` or `"latent"`. Latent output is the untrimmed canvas.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [LTX2DFRPipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipelineOutput) or a plain `(frames, audio, keyframes, keyframe_positions)` tuple.

attention_kwargs (`dict`, *optional*) : Additional kwargs passed to the attention processor.

callback_on_step_end (`Callable`, *optional*) : A function called at the end of each denoising step.

callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`) : Tensor inputs for the callback function.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length for the text prompt.

**Returns:** [LTX2DFRPipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipelineOutput) or `tuple`

If `return_dict` is `True`, [LTX2DFRPipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipelineOutput) is returned, otherwise a `tuple` of `(video,
audio, keyframes, keyframe_positions)` is returned.

Function invoked when calling the pipeline for generation.

One denoise pass at `height` × `width`. Compose stages in the caller: this pipeline at half-res, spatial
upsample, this pipeline again with `latents` / `keyframes_latents` / `reference_latents`, then
[LTX2DFRTemporalRefinePipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRTemporalRefinePipeline) for each temporal round.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2DFRPipeline
>>> from diffusers.utils import encode_video

>>> pipe = LTX2DFRPipeline.from_pretrained("Lightricks/LTX-2.5-Diffusers", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> frame_rate = 24.0
>>> video, audio, _, _ = pipe(
...     prompt="A tabby cat stretching in a sunlit window, dust motes drifting in the light",
...     height=704,
...     width=1216,
...     num_frames=121,
...     frame_rate=frame_rate,
...     output_type="np",
...     return_dict=False,
... )

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
...     output_path="dfr_output.mp4",
... )
```

#### denoise[[diffusers.LTX2DFRPipeline.denoise]]

```python
denoise(latents: Tensor, conditioning_mask: Tensor, clean_latents: Tensor, video_coords: Tensor, keyframes_mask: Tensor, prompt_embeds: Tensor, audio_prompt_embeds: Tensor, prompt_attention_mask: Tensor, sigmas: list, frame_rate: float, audio_latents: Tensor, freeze_audio: bool = False, video_tile_plan: list | None = None, generator: typing.Optional[torch.Generator] = None, use_cross_timestep: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, progress_bar = None, step_offset: int = 0, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L1225)

**Parameters:**

sigmas (`list[float]`) : Noise schedule for this pass, without the terminal `0.0`.

freeze_audio (`bool`, *optional*, defaults to `False`) : Hold `audio_latents` clean (timestep/sigma 0, no audio Euler step) while still running audio-to-video cross-attention. Ignored when `audio_latents` is `None`.

video_tile_plan (`list`, *optional*) : Per-tile token plan from `video_tile_plan`[`~diffusers.pipelines.ltx2.dfr_layout.video_tile_plan`]. When given, each step runs the transformer once per tile and blends the predictions, so the sampler still steps a single full canvas and the tiles agree on their overlaps at every step.

generator (`torch.Generator`, *optional*) : Forwarded to `LTXEulerAncestralRFScheduler.step()`. Temporal tiles pass a per-tile seed so ancestral draws do not share a stream or consume the state the next tile's initial noising reads. Distilled Euler does not read it.

step_offset (`int`) : Index of this pass's first step within the pipeline's whole schedule, used for `callback_on_step_end` and the shared progress bar.

Run one DFR denoising pass over `sigmas` and return `(latents, audio_latents)`, both still packed.

The distilled schedule is used without classifier-free guidance, so this is a single transformer call per step.
Every pass runs both streams, because the video branch needs the cross-modal attention even where the audio it
produces is thrown away. `freeze_audio=True` keeps the audio stream at sigma 0 (no Euler step) so video can
still cross-attend to it — the temporal refine tiles and the epilogue use this to follow stage-1 speech without
each tile re-denoising a different audio realization.

After the x0 conditioning blend, the velocity is `(latents - denoised) / sigma` so an RF step `x0 = x - σ v`
recovers the blended `denoised`. Stage 1 / 2 / the epilogue keep [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) and take
that Euler step as-is. Temporal refine swaps in `LTXEulerAncestralRFScheduler` (`eta=0.5`); that step
renoises every token, so the conditioning blend is applied again afterwards or strength-0.95 seam anchors
erode.

#### encode_conditions[[diffusers.LTX2DFRPipeline.encode_conditions]]

```python
encode_conditions(conditions: list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None, height: int, width: int, num_frames: int, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L1189)

Preprocess and VAE-encode frame conditions, positioned by pixel frame.

Returns `(pixel_frame_index, latent, strength, num_pixel_frames)` per condition, ready for
[prepare_latents()](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipeline.prepare_latents)'s `condition_latents`. Encoding is kept separate from placement because
the temporal refine rounds scale a condition's position by `2 ** round` and re-base it per tile, and should not
re-encode the same still once per tile to do so.

The returned index is on `num_frames`' own pixel grid; carrying it onto a refined canvas is the caller's job.

#### encode_prompt[[diffusers.LTX2DFRPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L529)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for `prompt_embeds`.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

DFR runs the distilled sigma schedule, which is trained to be used without classifier-free guidance, so there
is no negative branch here.

#### enhance_prompt[[diffusers.LTX2DFRPipeline.enhance_prompt]]

```python
enhance_prompt(prompt: str, system_prompt: str, max_new_tokens: int | None = None, seed: int = 10, generator: typing.Optional[torch.Generator] = None, generation_kwargs: dict[str, typing.Any] | None = None, device: typing.Union[str, torch.device, NoneType] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L389)

Enhances the supplied `prompt` by generating a new prompt using the prompt enhancer (a Gemma
conditional-generation model) from it and a system prompt. When `image` is supplied, the enhancer is also
conditioned on that reference frame (I2V / keyframe-style enhancement). Uses the dedicated `prompt_enhancer`
component if one is configured (e.g. LTX-2.5, whose text encoder isn't trained for enhancement), otherwise
falls back to the main `text_encoder` (LTX-2.0/2.3, which double as their own enhancer).

Message templates, decoding kwargs, response cleaning, and image long-side prep match `ltx-core` /
`ltx-pipelines` (`enhance_t2v` / `enhance_i2v` / `generate_enhanced_prompt`).

#### prepare_latents[[diffusers.LTX2DFRPipeline.prepare_latents]]

```python
prepare_latents(conditions: list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, condition_latents: list[tuple[int, torch.Tensor, float, int]] | None = None, keyframe_latents: list[tuple[int, torch.Tensor, float]] | None = None, slot_frame_indices: list[int] | None = None, slot_initial_latents: typing.Optional[torch.Tensor] = None, reference_latents: typing.Optional[torch.Tensor] = None, reference_downscale_factor: int = 1, batch_size: int = 1, num_channels_latents: int = 128, height: int = 512, width: int = 768, num_frames: int = 121, frame_rate: float = 24.0, noise_scale: float = 1.0, dtype: typing.Optional[torch.dtype] = None, device: typing.Optional[torch.device] = None, generator: typing.Optional[torch.Generator] = None, latents: typing.Optional[torch.Tensor] = None, latents_normalized: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L886)

**Parameters:**

conditions (`list[LTX2VideoCondition]`, *optional*) : Frame-level image / video conditions, positioned by latent index.

condition_latents (`list[tuple[int, torch.Tensor, float, int]]`, *optional*) : Already-encoded stand-in for `conditions`, as `(pixel_frame_index, latent, strength, num_pixel_frames)`. Pixel rather than latent index, because a temporal refine round scales a condition's position by `2 ** round` and the result does not generally land on a latent boundary -- only an appended keyframe token can sit there, and it is placed by pixel. `pixel_frame_index == 0` still means "replace the first frame".

keyframe_latents (`list[tuple[int, torch.Tensor, float]]`, *optional*) : Already-encoded keyframe guidance as `(pixel_frame_index, latent, strength)`, where `latent` has shape `(batch_size, num_channels_latents, 1, latent_height, latent_width)`. Used by the temporal refine rounds to pin the seam keyframes carried in from the previous round.

slot_frame_indices (`list[int]`, *optional*) : Pixel-frame positions of the generated keyframe slots.

slot_initial_latents (`torch.Tensor`, *optional*) : `(batch_size, num_channels_latents, len(slot_frame_indices), latent_height, latent_width)` content written into the slot tokens before noising.

reference_latents (`torch.Tensor`, *optional*) : `(batch_size, num_channels_latents, F, H, W)` IC-LoRA reference latent.

reference_downscale_factor (`int`, defaults to `1`) : Ratio between the target and the reference resolution.

latents (`torch.Tensor`, *optional*) : `(batch_size, num_channels_latents, F, H, W)` initial content for the base tokens. Public pipeline latents are raw (denormalized); pass `latents_normalized=False` at that boundary. Tile loops that already sit in VAE-normalized space leave the default.

latents_normalized (`bool`, defaults to `True`) : Whether `latents`, `slot_initial_latents`, `reference_latents`, and `keyframe_latents` are already VAE-normalized. Internal tile loops pass `True`; each pipeline `__call__` passes `False`.

noise_scale (`float`, defaults to `1.0`) : Noise level the unconditioned tokens are initialized at, i.e. the schedule's first sigma.

**Returns:** `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, slice | None]`

`(latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, slot_token_slice)`.
`slot_token_slice` indexes the generated keyframe slot tokens in the packed sequence, or `None` when no
slots were requested.

Prepare the noisy packed video latents for one DFR denoising pass.

The packed sequence is laid out as `[base | keyframes | slots | reference]`:

- Base tokens cover the target latent grid, seeded from `latents` when supplied.
- Frame conditions with `index == 0` set the clean target at the first-frame positions; those with `index > 0`
  and every entry of `keyframe_latents` are appended as extra keyframe tokens with a per-token conditioning
  mask equal to their strength.
- `slot_frame_indices` appends one latent frame's worth of *generated* keyframe tokens per position, with
  conditioning mask `0` (fully denoised) and a RoPE temporal extent of exactly one pixel frame. These are the
  keyframe slots that give DFR its extra frames; `slot_initial_latents` seeds their content.
- `reference_latents` appends the stage-1 half-resolution latent as a fully clean IC-LoRA reference, with
  spatial coordinates scaled by `reference_downscale_factor` so it maps into the target coordinate space.

Appended conditioning tokens carry their content in `clean_latents` and a zero placeholder in `latents`, while
keyframe slots carry their seed in `latents` and zeros in `clean_latents` -- the returned `latents` are the
noised mix of the two (see the noising step at the end of this method).

#### preprocess_conditions[[diffusers.LTX2DFRPipeline.preprocess_conditions]]

```python
preprocess_conditions(conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, height: int = 512, width: int = 768, num_frames: int = 121, device: typing.Optional[torch.device] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L707)

**Parameters:**

conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*, defaults to `None`) : A list of image/video condition instances.

height (`int`, *optional*, defaults to `512`) : The desired height in pixels.

width (`int`, *optional*, defaults to `768`) : The desired width in pixels.

num_frames (`int`, *optional*, defaults to `121`) : The desired number of frames in the generated video.

device (`torch.device`, *optional*, defaults to `None`) : The device on which to put the preprocessed image/video tensors.

**Returns:** `Tuple[List[torch.Tensor], List[float], List[int], List[int]]`

Returns a 4-tuple of lists of length `len(conditions)` as follows:
1. The first list is a list of preprocessed video tensors of shape [batch_size=1, num_channels,
   num_frames, height, width].
2. The second list is a list of conditioning strengths.
3. The third list is a list of latent-space indices for each condition.
4. The fourth list is a list of (trimmed) pixel-space frame counts per condition. This is needed
   for keyframe coord semantics (single-pixel-frame keyframes have a clamped temporal extent).

Preprocesses the condition images/videos to torch tensors.

#### rebuild_epilogue_keyframes[[diffusers.LTX2DFRPipeline.rebuild_epilogue_keyframes]]

```python
rebuild_epilogue_keyframes(keyframe_latents: Tensor, decode_timestep: float, decode_noise_scale: float, seed: int, device: device, dtype: dtype)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L1454)

**Parameters:**

keyframe_latents (`torch.Tensor`) : Raw `(batch_size, C, K, H, W)` carry keyframes, as a previous pass returned them.

seed (`int`) : Base seed; plane `i` decodes under `seed + 4000 + i`, so a plane's pixels do not depend on how many planes were decoded before it.

**Returns:** `torch.Tensor`

Raw `(batch_size, C, K, 2H, 2W)` latents, ready to pass straight back in as
`guidance_keyframe_latents`.

Re-encode the carry keyframes at twice their resolution by way of RGB.

These are frames the refine rounds already settled, so the epilogue is *given* them rather than asked to
generate them. Decoding to pixels, stretching x2 with Lanczos and encoding again preserves the frame while
landing it on the output grid, which is what lets the epilogue pin it fully clean.

Each plane is decoded as its own one-frame clip. The VAE is causal, so a stacked decode would let neighbouring
planes bleed into each other -- they are independent stills, not a sequence.

#### trim_conditioning_sequence[[diffusers.LTX2DFRPipeline.trim_conditioning_sequence]]

```python
trim_conditioning_sequence(start_frame: int, sequence_num_frames: int, target_num_frames: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr.py#L689)

**Parameters:**

start_frame (int) : The target frame number of the first frame in the sequence.

sequence_num_frames (int) : The number of frames in the sequence.

target_num_frames (int) : The target number of frames in the generated video.

**Returns:** `int`

updated sequence length

Trim a conditioning sequence to the allowed number of frames.

## LTX2DFRTemporalRefinePipeline[[diffusers.LTX2DFRTemporalRefinePipeline]]

#### diffusers.LTX2DFRTemporalRefinePipeline[[diffusers.LTX2DFRTemporalRefinePipeline]]

```python
diffusers.LTX2DFRTemporalRefinePipeline(scheduler: LTXEulerAncestralRFScheduler, vae: AutoencoderKLLTX2Video, audio_vae: AutoencoderKLLTX2Audio, text_encoder: transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration | transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration, tokenizer: GemmaTokenizer, connectors: LTX2TextConnectors, transformer: LTX2VideoTransformer3DModel, vocoder: diffusers.pipelines.ltx2.vocoder.LTX2Vocoder | diffusers.pipelines.ltx2.vocoder.LTX2VocoderWithBWE, temporal_latent_upsampler: LTX2LatentUpsamplerModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L151)

**Parameters:**

scheduler (`LTXEulerAncestralRFScheduler`) : Ancestral Euler scheduler in the rectified-flow parameterization. Construct with `eta=0.5` to match the DFR temporal recipe.

vae ([AutoencoderKLLTX2Video](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_ltx_2#diffusers.AutoencoderKLLTX2Video)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

audio_vae ([AutoencoderKLLTX2Audio](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_audio_ltx_2#diffusers.AutoencoderKLLTX2Audio)) : Audio VAE to encode and decode audio spectrograms.

text_encoder (`Gemma3ForConditionalGeneration` or `Gemma4UnifiedForConditionalGeneration`) : Text encoder model.

tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`) : Tokenizer for the text encoder.

connectors (`LTX2TextConnectors`) : Text connector stack used to adapt text encoder hidden states for the video and audio branches.

transformer ([LTX2VideoTransformer3DModel](/docs/diffusers/v0.41.0/en/api/models/ltx2_video_transformer3d#diffusers.LTX2VideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

vocoder (`LTX2Vocoder` or `LTX2VocoderWithBWE`) : Vocoder to convert mel spectrograms to audio waveforms.

temporal_latent_upsampler (`LTX2LatentUpsamplerModel`) : Temporal x2 latent upsampler applied at the start of the round.

One temporal DFR refine round: interpolate the canvas x2 in time, tile on keyframe seams, ancestral-denoise each
tile, stitch by dropping the later tile's lead-in, and merge the carry-keyframe bag.

The scheduler is `LTXEulerAncestralRFScheduler` (`eta=0.5` by default). Stage 1 / 2 / the spatial epilogue stay
on [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) via [LTX2DFRPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipeline). Call this pipeline once per round; the caller loops
for 2x / 4x.

Incoming `latents` / `keyframes_latents` / `audio_latents` are raw (denormalized), matching [LTX2Pipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2Pipeline).
`keyframe_positions` must be the positions returned by the previous pass — they cannot be re-derived from the
original `num_frames` after a round has run.

#### __call__[[diffusers.LTX2DFRTemporalRefinePipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, latents: Tensor = None, keyframes_latents: Tensor = None, keyframe_positions: list = None, audio_latents: Tensor = None, conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, height: int = 704, width: int = 1216, num_frames: int = 121, frame_rate: float = 24.0, source_seconds: float | None = None, condition_num_frames: int | None = None, round_index: int = 1, sigmas: list = [0.975, 0.909375, 0.725, 0.421875], noise_scale: float | None = None, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, decode_timestep: float | list[float] = 0.0, decode_noise_scale: float | list[float] | None = None, use_cross_timestep: bool = True, output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L1349)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.

latents (`torch.Tensor`) : Raw video latents of the **input** canvas, `(batch_size, channels, frames, height, width)`.

keyframes_latents (`torch.Tensor`) : Raw carry keyframes `(batch_size, channels, K, height, width)` from the previous pass.

keyframe_positions (`list[int]`) : Pixel-frame indices of `keyframes_latents` on the **input** canvas.

audio_latents (`torch.Tensor`) : Frozen stage-1 audio, unpacked and denormalized. Each tile is handed the slice covering its playback window; the returned audio is this waveform, not a per-tile re-denoise.

conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*) : Frame-level conditions indexed on `condition_num_frames` (the original request). Each round scales a condition's pixel position by `2 ** round_index`.

height (`int`, *optional*, defaults to `704`) : Pixel height of this pass (same as the incoming video).

width (`int`, *optional*, defaults to `1216`) : Pixel width of this pass.

num_frames (`int`, *optional*, defaults to `121`) : Pixel frame count of the **input** canvas (untrimmed).

frame_rate (`float`, *optional*, defaults to `24.0`) : Playback fps of the **input**. The round doubles it.

source_seconds (`float`, *optional*) : Duration of the frozen stage-1 audio. Defaults to `num_frames / frame_rate`, which is correct for the first round; later rounds must pass the original stage-1 duration so tiles do not drift.

condition_num_frames (`int`, *optional*) : Original generation `num_frames` used to encode `conditions`. Defaults to `num_frames`. After padding or a prior round, pass the original request so `index=-1` does not wrap to the padded tail.

round_index (`int`, *optional*, defaults to `1`) : 1-based round number. Tiles seed ancestral noise as `seed + 1000 * round_index + tile`, and conditions are scaled by `2 ** round_index`.

sigmas (`list[float]`, *optional*) : Distilled schedule for this round's tiles, without the terminal `0.0`.

noise_scale (`float`, *optional*) : Noise level unconditioned tokens start at. Defaults to `sigmas[0]`.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Random generator(s) for reproducibility. Ancestral draws use a separate per-tile seed derived from this generator's `initial_seed()`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : Noise scale at decode time.

use_cross_timestep (`bool`, *optional*, defaults to `True`) : Whether to use cross-modality sigma for cross attention modulation. `True` for LTX-2.3+.

output_type (`str`, *optional*, defaults to `"pil"`) : Output format. Choose `"pil"`, `"np"`, `"pt"` or `"latent"`. Latent output is the untrimmed canvas.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [LTX2DFRPipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipelineOutput) or a plain `(frames, audio, keyframes, keyframe_positions)` tuple.

attention_kwargs (`dict`, *optional*) : Additional kwargs passed to the attention processor.

callback_on_step_end (`Callable`, *optional*) : A function called at the end of each denoising step, across every tile.

callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`) : Tensor inputs for the callback function.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length for the text prompt.

**Returns:** [LTX2DFRPipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipelineOutput) or `tuple`

If `return_dict` is `True`, [LTX2DFRPipelineOutput](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipelineOutput) is returned, otherwise a `tuple` of `(video,
audio, keyframes, keyframe_positions)` is returned.

Run one temporal refine round.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2DFRPipeline, LTX2DFRTemporalRefinePipeline, LTXEulerAncestralRFScheduler
>>> from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel

>>> pipe = LTX2DFRPipeline.from_pretrained("Lightricks/LTX-2.5-Diffusers", torch_dtype=torch.bfloat16)
>>> temporal_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
...     "path/to/converted/temporal_latent_upsampler", torch_dtype=torch.bfloat16
... )
>>> temporal_pipe = LTX2DFRTemporalRefinePipeline(
...     scheduler=LTXEulerAncestralRFScheduler(eta=0.5),
...     vae=pipe.vae,
...     audio_vae=pipe.audio_vae,
...     text_encoder=pipe.text_encoder,
...     tokenizer=pipe.tokenizer,
...     connectors=pipe.connectors,
...     transformer=pipe.transformer,
...     vocoder=pipe.vocoder,
...     temporal_latent_upsampler=temporal_upsampler,
... )
>>> # `out` is a prior DFR pass at the same spatial size, `return_dict=True`.
>>> out = temporal_pipe(
...     latents=out.frames,
...     keyframes_latents=out.keyframes,
...     keyframe_positions=out.keyframe_positions,
...     audio_latents=out.audio,
...     prompt="A tabby cat stretching in a sunlit window",
...     height=1088,
...     width=1920,
...     num_frames=121,
...     output_type="latent",
... )
```

#### denoise[[diffusers.LTX2DFRTemporalRefinePipeline.denoise]]

```python
denoise(latents: Tensor, conditioning_mask: Tensor, clean_latents: Tensor, video_coords: Tensor, keyframes_mask: Tensor, prompt_embeds: Tensor, audio_prompt_embeds: Tensor, prompt_attention_mask: Tensor, sigmas: list, frame_rate: float, audio_latents: Tensor, freeze_audio: bool = False, video_tile_plan: list | None = None, generator: typing.Optional[torch.Generator] = None, use_cross_timestep: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, progress_bar = None, step_offset: int = 0, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L1138)

**Parameters:**

sigmas (`list[float]`) : Noise schedule for this pass, without the terminal `0.0`.

freeze_audio (`bool`, *optional*, defaults to `False`) : Hold `audio_latents` clean (timestep/sigma 0, no audio Euler step) while still running audio-to-video cross-attention. Ignored when `audio_latents` is `None`.

video_tile_plan (`list`, *optional*) : Per-tile token plan from `video_tile_plan`[`~diffusers.pipelines.ltx2.dfr_layout.video_tile_plan`]. When given, each step runs the transformer once per tile and blends the predictions, so the sampler still steps a single full canvas and the tiles agree on their overlaps at every step.

generator (`torch.Generator`, *optional*) : Forwarded to `LTXEulerAncestralRFScheduler.step()`. Temporal tiles pass a per-tile seed so ancestral draws do not share a stream or consume the state the next tile's initial noising reads. Distilled Euler does not read it.

step_offset (`int`) : Index of this pass's first step within the pipeline's whole schedule, used for `callback_on_step_end` and the shared progress bar.

Run one DFR denoising pass over `sigmas` and return `(latents, audio_latents)`, both still packed.

The distilled schedule is used without classifier-free guidance, so this is a single transformer call per step.
Every pass runs both streams, because the video branch needs the cross-modal attention even where the audio it
produces is thrown away. `freeze_audio=True` keeps the audio stream at sigma 0 (no Euler step) so video can
still cross-attend to it — the temporal refine tiles and the epilogue use this to follow stage-1 speech without
each tile re-denoising a different audio realization.

After the x0 conditioning blend, the velocity is `(latents - denoised) / sigma` so an RF step `x0 = x - σ v`
recovers the blended `denoised`. Stage 1 / 2 / the epilogue keep [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) and take
that Euler step as-is. Temporal refine swaps in `LTXEulerAncestralRFScheduler` (`eta=0.5`); that step
renoises every token, so the conditioning blend is applied again afterwards or strength-0.95 seam anchors
erode.

#### encode_conditions[[diffusers.LTX2DFRTemporalRefinePipeline.encode_conditions]]

```python
encode_conditions(conditions: list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None, height: int, width: int, num_frames: int, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L1101)

Preprocess and VAE-encode frame conditions, positioned by pixel frame.

Returns `(pixel_frame_index, latent, strength, num_pixel_frames)` per condition, ready for
[prepare_latents()](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2DFRPipeline.prepare_latents)'s `condition_latents`. Encoding is kept separate from placement because
the temporal refine rounds scale a condition's position by `2 ** round` and re-base it per tile, and should not
re-encode the same still once per tile to do so.

The returned index is on `num_frames`' own pixel grid; carrying it onto a refined canvas is the caller's job.

#### encode_prompt[[diffusers.LTX2DFRTemporalRefinePipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L470)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for `prompt_embeds`.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

DFR runs the distilled sigma schedule, which is trained to be used without classifier-free guidance, so there
is no negative branch here.

#### prepare_latents[[diffusers.LTX2DFRTemporalRefinePipeline.prepare_latents]]

```python
prepare_latents(conditions: list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, condition_latents: list[tuple[int, torch.Tensor, float, int]] | None = None, keyframe_latents: list[tuple[int, torch.Tensor, float]] | None = None, slot_frame_indices: list[int] | None = None, slot_initial_latents: typing.Optional[torch.Tensor] = None, reference_latents: typing.Optional[torch.Tensor] = None, reference_downscale_factor: int = 1, batch_size: int = 1, num_channels_latents: int = 128, height: int = 512, width: int = 768, num_frames: int = 121, frame_rate: float = 24.0, noise_scale: float = 1.0, dtype: typing.Optional[torch.dtype] = None, device: typing.Optional[torch.device] = None, generator: typing.Optional[torch.Generator] = None, latents: typing.Optional[torch.Tensor] = None, latents_normalized: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L828)

**Parameters:**

conditions (`list[LTX2VideoCondition]`, *optional*) : Frame-level image / video conditions, positioned by latent index.

condition_latents (`list[tuple[int, torch.Tensor, float, int]]`, *optional*) : Already-encoded stand-in for `conditions`, as `(pixel_frame_index, latent, strength, num_pixel_frames)`. Pixel rather than latent index, because a temporal refine round scales a condition's position by `2 ** round` and the result does not generally land on a latent boundary -- only an appended keyframe token can sit there, and it is placed by pixel. `pixel_frame_index == 0` still means "replace the first frame".

keyframe_latents (`list[tuple[int, torch.Tensor, float]]`, *optional*) : Already-encoded keyframe guidance as `(pixel_frame_index, latent, strength)`, where `latent` has shape `(batch_size, num_channels_latents, 1, latent_height, latent_width)`. Used by the temporal refine rounds to pin the seam keyframes carried in from the previous round.

slot_frame_indices (`list[int]`, *optional*) : Pixel-frame positions of the generated keyframe slots.

slot_initial_latents (`torch.Tensor`, *optional*) : `(batch_size, num_channels_latents, len(slot_frame_indices), latent_height, latent_width)` content written into the slot tokens before noising.

reference_latents (`torch.Tensor`, *optional*) : `(batch_size, num_channels_latents, F, H, W)` IC-LoRA reference latent.

reference_downscale_factor (`int`, defaults to `1`) : Ratio between the target and the reference resolution.

latents (`torch.Tensor`, *optional*) : `(batch_size, num_channels_latents, F, H, W)` initial content for the base tokens. Public pipeline latents are raw (denormalized); pass `latents_normalized=False` at that boundary. Tile loops that already sit in VAE-normalized space leave the default.

latents_normalized (`bool`, defaults to `True`) : Whether `latents`, `slot_initial_latents`, `reference_latents`, and `keyframe_latents` are already VAE-normalized. Internal tile loops pass `True`; each pipeline `__call__` passes `False`.

noise_scale (`float`, defaults to `1.0`) : Noise level the unconditioned tokens are initialized at, i.e. the schedule's first sigma.

**Returns:** `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, slice | None]`

`(latents, conditioning_mask, clean_latents, video_coords, keyframes_mask, slot_token_slice)`.
`slot_token_slice` indexes the generated keyframe slot tokens in the packed sequence, or `None` when no
slots were requested.

Prepare the noisy packed video latents for one DFR denoising pass.

The packed sequence is laid out as `[base | keyframes | slots | reference]`:

- Base tokens cover the target latent grid, seeded from `latents` when supplied.
- Frame conditions with `index == 0` set the clean target at the first-frame positions; those with `index > 0`
  and every entry of `keyframe_latents` are appended as extra keyframe tokens with a per-token conditioning
  mask equal to their strength.
- `slot_frame_indices` appends one latent frame's worth of *generated* keyframe tokens per position, with
  conditioning mask `0` (fully denoised) and a RoPE temporal extent of exactly one pixel frame. These are the
  keyframe slots that give DFR its extra frames; `slot_initial_latents` seeds their content.
- `reference_latents` appends the stage-1 half-resolution latent as a fully clean IC-LoRA reference, with
  spatial coordinates scaled by `reference_downscale_factor` so it maps into the target coordinate space.

Appended conditioning tokens carry their content in `clean_latents` and a zero placeholder in `latents`, while
keyframe slots carry their seed in `latents` and zeros in `clean_latents` -- the returned `latents` are the
noised mix of the two (see the noising step at the end of this method).

#### preprocess_conditions[[diffusers.LTX2DFRTemporalRefinePipeline.preprocess_conditions]]

```python
preprocess_conditions(conditions: diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None, height: int = 512, width: int = 768, num_frames: int = 121, device: typing.Optional[torch.device] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L648)

**Parameters:**

conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*, defaults to `None`) : A list of image/video condition instances.

height (`int`, *optional*, defaults to `512`) : The desired height in pixels.

width (`int`, *optional*, defaults to `768`) : The desired width in pixels.

num_frames (`int`, *optional*, defaults to `121`) : The desired number of frames in the generated video.

device (`torch.device`, *optional*, defaults to `None`) : The device on which to put the preprocessed image/video tensors.

**Returns:** `Tuple[List[torch.Tensor], List[float], List[int], List[int]]`

Returns a 4-tuple of lists of length `len(conditions)` as follows:
1. The first list is a list of preprocessed video tensors of shape [batch_size=1, num_channels,
   num_frames, height, width].
2. The second list is a list of conditioning strengths.
3. The third list is a list of latent-space indices for each condition.
4. The fourth list is a list of (trimmed) pixel-space frame counts per condition. This is needed
   for keyframe coord semantics (single-pixel-frame keyframes have a clamped temporal extent).

Preprocesses the condition images/videos to torch tensors.

#### trim_conditioning_sequence[[diffusers.LTX2DFRTemporalRefinePipeline.trim_conditioning_sequence]]

```python
trim_conditioning_sequence(start_frame: int, sequence_num_frames: int, target_num_frames: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L630)

**Parameters:**

start_frame (int) : The target frame number of the first frame in the sequence.

sequence_num_frames (int) : The number of frames in the sequence.

target_num_frames (int) : The target number of frames in the generated video.

**Returns:** `int`

updated sequence length

Trim a conditioning sequence to the allowed number of frames.

#### upsample_latents[[diffusers.LTX2DFRTemporalRefinePipeline.upsample_latents]]

```python
upsample_latents(latents: Tensor, upsampler: LTX2LatentUpsamplerModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_dfr_temporal_refine.py#L1090)

Run `upsampler` on normalized latents, round-tripping through raw VAE latent space as it expects.

## LTX2DFRPipelineOutput[[diffusers.LTX2DFRPipelineOutput]]

#### diffusers.LTX2DFRPipelineOutput[[diffusers.LTX2DFRPipelineOutput]]

```python
diffusers.LTX2DFRPipelineOutput(frames: Tensor, audio: Tensor, keyframes: typing.Optional[torch.Tensor] = None, keyframe_positions: list[int] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_output.py#L27)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : Denoised video. Latent output is the untrimmed canvas, shape `(batch_size, num_channels, latent_frames, latent_height, latent_width)`.

audio (`torch.Tensor`, `np.ndarray`) : Accompanying audio latents or waveform.

keyframes (`torch.Tensor`, *optional*) : Generated or carried keyframe latents of shape `(batch_size, num_channels, num_keyframes, latent_height, latent_width)`. `None` when `output_type != "latent"`, or when the pass did not produce slots (e.g. a tiled epilogue).

keyframe_positions (`list[int]`, *optional*) : Pixel-frame index of each keyframe on this pass's canvas; `None` whenever `keyframes` is `None`. After a temporal round these cannot be re-derived from the original `num_frames` and must be passed into the next stage.

Output class for DFR pipelines.

## LTX2LatentUpsamplePipeline[[diffusers.LTX2LatentUpsamplePipeline]]

#### diffusers.LTX2LatentUpsamplePipeline[[diffusers.LTX2LatentUpsamplePipeline]]

```python
diffusers.LTX2LatentUpsamplePipeline(vae: AutoencoderKLLTX2Video, latent_upsampler: LTX2LatentUpsamplerModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L104)

#### __call__[[diffusers.LTX2LatentUpsamplePipeline.__call__]]

```python
__call__(video: list[typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]] | None = None, height: int = 512, width: int = 768, num_frames: int = 121, spatial_patch_size: int = 1, temporal_patch_size: int = 1, latents: typing.Optional[torch.Tensor] = None, latents_normalized: bool = False, decode_timestep: float | list[float] = 0.0, decode_noise_scale: float | list[float] | None = None, adain_factor: float = 0.0, tone_map_compression_ratio: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str | None = 'pil', return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L264)

**Parameters:**

video (`list[PipelineImageInput]`, *optional*) : The video to be upsampled (such as a LTX 2.0 first stage output). If not supplied, `latents` should be supplied.

height (`int`, *optional*, defaults to `512`) : The height in pixels of the input video (not the generated video, which will have a larger resolution).

width (`int`, *optional*, defaults to `768`) : The width in pixels of the input video (not the generated video, which will have a larger resolution).

num_frames (`int`, *optional*, defaults to `121`) : The number of frames in the input video.

spatial_patch_size (`int`, *optional*, defaults to `1`) : The spatial patch size of the video latents. Used when `latents` is supplied if unpacking is necessary.

temporal_patch_size (`int`, *optional*, defaults to `1`) : The temporal patch size of the video latents. Used when `latents` is supplied if unpacking is necessary.

latents (`torch.Tensor`, *optional*) : Pre-generated video latents. This can be supplied in place of the `video` argument. Can either be a patch sequence of shape `(batch_size, seq_len, hidden_dim)` or a video latent of shape `(batch_size, latent_channels, latent_frames, latent_height, latent_width)`.

latents_normalized (`bool`, *optional*, defaults to `False`) : If `latents` are supplied, whether the `latents` are normalized using the VAE latent mean and std. If `True`, the `latents` will be denormalized before being supplied to the latent upsampler.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

adain_factor (`float`, *optional*, defaults to `0.0`) : Adaptive Instance Normalization (AdaIN) blending factor between the upsampled and original latents. Should be in [-10.0, 10.0]; supplying 0.0 (the default) means that AdaIN is not performed.

tone_map_compression_ratio (`float`, *optional*, defaults to `0.0`) : The compression strength for tone mapping, which will reduce the dynamic range of the latent values. This is useful for regularizing high-variance latents or for conditioning outputs during generation. Should be in [0, 1], where 0.0 (the default) means tone mapping is not applied and 1.0 corresponds to the full compression effect.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.

**Returns:** `~pipelines.ltx.LTXPipelineOutput` or `tuple`

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is the upsampled video.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
>>> from diffusers.utils import encode_video
>>> from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
>>> from diffusers.utils import load_image

>>> pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> image = load_image(
...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
... )
>>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> frame_rate = 24.0
>>> video, audio = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=40,
...     guidance_scale=4.0,
...     output_type="pil",
...     return_dict=False,
... )

>>> latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
...     "Lightricks/LTX-2", subfolder="latent_upsampler", torch_dtype=torch.bfloat16
... )
>>> upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
>>> upsample_pipe.vae.enable_tiling()
>>> upsample_pipe.to(device="cuda", dtype=torch.bfloat16)

>>> video = upsample_pipe(
...     video=video,
...     width=768,
...     height=512,
...     output_type="np",
...     return_dict=False,
... )[0]

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

#### adain_filter_latent[[diffusers.LTX2LatentUpsamplePipeline.adain_filter_latent]]

```python
adain_filter_latent(latents: Tensor, reference_latents: Tensor, factor: float = 1.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L168)

**Parameters:**

latents (`torch.Tensor`) : Input latents to normalize

reference_latents (`torch.Tensor`) : The reference latents providing style statistics.

factor (`float`) : Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

**Returns:** `torch.Tensor`

The transformed latent tensor

Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
tensor.

#### tone_map_latents[[diffusers.LTX2LatentUpsamplePipeline.tone_map_latents]]

```python
tone_map_latents(latents: Tensor, compression: float)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L196)

**Parameters:**

latents : torch.Tensor Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.

compression : float Compression strength in the range [0, 1]. - 0.0: No tone-mapping (identity transform) - 1.0: Full compression effect

**Returns:**

torch.Tensor
The tone-mapped latent tensor of the same shape as input.

Applies a non-linear tone-mapping function to latent values to reduce their dynamic range in a perceptually
smooth way using a sigmoid-based compression.

This is useful for regularizing high-variance latents or for conditioning outputs during generation, especially
when controlling dynamic behavior with a `compression` factor.

## LTX2VideoDiffusionDecodePipeline[[diffusers.LTX2VideoDiffusionDecodePipeline]]

#### diffusers.LTX2VideoDiffusionDecodePipeline[[diffusers.LTX2VideoDiffusionDecodePipeline]]

```python
diffusers.LTX2VideoDiffusionDecodePipeline(diffusion_decoder: LTX2VideoDiffusionDecoderModel, scheduler, vae: AutoencoderKLLTX2Video = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_diffusion_decode.py#L27)

**Parameters:**

diffusion_decoder ([LTX2VideoDiffusionDecoderModel](/docs/diffusers/v0.41.0/en/api/models/ltx2_diffusion_decoder#diffusers.LTX2VideoDiffusionDecoderModel)) : The diffusion video decoder.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : Scheduler driving the decoder's denoising steps.

vae ([AutoencoderKLLTX2Video](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_ltx_2#diffusers.AutoencoderKLLTX2Video), *optional*) : Only consulted for the latent statistics used to denormalize. When omitted the pipeline falls back to the LTX-2 defaults, so a decode-only workflow does not have to load a second autoencoder.

Decode LTX-2 video latents with the diffusion decoder introduced in LTX-2.5.

Unlike a convolutional decoder this one is itself a small diffusion model: it denoises pixels conditioned on a
context volume built from the latents, so it needs a scheduler and a generator. Pair it with any LTX-2 pipeline run
with `output_type="latent"`, passing `denormalize=False` since that path already applied the latent statistics.

#### __call__[[diffusers.LTX2VideoDiffusionDecodePipeline.__call__]]

```python
__call__(latents: Tensor, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str = 'pil', return_dict: bool = True, denormalize: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_diffusion_decode.py#L79)

**Parameters:**

latents (`torch.Tensor`) : Latents of shape `(B, C, F, H, W)`. Note that an LTX-2 pipeline run with `output_type="latent"` returns latents that are *already* denormalized, so pass `denormalize=False` for those.

generator (`torch.Generator`, *optional*) : The decoder samples the noise it denoises, so pass a generator to make decoding reproducible.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the decoded video. Choose between `"pil"`, `"np"`, `"pt"` and `"latent"`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `LTX2VideoDecodeOutput` instead of a plain tuple.

denormalize (`bool`, *optional*, defaults to `True`) : Whether to apply the latent statistics before decoding. Set to `False` if the latents are already denormalized.

**Returns:** `LTX2VideoDecodeOutput` or `tuple`

## LTX2DurationHead[[diffusers.pipelines.ltx2.LTX2DurationHead]]

#### diffusers.pipelines.ltx2.LTX2DurationHead[[diffusers.pipelines.ltx2.LTX2DurationHead]]

```python
diffusers.pipelines.ltx2.LTX2DurationHead(video_cross_attention_dim: int = 4096, audio_cross_attention_dim: int = 2048, pooler_hidden_dim: int = 256, num_queries: int = 1, num_pooler_heads: int = 4, mlp_hidden_dim: int = 256)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/duration_head.py#L81)

**Parameters:**

video_cross_attention_dim (`int`, defaults to `4096`) : Width of the video connector output.

audio_cross_attention_dim (`int`, defaults to `2048`) : Width of the audio connector output.

pooler_hidden_dim (`int`, defaults to `256`) : Shared hidden dimension both modalities are projected into.

num_queries (`int`, defaults to `1`) : Number of learnable pooling queries.

num_pooler_heads (`int`, defaults to `4`) : Attention heads used by the pooler.

mlp_hidden_dim (`int`, defaults to `256`) : Hidden width of the output MLP. Named with a `_dim` suffix to avoid colliding with the `mlp_hidden` submodule, which `ConfigMixin.__getattr__` would otherwise shadow with this config value.

Predicts the natural duration of the shot implied by a caption, from the LTX-2 text connector outputs.

The head is modality-agnostic: pass either or both of the video and audio connector outputs. Modality-specific
input projections map each stream into a shared pooler dimension, learnable modality embeddings tag the streams so
the pooler can tell them apart, and a small MLP turns the pooled vector into a log-duration. The regression target
is trained in log-seconds, so `forward` exponentiates and callers always get seconds.

Ships from LTX-2.5 checkpoints onward.

#### forward[[diffusers.pipelines.ltx2.LTX2DurationHead.forward]]

```python
forward(video_tokens: typing.Optional[torch.Tensor] = None, audio_tokens: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/duration_head.py#L134)

**Parameters:**

video_tokens (`torch.Tensor` of shape `(batch_size, seq_len, video_cross_attention_dim)`, *optional*) : Video connector output.

audio_tokens (`torch.Tensor` of shape `(batch_size, seq_len, audio_cross_attention_dim)`, *optional*) : Audio connector output.

**Returns:** `torch.Tensor` of shape `(batch_size,)`

the predicted duration in seconds.

#### predict_num_frames[[diffusers.pipelines.ltx2.LTX2DurationHead.predict_num_frames]]

```python
predict_num_frames(video_tokens: typing.Optional[torch.Tensor] = None, audio_tokens: typing.Optional[torch.Tensor] = None, frame_rate: float, temporal_compression_ratio: int, min_seconds: float = 1.0, max_seconds: float = 20.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/duration_head.py#L172)

**Parameters:**

video_tokens (`torch.Tensor`, *optional*) : Video connector output for a single prompt.

audio_tokens (`torch.Tensor`, *optional*) : Audio connector output for a single prompt.

frame_rate (`float`) : Frames per second used to convert the predicted duration into a frame count.

temporal_compression_ratio (`int`) : The VAE's temporal compression ratio, which defines the frame grid.

min_seconds (`float`, defaults to `1.0`) : Lower bound on the prediction.

max_seconds (`float`, defaults to `20.0`) : Upper bound on the prediction.

**Returns:** `int`

a frame count lying on the VAE's temporal grid.

Predicts a frame count from connector tokens, clamped to `[min_seconds, max_seconds]` and snapped to the VAE's
causal temporal grid (`k * temporal_compression_ratio + 1`).

The clamp is applied before snapping: a clamped frame count is not necessarily grid-aligned, so snapping first
would give a different result. Because snapping floors, it can land below the minimum; when that happens the
result is snapped up to the next grid point instead, so the frame count stays within bounds.

Narrow bounds can convert to a frame window containing no grid point at all -- at 24 fps, `[1.0s, 1.02s]`
rounds to `[24, 24]`, and 24 is not `8k + 1`. The nearest grid point is used and a warning is logged, since
overshooting by under one grid step beats refusing to generate. The returned count is therefore always on the
grid, but may fall just outside the requested bounds in this case.

## LTX2PipelineOutput[[diffusers.pipelines.ltx2.LTX2PipelineOutput]]

#### diffusers.pipelines.ltx2.LTX2PipelineOutput[[diffusers.pipelines.ltx2.LTX2PipelineOutput]]

```python
diffusers.pipelines.ltx2.LTX2PipelineOutput(frames: Tensor, audio: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/ltx2/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

audio (`torch.Tensor`, `np.ndarray`) : TODO

Output class for LTX pipelines.

## LTX2ModularPipeline[[diffusers.LTX2ModularPipeline]]

#### diffusers.LTX2ModularPipeline[[diffusers.LTX2ModularPipeline]]

```python
diffusers.LTX2ModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/ltx2/modular_pipeline.py#L26)

A ModularPipeline for LTX-2 (joint video + audio generation).

## LTX2AutoBlocks[[diffusers.LTX2AutoBlocks]]

#### diffusers.LTX2AutoBlocks[[diffusers.LTX2AutoBlocks]]

```python
diffusers.LTX2AutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/ltx2/modular_blocks_ltx2.py#L1721)

Auto blocks for LTX-2 supporting text-to-video, image-to-video, condition-to-video and in-context (IC-LoRA)
generation (joint video + audio).

Supported workflows:
- `text2video`: requires `prompt`
- `image2video`: requires `image`, `prompt`
- `condition`: requires `conditions`, `prompt`
- `in_context`: requires `reference_conditions`, `num_frames`, `prompt`

Components:
prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
(`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
(`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) transformer (`LTX2VideoTransformer3DModel`)
scheduler (`FlowMatchEulerDiscreteScheduler`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`)
audio_guider (`LTX2Guidance`) vocoder (`LTX2Vocoder`)

Inputs:
prompt (`str`, *optional*):
The prompt or prompts to guide image generation.
conditions (`list`, *optional*):
`LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
generated video.
enable_prompt_enhancement (`bool`, *optional*, defaults to False):
Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.
system_prompt (`str`, *optional*):
System prompt for enhancement. Defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` when a `PIL.Image.Image`
condition frame is available, else `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`.
prompt_max_new_tokens (`int`, *optional*):
Maximum number of new tokens to generate during prompt enhancement. Defaults to 600, the LTX-2.5 Gemma-4
enhancer's budget.
prompt_enhancement_kwargs (`dict`, *optional*):
Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.
prompt_enhancement_seed (`int`, *optional*, defaults to 10):
Random seed for prompt enhancement (inert under LTX-2.5's greedy decoding).
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
image (`Image | list`, *optional*):
Reference image(s) for denoising. Can be a single image or list of images.
negative_prompt (`str`, *optional*):
The prompt or prompts not to guide the image generation.
max_sequence_length (`int`, *optional*, defaults to 1024):
Maximum sequence length for prompt encoding.
min_seconds (`float`, *optional*, defaults to 1.0):
Lower bound on the auto-predicted duration.
max_seconds (`float`, *optional*, defaults to 20.0):
Upper bound on the auto-predicted duration. Must be strictly greater than `min_seconds`.
frame_rate (`float`, *optional*, defaults to 24.0):
Frames per second of the generated video.
height (`int`, *optional*, defaults to 512):
The height in pixels of the generated image.
width (`int`, *optional*, defaults to 704):
The width in pixels of the generated image.
image_crf (`int`, *optional*):
H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the
model was trained against. `None` (default) resolves from the text-encoder generation (33 through
LTX-2.3, 18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when
re-compression runs.
num_frames (`int`, *optional*):
The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
`LTX2AutoDurationStep`).
reference_conditions (`list`, *optional*):
`LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens the IC-LoRA
adapter attends to.
reference_downscale_factor (`int`, *optional*, defaults to 1):
Ratio between the target and reference resolutions; 2 means the reference is preprocessed at half the
target resolution. Spatial coordinates are scaled by this factor so the reference tokens land in the
target coordinate space. Must match the factor the IC-LoRA was trained with.
conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
(default) leaves attention unmasked.
conditioning_attention_mask (`Tensor`, *optional*):
Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
attention strength. Downsampled to the reference's latent grid and multiplied by
`conditioning_attention_strength`.
num_videos_per_prompt (`int`, *optional*, defaults to 1):
The number of images to generate per prompt.
condition_latents (`list`, *optional*):
Per-condition normalized VAE latents of shape [1, C, F, H, W].
condition_strengths (`list`, *optional*):
Per-condition conditioning strengths.
condition_indices (`list`, *optional*):
Per-condition latent frame index at which the condition is applied.
condition_pixel_frames (`list`, *optional*):
Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
reference_latents (`Tensor`, *optional*):
Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference conditions
were supplied (`LTX2AutoReferenceEncoderStep` is skipped).
reference_coords (`Tensor`, *optional*):
RoPE coordinates for the reference tokens.
reference_token_counts (`list`, *optional*):
Per-reference token counts, in `reference_conditions` order.
latents (`Tensor`):
Pre-generated noisy latents for image generation.
noise_scale (`float`, *optional*):
Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
`sigmas` are supplied, else 1.0.
sigmas (`list`, *optional*):
Custom sigmas for the denoising process.
reference_cross_mask (`Tensor`, *optional*):
Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
num_inference_steps (`int`):
The number of denoising steps.
timesteps (`Tensor`):
Timesteps for the denoising process.
audio_latents (`Tensor`):
Optional pre-encoded audio latents; random noise is used when not provided.
**denoiser_input_fields (`None`, *optional*):
conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
use_cross_timestep (`bool`, *optional*, defaults to True):
Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
image_latents (`Tensor`, *optional*):
VAE-encoded reference-image latents used for image-to-video conditioning.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.
decode_timestep (`None`, *optional*, defaults to 0.0):
The timestep at which the VAE decodes the final latents.
decode_noise_scale (`None`, *optional*):
Noise interpolation factor applied to the latents at the decode timestep.

Outputs:
videos (`list`):
The generated videos.
audio (`Tensor`):
The generated audio waveform.

## LTX25ModularPipeline[[diffusers.LTX25ModularPipeline]]

#### diffusers.LTX25ModularPipeline[[diffusers.LTX25ModularPipeline]]

```python
diffusers.LTX25ModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/ltx2/modular_pipeline.py#L125)

A ModularPipeline for LTX-2.5 (joint video + audio generation).

Identical to [LTX2ModularPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/ltx2#diffusers.LTX2ModularPipeline) except that its default blocks decode with the diffusion video decoder, which
is the native default from LTX-2.5 on. A checkpoint routes here through `modular_model_index.json`.

## LTX25AutoBlocks[[diffusers.LTX25AutoBlocks]]

#### diffusers.LTX25AutoBlocks[[diffusers.LTX25AutoBlocks]]

```python
diffusers.LTX25AutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/ltx2/modular_blocks_ltx25.py#L215)

Auto blocks for LTX-2.5 supporting text-to-video, image-to-video, condition-to-video and in-context (IC-LoRA)
generation (joint video + audio). Identical to `LTX2AutoBlocks` except that the video decoder is
`LTX2DiffusionVaeDecoderStep`, since the diffusion decoder is the native default from LTX-2.5 on. To decode with
the convolutional VAE instead, swap the decode block: `blocks.sub_blocks["decode"] = LTX2AutoDecoderStep()`.

Supported workflows:
- `text2video`: requires `prompt`
- `image2video`: requires `image`, `prompt`
- `condition`: requires `conditions`, `prompt`
- `in_context`: requires `reference_conditions`, `num_frames`, `prompt`

Components:
prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
(`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
(`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) transformer (`LTX2VideoTransformer3DModel`)
scheduler (`FlowMatchEulerDiscreteScheduler`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`)
audio_guider (`LTX2Guidance`) diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) vocoder (`LTX2Vocoder`)

Inputs:
prompt (`str`, *optional*):
The prompt or prompts to guide image generation.
conditions (`list`, *optional*):
`LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
generated video.
enable_prompt_enhancement (`bool`, *optional*, defaults to False):
Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.
system_prompt (`str`, *optional*):
System prompt for enhancement. Defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` when a `PIL.Image.Image`
condition frame is available, else `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`.
prompt_max_new_tokens (`int`, *optional*):
Maximum number of new tokens to generate during prompt enhancement. Defaults to 600, the LTX-2.5 Gemma-4
enhancer's budget.
prompt_enhancement_kwargs (`dict`, *optional*):
Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.
prompt_enhancement_seed (`int`, *optional*, defaults to 10):
Random seed for prompt enhancement (inert under LTX-2.5's greedy decoding).
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
image (`Image | list`, *optional*):
Reference image(s) for denoising. Can be a single image or list of images.
negative_prompt (`str`, *optional*):
The prompt or prompts not to guide the image generation.
max_sequence_length (`int`, *optional*, defaults to 1024):
Maximum sequence length for prompt encoding.
min_seconds (`float`, *optional*, defaults to 1.0):
Lower bound on the auto-predicted duration.
max_seconds (`float`, *optional*, defaults to 20.0):
Upper bound on the auto-predicted duration. Must be strictly greater than `min_seconds`.
frame_rate (`float`, *optional*, defaults to 24.0):
Frames per second of the generated video.
height (`int`, *optional*, defaults to 512):
The height in pixels of the generated image.
width (`int`, *optional*, defaults to 704):
The width in pixels of the generated image.
image_crf (`int`, *optional*):
H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the
model was trained against. `None` (default) resolves from the text-encoder generation (33 through
LTX-2.3, 18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when
re-compression runs.
num_frames (`int`, *optional*):
The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
`LTX2AutoDurationStep`).
reference_conditions (`list`, *optional*):
`LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens the IC-LoRA
adapter attends to.
reference_downscale_factor (`int`, *optional*, defaults to 1):
Ratio between the target and reference resolutions; 2 means the reference is preprocessed at half the
target resolution. Spatial coordinates are scaled by this factor so the reference tokens land in the
target coordinate space. Must match the factor the IC-LoRA was trained with.
conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
(default) leaves attention unmasked.
conditioning_attention_mask (`Tensor`, *optional*):
Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
attention strength. Downsampled to the reference's latent grid and multiplied by
`conditioning_attention_strength`.
num_videos_per_prompt (`int`, *optional*, defaults to 1):
The number of images to generate per prompt.
condition_latents (`list`, *optional*):
Per-condition normalized VAE latents of shape [1, C, F, H, W].
condition_strengths (`list`, *optional*):
Per-condition conditioning strengths.
condition_indices (`list`, *optional*):
Per-condition latent frame index at which the condition is applied.
condition_pixel_frames (`list`, *optional*):
Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
reference_latents (`Tensor`, *optional*):
Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference conditions
were supplied (`LTX2AutoReferenceEncoderStep` is skipped).
reference_coords (`Tensor`, *optional*):
RoPE coordinates for the reference tokens.
reference_token_counts (`list`, *optional*):
Per-reference token counts, in `reference_conditions` order.
latents (`Tensor`):
Pre-generated noisy latents for image generation.
noise_scale (`float`, *optional*):
Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
`sigmas` are supplied, else 1.0.
sigmas (`list`, *optional*):
Custom sigmas for the denoising process.
reference_cross_mask (`Tensor`, *optional*):
Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
num_inference_steps (`int`):
The number of denoising steps.
timesteps (`Tensor`):
Timesteps for the denoising process.
audio_latents (`Tensor`):
Optional pre-encoded audio latents; random noise is used when not provided.
**denoiser_input_fields (`None`, *optional*):
conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
use_cross_timestep (`bool`, *optional*, defaults to True):
Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
image_latents (`Tensor`, *optional*):
VAE-encoded reference-image latents used for image-to-video conditioning.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
videos (`list`):
The generated videos.
audio (`Tensor`):
The generated audio waveform.

### Stable Diffusion XL
https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/stable_diffusion/stable_diffusion_xl.md
