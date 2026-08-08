# JoyAI-Image-Edit

[JoyAI-Image](https://github.com/jd-opensource/JoyAI-Image) is a unified multimodal foundation model for image understanding, text-to-image generation, and instruction-guided image editing. It combines an 8B Multimodal Large Language Model (MLLM) with a 16B Multimodal Diffusion Transformer (MMDiT). A central principle of JoyAI-Image is the closed-loop collaboration between understanding, generation, and editing.

JoyAI-Image-Edit supports general image editing as well as spatial editing capabilities including object move, object rotation, and camera control.

| Model | Description | Download |
|:-----:|:-----------:|:--------:|
| JoyAI-Image-Edit | Instruction-guided image editing with precise and controllable spatial manipulation | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers) |

```python
import torch
from diffusers import JoyImageEditPipeline
from diffusers.utils import load_image

pipeline = JoyImageEditPipeline.from_pretrained(
    "jdopensource/JoyAI-Image-Edit-Diffusers", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")
prompt = "Add wings to the astronaut."

output = pipeline(
    image=image,
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
output.save("joyimage_edit_output.png")
```

## Spatial editing

JoyAI-Image supports three spatial editing prompt patterns: **Object Move**, **Object Rotation**, and **Camera Control**. For best results, follow the prompt templates below as closely as possible. For more information, refer to [SpatialEdit](https://github.com/EasonXiao-888/SpatialEdit).

### Object Move

Move a target object into a specified region marked by a red box in the input image.

```text
Move the <object> into the red box and finally remove the red box.
```

### Object Rotation

Rotate an object to a specific canonical view. Supported `<view>` values: `front`, `right`, `left`, `rear`, `front right`, `front left`, `rear right`, `rear left`.

```text
Rotate the <object> to show the <view> side view.
```

### Camera Control

Change the camera viewpoint while keeping the 3D scene unchanged.

```text
Move the camera.
- Camera rotation: Yaw {y_rotation}°, Pitch {p_rotation}°.
- Camera zoom: in/out/unchanged.
- Keep the 3D scene static; only change the viewpoint.
```

## JoyImageEditPipeline[[diffusers.JoyImageEditPipeline]]

#### diffusers.JoyImageEditPipeline[[diffusers.JoyImageEditPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L100)

Diffusion pipeline for image editing using the JoyImage architecture.

The pipeline encodes text and image conditioning via a Qwen3-VL text encoder, denoises latents with a 3-D
transformer, and decodes the result with a WAN VAE.

Model offloading order: text_encoder -> transformer -> vae.

__call__diffusers.JoyImageEditPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L600[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 40"}, {"name": "timesteps", "val": ": typing.List[int] = None"}, {"name": "sigmas", "val": ": typing.List[float] = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "negative_prompt", "val": ": typing.Union[str, typing.List[str], NoneType] = None"}, {"name": "num_images_per_prompt", "val": ": typing.Optional[int] = 1"}, {"name": "generator", "val": ": typing.Union[torch._C.Generator, typing.List[torch._C.Generator], NoneType] = None"}, {"name": "latents", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "prompt_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "prompt_embeds_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "negative_prompt_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "negative_prompt_embeds_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_type", "val": ": typing.Optional[str] = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": typing.List[str] = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 4096"}, {"name": "enable_denormalization", "val": ": bool = True"}]- **prompt** (*str* or *List[str]*) --
  The prompt or prompts to guide generation.
- **height** (*int*) --
  Height of the generated output in pixels.
- **width** (*int*) --
  Width of the generated output in pixels.
- **image** (*PipelineImageInput*, *optional*) --
  Reference image used for conditioning. When provided the pipeline operates in image-editing mode with
  `num_items=2`.
- **num_inference_steps** (*int*, *optional*, defaults to 40) --
  Number of denoising steps. More steps generally improve quality at the cost of slower inference.
- **timesteps** (*List[int]*, *optional*) --
  Custom timesteps for the denoising process. When provided, `num_inference_steps` is inferred from the
  list length.
- **sigmas** (*List[float]*, *optional*) --
  Custom sigmas for the denoising process. Mutually exclusive with `timesteps`.
- **guidance_scale** (*float*, *optional*, defaults to 4.0) --
  Classifier-free guidance scale.
- **negative_prompt** (*str* or *List[str]*, *optional*) --
  Negative prompt(s) used to suppress undesired content.
- **num_images_per_prompt** (*int*, *optional*, defaults to 1) --
  Number of generated samples per prompt.
- **generator** (*torch.Generator* or *List[torch.Generator]*, *optional*) --
  RNG generator(s) for deterministic sampling.
- **latents** (*torch.Tensor*, *optional*) --
  Pre-generated noisy latents for the target slot. Sampled from a Gaussian distribution when not
  provided. Can be used to seed generation from a specific starting noise tensor.
- **prompt_embeds** (*torch.Tensor*, *optional*) --
  Pre-computed prompt embeddings. When provided `prompt` can be omitted.
- **prompt_embeds_mask** (*torch.Tensor*, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (*torch.Tensor*, *optional*) --
  Pre-computed negative prompt embeddings.
- **negative_prompt_embeds_mask** (*torch.Tensor*, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (*str*, *optional*, defaults to `"pil"`) --
  Output format. Pass `"latent"` to return raw latents.
- **return_dict** (*bool*, *optional*, defaults to *True*) --
  Whether to return a [JoyImageEditPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/joyimage_edit#diffusers.JoyImageEditPipelineOutput) or a plain tensor.
- **callback_on_step_end** (*Callable*, *PipelineCallback*, *MultiPipelineCallbacks*, *optional*) --
  Callback invoked at the end of each denoising step with signature `(self, step: int, timestep: int, callback_kwargs: Dict)`.
- **callback_on_step_end_tensor_inputs** (*List[str]*, *optional*, defaults to `["latents"]`) --
  Tensor keys included in `callback_kwargs` for `callback_on_step_end`.
- **max_sequence_length** (*int*, *optional*, defaults to 4096) --
  Maximum sequence length for prompt encoding.
- **enable_denormalization** (*bool*, *optional*, defaults to *True*) --
  Denormalise latents before VAE decoding.0[*~pipelines.joyimage.JoyImageEditPipelineOutput*] or *torch.Tensor*If `return_dict` is `True`, returns a pipeline output object containing the generated image(s).
Otherwise returns the image tensor directly.

Generate an edited image conditioned on a reference image and a text prompt.

Examples:
```python
>>> import torch
>>> from diffusers import JoyImageEditPipeline
>>> from diffusers.utils import load_image

>>> model_id = "jdopensource/JoyAI-Image-Edit-Diffusers"
>>> pipe = JoyImageEditPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> image = load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/astronaut.jpg")
>>> output = pipe(
...     image=image,  # pass an image for editing; omit for text-to-image generation
...     prompt="Add wings to the astronaut.",
...     num_inference_steps=40,
...     guidance_scale=4.0,
...     generator=torch.manual_seed(0),
... )
>>> output.images[0].save("joyimage_edit.png")
```

**Parameters:**

prompt (*str* or *List[str]*) : The prompt or prompts to guide generation.

height (*int*) : Height of the generated output in pixels.

width (*int*) : Width of the generated output in pixels.

image (*PipelineImageInput*, *optional*) : Reference image used for conditioning. When provided the pipeline operates in image-editing mode with `num_items=2`.

num_inference_steps (*int*, *optional*, defaults to 40) : Number of denoising steps. More steps generally improve quality at the cost of slower inference.

timesteps (*List[int]*, *optional*) : Custom timesteps for the denoising process. When provided, `num_inference_steps` is inferred from the list length.

sigmas (*List[float]*, *optional*) : Custom sigmas for the denoising process. Mutually exclusive with `timesteps`.

guidance_scale (*float*, *optional*, defaults to 4.0) : Classifier-free guidance scale.

negative_prompt (*str* or *List[str]*, *optional*) : Negative prompt(s) used to suppress undesired content.

num_images_per_prompt (*int*, *optional*, defaults to 1) : Number of generated samples per prompt.

generator (*torch.Generator* or *List[torch.Generator]*, *optional*) : RNG generator(s) for deterministic sampling.

latents (*torch.Tensor*, *optional*) : Pre-generated noisy latents for the target slot. Sampled from a Gaussian distribution when not provided. Can be used to seed generation from a specific starting noise tensor.

prompt_embeds (*torch.Tensor*, *optional*) : Pre-computed prompt embeddings. When provided `prompt` can be omitted.

prompt_embeds_mask (*torch.Tensor*, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (*torch.Tensor*, *optional*) : Pre-computed negative prompt embeddings.

negative_prompt_embeds_mask (*torch.Tensor*, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (*str*, *optional*, defaults to `"pil"`) : Output format. Pass `"latent"` to return raw latents.

return_dict (*bool*, *optional*, defaults to *True*) : Whether to return a [JoyImageEditPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/joyimage_edit#diffusers.JoyImageEditPipelineOutput) or a plain tensor.

callback_on_step_end (*Callable*, *PipelineCallback*, *MultiPipelineCallbacks*, *optional*) : Callback invoked at the end of each denoising step with signature `(self, step: int, timestep: int, callback_kwargs: Dict)`.

callback_on_step_end_tensor_inputs (*List[str]*, *optional*, defaults to `["latents"]`) : Tensor keys included in `callback_kwargs` for `callback_on_step_end`.

max_sequence_length (*int*, *optional*, defaults to 4096) : Maximum sequence length for prompt encoding.

enable_denormalization (*bool*, *optional*, defaults to *True*) : Denormalise latents before VAE decoding.

**Returns:**

`[*~pipelines.joyimage.JoyImageEditPipelineOutput*] or *torch.Tensor*`

If `return_dict` is `True`, returns a pipeline output object containing the generated image(s).
Otherwise returns the image tensor directly.
#### check_inputs[[diffusers.JoyImageEditPipeline.check_inputs]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L409)

Validate pipeline inputs before the forward pass.
#### denormalize_latents[[diffusers.JoyImageEditPipeline.denormalize_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L476)

Invert `normalize_latents` to recover the original latent scale.

**Parameters:**

latent : Normalised latent tensor.

**Returns:**

Latent tensor in the scale expected by `vae.decode`.
#### encode_prompt[[diffusers.JoyImageEditPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L364)

Encode a text prompt into embeddings (text-only path).

Pre-computed `prompt_embeds` bypass encoding entirely.

**Parameters:**

prompt : Prompt string or list of prompt strings.

device : Target device.

num_images_per_prompt : Number of outputs to generate per prompt.

prompt_embeds : Pre-computed prompt embeddings.

prompt_embeds_mask : Attention mask for pre-computed embeddings.

max_sequence_length : Maximum output sequence length.

template_type : Prompt template key (`"image"` or `"multiple_images"`).

**Returns:**

Tuple of (prompt_embeds, prompt_embeds_mask).
#### encode_prompt_multiple_images[[diffusers.JoyImageEditPipeline.encode_prompt_multiple_images]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L286)

Encode prompts that contain inline image tokens via the Qwen processor.

`&amp;lt;image>\n` placeholders in each prompt string are replaced by the Qwen vision special tokens before being
fed to the multimodal encoder.

**Parameters:**

prompt : Prompt string(s), optionally containing `&amp;lt;image>\n` tokens.

device : Target device.

num_images_per_prompt : Number of outputs to generate per prompt.

images : Pixel tensors corresponding to the inline image tokens.

prompt_embeds : Pre-computed prompt embeddings.

prompt_embeds_mask : Attention mask for pre-computed embeddings.

template_type : Must be `"multiple_images"`.

max_sequence_length : If set, truncate the output to this length (keeping the last `max_sequence_length` tokens).

**Returns:**

Tuple of (prompt_embeds, prompt_embeds_mask).
#### normalize_latents[[diffusers.JoyImageEditPipeline.normalize_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L447)

Normalise latents using per-channel statistics from the VAE config.

Uses (latent - mean) / std when the VAE exposes `latents_mean` and `latents_std`; otherwise falls back to
scaling by `scaling_factor`.

**Parameters:**

latent : Raw latent tensor from `vae.encode`.

**Returns:**

Normalised latent tensor.
#### prepare_latents[[diffusers.JoyImageEditPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_joyimage_edit.py#L502)

Prepare the initial noisy latent tensor for the denoising loop.

**Parameters:**

batch_size : Number of samples in the batch.

num_channels_latents : Latent channel dimension from the transformer config.

height : Spatial height in pixels.

width : Spatial width in pixels.

video_length : Number of frames (1 for image inference).

dtype : Floating-point dtype for the latent tensor.

device : Target device.

generator : RNG generator(s) for reproducible sampling.

latents : Optional user-provided initial noise for the target slot. When `None` random noise is sampled.

image : Optional list of PIL reference images to VAE-encode as conditioning slots.

enable_denormalization : Whether to normalise encoded reference latents.

**Returns:**

Tuple of `(latents, image_latents)` where `latents` has shape `(B, 1, C, T, H', W')` and
`image_latents` has shape `(B, N_ref, C, T, H', W')` or `None` when no reference images are given.

## JoyImageEditPipelineOutput[[diffusers.JoyImageEditPipelineOutput]]

#### diffusers.JoyImageEditPipelineOutput[[diffusers.JoyImageEditPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/joyimage/pipeline_output.py#L11)

Output class for JoyImageEdit generation pipelines.

### Prx Pixel
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/prx_pixel.md

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

# PRX Pixel

PRXPixel is a pixel-space text-to-image generation model by Photoroom. A ~7B `PRXTransformer2DModel`
denoises raw RGB images directly — no VAE is needed. The model is conditioned on a Qwen3-VL text encoder
and uses flow matching where the transformer predicts the clean image at each step (x-prediction). The
generation resolution is fed into the timestep modulation so the model is aware of the target size.

## Available models

| Model | Resolution | Description | Suggested parameters | Recommended dtype |
|:-----:|:---------:|:----------:|:----------:|:----------:|
| [`Photoroom/prxpixel-t2i`](https://huggingface.co/Photoroom/prxpixel-t2i) | 1024 | Pixel-space ~7B model with Qwen3-VL text encoder | 28 steps, cfg=5.0 | `torch.bfloat16` |

## Loading the pipeline

[PRXPixelPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/prx_pixel#diffusers.PRXPixelPipeline) requires `transformers >= 4.57` (the version that introduced `Qwen3VLTextModel`). Load it with [from_pretrained()](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained):

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("prxpixel_output.png")
```

## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PRXPixelPipeline[[diffusers.PRXPixelPipeline]]

#### diffusers.PRXPixelPipeline[[diffusers.PRXPixelPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L98)

Pipeline for text-to-image generation with the PRXPixel model.

PRXPixel is a standalone, pixel-space text-to-image pipeline. It denoises raw RGB directly with a ~7B-parameter
`PRXTransformer2DModel` and has no VAE (generation happens entirely in pixel space, so the denoised output *is*
the image). Prompts are encoded with a Qwen3-VL text encoder (the vision tower is discarded). Unlike
[PRXPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/prx#diffusers.PRXPipeline) the transformer is trained with x-prediction: at every step it predicts the clean image `x0`, which
is converted to a flow-matching velocity before the scheduler step. Sampling starts from `randn * noise_scale`
(`noise_scale=2.0` by default) and the default resolution is 1024px.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

Examples:
```py
>>> import torch
>>> from diffusers import PRXPixelPipeline

>>> pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
>>> image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
>>> image.save("prxpixel_output.png")
```

__call__diffusers.PRXPixelPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L406[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str = ''"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 28"}, {"name": "timesteps", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.BoolTensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.BoolTensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "use_resolution_binning", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`
  instead.
- **negative_prompt** (`str`, *optional*, defaults to `""`) --
  The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
  if `guidance_scale` is less than `1`).
- **height** (`int`, *optional*, defaults to `default_sample_size`) --
  The height in pixels of the generated image.
- **width** (`int`, *optional*, defaults to `default_sample_size`) --
  The width in pixels of the generated image.
- **num_inference_steps** (`int`, *optional*, defaults to 28) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **guidance_scale** (`float`, *optional*, defaults to 4.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided and `guidance_scale > 1`, negative embeddings will be generated from an
  empty string.
- **prompt_attention_mask** (`torch.BoolTensor`, *optional*) --
  Pre-generated attention mask for `prompt_embeds`. If not provided, attention mask will be generated
  from `prompt` input argument.
- **negative_prompt_attention_mask** (`torch.BoolTensor`, *optional*) --
  Pre-generated attention mask for `negative_prompt_embeds`. If not provided and `guidance_scale > 1`,
  attention mask will be generated from an empty string.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a [PRXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/prx#diffusers.pipelines.prx.PRXPipelineOutput) instead of a plain tuple.
- **use_resolution_binning** (`bool`, *optional*, defaults to `True`) --
  If set to `True`, the requested height and width are first mapped to the closest resolutions using
  predefined aspect ratio bins. After the produced latents are decoded into images, they are resized back
  to the requested resolution. Useful for generating non-square images at optimal resolutions.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self, step, timestep, callback_kwargs)`.
  `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include tensors that are listed
  in the `._callback_tensor_inputs` attribute.0[PRXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/prx#diffusers.pipelines.prx.PRXPipelineOutput) or `tuple`[PRXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/prx#diffusers.pipelines.prx.PRXPipelineOutput) if `return_dict` is
True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import PRXPixelPipeline

>>> pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
>>> image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
>>> image.save("prxpixel_output.png")
```

**Parameters:**

transformer (`PRXTransformer2DModel`) : The ~7B-parameter PRX denoiser. For PRXPixel this is built with `in_channels=3`, a bottleneck `img_in`, and `resolution_embeds=True`, and it is trained to predict the clean image `x0`.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : Flow-matching scheduler used to denoise the (pixel-space) latents.

text_encoder (`PreTrainedModel`) : The Qwen3-VL text backbone used to encode prompts (the vision tower is discarded). Must return a `last_hidden_state`.

tokenizer (`PreTrainedTokenizerBase`) : Tokenizer for `text_encoder` (typically loaded via `AutoTokenizer`).

default_sample_size (`int`, *optional*, defaults to 1024) : Default height/width used when none is provided to `__call__`.

prompt_max_tokens (`int`, *optional*, defaults to 256) : Number of text tokens the prompt is padded/truncated to before encoding.

noise_scale (`float`, *optional*, defaults to 2.0) : Scale applied to the initial Gaussian noise. PRXPixel trains with a non-unit initial-noise scale, so sampling must start from `randn * noise_scale`.

**Returns:**

`[PRXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/prx#diffusers.pipelines.prx.PRXPipelineOutput) or `tuple``

[PRXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/prx#diffusers.pipelines.prx.PRXPipelineOutput) if `return_dict` is
True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
#### check_inputs[[diffusers.PRXPixelPipeline.check_inputs]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L324)

Check that all inputs are in correct format.
#### encode_prompt[[diffusers.PRXPixelPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L263)

Encode text prompt using standard text encoder and tokenizer, or use precomputed embeddings.
#### prepare_latents[[diffusers.PRXPixelPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L378)

Prepare initial latents for the diffusion process.

PRXPixel trains with a non-unit initial-noise scale, so the sampled noise is multiplied by
`self.config.noise_scale`.

## PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

#### diffusers.pipelines.prx.PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_output.py#L24)

Output class for PRX pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Z-Image
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/z_image.md
