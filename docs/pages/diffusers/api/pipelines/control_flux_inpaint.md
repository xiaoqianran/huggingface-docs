# FluxControlInpaint

  

FluxControlInpaintPipeline is an implementation of Inpainting for Flux.1 Depth/Canny models. It is a pipeline that allows you to inpaint images using the Flux.1 Depth/Canny models. The pipeline takes an image and a mask as input and returns the inpainted image.

FLUX.1 Depth and Canny [dev] is a 12 billion parameter rectified flow transformer capable of generating an image based on a text description while following the structure of a given input image. **This is not a ControlNet model**.

| Control type | Developer | Link |
| -------- | ---------- | ---- |
| Depth | [Black Forest Labs](https://huggingface.co/black-forest-labs) | [Link](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) |
| Canny | [Black Forest Labs](https://huggingface.co/black-forest-labs) | [Link](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) |

> [!TIP]
> Flux can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. Additionally, Flux can benefit from quantization for memory efficiency with a trade-off in inference latency. Refer to [this blog post](https://huggingface.co/blog/quanto-diffusers) to learn more. For an exhaustive list of resources, check out [this gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c).

```python
import torch
from diffusers import FluxControlInpaintPipeline
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import T5EncoderModel
from diffusers.utils import load_image, make_image_grid
from image_gen_aux import DepthPreprocessor # https://github.com/huggingface/image_gen_aux
from PIL import Image
import numpy as np

pipe = FluxControlInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    dtype=torch.bfloat16,
)
# use following lines if you have GPU constraints
# ---------------------------------------------------------------
transformer = FluxTransformer2DModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="transformer", dtype=torch.bfloat16
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="text_encoder_2", dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
pipe.enable_model_cpu_offload()
# ---------------------------------------------------------------
pipe.to("cuda")

prompt = "a blue robot singing opera with human-like expressions"
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

head_mask = np.zeros_like(image)
head_mask[65:580,300:642] = 255
mask_image = Image.fromarray(head_mask)

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(image)[0].convert("RGB")

output = pipe(
    prompt=prompt,
    image=image,
    control_image=control_image,
    mask_image=mask_image,
    num_inference_steps=30,
    strength=0.9,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
make_image_grid([image, control_image, mask_image, output.resize(image.size)], rows=1, cols=4).save("output.png")
```

## FluxControlInpaintPipeline[[diffusers.FluxControlInpaintPipeline]]

#### diffusers.FluxControlInpaintPipeline[[diffusers.FluxControlInpaintPipeline]]

```python
diffusers.FluxControlInpaintPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: T5Tokenizer, transformer: FluxTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux/pipeline_flux_control_inpaint.py#L204)

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.40.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

The Flux pipeline for image inpainting using Flux-dev-Depth/Canny.

Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

#### __call__[[diffusers.FluxControlInpaintPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, prompt_2: str | list[str] | None = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, control_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, mask_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, masked_image_latents: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, height: int | None = None, width: int | None = None, strength: float = 0.6, num_inference_steps: int = 28, sigmas: list[float] | None = None, guidance_scale: float = 7.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux/pipeline_flux_control_inpaint.py#L751)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, `list[np.ndarray]`, : `list[list[torch.Tensor]]`, `list[list[np.ndarray]]` or `list[list[PIL.Image.Image]]`): The ControlNet input condition to provide guidance to the `unet` for generation. If the type is specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`, images must be passed as a list such that each element of the list can be correctly batched for input to a single ControlNet.

mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B, H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W, 1)`, or `(H, W)`.

masked_image_latents (`torch.Tensor`, `list[torch.Tensor]`) : `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask latents tensor will be generated by `mask_image`.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

strength (`float`, *optional*, defaults to 1.0) : Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 1 essentially ignores `image`.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 7.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

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

**Returns:** `~pipelines.flux.FluxPipelineOutput` or `tuple`

`~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.

Function invoked when calling the pipeline for generation.

Examples:
```py
import torch
from diffusers import FluxControlInpaintPipeline
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import T5EncoderModel
from diffusers.utils import load_image, make_image_grid
from image_gen_aux import DepthPreprocessor  # https://github.com/huggingface/image_gen_aux
from PIL import Image
import numpy as np

pipe = FluxControlInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    torch_dtype=torch.bfloat16,
)
# use following lines if you have GPU constraints
# ---------------------------------------------------------------
transformer = FluxTransformer2DModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
pipe.enable_model_cpu_offload()
# ---------------------------------------------------------------
pipe.to("cuda")

prompt = "a blue robot singing opera with human-like expressions"
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

head_mask = np.zeros_like(image)
head_mask[65:580, 300:642] = 255
mask_image = Image.fromarray(head_mask)

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(image)[0].convert("RGB")

output = pipe(
    prompt=prompt,
    image=image,
    control_image=control_image,
    mask_image=mask_image,
    num_inference_steps=30,
    strength=0.9,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
make_image_grid([image, control_image, mask_image, output.resize(image.size)], rows=1, cols=4).save(
    "output.png"
)
```

#### encode_prompt[[diffusers.FluxControlInpaintPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], prompt_2: str | list[str] | None = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, max_sequence_length: int = 512, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux/pipeline_flux_control_inpaint.py#L374)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is used in all text-encoders

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

lora_scale (`float`, *optional*) : A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.

## FluxPipelineOutput[[diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput]]

#### diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput[[diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput]]

```python
diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux/pipeline_output.py#L11)

**Parameters:**

images (`list[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be passed to the decoder.

Output class for Flux image generation pipelines.

### Prx Pixel
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/prx_pixel.md

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

[PRXPixelPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/prx_pixel#diffusers.PRXPixelPipeline) requires `transformers >= 4.57` (the version that introduced `Qwen3VLTextModel`). Load it with [from_pretrained()](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained):

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", dtype=torch.bfloat16)
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

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PRXPixelPipeline[[diffusers.PRXPixelPipeline]]

#### diffusers.PRXPixelPipeline[[diffusers.PRXPixelPipeline]]

```python
diffusers.PRXPixelPipeline(transformer: PRXTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder: PreTrainedModel, tokenizer: transformers.models.auto.tokenization_auto.AutoTokenizer | transformers.tokenization_utils_base.PreTrainedTokenizerBase, default_sample_size: int | None = 1024, prompt_max_tokens: int = 256, noise_scale: float = 2.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L98)

**Parameters:**

transformer (`PRXTransformer2DModel`) : The ~7B-parameter PRX denoiser. For PRXPixel this is built with `in_channels=3`, a bottleneck `img_in`, and `resolution_embeds=True`, and it is trained to predict the clean image `x0`.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : Flow-matching scheduler used to denoise the (pixel-space) latents.

text_encoder (`PreTrainedModel`) : The Qwen3-VL text backbone used to encode prompts (the vision tower is discarded). Must return a `last_hidden_state`.

tokenizer (`PreTrainedTokenizerBase`) : Tokenizer for `text_encoder` (typically loaded via `AutoTokenizer`).

default_sample_size (`int`, *optional*, defaults to 1024) : Default height/width used when none is provided to `__call__`.

prompt_max_tokens (`int`, *optional*, defaults to 256) : Number of text tokens the prompt is padded/truncated to before encoding.

noise_scale (`float`, *optional*, defaults to 2.0) : Scale applied to the initial Gaussian noise. PRXPixel trains with a non-unit initial-noise scale, so sampling must start from `randn * noise_scale`.

Pipeline for text-to-image generation with the PRXPixel model.

PRXPixel is a standalone, pixel-space text-to-image pipeline. It denoises raw RGB directly with a ~7B-parameter
`PRXTransformer2DModel` and has no VAE (generation happens entirely in pixel space, so the denoised output *is*
the image). Prompts are encoded with a Qwen3-VL text encoder (the vision tower is discarded). Unlike
[PRXPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/prx#diffusers.PRXPipeline) the transformer is trained with x-prediction: at every step it predicts the clean image `x0`, which
is converted to a flow-matching velocity before the scheduler step. Sampling starts from `randn * noise_scale`
(`noise_scale=2.0` by default) and the default resolution is 1024px.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
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

#### __call__[[diffusers.PRXPixelPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str = '', height: int | None = None, width: int | None = None, num_inference_steps: int = 28, timesteps: list = None, guidance_scale: float = 4.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, negative_prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, use_resolution_binning: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L406)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead.

negative_prompt (`str`, *optional*, defaults to `""`) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, *optional*, defaults to `default_sample_size`) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to `default_sample_size`) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 28) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided and `guidance_scale > 1`, negative embeddings will be generated from an empty string.

prompt_attention_mask (`torch.BoolTensor`, *optional*) : Pre-generated attention mask for `prompt_embeds`. If not provided, attention mask will be generated from `prompt` input argument.

negative_prompt_attention_mask (`torch.BoolTensor`, *optional*) : Pre-generated attention mask for `negative_prompt_embeds`. If not provided and `guidance_scale > 1`, attention mask will be generated from an empty string.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [PRXPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/prx_pixel#diffusers.pipelines.prx.PRXPipelineOutput) instead of a plain tuple.

use_resolution_binning (`bool`, *optional*, defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using predefined aspect ratio bins. After the produced latents are decoded into images, they are resized back to the requested resolution. Useful for generating non-square images at optimal resolutions.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self, step, timestep, callback_kwargs)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include tensors that are listed in the `._callback_tensor_inputs` attribute.

**Returns:** [PRXPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/prx_pixel#diffusers.pipelines.prx.PRXPipelineOutput) or `tuple`

[PRXPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/prx_pixel#diffusers.pipelines.prx.PRXPipelineOutput) if `return_dict` is
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

#### check_inputs[[diffusers.PRXPixelPipeline.check_inputs]]

```python
check_inputs(prompt: str | list[str], height: int, width: int, guidance_scale: float, callback_on_step_end_tensor_inputs: list[str] | None = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L324)

Check that all inputs are in correct format.

#### encode_prompt[[diffusers.PRXPixelPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, do_classifier_free_guidance: bool = True, negative_prompt: str = '', num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, negative_prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, tokenizer_max_length: int | None = None, skip_text_cleaning: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L263)

Encode text prompt using standard text encoder and tokenizer, or use precomputed embeddings.

#### prepare_latents[[diffusers.PRXPixelPipeline.prepare_latents]]

```python
prepare_latents(batch_size: int, num_channels_latents: int, height: int, width: int, dtype: dtype, device: device, generator: typing.Optional[torch.Generator] = None, latents: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx_pixel.py#L378)

Prepare initial latents for the diffusion process.

PRXPixel trains with a non-unit initial-noise scale, so the sampled noise is multiplied by
`self.config.noise_scale`.

## PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

#### diffusers.pipelines.prx.PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

```python
diffusers.pipelines.prx.PRXPipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_output.py#L24)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for PRX pipelines.

### Lumina2
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/lumina2.md

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
