# Kandinsky 2.2

Kandinsky 2.2 is created by [Arseniy Shakhmatov](https://github.com/cene555), [Anton Razzhigaev](https://github.com/razzant), [Aleksandr Nikolich](https://github.com/AlexWortega), [Vladimir Arkhipkin](https://github.com/oriBetelgeuse), [Igor Pavlov](https://github.com/boomb0om), [Andrey Kuznetsov](https://github.com/kuznetsoffandrey), and [Denis Dimitrov](https://github.com/denndimitrov).

The description from it's GitHub page is:

*Kandinsky 2.2 brings substantial improvements upon its predecessor, Kandinsky 2.1, by introducing a new, more powerful image encoder - CLIP-ViT-G and the ControlNet support. The switch to CLIP-ViT-G as the image encoder significantly increases the model's capability to generate more aesthetic pictures and better understand text, thus enhancing the model's overall performance. The addition of the ControlNet mechanism allows the model to effectively control the process of generating images. This leads to more accurate and visually appealing outputs and opens new possibilities for text-guided image manipulation.*

The original codebase can be found at [ai-forever/Kandinsky-2](https://github.com/ai-forever/Kandinsky-2).

> [!TIP]
> Check out the [Kandinsky Community](https://huggingface.co/kandinsky-community) organization on the Hub for the official model checkpoints for tasks like text-to-image, image-to-image, and inpainting.

> [!TIP]
> Make sure to check out the schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## KandinskyV22PriorPipeline[[diffusers.KandinskyV22PriorPipeline]]

#### diffusers.KandinskyV22PriorPipeline[[diffusers.KandinskyV22PriorPipeline]]

```python
diffusers.KandinskyV22PriorPipeline(prior: PriorTransformer, image_encoder: CLIPVisionModelWithProjection, text_encoder: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, scheduler: UnCLIPScheduler, image_processor: CLIPImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py#L89)

**Parameters:**

prior ([PriorTransformer](/docs/diffusers/v0.40.0/en/api/models/prior_transformer#diffusers.PriorTransformer)) : The canonical unCLIP prior to approximate the image embedding from the text embedding.

image_encoder (`CLIPVisionModelWithProjection`) : Frozen image-encoder.

text_encoder (`CLIPTextModelWithProjection`) : Frozen text-encoder.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

scheduler (`UnCLIPScheduler`) : A scheduler to be used in combination with `prior` to generate image embedding.

image_processor (`CLIPImageProcessor`) : A image_processor to be used to preprocess image from clip.

Pipeline for generating image prior for Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22PriorPipeline.__call__]]

```python
__call__(prompt: str | list[str], negative_prompt: str | list[str] | None = None, num_images_per_prompt: int = 1, num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, guidance_scale: float = 4.0, output_type: str | None = 'pt', return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py#L375)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

output_type (`str`, *optional*, defaults to `"pt"`) : The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** `KandinskyPriorPipelineOutput` or `tuple`

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
>>> import torch

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
>>> pipe_prior.to("cuda")
>>> prompt = "red cat, 4k photo"
>>> image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()

>>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
>>> pipe.to("cuda")
>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=negative_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=50,
... ).images
>>> image[0].save("cat.png")
```

#### interpolate[[diffusers.KandinskyV22PriorPipeline.interpolate]]

```python
interpolate(images_and_prompts: list, weights: list, num_images_per_prompt: int = 1, num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, negative_prior_prompt: str | None = None, negative_prompt: str = '', guidance_scale: float = 4.0, device = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior.py#L136)

**Parameters:**

images_and_prompts (`list[str | PIL.Image.Image | torch.Tensor]`) : list of prompts and images to guide the image generation.

weights : (`list[float]`): list of weights for each condition in `images_and_prompts`

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

negative_prior_prompt (`str`, *optional*) : The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

**Returns:** `KandinskyPriorPipelineOutput` or `tuple`

Function invoked when using the prior pipeline for interpolation.

Examples:
```py
>>> from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
>>> from diffusers.utils import load_image
>>> import PIL
>>> import torch
>>> from torchvision import transforms

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
... )
>>> pipe_prior.to("cuda")
>>> img1 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... )
>>> img2 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/starry_night.jpeg"
... )
>>> images_texts = ["a cat", img1, img2]
>>> weights = [0.3, 0.3, 0.4]
>>> out = pipe_prior.interpolate(images_texts, weights)
>>> pipe = KandinskyV22Pipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
... )
>>> pipe.to("cuda")
>>> image = pipe(
...     image_embeds=out.image_embeds,
...     negative_image_embeds=out.negative_image_embeds,
...     height=768,
...     width=768,
...     num_inference_steps=50,
... ).images[0]
>>> image.save("starry_cat.png")
```

## KandinskyV22Pipeline[[diffusers.KandinskyV22Pipeline]]

#### diffusers.KandinskyV22Pipeline[[diffusers.KandinskyV22Pipeline]]

```python
diffusers.KandinskyV22Pipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2.py#L72)

**Parameters:**

scheduler (`DDIMScheduler` | `DDPMScheduler`) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

Pipeline for text-to-image generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22Pipeline.__call__]]

```python
__call__(image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], negative_image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], height: int = 512, width: int = 512, num_inference_steps: int = 100, guidance_scale: float = 4.0, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2.py#L130)

**Parameters:**

image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for text prompt, that will be used to condition the image generation.

negative_image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for negative text prompt, will be used to condition the image generation.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
>>> import torch

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
>>> pipe_prior.to("cuda")
>>> prompt = "red cat, 4k photo"
>>> out = pipe_prior(prompt)
>>> image_emb = out.image_embeds
>>> zero_image_emb = out.negative_image_embeds
>>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
>>> pipe.to("cuda")
>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=zero_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=50,
... ).images
>>> image[0].save("cat.png")
```

## KandinskyV22CombinedPipeline[[diffusers.KandinskyV22CombinedPipeline]]

#### diffusers.KandinskyV22CombinedPipeline[[diffusers.KandinskyV22CombinedPipeline]]

```python
diffusers.KandinskyV22CombinedPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel, prior_prior: PriorTransformer, prior_image_encoder: CLIPVisionModelWithProjection, prior_text_encoder: CLIPTextModelWithProjection, prior_tokenizer: CLIPTokenizer, prior_scheduler: UnCLIPScheduler, prior_image_processor: CLIPImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L107)

**Parameters:**

scheduler (`DDIMScheduler` | `DDPMScheduler`) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

prior_prior ([PriorTransformer](/docs/diffusers/v0.40.0/en/api/models/prior_transformer#diffusers.PriorTransformer)) : The canonical unCLIP prior to approximate the image embedding from the text embedding.

prior_image_encoder (`CLIPVisionModelWithProjection`) : Frozen image-encoder.

prior_text_encoder (`CLIPTextModelWithProjection`) : Frozen text-encoder.

prior_tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

prior_scheduler (`UnCLIPScheduler`) : A scheduler to be used in combination with `prior` to generate image embedding.

prior_image_processor (`CLIPImageProcessor`) : A image_processor to be used to preprocess image from clip.

Combined Pipeline for text-to-image generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22CombinedPipeline.__call__]]

```python
__call__(prompt: str | list[str], negative_prompt: str | list[str] | None = None, num_inference_steps: int = 100, guidance_scale: float = 4.0, num_images_per_prompt: int = 1, height: int = 512, width: int = 512, prior_guidance_scale: float = 4.0, prior_num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', callback: typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None, callback_steps: int = 1, return_dict: bool = True, prior_callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, prior_callback_on_step_end_tensor_inputs: list = ['latents'], callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L202)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

prior_guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

prior_num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

callback (`Callable`, *optional*) : A function that calls every `callback_steps` steps during inference. The function is called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.

callback_steps (`int`, *optional*, defaults to 1) : The frequency at which the `callback` function is called. If not specified, the callback is called at every step.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

prior_callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference of the prior pipeline. The function is called with the following arguments: `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.

prior_callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `prior_callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your prior pipeline class.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference of the decoder pipeline. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:
```py
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

image = pipe(prompt=prompt, num_inference_steps=25).images[0]
```

#### enable_sequential_cpu_offload[[diffusers.KandinskyV22CombinedPipeline.enable_sequential_cpu_offload]]

```python
enable_sequential_cpu_offload(gpu_id: int | None = None, device: typing.Union[torch.device, str] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L182)

Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
`torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
Note that offloading happens on a submodule basis. Memory savings are higher than with
`enable_model_cpu_offload`, but performance is lower.

## KandinskyV22ControlnetPipeline[[diffusers.KandinskyV22ControlnetPipeline]]

#### diffusers.KandinskyV22ControlnetPipeline[[diffusers.KandinskyV22ControlnetPipeline]]

```python
diffusers.KandinskyV22ControlnetPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py#L115)

**Parameters:**

scheduler ([DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

Pipeline for text-to-image generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22ControlnetPipeline.__call__]]

```python
__call__(image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], negative_image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], hint: Tensor, height: int = 512, width: int = 512, num_inference_steps: int = 100, guidance_scale: float = 4.0, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', callback: typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None, callback_steps: int = 1, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet.py#L160)

**Parameters:**

hint (`torch.Tensor`) : The controlnet condition.

image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for text prompt, that will be used to condition the image generation.

negative_image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for negative text prompt, will be used to condition the image generation.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

callback (`Callable`, *optional*) : A function that calls every `callback_steps` steps during inference. The function is called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.

callback_steps (`int`, *optional*, defaults to 1) : The frequency at which the `callback` function is called. If not specified, the callback is called at every step.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:

## KandinskyV22PriorEmb2EmbPipeline[[diffusers.KandinskyV22PriorEmb2EmbPipeline]]

#### diffusers.KandinskyV22PriorEmb2EmbPipeline[[diffusers.KandinskyV22PriorEmb2EmbPipeline]]

```python
diffusers.KandinskyV22PriorEmb2EmbPipeline(prior: PriorTransformer, image_encoder: CLIPVisionModelWithProjection, text_encoder: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, scheduler: UnCLIPScheduler, image_processor: CLIPImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py#L105)

**Parameters:**

prior ([PriorTransformer](/docs/diffusers/v0.40.0/en/api/models/prior_transformer#diffusers.PriorTransformer)) : The canonical unCLIP prior to approximate the image embedding from the text embedding.

image_encoder (`CLIPVisionModelWithProjection`) : Frozen image-encoder.

text_encoder (`CLIPTextModelWithProjection`) : Frozen text-encoder.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

scheduler (`UnCLIPScheduler`) : A scheduler to be used in combination with `prior` to generate image embedding.

Pipeline for generating image prior for Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22PriorEmb2EmbPipeline.__call__]]

```python
__call__(prompt: str | list[str], image: typing.Union[torch.Tensor, list[torch.Tensor], PIL.Image.Image, list[PIL.Image.Image]], strength: float = 0.3, negative_prompt: str | list[str] | None = None, num_images_per_prompt: int = 1, num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, guidance_scale: float = 4.0, output_type: str | None = 'pt', return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py#L399)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

image (`torch.Tensor`, `PIL.Image.Image`, `list[torch.Tensor]` or `list[PIL.Image.Image]`) : `Image`, or tensor representing an image batch, that will be used as the starting point for the image embedding. Can also accept image latents as `image`, if passing latents directly, it will not be encoded again.

strength (`float`, *optional*, defaults to 0.8) : Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

output_type (`str`, *optional*, defaults to `"pt"`) : The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** `KandinskyPriorPipelineOutput` or `tuple`

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorEmb2EmbPipeline
>>> import torch

>>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
... )
>>> pipe_prior.to("cuda")

>>> prompt = "red cat, 4k photo"
>>> img = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... )
>>> image_emb, nagative_image_emb = pipe_prior(prompt, image=img, strength=0.2).to_tuple()

>>> pipe = KandinskyPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-decoder, torch_dtype=torch.float16"
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=negative_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=100,
... ).images

>>> image[0].save("cat.png")
```

#### interpolate[[diffusers.KandinskyV22PriorEmb2EmbPipeline.interpolate]]

```python
interpolate(images_and_prompts: list, weights: list, num_images_per_prompt: int = 1, num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, negative_prior_prompt: str | None = None, negative_prompt: str = '', guidance_scale: float = 4.0, device = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_prior_emb2emb.py#L158)

**Parameters:**

images_and_prompts (`list[str | PIL.Image.Image | torch.Tensor]`) : list of prompts and images to guide the image generation.

weights : (`list[float]`): list of weights for each condition in `images_and_prompts`

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

negative_prior_prompt (`str`, *optional*) : The prompt not to guide the prior diffusion process. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

**Returns:** `KandinskyPriorPipelineOutput` or `tuple`

Function invoked when using the prior pipeline for interpolation.

Examples:
```py
>>> from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22Pipeline
>>> from diffusers.utils import load_image
>>> import PIL

>>> import torch
>>> from torchvision import transforms

>>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
... )
>>> pipe_prior.to("cuda")

>>> img1 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/cat.png"
... )

>>> img2 = load_image(
...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
...     "/kandinsky/starry_night.jpeg"
... )

>>> images_texts = ["a cat", img1, img2]
>>> weights = [0.3, 0.3, 0.4]
>>> image_emb, zero_image_emb = pipe_prior.interpolate(images_texts, weights)

>>> pipe = KandinskyV22Pipeline.from_pretrained(
...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
... )
>>> pipe.to("cuda")

>>> image = pipe(
...     image_embeds=image_emb,
...     negative_image_embeds=zero_image_emb,
...     height=768,
...     width=768,
...     num_inference_steps=150,
... ).images[0]

>>> image.save("starry_cat.png")
```

## KandinskyV22Img2ImgPipeline[[diffusers.KandinskyV22Img2ImgPipeline]]

#### diffusers.KandinskyV22Img2ImgPipeline[[diffusers.KandinskyV22Img2ImgPipeline]]

```python
diffusers.KandinskyV22Img2ImgPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py#L78)

**Parameters:**

scheduler ([DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

Pipeline for image-to-image generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22Img2ImgPipeline.__call__]]

```python
__call__(image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], image: typing.Union[torch.Tensor, list[torch.Tensor], PIL.Image.Image, list[PIL.Image.Image]], negative_image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], height: int = 512, width: int = 512, num_inference_steps: int = 100, guidance_scale: float = 4.0, strength: float = 0.3, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str | None = 'pil', return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py#L183)

**Parameters:**

image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for text prompt, that will be used to condition the image generation.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, or tensor representing an image batch, that will be used as the starting point for the process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded again.

strength (`float`, *optional*, defaults to 0.8) : Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.

negative_image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for negative text prompt, will be used to condition the image generation.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:

## KandinskyV22Img2ImgCombinedPipeline[[diffusers.KandinskyV22Img2ImgCombinedPipeline]]

#### diffusers.KandinskyV22Img2ImgCombinedPipeline[[diffusers.KandinskyV22Img2ImgCombinedPipeline]]

```python
diffusers.KandinskyV22Img2ImgCombinedPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel, prior_prior: PriorTransformer, prior_image_encoder: CLIPVisionModelWithProjection, prior_text_encoder: CLIPTextModelWithProjection, prior_tokenizer: CLIPTokenizer, prior_scheduler: UnCLIPScheduler, prior_image_processor: CLIPImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L341)

**Parameters:**

scheduler (`DDIMScheduler` | `DDPMScheduler`) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

prior_prior ([PriorTransformer](/docs/diffusers/v0.40.0/en/api/models/prior_transformer#diffusers.PriorTransformer)) : The canonical unCLIP prior to approximate the image embedding from the text embedding.

prior_image_encoder (`CLIPVisionModelWithProjection`) : Frozen image-encoder.

prior_text_encoder (`CLIPTextModelWithProjection`) : Frozen text-encoder.

prior_tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

prior_scheduler (`UnCLIPScheduler`) : A scheduler to be used in combination with `prior` to generate image embedding.

prior_image_processor (`CLIPImageProcessor`) : A image_processor to be used to preprocess image from clip.

Combined Pipeline for image-to-image generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22Img2ImgCombinedPipeline.__call__]]

```python
__call__(prompt: str | list[str], image: typing.Union[torch.Tensor, list[torch.Tensor], PIL.Image.Image, list[PIL.Image.Image]], negative_prompt: str | list[str] | None = None, num_inference_steps: int = 100, guidance_scale: float = 4.0, strength: float = 0.3, num_images_per_prompt: int = 1, height: int = 512, width: int = 512, prior_guidance_scale: float = 4.0, prior_num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', callback: typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None, callback_steps: int = 1, return_dict: bool = True, prior_callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, prior_callback_on_step_end_tensor_inputs: list = ['latents'], callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L446)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, or tensor representing an image batch, that will be used as the starting point for the process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded again.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

strength (`float`, *optional*, defaults to 0.3) : Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

prior_guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

prior_num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

callback (`Callable`, *optional*) : A function that calls every `callback_steps` steps during inference. The function is called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.

callback_steps (`int`, *optional*, defaults to 1) : The frequency at which the `callback` function is called. If not specified, the callback is called at every step.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

prior_callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference of the prior pipeline. The function is called with the following arguments: `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.

prior_callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `prior_callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your prior pipeline class.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference of the decoder pipeline. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:
```py
from diffusers import AutoPipelineForImage2Image
import torch
import requests
from io import BytesIO
from PIL import Image
import os

pipe = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A fantasy landscape, Cinematic lighting"
negative_prompt = "low quality, bad quality"

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.thumbnail((768, 768))

image = pipe(prompt=prompt, image=original_image, num_inference_steps=25).images[0]
```

#### enable_model_cpu_offload[[diffusers.KandinskyV22Img2ImgCombinedPipeline.enable_model_cpu_offload]]

```python
enable_model_cpu_offload(gpu_id: int | None = None, device: typing.Union[torch.device, str] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L416)

Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
`enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

#### enable_sequential_cpu_offload[[diffusers.KandinskyV22Img2ImgCombinedPipeline.enable_sequential_cpu_offload]]

```python
enable_sequential_cpu_offload(gpu_id: int | None = None, device: typing.Union[torch.device, str] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L426)

Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
`torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
Note that offloading happens on a submodule basis. Memory savings are higher than with
`enable_model_cpu_offload`, but performance is lower.

## KandinskyV22ControlnetImg2ImgPipeline[[diffusers.KandinskyV22ControlnetImg2ImgPipeline]]

#### diffusers.KandinskyV22ControlnetImg2ImgPipeline[[diffusers.KandinskyV22ControlnetImg2ImgPipeline]]

```python
diffusers.KandinskyV22ControlnetImg2ImgPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py#L107)

**Parameters:**

scheduler ([DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

Pipeline for image-to-image generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22ControlnetImg2ImgPipeline.__call__]]

```python
__call__(image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], image: typing.Union[torch.Tensor, list[torch.Tensor], PIL.Image.Image, list[PIL.Image.Image]], negative_image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], hint: Tensor, height: int = 512, width: int = 512, num_inference_steps: int = 100, guidance_scale: float = 4.0, strength: float = 0.3, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str | None = 'pil', callback: typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None, callback_steps: int = 1, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py#L200)

**Parameters:**

image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for text prompt, that will be used to condition the image generation.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, or tensor representing an image batch, that will be used as the starting point for the process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded again.

strength (`float`, *optional*, defaults to 0.8) : Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.

hint (`torch.Tensor`) : The controlnet condition.

negative_image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for negative text prompt, will be used to condition the image generation.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

callback (`Callable`, *optional*) : A function that calls every `callback_steps` steps during inference. The function is called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.

callback_steps (`int`, *optional*, defaults to 1) : The frequency at which the `callback` function is called. If not specified, the callback is called at every step.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:

## KandinskyV22InpaintPipeline[[diffusers.KandinskyV22InpaintPipeline]]

#### diffusers.KandinskyV22InpaintPipeline[[diffusers.KandinskyV22InpaintPipeline]]

```python
diffusers.KandinskyV22InpaintPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py#L243)

**Parameters:**

scheduler ([DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler)) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

Pipeline for text-guided image inpainting using Kandinsky2.1

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22InpaintPipeline.__call__]]

```python
__call__(image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], image: typing.Union[torch.Tensor, PIL.Image.Image], mask_image: typing.Union[PIL.Image.Image, torch.Tensor, numpy.ndarray], negative_image_embeds: typing.Union[torch.Tensor, list[torch.Tensor]], height: int = 512, width: int = 512, num_inference_steps: int = 100, guidance_scale: float = 4.0, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py#L302)

**Parameters:**

image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for text prompt, that will be used to condition the image generation.

image (`PIL.Image.Image`) : `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will be masked out with `mask_image` and repainted according to `prompt`.

mask_image (`np.array`) : Tensor representing an image batch, to mask `image`. White pixels in the mask will be repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.

negative_image_embeds (`torch.Tensor` or `list[torch.Tensor]`) : The clip image embeddings for negative text prompt, will be used to condition the image generation.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:

## KandinskyV22InpaintCombinedPipeline[[diffusers.KandinskyV22InpaintCombinedPipeline]]

#### diffusers.KandinskyV22InpaintCombinedPipeline[[diffusers.KandinskyV22InpaintCombinedPipeline]]

```python
diffusers.KandinskyV22InpaintCombinedPipeline(unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel, prior_prior: PriorTransformer, prior_image_encoder: CLIPVisionModelWithProjection, prior_text_encoder: CLIPTextModelWithProjection, prior_tokenizer: CLIPTokenizer, prior_scheduler: UnCLIPScheduler, prior_image_processor: CLIPImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L607)

**Parameters:**

scheduler (`DDIMScheduler` | `DDPMScheduler`) : A scheduler to be used in combination with `unet` to generate image latents.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the image embedding.

movq ([VQModel](/docs/diffusers/v0.40.0/en/api/models/vq#diffusers.VQModel)) : MoVQ Decoder to generate the image from the latents.

prior_prior ([PriorTransformer](/docs/diffusers/v0.40.0/en/api/models/prior_transformer#diffusers.PriorTransformer)) : The canonical unCLIP prior to approximate the image embedding from the text embedding.

prior_image_encoder (`CLIPVisionModelWithProjection`) : Frozen image-encoder.

prior_text_encoder (`CLIPTextModelWithProjection`) : Frozen text-encoder.

prior_tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

prior_scheduler (`UnCLIPScheduler`) : A scheduler to be used in combination with `prior` to generate image embedding.

prior_image_processor (`CLIPImageProcessor`) : A image_processor to be used to preprocess image from clip.

Combined Pipeline for inpainting generation using Kandinsky

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.KandinskyV22InpaintCombinedPipeline.__call__]]

```python
__call__(prompt: str | list[str], image: typing.Union[torch.Tensor, list[torch.Tensor], PIL.Image.Image, list[PIL.Image.Image]], mask_image: typing.Union[torch.Tensor, list[torch.Tensor], PIL.Image.Image, list[PIL.Image.Image]], negative_prompt: str | list[str] | None = None, num_inference_steps: int = 100, guidance_scale: float = 4.0, num_images_per_prompt: int = 1, height: int = 512, width: int = 512, prior_guidance_scale: float = 4.0, prior_num_inference_steps: int = 25, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, prior_callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, prior_callback_on_step_end_tensor_inputs: list = ['latents'], callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L702)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, or tensor representing an image batch, that will be used as the starting point for the process. Can also accept image latents as `image`, if passing latents directly, it will not be encoded again.

mask_image (`np.array`) : Tensor representing an image batch, to mask `image`. White pixels in the mask will be repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

height (`int`, *optional*, defaults to 512) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to 512) : The width in pixels of the generated image.

prior_guidance_scale (`float`, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

prior_num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) instead of a plain tuple.

prior_callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `prior_callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.

prior_callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `prior_callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

Function invoked when calling the pipeline for generation.

Examples:
```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import numpy as np

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A fantasy landscape, Cinematic lighting"
negative_prompt = "low quality, bad quality"

original_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
)

mask = np.zeros((768, 768), dtype=np.float32)
# Let's mask out an area above the cat's head
mask[:250, 250:-250] = 1

image = pipe(prompt=prompt, image=original_image, mask_image=mask, num_inference_steps=25).images[0]
```

#### enable_sequential_cpu_offload[[diffusers.KandinskyV22InpaintCombinedPipeline.enable_sequential_cpu_offload]]

```python
enable_sequential_cpu_offload(gpu_id: int | None = None, device: typing.Union[torch.device, str] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py#L682)

Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
`torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
Note that offloading happens on a submodule basis. Memory savings are higher than with
`enable_model_cpu_offload`, but performance is lower.

### Hunyuan Video15
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/hunyuan_video15.md

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

# HunyuanVideo-1.5

HunyuanVideo-1.5 is a lightweight yet powerful video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture with selective and sliding tile attention (SSTA), enhanced bilingual understanding through glyph-aware text encoding, progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions. Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source models.

You can find all the original HunyuanVideo checkpoints under the [Tencent](https://huggingface.co/tencent) organization.

> [!TIP]
> Click on the HunyuanVideo models in the right sidebar for more examples of video generation tasks.
>
> The examples below use a checkpoint from [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) because the weights are stored in a layout compatible with Diffusers.

The example below demonstrates how to generate a video optimized for memory or inference speed.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

```py
import torch
from diffusers import AutoModel, HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

pipeline = HunyuanVideo15Pipeline.from_pretrained(
    "HunyuanVideo-1.5-Diffusers-480p_t2v",
    dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

## Notes

- HunyuanVideo1.5 use attention masks with variable-length sequences. For best performance, we recommend using an attention backend that handles padding efficiently.

    - **H100/H800:** `_flash_3_hub` or `_flash_3_varlen_hub`
    - **A100/A800/RTX 4090:** `flash_hub` or `flash_varlen_hub`
    - **Other GPUs:** `sage_hub`

Refer to the [Attention backends](../../optimization/attention_backends) guide for more details about using a different backend.

```py
pipe.transformer.set_attention_backend("flash_hub")  # or your preferred backend
```

- [HunyuanVideo15Pipeline](/docs/diffusers/v0.40.0/en/api/pipelines/hunyuan_video15#diffusers.HunyuanVideo15Pipeline) use guider and does not take `guidance_scale` parameter at runtime. 

You can check the default guider configuration using `pipe.guider`:

```py
>>> pipe.guider 
ClassifierFreeGuidance {
  "_class_name": "ClassifierFreeGuidance",
  "_diffusers_version": "0.36.0.dev0",
  "enabled": true,
  "guidance_rescale": 0.0,
  "guidance_scale": 6.0,
  "start": 0.0,
  "stop": 1.0,
  "use_original_formulation": false
}

State:
  step: None
  num_inference_steps: None
  timestep: None
  count_prepared: 0
  enabled: True
  num_conditions: 2
```

To update guider configuration, you can run `pipe.guider = pipe.guider.new(...)`

```py
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

Read more on Guider [here](../../using-diffusers/guiders).

## HunyuanVideo15Pipeline[[diffusers.HunyuanVideo15Pipeline]]

#### diffusers.HunyuanVideo15Pipeline[[diffusers.HunyuanVideo15Pipeline]]

```python
diffusers.HunyuanVideo15Pipeline(text_encoder: Qwen2_5_VLTextModel, tokenizer: Qwen2Tokenizer, transformer: HunyuanVideo15Transformer3DModel, vae: AutoencoderKLHunyuanVideo15, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder_2: T5EncoderModel, tokenizer_2: ByT5Tokenizer, guider: ClassifierFreeGuidance)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L166)

**Parameters:**

transformer ([HunyuanVideo15Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.HunyuanVideo15Transformer3DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLHunyuanVideo15](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_hunyuan_video15#diffusers.AutoencoderKLHunyuanVideo15)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider ([ClassifierFreeGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance)) : [ClassifierFreeGuidance]for classifier free guidance.

Pipeline for text-to-video generation using HunyuanVideo1.5.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideo15Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int | None = None, width: int | None = None, num_frames: int = 121, num_inference_steps: int = 50, sigmas: list = None, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L542)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead.

height (`int`, *optional*) : The height in pixels of the generated video.

width (`int`, *optional*) : The width in pixels of the generated video.

num_frames (`int`, defaults to `121`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings from the second text encoder.

negative_prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings from the second text encoder.

negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings from the second text encoder.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated video. Choose between "np", "pt", or "latent".

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `HunyuanVideo15PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

**Returns:** `~HunyuanVideo15PipelineOutput` or `tuple`

If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideo15Pipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v"
>>> pipe = HunyuanVideo15Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> output = pipe(
...     prompt="A cat walks on the grass, realistic",
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=15)
```

#### encode_prompt[[diffusers.HunyuanVideo15Pipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, batch_size: int = 1, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L334)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

#### prepare_cond_latents_and_mask[[diffusers.HunyuanVideo15Pipeline.prepare_cond_latents_and_mask]]

```python
prepare_cond_latents_and_mask(latents, dtype: typing.Optional[torch.dtype], device: typing.Optional[torch.device])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L508)

**Parameters:**

latents : Main latents tensor (B, C, F, H, W)

**Returns:** `tuple`

(cond_latents_concat, mask_concat) - both are zero tensors for t2v

Prepare conditional latents and mask for t2v generation.

## HunyuanVideo15ImageToVideoPipeline[[diffusers.HunyuanVideo15ImageToVideoPipeline]]

#### diffusers.HunyuanVideo15ImageToVideoPipeline[[diffusers.HunyuanVideo15ImageToVideoPipeline]]

```python
diffusers.HunyuanVideo15ImageToVideoPipeline(text_encoder: Qwen2_5_VLTextModel, tokenizer: Qwen2Tokenizer, transformer: HunyuanVideo15Transformer3DModel, vae: AutoencoderKLHunyuanVideo15, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder_2: T5EncoderModel, tokenizer_2: ByT5Tokenizer, guider: ClassifierFreeGuidance, image_encoder: SiglipVisionModel, feature_extractor: SiglipImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L193)

**Parameters:**

transformer ([HunyuanVideo15Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.HunyuanVideo15Transformer3DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLHunyuanVideo15](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_hunyuan_video15#diffusers.AutoencoderKLHunyuanVideo15)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider ([ClassifierFreeGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance)) : [ClassifierFreeGuidance]for classifier free guidance.

image_encoder (`SiglipVisionModel`) : [SiglipVisionModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipVisionModel) variant.

feature_extractor (`SiglipImageProcessor`) : [SiglipImageProcessor](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipImageProcessor) variant.

Pipeline for image-to-video generation using HunyuanVideo1.5.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideo15ImageToVideoPipeline.__call__]]

```python
__call__(image: Image, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, num_frames: int = 121, num_inference_steps: int = 50, sigmas: list = None, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L650)

**Parameters:**

image (`PIL.Image.Image`) : The input image to condition video generation on.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds` instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead.

num_frames (`int`, defaults to `121`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings from the second text encoder.

negative_prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings from the second text encoder.

negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings from the second text encoder.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated video. Choose between "np", "pt", or "latent".

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `HunyuanVideo15PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

**Returns:** `~HunyuanVideo15PipelineOutput` or `tuple`

If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideo15ImageToVideoPipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v"
>>> pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")

>>> output = pipe(
...     prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
...     image=image,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=24)
```

#### encode_prompt[[diffusers.HunyuanVideo15ImageToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, batch_size: int = 1, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L422)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

#### prepare_cond_latents_and_mask[[diffusers.HunyuanVideo15ImageToVideoPipeline.prepare_cond_latents_and_mask]]

```python
prepare_cond_latents_and_mask(latents: Tensor, image: Image, batch_size: int, height: int, width: int, dtype: dtype, device: device)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L594)

**Parameters:**

latents : Main latents tensor (B, C, F, H, W)

**Returns:** `tuple`

(cond_latents_concat, mask_concat) - both are zero tensors for t2v

Prepare conditional latents and mask for t2v generation.

## HunyuanVideo15PipelineOutput[[diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput]]

#### diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput[[diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput]]

```python
diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for HunyuanVideo1.5 pipelines.

### Omnigen
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/omnigen.md

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

# OmniGen

[OmniGen: Unified Image Generation](https://huggingface.co/papers/2409.11340) from BAAI, by Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, Zheng Liu.

The abstract from the paper is:

*The emergence of Large Language Models (LLMs) has unified language  generation tasks and revolutionized human-machine interaction.  However, in the realm of image generation, a unified model capable of handling various tasks within a single framework remains largely unexplored. In this work, we introduce OmniGen, a new diffusion model for unified image generation. OmniGen is characterized by the following features: 1) Unification: OmniGen not only demonstrates text-to-image generation capabilities but also inherently supports various downstream tasks, such as image editing, subject-driven generation, and visual conditional generation. 2) Simplicity: The architecture of OmniGen is highly simplified, eliminating the need for additional plugins. Moreover, compared to existing diffusion models, it is more user-friendly and can complete complex tasks end-to-end through instructions without the need for extra intermediate steps, greatly simplifying the image generation workflow. 3) Knowledge Transfer: Benefit from learning in a unified format, OmniGen effectively transfers knowledge across different tasks, manages unseen tasks and domains, and exhibits novel capabilities. We also explore the model’s reasoning capabilities and potential applications of the chain-of-thought mechanism.  This work represents the first attempt at a general-purpose image generation model,  and we will release our resources at https://github.com/VectorSpaceLab/OmniGen to foster future advancements.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [staoxiao](https://github.com/staoxiao). The original codebase can be found [here](https://github.com/VectorSpaceLab/OmniGen). The original weights can be found under [hf.co/shitao](https://huggingface.co/Shitao/OmniGen-v1).

## Load model checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [from_pretrained()](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) method.

```python
import torch
from diffusers import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1-diffusers", dtype=torch.bfloat16)
```

## Text-to-image

For text-to-image, pass a text prompt. By default, OmniGen generates a 1024x1024 image. 
You can try setting the `height` and `width` parameters to generate images with different size.

```python
import torch
from diffusers import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")

prompt = "Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3,
    generator=torch.Generator(device="cpu").manual_seed(111),
).images[0]
image.save("output.png")
```

    

## Image edit

OmniGen supports multimodal inputs. 
When the input includes an image, you need to add a placeholder `<img><|image_1|></img>` in the text prompt to represent the image. 
It is recommended to enable `use_input_image_size_as_output` to keep the edited image the same size as the original image.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="<img><|image_1|></img> Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/t2i_woman_with_book.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(222)
).images[0]
image.save("output.png")
```

  
    
    original image
  
  
    
    edited image
  

OmniGen has some interesting features, such as visual reasoning, as shown in the example below.

```python
prompt="If the woman is thirsty, what should she take? Find it in the image and highlight it in blue. <img><|image_1|></img>"
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]
image.save("output.png")
```

    

## Controllable generation

OmniGen can handle several classic computer vision tasks. As shown below, OmniGen can detect human skeletons in input images, which can be used as control conditions to generate new images.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="Detect the skeleton of human in this image: <img><|image_1|></img>"
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image1 = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(333)
).images[0]
image1.save("image1.png")

prompt="Generate a new photo using the following picture and text as conditions: <img><|image_1|></img>\n A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/skeletal.png")]
image2 = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(333)
).images[0]
image2.save("image2.png")
```

  
    
    original image
  
  
    
    detected skeleton
  
  
    
    skeleton to image
  

OmniGen can also directly use relevant information from input images to generate new images.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="Following the pose of this image <img><|image_1|></img>, generate a new photo: A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]
image.save("output.png")
```

  
    
    generated image
  

## ID and object preserving

OmniGen can generate multiple images based on the people and objects in the input image and supports inputting multiple images simultaneously. 
Additionally, OmniGen can extract desired objects from an image containing multiple objects based on instructions.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="A man and a woman are sitting at a classroom desk. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>"
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/3.png")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/4.png")
input_images=[input_image_1, input_image_2]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    height=1024,
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator(device="cpu").manual_seed(666)
).images[0]
image.save("output.png")
```

  
    
    input_image_1
  
  
    
    input_image_2
  
  
    
    generated image
  

```py
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")

prompt="A woman is walking down the street, wearing a white long-sleeve blouse with lace details on the sleeves, paired with a blue pleated skirt. The woman is <img><|image_1|></img>. The long-sleeve blouse and a pleated skirt are <img><|image_2|></img>."
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/emma.jpeg")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/dress.jpg")
input_images=[input_image_1, input_image_2]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    height=1024,
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator(device="cpu").manual_seed(666)
).images[0]
image.save("output.png")
```

  
    
    person image
  
  
    
    clothe image
  
  
    
    generated image
  

## Optimization when using multiple images 

For text-to-image task, OmniGen requires minimal memory and time costs (9GB memory and 31s for a 1024x1024 image on A800 GPU). 
However, when using input images, the computational cost increases. 

Here are some guidelines to help you reduce computational costs when using multiple images. The experiments are conducted on an A800 GPU with two input images.

Like other pipelines, you can reduce memory usage by offloading the model: `pipe.enable_model_cpu_offload()` or `pipe.enable_sequential_cpu_offload() `. 
In OmniGen, you can also decrease computational overhead by reducing the `max_input_image_size`. 
The memory consumption for different image sizes is shown in the table below:

| Method                    | Memory Usage |
|---------------------------|--------------|
| max_input_image_size=1024 | 40GB         |
| max_input_image_size=512  | 17GB         |
| max_input_image_size=256  | 14GB         |

## OmniGenPipeline[[diffusers.OmniGenPipeline]]

#### diffusers.OmniGenPipeline[[diffusers.OmniGenPipeline]]

```python
diffusers.OmniGenPipeline(transformer: OmniGenTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, tokenizer: LlamaTokenizer)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/omnigen/pipeline_omnigen.py#L119)

**Parameters:**

transformer ([OmniGenTransformer2DModel](/docs/diffusers/v0.40.0/en/api/models/omnigen_transformer#diffusers.OmniGenTransformer2DModel)) : Autoregressive Transformer architecture for OmniGen.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

tokenizer (`LlamaTokenizer`) : Text tokenizer of class. [LlamaTokenizer](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaTokenizer).

The OmniGen pipeline for multimodal-to-image generation.

Reference: https://huggingface.co/papers/2409.11340

#### __call__[[diffusers.OmniGenPipeline.__call__]]

```python
__call__(prompt: str | list[str], input_images: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], list[typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]]] = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, max_input_image_size: int = 1024, timesteps: list = None, guidance_scale: float = 2.5, img_guidance_scale: float = 1.6, use_input_image_size_as_output: bool = False, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/omnigen/pipeline_omnigen.py#L277)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If the input includes images, need to add placeholders `<img><|image_i|></img>` in the prompt to indicate the position of the i-th images.

input_images (`PipelineImageInput` or `list[PipelineImageInput]`, *optional*) : The list of input images. We will replace the "<|image_i|>" in prompt with the i-th image in list.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

max_input_image_size (`int`, *optional*, defaults to 1024) : the maximum size of input image, which will be used to crop the input image to the maximum size

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 2.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

img_guidance_scale (`float`, *optional*, defaults to 1.6) : Defined as equation 3 in [Instrucpix2pix](https://huggingface.co/papers/2211.09800).

use_input_image_size_as_output (bool, defaults to False) : whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import OmniGenPipeline

>>> pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1-diffusers", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, num_inference_steps=50, guidance_scale=2.5).images[0]
>>> image.save("output.png")
```

Returns: [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`:
If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images.

#### encode_input_images[[diffusers.OmniGenPipeline.encode_input_images]]

```python
encode_input_images(input_pixel_values: list, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/omnigen/pipeline_omnigen.py#L171)

**Parameters:**

input_pixel_values : normalized pixel of input images

device --

get the continue embedding of input images by VAE

Returns: torch.Tensor

### Chronoedit
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/chronoedit.md

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

  
    
      
    
  

# ChronoEdit

[ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation](https://huggingface.co/papers/2510.04290) from NVIDIA and University of Toronto, by Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, Huan Ling.

> **TL;DR:** ChronoEdit reframes image editing as a video generation task, using input and edited images as start/end frames to leverage pretrained video models with temporal consistency. A temporal reasoning stage introduces reasoning tokens to ensure physically plausible edits and visualize the editing trajectory.

*Recent advances in large generative models have greatly enhanced both image editing and in-context image generation, yet a critical gap remains in ensuring physical consistency, where edited objects must remain coherent. This capability is especially vital for world simulation related tasks. In this paper, we present ChronoEdit, a framework that reframes image editing as a video generation problem. First, ChronoEdit treats the input and edited images as the first and last frames of a video, allowing it to leverage large pretrained video generative models that capture not only object appearance but also the implicit physics of motion and interaction through learned temporal consistency. Second, ChronoEdit introduces a temporal reasoning stage that explicitly performs editing at inference time. Under this setting, target frame is jointly denoised with reasoning tokens to imagine a plausible editing trajectory that constrains the solution space to physically viable transformations. The reasoning tokens are then dropped after a few steps to avoid the high computational cost of rendering a full video. To validate ChronoEdit, we introduce PBench-Edit, a new benchmark of image-prompt pairs for contexts that require physical consistency, and demonstrate that ChronoEdit surpasses state-of-the-art baselines in both visual fidelity and physical plausibility. Project page for code and models: [this https URL](https://research.nvidia.com/labs/toronto-ai/chronoedit).*

The ChronoEdit pipeline is developed by the ChronoEdit Team. The original code is available on [GitHub](https://github.com/nv-tlabs/ChronoEdit), and pretrained models can be found in the [nvidia/ChronoEdit](https://huggingface.co/collections/nvidia/chronoedit) collection on Hugging Face.

Available Models/LoRAs:
- [nvidia/ChronoEdit-14B-Diffusers](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers)
- [nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora)
- [nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora)

### Image Editing

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

Optionally, enable **temporal reasoning** for improved physical consistency:
```py
output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=29,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_temporal_reasoning=True,
    num_temporal_reasoning_steps=50,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

### Inference with 8-Step Distillation Lora

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, dtype=torch.bfloat16)
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers", weight_name="lora/chronoedit_distill_lora.safetensors", adapter_name="distill")
pipe.fuse_lora(adapter_names=["distill"], lora_scale=1.0)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=8,
    guidance_scale=1.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

### Inference with Multiple LoRAs

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, dtype=torch.bfloat16)
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora", weight_name="paintbrush_lora_diffusers.safetensors", adapter_name="paintbrush")
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers", weight_name="lora/chronoedit_distill_lora.safetensors", adapter_name="distill")
pipe.fuse_lora(adapter_names=["paintbrush", "distill"], lora_scale=1.0)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)
pipe.to("cuda")

image = load_image(
    "https://raw.githubusercontent.com/nv-tlabs/ChronoEdit/refs/heads/main/assets/images/input_paintbrush.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "Turn the pencil sketch in the image into an actual object that is consistent with the image’s content. The user wants to change the sketch to a crown and a hat."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=8,
    guidance_scale=1.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output_1.png")
```

## ChronoEditPipeline[[diffusers.ChronoEditPipeline]]

#### diffusers.ChronoEditPipeline[[diffusers.ChronoEditPipeline]]

```python
diffusers.ChronoEditPipeline(tokenizer: AutoTokenizer, text_encoder: UMT5EncoderModel, image_encoder: CLIPVisionModel, image_processor: CLIPImageProcessorPil, transformer: ChronoEditTransformer3DModel, vae: AutoencoderKLWan, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L128)

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

image_encoder (`CLIPVisionModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.

transformer ([WanTransformer3DModel](/docs/diffusers/v0.40.0/en/api/models/wan_transformer_3d#diffusers.WanTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for image-to-video generation using Wan.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.ChronoEditPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 480, width: int = 832, num_frames: int = 81, num_inference_steps: int = 50, guidance_scale: float = 5.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, image_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int, NoneType], diffusers.callbacks.PipelineCallback | diffusers.callbacks.MultiPipelineCallbacks]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512, enable_temporal_reasoning: bool = False, num_temporal_reasoning_steps: int = 0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L467)

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, defaults to `480`) : The height of the generated video.

width (`int`, defaults to `832`) : The width of the generated video.

num_frames (`int`, defaults to `81`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `5.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `negative_prompt` input argument.

image_embeds (`torch.Tensor`, *optional*) : Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided, image embeddings are generated from the `image` input argument.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `ChronoEditPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, defaults to `512`) : The maximum sequence length of the text encoder. If the prompt is longer than this, it will be truncated. If the prompt is shorter, it will be padded to this length.

enable_temporal_reasoning (`bool`, *optional*, defaults to `False`) : Whether to enable temporal reasoning.

num_temporal_reasoning_steps (`int`, *optional*, defaults to `0`) : The number of steps to enable temporal reasoning.

**Returns:** `~ChronoEditPipelineOutput` or `tuple`

If `return_dict` is `True`, `ChronoEditPipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> import numpy as np
>>> from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
>>> from diffusers.utils import export_to_video, load_image
>>> from transformers import CLIPVisionModel

>>> # Available models: nvidia/ChronoEdit-14B-Diffusers
>>> model_id = "nvidia/ChronoEdit-14B-Diffusers"
>>> image_encoder = CLIPVisionModel.from_pretrained(
...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
... )
>>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
>>> transformer = ChronoEditTransformer3DModel.from_pretrained(
...     model_id, subfolder="transformer", torch_dtype=torch.bfloat16
... )
>>> pipe = ChronoEditPipeline.from_pretrained(
...     model_id, vae=vae, image_encoder=image_encoder, transformer=transformer, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = load_image("https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png")
>>> max_area = 720 * 1280
>>> aspect_ratio = image.height / image.width
>>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
>>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
>>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
>>> image = image.resize((width, height))
>>> prompt = (
...     "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
...     "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
... )

>>> output = pipe(
...     image=image,
...     prompt=prompt,
...     height=height,
...     width=width,
...     num_frames=5,
...     guidance_scale=5.0,
...     enable_temporal_reasoning=False,
...     num_temporal_reasoning_steps=0,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=16)
```

#### encode_prompt[[diffusers.ChronoEditPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L239)

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

## ChronoEditPipelineOutput[[diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput]]

#### diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput[[diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput]]

```python
diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/chronoedit/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for ChronoEdit pipelines.

### Consisid
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/consisid.md

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

# ConsisID

  

[Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://huggingface.co/papers/2411.17440) from Peking University & University of Rochester & etc, by Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, Li Yuan.

The abstract from the paper is:

*Identity-preserving text-to-video (IPT2V) generation aims to create high-fidelity videos with consistent human identity. It is an important task in video generation but remains an open problem for generative models. This paper pushes the technical frontier of IPT2V in two directions that have not been resolved in the literature: (1) A tuning-free pipeline without tedious case-by-case finetuning, and (2) A frequency-aware heuristic identity-preserving Diffusion Transformer (DiT)-based control scheme. To achieve these goals, we propose **ConsisID**, a tuning-free DiT-based controllable IPT2V model to keep human-**id**entity **consis**tent in the generated video. Inspired by prior findings in frequency analysis of vision/diffusion transformers, it employs identity-control signals in the frequency domain, where facial features can be decomposed into low-frequency global features (e.g., profile, proportions) and high-frequency intrinsic features (e.g., identity markers that remain unaffected by pose changes). First, from a low-frequency perspective, we introduce a global facial extractor, which encodes the reference image and facial key points into a latent space, generating features enriched with low-frequency information. These features are then integrated into the shallow layers of the network to alleviate training challenges associated with DiT. Second, from a high-frequency perspective, we design a local facial extractor to capture high-frequency details and inject them into the transformer blocks, enhancing the model's ability to preserve fine-grained features. To leverage the frequency information for identity preservation, we propose a hierarchical training strategy, transforming a vanilla pre-trained video generation model into an IPT2V model. Extensive experiments demonstrate that our frequency-aware heuristic scheme provides an optimal control solution for DiT-based models. Thanks to this scheme, our **ConsisID** achieves excellent results in generating high-quality, identity-preserving videos, making strides towards more effective IPT2V. The model weight of ConsID is publicly available at https://github.com/PKU-YuanGroup/ConsisID.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [SHYuanBest](https://github.com/SHYuanBest). The original codebase can be found [here](https://github.com/PKU-YuanGroup/ConsisID). The original weights can be found under [hf.co/BestWishYsh](https://huggingface.co/BestWishYsh).

There are two official ConsisID checkpoints for identity-preserving text-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`BestWishYsh/ConsisID-preview`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |
| [`BestWishYsh/ConsisID-1.5`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |

### Memory optimization

ConsisID requires about 44 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/SHYuanBest/bc4207c36f454f9e969adbb50eaf8258) script.

| Feature (overlay the previous) | Max Memory Allocated | Max Memory Reserved |
| :----------------------------- | :------------------- | :------------------ |
| -                              | 37 GB                | 44 GB               |
| enable_model_cpu_offload       | 22 GB                | 25 GB               |
| enable_sequential_cpu_offload  | 16 GB                | 22 GB               |
| vae.enable_slicing             | 16 GB                | 22 GB               |
| vae.enable_tiling              | 5 GB                 | 7 GB                |

## Load Model Checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [from_pretrained()](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) method.

```python
# !pip install consisid_eva_clip insightface facexlib
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from huggingface_hub import snapshot_download

# Download ckpts
snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")

# Load face helper model to preprocess input face image
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)

# Load consisid base model
pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", dtype=torch.bfloat16)
pipe.to("cuda")
```

## Identity-Preserving Text-to-Video

For identity-preserving text-to-video, pass a text prompt and an image contain clear face (e.g., preferably half-body or full-body). By default, ConsisID generates a 720x480 video for the best results.

```python
from diffusers.utils import export_to_video

prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_input.png?download=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, "cuda", torch.bfloat16, image, is_align_face=True)

video = pipe(image=image, prompt=prompt, num_inference_steps=50, guidance_scale=6.0, use_dynamic_cfg=False, id_vit_hidden=id_vit_hidden, id_cond=id_cond, kps_cond=face_kps, generator=torch.Generator("cuda").manual_seed(42))
export_to_video(video.frames[0], "output.mp4", fps=8)
```

  
    Face Image
    Video
    Description
  
  
    
    
    The video, in a beautifully crafted animated style, features a confident woman riding a horse through a lush forest clearing. Her expression is focused yet serene as she adjusts her wide-brimmed hat with a practiced hand. She wears a flowy bohemian dress, which moves gracefully with the rhythm of the horse, the fabric flowing fluidly in the animated motion. The dappled sunlight filters through the trees, casting soft, painterly patterns on the forest floor. Her posture is poised, showing both control and elegance as she guides the horse with ease. The animation's gentle, fluid style adds a dreamlike quality to the scene, with the woman’s calm demeanor and the peaceful surroundings evoking a sense of freedom and harmony.
  
  
    
    
    The video, in a captivating animated style, shows a woman standing in the center of a snowy forest, her eyes narrowed in concentration as she extends her hand forward. She is dressed in a deep blue cloak, her breath visible in the cold air, which is rendered with soft, ethereal strokes. A faint smile plays on her lips as she summons a wisp of ice magic, watching with focus as the surrounding trees and ground begin to shimmer and freeze, covered in delicate ice crystals. The animation’s fluid motion brings the magic to life, with the frost spreading outward in intricate, sparkling patterns. The environment is painted with soft, watercolor-like hues, enhancing the magical, dreamlike atmosphere. The overall mood is serene yet powerful, with the quiet winter air amplifying the delicate beauty of the frozen scene.
  
  
    
    
    The animation features a whimsical portrait of a balloon seller standing in a gentle breeze, captured with soft, hazy brushstrokes that evoke the feel of a serene spring day. His face is framed by a gentle smile, his eyes squinting slightly against the sun, while a few wisps of hair flutter in the wind. He is dressed in a light, pastel-colored shirt, and the balloons around him sway with the wind, adding a sense of playfulness to the scene. The background blurs softly, with hints of a vibrant market or park, enhancing the light-hearted, yet tender mood of the moment.
  
  
    
    
    The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.
  
  
    
    
    The video features a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose. The baby has a determined look on their face, with eyes wide and lips pursed in concentration, as if ready to take on a challenge. The setting appears playful, with colorful toys scattered around and a soft rug underfoot, while sunlight streams through a nearby window, highlighting the fluttering cape and adding to the impression of heroism. The overall atmosphere is lighthearted and fun, with the baby's expressions capturing a mix of innocence and an adorable attempt at bravery, as if truly ready to save the day.
  

## Resources

Learn more about ConsisID with the following resources.
- A [video](https://www.youtube.com/watch?v=PhlgC-bI5SQ) demonstrating ConsisID's main features.
- The research paper, [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://hf.co/papers/2411.17440) for more details.

## ConsisIDPipeline[[diffusers.ConsisIDPipeline]]

#### diffusers.ConsisIDPipeline[[diffusers.ConsisIDPipeline]]

```python
diffusers.ConsisIDPipeline(*args, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/utils/dummy_torch_and_transformers_and_opencv_objects.py#L5)

#### __call__[[diffusers.ConsisIDPipeline.__call__]]

```python
__call__(*args, **kwargs)
```

Call self as a function.

## ConsisIDPipelineOutput[[diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput]]

#### diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput[[diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput]]

```python
diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/consisid/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for ConsisID pipelines.

### Prx
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/prx.md

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

# PRX

PRX generates high-quality images from text using a simplified MMDIT architecture where text tokens don't update through transformer blocks. It employs flow matching with discrete scheduling for efficient sampling and uses Google's T5Gemma-2B-2B-UL2 model for multi-language text encoding. The ~1.3B parameter transformer delivers fast inference without sacrificing quality. You can choose between Flux VAE (8x compression, 16 latent channels) for balanced quality and speed or DC-AE (32x compression, 32 latent channels) for latent compression and faster processing.

## Available models

PRX offers multiple variants with different VAE configurations, each optimized for specific resolutions. Base models excel with detailed prompts, capturing complex compositions and subtle details. Fine-tuned models trained on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) improve aesthetic quality, especially with simpler prompts.

| Model | Resolution | Fine-tuned | Distilled | Description | Suggested prompts | Suggested parameters | Recommended dtype |
|:-----:|:-----------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [`Photoroom/prx-256-t2i`](https://huggingface.co/Photoroom/prx-256-t2i)| 256 | No | No | Base model pre-trained at 256 with Flux VAE|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-256-t2i-sft`](https://huggingface.co/Photoroom/prx-256-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i`](https://huggingface.co/Photoroom/prx-512-t2i)| 512 | No | No | Base model pre-trained at 512 with Flux VAE |Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-sft`](https://huggingface.co/Photoroom/prx-512-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/prx-512-t2i-sft`](https://huggingface.co/Photoroom/prx-512-t2i-sft) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae)| 512 | No | No | Base model pre-trained at 512 with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae)|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae-sft`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae) | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/prx-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft-distilled) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |s

Refer to [this](https://huggingface.co/collections/Photoroom/prx-models-68e66254c202ebfab99ad38e) collection for more information.

## Loading the pipeline

Load the pipeline with [from_pretrained()](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained).

```py
from diffusers.pipelines.prx import PRXPipeline

# Load pipeline - VAE and text encoder will be loaded from HuggingFace
pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft", dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("prx_output.png")
```

### Manual Component Loading

Load components individually to customize the pipeline for instance to use quantized models.

```py
import torch
from diffusers.pipelines.prx import PRXPipeline
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5GemmaModel, GemmaTokenizerFast
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
# Load transformer
transformer = PRXTransformer2DModel.from_pretrained(
    "checkpoints/prx-512-t2i-sft",
    subfolder="transformer",
    quantization_config=quant_config,
    dtype=torch.bfloat16,
)

# Load scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "checkpoints/prx-512-t2i-sft", subfolder="scheduler"
)

# Load T5Gemma text encoder
t5gemma_model = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2",
                                            quantization_config=quant_config,
                                            dtype=torch.bfloat16)
text_encoder = t5gemma_model.encoder.to(dtype=torch.bfloat16)
tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")
tokenizer.model_max_length = 256

# Load VAE - choose either Flux VAE or DC-AE
# Flux VAE
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev",
                                    subfolder="vae",
                                    quantization_config=quant_config,
                                    dtype=torch.bfloat16)

pipe = PRXPipeline(
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    vae=vae
)
pipe.to("cuda")
```

## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers.pipelines.prx import PRXPipeline

pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft", dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Offload components to CPU when not in use

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PRXPipeline[[diffusers.PRXPipeline]]

#### diffusers.PRXPipeline[[diffusers.PRXPipeline]]

```python
diffusers.PRXPipeline(transformer: PRXTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder: T5GemmaEncoder, tokenizer: transformers.models.t5.tokenization_t5.T5Tokenizer | transformers.models.gemma.tokenization_gemma.GemmaTokenizer | transformers.models.auto.tokenization_auto.AutoTokenizer, vae: diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL | diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC | None = None, default_sample_size: int | None = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx.py#L257)

**Parameters:**

transformer (`PRXTransformer2DModel`) : The PRX transformer model to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

text_encoder (`T5GemmaEncoder`) : Text encoder model for encoding prompts.

tokenizer ([`T5TokenizerFast` or `GemmaTokenizerFast`]) : Tokenizer for the text encoder.

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) or [AutoencoderDC](/docs/diffusers/v0.40.0/en/api/models/autoencoder_dc#diffusers.AutoencoderDC)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. Supports both AutoencoderKL (8x compression) and AutoencoderDC (32x compression).

Pipeline for text-to-image generation using PRX Transformer.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.PRXPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str = '', height: int | None = None, width: int | None = None, num_inference_steps: int = 28, timesteps: list = None, guidance_scale: float = 4.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, negative_prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, use_resolution_binning: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], tokenizer_max_length: int | None = None, skip_text_cleaning: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx.py#L554)

**Parameters:**

prompt (*str* or *list[str]*, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass *prompt_embeds* instead.

negative_prompt (*str*, *optional*, defaults to *""*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if *guidance_scale* is less than *1*).

height (*int*, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image.

width (*int*, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image.

num_inference_steps (*int*, *optional*, defaults to 28) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (*list[int]*, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a *timesteps* argument in their *set_timesteps* method. If not defined, the default behavior when *num_inference_steps* is passed will be used. Must be in descending order.

guidance_scale (*float*, *optional*, defaults to 4.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). *guidance_scale* is defined as *w* of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting *guidance_scale > 1*. Higher guidance scale encourages to generate images that are closely linked to the text *prompt*, usually at the expense of lower image quality.

num_images_per_prompt (*int*, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (*torch.Generator* or *list[torch.Generator]*, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (*torch.Tensor*, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random *generator*.

prompt_embeds (*torch.FloatTensor*, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from *prompt* input argument.

negative_prompt_embeds (*torch.FloatTensor*, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided and *guidance_scale > 1*, negative embeddings will be generated from an empty string.

prompt_attention_mask (*torch.BoolTensor*, *optional*) : Pre-generated attention mask for *prompt_embeds*. If not provided, attention mask will be generated from *prompt* input argument.

negative_prompt_attention_mask (*torch.BoolTensor*, *optional*) : Pre-generated attention mask for *negative_prompt_embeds*. If not provided and *guidance_scale > 1*, attention mask will be generated from an empty string.

output_type (*str*, *optional*, defaults to *"pil"*) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): *PIL.Image.Image* or *np.array*.

tokenizer_max_length (*int*, *optional*) : Override the maximum number of tokens used when tokenizing the prompt. Defaults to the tokenizer's own `model_max_length` when not set.

skip_text_cleaning (*bool*, *optional*, defaults to *False*) : If *True*, uses only light prompt cleaning (fix encoding + unescape HTML) instead of the full DeepFloyd cleaning pipeline.

return_dict (*bool*, *optional*, defaults to *True*) : Whether or not to return a [*~pipelines.prx.PRXPipelineOutput*] instead of a plain tuple.

use_resolution_binning (*bool*, *optional*, defaults to *True*) : If set to *True*, the requested height and width are first mapped to the closest resolutions using predefined aspect ratio bins. After the produced latents are decoded into images, they are resized back to the requested resolution. Useful for generating non-square images at optimal resolutions.

callback_on_step_end (*Callable*, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: *callback_on_step_end(self, step, timestep, callback_kwargs)*. *callback_kwargs* will include a list of all tensors as specified by *callback_on_step_end_tensor_inputs*.

callback_on_step_end_tensor_inputs (*list*, *optional*) : The list of tensor inputs for the *callback_on_step_end* function. The tensors specified in the list will be passed as *callback_kwargs* argument. You will only be able to include tensors that are listed in the *._callback_tensor_inputs* attribute.

**Returns:** [*~pipelines.prx.PRXPipelineOutput*] or *tuple`

[*~pipelines.prx.PRXPipelineOutput*] if *return_dict* is
True, otherwise a *tuple. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import PRXPipeline

>>> # Load pipeline with from_pretrained
>>> pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft")
>>> pipe.to("cuda")

>>> prompt = "A digital painting of a rusty, vintage tram on a sandy beach"
>>> image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
>>> image.save("prx_output.png")
```

#### check_inputs[[diffusers.PRXPipeline.check_inputs]]

```python
check_inputs(prompt: str | list[str], height: int, width: int, guidance_scale: float, callback_on_step_end_tensor_inputs: list[str] | None = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx.py#L500)

Check that all inputs are in correct format.

#### encode_prompt[[diffusers.PRXPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, do_classifier_free_guidance: bool = True, negative_prompt: str = '', num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, negative_prompt_attention_mask: typing.Optional[torch.BoolTensor] = None, tokenizer_max_length: int | None = None, skip_text_cleaning: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx.py#L376)

Encode text prompt using standard text encoder and tokenizer, or use precomputed embeddings.

#### get_default_resolution[[diffusers.PRXPipeline.get_default_resolution]]

```python
get_default_resolution()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx.py#L340)

**Returns:** `int`

The default sample size (height/width) to use for generation.

Determine the default resolution based on the loaded VAE and config.

#### prepare_latents[[diffusers.PRXPipeline.prepare_latents]]

```python
prepare_latents(batch_size: int, num_channels_latents: int, height: int, width: int, dtype: dtype, device: device, generator: typing.Optional[torch.Generator] = None, latents: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_prx.py#L352)

Prepare initial latents for the diffusion process.

## PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

#### diffusers.pipelines.prx.PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

```python
diffusers.pipelines.prx.PRXPipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/prx/pipeline_output.py#L24)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for PRX pipelines.

### Kandinsky 2.1
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/kandinsky.md
