# Flux2

  
  

Flux.2 is the recent series of image generation models from Black Forest Labs, preceded by the [Flux.1](./flux) series. It is an entirely new model with a new architecture and pre-training done from scratch!

Original model checkpoints for Flux can be found [here](https://huggingface.co/black-forest-labs). Original inference code can be found [here](https://github.com/black-forest-labs/flux2).

> [!TIP]
> Flux2 can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. Additionally, Flux can benefit from quantization for memory efficiency with a trade-off in inference latency. Refer to [this blog post](https://huggingface.co/blog/quanto-diffusers) to learn more.
>
> [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

## Caption upsampling

Flux.2 can potentially generate better better outputs with better prompts. We can "upsample"
an input prompt by setting the `caption_upsample_temperature` argument in the pipeline call arguments.
The [official implementation](https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L140) recommends this value to be 0.15.

## Flux2Pipeline[[diffusers.Flux2Pipeline]]

#### diffusers.Flux2Pipeline[[diffusers.Flux2Pipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2.py#L251)

The Flux2 pipeline for text-to-image generation.

Reference: [https://bfl.ai/blog/flux-2](https://bfl.ai/blog/flux-2)

__call__diffusers.Flux2Pipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2.py#L745[{"name": "image", "val": ": PIL.Image.Image | list[PIL.Image.Image] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = 4.0"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "text_encoder_out_layers", "val": ": tuple = (10, 20, 30)"}, {"name": "caption_upsample_temperature", "val": ": float = None"}]- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **guidance_scale** (`float`, *optional*, defaults to 1.0) --
  Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
  a model to generate images more aligned with `prompt` at the expense of lower image quality.

  Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
  the [paper](https://huggingface.co/papers/2210.03142) to learn more.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.
- **text_encoder_out_layers** (`tuple[int]`) --
  Layer indices to use in the `text_encoder` to derive the final prompt embeddings.
- **caption_upsample_temperature** (`float`) --
  When specified, we will try to perform caption upsampling for potentially improved outputs. We
  recommend setting it to 0.15 if caption upsampling is to be performed.0`~pipelines.flux2.Flux2PipelineOutput` or `tuple``~pipelines.flux2.Flux2PipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import Flux2Pipeline

>>> pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, num_inference_steps=50, guidance_scale=2.5).images[0]
>>> image.save("flux.png")
```

**Parameters:**

transformer ([Flux2Transformer2DModel](/docs/diffusers/v0.39.0/en/api/models/flux2_transformer#diffusers.Flux2Transformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae (`AutoencoderKLFlux2`) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Mistral3ForConditionalGeneration`) : [Mistral3ForConditionalGeneration](https://huggingface.co/docs/transformers/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration)

tokenizer (`AutoProcessor`) : Tokenizer of class [PixtralProcessor](https://huggingface.co/docs/transformers/en/model_doc/pixtral#transformers.PixtralProcessor).

**Returns:**

``~pipelines.flux2.Flux2PipelineOutput` or `tuple``

`~pipelines.flux2.Flux2PipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.

## Flux2KleinPipeline[[diffusers.Flux2KleinPipeline]]

#### diffusers.Flux2KleinPipeline[[diffusers.Flux2KleinPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L155)

The Flux2 Klein pipeline for text-to-image generation.

Reference:
[https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)

__call__diffusers.Flux2KleinPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L612[{"name": "image", "val": ": list[PIL.Image.Image] | PIL.Image.Image | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": str | list[str] | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, dict], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "text_encoder_out_layers", "val": ": tuple = (9, 18, 27)"}]- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **guidance_scale** (`float`, *optional*, defaults to 4.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality. For step-wise distilled models,
  `guidance_scale` is ignored.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`List[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `List[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Note that "" is used as the negative prompt in this pipeline.
  If not provided, will be generated from "".
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.
- **text_encoder_out_layers** (`tuple[int]`) --
  Layer indices to use in the `text_encoder` to derive the final prompt embeddings.0`~pipelines.flux2.Flux2PipelineOutput` or `tuple``~pipelines.flux2.Flux2PipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import Flux2KleinPipeline

>>> pipe = Flux2KleinPipeline.from_pretrained(
...     "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, num_inference_steps=50, guidance_scale=4.0).images[0]
>>> image.save("flux2_output.png")
```

**Parameters:**

transformer ([Flux2Transformer2DModel](/docs/diffusers/v0.39.0/en/api/models/flux2_transformer#diffusers.Flux2Transformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae (`AutoencoderKLFlux2`) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen3ForCausalLM`) : [Qwen3ForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/qwen3#transformers.Qwen3ForCausalLM)

tokenizer (`Qwen2TokenizerFast`) : Tokenizer of class [Qwen2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2TokenizerFast).

**Returns:**

``~pipelines.flux2.Flux2PipelineOutput` or `tuple``

`~pipelines.flux2.Flux2PipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.

## Flux2KleinKVPipeline[[diffusers.Flux2KleinKVPipeline]]

#### diffusers.Flux2KleinKVPipeline[[diffusers.Flux2KleinKVPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein_kv.py#L155)

The Flux2 Klein KV pipeline for text-to-image generation with KV-cached reference image conditioning.

On the first denoising step, reference image tokens are included in the forward pass and their attention K/V
projections are cached. On subsequent steps, the cached K/V are reused without recomputing, providing faster
inference when using reference images.

Reference:
[https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)

__call__diffusers.Flux2KleinKVPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein_kv.py#L609[{"name": "image", "val": ": list[PIL.Image.Image] | PIL.Image.Image | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 4"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, dict], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "text_encoder_out_layers", "val": ": tuple = (9, 18, 27)"}]- **image** (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*) --
  Reference image(s) for conditioning. On the first denoising step, reference tokens are included in the
  forward pass and their attention K/V are cached. On subsequent steps, the cached K/V are reused without
  recomputing.
- **prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts to guide the image generation.
- **height** (`int`, *optional*) --
  The height in pixels of the generated image.
- **width** (`int`, *optional*) --
  The width in pixels of the generated image.
- **num_inference_steps** (`int`, *optional*, defaults to 4) --
  The number of denoising steps.
- **sigmas** (`List[float]`, *optional*) --
  Custom sigmas for the denoising schedule.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `List[torch.Generator]`, *optional*) --
  Generator(s) for deterministic generation.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  Output format: `"pil"` or `"np"`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether to return a `Flux2PipelineOutput` or a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  Extra kwargs passed to attention processors.
- **callback_on_step_end** (`Callable`, *optional*) --
  Callback function called at the end of each denoising step.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  Tensor inputs for the callback function.
- **max_sequence_length** (`int`, defaults to 512) --
  Maximum sequence length for the prompt.
- **text_encoder_out_layers** (`tuple[int]`) --
  Layer indices for text encoder hidden state extraction.0`~pipelines.flux2.Flux2PipelineOutput` or `tuple`.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import Flux2KleinKVPipeline

>>> pipe = Flux2KleinKVPipeline.from_pretrained(
...     "black-forest-labs/FLUX.2-klein-9b-kv", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> ref_image = Image.open("reference.png")
>>> image = pipe("A cat dressed like a wizard", image=ref_image, num_inference_steps=4).images[0]
>>> image.save("flux2_kv_output.png")
```

**Parameters:**

transformer ([Flux2Transformer2DModel](/docs/diffusers/v0.39.0/en/api/models/flux2_transformer#diffusers.Flux2Transformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae (`AutoencoderKLFlux2`) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen3ForCausalLM`) : [Qwen3ForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/qwen3#transformers.Qwen3ForCausalLM)

tokenizer (`Qwen2TokenizerFast`) : Tokenizer of class [Qwen2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2TokenizerFast).

**Returns:**

`~pipelines.flux2.Flux2PipelineOutput` or `tuple`.

### Chronoedit
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/chronoedit.md

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
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
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
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
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
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
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

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L128)

Pipeline for image-to-video generation using Wan.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.ChronoEditPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L467[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 480"}, {"name": "width", "val": ": int = 832"}, {"name": "num_frames", "val": ": int = 81"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 5.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "image_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, NoneType], diffusers.callbacks.PipelineCallback | diffusers.callbacks.MultiPipelineCallbacks]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "enable_temporal_reasoning", "val": ": bool = False"}, {"name": "num_temporal_reasoning_steps", "val": ": int = 0"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, defaults to `480`) --
  The height of the generated video.
- **width** (`int`, defaults to `832`) --
  The width of the generated video.
- **num_frames** (`int`, defaults to `81`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `5.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
  `guidance_scale` is defined as `w` of equation 2. of [Imagen
  Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
  1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
  usually at the expense of lower image quality.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `negative_prompt` input argument.
- **image_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
  image embeddings are generated from the `image` input argument.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `ChronoEditPipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
  truncated. If the prompt is shorter, it will be padded to this length.
- **enable_temporal_reasoning** (`bool`, *optional*, defaults to `False`) --
  Whether to enable temporal reasoning.
- **num_temporal_reasoning_steps** (`int`, *optional*, defaults to `0`) --
  The number of steps to enable temporal reasoning.0`~ChronoEditPipelineOutput` or `tuple`If `return_dict` is `True`, `ChronoEditPipelineOutput` is returned, otherwise a `tuple` is returned
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

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

image_encoder (`CLIPVisionModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.

transformer ([WanTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/wan_transformer_3d#diffusers.WanTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~ChronoEditPipelineOutput` or `tuple``

If `return_dict` is `True`, `ChronoEditPipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.ChronoEditPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L239)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

## ChronoEditPipelineOutput[[diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput]]

#### diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput[[diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/chronoedit/pipeline_output.py#L9)

Output class for ChronoEdit pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Kandinsky 3
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/kandinsky3.md
