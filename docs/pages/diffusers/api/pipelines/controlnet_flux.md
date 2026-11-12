# ControlNet with Flux.1

  

FluxControlNetPipeline is an implementation of ControlNet for Flux.1.

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, eg, edges, depth, segmentation, human pose, etc, with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (<50k) and large (>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models.*

This controlnet code is implemented by [The InstantX Team](https://huggingface.co/InstantX). You can find pre-trained checkpoints for Flux-ControlNet in the table below:

| ControlNet type | Developer | Link |
| -------- | ---------- | ---- |
| Canny | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny) |
| Depth | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth) |
| Union | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union) |

XLabs ControlNets are also supported, which was contributed by the [XLabs team](https://huggingface.co/XLabs-AI).

| ControlNet type | Developer | Link |
| -------- | ---------- | ---- |
| Canny | [The XLabs Team](https://huggingface.co/XLabs-AI) | [Link](https://huggingface.co/XLabs-AI/flux-controlnet-canny-diffusers) |
| Depth | [The XLabs Team](https://huggingface.co/XLabs-AI) | [Link](https://huggingface.co/XLabs-AI/flux-controlnet-depth-diffusers) |
| HED | [The XLabs Team](https://huggingface.co/XLabs-AI) | [Link](https://huggingface.co/XLabs-AI/flux-controlnet-hed-diffusers) |

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

## FluxControlNetPipeline[[diffusers.FluxControlNetPipeline]]

#### diffusers.FluxControlNetPipeline[[diffusers.FluxControlNetPipeline]]

```python
diffusers.FluxControlNetPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: T5Tokenizer, transformer: FluxTransformer2DModel, controlnet: diffusers.models.controlnets.controlnet_flux.FluxControlNetModel | list[diffusers.models.controlnets.controlnet_flux.FluxControlNetModel] | tuple[diffusers.models.controlnets.controlnet_flux.FluxControlNetModel] | diffusers.models.controlnets.controlnet_flux.FluxMultiControlNetModel, image_encoder: CLIPVisionModelWithProjection = None, feature_extractor: CLIPImageProcessorPil = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L177)

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

The Flux pipeline for text-to-image generation.

Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

#### __call__[[diffusers.FluxControlNetPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, prompt_2: str | list[str] | None = None, negative_prompt: str | list[str] = None, negative_prompt_2: str | list[str] | None = None, true_cfg_scale: float = 1.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 28, sigmas: list[float] | None = None, guidance_scale: float = 7.0, control_guidance_start: float | list[float] = 0.0, control_guidance_end: float | list[float] = 1.0, control_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, control_mode: int | list[int] | None = None, controlnet_conditioning_scale: float | list[float] = 1.0, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, ip_adapter_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, ip_adapter_image_embeds: list[torch.Tensor] | None = None, negative_ip_adapter_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, negative_ip_adapter_image_embeds: list[torch.Tensor] | None = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L677)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

negative_prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.

true_cfg_scale (`float`, *optional*, defaults to 1.0) : True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and `negative_prompt` is provided.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 7.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

control_guidance_start (`float` or `list[float]`, *optional*, defaults to 0.0) : The percentage of total steps at which the ControlNet starts applying.

control_guidance_end (`float` or `list[float]`, *optional*, defaults to 1.0) : The percentage of total steps at which the ControlNet stops applying.

control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, `list[np.ndarray]`, : `list[list[torch.Tensor]]`, `list[list[np.ndarray]]` or `list[list[PIL.Image.Image]]`): The ControlNet input condition to provide guidance to the `unet` for generation. If the type is specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`, images must be passed as a list such that each element of the list can be correctly batched for input to a single ControlNet.

controlnet_conditioning_scale (`float` or `list[float]`, *optional*, defaults to 1.0) : The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set the corresponding scale as a list.

control_mode (`int` or `list[int]`,, *optional*, defaults to None) : The control mode when applying ControlNet-Union.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt` input argument.

ip_adapter_image : (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.

ip_adapter_image_embeds (`list[torch.Tensor]`, *optional*) : Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.

negative_ip_adapter_image : (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.

negative_ip_adapter_image_embeds (`list[torch.Tensor]`, *optional*) : Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.

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
>>> import torch
>>> from diffusers.utils import load_image
>>> from diffusers import FluxControlNetPipeline
>>> from diffusers import FluxControlNetModel

>>> base_model = "black-forest-labs/FLUX.1-dev"
>>> controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
>>> controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
>>> pipe = FluxControlNetPipeline.from_pretrained(
...     base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
>>> prompt = "A girl in city, 25 years old, cool, futuristic"
>>> image = pipe(
...     prompt,
...     control_image=control_image,
...     control_guidance_start=0.2,
...     control_guidance_end=0.8,
...     controlnet_conditioning_scale=1.0,
...     num_inference_steps=28,
...     guidance_scale=3.5,
... ).images[0]
>>> image.save("flux.png")
```

#### encode_prompt[[diffusers.FluxControlNetPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], prompt_2: str | list[str] | None = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, max_sequence_length: int = 512, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L344)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/flux/pipeline_output.py#L11)

**Parameters:**

images (`list[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be passed to the decoder.

Output class for Flux image generation pipelines.

### Hidream
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/hidream.md

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

# HiDreamImage

[HiDream-I1](https://huggingface.co/HiDream-ai) by HiDream.ai

> [!TIP]
> [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

## Available models

The following models are available for the [HiDreamImagePipeline](/docs/diffusers/v0.43.0/en/api/pipelines/hidream#diffusers.HiDreamImagePipeline) pipeline:

| Model name | Description |
|:---|:---|
| [`HiDream-ai/HiDream-I1-Full`](https://huggingface.co/HiDream-ai/HiDream-I1-Full) | - |
| [`HiDream-ai/HiDream-I1-Dev`](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) | - |
| [`HiDream-ai/HiDream-I1-Fast`](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) | - |

## HiDreamImagePipeline[[diffusers.HiDreamImagePipeline]]

#### diffusers.HiDreamImagePipeline[[diffusers.HiDreamImagePipeline]]

```python
diffusers.HiDreamImagePipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, text_encoder_2: CLIPTextModelWithProjection, tokenizer_2: CLIPTokenizer, text_encoder_3: T5EncoderModel, tokenizer_3: T5Tokenizer, text_encoder_4: LlamaForCausalLM, tokenizer_4: TokenizersBackend, transformer: HiDreamImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hidream_image/pipeline_hidream_image.py#L159)

#### __call__[[diffusers.HiDreamImagePipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, prompt_2: str | list[str] | None = None, prompt_3: str | list[str] | None = None, prompt_4: str | list[str] | None = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float = 5.0, negative_prompt: str | list[str] | None = None, negative_prompt_2: str | list[str] | None = None, negative_prompt_3: str | list[str] | None = None, negative_prompt_4: str | list[str] | None = None, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds_t5: typing.Optional[torch.FloatTensor] = None, prompt_embeds_llama3: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds_t5: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds_llama3: typing.Optional[torch.FloatTensor] = None, pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 128, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hidream_image/pipeline_hidream_image.py#L674)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead.

prompt_3 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is will be used instead.

prompt_4 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_4` and `text_encoder_4`. If not defined, `prompt` is will be used instead.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 3.5) : Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages a model to generate images more aligned with `prompt` at the expense of lower image quality.  Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to the [paper](https://huggingface.co/papers/2210.03142) to learn more.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

negative_prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.

negative_prompt_3 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `negative_prompt` is used in all the text-encoders.

negative_prompt_4 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_4` and `text_encoder_4`. If not defined, `negative_prompt` is used in all the text-encoders.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds_t5 (`torch.FloatTensor`, *optional*) : Pre-generated T5 text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_llama3 (`torch.FloatTensor`, *optional*) : Pre-generated LLaMA3 text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds_t5 (`torch.FloatTensor`, *optional*) : Pre-generated negative T5 text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, embeddings will be generated from `negative_prompt` input argument.

negative_prompt_embeds_llama3 (`torch.FloatTensor`, *optional*) : Pre-generated negative LLaMA3 text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, embeddings will be generated from `negative_prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 128) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.hidream_image.HiDreamImagePipelineOutput` or `tuple`

`~pipelines.hidream_image.HiDreamImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated. images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from transformers import AutoTokenizer, LlamaForCausalLM
>>> from diffusers import HiDreamImagePipeline

>>> tokenizer_4 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
>>> text_encoder_4 = LlamaForCausalLM.from_pretrained(
...     "meta-llama/Meta-Llama-3.1-8B-Instruct",
...     output_hidden_states=True,
...     output_attentions=True,
...     torch_dtype=torch.bfloat16,
... )

>>> pipe = HiDreamImagePipeline.from_pretrained(
...     "HiDream-ai/HiDream-I1-Full",
...     tokenizer_4=tokenizer_4,
...     text_encoder_4=text_encoder_4,
...     torch_dtype=torch.bfloat16,
... )
>>> pipe.enable_model_cpu_offload()

>>> image = pipe(
...     'A cat holding a sign that says "Hi-Dreams.ai".',
...     height=1024,
...     width=1024,
...     guidance_scale=5.0,
...     num_inference_steps=50,
...     generator=torch.Generator("cuda").manual_seed(0),
... ).images[0]
>>> image.save("output.png")
```

## HiDreamImagePipelineOutput[[diffusers.pipelines.hidream_image.pipeline_output.HiDreamImagePipelineOutput]]

#### diffusers.pipelines.hidream_image.pipeline_output.HiDreamImagePipelineOutput[[diffusers.pipelines.hidream_image.pipeline_output.HiDreamImagePipelineOutput]]

```python
diffusers.pipelines.hidream_image.pipeline_output.HiDreamImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hidream_image/pipeline_output.py#L24)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for HiDreamImage pipelines.

### Sana Sprint
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/sana_sprint.md

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

# SANA-Sprint

  

[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](https://huggingface.co/papers/2503.09641) from NVIDIA, MIT HAN Lab, and Hugging Face by Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Enze Xie, Song Han

The abstract from the paper is:

*This paper presents SANA-Sprint, an efficient diffusion model for ultra-fast text-to-image (T2I) generation. SANA-Sprint is built on a pre-trained foundation model and augmented with hybrid distillation, dramatically reducing inference steps from 20 to 1-4. We introduce three key innovations: (1) We propose a training-free approach that transforms a pre-trained flow-matching model for continuous-time consistency distillation (sCM), eliminating costly training from scratch and achieving high training efficiency. Our hybrid distillation strategy combines sCM with latent adversarial distillation (LADD): sCM ensures alignment with the teacher model, while LADD enhances single-step generation fidelity. (2) SANA-Sprint is a unified step-adaptive model that achieves high-quality generation in 1-4 steps, eliminating step-specific training and improving efficiency. (3) We integrate ControlNet with SANA-Sprint for real-time interactive image generation, enabling instant visual feedback for user interaction. SANA-Sprint establishes a new Pareto frontier in speed-quality tradeoffs, achieving state-of-the-art performance with 7.59 FID and 0.74 GenEval in only 1 step — outperforming FLUX-schnell (7.94 FID / 0.71 GenEval) while being 10× faster (0.1s vs 1.1s on H100). It also achieves 0.1s (T2I) and 0.25s (ControlNet) latency for 1024×1024 images on H100, and 0.31s (T2I) on an RTX 4090, showcasing its exceptional efficiency and potential for AI-powered consumer applications (AIPC). Code and pre-trained models will be open-sourced.*

This pipeline was contributed by [lawrence-cj](https://github.com/lawrence-cj), [shuchen Xue](https://github.com/scxue) and [Enze Xie](https://github.com/xieenze). The original codebase can be found [here](https://github.com/NVlabs/Sana). The original weights can be found under [hf.co/Efficient-Large-Model](https://huggingface.co/Efficient-Large-Model/).

Available models:

|                                                                    Model                                                                    | Recommended dtype |
|:-------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
| [`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) | `torch.bfloat16`  |
| [`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) | `torch.bfloat16`  |

Refer to [this](https://huggingface.co/collections/Efficient-Large-Model/sana-sprint-67d6810d65235085b3b17c76) collection for more information.

Note: The recommended dtype mentioned is for the transformer weights. The text encoder must stay in `torch.bfloat16` and VAE weights must stay in `torch.bfloat16` or `torch.float32` for the model to work correctly. Please refer to the inference example below to see how to load the model with the recommended dtype. 

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [SanaSprintPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/sana_sprint#diffusers.SanaSprintPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaTransformer2DModel, SanaSprintPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    dtype=torch.bfloat16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaTransformer2DModel.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    dtype=torch.bfloat16,
)

pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    dtype=torch.bfloat16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("sana.png")
```

## Setting `max_timesteps`

Users can tweak the `max_timesteps` value for experimenting with the visual quality of the generated outputs. The default `max_timesteps` value was obtained with an inference-time search process. For more details about it, check out the paper.

## Image to Image 

The [SanaSprintImg2ImgPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/sana_sprint#diffusers.SanaSprintImg2ImgPipeline) is a pipeline for image-to-image generation. It takes an input image and a prompt, and generates a new image based on the input image and the prompt.

```py
import torch
from diffusers import SanaSprintImg2ImgPipeline
from diffusers.utils.loading_utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)

pipe = SanaSprintImg2ImgPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", 
    dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

image = pipe(
    prompt="a cute pink bear", 
    image=image, 
    strength=0.5, 
    height=832, 
    width=480
).images[0]
image.save("output.png")
```

## SanaSprintPipeline[[diffusers.SanaSprintPipeline]]

#### diffusers.SanaSprintPipeline[[diffusers.SanaSprintPipeline]]

```python
diffusers.SanaSprintPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderDC, transformer: SanaTransformer2DModel, scheduler: DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py#L140)

Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641).

#### __call__[[diffusers.SanaSprintPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, num_inference_steps: int = 2, timesteps: list[int] = None, max_timesteps: float = 1.5708, intermediate_timesteps: float = 1.3, guidance_scale: float = 4.5, num_images_per_prompt: int | None = 1, height: int = 1024, width: int = 1024, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list[str] = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py#L561)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

max_timesteps (`float`, *optional*, defaults to 1.57080) : The maximum timestep value used in the SCM scheduler.

intermediate_timesteps (`float`, *optional*, defaults to 1.3) : The intermediate timestep value used in SCM scheduler (only used when num_inference_steps=2).

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 4.5) : Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages a model to generate images more aligned with `prompt` at the expense of lower image quality.  Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to the [paper](https://huggingface.co/papers/2210.03142) to learn more.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion.IFPipelineOutput` instead of a plain tuple.

attention_kwargs : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

clean_caption (`bool`, *optional*, defaults to `True`) : Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to be installed. If the dependencies are not installed, the embeddings will be created from the raw prompt.

use_resolution_binning (`bool` defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to the requested resolution. Useful for generating non-square images.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to `300`) : Maximum sequence length to use with the `prompt`.

complex_human_instruction (`list[str]`, *optional*) : Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

**Returns:** [SanaPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) or `tuple`

If `return_dict` is `True`, [SanaPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaSprintPipeline

>>> pipe = SanaSprintPipeline.from_pretrained(
...     "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = pipe(prompt="a tiny astronaut hatching from an egg on the moon")[0]
>>> image[0].save("output.png")
```

#### encode_prompt[[diffusers.SanaSprintPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], num_images_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py#L232)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded 

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

Encodes the prompt into text encoder hidden states.

## SanaSprintImg2ImgPipeline[[diffusers.SanaSprintImg2ImgPipeline]]

#### diffusers.SanaSprintImg2ImgPipeline[[diffusers.SanaSprintImg2ImgPipeline]]

```python
diffusers.SanaSprintImg2ImgPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderDC, transformer: SanaTransformer2DModel, scheduler: DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py#L146)

Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641).

#### __call__[[diffusers.SanaSprintImg2ImgPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, num_inference_steps: int = 2, timesteps: list[int] = None, max_timesteps: float = 1.5708, intermediate_timesteps: float = 1.3, guidance_scale: float = 4.5, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, strength: float = 0.6, num_images_per_prompt: int | None = 1, height: int = 1024, width: int = 1024, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list[str] = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py#L629)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

max_timesteps (`float`, *optional*, defaults to 1.57080) : The maximum timestep value used in the SCM scheduler.

intermediate_timesteps (`float`, *optional*, defaults to 1.3) : The intermediate timestep value used in SCM scheduler (only used when num_inference_steps=2).

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. Can also accept image latents as `image`, but if passing latents directly it is not encoded again.

strength (`float`, *optional*, defaults to 0.6) : Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 1 essentially ignores `image`.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion.IFPipelineOutput` instead of a plain tuple.

attention_kwargs : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

clean_caption (`bool`, *optional*, defaults to `True`) : Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to be installed. If the dependencies are not installed, the embeddings will be created from the raw prompt.

use_resolution_binning (`bool` defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to the requested resolution. Useful for generating non-square images.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to `300`) : Maximum sequence length to use with the `prompt`.

complex_human_instruction (`list[str]`, *optional*) : Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

**Returns:** [SanaPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) or `tuple`

If `return_dict` is `True`, [SanaPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaSprintImg2ImgPipeline
>>> from diffusers.utils.loading_utils import load_image

>>> pipe = SanaSprintImg2ImgPipeline.from_pretrained(
...     "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
... )

>>> image = pipe(prompt="a cute pink bear", image=image, strength=0.5, height=832, width=480).images[0]
>>> image[0].save("output.png")
```

#### encode_prompt[[diffusers.SanaSprintImg2ImgPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], num_images_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py#L240)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded 

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

Encodes the prompt into text encoder hidden states.

## SanaPipelineOutput[[diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput]]

#### diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput[[diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput]]

```python
diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for Sana pipelines.

### Flux3
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/flux3.md

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

# FLUX 3 Image Klein

FLUX 3 Image Klein is a 5.6B image generation model from Black Forest Labs: a single transformer ([Flux3Transformer2DModel](/docs/diffusers/v0.43.0/en/api/models/flux3_transformer#diffusers.Flux3Transformer2DModel)) generates images from a text prompt, alone or together with one or more reference images. Text conditioning comes from Qwen3-VL, and images use the FLUX.2 packed autoencoder (`AutoencoderKLFlux2`).

FLUX 3 Image Klein ships as a [modular pipeline](../../modular_diffusers/overview). Get started with a text-to-image generation:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to("cuda")

image = pipe(
    prompt="A red fox in a snow-covered forest clearing at golden hour",
    generator=torch.Generator().manual_seed(0),
    output="images",
)[0]
image.save("fox.png")
```

`prompt` is the only required input: the pipeline carries the recommended sampling settings (50 steps, guidance scale 3.5, scheduler shift 6.93) and generates 1024 x 1024 images by default. It takes one prompt per call; use `num_images_per_prompt` for several images of the same prompt.

As you can see in the example above, loading and running differ a little from a `DiffusionPipeline`. Loading is lazy: [from_pretrained()](/docs/diffusers/v0.43.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.from_pretrained) only builds the pipeline, and you call [load_components()](/docs/diffusers/v0.43.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.load_components) to load the weights. A call returns the pipeline state, and you ask for what you need with `output`: `output="images"` here, or `output=["images", "prompt"]` to also get the caption the prompt upsampler wrote.

> [!TIP]
> If you're working with `ModularPipeline` for the first time, read the [quickstart](../../modular_diffusers/quickstart), and [Coming from DiffusionPipeline](../../modular_diffusers/coming_from_diffusion_pipeline) if you know the standard pipelines. Besides loading and outputs, a few other things differ: [guidance](#guidance-scale), [callbacks](#per-step-callbacks) and [offloading](#memory).

## Prompt upsampling

Prompting matters: FLUX 3 Image Klein takes full scene captions that describe the subject, composition, lighting, and atmosphere, not short prompts; a request sent as typed gives generic output. That is why the pipeline runs the prompt upsampler first on every request, text-to-image and editing alike. It is a separate 9B model from Black Forest Labs ([black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)) that expands `prompt` into such a caption before the text encoder, and a short prompt is all it takes.

The prompt upsampler turns a short request, together with the reference images when editing, into the full scene caption the model takes: medium and style, scene, lighting and camera, each subject with its colors, materials and position in the frame. One model covers text-to-image, editing with reference images and layout with boxes. The caption it wrote comes back as the `prompt` output. In the example above, ask for it alongside the images to see what the image model received:

```python
out = pipe(
    prompt="A red fox in a snow-covered forest clearing at golden hour",
    generator=torch.Generator().manual_seed(0),
    output=["images", "prompt"],
)
print(out["prompt"])
```

```
Full-color outdoor wildlife photograph taken from a low eye-level perspective in a snow-covered forest clearing at golden hour. In the center foreground, a red fox stands in profile facing left, its body turned toward the viewer and its head raised as it looks across the clearing. The fox has dense rust-red winter fur, a white throat and chest, black lower legs, pointed ears with dark tips, and a long bushy tail ending in white. ...
```

The full caption runs to about 240 words; `max_new_tokens` caps its length (3584 by default).

You can print the step's `doc` to see what it takes and what it writes:

```python
print(pipe.blocks.sub_blocks["prompt_upsampler"].doc)
```

```
  Inputs:
      prompt (`str`):
          The edit instruction to expand into the full caption (one per call).
      image (`Image | list`, *optional*):
          The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
      boxes (`list`, *optional*):
          Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max, x_max] in
          0-1: ...
      max_new_tokens (`int`, *optional*, defaults to 3584):
          The most tokens the upsampler may generate for the caption.

  Outputs:
      prompt (`str`):
          The full caption, replacing the request.
```

Like other blocks in the pipeline, the `prompt_upsampler` block also runs on its own. The recommended way is to pop it out of the blocks: the popped block becomes a standalone upsampler, and the remaining blocks become a pipeline that takes the expanded caption as `prompt`:

```python
blocks = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base").blocks
pup_block = blocks.sub_blocks.pop("prompt_upsampler")

pup = pup_block.init_pipeline("black-forest-labs/flux-3-image-klein-base")
pup.load_components(dtype=torch.bfloat16)
pup.to("cuda")
caption = pup(prompt="A red fox in a snow-covered forest clearing at golden hour", output="prompt")

pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to("cuda")
image = pipe(prompt=caption, generator=torch.Generator().manual_seed(0), output="images")[0]
```

This way it can be served as a separate stage, on its own GPU or behind its own endpoint, and a UI can show the caption it wrote and let the user edit it before generating. Any other LLM can take the upsampler's place, with the same system prompts: give it `system_prompts["t2i"]` from the upsampler repository (or `"i2i"`, with the references first in the user message) as the system prompt and the request as the user message, and pass the caption it returns as `prompt` to `pipe`.

## Image-to-image

Pass one or more reference images with `image`. A single reference is kept whole: it is downscaled only above 4 megapixels, padded to the 16-pixel grid, and the generation defaults to its size, so the edit comes back at the size of the input. Several references are each fitted under 1 megapixel and cropped to the grid, and the generation defaults to the first one's size. A given `height`/`width` overrides the default. Describe the *target* image as a complete caption of the reference with your edit folded in, rather than a one-line instruction, or let the [prompt upsampler](#prompt-upsampling) write it from the instruction and the references:

```python
night = pipe(
    prompt="the same scene at night under a full moon",
    image=image,
    generator=torch.Generator().manual_seed(0),
    output=["images", "prompt"],
)
```

With several references, refer to them as the first image, the second image, ... in the order they are passed to `image`; the caption names them `<ref_image_0>`, `<ref_image_1>`, ... for the image model:

```python
combined = pipe(
    prompt="the fox from the first image sitting on the wooden bench from the second image",
    image=[image, bench],
    height=768,
    width=1360,
    output="images",
)[0]
```

Pass `height` and/or `width` to override the derived size.

## Layout control

To place content explicitly, pass `boxes`: for each element an `id`, a short `desc`, and a `bbox` of `[y_min, x_min, y_max, x_max]` in 0-1 (origin top-left), so it is independent of the output size. The upsampler writes the caption around them, citing each element as `<id>`, and appends the table of boxes, yours kept exactly:

```python
boxes = [
    {"id": "balloon_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]},
    {"id": "balloon_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]},
]
out = pipe(prompt="hot-air balloons over a green valley at sunrise", boxes=boxes, output=["images", "prompt"])
```

When editing, a box names its source with `from`: `"target"` with a `tgt_bbox` adds a new element there, `"ref_image_0"` with a `src_bbox` and a `tgt_bbox` moves an element of that reference (equal boxes keep it in place):

```python
boxes = [{"id": "cactus", "desc": "a small potted cactus", "from": "target", "src_bbox": None, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]}]
out = pipe(prompt="add a small potted cactus in the marked spot", image=mug, boxes=boxes, output=["images", "prompt"])

boxes = [{"id": "knight", "desc": "the knight figure", "from": "ref_image_0", "src_bbox": [0.50, 0.15, 0.85, 0.35], "tgt_bbox": [0.19, 0.06, 0.54, 0.26]}]
out = pipe(prompt="move the knight up and to the left, keep the rest unchanged", image=knight, boxes=boxes, output=["images", "prompt"])
```

The caption the image model reads has the elements tagged `<id>` in the text and the table appended as JSON, with the boxes on a 0-1000 canvas; print the `prompt` output to see it. A caption written in that form by hand goes to the pipeline without the step, as in [Prompt upsampling](#prompt-upsampling).

## Guidance scale

The pipeline applies classifier-free guidance through its `guider` component and does not take a `guidance_scale` argument at call time. The default guider is [ClassifierFreeGuidance](/docs/diffusers/v0.43.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance) with `guidance_scale=3.5`, the recommended setting for both workflows. To change it, create an updated copy of the guider:

```python
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

See the [guiders guide](../../using-diffusers/guiders) to switch to a different guidance method.

## Step-distilled model

The step-distilled checkpoint generates in 16 steps without classifier-free guidance. Its transformer is a flow map: at every step it is told the current timestep and the one the step jumps to. It loads like the base model (the repo's `modular_model_index.json` resolves to [Flux3DistilledModularPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/flux3#diffusers.Flux3DistilledModularPipeline) and [Flux3DistilledAutoBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/flux3#diffusers.Flux3DistilledAutoBlocks)); the pipeline has no `guider` and does not take `negative_prompt`:

```python
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="images")[0]
```

It serves the same two workflows as the base model. `num_inference_steps` defaults to 16; the model is trained for 4 to 16 steps.

## Reproducible generation

A `generator` fixes the noise the pipeline draws (see [Reproducibility](../../using-diffusers/reusing_seeds)). To reuse noise drawn elsewhere, pass it as `latents` with shape `(batch_size, 128, height // 16, width // 16)`. For example, the [BFL implementation](https://github.com/black-forest-labs/flux-3) draws its noise as image tokens on the CPU and casts it to bf16; this reproduces its sample for a given seed:

```python
height, width = 1024, 1024
h, w = height // 16, width // 16

noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
latents = noise.T.reshape(1, 128, h, w)

image = pipe(prompt=prompt, latents=latents, height=height, width=width, output="images")[0]
```

## Workflows

The pipeline serves two workflows, `text2image` and `image_conditioned`; the one that runs is picked by the inputs (`image` selects image-to-image). [from_pretrained()](/docs/diffusers/v0.43.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.from_pretrained) can also build a pipeline for one workflow with the `workflow` argument, which then only expects and loads the components that workflow needs:

```python
from diffusers.modular_pipelines import Flux3AutoBlocks

print(Flux3AutoBlocks().available_workflows)
# ["text2image", "image_conditioned"]

t2i = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base", workflow="text2image")
print(t2i.blocks.doc)
```

## Per-step callbacks

To use an existing callback function, insert a block into the denoising loop. The [callbacks guide](../../using-diffusers/callback#modular-pipelines) has a simple example with the `CallbackOnStepEnd` block and the `walk_denoise_loops` helper:

```python
blocks = Flux3AutoBlocks()  # or Flux3DistilledAutoBlocks
for loop in walk_denoise_loops(blocks):
    loop.sub_blocks["callback"] = CallbackOnStepEnd()
pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")

image = pipe(prompt=caption, callback_on_step_end=my_callback, output="images")[0]
```

To stop early, raise from the callback; there is no `interrupt` flag.

## Memory

Loading everything in bf16 takes about 39 GB of weights: the transformer is 5.5B parameters, the Qwen3-VL-4B text encoder 4B and the [prompt upsampler](#prompt-upsampling) 9B (about 19 GB on its own), with the image VAE in fp32. A pipeline built without the upsampler step (see there) loads about 20 GB. For GPUs with less memory, enable component offloading; the manager moves components between the CPU and the GPU as each step needs them, the upsampler included:

```python
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base", components_manager=manager)
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
manager.enable_auto_cpu_offload(device="cuda")
```

## Single-file loading

[Flux3Transformer2DModel](/docs/diffusers/v0.43.0/en/api/models/flux3_transformer#diffusers.Flux3Transformer2DModel) loads the transformer straight from a Black Forest Labs model package with [from_single_file()](/docs/diffusers/v0.43.0/en/api/loaders/single_file#diffusers.loaders.FromOriginalModelMixin.from_single_file). The package `config.json` is not read; pass the diffusers config with `config` and `subfolder`.

```python
import torch
from diffusers import Flux3Transformer2DModel, ModularPipeline

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/model.safetensors",
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    dtype=torch.bfloat16,
)

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.update_components(transformer=transformer)
pipe.to("cuda")
```

Load the other components by name first: [load_components()](/docs/diffusers/v0.43.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.load_components) without `names` would also load the transformer from the pipeline repo.

Loading the FP8r and NVFP4 single-file checkpoints are also supported. Both load through the same call with a [BFLQuantizationConfig](/docs/diffusers/v0.43.0/en/api/quantization#diffusers.BFLQuantizationConfig). See the [BFL quantization guide](../../quantization/bfl) for the kernel requirements of each scheme.

```python
from diffusers import BFLQuantizationConfig

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/variants/nvfp4/model.safetensors",
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
```

When loading the step-distilled checkpoints, we need their
configs (`flow_map=True`):

```python
transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/variants/nvfp4/model.safetensors",
    config="black-forest-labs/flux-3-image-klein",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
```

## Flux3ModularPipeline[[diffusers.Flux3ModularPipeline]]

#### diffusers.Flux3ModularPipeline[[diffusers.Flux3ModularPipeline]]

```python
diffusers.Flux3ModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/flux3/modular_pipeline.py#L24)

A ModularPipeline for FLUX.3.

## Flux3AutoBlocks[[diffusers.Flux3AutoBlocks]]

#### diffusers.Flux3AutoBlocks[[diffusers.Flux3AutoBlocks]]

```python
diffusers.Flux3AutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/flux3/modular_blocks_flux3.py#L339)

Auto blocks that perform text-to-image and image-to-image generation using FLUX.3. `prompt` goes through the prompt
upsampler first; pop the `prompt_upsampler` step to encode it as given.

Supported workflows:
- `text2image`: requires `prompt`
- `image_conditioned`: requires `prompt`, `image`

Components:
prompt_upsampler (`Qwen3_5ForConditionalGeneration`): The prompt upsampler. prompt_upsampler_processor
(`AutoProcessor`): Its processor. text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
(`AutoProcessor`) image_processor (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`) transformer
(`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`)

Inputs:
prompt (`str`):
The edit instruction to expand into the full caption (one per call).
image (`Image | list`, *optional*):
The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
boxes (`list`, *optional*):
Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max,
x_max] in 0-1: 'from': 'target' with `tgt_bbox` adds an element there, 'from': 'ref_image_0', ... with
`src_bbox` and `tgt_bbox` moves one from a reference image.
max_new_tokens (`int`, *optional*, defaults to 3584):
The most tokens the upsampler may generate for the caption.
negative_prompt (`str`, *optional*):
The prompt not to guide the generation
max_sequence_length (`int`, *optional*, defaults to 8192):
Maximum number of prompt tokens; longer prompts are truncated.
text_encoder_out_layers (`tuple`, *optional*, defaults to (4, 8, 12, 16, 20, 24, 28, 32)):
Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings
num_images_per_prompt (`int`, *optional*, defaults to 1):
The number of images to generate per prompt.
image_cond_latents (`list`, *optional*):
Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
per image
height (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The height in pixels of the generated image; defaults to the first reference image's own height
width (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The width in pixels of the generated image; defaults to the first reference image's own width
reference_crop (`list | NoneType`, *optional*):
Per reference, the box of the reference inside its padded canvas or None, from the preprocess step
latents (`Tensor`):
Pre-generated noisy latents for image generation.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`):
The number of denoising steps.
timesteps (`Tensor`):
Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
sigmas (`list`, *optional*):
Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
**denoiser_input_fields (`None`, *optional*):
conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
images (`list`):
Generated images.
latents (`Tensor`):
Denoised latents.

## Flux3DistilledAutoBlocks[[diffusers.Flux3DistilledAutoBlocks]]

#### diffusers.Flux3DistilledAutoBlocks[[diffusers.Flux3DistilledAutoBlocks]]

```python
diffusers.Flux3DistilledAutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/flux3/modular_blocks_flux3_distilled.py#L258)

Auto blocks that perform text-to-image and image-to-image generation using the step-distilled FLUX.3 model (16
steps, no classifier-free guidance). `prompt` goes through the prompt upsampler first; pop the `prompt_upsampler`
step to encode it as given.

Supported workflows:
- `text2image`: requires `prompt`
- `image_conditioned`: requires `prompt`, `image`

Components:
prompt_upsampler (`Qwen3_5ForConditionalGeneration`): The prompt upsampler. prompt_upsampler_processor
(`AutoProcessor`): Its processor. text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
(`AutoProcessor`) image_processor (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`) transformer
(`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

Inputs:
prompt (`str`):
The edit instruction to expand into the full caption (one per call).
image (`Image | list`, *optional*):
The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
boxes (`list`, *optional*):
Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max,
x_max] in 0-1: 'from': 'target' with `tgt_bbox` adds an element there, 'from': 'ref_image_0', ... with
`src_bbox` and `tgt_bbox` moves one from a reference image.
max_new_tokens (`int`, *optional*, defaults to 3584):
The most tokens the upsampler may generate for the caption.
max_sequence_length (`int`, *optional*, defaults to 8192):
Maximum number of prompt tokens; longer prompts are truncated.
text_encoder_out_layers (`tuple`, *optional*, defaults to (4, 8, 12, 16, 20, 24, 28, 32)):
Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings
num_images_per_prompt (`int`, *optional*, defaults to 1):
The number of images to generate per prompt.
image_cond_latents (`list`, *optional*):
Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
per image
height (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The height in pixels of the generated image; defaults to the first reference image's own height
width (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The width in pixels of the generated image; defaults to the first reference image's own width
reference_crop (`list | NoneType`, *optional*):
Per reference, the box of the reference inside its padded canvas or None, from the preprocess step
latents (`Tensor`):
Pre-generated noisy latents for image generation.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`):
The number of denoising steps.
timesteps (`Tensor`):
Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
sigmas (`list`, *optional*):
Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
**denoiser_input_fields (`None`, *optional*):
conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
images (`list`):
Generated images.
latents (`Tensor`):
Denoised latents.

## Flux3DistilledModularPipeline[[diffusers.Flux3DistilledModularPipeline]]

#### diffusers.Flux3DistilledModularPipeline[[diffusers.Flux3DistilledModularPipeline]]

```python
diffusers.Flux3DistilledModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/flux3/modular_pipeline.py#L52)

A ModularPipeline for the step-distilled FLUX.3 model (16 steps, no classifier-free guidance).

### Anima
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/anima.md

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
