# AuraFlow

AuraFlow is inspired by [Stable Diffusion 3](../pipelines/stable_diffusion/stable_diffusion_3) and is by far the largest text-to-image generation model that comes with an Apache 2.0 license. This model achieves state-of-the-art results on the [GenEval](https://github.com/djghosh13/geneval) benchmark.

It was developed by the Fal team and more details about it can be found in [this blog post](https://blog.fal.ai/auraflow/).

> [!TIP]
> AuraFlow can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details.

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [AuraFlowPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/aura_flow#diffusers.AuraFlowPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, AuraFlowTransformer2DModel, AuraFlowPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "fal/AuraFlow",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = AuraFlowTransformer2DModel.from_pretrained(
    "fal/AuraFlow",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = AuraFlowPipeline.from_pretrained(
    "fal/AuraFlow",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("auraflow.png")
```

Loading [GGUF checkpoints](https://huggingface.co/docs/diffusers/quantization/gguf) are also supported:

```py
import torch
from diffusers import (
    AuraFlowPipeline,
    GGUFQuantizationConfig,
    AuraFlowTransformer2DModel,
)

transformer = AuraFlowTransformer2DModel.from_single_file(
    "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipeline = AuraFlowPipeline.from_pretrained(
    "fal/AuraFlow-v0.3",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

prompt = "a cute pony in a field of flowers"
image = pipeline(prompt).images[0]
image.save("auraflow.png")
```

## Support for `torch.compile()`

AuraFlow can be compiled with `torch.compile()` to speed up inference latency even for different resolutions. First, install PyTorch nightly following the instructions from [here](https://pytorch.org/). The snippet below shows the changes needed to enable this:

```diff
+ torch.fx.experimental._config.use_duck_shape = False
+ pipeline.transformer = torch.compile(
    pipeline.transformer, fullgraph=True, dynamic=True
)
```

Specifying `use_duck_shape` to be `False` instructs the compiler if it should use the same symbolic variable to represent input sizes that are the same. For more details, check out [this comment](https://github.com/huggingface/diffusers/pull/11327#discussion_r2047659790).

This enables from 100% (on low resolutions) to a 30% (on 1536x1536 resolution) speed improvements.

Thanks to [AstraliteHeart](https://github.com/huggingface/diffusers/pull/11297/) who helped us rewrite the [AuraFlowTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/aura_flow_transformer2d#diffusers.AuraFlowTransformer2DModel) class so that the above works for different resolutions ([PR](https://github.com/huggingface/diffusers/pull/11297/)).

## AuraFlowPipeline[[diffusers.AuraFlowPipeline]]

#### diffusers.AuraFlowPipeline[[diffusers.AuraFlowPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py#L123)

__call__diffusers.AuraFlowPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py#L428[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 3.5"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "height", "val": ": int | None = 1024"}, {"name": "width", "val": ": int | None = 1024"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "max_sequence_length", "val": ": int = 256"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for best results.
- **width** (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
  `num_inference_steps` and `timesteps` must be `None`.
- **guidance_scale** (`float`, *optional*, defaults to 5.0) --
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
- **latents** (`torch.FloatTensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for text embeddings.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for negative text embeddings.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int` defaults to 256) -- Maximum sequence length to use with the `prompt`.0[ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import AuraFlowPipeline

>>> pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> image = pipe(prompt).images[0]
>>> image.save("aura_flow.png")
```

**Parameters:**

tokenizer (`T5TokenizerFast`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

text_encoder (`T5EncoderModel`) : Frozen text-encoder. AuraFlow uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the [EleutherAI/pile-t5-xl](https://huggingface.co/EleutherAI/pile-t5-xl) variant.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

transformer ([AuraFlowTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/aura_flow_transformer2d#diffusers.AuraFlowTransformer2DModel)) : Conditional Transformer (MMDiT and DiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

`[ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple``

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### encode_prompt[[diffusers.AuraFlowPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py#L232)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : whether to use classifier free guidance or not

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

max_sequence_length (`int`, defaults to 256) : Maximum sequence length to use for the prompt.

lora_scale (`float`, *optional*) : A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.

### Nucleusmoe Image
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/nucleusmoe_image.md

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

# NucleusMoE-Image

[NucleusMoE-Image](https://huggingface.co/NucleusAI/NucleusMoE-Image) is a text-to-image model that pairs a single-stream DiT with Mixture-of-Experts feed-forward layers, cross-attention to a Qwen3-VL text encoder, and a flow-matching Euler discrete scheduler.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## NucleusMoEImagePipeline[[diffusers.NucleusMoEImagePipeline]]

#### diffusers.NucleusMoEImagePipeline[[diffusers.NucleusMoEImagePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py#L132)

Pipeline for text-to-image generation using NucleusMoE.

This pipeline uses a single-stream DiT with Mixture-of-Experts feed-forward layers, cross-attention to a Qwen3-VL
text encoder, and a flow-matching Euler discrete scheduler.

__call__diffusers.NucleusMoEImagePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py#L379[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "max_sequence_length", "val": ": int | None = None"}, {"name": "return_index", "val": ": int | None = None"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, dict], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, an empty string is used when
  `true_cfg_scale > 1`.
- **guidance_scale** (`float`, *optional*, defaults to 4.0) --
  Classifier-free guidance scale. Values greater than 1 enable CFG.
- **return_index** (`int`, *optional*) --
  Layer index of the text encoder output to use for the prompt embeddings.
- **height** (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`) --
  The height in pixels of the generated image.
- **width** (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`) --
  The width in pixels of the generated image.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas for the denoising schedule. If not defined, a linear schedule is used.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of torch generators to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents to be used as inputs for image generation.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings.
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for pre-generated text embeddings.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for pre-generated negative text embeddings.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `"pil"`, `"np"`, or `"latent"`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `NucleusMoEImagePipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  Kwargs passed to the attention processor.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function called at the end of each denoising step.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  Tensor inputs for the `callback_on_step_end` function.
- **max_sequence_length** (`int`, defaults to 512) --
  Maximum sequence length for the text prompt.0`NucleusMoEImagePipelineOutput` or `tuple``NucleusMoEImagePipelineOutput` if `return_dict` is True, otherwise a `tuple` where the first element
is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import NucleusMoEImagePipeline

>>> pipe = NucleusMoEImagePipeline.from_pretrained("NucleusAI/NucleusMoE-Image", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> image = pipe(prompt, num_inference_steps=50).images[0]
>>> image.save("nucleus_moe.png")
```

**Parameters:**

transformer (`NucleusMoEImageTransformer2DModel`) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLQwenImage](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_qwenimage#diffusers.AutoencoderKLQwenImage)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen3VLForConditionalGeneration`) : Text encoder for computing prompt embeddings.

processor (`Qwen3VLProcessor`) : Processor for tokenizing text inputs.

**Returns:**

``NucleusMoEImagePipelineOutput` or `tuple``

`NucleusMoEImagePipelineOutput` if `return_dict` is True, otherwise a `tuple` where the first element
is a list with the generated images.
#### encode_prompt[[diffusers.NucleusMoEImagePipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py#L187)

Encode text prompt(s) into embeddings using the Qwen3-VL text encoder.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to encode.

device (`torch.device`, *optional*) : Torch device for the resulting tensors.

num_images_per_prompt (`int`, defaults to 1) : Number of images to generate per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Skips encoding when provided.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for pre-generated embeddings.

max_sequence_length (`int`, defaults to 1024) : Maximum token length for the encoded prompt.

## NucleusMoEImagePipelineOutput[[diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput]]

#### diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput[[diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_output.py#L10)

Output class for NucleusMoE Image pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Flux2
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/flux2.md
