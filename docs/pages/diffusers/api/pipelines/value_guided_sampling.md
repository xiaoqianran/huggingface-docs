# Value-guided planning

> [!WARNING]
> 🧪 This is an experimental pipeline for reinforcement learning!

This pipeline is based on the [Planning with Diffusion for Flexible Behavior Synthesis](https://huggingface.co/papers/2205.09991) paper by Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine.

The abstract from the paper is:

*Model-based reinforcement learning methods often use learning only for the purpose of estimating an approximate dynamics model, offloading the rest of the decision-making work to classical trajectory optimizers. While conceptually simple, this combination has a number of empirical shortcomings, suggesting that learned models may not be well-suited to standard trajectory optimization. In this paper, we consider what it would look like to fold as much of the trajectory optimization pipeline as possible into the modeling problem, such that sampling from the model and planning with it become nearly identical. The core of our technical approach lies in a diffusion probabilistic model that plans by iteratively denoising trajectories. We show how classifier-guided sampling and image inpainting can be reinterpreted as coherent planning strategies, explore the unusual and useful properties of diffusion-based planning methods, and demonstrate the effectiveness of our framework in control settings that emphasize long-horizon decision-making and test-time flexibility.*

You can find additional information about the model on the [project page](https://diffusion-planning.github.io/), the [original codebase](https://github.com/jannerm/diffuser), or try it out in a demo [notebook](https://colab.research.google.com/drive/1rXm8CX4ZdN5qivjJ2lhwhkOmt_m0CvU0#scrollTo=6HXJvhyqcITc&uniqifier=1).

The script to run the model is available [here](https://github.com/huggingface/diffusers/tree/main/examples/reinforcement_learning).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## ValueGuidedRLPipeline[[diffusers.experimental.ValueGuidedRLPipeline]]

#### diffusers.experimental.ValueGuidedRLPipeline[[diffusers.experimental.ValueGuidedRLPipeline]]

```python
diffusers.experimental.ValueGuidedRLPipeline(value_function: UNet1DModel, unet: UNet1DModel, scheduler: DDPMScheduler, env)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/experimental/rl/value_guided_sampling.py#L25)

**Parameters:**

value_function ([UNet1DModel](/docs/diffusers/v0.40.0/en/api/models/unet#diffusers.UNet1DModel)) : A specialized UNet for fine-tuning trajectories base on reward.

unet ([UNet1DModel](/docs/diffusers/v0.40.0/en/api/models/unet#diffusers.UNet1DModel)) : UNet architecture to denoise the encoded trajectories.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded trajectories. Default for this application is [DDPMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddpm#diffusers.DDPMScheduler).

env () : An environment following the OpenAI gym API to act in. For now only Hopper has pretrained models.

Pipeline for value-guided sampling from a diffusion model trained to predict sequences of states.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

### Nucleusmoe Image
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/nucleusmoe_image.md

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

```python
diffusers.NucleusMoEImagePipeline(transformer: NucleusMoEImageTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py#L132)

**Parameters:**

transformer (`NucleusMoEImageTransformer2DModel`) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLQwenImage](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl_qwenimage#diffusers.AutoencoderKLQwenImage)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen3VLForConditionalGeneration`) : Text encoder for computing prompt embeddings.

processor (`Qwen3VLProcessor`) : Processor for tokenizing text inputs.

Pipeline for text-to-image generation using NucleusMoE.

This pipeline uses a single-stream DiT with Mixture-of-Experts feed-forward layers, cross-attention to a Qwen3-VL
text encoder, and a flow-matching Euler discrete scheduler.

#### __call__[[diffusers.NucleusMoEImagePipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, guidance_scale: float = 4.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, num_images_per_prompt: int = 1, max_sequence_length: int | None = None, return_index: int | None = None, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int, dict], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py#L379)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, an empty string is used when `true_cfg_scale > 1`.

guidance_scale (`float`, *optional*, defaults to 4.0) : Classifier-free guidance scale. Values greater than 1 enable CFG.

return_index (`int`, *optional*) : Layer index of the text encoder output to use for the prompt embeddings.

height (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to `self.default_sample_size * self.vae_scale_factor`) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps.

sigmas (`list[float]`, *optional*) : Custom sigmas for the denoising schedule. If not defined, a linear schedule is used.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of torch generators to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents to be used as inputs for image generation.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for pre-generated text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for pre-generated negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `"pil"`, `"np"`, or `"latent"`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `NucleusMoEImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : Kwargs passed to the attention processor.

callback_on_step_end (`Callable`, *optional*) : A function called at the end of each denoising step.

callback_on_step_end_tensor_inputs (`list`, *optional*) : Tensor inputs for the `callback_on_step_end` function.

max_sequence_length (`int`, defaults to 512) : Maximum sequence length for the text prompt.

**Returns:** `NucleusMoEImagePipelineOutput` or `tuple`

`NucleusMoEImagePipelineOutput` if `return_dict` is True, otherwise a `tuple` where the first element
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

#### encode_prompt[[diffusers.NucleusMoEImagePipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str] = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int | None = None, return_index: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py#L187)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to encode.

device (`torch.device`, *optional*) : Torch device for the resulting tensors.

num_images_per_prompt (`int`, defaults to 1) : Number of images to generate per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Skips encoding when provided.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for pre-generated embeddings.

max_sequence_length (`int`, defaults to 1024) : Maximum token length for the encoded prompt.

Encode text prompt(s) into embeddings using the Qwen3-VL text encoder.

## NucleusMoEImagePipelineOutput[[diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput]]

#### diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput[[diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput]]

```python
diffusers.pipelines.nucleusmoe_image.pipeline_output.NucleusMoEImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/nucleusmoe_image/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for NucleusMoE Image pipelines.

### ControlNet with Stable Diffusion XL
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/controlnet_sdxl.md
