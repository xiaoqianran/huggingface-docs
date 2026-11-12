# PAN 2

PAN2 is a foundation world model that simulates how a scene evolves under a time-grounded plan of actions and camera movements. Its 29B generative simulator is a dual-stream multimodal diffusion transformer, integrated here; the [reasoning backbone](https://huggingface.co/IFM/PAN2#reasoning-backbone) that writes its prompts is not.

PAN 2 is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only. [PAN2Blocks](/docs/diffusers/v0.43.0/en/api/pipelines/pan2#diffusers.PAN2Blocks) supports two workflows:

| Workflow | Inputs |
|---|---|
| `t2v` | `prompt` |
| `i2v` | `prompt`, `image` |

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video, load_image

pipe = ModularPipeline.from_pretrained("IFM/PAN2")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A red fox trots through fresh snow in a pine forest at dawn."

video = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="videos")[0]
export_to_video(video, "pan2_t2v.mp4", fps=24)

image = load_image("path/to/first_frame.png")
video = pipe(prompt=prompt, image=image, generator=torch.Generator().manual_seed(0), output="videos")[0]
export_to_video(video, "pan2_i2v.mp4", fps=24)
```

Classifier-free guidance is configured on the guider (`guidance_scale=2.5` by default):

```py
pipe.update_components(guider=pipe.guider.new(guidance_scale=5.0))
```

Every call takes a single prompt, which is encoded without padding. The transformer therefore never needs an attention mask, and every [attention backend](../../optimization/attention_backends) is available:

```py
pipe.transformer.set_attention_backend("flash_hub")  # FlashAttention-2
```

## Safety checker

PAN 2 runs guardrails on the prompt before generation and on every generated video after it, and a blocked prompt or video raises a `ValueError`. The checks are on by default, and `load_components()` loads the guardrails, which need the [PAN 2 guardrail](https://github.com/ifm-ai/pan2-guardrail) package:

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail
```

## Context parallelism

[PAN2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/pan2_transformer3d#diffusers.PAN2Transformer3DModel) supports [context parallelism](../../optimization/context_parallelism). The text and video token counts are usually not divisible by the number of devices (the number of text tokens depends on the prompt), so use `ulysses_anything` or `ring_anything` for inference. Plain `ulysses_degree` / `ring_degree`, for example for training, requires both token counts to be divisible by the context parallel size.

```py
# torchrun --nproc_per_node=2 pan2_context_parallel.py
import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig, ModularPipeline

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

pipe = ModularPipeline.from_pretrained("IFM/PAN2")
pipe.load_components(dtype=torch.bfloat16)
pipe.to(f"cuda:{rank}")
pipe.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2, ulysses_anything=True))

video = pipe(prompt="A red fox trots through fresh snow.", generator=torch.Generator().manual_seed(0), output="videos")[0]
dist.destroy_process_group()
```

## PAN2ModularPipeline[[diffusers.PAN2ModularPipeline]]

#### diffusers.PAN2ModularPipeline[[diffusers.PAN2ModularPipeline]]

```python
diffusers.PAN2ModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pan2/modular_pipeline.py#L42)

A ModularPipeline for PAN 2: text-to-video, and image-to-video from a first frame.

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("IFM/PAN2")
pipe.load_components(dtype=torch.bfloat16)
```

#### check_safety_checker_loaded[[diffusers.PAN2ModularPipeline.check_safety_checker_loaded]]

```python
check_safety_checker_loaded()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pan2/modular_pipeline.py#L103)

Raise if the checks are on but the PAN 2 guardrail is not loaded.

#### disable_safety_checker[[diffusers.PAN2ModularPipeline.disable_safety_checker]]

```python
disable_safety_checker()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pan2/modular_pipeline.py#L118)

Turn off the prompt and video checks, at your own risk.

#### enable_safety_checker[[diffusers.PAN2ModularPipeline.enable_safety_checker]]

```python
enable_safety_checker()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pan2/modular_pipeline.py#L97)

Check the prompt and the generated videos with the PAN 2 guardrail.

## PAN2Blocks[[diffusers.PAN2Blocks]]

#### diffusers.PAN2Blocks[[diffusers.PAN2Blocks]]

```python
diffusers.PAN2Blocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pan2/modular_blocks_pan2.py#L251)

Modular pipeline blocks for text-to-video and image-to-video generation with PAN 2.

Supported workflows:
- `t2v`: requires `prompt`
- `i2v`: requires `prompt`, `image`

Components:
text_encoder (`Qwen3_5TextModel`) tokenizer (`Qwen2Tokenizer`) safety_checker (`PAN2SafetyChecker`) vae
(`AutoencoderKLWan`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
transformer (`PAN2Transformer3DModel`) guider (`ClassifierFreeGuidance`)

Configs:
enable_safety_checker (default: True)

Inputs:
prompt (`str`):
The prompt to guide generation, a single string.
negative_prompt (`str`, *optional*):
The prompt not to guide generation, a single string. When guidance needs one and none (or an empty one)
is passed, PAN 2's default negative prompt is used, its time ranges set to the clip's duration.
max_sequence_length (`int`, *optional*, defaults to 1536):
Maximum sequence length for prompt encoding.
num_frames (`int`, *optional*, defaults to 313):
Number of video frames to generate, of the form `4 * n + 1`.
fps (`float`, *optional*, defaults to 24.0):
The frame rate of the video, which sets its duration in the default negative prompt.
image (`Image`, *optional*):
The first frame, a single image.
height (`int`, *optional*):
The height in pixels of the generated video, 704 by default.
width (`int`, *optional*):
The width in pixels of the generated video, 1248 by default.
num_videos_per_prompt (`int`, *optional*, defaults to 1):
The number of videos to generate per prompt.
image_latents (`Tensor`, *optional*):
The normalized first-frame latents. Can be generated from the vae_encoder step.
num_inference_steps (`int`):
The number of denoising steps.
sigmas (`list`, *optional*):
Custom sigmas for the denoising process.
latents (`Tensor`):
Pre-generated noisy latents to use as inputs for video generation.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
output_type (`str`, *optional*, defaults to np):
Output format: 'pil', 'np', 'pt'.

Outputs:
videos (`list`):
The generated videos.

### Qwenimage
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/qwenimage.md

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

# QwenImage

  

Qwen-Image from the Qwen team is an image generation foundation model in the Qwen series that achieves significant advances in complex text rendering and precise image editing. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.

Qwen-Image comes in the following variants:

| model type | model id |
|:----------:|:--------:|
| Qwen-Image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-Edit | [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| Qwen-Image-Edit Plus | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |

> [!TIP]
> See the [Caching](../../optimization/cache) guide to speed up inference by storing and reusing intermediate outputs.

## LoRA for faster inference

Use a LoRA from `lightx2v/Qwen-Image-Lightning` to speed up inference by reducing the
number of steps. Refer to the code snippet below:

Code

```py
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch 
import math

ckpt_id = "Qwen/Qwen-Image"

# From
# https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    ckpt_id, scheduler=scheduler, dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
)

prompt = "a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition."
negative_prompt = " "
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=8,
    true_cfg_scale=1.0,
    generator=torch.manual_seed(0),
).images[0]
image.save("qwen_fewsteps.png")
```

> [!TIP]
> The `guidance_scale` parameter in the pipeline is there to support future guidance-distilled models when they come up. Note that passing `guidance_scale` to the pipeline is ineffective. To enable classifier-free guidance, please pass `true_cfg_scale` and `negative_prompt` (even an empty negative prompt like " ") should enable classifier-free guidance computations.

## Multi-image reference with QwenImageEditPlusPipeline

With [QwenImageEditPlusPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/qwenimage#diffusers.QwenImageEditPlusPipeline), one can provide multiple images as input reference.

```py
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"

image_1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/grumpy.jpg")
image_2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peng.png")
image = pipe(
    image=[image_1, image_2],
    prompt='''put the penguin and the cat at a game show called "Qwen Edit Plus Games"''',
    num_inference_steps=50
).images[0]
```

## Performance

### torch.compile

Using `torch.compile` on the transformer provides ~2.4x speedup (A100 80GB: 4.70s → 1.93s):

```python
import torch
from diffusers import QwenImagePipeline

pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", dtype=torch.bfloat16).to("cuda")  # or "mps", "xpu", "cpu"
pipe.transformer = torch.compile(pipe.transformer)

# First call triggers compilation (~7s overhead)
# Subsequent calls run at ~2.4x faster
image = pipe("a cat", num_inference_steps=50).images[0]
```

### Batched Inference with Variable-Length Prompts

When using classifier-free guidance (CFG) with prompts of different lengths, the pipeline properly handles padding through attention masking. This ensures padding tokens do not influence the generated output.

```python
# CFG with different prompt lengths works correctly
image = pipe(
    prompt="A cat",
    negative_prompt="blurry, low quality, distorted",
    true_cfg_scale=3.5,
    num_inference_steps=50,
).images[0]
```

For detailed benchmark scripts and results, see [this gist](https://gist.github.com/cdutr/bea337e4680268168550292d7819dc2f).

## QwenImagePipeline[[diffusers.QwenImagePipeline]]

#### diffusers.QwenImagePipeline[[diffusers.QwenImagePipeline]]

```python
diffusers.QwenImagePipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L132)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The QwenImage pipeline for text-to-image generation.

#### __call__[[diffusers.QwenImagePipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L408)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

true_cfg_scale (`float`, *optional*, defaults to 1.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImagePipeline

>>> pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, num_inference_steps=50).images[0]
>>> image.save("qwenimage.png")
```

#### encode_prompt[[diffusers.QwenImagePipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L225)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageImg2ImgPipeline[[diffusers.QwenImageImg2ImgPipeline]]

#### diffusers.QwenImageImg2ImgPipeline[[diffusers.QwenImageImg2ImgPipeline]]

```python
diffusers.QwenImageImg2ImgPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L134)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The QwenImage pipeline for text-to-image generation.

#### __call__[[diffusers.QwenImageImg2ImgPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, height: int | None = None, width: int | None = None, strength: float = 0.6, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L482)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

true_cfg_scale (`float`, *optional*, defaults to 1.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

strength (`float`, *optional*, defaults to 1.0) : Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 1 essentially ignores `image`.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImageImg2ImgPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageImg2ImgPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
>>> pipe = pipe.to("cuda")
>>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
>>> init_image = load_image(url).resize((1024, 1024))
>>> prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney"
>>> images = pipe(prompt=prompt, negative_prompt=" ", image=init_image, strength=0.95).images[0]
>>> images.save("qwenimage_img2img.png")
```

#### encode_prompt[[diffusers.QwenImageImg2ImgPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L268)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageInpaintPipeline[[diffusers.QwenImageInpaintPipeline]]

#### diffusers.QwenImageInpaintPipeline[[diffusers.QwenImageInpaintPipeline]]

```python
diffusers.QwenImageInpaintPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L137)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The QwenImage pipeline for text-to-image generation.

#### __call__[[diffusers.QwenImageInpaintPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, mask_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, masked_image_latents: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, height: int | None = None, width: int | None = None, padding_mask_crop: int | None = None, strength: float = 0.6, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L592)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

true_cfg_scale (`float`, *optional*, defaults to 1.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B, H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W, 1)`, or `(H, W)`.

masked_image_latents (`torch.Tensor`, `list[torch.Tensor]`) : `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask latents tensor will be generated by `mask_image`.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

padding_mask_crop (`int`, *optional*, defaults to `None`) : The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region with the same aspect ration of the image and contains all masked area, and then expand that area based on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before resizing to the original image size for inpainting. This is useful when the masked area is small while the image is large and contain information irrelevant for inpainting, such as background.

strength (`float`, *optional*, defaults to 1.0) : Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 1 essentially ignores `image`.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImageInpaintPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageInpaintPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
>>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
>>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
>>> source = load_image(img_url)
>>> mask = load_image(mask_url)
>>> image = pipe(prompt=prompt, negative_prompt=" ", image=source, mask_image=mask, strength=0.85).images[0]
>>> image.save("qwenimage_inpainting.png")
```

#### encode_prompt[[diffusers.QwenImageInpaintPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L279)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageEditPipeline[[diffusers.QwenImageEditPipeline]]

#### diffusers.QwenImageEditPipeline[[diffusers.QwenImageEditPipeline]]

```python
diffusers.QwenImageEditPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, processor: Qwen2VLProcessor, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L165)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The Qwen-Image-Edit pipeline for image editing.

#### __call__[[diffusers.QwenImageEditPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L504)

**Parameters:**

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

true_cfg_scale (`float`, *optional*, defaults to 1.0) : true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageEditPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGB")
>>> prompt = (
...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(image, prompt, num_inference_steps=50).images[0]
>>> image.save("qwenimage_edit.png")
```

#### encode_prompt[[diffusers.QwenImageEditPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], image: typing.Optional[torch.Tensor] = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L272)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

image (`torch.Tensor`, *optional*) : image to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageEditInpaintPipeline[[diffusers.QwenImageEditInpaintPipeline]]

#### diffusers.QwenImageEditInpaintPipeline[[diffusers.QwenImageEditInpaintPipeline]]

```python
diffusers.QwenImageEditInpaintPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, processor: Qwen2VLProcessor, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L167)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The Qwen-Image-Edit pipeline for image editing.

#### __call__[[diffusers.QwenImageEditInpaintPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, mask_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, masked_image_latents: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, true_cfg_scale: float = 4.0, height: int | None = None, width: int | None = None, padding_mask_crop: int | None = None, strength: float = 0.6, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L637)

**Parameters:**

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

true_cfg_scale (`float`, *optional*, defaults to 1.0) : true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B, H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W, 1)`, or `(H, W)`.

masked_image_latents (`torch.Tensor`, `list[torch.Tensor]`) : `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask latents tensor will ge generated by `mask_image`.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

padding_mask_crop (`int`, *optional*, defaults to `None`) : The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region with the same aspect ration of the image and contains all masked area, and then expand that area based on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before resizing to the original image size for inpainting. This is useful when the masked area is small while the image is large and contain information irrelevant for inpainting, such as background.

strength (`float`, *optional*, defaults to 1.0) : Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 1 essentially ignores `image`.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageEditInpaintPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditInpaintPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

>>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
>>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
>>> source = load_image(img_url)
>>> mask = load_image(mask_url)
>>> image = pipe(
...     prompt=prompt, negative_prompt=" ", image=source, mask_image=mask, strength=1.0, num_inference_steps=50
... ).images[0]
>>> image.save("qwenimage_inpainting.png")
```

#### encode_prompt[[diffusers.QwenImageEditInpaintPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], image: typing.Optional[torch.Tensor] = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L284)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

image (`torch.Tensor`, *optional*) : image to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageControlNetPipeline[[diffusers.QwenImageControlNetPipeline]]

#### diffusers.QwenImageControlNetPipeline[[diffusers.QwenImageControlNetPipeline]]

```python
diffusers.QwenImageControlNetPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, transformer: QwenImageTransformer2DModel, controlnet: diffusers.models.controlnets.controlnet_qwenimage.QwenImageControlNetModel | diffusers.models.controlnets.controlnet_qwenimage.QwenImageMultiControlNetModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L192)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The QwenImage pipeline for text-to-image generation.

#### __call__[[diffusers.QwenImageControlNetPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, control_guidance_start: float | list[float] = 0.0, control_guidance_end: float | list[float] = 1.0, control_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, controlnet_conditioning_scale: float | list[float] = 1.0, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L511)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

true_cfg_scale (`float`, *optional*, defaults to 1.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

control_guidance_start (`float` or `list[float]`, *optional*, defaults to 0.0) : The percentage of total steps at which the ControlNet starts applying.

control_guidance_end (`float` or `list[float]`, *optional*, defaults to 1.0) : The percentage of total steps at which the ControlNet stops applying.

control_image (`PipelineImageInput`, *optional*) : The ControlNet input condition to provide guidance for the generation.

controlnet_conditioning_scale (`float` or `list[float]`, *optional*, defaults to 1.0) : The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original `transformer`.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers.utils import load_image
>>> from diffusers import QwenImageControlNetModel, QwenImageMultiControlNetModel, QwenImageControlNetPipeline

>>> # QwenImageControlNetModel
>>> controlnet = QwenImageControlNetModel.from_pretrained(
...     "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch.bfloat16
... )
>>> pipe = QwenImageControlNetPipeline.from_pretrained(
...     "Qwen/Qwen-Image", controlnet=controlnet, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation."
>>> negative_prompt = " "
>>> control_image = load_image(
...     "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/canny.png"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     control_image=control_image,
...     controlnet_conditioning_scale=1.0,
...     num_inference_steps=30,
...     true_cfg_scale=4.0,
... ).images[0]
>>> image.save("qwenimage_cn_union.png")

>>> # QwenImageMultiControlNetModel
>>> controlnet = QwenImageControlNetModel.from_pretrained(
...     "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch.bfloat16
... )
>>> controlnet = QwenImageMultiControlNetModel([controlnet])
>>> pipe = QwenImageControlNetPipeline.from_pretrained(
...     "Qwen/Qwen-Image", controlnet=controlnet, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation."
>>> negative_prompt = " "
>>> control_image = load_image(
...     "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/canny.png"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     control_image=[control_image, control_image],
...     controlnet_conditioning_scale=[0.5, 0.5],
...     num_inference_steps=30,
...     true_cfg_scale=4.0,
... ).images[0]
>>> image.save("qwenimage_cn_union_multi.png")
```

#### encode_prompt[[diffusers.QwenImageControlNetPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L290)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageEditPlusPipeline[[diffusers.QwenImageEditPlusPipeline]]

#### diffusers.QwenImageEditPlusPipeline[[diffusers.QwenImageEditPlusPipeline]]

```python
diffusers.QwenImageEditPlusPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, processor: Qwen2VLProcessor, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L168)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The Qwen-Image-Edit pipeline for image editing.

#### __call__[[diffusers.QwenImageEditPlusPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L526)

**Parameters:**

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

true_cfg_scale (`float`, *optional*, defaults to 1.0) : true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageEditPlusPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGB")
>>> prompt = (
...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(image, prompt, num_inference_steps=50).images[0]
>>> image.save("qwenimage_edit_plus.png")
```

#### encode_prompt[[diffusers.QwenImageEditPlusPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], image: typing.Optional[torch.Tensor] = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L286)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

image (`torch.Tensor`, *optional*) : image to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageLayeredPipeline[[diffusers.QwenImageLayeredPipeline]]

#### diffusers.QwenImageLayeredPipeline[[diffusers.QwenImageLayeredPipeline]]

```python
diffusers.QwenImageLayeredPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, processor: Qwen2VLProcessor, transformer: QwenImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py#L174)

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

The Qwen-Image-Layered pipeline for image decomposing.

#### __call__[[diffusers.QwenImageLayeredPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 4.0, layers: int | None = 4, num_inference_steps: int = 50, sigmas: list[float] | None = None, guidance_scale: float | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512, resolution: int = 640, cfg_normalize: bool = False, use_en_prompt: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py#L537)

**Parameters:**

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but if passing latents directly it is not encoded again.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

true_cfg_scale (`float`, *optional*, defaults to 1.0) : true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

layers (`int`, *optional*, defaults to 4) : Number of latent layers to generate for the layered output.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. This parameter in the pipeline is there to support future guidance-distilled models when they come up. It is ignored when not using guidance distilled models. To enable traditional classifier-free guidance, please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should enable classifier-free guidance computations).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

resolution (`int`, *optional*, defaults to 640) : using different bucket in (640, 1024) to determin the condition and output resolution

cfg_normalize (`bool`, *optional*, defaults to `False`) : whether enable cfg normalization.

use_en_prompt (`bool`, *optional*, defaults to `False`) : automatic caption language if user does not provide caption

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageLayeredPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGBA")
>>> prompt = ""
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> images = pipe(
...     image,
...     prompt,
...     num_inference_steps=50,
...     true_cfg_scale=4.0,
...     layers=4,
...     resolution=640,
...     cfg_normalize=False,
...     use_en_prompt=True,
... ).images[0]
>>> for i, image in enumerate(images):
...     image.save(f"{i}.out.png")
```

#### encode_prompt[[diffusers.QwenImageLayeredPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py#L291)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImagePipelineOutput[[diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput]]

#### diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput[[diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput]]

```python
diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/qwenimage/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for Stable Diffusion pipelines.

### ControlNet with Flux.1
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/controlnet_flux.md
