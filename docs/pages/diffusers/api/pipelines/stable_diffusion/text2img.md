# Text-to-image
image = pipe("A capybara wearing a wizard hat, oil painting").images[0]
image.save("t2i.png")

# Image-conditioned editing
edited = pipe("Move it to a snowy mountain top", image=image).images[0]
edited.save("edit.png")
```

## Multiple condition images

Pass a list to `image` and every entry becomes its own block in the joint sequence: the Qwen3-VL encoder sees them as
vision context and the VAE contributes their latent tokens. Block-causal attention keeps each block internally
bidirectional while letting later blocks and the target image attend to the earlier ones, so the order you pass them
in is the order the model reads them.

```python
edited = pipe("Put the flowers from the first image into the second scene", image=[flowers, scene]).images[0]
```

## Faster attention with flex_attention

The default `QwenImage21AttnProcessor` runs the block-causal prefill as one attention call per prefix segment. It
needs no compilation and works on any PyTorch build. `QwenImage21FlexAttnProcessor` expresses the same mask as a
single `flex_attention` call, which is faster once the model is **_compiled_**.

> [!TIP]
> Compile the model when you switch to the flex processor. An uncompiled `flex_attention` materializes the full
> attention score matrix in fp32, which is much slower and runs out of memory at high resolution.

```python
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
pipe.transformer.compile()
```

## Loading single-file checkpoints

```python
import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

transformer = QwenImage21Transformer2DModel.from_single_file(
    "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_bf16.safetensors",
    dtype=torch.bfloat16,
)
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", transformer=transformer, dtype=torch.bfloat16).to(
    "cuda"
)
```

## Sampling sigmas

Model authors can configure a default sampling grid with `sample_sigmas` in the pipeline config. When you load a
released checkpoint, its default grid and scheduler settings are restored automatically.

To experiment with a different grid at runtime, pass `sigmas` to the pipeline call:

```python
# Use the checkpoint's default sampling grid.
image = pipe(prompt).images[0]

# Override the default grid for this call.
image = pipe(prompt, sigmas=[1.0, 0.8, 0.5, 0.2]).images[0]
```

The custom grid above illustrates the API; generation quality depends on the checkpoint and grid. Sigma lists exclude
the terminal sigma, which the scheduler appends. Explicit `sigmas` override the configured `sample_sigmas`, and either
list determines the number of steps instead of `num_inference_steps`. If neither is provided, the pipeline uses
`num_inference_steps` to generate the schedule. The scheduler applies its configured processing to either grid.

## QwenImage21Pipeline[[diffusers.QwenImage21Pipeline]]

#### diffusers.QwenImage21Pipeline[[diffusers.QwenImage21Pipeline]]

```python
diffusers.QwenImage21Pipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLQwenImage21, text_encoder: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor, transformer: QwenImage21Transformer2DModel, sample_sigmas: list[float] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L159)

**Parameters:**

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : Scheduler used to denoise the encoded image latents.

vae ([AutoencoderKLQwenImage21](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_qwenimage21#diffusers.AutoencoderKLQwenImage21)) : Variational auto-encoder mapping images to and from the 64-channel latent space.

text_encoder (`Qwen3VLForConditionalGeneration`) : Qwen3-VL model producing the joint text/image embeddings.

processor (`Qwen3VLProcessor`) : Processor that builds the chat template and tokenizes prompt and condition images.

transformer ([QwenImage21Transformer2DModel](/docs/diffusers/v0.41.0/en/api/models/qwenimage21_transformer2d#diffusers.QwenImage21Transformer2DModel)) : The single-stream block-causal transformer that denoises the latents.

sample_sigmas (`list[float]`, *optional*) : Default sampling sigmas configured by the model author, excluding the terminal sigma. Their length determines the number of denoising steps. To experiment with a different grid at runtime, pass `sigmas` to `__call__`.

Text-to-image and image-conditioned generation with Qwen-Image 2.1.

Prompt and condition images are encoded together by a Qwen3-VL model, so a condition image occupies the vision
slots the encoder reserved for it and the transformer sees one interleaved text/image sequence.

#### __call__[[diffusers.QwenImage21Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, negative_prompt: str | list[str] = None, true_cfg_scale: float = 1.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 40, sigmas: list[float] | None = None, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int, dict], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], output_resolution: int = 1024, use_kv_cache: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L509)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt to guide image generation. Pass `prompt_embeds` instead to supply embeddings directly.

image (`PipelineImageInput`, *optional*) : One or more condition images, as a PIL image or a numpy array. They are encoded by the text encoder as vision context and by the VAE into latent tokens prepended to the noise. A list is one set of images shared by every prompt in the batch, not one entry per prompt.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide image generation. Ignored when `true_cfg_scale` is not greater than 1.

true_cfg_scale (`float`, *optional*, defaults to 1.0) : Classifier-free guidance scale. Enabled by `true_cfg_scale > 1` together with a negative prompt. Qwen-Image 2.1 is meant to be sampled without guidance, hence the default of 1.0.

height (`int`, *optional*) : Height in pixels of the generated image. Derived from the condition image's aspect ratio if omitted.

width (`int`, *optional*) : Width in pixels of the generated image. Derived from the condition image's aspect ratio if omitted.

num_inference_steps (`int`, *optional*, defaults to 40) : Number of denoising steps. Ignored when `sigmas` or the pipeline's configured `sample_sigmas` is used; the length of that schedule determines the number of steps.

sigmas (`list[float]`, *optional*) : Sampling sigmas to try for this call, excluding the terminal sigma. Overrides the pipeline's configured `sample_sigmas` and determines the number of denoising steps. The scheduler applies its configured processing to these values.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images generated per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Generator(s) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings, which skip prompt encoding. Pass `prompt_embeds_mask` with them.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Bool mask marking the valid positions of `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings, used in place of `negative_prompt`. Pass `negative_prompt_embeds_mask` with them.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Bool mask marking the valid positions of `negative_prompt_embeds`.

output_type (`str`, *optional*, defaults to `"pil"`) : Output format, `"pil"`, `"np"`, `"pt"` or `"latent"`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : Passed through to the attention processor.

callback_on_step_end (`Callable`, *optional*) : Called at the end of each denoising step.

callback_on_step_end_tensor_inputs (`list[str]`, *optional*, defaults to `["latents"]`) : Tensors from the denoising loop to hand to `callback_on_step_end`. They must be listed in the pipeline's `_callback_tensor_inputs`.

output_resolution (`int`, *optional*, defaults to 1024) : Target side length used to derive `height`/`width` and to resize condition images.

use_kv_cache (`bool`, *optional*, defaults to `True`) : Cache the text and condition-image keys and values after the first step. Valid because `causal_condition` modulates those tokens from `t = 0`, making their activations step-independent.  Toggling this does not reproduce the same image bit-for-bit in reduced precision. Caching makes the decode step attend with a different sequence layout than the prefill step, so the two tile differently and land on different rounding; both agree with an fp32 reference to the same tolerance. A one-ULP difference at the first block is then amplified by 32 blocks and every sampler step, so the two settings give equally valid but visibly distinct samples. Fix a sample by fixing this flag.

**Returns:** `~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple`

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple` whose first
element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImage21Pipeline

>>> pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "A capybara wearing a wizard hat, reading a book by candlelight, oil painting"
>>> image = pipe(prompt).images[0]
>>> image.save("qwenimage21.png")
```

#### encode_prompt[[diffusers.QwenImage21Pipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], image: list[typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]] | None = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, image_pad_mask: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py#L339)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : Prompt to be encoded.

image (`list[PipelineImageInput]`, *optional*) : Condition images to encode alongside the prompt.

device (`torch.device`) : Torch device.

num_images_per_prompt (`int`) : Number of images generated per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Skips encoding when provided.

## QwenImagePipelineOutput[[diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput]]

#### diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput[[diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput]]

```python
diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/qwenimage/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for Stable Diffusion pipelines.

### Ideogram 4
https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/ideogram4.md
