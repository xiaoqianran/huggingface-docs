# Kandinsky 6

Kandinsky 6 is a family of video generation models from [Kandinsky Lab](https://huggingface.co/kandinskylab). The
main model generates video and synchronized audio from text or a reference image with a single multimodal diffusion
transformer: video and audio latents are denoised together through fused blocks that cross-attend between the two
modalities, each conditioned on its own Qwen2.5-VL text branch and a CLIP pooled embedding. A separate
super-resolution model upscales the generated video tile by tile in the latent space of a causal 3D K-VAE.

> [!TIP]
> Check out the [Kandinsky Lab](https://huggingface.co/kandinskylab) organization on the Hub for the full set of
> official checkpoints, including flow-matching and distilled variants of both the base and super-resolution models.
>
> Distilled checkpoints ship with the few-step [PiflowScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/piflow#diffusers.PiflowScheduler) and must be run with `guidance_scale=1.0`.

## Available models

| Model | Pipeline | Notes |
|---|---|---|
| [`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers) | [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) | [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) | Distilled, `guidance_scale=1.0`, 10 steps |
| [`kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers) | [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers) | [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) | Distilled, `guidance_scale=1.0`, 10 steps |
| [`kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-pretrain-5s-Diffusers) | [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-pretrain-5s-Diffusers) | [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) | Flow matching, `guidance_scale=5.0`, 50 steps |
| [`kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers) | [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) | Flow matching super-resolution |
| [`kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) | [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) | Distilled super-resolution, 2 steps |

## Text/image-to-video-and-audio

```python
import torch
from diffusers import Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video

pipe = Kandinsky6TI2VAPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A cat and a dog baking a cake together in a kitchen.",
    height=480,
    width=864,
    num_frames=121,
    num_inference_steps=10,
    guidance_scale=1.0,
)
encode_video(
    output.frames[0],
    fps=24,
    output_path="output.mp4",
    audio=output.audio[0][None],
    audio_sample_rate=pipe.audio_sample_rate,
)
```

Pass `image=` to condition the first frame on a reference image, `sample_audio=False` to generate video only, and
`expand_prompts=True` to let the Qwen2.5-VL text encoder rewrite short prompts into detailed ones first.

## Video super-resolution

[Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) takes the frames produced by [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) and upscales them by `2`, `4`, or
`2.25` (a 1.125x bilinear pre-upscale followed by the 2x path). The video is split into overlapping tiles, every tile
is refined at one of the tile sizes the SR transformer was trained on, and the tiles are blended back with Hann
windows.

```python
# required: lets inductor pick flex-attention tiles that fit the SR block mask
torch._inductor.config.max_autotune = True  

sr_pipe = Kandinsky6SRPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", torch_dtype=torch.bfloat16
)
# The SR transformer always runs NABLA sparse attention on the `flex` backend. Compile it, otherwise flex falls
# back to an eager implementation that needs far more memory at video resolutions.
sr_pipe.enable_model_cpu_offload()
sr_pipe.transformer.set_attention_backend("flex")
sr_pipe.transformer.compile_repeated_blocks(fullgraph=True)

upscaled = sr_pipe(video=output.frames[0], resolution_scale=2.25, num_inference_steps=2).frames[0]
```

## Memory optimization

Refer to the [Reduce memory usage](../../optimization/memory) guide for the general set of techniques. Both
[Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) and [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) support [model offloading](../../optimization/memory#model-offloading)
(used above) and, for a smaller footprint at the cost of speed, [sequential CPU offloading](../../optimization/memory#cpu-offloading):

```python
pipe.enable_sequential_cpu_offload()
```

[Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline)'s video VAE also supports [tiled decoding](../../optimization/memory#vae-tiling) for high
resolutions or long videos:

```python
pipe.vae.enable_tiling()
```

## Notes

- `height` and `width` must be divisible by the video VAE's spatial compression ratio times the transformer's patch
  size — `16` with the default [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) configuration (`AutoencoderKLHunyuanVideo` at a
  compression ratio of `8`, `patch_size=(1, 2, 2)`). `480x864`, used in the example above, satisfies this.
- [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline)'s input `video` must have `1 + k * vae_scale_factor_temporal` frames for some integer `k`
  (a temporal compression ratio of `4` with the default K-VAE configuration, so `121` frames works but `120` doesn't)
  — trim or pad a video that doesn't already satisfy this before upscaling it.
- `expand_prompts=True` reuses the already-loaded Qwen2.5-VL text encoder for an extra generation pass before
  denoising, so it adds latency but no extra model weights.
- Compile the repeated transformer blocks for faster repeated inference:
  ```python
  pipe.transformer.compile_repeated_blocks(fullgraph=True)
  ```

## Kandinsky6TI2VAPipeline[[diffusers.Kandinsky6TI2VAPipeline]]

#### diffusers.Kandinsky6TI2VAPipeline[[diffusers.Kandinsky6TI2VAPipeline]]

```python
diffusers.Kandinsky6TI2VAPipeline(transformer: Kandinsky6Transformer3DModel, vae: AutoencoderKLHunyuanVideo, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2_5_VLProcessor, text_encoder_2: CLIPTextModel, tokenizer_2: CLIPTokenizer, scheduler: diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler | diffusers.schedulers.scheduling_piflow.PiflowScheduler, audio_vae: diffusers.models.autoencoders.autoencoder_mmaudio.MMAudioVAE | None = None, vocoder: diffusers.models.autoencoders.mmaudio_vocoder.MMAudioVocoder | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L178)

**Parameters:**

transformer ([Kandinsky6Transformer3DModel](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_transformer#diffusers.Kandinsky6Transformer3DModel)) : Multimodal transformer that denoises the video and audio latents.

vae ([AutoencoderKLHunyuanVideo](/docs/diffusers/v0.42.1/en/api/models/autoencoder_kl_hunyuan_video#diffusers.AutoencoderKLHunyuanVideo)) : Video VAE used to encode the reference image and decode the generated video.

text_encoder ([Qwen2_5_VLForConditionalGeneration](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLForConditionalGeneration)) : Qwen2.5-VL model providing the token-level text embeddings and, optionally, prompt expansion.

tokenizer ([Qwen2_5_VLProcessor](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor)) : Processor of `text_encoder`.

text_encoder_2 ([CLIPTextModel](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/clip#transformers.CLIPTextModel)) : CLIP text encoder providing the pooled text embedding.

tokenizer_2 ([CLIPTokenizer](https://huggingface.co/docs/transformers/v5.20.2/en/model_doc/clip#transformers.CLIPTokenizer)) : Tokenizer of `text_encoder_2`.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) or [PiflowScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/piflow#diffusers.PiflowScheduler)) : Scheduler used with `transformer` to denoise the latents. Distilled checkpoints ship with a [PiflowScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/piflow#diffusers.PiflowScheduler) and must be run with `guidance_scale=1.0`.

audio_vae ([MMAudioVAE](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVAE), *optional*) : Audio VAE used to decode the generated audio latents into a mel spectrogram. Only needed when `sample_audio=True`.

vocoder ([MMAudioVocoder](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVocoder), *optional*) : Vocoder used to turn the mel spectrogram `audio_vae` decodes into a waveform. Only needed when `sample_audio=True`.

Pipeline for text/image-to-video-and-audio generation with Kandinsky 6.

Video and audio latents are denoised together by a single multimodal transformer, conditioned on Qwen2.5-VL text
tokens and a CLIP pooled embedding. An optional reference image conditions the first frame.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.Kandinsky6TI2VAPipeline.__call__]]

```python
__call__(prompt: str | list[str] | None = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, negative_prompt: str | list[str] | None = None, height: int = 512, width: int = 768, num_frames: int = 121, frame_rate: float = 24.0, num_inference_steps: int = 50, timesteps: list[int] | None = None, sigmas: list[float] | None = None, guidance_scale: float = 5.0, num_videos_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, audio_latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, sample_audio: bool = True, expand_prompts: bool = False, max_sequence_length: int = 1024, output_type: str = 'pil', return_dict: bool = True, callback_on_step_end: collections.abc.Callable[[int, int, dict], None] | None = None, callback_on_step_end_tensor_inputs: list = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L644)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the generation. Required unless `prompt_embeds` is given.

image (`PipelineImageInput`, *optional*) : Reference image(s) conditioning the first frame (image-to-video-and-audio).

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the generation. Defaults to the Kandinsky 6 negative prompt.

height (`int`, defaults to `512`) : Height of the generated video in pixels.

width (`int`, defaults to `768`) : Width of the generated video in pixels.

num_frames (`int`, defaults to `121`) : Number of generated frames.

frame_rate (`float`, defaults to `24.0`) : Frame rate the video is generated at; sets the length of the synchronized audio.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. Use `16` with the distilled checkpoints.

timesteps (`list[int]`, *optional*) : Custom timesteps for schedulers that support them.

sigmas (`list[float]`, *optional*) : Custom sigmas for schedulers that support them.

guidance_scale (`float`, defaults to `5.0`) : Classifier-free guidance scale. Must be `1.0` with a [PiflowScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/piflow#diffusers.PiflowScheduler).

num_videos_per_prompt (`int`, defaults to `1`) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Generator(s) used for the initial noise and the reference image encoding.

latents (`torch.Tensor`, *optional*) : Pre-generated video latents of shape `(batch_size, channels, num_latent_frames, latent_height, latent_width)`.

audio_latents (`torch.Tensor`, *optional*) : Pre-generated audio latents of shape `(batch_size, channels, audio_length)`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated Qwen2.5-VL text embeddings.

pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated CLIP pooled text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative Qwen2.5-VL text embeddings.

negative_pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative CLIP pooled text embeddings.

sample_audio (`bool`, defaults to `True`) : Whether to generate synchronized audio. Requires the pipeline to have an `audio_vae` and a `vocoder`.

expand_prompts (`bool`, defaults to `False`) : Whether to rewrite the prompts with [expand_prompts()](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline.expand_prompts) before encoding.

max_sequence_length (`int`, defaults to `1024`) : Maximum number of prompt tokens after the chat template.

output_type (`str`, defaults to `"pil"`) : The output format of the generated video: `"pil"`, `"np"`, `"pt"` or `"latent"`.

return_dict (`bool`, defaults to `True`) : Whether or not to return a [Kandinsky6TI2VAPipelineOutput](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipelineOutput) instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function called at the end of each denoising step with `callback_on_step_end(self, step, timestep, callback_kwargs)`. It may return a dict overriding the listed tensors.

callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`) : Tensor inputs passed to `callback_on_step_end`; a subset of `_callback_tensor_inputs`.

**Returns:** [Kandinsky6TI2VAPipelineOutput](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipelineOutput) or `tuple`

The generated video and audio; a `(frames, audio)` tuple when `return_dict=False`.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import Kandinsky6TI2VAPipeline
>>> from diffusers.utils import encode_video

>>> pipe = Kandinsky6TI2VAPipeline.from_pretrained(
...     "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.enable_model_cpu_offload()

>>> output = pipe(
...     prompt="A cat and a dog baking a cake together in a kitchen.",
...     height=480,
...     width=864,
...     num_frames=121,
...     num_inference_steps=16,
...     guidance_scale=1.0,
... )
>>> encode_video(
...     output.frames[0],
...     fps=24,
...     output_path="output.mp4",
...     audio=output.audio[0][None],
...     audio_sample_rate=pipe.audio_sample_rate,
... )
```

#### encode_image[[diffusers.Kandinsky6TI2VAPipeline.encode_image]]

```python
encode_image(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], height: int, width: int, device: device, dtype: dtype, num_videos_per_prompt: int = 1, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L551)

Encodes the reference image(s) into first-frame latents of shape `(batch_size, latent_height, latent_width,
latent_channels)`, scaled by the VAE `scaling_factor`. PIL images are resized and center-cropped to `height x
width`; tensors and arrays must already have that size. The latents are repeated `num_videos_per_prompt` times
along the batch dimension.

#### encode_prompt[[diffusers.Kandinsky6TI2VAPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 1024, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L309)

**Parameters:**

prompt (`str` or `list[str]`) : Prompt to be encoded.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the generation. Ignored when `do_classifier_free_guidance` is `False`.

do_classifier_free_guidance (`bool`, defaults to `True`) : Whether to also encode the negative prompt.

num_videos_per_prompt (`int`, defaults to `1`) : Number of videos generated per prompt; the embeddings are repeated accordingly.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated Qwen2.5-VL text embeddings. Skips encoding `prompt`.

pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated CLIP pooled text embeddings. Must be given together with `prompt_embeds`.

prompt_attention_mask (`torch.Tensor`, *optional*) : Boolean padding mask of `prompt_embeds`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative Qwen2.5-VL text embeddings.

negative_pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative CLIP pooled text embeddings.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Boolean padding mask of `negative_prompt_embeds`.

max_sequence_length (`int`, defaults to `1024`) : Maximum number of prompt tokens after the chat template.

device (`torch.device`, *optional*) : Device to run the text encoders on.

dtype (`torch.dtype`, *optional*) : Dtype of the returned embeddings.

Encodes the prompt into text encoder hidden states.

#### expand_prompts[[diffusers.Kandinsky6TI2VAPipeline.expand_prompts]]

```python
expand_prompts(prompt: str | list[str], tokenizer, text_encoder, device: device, image: PIL.Image.Image | list[PIL.Image.Image] | None = None, max_sequence_length: int = 1024, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L412)

**Parameters:**

prompt (`str` or `list[str]`) : Prompt or prompts to expand.

tokenizer : The Qwen2.5-VL processor, e.g. `pipe.tokenizer`.

text_encoder : The Qwen2.5-VL model, e.g. `pipe.text_encoder`.

device (`torch.device`) : Device to run the text encoder on.

image (`PIL.Image.Image` or `list[PIL.Image.Image]`, *optional*) : Reference image(s) of an image-to-video call.

max_sequence_length (`int`, defaults to `1024`) : Maximum number of generated tokens per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Seeds the sampled expansion; a list must match `prompt`'s length, one generator per item. `generate` draws from the global RNG, so the global RNG is seeded from this generator's seed; later `randn_tensor` calls keep using `generator` directly.

**Returns:** `str` or `list[str]`

The expanded prompt(s).

Rewrites short prompts into detailed video+audio prompts with the Qwen2.5-VL text encoder, grounding them on
the reference image when one is given. A `staticmethod` so it can be used standalone, before running the
pipeline.

#### prepare_audio_latents[[diffusers.Kandinsky6TI2VAPipeline.prepare_audio_latents]]

```python
prepare_audio_latents(batch_size: int, num_channels_latents: int, audio_length: int, dtype: dtype, device: device, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType], audio_latents: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L607)

Returns audio latents in the transformer's `(batch_size, audio_length, channels)` layout. A user-provided
`audio_latents` tensor is expected in the `(batch_size, channels, audio_length)` layout.

#### prepare_latents[[diffusers.Kandinsky6TI2VAPipeline.prepare_latents]]

```python
prepare_latents(batch_size: int, num_channels_latents: int, height: int, width: int, num_frames: int, dtype: dtype, device: device, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType], latents: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py#L578)

Returns video latents in the transformer's `(batch_size, num_frames, height, width, channels)` layout.
A user-provided `latents` tensor is expected in the `(batch_size, channels, num_frames, height, width)` layout.

## Kandinsky6SRPipeline[[diffusers.Kandinsky6SRPipeline]]

#### diffusers.Kandinsky6SRPipeline[[diffusers.Kandinsky6SRPipeline]]

```python
diffusers.Kandinsky6SRPipeline(transformer: Kandinsky6SRTransformer3DModel, vae: Kandinsky6SRVAE, scheduler: diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler | diffusers.schedulers.scheduling_piflow.PiflowScheduler, latent_upscaler: diffusers.models.latent_upscaler.latent_upscaler_kandinsky6_sr.Kandinsky6SRLatentUpscalerBank | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_sr.py#L143)

**Parameters:**

transformer ([Kandinsky6SRTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_transformer#diffusers.Kandinsky6SRTransformer3DModel)) : Transformer that refines the latent tiles.

vae ([Kandinsky6SRVAE](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.Kandinsky6SRVAE)) : Causal video K-VAE used to encode the input video and decode the refined tiles.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) or [PiflowScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/piflow#diffusers.PiflowScheduler)) : Scheduler used with `transformer` to denoise the tiles. Distilled checkpoints ship with a [PiflowScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/piflow#diffusers.PiflowScheduler).

latent_upscaler ([Kandinsky6SRLatentUpscalerBank](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRLatentUpscalerBank), *optional*) : Latent upscalers for the supported scales.

Pipeline for video super-resolution with Kandinsky 6.

The video is split into overlapping spatial tiles, every tile is refined by the SR transformer at one of the tile
sizes the model was trained on (`transformer.config.tile_sizes`), and the refined tiles are blended back with Hann
windows. When the pipeline has a `latent_upscaler`, the tiles are cut from the K-VAE latents of the whole video and
upscaled in latent space; otherwise the pixel tiles are bilinearly upscaled and encoded.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.Kandinsky6SRPipeline.__call__]]

```python
__call__(video: typing.Union[list[PIL.Image.Image], list[list[PIL.Image.Image]], numpy.ndarray, torch.Tensor], resolution_scale: float = 2.25, num_inference_steps: int = 4, timesteps: list[int] | None = None, sigmas: list[float] | None = None, lq_noise_scale: float = 0.7, min_overlap: float = 0.2, tiles_batch_size: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str = 'pil', return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_sr.py#L228)

**Parameters:**

video (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`) : The low-resolution video(s), in any format [preprocess_video()](/docs/diffusers/v0.42.1/en/api/video_processor#diffusers.VideoProcessor.preprocess_video) accepts, with `1 + k * 4` frames. Sizes are rounded down to a multiple of the VAE spatial factor.

resolution_scale (`float`, defaults to `2.25`) : Total spatial upscale: `2`, `4`, or `2.25` (a 1.125x bilinear pre-upscale followed by the 2x path).

num_inference_steps (`int`, defaults to `4`) : The number of denoising steps per tile. Use `2` with the distilled checkpoints.

timesteps (`list[int]`, *optional*) : Custom timesteps for schedulers that support them.

sigmas (`list[float]`, *optional*) : Custom sigmas for schedulers that support them.

lq_noise_scale (`float`, defaults to `0.7`) : Amount of Gaussian noise mixed into the low-resolution latents (variance preserving) before denoising.

min_overlap (`float`, defaults to `0.2`) : Minimum overlap between neighbouring tiles as a fraction of the tile size.

tiles_batch_size (`int`, defaults to `1`) : Number of tiles denoised per transformer call.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Generator(s) used for the noise mixed into the tiles.

output_type (`str`, defaults to `"pil"`) : The output format of the generated video: `"pil"`, `"np"` or `"pt"`.

return_dict (`bool`, defaults to `True`) : Whether or not to return a [Kandinsky6SRPipelineOutput](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipelineOutput) instead of a plain tuple.

**Returns:** [Kandinsky6SRPipelineOutput](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipelineOutput) or `tuple`

The super-resolved video; a one-element tuple when `return_dict=False`.

The call function to the pipeline for super-resolution.

Examples:
```python
>>> import torch
>>> from diffusers import Kandinsky6SRPipeline, Kandinsky6TI2VAPipeline
>>> from diffusers.utils import export_to_video

>>> pipe = Kandinsky6TI2VAPipeline.from_pretrained(
...     "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.enable_model_cpu_offload()
>>> video = pipe(
...     prompt="A cat and a dog baking a cake together in a kitchen.",
...     height=480,
...     width=864,
...     num_inference_steps=16,
...     guidance_scale=1.0,
...     sample_audio=False,
... ).frames[0]

>>> sr_pipe = Kandinsky6SRPipeline.from_pretrained(
...     "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", torch_dtype=torch.bfloat16
... )
>>> # The transformer always runs attention through the `flex` backend; compiling avoids the eager
>>> # fallback's much higher memory use at video resolutions.
>>> sr_pipe.transformer.compile_repeated_blocks(fullgraph=True)
>>> sr_pipe.enable_model_cpu_offload()
>>> output = sr_pipe(video=video, resolution_scale=2.25, num_inference_steps=2)
>>> export_to_video(output.frames[0], "output_sr.mp4", fps=24)
```

#### decode_latents[[diffusers.Kandinsky6SRPipeline.decode_latents]]

```python
decode_latents(latents: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_sr.py#L223)

Decodes scaled K-VAE latents into a `(batch_size, channels, num_frames, height, width)` video in `[-1, 1]`.

#### encode_video[[diffusers.Kandinsky6SRPipeline.encode_video]]

```python
encode_video(video: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_kandinsky6_sr.py#L213)

Encodes a `(batch_size, channels, num_frames, height, width)` video in `[-1, 1]` into K-VAE latents scaled by
the VAE `scaling_factor`.

## Kandinsky6SRLatentUpscalerBank[[diffusers.Kandinsky6SRLatentUpscalerBank]]

#### diffusers.Kandinsky6SRLatentUpscalerBank[[diffusers.Kandinsky6SRLatentUpscalerBank]]

```python
diffusers.Kandinsky6SRLatentUpscalerBank(in_channels: int = 64, stage_channels: tuple[int, int, int] = (2048, 1024, 512), num_pre_blocks: int = 5, num_mid_blocks: int = 3, num_post_blocks: int = 3, num_x2_adapter_blocks: int = 2, scales: tuple[int, ...] = (2, 4), scaling_factor: float = 0.910344004631042)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/latent_upscaler/latent_upscaler_kandinsky6_sr.py#L251)

**Parameters:**

in_channels (`int`, defaults to `64`) : Number of latent channels.

stage_channels (`tuple[int, int, int]`, defaults to `(2048, 1024, 512)`) : Feature widths of the three stages of the cascade.

num_pre_blocks (`int`, defaults to `5`) : Residual blocks before the first upsample of the x4 model.

num_mid_blocks (`int`, defaults to `3`) : Residual blocks between the two upsamples.

num_post_blocks (`int`, defaults to `3`) : Residual blocks after the last upsample.

num_x2_adapter_blocks (`int`, defaults to `2`) : Residual blocks of the x2 model's adapter.

scales (`tuple[int, ...]`, defaults to `(2, 4)`) : Spatial scales the bank provides an upscaler for.

scaling_factor (`float`, defaults to `0.910344`) : Scale the input latents are expected to carry (the K-VAE `scaling_factor`).

Bank of latent upscalers used by [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline): one `Kandinsky6SRLatentUpscaler` per supported spatial
scale, operating on K-VAE latents.

#### forward[[diffusers.Kandinsky6SRLatentUpscalerBank.forward]]

```python
forward(latents: Tensor, scale: int, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/latent_upscaler/latent_upscaler_kandinsky6_sr.py#L305)

**Parameters:**

latents (`torch.Tensor` of shape `(batch_size, in_channels, num_frames, height, width)`) : K-VAE latents scaled by `scaling_factor`.

scale (`int`) : Spatial upscale factor; one of `scales`.

return_dict (`bool`, defaults to `True`) : Whether to return a `~models.autoencoder_kl.DecoderOutput` instead of a plain tuple.

**Returns:**

The upscaled latents of shape `(batch_size, in_channels, num_frames, height * scale, width * scale)`.

## Kandinsky6TI2VAPipelineOutput[[diffusers.Kandinsky6TI2VAPipelineOutput]]

#### diffusers.Kandinsky6TI2VAPipelineOutput[[diffusers.Kandinsky6TI2VAPipelineOutput]]

```python
diffusers.Kandinsky6TI2VAPipelineOutput(frames: typing.Union[torch.Tensor, numpy.ndarray, list[list[PIL.Image.Image]]], audio: typing.Union[torch.Tensor, numpy.ndarray, NoneType] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_output.py#L25)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`) : The generated video. A nested list of length `batch_size` holding `num_frames` PIL images each, or a NumPy array or torch tensor of shape `(batch_size, num_frames, height, width, channels)` / `(batch_size, num_frames, channels, height, width)`. With `output_type="latent"`, the video latents of shape `(batch_size, channels, num_latent_frames, latent_height, latent_width)`.

audio (`torch.Tensor` or `np.ndarray`, *optional*) : The generated waveforms of shape `(batch_size, num_samples)` in `[-1, 1]` at the audio VAE's sample rate, or `None` when audio was not sampled. With `output_type="latent"`, the audio latents of shape `(batch_size, channels, audio_length)`.

Output class for [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline).

## Kandinsky6SRPipelineOutput[[diffusers.Kandinsky6SRPipelineOutput]]

#### diffusers.Kandinsky6SRPipelineOutput[[diffusers.Kandinsky6SRPipelineOutput]]

```python
diffusers.Kandinsky6SRPipelineOutput(frames: typing.Union[torch.Tensor, numpy.ndarray, list[list[PIL.Image.Image]]])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/kandinsky6/pipeline_output.py#L46)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or `list[list[PIL.Image.Image]]`) : The super-resolved video. A nested list of length `batch_size` holding `num_frames` PIL images each, or a NumPy array or torch tensor of shape `(batch_size, num_frames, height, width, channels)` / `(batch_size, num_frames, channels, height, width)`.

Output class for [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline).

### ACE-Step 1.5
https://huggingface.co/docs/diffusers/v0.42.1/api/pipelines/ace_step.md
