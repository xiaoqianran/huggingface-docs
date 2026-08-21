# MiniMax Music 3

[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) is a music generation model that produces complete
songs up to five minutes long from lyrics and a music description, with expressive vocals and long-range structure.

The model is a hybrid of an autoregressive and a diffusion stage: an 8B Qwen3-based global language model predicts one
semantic audio token per frame while a small depth decoder fills in seven residual RVQ codebooks, and their fused
hidden states condition a 2.4B flow-matching transformer that produces Flow-VAE latents in overlapping chunks. A
DAC-style decoder turns the latents into 44.1 kHz stereo audio.

## Usage

MiniMax Music 3 is available as a modular pipeline.

```py
import soundfile as sf
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

lyrics = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""

prompt = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    audio_duration=60.0,
    generator=torch.Generator("cuda").manual_seed(7),
    output="audios",
)[0]

sf.write("minimax_music3.wav", audio.T, pipe.sampling_rate)
```

## Reduce memory usage

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving
techniques.

The full pipeline needs ~23 GB of VRAM in bfloat16. With automatic CPU offloading a generation runs in ~22 GB of free
VRAM, and additionally group-offloading the language model fits in 8 GB.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

# Only needed below ~22 GB of free VRAM — slower, but fits in 8 GB.
apply_group_offloading(
    pipe.language_model, onload_device=torch.device("cuda"), offload_type="leaf_level", use_stream=True
)
```

## Tips

- Structure tags such as `[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`, `[instrumental]`, `[solo]`, and
  `[outro]` must each be on their own line in `lyrics`. Text on the same line as a leading tag is dropped by the
  model's input contract.
- The music description controls the vocals: describe the vocal gender and timbre explicitly (e.g. "warm female
  vocal") or the model may drift instrumental. For fine-grained control, structure the description into global
  metadata (genre, BPM, key, emotional progression), vocal details, and arrangement.
- `audio_duration` is an upper bound — the language model may end the song earlier with a stop token. The
  autoregressive stage generates 25 frames per second of audio and dominates the runtime.
- The classifier-free guidance scale of the flow-matching stage is a guider setting (the reference inference value is
  1.7): swap it with `pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=...))`.
- The pipeline returns the vocoder's native 44.1 kHz stereo output. The reference server additionally resamples to 32
  kHz; apply your own resampling if you need that exact rate.

## MiniMaxMusic3ModularPipeline[[diffusers.MiniMaxMusic3ModularPipeline]]

#### diffusers.MiniMaxMusic3ModularPipeline[[diffusers.MiniMaxMusic3ModularPipeline]]

```python
diffusers.MiniMaxMusic3ModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_music3/modular_pipeline.py#L22)

A ModularPipeline for lyrics- and caption-conditioned music generation with MiniMax Music 3.

## MiniMaxMusic3Blocks[[diffusers.MiniMaxMusic3Blocks]]

#### diffusers.MiniMaxMusic3Blocks[[diffusers.MiniMaxMusic3Blocks]]

```python
diffusers.MiniMaxMusic3Blocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_music3/modular_blocks_minimax_music3.py#L130)

Modular pipeline for lyrics- and caption-conditioned music generation using MiniMax Music 3. An autoregressive
Qwen3 language model generates per-frame semantic codes and hidden states from the lyrics and the music
description; a flow-matching transformer turns the hidden states into Flow-VAE latents chunk by chunk; and a
DAC-style vocoder decodes them into a stereo waveform at 44.1 kHz.

Components:
tokenizer (`Qwen2Tokenizer`) language_model (`Qwen3ForCausalLM`) rvq_depth_decoder
(`MiniMaxMusic3RVQDepthDecoder`) condition_encoder (`MiniMaxMusic3ConditionEncoder`) transformer
(`MiniMaxMusic3Transformer1DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
(`ClassifierFreeGuidance`) vocoder (`MiniMaxMusic3Vocoder`)

Inputs:
prompt (`str`):
The music description (genre, mood, vocals, instrumentation, arrangement).
lyrics (`str`):
The lyrics to sing. Structure tags such as `[verse]` or `[chorus]` must each be on their own line; text
on the same line as a leading tag is dropped by the checkpoint's input contract.
audio_duration (`float`, *optional*, defaults to 60.0):
Upper bound on the generated audio length in seconds. The language model may stop earlier. Capped at 9000
frames (six minutes).
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`, *optional*, defaults to 30):
Number of flow-matching Euler steps per chunk.
output_type (`str`, *optional*, defaults to np):
Output format: 'np' or 'pt'.

Outputs:
audios (`Tensor | ndarray`):
The generated stereo waveform of shape `(batch, channels, samples)` in `[-1, 1]`.

## MiniMaxMusic3ConditionEncoder[[diffusers.MiniMaxMusic3ConditionEncoder]]

#### diffusers.MiniMaxMusic3ConditionEncoder[[diffusers.MiniMaxMusic3ConditionEncoder]]

```python
diffusers.MiniMaxMusic3ConditionEncoder(condition_hidden_dim: int = 4096, num_condition_layers: int = 8, out_dim: int = 2048, input_sampling_rate: int = 24000, input_hop_length: int = 960, output_sampling_rate: int = 44100, output_hop_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/condition_embedders/condition_embedder_minimax_music3.py#L23)

Projects the per-frame hidden states of the autoregressive stage onto the Flow-VAE latent timeline.

Each generated frame carries `num_condition_layers` hidden states of size `condition_hidden_dim` (one from the
language model and one per residual codebook step). They are mixed with learned softmax weights, projected, and
resampled from the language-model frame rate to the latent frame rate with nearest-neighbor interpolation.

#### forward[[diffusers.MiniMaxMusic3ConditionEncoder.forward]]

```python
forward(hidden_states: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/condition_embedders/condition_embedder_minimax_music3.py#L48)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch, frames, num_condition_layers * condition_hidden_dim)`) : Concatenated per-frame hidden states from the autoregressive stage.

**Returns:** `torch.Tensor` of shape `(batch, latent_length, out_dim)`

the latent-aligned conditioning sequence.

## MiniMaxMusic3RVQDepthDecoder[[diffusers.MiniMaxMusic3RVQDepthDecoder]]

#### diffusers.MiniMaxMusic3RVQDepthDecoder[[diffusers.MiniMaxMusic3RVQDepthDecoder]]

```python
diffusers.MiniMaxMusic3RVQDepthDecoder(hidden_size: int = 4096, num_layers: int = 4, num_attention_heads: int = 16, intermediate_size: int = 6144, audio_vocab_size: int = 1024, num_codebooks: int = 8, max_position_embeddings: int = 16)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/minimax_music3_rvq_depth_decoder.py#L91)

The local language model of MiniMax Music 3. Within each audio frame it autoregressively predicts the seven
residual RVQ codebooks (c1..c7) from the global language model's hidden state and the frame's semantic code, and
exposes the per-step hidden states that condition the flow-matching transformer.

It also owns the embedding table for the residual codebooks, which the pipeline uses to embed complete frames for
the global language model's feedback loop.

#### forward[[diffusers.MiniMaxMusic3RVQDepthDecoder.forward]]

```python
forward(inputs_embeds: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/minimax_music3_rvq_depth_decoder.py#L127)

**Parameters:**

inputs_embeds (`torch.Tensor` of shape `(batch, steps, hidden_size)`) : Projected depth-sequence embeddings: the global hidden state followed by the embedded codes sampled so far, each passed through `projection`.

**Returns:** `torch.Tensor` of shape `(batch, steps, hidden_size)`

normalized hidden states; the last step feeds the
next codebook head.

## MiniMaxMusic3Vocoder[[diffusers.MiniMaxMusic3Vocoder]]

#### diffusers.MiniMaxMusic3Vocoder[[diffusers.MiniMaxMusic3Vocoder]]

```python
diffusers.MiniMaxMusic3Vocoder(latent_channels: int = 128, decoder_input_dim: int = 1024, decoder_hidden_dim: int = 1536, upsampling_ratios: tuple = (8, 8, 4, 2), sampling_rate: int = 44100)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/minimax_music3_vocoder.py#L71)

The Flow-VAE waveform decoder of MiniMax Music 3 (a DAC-style decoder). It decodes flow-matched latents of shape
`(batch, latent_channels, length)` into stereo waveforms at `sampling_rate`; the two audio channels are decoded as
two folded `latent_channels // 2` streams.

#### forward[[diffusers.MiniMaxMusic3Vocoder.forward]]

```python
forward(latents: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/minimax_music3_vocoder.py#L100)

**Parameters:**

latents (`torch.Tensor` of shape `(batch, latent_channels, length)`) : Flow-matched Flow-VAE latents.

**Returns:** `torch.Tensor` of shape `(batch, 2, samples)`

the stereo waveform in `[-1, 1]`.

### Cogview3
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/cogview3.md

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

# CogView3Plus

[CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion](https://huggingface.co/papers/2403.05121) from Tsinghua University & ZhipuAI, by Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, Jie Tang.

The abstract from the paper is:

*Recent advancements in text-to-image generative systems have been largely driven by diffusion models. However, single-stage text-to-image diffusion models still face challenges, in terms of computational efficiency and the refinement of image details. To tackle the issue, we propose CogView3, an innovative cascaded framework that enhances the performance of text-to-image diffusion. CogView3 is the first model implementing relay diffusion in the realm of text-to-image generation, executing the task by first creating low-resolution images and subsequently applying relay-based super-resolution. This methodology not only results in competitive text-to-image outputs but also greatly reduces both training and inference costs. Our experimental results demonstrate that CogView3 outperforms SDXL, the current state-of-the-art open-source text-to-image diffusion model, by 77.0% in human evaluations, all while requiring only about 1/2 of the inference time. The distilled variant of CogView3 achieves comparable performance while only utilizing 1/10 of the inference time by SDXL.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

## CogView3PlusPipeline[[diffusers.CogView3PlusPipeline]]

#### diffusers.CogView3PlusPipeline[[diffusers.CogView3PlusPipeline]]

```python
diffusers.CogView3PlusPipeline(tokenizer: T5Tokenizer, text_encoder: T5EncoderModel, vae: AutoencoderKL, transformer: CogView3PlusTransformer2DModel, scheduler: diffusers.schedulers.scheduling_ddim_cogvideox.CogVideoXDDIMScheduler | diffusers.schedulers.scheduling_dpm_cogvideox.CogVideoXDPMScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py#L118)

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. CogView3Plus uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CogView3PlusTransformer2DModel](/docs/diffusers/v0.40.0/en/api/models/cogview3plus_transformer2d#diffusers.CogView3PlusTransformer2DModel)) : A text conditioned `CogView3PlusTransformer2DModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

Pipeline for text-to-image generation using CogView3Plus.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.CogView3PlusPipeline.__call__]]

```python
__call__(prompt: str | list[str] | None = None, negative_prompt: str | list[str] | None = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, timesteps: list[int] | None = None, guidance_scale: float = 5.0, num_images_per_prompt: int = 1, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, original_size: tuple[int, int] | None = None, crops_coords_top_left: tuple = (0, 0), output_type: str = 'pil', return_dict: bool = True, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 224)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py#L407)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. If not provided, it is set to 1024.

width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. If not provided it is set to 1024.

num_inference_steps (`int`, *optional*, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to `5.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to `1`) : The number of images to generate per prompt.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies to [DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

original_size (`tuple[int]`, *optional*, defaults to (1024, 1024)) : If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled. `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

crops_coords_top_left (`tuple[int]`, *optional*, defaults to (0, 0)) : `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, defaults to `224`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

**Returns:** [CogView3PipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/cogview3#diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput) or `tuple`

[CogView3PipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/cogview3#diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CogView3PlusPipeline

>>> pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A photo of an astronaut riding a horse on mars"
>>> image = pipe(prompt).images[0]
>>> image.save("output.png")
```

#### encode_prompt[[diffusers.CogView3PlusPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 224, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py#L210)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

max_sequence_length (`int`, defaults to `224`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

## CogView3PipelineOutput[[diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput]]

#### diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput[[diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput]]

```python
diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogview3/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for CogView3 pipelines.

### DiT
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/dit.md
