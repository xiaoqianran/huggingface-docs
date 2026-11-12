# Stable Audio 3

Stable Audio 3 (SA3) is a text-to-audio model from [Stability AI](https://stability.ai/) that generates high-quality
stereo audio at 44.1 kHz. It uses a rectified-flow DiT conditioned on two signals:

* **Text** — encoded by a frozen T5Gemma encoder and injected via cross-attention.
* **Duration** — a float (seconds) embedded by [StableAudio3DurationEmbedder](/docs/diffusers/v0.43.0/en/api/pipelines/stable_audio_3#diffusers.StableAudio3DurationEmbedder) and used as a global conditioning
  vector for adaptive layer normalisation.

Audio is decoded by the SAME (Semantically-Aligned Music Encoder) autoencoder, [AutoencoderSAME](/docs/diffusers/v0.43.0/en/api/models/autoencoder_same#diffusers.AutoencoderSAME).

Both checkpoints use [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) with a log-SNR-uniform sigma schedule, differing only in
`stochastic_sampling` and the default step count:

| Checkpoint | `diffusion_objective` | `stochastic_sampling` | `num_inference_steps` |
|---|---|---|---|
| `stable-audio-3-medium-base` | `rectified_flow` | `False` (deterministic Euler) | **100** (not distilled) |
| `stable-audio-3-medium` (distilled) | `rf_denoiser` | `True` (ping-pong re-noise) | **8** (distilled for 8 steps) |

The correct scheduler config is baked into each converted checkpoint, so `num_inference_steps` defaults to the right
value when you leave it unset. Only pass it to override.

Original codebase: [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3).

## Converting original checkpoints

The Stability AI checkpoints are not published in diffusers format, so convert them locally. The script downloads the
checkpoint's `model_config.json` and selects the scheduler from its `diffusion_objective`:

```bash
python scripts/convert_stable_audio_3_to_diffusers.py \
  --checkpoint_path stabilityai/stable-audio-3-medium-base \
  --text_encoder_repo google/t5gemma-b-b-ul2 \
  --output_dir /tmp/sa3-diffusers-euler \
  --dtype float32
```

> [!TIP]
> `stable-audio-3-medium-base` is a **gated** repo. Run `hf auth login` with an account that has access before
> converting, otherwise the download fails with a 401.

## Usage example

Load the converted checkpoint from its local output directory (install
[`soundfile`](https://pypi.org/project/soundfile/) with `pip install soundfile`):

```py
import torch
import soundfile as sf
from diffusers import StableAudio3Pipeline

pipe = StableAudio3Pipeline.from_pretrained("/tmp/sa3-diffusers-euler", torch_dtype=torch.float32)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

generator = torch.Generator("cuda").manual_seed(0)
audio = pipe(
    "A gentle piano melody with soft strings in a concert hall",
    duration=10.0,  # seconds; latent length is computed automatically
    generator=generator,
).audios

sf.write("sa3_output.wav", audio[0].T.cpu().float().numpy(), samplerate=44100)
```

The pipeline is also registered with [AutoPipelineForText2Audio](/docs/diffusers/v0.43.0/en/api/pipelines/auto_pipeline#diffusers.AutoPipelineForText2Audio), which resolves the checkpoint to
`StableAudio3Pipeline` automatically:

```py
from diffusers import AutoPipelineForText2Audio

pipe = AutoPipelineForText2Audio.from_pretrained("/tmp/sa3-diffusers-euler", torch_dtype=torch.float32)
```

> [!NOTE]
> The examples use a local path because `stabilityai/stable-audio-3-medium` and `stable-audio-3-medium-base` are not
> yet published in diffusers format (loading by repo id returns a 404). Once published, the repo id works in place of
> the local path.

## Tips

* Use `torch.float32` on CPU or MPS (Apple Silicon) — `torch.float16` on MPS produces noise.
* The distilled model (`stable-audio-3-medium`) is **adversarially distilled** — guidance is baked into the weights.
  Leave `guidance_scale=1.0` (the default) and don't pass a `negative_prompt` for that checkpoint; both only do
  something useful for the non-distilled `stable-audio-3-medium-base` checkpoint.
* `silence_padding_duration` (default `0.0`) adds silent headroom at the end of the latent sequence. Leave it at `0.0`
  unless the model is trained to mask that padding — otherwise the extra frames drain output energy and the result
  gets quiet.
* Set `num_waveforms_per_prompt > 1` to generate multiple clips per prompt.

## StableAudio3Pipeline[[diffusers.StableAudio3Pipeline]]

#### diffusers.StableAudio3Pipeline[[diffusers.StableAudio3Pipeline]]

```python
diffusers.StableAudio3Pipeline(vae: AutoencoderSAME, text_encoder: T5GemmaEncoderModel, tokenizer: GemmaTokenizer, duration_embedder: StableAudio3DurationEmbedder, transformer: StableAudio3DiTModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3.py#L157)

**Parameters:**

vae ([*AutoencoderSAME*]) : SAME autoencoder used to encode and decode audio latents.

text_encoder ([*~transformers.T5GemmaEncoderModel*]) : Frozen T5Gemma text encoder (`google/t5gemma-b-b-ul2`).

tokenizer ([*~transformers.GemmaTokenizerFast*]) : Tokenizer for the text encoder.

duration_embedder ([*StableAudio3DurationEmbedder*]) : Maps `duration` in seconds to a global conditioning vector for AdaLN in each DiT block.

transformer ([*StableAudio3DiTModel*]) : The rectified-flow velocity-prediction DiT.

scheduler ([*FlowMatchEulerDiscreteScheduler*]) : Scheduler for the iterative denoising loop. The production (distilled) SA3 Medium checkpoint uses *stochastic_sampling=True* for exactly 8 ping-pong steps; the non-distilled base checkpoint uses *stochastic_sampling=False* for ~100 deterministic Euler steps.

Pipeline for text-to-audio generation using Stable Audio 3.

SA3 uses a distilled rectified-flow DiT with ping-pong sampling. Classifier-free guidance (`guidance_scale` /
`negative_prompt`) is unnecessary for the distilled checkpoint (leave `guidance_scale=1.0`, the default) but is
meaningful for the non-distilled `stable-audio-3-medium-base` checkpoint.

This model inherits from [*DiffusionPipeline*]. Check the superclass documentation for the generic methods
implemented for all pipelines.

#### __call__[[diffusers.StableAudio3Pipeline.__call__]]

```python
__call__(prompt: typing.Union[str, typing.List[str], NoneType] = None, duration: float = 10.0, num_inference_steps: typing.Optional[int] = None, logsnr_min: float = -6.2, logsnr_max: float = 2.0, silence_padding_duration: float = 0.0, guidance_scale: float = 1.0, negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, num_waveforms_per_prompt: int = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.LongTensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_encoder_attention_mask: typing.Optional[torch.LongTensor] = None, return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int, dict], dict]] = None, callback_on_step_end_tensor_inputs: typing.List[str] = ['latents'], output_type: str = 'pt')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3.py#L398)

**Parameters:**

prompt (*str* or *list[str]*, *optional*) : Text prompt(s). Pass `prompt_embeds` instead to skip tokenization and encoding.

duration (*float*, defaults to 10.0) : Requested output duration in seconds.

num_inference_steps (*int*, *optional*) : Number of denoising steps. When `None` (default), the step count is chosen from the scheduler's *stochastic_sampling* config: **8** when *True* (the distilled ping-pong-style checkpoint) and **100** when *False* (the non-distilled base checkpoint). Pass an explicit value to override.

logsnr_min (*float*, defaults to -6.2) : Minimum log-SNR value for the noise schedule — maps to the high-noise start of the schedule.

logsnr_max (*float*, defaults to 2.0) : Maximum log-SNR value for the noise schedule — maps to the low-noise end of the schedule.

silence_padding_duration (*float*, defaults to 0.0) : Extra seconds of latent context generated beyond the target content, giving the model headroom at the boundary; the output is trimmed back to *duration*. Defaults to 0.0 (disabled). Increase only if the model is trained/distilled to mask this padding — otherwise the extra frames drain output energy.

guidance_scale (*float*, defaults to 1.0) : Classifier-free guidance scale. `1.0` disables guidance (the default, and the only sensible value for the distilled SA3 Medium checkpoint, whose CFG is baked into the weights). Values `> 1.0` are meaningful for the non-distilled `stable-audio-3-medium-base` checkpoint; higher values follow the prompt more closely at the cost of diversity.

negative_prompt (*str* or *list[str]*, *optional*) : Prompt(s) describing what to steer away from when `guidance_scale > 1.0`. Defaults to an empty string (unconditional) when `guidance_scale > 1.0` and neither this nor *negative_prompt_embeds* is given. Ignored when `guidance_scale &amp;lt;= 1.0`.

num_waveforms_per_prompt (*int*, defaults to 1) : Number of waveforms to generate per prompt.

generator (*torch.Generator* or *list[torch.Generator]*, *optional*) : For deterministic generation and reproducible re-noise in the ping-pong loop.

latents (*torch.Tensor*, *optional*) : Pre-generated starting latents. If `None` a fresh Gaussian tensor is sampled.

prompt_embeds (*torch.Tensor*, *optional*) : Pre-computed text embeddings `(batch, seq_len, 768)`.

encoder_attention_mask (*torch.LongTensor*, *optional*) : Boolean mask for pre-computed embeddings.

negative_prompt_embeds (*torch.Tensor*, *optional*) : Pre-computed negative text embeddings, as an alternative to *negative_prompt*.

negative_encoder_attention_mask (*torch.LongTensor*, *optional*) : Boolean mask for pre-computed negative embeddings.

return_dict (*bool*, defaults to *True*) : Return an *AudioPipelineOutput* or a plain tuple.

callback_on_step_end (*Callable*, *optional*) : Called at the end of each denoising step with *(self, step_idx, timestep, callback_kwargs)*, where *callback_kwargs* contains the tensors listed in *callback_on_step_end_tensor_inputs*. Must return a dict with the (optionally modified) tensors to use for the rest of the loop.

callback_on_step_end_tensor_inputs (*list[str]*, defaults to *["latents"]*) : The tensors passed to *callback_on_step_end*. Must be a subset of *self._callback_tensor_inputs*.

output_type (*str*, defaults to `"pt"`) : `"pt"` for a PyTorch tensor, `"np"` for a NumPy array, or `"latent"` to skip decoding and return the raw latents.

**Returns:** [*~pipelines.AudioPipelineOutput*] or *tuple*

`.audios` is a tensor / array of shape `(batch * num_waveforms_per_prompt, audio_channels, samples)`.

Generate audio from a text prompt.

Examples:
```py
>>> import torch
>>> import soundfile as sf
>>> from diffusers import StableAudio3Pipeline

>>> pipe = StableAudio3Pipeline.from_pretrained("stabilityai/stable-audio-3-medium", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> generator = torch.Generator("cuda").manual_seed(0)
>>> audio = pipe(
...     "A gentle piano melody with soft strings in a concert hall",
...     duration=10.0,
...     generator=generator,
... ).audios

>>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=pipe.vae.config.sampling_rate)
```

#### encode_duration[[diffusers.StableAudio3Pipeline.encode_duration]]

```python
encode_duration(duration: float, device: device, num_waveforms_per_prompt: int, batch_size: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3.py#L289)

**Parameters:**

duration : Duration in seconds, applied to every sample in the batch.

device : Target device.

num_waveforms_per_prompt : Tile factor.

batch_size : Number of prompts.

**Returns:**

`(batch * num_waveforms_per_prompt, output_dim)` tensor.

Embed the duration value into the global conditioning vector.

#### encode_prompt[[diffusers.StableAudio3Pipeline.encode_prompt]]

```python
encode_prompt(prompt: typing.Union[str, typing.List[str], NoneType], device: device, num_waveforms_per_prompt: int, prompt_embeds: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.LongTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3.py#L223)

**Parameters:**

prompt : Text prompt or list of prompts.  Ignored when `prompt_embeds` is provided.

device : Target device.

num_waveforms_per_prompt : How many output waveforms to generate per prompt; conditioning tensors are tiled accordingly.

prompt_embeds : Pre-computed text embeddings `(batch, seq_len, hidden_size)`.

encoder_attention_mask : Boolean mask `(batch, seq_len)` for pre-computed embeddings; `1` = real token, `0` = pad.

**Returns:**

`(prompt_embeds, encoder_attention_mask)` both tiled to `batch * num_waveforms_per_prompt`.

Encode text prompt(s) into cross-attention conditioning tensors.

#### prepare_cross_attention[[diffusers.StableAudio3Pipeline.prepare_cross_attention]]

```python
prepare_cross_attention(prompt_embeds: Tensor, encoder_attention_mask: Tensor, global_hidden_states: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3.py#L313)

**Returns:**

`(context, context_mask)` of shapes `(batch, T_text + 1, dim)` and `(batch, T_text + 1)`.

Build the cross-attention context by appending the duration embedding as an extra token.

SA3 routes the `seconds_total` conditioner to *both* the global (AdaLN) input and the cross-attention context
(`cross_attention_cond_ids = ["prompt", "seconds_total"]`). The duration embedding is concatenated after the
text tokens, and the attention mask is extended with one valid entry.

## StableAudio3InpaintPipeline[[diffusers.StableAudio3InpaintPipeline]]

#### diffusers.StableAudio3InpaintPipeline[[diffusers.StableAudio3InpaintPipeline]]

```python
diffusers.StableAudio3InpaintPipeline(vae: AutoencoderSAME, text_encoder: T5GemmaEncoderModel, tokenizer: GemmaTokenizer, duration_embedder: StableAudio3DurationEmbedder, transformer: StableAudio3DiTModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_inpaint.py#L158)

**Parameters:**

vae ([*AutoencoderSAME*]) : SAME autoencoder used to encode and decode audio latents.

text_encoder ([*~transformers.T5GemmaEncoderModel*]) : Frozen T5Gemma text encoder (`google/t5gemma-b-b-ul2`).

tokenizer ([*~transformers.GemmaTokenizerFast*]) : Tokenizer for the text encoder.

duration_embedder ([*StableAudio3DurationEmbedder*]) : Maps `duration` in seconds to a global conditioning vector for AdaLN in each DiT block.

transformer ([*StableAudio3DiTModel*]) : The rectified-flow velocity-prediction DiT.

scheduler ([*FlowMatchEulerDiscreteScheduler*]) : Scheduler for the iterative denoising loop. The production (distilled) SA3 Medium checkpoint uses *stochastic_sampling=True* for exactly 8 ping-pong steps; the non-distilled base checkpoint uses *stochastic_sampling=False* for ~100 deterministic Euler steps.

Audio inpainting pipeline for Stable Audio 3.

Shares its text-to-audio logic with [*StableAudio3Pipeline*] (kept in sync via *# Copied from*). When `audio` and
`mask` are provided, encodes the reference audio with the frozen SAME encoder and injects `masked_latent ∥ mask` as local-additive conditioning into each DiT block via the transformer's `local_add_cond` pathway
(`to_local_embed`).

Call signature extension (see `__call__`):
audio (*torch.Tensor* of shape `(batch, channels, samples)`):
Reference audio waveform at `vae.config.sampling_rate` Hz.
mask (*torch.Tensor* of shape `(batch, 1, latent_length)`):
Per-frame binary mask in latent space. `1` = preserve original audio; `0` = region to be inpainted.

#### __call__[[diffusers.StableAudio3InpaintPipeline.__call__]]

```python
__call__(prompt: typing.Union[str, typing.List[str], NoneType] = None, duration: float = 10.0, audio: typing.Optional[torch.Tensor] = None, mask: typing.Optional[torch.Tensor] = None, mask_start_seconds: typing.Union[float, typing.List[float], NoneType] = None, mask_end_seconds: typing.Union[float, typing.List[float], NoneType] = None, num_inference_steps: typing.Optional[int] = None, logsnr_min: float = -6.2, logsnr_max: float = 2.0, silence_padding_duration: float = 0.0, num_waveforms_per_prompt: int = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.LongTensor] = None, return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int, dict], dict]] = None, callback_on_step_end_tensor_inputs: typing.List[str] = ['latents'], output_type: str = 'pt')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_inpaint.py#L455)

**Parameters:**

prompt (*str* or *list[str]*, *optional*) : Text prompt(s).

duration (*float*, defaults to 10.0) : Output duration in seconds. Should match the reference audio.

audio (*torch.Tensor*, *optional*) : Reference waveform `(batch, channels, samples)` at `vae.config.sampling_rate` Hz. Required for inpainting.

mask (*torch.Tensor*, *optional*) : Per-frame latent-space mask `(batch, 1, L)` with 0 = inpaint region, 1 = preserve. Either `mask` or `mask_start_seconds` / `mask_end_seconds` must be provided.

mask_start_seconds (*float* or *list[float]*, *optional*) : Start time(s) of the inpaint region in seconds.

mask_end_seconds (*float* or *list[float]*, *optional*) : End time(s) of the inpaint region (must pair with `mask_start_seconds`).

num_inference_steps (*int*, *optional*) : Number of denoising steps. When `None` (default), the step count is chosen from the scheduler's *stochastic_sampling* config, matching [*StableAudio3Pipeline*].

logsnr_min (*float*, defaults to -6.2) : Minimum log-SNR value for the noise schedule — maps to the high-noise start of the schedule.

logsnr_max (*float*, defaults to 2.0) : Maximum log-SNR value for the noise schedule — maps to the low-noise end of the schedule.

silence_padding_duration (*float*, defaults to 0.0) : Extra latent headroom after the target content.

num_waveforms_per_prompt (*int*, defaults to 1) : Waveforms per prompt.

generator : RNG for reproducibility.

latents : Pre-generated starting noise (`None` → sample fresh).

prompt_embeds : Pre-computed text embeddings.

encoder_attention_mask : Mask for pre-computed embeddings.

return_dict (*bool*, defaults to *True*) : Return *AudioPipelineOutput* or tuple.

callback_on_step_end (*Callable*, *optional*) : Called at the end of each denoising step with *(self, step_idx, timestep, callback_kwargs)*. Must return a dict with the (optionally modified) tensors to use for the rest of the loop.

callback_on_step_end_tensor_inputs (*list[str]*, defaults to *["latents"]*) : The tensors passed to *callback_on_step_end*. Must be a subset of *self._callback_tensor_inputs*.

output_type (*str*, defaults to `"pt"`) : `"pt"` / `"np"` / `"latent"`.

**Returns:**

[*~pipelines.AudioPipelineOutput*] with `.audios`.

Generate inpainted audio conditioned on a text prompt and reference.

Examples:
```py
>>> import torch
>>> import soundfile as sf
>>> import torchaudio
>>> from diffusers import StableAudio3InpaintPipeline

>>> pipe = StableAudio3InpaintPipeline.from_pretrained(
...     "stabilityai/stable-audio-3-medium", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> audio, sr = torchaudio.load("reference.wav")
>>> audio = torchaudio.functional.resample(audio, sr, pipe.vae.config.sampling_rate).unsqueeze(0).to("cuda")

>>> generator = torch.Generator("cuda").manual_seed(0)
>>> audio = pipe(
...     "A gentle piano melody with soft strings in a concert hall",
...     duration=10.0,
...     audio=audio,
...     mask_start_seconds=4.0,
...     mask_end_seconds=6.0,
...     generator=generator,
... ).audios

>>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=pipe.vae.config.sampling_rate)
```

#### encode_duration[[diffusers.StableAudio3InpaintPipeline.encode_duration]]

```python
encode_duration(duration: float, device: device, num_waveforms_per_prompt: int, batch_size: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_inpaint.py#L290)

**Parameters:**

duration : Duration in seconds, applied to every sample in the batch.

device : Target device.

num_waveforms_per_prompt : Tile factor.

batch_size : Number of prompts.

**Returns:**

`(batch * num_waveforms_per_prompt, output_dim)` tensor.

Embed the duration value into the global conditioning vector.

#### encode_prompt[[diffusers.StableAudio3InpaintPipeline.encode_prompt]]

```python
encode_prompt(prompt: typing.Union[str, typing.List[str], NoneType], device: device, num_waveforms_per_prompt: int, prompt_embeds: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.LongTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_inpaint.py#L223)

**Parameters:**

prompt : Text prompt or list of prompts.  Ignored when `prompt_embeds` is provided.

device : Target device.

num_waveforms_per_prompt : How many output waveforms to generate per prompt; conditioning tensors are tiled accordingly.

prompt_embeds : Pre-computed text embeddings `(batch, seq_len, hidden_size)`.

encoder_attention_mask : Boolean mask `(batch, seq_len)` for pre-computed embeddings; `1` = real token, `0` = pad.

**Returns:**

`(prompt_embeds, encoder_attention_mask)` both tiled to `batch * num_waveforms_per_prompt`.

Encode text prompt(s) into cross-attention conditioning tensors.

#### prepare_cross_attention[[diffusers.StableAudio3InpaintPipeline.prepare_cross_attention]]

```python
prepare_cross_attention(prompt_embeds: Tensor, encoder_attention_mask: Tensor, global_hidden_states: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_inpaint.py#L315)

**Returns:**

`(context, context_mask)` of shapes `(batch, T_text + 1, dim)` and `(batch, T_text + 1)`.

Build the cross-attention context by appending the duration embedding as an extra token.

SA3 routes the `seconds_total` conditioner to *both* the global (AdaLN) input and the cross-attention context
(`cross_attention_cond_ids = ["prompt", "seconds_total"]`). The duration embedding is concatenated after the
text tokens, and the attention mask is extended with one valid entry.

## StableAudio3AudioToAudioPipeline[[diffusers.StableAudio3AudioToAudioPipeline]]

Generates a variation of a reference audio clip: the whole reference is noised to `init_noise_level` and denoised
from there, unlike [StableAudio3InpaintPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/stable_audio_3#diffusers.StableAudio3InpaintPipeline)'s per-frame local-additive conditioning which preserves specific
frames exactly.

#### diffusers.StableAudio3AudioToAudioPipeline[[diffusers.StableAudio3AudioToAudioPipeline]]

```python
diffusers.StableAudio3AudioToAudioPipeline(vae: AutoencoderSAME, text_encoder: T5GemmaEncoderModel, tokenizer: GemmaTokenizer, duration_embedder: StableAudio3DurationEmbedder, transformer: StableAudio3DiTModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_audio2audio.py#L154)

**Parameters:**

vae ([*AutoencoderSAME*]) : SAME autoencoder used to encode and decode audio latents.

text_encoder ([*~transformers.T5GemmaEncoderModel*]) : Frozen T5Gemma text encoder (`google/t5gemma-b-b-ul2`).

tokenizer ([*~transformers.GemmaTokenizerFast*]) : Tokenizer for the text encoder.

duration_embedder ([*StableAudio3DurationEmbedder*]) : Maps `duration` in seconds to a global conditioning vector for AdaLN in each DiT block.

transformer ([*StableAudio3DiTModel*]) : The rectified-flow velocity-prediction DiT.

scheduler ([*FlowMatchEulerDiscreteScheduler*]) : Scheduler for the iterative denoising loop. The production (distilled) SA3 Medium checkpoint uses *stochastic_sampling=True* for exactly 8 ping-pong steps; the non-distilled base checkpoint uses *stochastic_sampling=False* for ~100 deterministic Euler steps.

Audio-to-audio variation pipeline for Stable Audio 3.

Shares its text-to-audio logic with [*StableAudio3Pipeline*] (kept in sync via *# Copied from*). Encodes the
reference audio with the frozen SAME encoder, mixes it with fresh noise according to `init_noise_level`, and
denoises from there — the whole signal is noised/denoised globally, unlike [*StableAudio3InpaintPipeline*]'s
per-frame local-additive conditioning.

Call signature extension (see `__call__`):
audio (*torch.Tensor* of shape `(batch, channels, samples)`):
Reference audio waveform at `vae.config.sampling_rate` Hz.
init_noise_level (*float*):
How much noise to mix into the reference before denoising. `1.0` = full noise (equivalent to
text-to-audio); lower values retain more of the reference.

#### __call__[[diffusers.StableAudio3AudioToAudioPipeline.__call__]]

```python
__call__(prompt: typing.Union[str, typing.List[str], NoneType] = None, duration: float = 10.0, audio: typing.Optional[torch.Tensor] = None, init_noise_level: float = 1.0, num_inference_steps: typing.Optional[int] = None, logsnr_min: float = -6.2, logsnr_max: float = 2.0, silence_padding_duration: float = 0.0, num_waveforms_per_prompt: int = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.LongTensor] = None, return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int, dict], dict]] = None, callback_on_step_end_tensor_inputs: typing.List[str] = ['latents'], output_type: str = 'pt')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_audio2audio.py#L424)

**Parameters:**

prompt (*str* or *list[str]*, *optional*) : Text prompt(s).

duration (*float*, defaults to 10.0) : Output duration in seconds. Should match the reference audio.

audio (*torch.Tensor*, *optional*) : Reference waveform `(batch, channels, samples)` at `vae.config.sampling_rate` Hz. Required.

init_noise_level (*float*, defaults to 1.0) : Noise level (in `(0, 1]`) mixed into the reference before denoising: `x_start = (1 - init_noise_level) * reference_latents + init_noise_level * noise`. `1.0` discards the reference entirely (equivalent to [*StableAudio3Pipeline*]); lower values retain progressively more of the reference's structure while still running the full step count.

num_inference_steps (*int*, *optional*) : Number of denoising steps. When `None` (default), the step count is chosen from the scheduler's *stochastic_sampling* config, matching [*StableAudio3Pipeline*].

logsnr_min (*float*, defaults to -6.2) : Minimum log-SNR value for the noise schedule — maps to the high-noise start of the schedule.

logsnr_max (*float*, defaults to 2.0) : Maximum log-SNR value for the noise schedule — maps to the low-noise end of the schedule.

silence_padding_duration (*float*, defaults to 0.0) : Extra latent headroom after the target content.

num_waveforms_per_prompt (*int*, defaults to 1) : Waveforms per prompt.

generator : RNG for reproducibility.

latents : Pre-generated starting noise (`None` → sample fresh).

prompt_embeds : Pre-computed text embeddings.

encoder_attention_mask : Mask for pre-computed embeddings.

return_dict (*bool*, defaults to *True*) : Return *AudioPipelineOutput* or tuple.

callback_on_step_end (*Callable*, *optional*) : Called at the end of each denoising step with *(self, step_idx, timestep, callback_kwargs)*. Must return a dict with the (optionally modified) tensors to use for the rest of the loop.

callback_on_step_end_tensor_inputs (*list[str]*, defaults to *["latents"]*) : The tensors passed to *callback_on_step_end*. Must be a subset of *self._callback_tensor_inputs*.

output_type (*str*, defaults to `"pt"`) : `"pt"` / `"np"` / `"latent"`.

**Returns:**

[*~pipelines.AudioPipelineOutput*] with `.audios`.

Generate an audio variation conditioned on a text prompt and a reference waveform.

Examples:
```py
>>> import torch
>>> import soundfile as sf
>>> import torchaudio
>>> from diffusers import StableAudio3AudioToAudioPipeline

>>> pipe = StableAudio3AudioToAudioPipeline.from_pretrained(
...     "stabilityai/stable-audio-3-medium", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> audio, sr = torchaudio.load("reference.wav")
>>> audio = torchaudio.functional.resample(audio, sr, pipe.vae.config.sampling_rate).unsqueeze(0).to("cuda")

>>> generator = torch.Generator("cuda").manual_seed(0)
>>> audio = pipe(
...     "A gentle piano melody with soft strings in a concert hall",
...     duration=10.0,
...     audio=audio,
...     init_noise_level=0.6,
...     generator=generator,
... ).audios

>>> sf.write("output.wav", audio[0].T.cpu().float().numpy(), samplerate=pipe.vae.config.sampling_rate)
```

#### encode_duration[[diffusers.StableAudio3AudioToAudioPipeline.encode_duration]]

```python
encode_duration(duration: float, device: device, num_waveforms_per_prompt: int, batch_size: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_audio2audio.py#L287)

**Parameters:**

duration : Duration in seconds, applied to every sample in the batch.

device : Target device.

num_waveforms_per_prompt : Tile factor.

batch_size : Number of prompts.

**Returns:**

`(batch * num_waveforms_per_prompt, output_dim)` tensor.

Embed the duration value into the global conditioning vector.

#### encode_prompt[[diffusers.StableAudio3AudioToAudioPipeline.encode_prompt]]

```python
encode_prompt(prompt: typing.Union[str, typing.List[str], NoneType], device: device, num_waveforms_per_prompt: int, prompt_embeds: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.LongTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_audio2audio.py#L220)

**Parameters:**

prompt : Text prompt or list of prompts.  Ignored when `prompt_embeds` is provided.

device : Target device.

num_waveforms_per_prompt : How many output waveforms to generate per prompt; conditioning tensors are tiled accordingly.

prompt_embeds : Pre-computed text embeddings `(batch, seq_len, hidden_size)`.

encoder_attention_mask : Boolean mask `(batch, seq_len)` for pre-computed embeddings; `1` = real token, `0` = pad.

**Returns:**

`(prompt_embeds, encoder_attention_mask)` both tiled to `batch * num_waveforms_per_prompt`.

Encode text prompt(s) into cross-attention conditioning tensors.

#### prepare_cross_attention[[diffusers.StableAudio3AudioToAudioPipeline.prepare_cross_attention]]

```python
prepare_cross_attention(prompt_embeds: Tensor, encoder_attention_mask: Tensor, global_hidden_states: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/stable_audio_3/pipeline_stable_audio_3_audio2audio.py#L312)

**Returns:**

`(context, context_mask)` of shapes `(batch, T_text + 1, dim)` and `(batch, T_text + 1)`.

Build the cross-attention context by appending the duration embedding as an extra token.

SA3 routes the `seconds_total` conditioner to *both* the global (AdaLN) input and the cross-attention context
(`cross_attention_cond_ids = ["prompt", "seconds_total"]`). The duration embedding is concatenated after the
text tokens, and the attention mask is extended with one valid entry.

## StableAudio3DurationEmbedder[[diffusers.StableAudio3DurationEmbedder]]

#### diffusers.StableAudio3DurationEmbedder[[diffusers.StableAudio3DurationEmbedder]]

```python
diffusers.StableAudio3DurationEmbedder(output_dim: int = 768, fourier_dim: int = 256, min_val: float = 0.0, max_val: float = 384.0, min_freq: float = 0.5, max_freq: float = 10000.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/condition_embedders/condition_embedder_stable_audio_3.py#L43)

**Parameters:**

output_dim (*int*, defaults to 768) : Dimension of the output embedding. Must match the DiT's `global_cond_dim`.

fourier_dim (*int*, defaults to 256) : Internal Fourier feature dimension (must be even).

min_val (*float*, defaults to 0.0) : Minimum duration value for normalization clamping.

max_val (*float*, defaults to 384.0) : Maximum duration value for normalization clamping. Values above this are clamped. 384 seconds is the production SA3 Medium upper bound for the `seconds_total` conditioner.

min_freq (*float*, defaults to 0.5) : Minimum frequency for the exponential Fourier basis.

max_freq (*float*, defaults to 10000.0) : Maximum frequency for the exponential Fourier basis.

Embeds a duration value (in seconds) into a global conditioning vector for the Stable Audio 3 DiT (used as the
`global_hidden_states` AdaLN input).

Replicates `NumberConditioner(fourier_features_type="expo")` from the SA3 reference implementation.

#### forward[[diffusers.StableAudio3DurationEmbedder.forward]]

```python
forward(seconds: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/condition_embedders/condition_embedder_stable_audio_3.py#L90)

**Parameters:**

seconds (`torch.Tensor` of shape `(batch,)`) : Duration values in seconds.

**Returns:**

`torch.Tensor` of shape `(batch, output_dim)` — duration embeddings.

### Hunyuanimage21
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/hunyuanimage21.md

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

# HunyuanImage2.1

HunyuanImage-2.1 is a 17B text-to-image model that is capable of generating 2K (2048 x 2048) resolution images

HunyuanImage-2.1 comes in the following variants:

| model type | model id |
|:----------:|:--------:|
| HunyuanImage-2.1 | [hunyuanvideo-community/HunyuanImage-2.1-Diffusers](https://huggingface.co/hunyuanvideo-community/HunyuanImage-2.1-Diffusers) |
| HunyuanImage-2.1-Distilled | [hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers](https://huggingface.co/hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers) |
| HunyuanImage-2.1-Refiner | [hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers](https://huggingface.co/hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers) |

> [!TIP]
> [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

## HunyuanImage-2.1

HunyuanImage-2.1 applies [Adaptive Projected Guidance (APG)](https://huggingface.co/papers/2410.02416) combined with Classifier-Free Guidance (CFG) in the denoising loop. `HunyuanImagePipeline` has a `guider` component (read more about [Guider](../../using-diffusers/guiders)) and does not take a `guidance_scale` parameter at runtime. To change guider-related parameters, e.g., `guidance_scale`, you can update the `guider` configuration instead.

```python
import torch
from diffusers import HunyuanImagePipeline

pipe = HunyuanImagePipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 
    dtype=torch.bfloat16
)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"
``` 

You can inspect the `guider` object:

```py
>>> pipe.guider
AdaptiveProjectedMixGuidance {
  "_class_name": "AdaptiveProjectedMixGuidance",
  "_diffusers_version": "0.36.0.dev0",
  "adaptive_projected_guidance_momentum": -0.5,
  "adaptive_projected_guidance_rescale": 10.0,
  "adaptive_projected_guidance_scale": 10.0,
  "adaptive_projected_guidance_start_step": 5,
  "enabled": true,
  "eta": 0.0,
  "guidance_rescale": 0.0,
  "guidance_scale": 3.5,
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
  momentum_buffer: None
  is_apg_enabled: False
  is_cfg_enabled: True
```

To update the guider with a different configuration, use the `new()` method. For example, to generate an image with `guidance_scale=5.0` while keeping all other default guidance parameters:

```py
import torch
from diffusers import HunyuanImagePipeline

pipe = HunyuanImagePipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 
    dtype=torch.bfloat16
)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

# Update the guider configuration
pipe.guider = pipe.guider.new(guidance_scale=5.0)

prompt = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, "
    "wearing a red knitted scarf and a red beret with the word 'Tencent' on it, holding a paintbrush with a "
    "focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
)

image = pipe(
    prompt=prompt, 
    num_inference_steps=50, 
    height=2048, 
    width=2048,
).images[0]
image.save("image.png")
```

## HunyuanImage-2.1-Distilled

use `distilled_guidance_scale` with the guidance-distilled checkpoint, 

```py
import torch
from diffusers import HunyuanImagePipeline
pipe = HunyuanImagePipeline.from_pretrained("hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers", dtype=torch.bfloat16)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt = (
    "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, "
    "wearing a red knitted scarf and a red beret with the word 'Tencent' on it, holding a paintbrush with a "
    "focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
)

out = pipe(
    prompt,
    num_inference_steps=8,
    distilled_guidance_scale=3.25,
    height=2048,
    width=2048,
    generator=generator,
).images[0]

```

## HunyuanImagePipeline[[diffusers.HunyuanImagePipeline]]

#### diffusers.HunyuanImagePipeline[[diffusers.HunyuanImagePipeline]]

```python
diffusers.HunyuanImagePipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLHunyuanImage, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, text_encoder_2: T5EncoderModel, tokenizer_2: ByT5Tokenizer, transformer: HunyuanImageTransformer2DModel, guider: AdaptiveProjectedMixGuidance | None = None, ocr_guider: AdaptiveProjectedMixGuidance | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py#L160)

**Parameters:**

transformer ([HunyuanImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/hunyuanimage_transformer_2d#diffusers.HunyuanImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLHunyuanImage](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_hunyuanimage#diffusers.AutoencoderKLHunyuanImage)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider (`AdaptiveProjectedMixGuidance`) : [AdaptiveProjectedMixGuidance]to be used to guide the image generation.

ocr_guider (`AdaptiveProjectedMixGuidance`, *optional*) : [AdaptiveProjectedMixGuidance] to be used to guide the image generation when text rendering is needed.

The HunyuanImage pipeline for text-to-image generation.

#### __call__[[diffusers.HunyuanImagePipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, distilled_guidance_scale: float | None = 3.25, sigmas: list[float] | None = None, num_images_per_prompt: int = 1, generator: torch.Generator | list[torch.Generator] | None = None, latents: torch.Tensor | None = None, prompt_embeds: torch.Tensor | None = None, prompt_embeds_mask: torch.Tensor | None = None, negative_prompt_embeds: torch.Tensor | None = None, negative_prompt_embeds_mask: torch.Tensor | None = None, prompt_embeds_2: torch.Tensor | None = None, prompt_embeds_mask_2: torch.Tensor | None = None, negative_prompt_embeds_2: torch.Tensor | None = None, negative_prompt_embeds_mask_2: torch.Tensor | None = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, Any] | None = None, callback_on_step_end: Callable[[int, int], None] | None = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py#L504)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined and negative_prompt_embeds is not provided, will use an empty negative prompt. Ignored when not using guidance. ).

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

distilled_guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance is enabled by setting `distilled_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. For guidance distilled models, this parameter is required. For non-distilled models, this parameter will be ignored.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text embeddings mask. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated text embeddings for ocr. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings for ocr will be generated from `prompt` input argument.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated text embeddings mask for ocr. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings mask for ocr will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings mask. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative text embeddings mask will be generated from `negative_prompt` input argument.

negative_prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings for ocr. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative text embeddings for ocr will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings mask for ocr. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative text embeddings mask for ocr will be generated from `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** `~pipelines.hunyuan_image.HunyuanImagePipelineOutput` or `tuple`

`~pipelines.hunyuan_image.HunyuanImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import HunyuanImagePipeline

>>> pipe = HunyuanImagePipeline.from_pretrained(
...     "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, negative_prompt="", num_inference_steps=50).images[0]
>>> image.save("hunyuanimage.png")
```

#### encode_prompt[[diffusers.HunyuanImagePipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: torch.device | None = None, batch_size: int = 1, num_images_per_prompt: int = 1, prompt_embeds: torch.Tensor | None = None, prompt_embeds_mask: torch.Tensor | None = None, prompt_embeds_2: torch.Tensor | None = None, prompt_embeds_mask_2: torch.Tensor | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage.py#L296)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

## HunyuanImageRefinerPipeline[[diffusers.HunyuanImageRefinerPipeline]]

#### diffusers.HunyuanImageRefinerPipeline[[diffusers.HunyuanImageRefinerPipeline]]

```python
diffusers.HunyuanImageRefinerPipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLHunyuanImageRefiner, text_encoder: Qwen2_5_VLForConditionalGeneration, tokenizer: Qwen2Tokenizer, transformer: HunyuanImageTransformer2DModel, guider: AdaptiveProjectedMixGuidance | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage_refiner.py#L138)

**Parameters:**

transformer ([HunyuanImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/hunyuanimage_transformer_2d#diffusers.HunyuanImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLHunyuanImageRefiner](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_hunyuanimage_refiner#diffusers.AutoencoderKLHunyuanImageRefiner)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

The HunyuanImage pipeline for text-to-image generation.

#### __call__[[diffusers.HunyuanImageRefinerPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, distilled_guidance_scale: float | None = 3.25, image: PipelineImageInput | None = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 4, sigmas: list[float] | None = None, num_images_per_prompt: int = 1, generator: torch.Generator | list[torch.Generator] | None = None, latents: torch.Tensor | None = None, prompt_embeds: torch.Tensor | None = None, prompt_embeds_mask: torch.Tensor | None = None, negative_prompt_embeds: torch.Tensor | None = None, negative_prompt_embeds_mask: torch.Tensor | None = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, Any] | None = None, callback_on_step_end: Callable[[int, int], None] | None = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage_refiner.py#L436)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, will use an empty negative prompt. Ignored when not using guidance.

distilled_guidance_scale (`float`, *optional*, defaults to None) : A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance where the guidance scale is applied during inference through noise prediction rescaling, guidance distilled models take the guidance scale directly as an input parameter during forward pass. Guidance is enabled by setting `distilled_guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. For guidance distilled models, this parameter is required. For non-distilled models, this parameter will be ignored.

image (`PipelineImageInput`, *optional*) : The input image to be refined.

num_images_per_prompt (`int`, *optional*, defaults to 1) --

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

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

**Returns:** `~pipelines.hunyuan_image.HunyuanImagePipelineOutput` or `tuple`

`~pipelines.hunyuan_image.HunyuanImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import HunyuanImageRefinerPipeline

>>> pipe = HunyuanImageRefinerPipeline.from_pretrained(
...     "hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> image = load_image("path/to/image.png")
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, image=image, num_inference_steps=4).images[0]
>>> image.save("hunyuanimage.png")
```

#### encode_prompt[[diffusers.HunyuanImageRefinerPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str] | None = None, device: torch.device | None = None, batch_size: int = 1, num_images_per_prompt: int = 1, prompt_embeds: torch.Tensor | None = None, prompt_embeds_mask: torch.Tensor | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_hunyuanimage_refiner.py#L225)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

## HunyuanImagePipelineOutput[[diffusers.pipelines.hunyuan_image.pipeline_output.HunyuanImagePipelineOutput]]

#### diffusers.pipelines.hunyuan_image.pipeline_output.HunyuanImagePipelineOutput[[diffusers.pipelines.hunyuan_image.pipeline_output.HunyuanImagePipelineOutput]]

```python
diffusers.pipelines.hunyuan_image.pipeline_output.HunyuanImagePipelineOutput(images: list[PIL.Image.Image, numpy.ndarray])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_image/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for HunyuanImage pipelines.

### ControlNet with Stable Diffusion XL
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/controlnet_sdxl.md
