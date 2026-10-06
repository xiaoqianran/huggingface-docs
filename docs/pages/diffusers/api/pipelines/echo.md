# Echo

[Echo](https://github.com/jd-opensource/JoyAI-Echo) is a long-video generation model. It supports an optional clean
first frame, ordered image/audio memory slots, and a stochastic few-step Distribution Matching Distillation (DMD)
sampler.
The pipeline generates synchronized video and audio.

Echo is implemented as a [ModularPipeline](/docs/diffusers/v0.41.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline) so its text encoding, memory conditioning, stochastic DMD denoising, and
decoding blocks can be run as a complete workflow or composed independently.

## Inference

Load the official [jdopensource/JoyAI-Echo](https://huggingface.co/jdopensource/JoyAI-Echo) checkpoint directly.
It uses the text encoder and tokenizer from [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it).
The Gemma repository is gated, so accept its license and authenticate with Hugging Face before loading the pipeline.

The released model uses 241 frames in its long-video example. The video RoPE coordinates remain at the training rate
of 24 fps, independently of the output container rate.

```py
import torch
import torchaudio
from PIL import Image

from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import encode_video

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("jdopensource/JoyAI-Echo", components_manager=manager)
pipe.load_components(dtype={"default": torch.bfloat16, "audio_vae": torch.float32})
manager.enable_auto_cpu_offload(device="cuda")
pipe.vae.enable_tiling()

first_frame = Image.open("first_frame.png").convert("RGB")
memory_images = [Image.open(path).convert("RGB") for path in ["memory_0.png", "memory_1.png"]]
memory_audio_with_rates = [torchaudio.load(path) for path in ["memory_0.wav", "memory_1.wav"]]
memory_audio = [waveform for waveform, _ in memory_audio_with_rates]
memory_audio_rates = [sample_rate for _, sample_rate in memory_audio_with_rates]

output = pipe(
    prompt="A cinematic dialogue scene in a quiet cafe.",
    image=first_frame,
    memory_images=memory_images,
    memory_audio_waveforms=memory_audio,
    memory_audio_sample_rates=memory_audio_rates,
    width=1280,
    height=736,
    num_frames=241,
    frame_rate=25.0,
    model_frame_rate=24.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="np",
    output=["videos", "audio"],
)

encode_video(
    output["videos"][0],
    fps=25,
    audio=output["audio"][0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="echo.mp4",
)
```

The default DMD sigma schedule is the released eight-step schedule. It predicts `x0` at every step and re-noises with
fresh Gaussian noise at the next sigma, so a seeded `torch.Generator` controls both the initial noise and all
intermediate re-noising.

Raw audio-memory encoding requires `torchaudio`. For reference parity, keep `audio_vae` in FP32 as shown above.
Modular workflows can cache the VAE encoder's normalized, unpacked tensors. Video latents have shape
`(batch, channels, frames, height, width)` and audio latents have shape `(batch, channels, time, mel_bins)`.
The core `denoise` block packs those tensors, expands conditioning for `num_videos_per_prompt`, and unpacks
denoised outputs back to the same VAE form. Pass initial `latents` and `audio_latents` in that unpacked form too.
Decoders take normalized VAE tensors and denormalize immediately before
decoding. `output=["latents", "audio_latents"]` returns normalized VAE tensors. With `output_type="latent"` and
`output=["videos", "audio"]`, you get the denormalized VAE tensors without decoding.

## EchoModularPipeline[[diffusers.EchoModularPipeline]]

#### diffusers.EchoModularPipeline[[diffusers.EchoModularPipeline]]

```python
diffusers.EchoModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/echo/modular_pipeline.py#L27)

A ModularPipeline for the Echo long-video DMD checkpoint.

## EchoBlocks[[diffusers.EchoBlocks]]

#### diffusers.EchoBlocks[[diffusers.EchoBlocks]]

```python
diffusers.EchoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/modular_pipelines/echo/modular_blocks_echo.py#L220)

Echo reference-to-video generation with clean first-frame conditioning, ordered image/audio memory slots, and
stochastic DMD denoising.

Components:
text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) vae
(`AutoencoderKLLTX2Video`) audio_vae (`AutoencoderKLLTX2Audio`) transformer (`LTX2VideoTransformer3DModel`)
video_processor (`VideoProcessor`) vocoder (`LTX2Vocoder`)

Inputs:
prompt (`str`):
The prompt or prompts to guide image generation.
max_sequence_length (`int`, *optional*, defaults to 1024):
Maximum sequence length for prompt encoding.
image (`Image | Tensor`, *optional*):
Optional single first frame used as a clean reference condition.
memory_images (`list`, *optional*):
Ordered reference images, one per Echo memory slot.
memory_audio_waveforms (`list`, *optional*):
Ordered memory waveforms as `(channels, samples)` tensors. Inputs longer than 9.62 seconds are cropped to
their highest-response window. Use `None` for a silent slot.
memory_audio_sample_rates (`int | list`, *optional*):
Sampling rate shared by all memory waveforms, or one rate per slot.
height (`int`, *optional*, defaults to 512):
The height in pixels of the generated image.
width (`int`, *optional*, defaults to 704):
The width in pixels of the generated image.
model_frame_rate (`float`, *optional*, defaults to 24.0):
Training-time frame rate used for Echo video RoPE coordinates.
memory_position_offset (`float`, *optional*, defaults to 500.0):
Temporal center assigned to the first memory slot.
memory_position_slot_stride (`float`, *optional*, defaults to 50.0):
Temporal distance between consecutive memory-slot centers.
num_videos_per_prompt (`int`, *optional*, defaults to 1):
Number of videos per prompt.
num_frames (`int`, *optional*, defaults to 241):
Number of generated pixel frames; must be `1 + k * vae_temporal_compression_ratio`.
frame_rate (`float`, *optional*, defaults to 25.0):
Frame rate of the generated video.
latents (`Tensor`, *optional*):
Optional initial video noise in VAE form (B, C, F, H, W).
audio_latents (`Tensor`, *optional*):
Optional initial audio noise in VAE form (B, C, L, M).
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
sigmas (`list | tuple`):
DMD sigma schedule, including the terminal zero.
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.
decode_timestep (`None`, *optional*, defaults to 0.0):
Timestep used to decode the final latents.
decode_noise_scale (`None`, *optional*):
Noise interpolation factor applied at the decode timestep.

Outputs:
videos (`list`):
The generated videos.
audio (`Tensor`):
The generated audio waveform.

### Motif-Video
https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/motif_video.md
