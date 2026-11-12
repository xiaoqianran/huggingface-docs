# Kandinsky S2V 1.0

Kandinsky S2V 1.0 generates an avatar video from a reference frame and a speech
track. The person in the frame is animated to match the speech — the whole figure,
not only the head. The speech is a condition, not an output: the model does not
generate audio, and the soundtrack of the result is the track you pass in.

Both conditions go through [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B):
its hidden states encode the prompt and its audio tower encodes the speech, which
enters the transformer through cross-attention adapters aligned frame by frame, so
each latent frame attends only to the speech it covers. A clip is generated in
chunks of 5 s (125 frames at 25 fps), each carrying FramePack memory of the
previous ones, so longer tracks stay consistent across chunk boundaries.

Kandinsky S2V 1.0 is integrated as a [modular pipeline](../../modular_diffusers/overview)
only; there is no standard `DiffusionPipeline` class for it.

## Checkpoints

| Checkpoint | Transformer | Blocks | Scheduler | Steps | Guidance |
|---|---|---|---|---|---|
| [Lite][lite] | 2B | [KandinskyS2VAutoBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/kandinsky_s2v#diffusers.KandinskyS2VAutoBlocks) | [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) | 50 | 3.0 |
| [Pro][pro] | 21B | [KandinskyS2VAutoBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/kandinsky_s2v#diffusers.KandinskyS2VAutoBlocks) | [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) | 50 | 3.0 |
| [Lite distilled][lite-distill] | 2B | [KandinskyS2VDistilledAutoBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/kandinsky_s2v#diffusers.KandinskyS2VDistilledAutoBlocks) | [PiflowScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/piflow#diffusers.PiflowScheduler) | 16 | off |
| [Pro distilled][pro-distill] | 21B | [KandinskyS2VDistilledAutoBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/kandinsky_s2v#diffusers.KandinskyS2VDistilledAutoBlocks) | [PiflowScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/piflow#diffusers.PiflowScheduler) | 16 | off |

The distilled checkpoints configure `PiflowScheduler` with `shift=10.0` and `n_grid=5`, not its defaults.

[lite]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-Diffusers
[pro]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers
[lite-distill]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers
[pro-distill]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers

`ModularPipeline.from_pretrained` picks the blocks from the checkpoint. The steps
default to the values above. Classifier-free guidance is the `guider` component,
not a call argument: the base checkpoints ship it at 3.0. The distilled ones have
no guider and no `negative_prompt`, because guidance is already part of the student.

## Usage

```py
import librosa
import torch
from diffusers import ModularPipeline
from diffusers.utils import encode_video, load_image

pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image("reference.png")
audio, _ = librosa.load("speech.wav", sr=16000)  # a mono waveform at 16 kHz, for the model
speech, sample_rate = librosa.load("speech.wav", sr=None)  # the original track, added to the output video

video = pipe(image=image, audio=audio, output="videos")[0]
encode_video(
    video,
    fps=25,
    output_path="avatar.mp4",
    audio=torch.from_numpy(speech)[None],
    audio_sample_rate=sample_rate,
)
```

The Pro checkpoints do not fit an 80 GB GPU next to the 7B speech and text encoder
when every component is on the device. Offload them through a `ComponentsManager`:

```py
import librosa
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import encode_video, load_image

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="40GiB")

image = load_image("reference.png")
audio, _ = librosa.load("speech.wav", sr=16000)
speech, sample_rate = librosa.load("speech.wav", sr=None)

video = pipe(image=image, audio=audio, output="videos")[0]
encode_video(
    video,
    fps=25,
    output_path="avatar.mp4",
    audio=torch.from_numpy(speech)[None],
    audio_sample_rate=sample_rate,
)
```

The margin is the memory kept free beyond the weights when a model is moved to the GPU.
It has to hold the VAE decode of a chunk, about 33 GiB at 512x768: with the default 3 GB
the transformer stays on the GPU through the decode, which then runs out of memory. With
40 GiB the peak is about 50 GiB on an 80 GB GPU. Enable the offload after `load_components`,
so it applies to every component.

## Notes

- `image` is a PIL image and `audio` a mono float waveform in [-1, 1] at 16 kHz, as a tensor
  or an array shaped `(samples,)`; the pipeline does not read files.
- The top-level blocks (`text_encoder`, `vae_encoder`, `denoise`, `decode`) run on their own too;
  pass the same `num_latent_frames` and `fps` to each when you run them separately.
- The number of chunks follows the length of the speech track unless `num_chunks`
  is passed. The video is trimmed to the track either way, so chunks past its end
  are computed and dropped, with a warning.
- With guidance, the unconditional pass swaps in the negative prompt *and* silences
  the speech, so the guidance is joint over text and speech.

## KandinskyS2VModularPipeline[[diffusers.KandinskyS2VModularPipeline]]

#### diffusers.KandinskyS2VModularPipeline[[diffusers.KandinskyS2VModularPipeline]]

```python
diffusers.KandinskyS2VModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/kandinsky_s2v/modular_pipeline.py#L91)

A ModularPipeline for Kandinsky S2V 1.0 speech-to-video generation.

## KandinskyS2VAutoBlocks[[diffusers.KandinskyS2VAutoBlocks]]

#### diffusers.KandinskyS2VAutoBlocks[[diffusers.KandinskyS2VAutoBlocks]]

```python
diffusers.KandinskyS2VAutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/kandinsky_s2v/modular_blocks_kandinsky_s2v.py#L108)

Modular pipeline for speech-to-video generation with Kandinsky S2V 1.0: an avatar video from a reference frame and
a speech track, generated chunk by chunk.

Components:
text_encoder (`Qwen2_5OmniThinkerForConditionalGeneration`) tokenizer (`Qwen2_5OmniProcessor`) text_encoder_2
(`CLIPTextModel`) tokenizer_2 (`CLIPTokenizer`) vae (`AutoencoderKLHunyuanVideo`) video_processor
(`VideoProcessor`) transformer (`KandinskyS2VTransformer3DModel`) scheduler
(`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`)

Inputs:
prompt (`str`, *optional*, defaults to The video presents a stable medium close-up of a subject facing the
camera directly. The subject exhibits natural, lifelike body dynamics: the head and shoulders move fluidly
with the rhythm of speech. The lighting is even, maintaining a balanced natural exposure to clearly define
facial features without washout. Throughout the sequence, the subject engages in continuous speech marked by
precise, high-fidelity articulation. The lips move in a rhythmic, natural pattern: alternating between brief
firm closures and relaxed openings. During these openings, the tips of the upper teeth are momentarily and
realistically visible, exhibiting natural enamel texture. The upper lip lifts subtly over the teeth during
enunciation, while the jaw performs minimal vertical oscillations synchronized with the speech cadence.
Facial micro-expressions are active yet restrained; eyebrows rise and lower slightly with minute asymmetry.
The eyes maintain a steady, engaged gaze forward, punctuated by smooth, deliberate blinks. ):
The prompt to guide the generation.
negative_prompt (`str`, *optional*):
The prompt for the unconditional pass of the guidance. Defaults to `DEFAULT_NEGATIVE_PROMPT` when the
guidance needs one.
audio (`Tensor | ndarray`):
The speech track that drives the generation: a mono float waveform in [-1, 1] at 16000 Hz, shaped
(samples,).
num_latent_frames (`int`, *optional*, defaults to 32):
Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).
fps (`int`, *optional*, defaults to 25):
Frame rate the chunk length is computed at.
num_chunks (`int`, *optional*):
Number of chunks to generate. Defaults to however many cover the speech track; the video is trimmed to
the track either way.
image (`Image`):
The reference frame to animate.
height (`int`, *optional*):
The height in pixels of the generated video. Defaults to the 512 bucket closest to the reference frame.
width (`int`, *optional*):
The width in pixels of the generated video, chosen together with `height`.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`, *optional*, defaults to 50):
The number of denoising steps.
scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
Per-axis RoPE scale for the resolution bucket.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
videos (`list`):
The generated videos.

## KandinskyS2VDistilledModularPipeline[[diffusers.KandinskyS2VDistilledModularPipeline]]

#### diffusers.KandinskyS2VDistilledModularPipeline[[diffusers.KandinskyS2VDistilledModularPipeline]]

```python
diffusers.KandinskyS2VDistilledModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/kandinsky_s2v/modular_pipeline.py#L109)

A ModularPipeline for the pi-Flow distilled Kandinsky S2V 1.0 checkpoints.

Guidance is distilled into the student, so its blocks have no guider and no negative prompt, and a single forward
pass runs per segment.

## KandinskyS2VDistilledAutoBlocks[[diffusers.KandinskyS2VDistilledAutoBlocks]]

#### diffusers.KandinskyS2VDistilledAutoBlocks[[diffusers.KandinskyS2VDistilledAutoBlocks]]

```python
diffusers.KandinskyS2VDistilledAutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/kandinsky_s2v/modular_blocks_kandinsky_s2v_distilled.py#L103)

Modular pipeline for speech-to-video generation with the pi-Flow distilled Kandinsky S2V 1.0 checkpoints: 16
segments per chunk and no guidance, which is distilled into the student.

Components:
text_encoder (`Qwen2_5OmniThinkerForConditionalGeneration`) tokenizer (`Qwen2_5OmniProcessor`) text_encoder_2
(`CLIPTextModel`) tokenizer_2 (`CLIPTokenizer`) vae (`AutoencoderKLHunyuanVideo`) video_processor
(`VideoProcessor`) transformer (`KandinskyS2VTransformer3DModel`) scheduler (`PiflowScheduler`)

Inputs:
prompt (`str`, *optional*, defaults to The video presents a stable medium close-up of a subject facing the
camera directly. The subject exhibits natural, lifelike body dynamics: the head and shoulders move fluidly
with the rhythm of speech. The lighting is even, maintaining a balanced natural exposure to clearly define
facial features without washout. Throughout the sequence, the subject engages in continuous speech marked by
precise, high-fidelity articulation. The lips move in a rhythmic, natural pattern: alternating between brief
firm closures and relaxed openings. During these openings, the tips of the upper teeth are momentarily and
realistically visible, exhibiting natural enamel texture. The upper lip lifts subtly over the teeth during
enunciation, while the jaw performs minimal vertical oscillations synchronized with the speech cadence.
Facial micro-expressions are active yet restrained; eyebrows rise and lower slightly with minute asymmetry.
The eyes maintain a steady, engaged gaze forward, punctuated by smooth, deliberate blinks. ):
The prompt to guide the generation.
audio (`Tensor | ndarray`):
The speech track that drives the generation: a mono float waveform in [-1, 1] at 16000 Hz, shaped
(samples,).
num_latent_frames (`int`, *optional*, defaults to 32):
Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).
fps (`int`, *optional*, defaults to 25):
Frame rate the chunk length is computed at.
num_chunks (`int`, *optional*):
Number of chunks to generate. Defaults to however many cover the speech track; the video is trimmed to
the track either way.
image (`Image`):
The reference frame to animate.
height (`int`, *optional*):
The height in pixels of the generated video. Defaults to the 512 bucket closest to the reference frame.
width (`int`, *optional*):
The width in pixels of the generated video, chosen together with `height`.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`, *optional*, defaults to 16):
The number of denoising steps.
scale_factor (`tuple`, *optional*, defaults to (1.0, 2.0, 2.0)):
Per-axis RoPE scale for the resolution bucket.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
videos (`list`):
The generated videos.

### Kandinsky 5.0 Video
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/kandinsky5_video.md
