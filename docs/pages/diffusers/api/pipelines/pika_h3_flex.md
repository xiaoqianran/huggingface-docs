# Pika H3 Flex

  
    
  

[Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) is Pika's post-training of [MiniMax-H3](./minimax_h3) into a six-step video-and-audio generator. It keeps MiniMax-H3's packed sequence, its Qwen3-VL conditioner and its two VAEs, and changes two things: the sampler, a fixed six-step schedule that re-noises the clean prediction at every step in place of the Euler update, and the weights, a pair of rank-128 LoRA adapters over the two MiniMax-H3 transformer partitions. Video and its synchronized stereo soundtrack come out of the same six forward passes.

The release is the adapters: the FL2VA adapter (`fl2va/adapter.safetensors`) drives the `transformer/` partition for text and endpoint-frame conditioning, the Ref2VA adapter (`ref2va/adapter.safetensors`) drives `transformer_ref/` for multimodal references. The base components are [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)'s, unchanged, and the repository carries a copy of them so that it loads on its own.

Pika H3 Flex is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only, like MiniMax-H3: the blocks and their [PikaH3FlexModularPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/pika_h3_flex#diffusers.PikaH3FlexModularPipeline) are the whole integration, and there is no `DiffusionPipeline` half.

## Loading

A pipeline is assembled in two steps: the components are loaded through a `modular_model_index.json` that names the MiniMax-H3 base components and the two scheduler configs, then the adapter of the selected workflow is loaded on top with [load_lora_weights()](/docs/diffusers/v0.43.0/en/api/loaders/lora#diffusers.loaders.MiniMaxH3LoraLoaderMixin.load_lora_weights). The adapters are plain PEFT dumps of the diffusers module names and load at their trained scale (`alpha == rank`, 1.0).

| Workflow | Inputs | Transformer partition | Adapter |
|---|---|---|---|
| `t2va` | `prompt` | `transformer/` | `fl2va/adapter.safetensors` |
| `fl2va` | `prompt` and `image` and/or `last_image` | `transformer/` | `fl2va/adapter.safetensors` |
| `ref2va` | `prompt` and `references` | `transformer_ref/` | `ref2va/adapter.safetensors` |

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", workflow="fl2va")
pipe.load_components(dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")
```

Passing the workflow prunes the blocks to that task, so only its own transformer partition is fetched, the adapter routes to the one partition present, and the pipeline accepts that task's requests alone — a `fl2va` pipeline needs a keyframe, a `t2va` one takes none. A pipeline that holds both partitions (loaded without a `workflow=`) needs to be told where the Ref2VA adapter goes: `pipe.load_lora_weights(..., weight_name="ref2va/adapter.safetensors", load_into_transformer_ref=True)`. The two VAEs are released in float32 and decode in float32, as the released package does; everything else runs in bfloat16.

> [!TIP]
> `pipe.doc` prints what the pipeline in front of you takes and returns — every input with its default, the components it expects and the outputs it produces.

### Reusing a loaded MiniMax-H3

The base components are MiniMax-H3's, so a MiniMax-H3 pipeline that is already loaded holds everything except the two schedulers and the adapters. This section assembles a Pika H3 Flex pipeline from those loaded components and keeps both pipelines ready side by side: one [ComponentsManager](/docs/diffusers/v0.43.0/en/api/modular_diffusers/pipeline_components#diffusers.ComponentsManager) holds each component once and offloads it for both, the Pika H3 Flex pipeline adds only its two schedulers, and the adapters decide which of the two models the shared transformers run.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

dtype = {"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16}
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

# The MiniMax-H3 pipeline already in use, with every workflow loaded. `collection` registers its components in the
# manager under that name, so they can be fetched by it later.
minimax = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager, collection="minimax-h3")
minimax.load_components(dtype=dtype)

# The Pika H3 Flex pipeline, in its own collection. Only its own components are loaded: the two schedulers.
pika = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", components_manager=manager, collection="pika-h3-flex")
pika.load_components(names=["scheduler", "audio_scheduler"])

# The rest comes from the MiniMax-H3 collection, and `update_components` puts it in the Pika H3 Flex pipeline.
shared = ["text_encoder", "tokenizer", "processor", "vae", "audio_vae", "transformer", "transformer_ref"]
pika.update_components(**{name: manager.get_one(name=name, collection="minimax-h3") for name in shared})

# The adapters go into the shared transformers.
pika.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")
pika.load_lora_weights(
    "Pika-Labs/Pika-H3-Flex", weight_name="ref2va/adapter.safetensors", load_into_transformer_ref=True
)
```

`print(manager)` shows the result: every shared component is listed once and belongs to both collections, the Pika H3 Flex collection adds its two schedulers, and the two transformers carry an adapter.

```text
Models:
Name_ID               | Class                           | Size (GB) | Collection
transformer_1397...   | MiniMaxH3Transformer3DModel     | 63.02     | minimax-h3
                      |                                 |           | pika-h3-flex
transformer_ref_13... | MiniMaxH3Transformer3DModel     | 63.02     | minimax-h3
                      |                                 |           | pika-h3-flex
text_encoder_1397...  | Qwen3VLForConditionalGeneration | 62.13     | minimax-h3
                      |                                 |           | pika-h3-flex
vae_1397...           | AutoencoderKLMiniMaxH3          | 9.70      | minimax-h3
                      |                                 |           | pika-h3-flex
audio_vae_1397...     | AutoencoderKLMiniMaxH3Audio     | 0.56      | minimax-h3
                      |                                 |           | pika-h3-flex

Other Components:
ID                        | Class              | Collection
scheduler_1397...         | MiniMaxH3Scheduler | minimax-h3
audio_scheduler_1397...   | MiniMaxH3Scheduler | minimax-h3
scheduler_1397...         | MiniMaxH3Scheduler | pika-h3-flex
audio_scheduler_1397...   | MiniMaxH3Scheduler | pika-h3-flex
tokenizer_1397...         | Qwen2Tokenizer     | minimax-h3
                          |                    | pika-h3-flex
processor_1397...         | Qwen3VLProcessor   | minimax-h3
                          |                    | pika-h3-flex

Additional Component Info:
transformer_1397...:
  Adapters: ['default_0']
transformer_ref_13...:
  Adapters: ['default_0']
```

The adapters decide which model the shared transformers are. To go back to MiniMax-H3, call `pika.disable_lora()`, and that is all: the next `minimax(...)` request runs the base model. `pika.enable_lora()` turns Pika H3 Flex back on. Nothing is reloaded in between; the manager moves each component on and off the accelerator as either pipeline reaches it.

## What differs from MiniMax-H3

- **Six steps, no `num_inference_steps`.** Both [MiniMaxH3Scheduler](/docs/diffusers/v0.43.0/en/api/schedulers/minimax_h3#diffusers.MiniMaxH3Scheduler) instances (`shift=12.0` for video, `shift=3.0` for audio) run with `stochastic_sampling` over the pipeline's `denoising_step_list` config, the released schedule `[1000, 750, 500, 250, 100, 40, 0]`: every step predicts the clean sample, and all but the last re-noise it with fresh noise to the next sigma. The step count is the adapter's contract and is not an input.
- **Fresh noise at every step.** A request draws from its `generator` in a fixed order: one draw per keyframe or visual reference for the conditioning noise, the video noise, the audio noise, and then the video and audio re-noising draws of each of the first five steps. Passing `latents` or `audio_latents` replaces the two initial draws and leaves the rest in place.
- **Keyframes are cropped, not stretched.** A `fl2va` keyframe is center-cropped to the canvas' aspect ratio at its own resolution and resized onto the canvas.
- **References follow the canvas' resolution tier.** Pika H3 Flex generates on a bucket table — a 480p tier and a 768p tier, eleven aspect ratios each — and a `ref2va` request conditions by the tier of its canvas: an image reference is encoded at a 1024 pixel short edge below the 768 tier and 2048 from it upwards, and a video reference is put on the tier's own canvas rule. Every video or standalone audio reference runs 2 to 15 seconds, and the video durations and the standalone audio durations each sum to at most 15 seconds.

## Generation

The released buckets are the canvases Pika H3 Flex was validated on. Pass one as `height` and `width`:

| | 21:9 | 16:9 | 3:2 | 4:3 | 5:4 | 1:1 | 4:5 | 3:4 | 2:3 | 9:16 | 9:21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 480p | 960x416 | 832x480 | 704x480 | 640x480 | 608x480 | 480x480 | 480x608 | 480x640 | 480x704 | 480x832 | 416x960 |
| 768p | 1536x672 | 1344x768 | 1152x768 | 1024x768 | 960x768 | 768x768 | 768x960 | 768x1024 | 768x1152 | 768x1344 | 672x1536 |

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

# 61.7GB of transformer and 62.1GB of conditioner do not sit on one accelerator, so the components are
# registered in a manager that moves each one on and off as the blocks reach it.
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")  # or "mps", "xpu", "cpu"

# Create one pipeline for all supported workflows, then load the `fl2va` workflow's components, which is
# enough to run both the fl2va and t2va requests below.
pipe = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", components_manager=manager)
pipe.load_components(
    workflow="fl2va", dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16}
)
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")

prompt = (
    "integrated_multimodal_description: A cinematic close shot of a red ceramic cup on a wooden table beside an "
    "open window. A soft breeze moves a linen curtain. The camera slowly pushes toward the cup as warm morning "
    "light crosses the table.\n"
    "overall_soundscape: Gentle wind through the window and faint birdsong outside. No speech.\n"
    "non_diegetic_music: None.\n"
)
outputs = ["videos", "audio", "sampling_rate"]

# Text to video + audio on the 480p 16:9 bucket, 5 seconds.
results = pipe(
    prompt=prompt, 
    height=480, 
    width=832, 
    num_frames=124, 
    generator=torch.Generator("cuda").manual_seed(42), 
    output=outputs
)

# First frame (and optionally `last_image`) to video + audio. The keyframe is center-cropped onto the canvas and
# held as the first frame, so the prompt describes the keyframe and what happens from it.
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
i2va_prompt = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    "integrated_multimodal_description: [Shot 1] Begin with the supplied image of a small toy astronaut figurine "
    "standing on a grey rocky moon surface, holding a white egg-shaped object. Preserve the figurine, its orange "
    "visor and the rocks. The astronaut slowly turns its helmet toward the camera and lifts the egg a little; fine "
    "dust drifts up around its boots. The camera slowly pushes in in one continuous shot. By the end the astronaut "
    "stands still, egg held up.\n\n"
    "overall_soundscape: Soft low wind and faint crunch of dust under the boots. No speech.\n\n"
    "non_diegetic_music: N/A\n"
)
results = pipe(
    prompt=i2va_prompt, 
    image=image, 
    height=480, 
    width=832, 
    num_frames=124,
    generator=torch.Generator("cuda").manual_seed(42), 
    output=outputs,
)

encode_video(
    results["videos"][0],
    fps=24,
    output_path="pika_h3_flex_fl2va.mp4",
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
```

The `ref2va` workflow takes an ordered list of references — up to 9 images, 3 videos and 3 audio clips, 12 in total — the way MiniMax-H3 does, with one reference class per modality. It is also the request that needs the most memory: the references add their own rows to the sequence, and at 768p the transformer alone fills an 80 GB card. The example below therefore streams the transformer's blocks from CPU memory one at a time, while the manager moves the other components on and off the device as before. It runs a 768p reference request on one 80 GB card; without the streaming, keep to 480p.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.pika_h3_flex import (
    PikaH3FlexAudioReference,
    PikaH3FlexImageReference,
    PikaH3FlexVideoReference,
)

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")  # or "mps", "xpu", "cpu"

pipe = ModularPipeline.from_pretrained(
    "Pika-Labs/Pika-H3-Flex", workflow="ref2va", components_manager=manager
)
pipe.load_components(dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="ref2va/adapter.safetensors")

# Keep the transformer in CPU memory and bring in one block at a time while it runs.
pipe.transformer_ref.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=1,
    use_stream=True,
)

results = pipe(
    prompt="subject_definitions: <Subject 1> is the woman in <Picture 1>. summary: [reference generation] <Subject 1> waves at the camera.",
    references=[
        PikaH3FlexImageReference.from_file("subject.png"),
        PikaH3FlexVideoReference.from_file("motion_ref.mp4"),  # frames, their `fps` and the soundtrack
        PikaH3FlexAudioReference.from_file("voice.wav"),  # waveform and its `sample_rate`
    ],
    height=768,
    width=1344,
    num_frames=124,
    generator=torch.Generator("cuda").manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)
```

## PikaH3FlexModularPipeline[[diffusers.PikaH3FlexModularPipeline]]

#### diffusers.PikaH3FlexModularPipeline[[diffusers.PikaH3FlexModularPipeline]]

```python
diffusers.PikaH3FlexModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/modular_pipeline.py#L211)

A ModularPipeline for joint video + audio generation with Pika H3 Flex, the post-trained six-step MiniMax-H3: the
`t2va` (text only) and `fl2va` (first and/or last keyframe) workflows against the `transformer/` checkpoint
partition with the `fl2va` adapter, and the `ref2va` (omni-reference) workflow against `transformer_ref/` with the
`ref2va` adapter.

The base components are MiniMax-H3's, and the Pika H3 Flex release is a pair of rank-128 LoRA adapters on top of
the two transformer partitions, so a pipeline is assembled in two steps: the components are loaded through a
`modular_model_index.json` that names the base components and the two schedulers, and the adapter of the selected
workflow is loaded with [load_lora_weights()](/docs/diffusers/v0.43.0/en/api/loaders/lora#diffusers.loaders.MiniMaxH3LoraLoaderMixin.load_lora_weights).

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", workflow="fl2va")
pipe.load_components(dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")
```

Like MiniMax-H3, Pika H3 Flex denoises **one packed sequence** that holds the text conditioning, the keyframe or
reference conditioning latents, the audio latents and the video latents at once, which is why the blocks pass a row
layout around rather than per-modality tensors, and why the pipeline carries two schedulers (`shift = 12.0` for
video, `shift = 3.0` for audio) that are stepped inside a single transformer call. Unlike MiniMax-H3, the two
schedulers are [MiniMaxH3Scheduler](/docs/diffusers/v0.43.0/en/api/schedulers/minimax_h3#diffusers.MiniMaxH3Scheduler) instances with `stochastic_sampling` over the fixed six-step
`denoising_step_list`: there is no `num_inference_steps`, and every step draws fresh noise from the request's
generator.

The adapters are guidance-distilled like their base: there is no guider, no `negative_prompt` and no
`guidance_scale`, and every step runs exactly one forward pass.

This module carries the model facts every block keys off — the config-derived geometry as properties, the canvas,
tier and frame-count arithmetic as functions — and the conditioning, encoding and noise contracts live on the
blocks themselves.

## PikaH3FlexBlocks[[diffusers.PikaH3FlexBlocks]]

#### diffusers.PikaH3FlexBlocks[[diffusers.PikaH3FlexBlocks]]

```python
diffusers.PikaH3FlexBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/modular_blocks_pika_h3_flex.py#L668)

Auto Modular pipeline blocks for joint video + audio generation with Pika H3 Flex: the `t2va` (text only), `fl2va`
(first and/or last keyframe) and `ref2va` (omni-reference) workflows, selected on the `references` and keyframe
inputs. The `t2va` and `fl2va` workflows run the `transformer` partition with the `fl2va` adapter and `ref2va` the
`transformer_ref` partition with the `ref2va` adapter, so load the adapter of the workflow the request takes.
Without a `workflow=`, loading the components pulls **both** 61.7GB transformer partitions — pass one to load only
what the task needs.

Supported workflows:
- `t2va`: requires `prompt`
- `fl2va`: requires `prompt`, `image` or `prompt`, `last_image`
- `ref2va`: requires `prompt`, `references`

Components:
image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
(`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`) vae (`AutoencoderKLMiniMaxH3`) audio_vae
(`AutoencoderKLMiniMaxH3Audio`) scheduler (`MiniMaxH3Scheduler`) audio_scheduler (`MiniMaxH3Scheduler`)
transformer_ref (`MiniMaxH3Transformer3DModel`) transformer (`MiniMaxH3Transformer3DModel`) video_processor
(`VideoProcessor`)

Configs:
canvas_short_edge (default: 768) canvas_max_pixels (default: 1032192) denoising_step_list (default: [1000,
750, 500, 250, 100, 40, 0])

Inputs:
references (`list`, *optional*):
The references to condition on, **in the order the model should read them**: the order labels them in the
prompt presentation and lays them out on the shared rotary clock, so a different order is a different
request. One dataclass per modality, all holding in-memory media — a `PikaH3FlexImageReference` (at
most 9), a `PikaH3FlexVideoReference` at its own `fps` (at most 3, whose `audio` soundtrack is
conditioned on as well), or a `PikaH3FlexAudioReference` at its own `sample_rate` (at most 3) — for at
most 12 references in total, and audio references cannot be the only ones. Every video or standalone
audio reference runs 2 to 15 seconds, and the video durations and the standalone audio durations each sum
to at most 15 seconds. These blocks never open a media file: decode with each class's `from_file`
classmethod, which brings the rates along.
height (`int`, *optional*):
Height of the generated video in pixels, a multiple of 32.
width (`int`, *optional*):
Width of the generated video in pixels, a multiple of 32.
num_frames (`int`, *optional*):
Number of frames to generate, at the fixed 24 fps: 120 to 360, i.e. 5 to 15 seconds, snapped up to the
next `17 * n + 5` the video VAE can decode. To generate a video as long as a reference soundtrack, pass
`round(samples / sample_rate * 24)`.
image (`Image`, *optional*):
Keyframe the video starts from. It is center-cropped to the canvas' aspect ratio and resized onto it; the
canvas by default is derived from its own aspect ratio.
last_image (`Image`, *optional*):
Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. It is center-cropped to
the canvas' aspect ratio and resized onto it, like `image`.
prompt (`str`):
The prompt to guide generation, a single string.
normalized_references (`list`, *optional*):
The references normalized by the setup step, in packed order.
keyframes (`list`, *optional*):
The keyframes put onto the target canvas, in packed order.
condition_latents (`list`, *optional*):
The encoded video conditioning latents, one per image and video reference in packed order. Their shape is
where every reference block's geometry comes from.
audio_condition_latents (`list`, *optional*):
The encoded audio conditioning rows, one per audio-bearing reference in packed order.
generator (`Generator`, *optional*):
The generator of the request. The conditioning noise is drawn from it first, one draw per condition,
before the noise of the generated rows.
latents (`Tensor`):
Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used
instead of the draw.
audio_latents (`Tensor`):
Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
**denoiser_input_fields (`None`, *optional*):
The structural description of the packed sequence the transformer reads by name: `token_tags`,
`position_ids` and the three row-index tensors.
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
keyframe_anchors (`tuple`, *optional*, defaults to ()):
Which end of the video every keyframe is anchored to, in packed order.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np' or 'pt'.

Outputs:
videos (`list`):
The generated video.
audio (`Tensor`):
The generated soundtrack, of shape `(1, 2, num_samples)`.
sampling_rate (`int`):
Sample rate of the generated soundtrack in Hz.

## PikaH3FlexImageReference[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexImageReference]]

#### diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexImageReference[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexImageReference]]

```python
diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexImageReference(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/references.py#L83)

**Parameters:**

image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`) : The reference image: an image, a `(height, width, 3)` array or a `(3, height, width)` tensor, `uint8` or floating point over `[0, 1]`. It never binds the generated geometry — it is encoded at a short edge of its own, 2048 for the released checkpoint, whatever canvas the request generates at.

A subject, style or scene reference: at most 9 per request.

#### from_file[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexImageReference.from_file]]

```python
from_file(media)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/references.py#L99)

**Parameters:**

media (`str` or `os.PathLike`) : Path or URL of the image.

Load an image file into a `PikaH3FlexImageReference`, through [load_image()](/docs/diffusers/v0.43.0/en/api/utilities#diffusers.utils.load_image).

## PikaH3FlexVideoReference[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexVideoReference]]

#### diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexVideoReference[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexVideoReference]]

```python
diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexVideoReference(frames: typing.Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor], fps: float | None = None, audio: typing.Optional[torch.Tensor] = None, sample_rate: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/references.py#L111)

**Parameters:**

frames (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`) : The reference frames: a list of images, a `(num_frames, height, width, 3)` array or a `(num_frames, 3, height, width)` tensor, `uint8` or floating point over `[0, 1]`.

fps (`float`, *optional*, defaults to 24.0) : The frame rate `frames` carries, which is what places the reference's vision blocks on the conditioner's 2 fps grid. MiniMax-H3's own clock is 24 fps, so any other rate is resampled onto it by dropping and duplicating whole frames — which makes this the field to get right when the frames came from a file.

audio (`torch.Tensor` of shape `(channels, num_samples)`, *optional*) : This video's soundtrack, mono or stereo, conditioned on as the reference's own rather than as a reference of its own. Left out, the reference conditions on motion alone.

sample_rate (`int`, *optional*) : The rate `audio` carries its samples at. Left out, it is the audio VAE's own, which leaves the samples untouched; any other rate is resampled onto it.

A motion and camera reference: at most 3 per request, conditioned on together with its own soundtrack.

#### from_file[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexVideoReference.from_file]]

```python
from_file(media)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/references.py#L147)

**Parameters:**

media (`str` or `os.PathLike`) : Path or URL of the video.

**Returns:** `PikaH3FlexVideoReference`

the `(num_frames, height, width, 3)` `uint8` frames at the frame rate the
container reports, carrying its soundtrack and that soundtrack's own sample rate when it has an audio
stream.

Decode a video file into a `PikaH3FlexVideoReference`, at the resolution, the frame rate and the soundtrack
it carries.

The rates land on the reference, which is the point of decoding this way rather than with
[load_video()](/docs/diffusers/v0.43.0/en/api/utilities#diffusers.utils.load_video): MiniMax-H3 resamples a reference onto its own 24 fps, so a frame rate lost on the way in
is a request conditioned at the wrong speed, with nothing to raise about it. A container whose metadata is
wrong is corrected by overriding `fps` or `sample_rate` on the returned reference.

Needs [PyAV](https://github.com/PyAV-Org/PyAV).

## PikaH3FlexAudioReference[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference]]

#### diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference]]

```python
diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference(audio: Tensor, sample_rate: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/references.py#L174)

**Parameters:**

audio (`torch.Tensor` of shape `(channels, num_samples)`) : The reference waveform, mono or stereo.

sample_rate (`int`, *optional*) : The rate `audio` carries its samples at. Left out, it is the audio VAE's own, which leaves the samples untouched; any other rate is resampled onto it.

A voice or music reference: at most 3 per request, and never on its own — an audio reference has to be paired with
at least one image or video reference. It never reaches the conditioner and is encoded by the audio VAE alone.

#### from_file[[diffusers.modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference.from_file]]

```python
from_file(media)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/pika_h3_flex/references.py#L193)

**Parameters:**

media (`str` or `os.PathLike`) : Path or URL of the audio, or of a video whose soundtrack is taken.

**Returns:** `PikaH3FlexAudioReference`

the `(channels, num_samples)` float32 waveform at the sample rate the
container reports.

Decode an audio file into a `PikaH3FlexAudioReference`, at the sample rate it carries.

Needs [PyAV](https://github.com/PyAV-Org/PyAV).

### Flux
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/flux.md
