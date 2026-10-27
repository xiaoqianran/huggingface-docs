# SANA-WM

SANA-WM is a camera-controlled image-to-video world model built on top of SANA. Given a first-frame image, a text
prompt, and a camera trajectory (either explicit `c2w` poses or a WASD/IJKL action string), it generates a video
whose motion follows the requested camera path.

Inference runs in two stages:

1. **Stage 1 — SANA-WM DiT.** A 1.6B-parameter bidirectional DiT with GDN-Triton linear attention and a UCPE
   camera-control branch. Sampling uses an LTX-style flow-matching Euler scheduler with per-token timesteps; the
   first latent frame is the conditioning anchor.
2. **Stage 2 — LTX-2 refiner (optional).** A separate sink-bidirectional Euler refiner pipeline
   ([SanaWMLTX2Refiner](/docs/diffusers/v0.42.1/en/api/pipelines/sana_wm#diffusers.SanaWMLTX2Refiner)) that wraps
   [SanaWMLTX2RefinerTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/sana_wm_refiner_transformer3d#diffusers.SanaWMLTX2RefinerTransformer3DModel) + `LTX2TextConnectors` and a Gemma-3 text encoder, run for 3
   distilled sigma steps.

Both stages decode through the [AutoencoderKLLTX2Video](/docs/diffusers/v0.42.1/en/api/models/autoencoderkl_ltx_2#diffusers.AutoencoderKLLTX2Video) VAE.

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`Efficient-Large-Model/SANA-WM_bidirectional-diffusers`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers) | `torch.bfloat16` |
| [`Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner`](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner) | `torch.bfloat16` |

> [!TIP]
> SANA-WM is trained at a fixed 704×1280 resolution. The recommended dtype is for the transformer weights — keep
> the text encoder in `torch.bfloat16` and the VAE in `torch.float32` for best numerics. The pipeline expects
> camera intrinsics `[fx, fy, cx, cy]` in *original-image* pixel coordinates; the resize-and-center-crop transform
> is applied internally.

## Inference

```python
import torch
from PIL import Image

from diffusers import SanaWMPipeline
from diffusers.utils import export_to_video

pipe = SanaWMPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # ~45 GB of weights — offload between stages

# SANA-WM was trained on the LTX-2 VAE in framewise mode with tiling enabled. Without these
# settings the VAE encodes the whole (B, C, T, H, W) clip in one shot, which gives subtly
# different numerics from the released checkpoint.
pipe.vae.enable_tiling()
pipe.vae.use_framewise_encoding = True
pipe.vae.use_framewise_decoding = True
pipe.vae.tile_sample_stride_num_frames = 64
pipe.vae.tile_sample_min_num_frames = 96

prompt = "A car driving across a vast desert plain at golden hour."
output = pipe(
    image=Image.open("input.png").convert("RGB"),
    prompt=prompt,
    action="w-80,jw-40,w-40",        # WASD-style action DSL: forward 80f, jump+forward 40f, forward 40f
    intrinsics=[800.0, 800.0, 845.0, 464.0],  # fx, fy, cx, cy in original-image pixels
    num_frames=161,
    num_inference_steps=60,
    guidance_scale=5.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="latent",            # hand the latents to the refiner below
)
```

Pass `action=None` and supply your own `c2w` poses (`(F, 4, 4)` numpy array) to drive the camera trajectory
explicitly. Drop `output_type="latent"` to get video straight out of stage 1 and skip the refiner.

### Stage 2 — the LTX-2 refiner

[SanaWMLTX2Refiner](/docs/diffusers/v0.42.1/en/api/pipelines/sana_wm#diffusers.SanaWMLTX2Refiner) ships as its own repository, the way SDXL splits base and refiner. Pass the base pipeline's
VAE so the weights are shared rather than loaded twice:

```python
from diffusers import SanaWMLTX2Refiner

refiner = SanaWMLTX2Refiner.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner",
    vae=pipe.vae,
    torch_dtype=torch.bfloat16,
)
refiner.enable_model_cpu_offload()

frames = refiner(output.latent, prompt, fps=16)
export_to_video(list(frames), "sana_wm.mp4", fps=16)
```

Without a `vae` the refiner returns refined latents instead of video, which is useful if you want to decode
yourself.

> [!TIP]
> Enable offloading on **both** pipelines, or free stage 1 before stage 2 (`pipe.transformer.to("cpu")`). The two
> stages together are around 45 GB in `torch.bfloat16`, and keeping both resident on one 80 GB card leaves too
> little room for activations. Note also that stage 2 honours `torch_dtype`: loading the refiner in
> `torch.float32` roughly doubles its memory and changes the output slightly.

If you don't have camera intrinsics, a hosted [modular block](../../modular_diffusers/overview) can estimate them
from a single frame. It lives outside `diffusers` because it pulls in Pi3X — an extra dependency and a second
checkpoint — so nothing is downloaded until you ask for it:

```python
from diffusers import ModularPipeline

# One-time per image. Requires `pip install pi3-vision`.
estimator = ModularPipeline.from_pretrained(
    "Efficient-Large-Model/pi3x-intrinsics-estimator", trust_remote_code=True
)
estimator.load_components(dtype=torch.bfloat16)
intrinsics = estimator(image=Image.open("input.png").convert("RGB"), output="intrinsics")
```

## Converting the released checkpoint

If you have the source SANA-WM release (not the pre-converted diffusers snapshot), run the conversion script once:

```bash
python scripts/convert_sana_wm_to_diffusers.py \
    --src Efficient-Large-Model/SANA-WM_bidirectional \
    --dst ./SANA-WM_bidirectional-diffusers
```

This writes two directories: the base pipeline at `--dst`, and the stage-2 refiner alongside it at
`./SANA-WM_bidirectional-diffusers-refiner` (override with `--dst-refiner`). Then load each from its local path
as usual.

## Components

- `tokenizer` — `GemmaTokenizerFast`
- `text_encoder` — Gemma-2 (returns decoder hidden states)
- `vae` — [AutoencoderKLLTX2Video](/docs/diffusers/v0.42.1/en/api/models/autoencoderkl_ltx_2#diffusers.AutoencoderKLLTX2Video) (LTX-2, spatial ×32 / temporal ×8)
- `transformer` — [SanaWMTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/sana_wm_transformer3d#diffusers.SanaWMTransformer3DModel), 1.6B-parameter bidirectional DiT
- `scheduler` — [FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)

The stage-2 refiner is a separate repository with its own `transformer`
([SanaWMLTX2RefinerTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/sana_wm_refiner_transformer3d#diffusers.SanaWMLTX2RefinerTransformer3DModel)), `connectors` (`LTX2TextConnectors`), `tokenizer`, Gemma-3
`text_encoder` and `scheduler`. It has no `vae` of its own — pass the base pipeline's.

## SanaWMPipeline[[diffusers.SanaWMPipeline]]

#### diffusers.SanaWMPipeline[[diffusers.SanaWMPipeline]]

```python
diffusers.SanaWMPipeline(tokenizer: GemmaTokenizer | GemmaTokenizerFast, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderKLLTX2Video, transformer: SanaWMTransformer3DModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_sana_wm.py#L143)

**Parameters:**

tokenizer ([*GemmaTokenizer*] or [*GemmaTokenizerFast*]) : The Gemma-2 tokenizer.

text_encoder ([*Gemma2PreTrainedModel*]) : The Gemma-2 text encoder.

vae ([*AutoencoderKLLTX2Video*]) : The LTX-2 VAE.

transformer ([*SanaWMTransformer3DModel*]) : The 1600M bidirectional SANA-WM DiT.

scheduler ([*FlowMatchEulerDiscreteScheduler*]) : Flow-matching Euler scheduler (LTX-style per-token timesteps). latents directly.

SANA-WM camera-controlled image-to-video pipeline.

Generates a video from a first-frame image, a text prompt, and a camera trajectory (explicit `c2w` poses or a
WASD/IJKL action string). Uses the 1600M bidirectional SANA DiT for stage-1 sampling and the LTX-2
sink-bidirectional Euler refiner for stage-2 polish; both decode through the LTX-2 VAE.

#### __call__[[diffusers.SanaWMPipeline.__call__]]

```python
__call__(image: PIL.Image.Image | str | Path, prompt: str, c2w: np.ndarray | None = None, action: str | None = None, intrinsics: np.ndarray | list[float] | None = None, height: int = 704, width: int = 1280, num_frames: int = 161, fps: int = 16, num_inference_steps: int = 60, guidance_scale: float = 5.0, negative_prompt: str = '', generator: torch.Generator | list[torch.Generator] | None = None, prompt_embeds: torch.Tensor | None = None, prompt_attention_mask: torch.Tensor | None = None, negative_prompt_embeds: torch.Tensor | None = None, negative_prompt_attention_mask: torch.Tensor | None = None, max_sequence_length: int = 300, chi_prompt: list[str] | None = None, output_type: Literal['np', 'pil', 'pt', 'latent'] = 'np', callback_on_step_end: Callable[[DiffusionPipeline, int, int, dict], dict] | None = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_sana_wm.py#L457)

**Parameters:**

image (*PIL.Image.Image* or *str*) : First-frame image (PIL or path).

prompt (*str*) : Text prompt.

c2w (*np.ndarray*, *optional*) : `(F, 4, 4)` camera-to-world poses. Mutually exclusive with *action*.

action (*str*, *optional*) : Action-DSL string e.g. `"w-80,jw-40,w-40"`. Mutually exclusive with *c2w*.

intrinsics (*np.ndarray* or *list[float]*) : `[fx, fy, cx, cy]` in **original-image** pixel coordinates. The pipeline applies the resize+crop transform internally.

height (*int*, defaults to 704) : Output frame height (fixed for the public model).

width (*int*, defaults to 1280) : Output frame width (fixed for the public model).

num_frames (*int*, defaults to 161) : Target frame count; snapped to `8k+1` (LTX-2 VAE constraint).

fps (*int*, defaults to 16) : Output frame rate (also fed to the refiner).

num_inference_steps (*int*, defaults to 60) : Number of stage-1 DiT sampling steps.

guidance_scale (*float*, defaults to 5.0) : Classifier-free guidance scale.

negative_prompt (*str*, defaults to "") : Optional negative prompt.

generator (*torch.Generator* or *list[torch.Generator]*, *optional*) : One or more torch generators to make the noise sampling deterministic. If *None*, sampling is non-deterministic.

prompt_embeds (*torch.Tensor*, *optional*) : Pre-computed text embeddings, to skip the text encoder. Must be passed together with *prompt_attention_mask*.

prompt_attention_mask (*torch.Tensor*, *optional*) : Attention mask for *prompt_embeds*.

negative_prompt_embeds (*torch.Tensor*, *optional*) : Pre-computed negative text embeddings. Must be passed together with *negative_prompt_attention_mask*.

negative_prompt_attention_mask (*torch.Tensor*, *optional*) : Attention mask for *negative_prompt_embeds*.

max_sequence_length (*int*, defaults to 300) : Max prompt tokens.

chi_prompt (*list[str]*, *optional*) : Override the chi-prompt prefix (default mirrors the public release).

output_type (*"np"*, *"pil"*, *"pt"*, or *"latent"*, defaults to *"np"*) : Output format.

callback_on_step_end (*Callable*, *optional*) : Called at the end of each denoising step as *callback_on_step_end(self, step, timestep, callback_kwargs)*, where *callback_kwargs* holds the tensors named in *callback_on_step_end_tensor_inputs*. Tensors in the returned dict replace the pipeline's.

callback_on_step_end_tensor_inputs (*list[str]*, defaults to *["latents"]*) : Which tensors to pass to *callback_on_step_end*; must be a subset of *._callback_tensor_inputs*.

return_dict (*bool*, defaults to True) : Return [*SanaWMPipelineOutput*] vs tuple.

**Returns:**

[*SanaWMPipelineOutput*] with *.frames* of shape `(T, H, W, 3)`, float `np.ndarray` in `[0, 1]` for
*output_type="np"*, a list of `PIL.Image.Image` of length `T` for *"pil"*, a `(T, 3, H, W)` tensor
for *"pt"*, or the raw latent tensor for *"latent"*.

Generate a SANA-WM camera-controlled video.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import SanaWMPipeline

>>> pipe = SanaWMPipeline.from_pretrained(
...     "Efficient-Large-Model/SANA-WM_bidirectional-diffusers", torch_dtype=torch.bfloat16
... ).to("cuda")

>>> output = pipe(
...     image=Image.open("input.png").convert("RGB"),
...     prompt="A car driving across a vast desert plain at golden hour.",
...     action="w-80,jw-40,w-40",
...     intrinsics=[800.0, 800.0, 845.0, 464.0],  # fx, fy, cx, cy in original-image pixels
...     num_inference_steps=60,
... )
>>> # output.frames is (T, H, W, 3) float np.ndarray in [0, 1] (diffusers convention).
```

#### check_inputs[[diffusers.SanaWMPipeline.check_inputs]]

```python
check_inputs(c2w: np.ndarray | None, action: str | None, intrinsics: np.ndarray | list[float] | None, callback_on_step_end_tensor_inputs: list[str] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_sana_wm.py#L335)

Validate `__call__` inputs. Raises on bad input and returns nothing.

#### encode_prompt[[diffusers.SanaWMPipeline.encode_prompt]]

```python
encode_prompt(prompt: str, negative_prompt: str = '', device: torch.device | None = None, max_sequence_length: int = 300, chi_prompt: list[str] | None = None, prompt_embeds: torch.Tensor | None = None, prompt_attention_mask: torch.Tensor | None = None, negative_prompt_embeds: torch.Tensor | None = None, negative_prompt_attention_mask: torch.Tensor | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_sana_wm.py#L204)

**Returns:**

`(cond, cond_mask, neg, neg_mask)` where `cond` and `neg` are `(1, 1, L, D)`-shaped Gemma hidden
states and the masks are `(1, L)`.

Encode prompt + negative prompt through Gemma-2.

Mirrors the SANA chi-prompt-prefix trick: the chi prompt is prepended to the user prompt, then a `select_index = [0, -L+1, ..., -1]` slice takes the BOS token plus the last `max_sequence_length - 1` tokens.

#### prepare_camera_trajectory[[diffusers.SanaWMPipeline.prepare_camera_trajectory]]

```python
prepare_camera_trajectory(c2w: np.ndarray | None, action: str | None, intrinsics: np.ndarray | list[float], num_frames: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_sana_wm.py#L365)

Normalize the camera inputs to `(c2w_(F, 4, 4), intrinsics_(F, 4))`.

Also snaps `num_frames` to the VAE-friendly `8k+1` and trims both arrays to match. The cropped image and
rescaled intrinsics come later, once the target resolution is known.

#### prepare_latents[[diffusers.SanaWMPipeline.prepare_latents]]

```python
prepare_latents(first_latent: torch.Tensor, num_frames: int, height: int, width: int, dtype: torch.dtype, device: torch.device, generator: torch.Generator)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_sana_wm.py#L403)

Sample initial latents and pin the first frame as the conditioning anchor.

Returns `(latents, condition_mask)` where `condition_mask` has ones on the first frame's tokens (they are
held clean throughout sampling) and zeros elsewhere.

## SanaWMLTX2Refiner[[diffusers.SanaWMLTX2Refiner]]

The LTX-2 stage-2 refiner is a standalone [DiffusionPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/overview#diffusers.DiffusionPipeline) that takes stage-1 latents. Give it a `vae` (the
base pipeline's, so the weights are shared) to have it decode to video; without one it returns refined latents.

#### diffusers.SanaWMLTX2Refiner[[diffusers.SanaWMLTX2Refiner]]

```python
diffusers.SanaWMLTX2Refiner(transformer: SanaWMLTX2RefinerTransformer3DModel, connectors: LTX2TextConnectors, tokenizer: GemmaTokenizer | GemmaTokenizerFast, text_encoder: Gemma3ForConditionalGeneration, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLLTX2Video | None = None, text_max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/refiner.py#L51)

**Parameters:**

transformer ([*SanaWMLTX2RefinerTransformer3DModel*]) : The LTX-2 video DiT with the chunk-causal sliding-window KV cache.

connectors ([*LTX2TextConnectors*]) : LTX-2 text connectors.

tokenizer : Gemma-3 tokenizer.

text_encoder : Gemma-3 text encoder.

scheduler ([*FlowMatchEulerDiscreteScheduler*]) : Flow-matching Euler scheduler. Constructed with `shift=1.0` so the distilled sigmas pass through unmodified.

vae ([*AutoencoderKLLTX2Video*], *optional*) : The same VAE used by [*SanaWMPipeline*]; pass *vae=pipe.vae* to share the weights. When given, the refiner decodes to video, otherwise it returns refined latents.

text_max_sequence_length (*int*, defaults to 1024) : Maximum tokens passed to the Gemma-3 tokenizer.

LTX-2 sink-bidirectional Euler refiner — SANA-WM stage 2, as a standalone pipeline.

Wraps the LTX-2 components (refiner transformer + text connectors + Gemma-3 text encoder + tokenizer) plus a
[*FlowMatchEulerDiscreteScheduler*] that carries the distilled sigma schedule and performs the Euler steps. It is
registered as an optional component of [*SanaWMPipeline*] and can also be used on its own to refine stage-1
latents.

#### __call__[[diffusers.SanaWMLTX2Refiner.__call__]]

```python
__call__(sana_latent: torch.Tensor, prompt: str, fps: float, sink_size: int = 1, generator: torch.Generator | None = None, block_size: int = 3, kv_max_frames: int = 11, sigmas: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0), output_type: str = 'np')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/refiner.py#L111)

**Parameters:**

sana_latent : `(B, C, F, H, W)` stage-1 latent.

prompt : text prompt.

fps : video frame rate (drives LTX-2 RoPE temporal scaling).

sink_size : how many leading raw `z_sana` frames to anchor as the attention sink (canonical: 1).

generator : torch.Generator for the FM endpoint noise. Defaults to a generator seeded with 42 so results are reproducible out of the box.

block_size : latent frames per AR block (canonical: 3).

kv_max_frames : maximum context+active frames retained in the sliding window (canonical: 11 = 1 sink + 10 recent).

sigmas : descending Euler schedule terminating at 0.0 (canonical 3-step distilled: `(0.909375, 0.725, 0.421875, 0.0)`). Fed to `self.scheduler` (minus the trailing 0.0, which the scheduler appends itself).

output_type : *"latent"* returns the refined latents. Anything else decodes through *self.vae* and post-processes to that type (*"np"*, *"pt"*, *"pil"*); without a *vae* the latents are returned regardless.

**Returns:** `*torch.Tensor*`

Refined VAE latents of shape `(B, C, F, H, W)` — the first `sink_size` frames carry the
raw stage-1 sink latents unchanged, the rest carry the refined output.

Run the LTX-2 refiner and return refined VAE latents.

Uses the chunk-causal AR recipe the model was trained on (`block_size=3`, `kv_max_frames=11`): a sliding
window of `[source_sink + recent_history + active_block]` K/V is fed to the transformer one block at a time,
so per-block compute is bounded and total refinement cost scales linearly with video length.

## SanaWMPipelineOutput[[diffusers.SanaWMPipelineOutput]]

#### diffusers.SanaWMPipelineOutput[[diffusers.SanaWMPipelineOutput]]

```python
diffusers.SanaWMPipelineOutput(frames: typing.Union[torch.Tensor, numpy.ndarray, list[list[PIL.Image.Image]]], c2w: numpy.ndarray | None = None, latent: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/sana_wm/pipeline_output.py#L25)

**Parameters:**

frames (*torch.Tensor*, *np.ndarray*, or *list[PIL.Image.Image]*) : Generated video: a float `np.ndarray` of shape `(T, H, W, 3)` in `[0, 1]` for `output_type="np"`, a `(T, 3, H, W)` tensor for `"pt"`, a list of `T` ``PIL.Image` for `"pil"`, or the latent tensor `(B, C, T_lat, H_lat, W_lat)` for `"latent"`.

c2w (*np.ndarray*) : Camera-to-world poses `(T, 4, 4)` aligned with `frames` (the refiner drops the sink anchor frame; this array is realigned accordingly when the refiner ran).

latent (*torch.Tensor*, optional) : Latent tensor in LTX-2 VAE space, shape `(B, C, T_lat, H_lat, W_lat)`. Returned when `output_type="latent"``.

Output class for SANA-WM image-to-video pipeline.

### Lumina-T2X
https://huggingface.co/docs/diffusers/v0.42.1/api/pipelines/lumina.md
