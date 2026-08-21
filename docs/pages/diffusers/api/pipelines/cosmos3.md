# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

## Text-to-image

Single-frame generation. The model is conditioned only on the text prompt; pass `num_frames=1`. Upsample with `--mode text2image` to produce the JSON prompt.

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

## Image-to-video

Pass a conditioning image via `image=`. The pipeline anchors frame 0 to the supplied image and denoises the rest. The image is resized while preserving its aspect ratio, center-cropped to the requested output size, and normalized with uint8-equivalent rounding. Upsample with `--mode image2video` to produce the JSON prompt.

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

## Video-to-video

Pass a conditioning clip via `video=` (e.g. from `load_video`). The pipeline anchors the leading latent frames given by `condition_frame_indexes_vision` (default `[0, 1]`) to the clip and denoises the rest. Use `condition_video_keep` (`"first"` or `"last"`) to choose which end of a longer source clip the conditioning frames are taken from. As with the other modes, the prompt should follow the descriptive JSON structure described in [Prompt upsampling](#prompt-upsampling).

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_v2v.mp4", fps=24, macro_block_size=1)
```

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_v2v.mp4", fps=24, macro_block_size=1)
```

## Video-to-video with sound

When the checkpoint carries a `sound_tokenizer`, add `enable_sound=True` to the video-to-video call to jointly generate a synchronized audio track. The waveform is returned alongside the video and can be muxed into the MP4 with [encode_video()](/docs/diffusers/v0.40.0/en/api/utilities#diffusers.utils.encode_video).

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_v2v_with_sound.mp4",
)
```

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_v2v_with_sound.mp4",
)
```

## Text-to-video with sound

When the checkpoint carries a `sound_tokenizer`, pass `enable_sound=True` to jointly generate a synchronized audio track. The waveform is returned alongside the video and can be muxed into the MP4 with [encode_video()](/docs/diffusers/v0.40.0/en/api/utilities#diffusers.utils.encode_video).

This is the same call as the text-to-video example above with `enable_sound=True` added:

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_with_sound.mp4",
)
```

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_with_sound.mp4",
)
```

## Action-conditioned generation

Action runs group every action-specific input into a [CosmosActionCondition](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.CosmosActionCondition) passed via the `action` argument instead of the top-level `image` / `video` / `height` / `width` arguments. Set `resolution_tier` (`256`/`480`/`704`/`720`) close to the input video's native resolution; it selects the conditioning canvas. Cosmos 3 supports three action modes — `policy`, `forward_dynamics`, and `inverse_dynamics`. `policy` and `forward_dynamics` condition only on the first frame (so an `image` or a `video` both work), while `inverse_dynamics` requires a `video`. The conditioning video for an action run is set on `action.video` (or `action.image`), not on the pipeline's top-level `video` argument.

Pass a plain task description as `prompt` and pick the camera with `action.view_point` (default `"ego_view"`; also `"third_person_view"`, `"wrist_view"`, `"concat_view"`). The pipeline turns these into the structured JSON caption the model was trained on, so action prompts should not be LLM-upsampled.

### Action policy

Action policy generation predicts future video and action tokens from the first observation frame, text prompt, and action domain metadata. The example below uses the Bridge robot domain and writes the predicted action chunk to JSON in model-normalized action space.

```python
import json

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

prompt = "Put the pot to the left of the purple item."
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

result = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "sample.mp4", fps=5, macro_block_size=1)

if result.action is not None:
    with open("sample_action.json", "w") as f:
        json.dump(result.action[0].tolist(), f)
```

```python
import json

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

prompt = "Put the pot to the left of the purple item."
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

result = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "sample.mp4", fps=5, macro_block_size=1)

if result.action is not None:
    with open("sample_action.json", "w") as f:
        json.dump(result.action[0].tolist(), f)
```

## Context parallelism

For long videos or high resolutions, a single forward pass can exceed the memory and latency budget of one GPU. Cosmos 3 supports **context parallelism (CP)** to shard the sequence dimension across multiple GPUs, splitting the attention computation so each device holds only a slice of the tokens.

Cosmos 3 supports **Ulysses** context parallelism (all-to-all sequence/head exchange). Ring attention is not supported.

Unlike most diffusers models, Cosmos 3 does **not** wire CP into the transformer or the declarative `enable_parallelism()` path: its grouped-query attention, separate understanding/generation streams (the generation stream attends to both), and ragged per-stream lengths can't be expressed as a `_cp_plan`. Instead, the model exposes small no-op shard/gather seams, and the implementation lives in [`examples/cosmos3/cosmos_parallel.py`](https://github.com/huggingface/diffusers/blob/main/examples/cosmos3/cosmos_parallel.py) — a self-contained module you can read end to end and adapt. It offers two orthogonal, composable sharding axes:

| Helper | Shards | Use for |
|---|---|---|
| `enable_cosmos3_context_parallel(transformer, cp_mesh)` | sequence (CP / Ulysses) | latency on a model that fits one GPU (`Nano`) |
| `enable_cosmos3_tensor_parallel(transformer, tp_mesh)` | weights (TP) | fitting a model that doesn't fit one GPU (`Super`) |

Use either alone or both together on a 2-D `(tp, cp)` mesh (see [Fitting large models with tensor parallelism](#fitting-large-models-with-tensor-parallelism)).

Two requirements are specific to Cosmos 3:

- Use the `native` attention backend. Cosmos 3 uses grouped-query attention (GQA), and the native SDPA backend is the only one that accepts `enable_gqa` (cuDNN and flash reject it). The helpers expand the KV heads to the query-head count and call SDPA with `enable_gqa=False` so it still dispatches to the flash kernel (the math fallback would materialize the full `[S, S]` scores and OOM on long sequences).
- The CP (Ulysses) degree must divide the query-head count (32 for `Nano`, 64 for `Super`); for TP, the degree must divide the KV heads (8). The understanding (text) and generation (video/sound) streams are sharded independently along the sequence, and ragged lengths are zero-padded internally to a multiple of the world size.

### Run it

The full CLI [`examples/cosmos3/inference_cosmos3.py`](https://github.com/huggingface/diffusers/blob/main/examples/cosmos3/inference_cosmos3.py) uses [Cosmos3OmniModularPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3OmniModularPipeline) and reuses these helpers, so **any modality** (text-to-image/video, image-to-video, sound, action modes) runs multi-GPU via `--tp-degree` / `--cp-degree`. Launch with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html); `--tp-degree * --cp-degree` must equal `--nproc_per_node`. Every rank produces the same output; rank 0 writes it.

```bash
# CP only — Nano (fits one GPU); CP degree must divide 32 query heads.
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py --model nano --cp-degree 4 --prompt "..."

# TP only — Super; TP degree must divide 64 query heads and 8 KV heads.
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py --model super --tp-degree 4 --prompt "..."

# TP + CP — Super, with sound (TP=2 x CP=2 across 4 GPUs).
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py \
    --model super --tp-degree 2 --cp-degree 2 --enable-sound --prompt "..."
```

`Super`'s ~120 GB of weights do not fit on one 96 GB GPU, so it needs TP; `Nano` fits on a single GPU, so CP for it is a pure latency optimization. (Omit both flags to run single-GPU.)

### Fitting large models with tensor parallelism

CP shards *activations* but replicates every weight on every rank, so it does not reduce a model's weight footprint — a model that doesn't fit on one GPU still won't fit under CP alone. To shard the **weights**, `enable_cosmos3_tensor_parallel(transformer, tp_mesh)` applies Megatron-style tensor parallelism on a second, orthogonal mesh axis:

- The attention and MLP projections are column/row sharded across the TP group (`to_q/to_k/to_v` + `add_q/k/v` and the MLPs' `gate/up` are column-parallel; `to_out/to_add_out` and the MLPs' `down` are row-parallel with an all-reduce). Each rank ends up owning `query_heads / tp` query heads and `kv_heads / tp` KV heads.
- TP composes with CP on a 2-D `(tp, cp)` device mesh: TP splits heads/weights persistently, CP shards the sequence on top. The constraints are `tp` divides the KV heads (8), and `tp * cp` divides the query heads (32 for `Nano`, 64 for `Super`).
- Weights are loaded to CPU and sharded onto the GPUs layer by layer, so the full model is never materialized on a single device.

> [!TIP]
> TP issues an all-reduce on every attention and MLP block, so it is bandwidth-heavy. On hosts without NVLink it is the dominant cost; prefer the smallest TP degree that makes the weights fit and put the remaining GPUs into CP.

### Use it in your own modular pipeline

The CLI flags are convenient, but you can call the helpers directly with [Cosmos3OmniModularPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3OmniModularPipeline). Load the pipeline configuration and components on CPU, apply TP *before* moving the pipeline to the rank-local GPU, switch to the `native` backend, and then enable CP. Do not use `device_map` for this flow:

```python
import os
import sys

import torch
import torch.distributed as dist
from diffusers import Cosmos3OmniModularPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from torch.distributed.device_mesh import init_device_mesh

# Make the helper module importable.
sys.path.insert(0, "examples/cosmos3")
from cosmos_parallel import (
    enable_cosmos3_context_parallel,
    enable_cosmos3_flash_attention,
    enable_cosmos3_tensor_parallel,
)

# torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Pick tp_degree * cp_degree == world size.
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (tp_degree, cp_degree), mesh_dim_names=("tp", "cp"))

# Load components on CPU first; a TP-sharded model may not fit one GPU.
pipe = Cosmos3OmniModularPipeline.from_pretrained(model_id)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()

if tp_degree > 1:
    enable_cosmos3_tensor_parallel(pipe.transformer, mesh["tp"])  # shard weights -> GPUs
pipe.to(f"cuda:{local_rank}")                                     # move the replicated remainder
pipe.transformer.set_attention_backend("native")
if cp_degree > 1:
    enable_cosmos3_context_parallel(pipe.transformer, mesh["cp"])  # shard the sequence
elif tp_degree > 1:
    enable_cosmos3_flash_attention(pipe.transformer)               # GQA-safe dense attention

# Modular pipelines replace components through update_components().
scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)
pipe.update_components(scheduler=scheduler)

# A single output name returns that value; a list returns a dictionary.
outputs = pipe(
    prompt='{"scene":"A robot arm in a kitchen"}',
    num_frames=189,
    height=720,
    width=1280,
    output=["videos", "sound", "sampling_rate", "action"],
)
videos = outputs["videos"]
sound = outputs["sound"]  # None unless sound generation was requested.
action = outputs["action"]  # None unless an action workflow produced actions.
```

For CP only (no weight sharding), use a 1-D mesh: `init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))` and just `enable_cosmos3_context_parallel`.

`enable_safety_checker()` loads and enables the default checker; `disable_safety_checker()` explicitly disables it. Use those pipeline methods instead of the task-pipeline `enable_safety_checker=` construction argument or `enable_safety_check=` call argument. Modular pipelines also do not return `Cosmos3OmniPipelineOutput`: use `output="videos"` for frames alone, or an output list and its returned dictionary as shown above instead of `result.video`, `result.sound`, or `result.action`.

> [!TIP]
> On some multi-GPU topologies the first NCCL all-to-all can hang. If a CP run stalls at the start of the first denoising step, set `NCCL_P2P_DISABLE=1` in the environment before launching `torchrun`.

CP and TP compose with all the workflows above (text-to-video, image-to-video, text-to-video with sound, and action-conditioned generation) and with both the `Nano` and `Super` checkpoints — only the pipeline construction and the parallelism setup lines change.

## Metadata templates

`tokenize_prompt` appends short metadata sentences inside the user message so the LLM sees the conditioning the model was trained with. The positive prompt gets sentences like *"The video is 7.9 seconds long and is of 24 FPS."* and *"This video is of 720x1280 resolution."*; the negative prompt gets the inverse (*"… is not …"*).

Both are on by default. Disable either pair through `__call__`:

```python
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    add_duration_template=False,    # skip the duration sentence on both prompts
    add_resolution_template=False,  # skip the resolution sentence on both prompts
)
```

`add_duration_template` has no effect when `num_frames == 1` (image mode); only the resolution sentence is appended in that case.

## Safety checker

Cosmos3 wires up the [`cosmos_guardrail`](https://pypi.org/project/cosmos-guardrail/) `CosmosSafetyChecker` and runs it **by default**. The text guardrail rejects unsafe prompts before generation (`ValueError`); the video guardrail runs on the decoded frames and either pixelates detected faces or rejects the output. Audio output is not guardrailed.

Install the optional dependency to enable the default checker:

```
pip install cosmos_guardrail
```

The checker is mandatory under the NVIDIA Open Model License Agreement. The two flags below exist for tests and development workflows where the guardrail would be redundant (e.g., the input has already been cleared, or you are intentionally exercising the pipeline on edge inputs).

**Disable at construction** (no checker is instantiated, so no guardrail models are downloaded or loaded into memory):

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)
```

**Disable for a single call** (checker stays loaded — useful for one-off bypass while keeping the default on for subsequent calls):

```python
result = pipe(
    prompt=prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_safety_check=False,
)
```

To supply a custom checker (e.g., a no-op subclass for fast tests), pass it as `safety_checker=`:

```python
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    dtype=torch.bfloat16,
    device_map="cuda",
    safety_checker=MyCustomSafetyChecker(),
)
```

## Cosmos3OmniPipeline[[diffusers.Cosmos3OmniPipeline]]

#### diffusers.Cosmos3OmniPipeline[[diffusers.Cosmos3OmniPipeline]]

```python
diffusers.Cosmos3OmniPipeline(transformer: Cosmos3OmniTransformer, text_tokenizer: AutoTokenizer, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler, sound_tokenizer: diffusers.models.autoencoders.autoencoder_cosmos3_audio.Cosmos3AVAEAudioTokenizer | None = None, safety_checker: diffusers.pipelines.cosmos.pipeline_cosmos3_omni.CosmosSafetyChecker | None = None, enable_safety_checker: bool = True, default_use_system_prompt: bool = True, use_native_flow_schedule: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L407)

#### decode_sound[[diffusers.Cosmos3OmniPipeline.decode_sound]]

```python
decode_sound(latent: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L511)

Decode a sound latent `[C, T]` to a waveform `[audio_ch, N]`.

Adds/removes the batch dimension expected by the sound tokenizer decoder.

#### prepare_latents[[diffusers.Cosmos3OmniPipeline.prepare_latents]]

```python
prepare_latents(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, NoneType] = None, video: typing.Union[list[PIL.Image.Image], torch.Tensor, numpy.ndarray, NoneType] = None, condition_frame_indexes_vision: Iterable = (0, 1), condition_video_keep: typing.Literal['first', 'last'] = 'first', num_frames: int | None = None, height: int | None = None, width: int | None = None, fps: float = 24.0, latents: typing.Optional[torch.Tensor] = None, sound_latents: typing.Optional[torch.Tensor] = None, action_latents: typing.Optional[torch.Tensor] = None, generator: typing.Optional[torch.Generator] = None, device: str = 'cuda', dtype: dtype = torch.bfloat16, enable_sound: bool = False, action: CosmosActionCondition | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L757)

**Returns:**

Initial noisy tensors plus condition masks/metadata for vision, sound, and optional action modalities.

Build conditioning + initial noise for a single sample.

#### tokenize_prompt[[diffusers.Cosmos3OmniPipeline.tokenize_prompt]]

```python
tokenize_prompt(prompt: str, negative_prompt: str | None = None, num_frames: int = 189, height: int = 720, width: int = 1280, fps: float = 24.0, use_system_prompt: bool | None = None, add_resolution_template: bool = True, add_duration_template: bool = True, action_mode: str | None = None, action_view_point: str | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L1126)

**Returns:**

`(cond_input_ids, uncond_input_ids)` — token-id lists for this sample.

Apply prompt-augmentation templates and tokenize cond/uncond prompts via the configured chat template.

This pipeline does not run a separate text encoder: the joint Cosmos3 transformer consumes raw token IDs
alongside vision (and optionally sound) tokens.

When `negative_prompt` is `None`, an empty string is used; the Cosmos3 docs page documents recommended
quality-control negative prompts to pass explicitly for text2video / image2video. The duration and resolution
templates are appended to the prompt, and inverse templates are appended to the negative prompt, when enabled.

When `action_mode` is set, the prompt is instead converted to the structured action JSON caption the model
was trained on (see `_build_action_json_prompt`), using `action_view_point` for the framing field; the
flat metadata templates are skipped because the JSON already carries duration/fps/resolution/aspect_ratio.

- all
- __call__

## Cosmos3OmniModularPipeline

Cosmos 3 is also available as a Modular Diffusers pipeline. The task-based [Cosmos3OmniPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3OmniPipeline) remains available; the modular pipeline coexists with it and covers the same modes (`text2image`, `text2video`, `image2video`, `video2video`, action-conditioned generation, and `transfer` (structural control), with optional sound when supported by the checkpoint).

```python
import torch
from diffusers import Cosmos3OmniModularPipeline

pipe = Cosmos3OmniModularPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16
)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()

videos = pipe(
    prompt='{"scene":"A robot arm in a kitchen"}',
    num_frames=1,
    height=720,
    width=1280,
    output="videos",
)

# Modular pipelines expose declared outputs directly instead of using the task pipeline's
# `return_dict`/`Cosmos3OmniPipelineOutput` API.
image = videos[0]
```

You can also load through [ModularPipeline](/docs/diffusers/v0.40.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline) and let the repository config select the blocks class:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()
videos = pipe(
    prompt='{"scene":"A robot arm in a kitchen"}', num_frames=1, height=720, width=1280, output="videos"
)
```

To inspect or customize a specific Cosmos modular workflow, use `available_workflows` + `get_workflow()`:

```python
available = pipe.blocks.available_workflows
image2video_blocks = pipe.blocks.get_workflow("image2video")
```

### Modular examples for all existing workflows

The modular pipeline supports the same call signatures as the task pipeline. The snippets below mirror every generation example shown above (`text2video`, `text2image`, `image2video`, `video2video`, `video2video_sound`, `text2video_sound`, and `action_policy`). Transfer (structural control) has its own inputs and is shown separately in [Modular transfer](#modular-transfer-structural-control) below.

```python
import json
import torch
from diffusers import Cosmos3OmniModularPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, export_to_video, load_image, load_video

pipe = Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

# text2video
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_t2v.mp4", fps=24, macro_block_size=1)

# text2image
json_prompt = json.load(open("assets/example_t2i_prompt.json"))
videos = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280, output="videos")
videos[0].save("cosmos3_modular_t2i.jpg", format="JPEG", quality=85)

# image2video
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))
image = load_image("https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg")
videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_i2v.mp4", fps=24, macro_block_size=1)

# video2video
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)
videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_v2v.mp4", fps=24, macro_block_size=1)

# video2video_sound
outputs = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
    output=["videos", "sound", "sampling_rate"],
)
encode_video(
    outputs["videos"],
    fps=24,
    audio=outputs["sound"],
    audio_sample_rate=outputs["sampling_rate"],
    output_path="cosmos3_modular_v2v_with_sound.mp4",
)

# text2video_sound
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
outputs = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
    output=["videos", "sound", "sampling_rate"],
)
encode_video(
    outputs["videos"],
    fps=24,
    audio=outputs["sound"],
    audio_sample_rate=outputs["sampling_rate"],
    output_path="cosmos3_modular_t2v_with_sound.mp4",
)

# action_policy
prompt = "Put the pot to the left of the purple item."
action_video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)
outputs = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=action_video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
    output=["videos", "action"],
)
export_to_video(outputs["videos"], "cosmos3_modular_action_policy.mp4", fps=5, macro_block_size=1)
if outputs["action"] is not None:
    with open("cosmos3_modular_action_policy.json", "w") as f:
        json.dump(outputs["action"][0].tolist(), f)
```

### Modular transfer (structural control)

Transfer follows a **precomputed control video** (edge, blur, depth, segmentation, or a world-scenario map) passed through `control_videos=` as a `{hint: video}` mapping. It is video-only (no `image` / `video` / `action` / `enable_sound`), the prompt is a pre-upsampled JSON caption (see [Prompt upsampling](#prompt-upsampling)), and long clips are generated autoregressively in chunks of `num_video_frames_per_chunk` and stitched automatically. `guidance_scale` is the usual text CFG; `control_guidance` (`!= 1.0`) additionally amplifies the control signal. Recommended starting values per hint:

| Hint | `guidance_scale` | `control_guidance` | `flow_shift` | Geometry |
| --- | --- | --- | --- | --- |
| Edge / Blur / Depth | 3.0 | 1.5 | 10.0 | 121 frames @ 30 FPS |
| Segmentation | 3.0 | 2.0 | 10.0 | 121 frames @ 30 FPS |
| World scenario (WSM) | 1.0 | 3.0 | 10.0 | 101 frames @ 10 FPS |

Diffusers does not ship the control assets. Ready-made ones (a control video + matching `prompt.json` per hint, plus a shared `negative_prompt.json`) live in the [Cosmos cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer/assets). For the edge example below, download them into a local `assets/` folder:

```bash
base=https://github.com/NVIDIA/cosmos/raw/refs/heads/main/cookbooks/cosmos3/generator/transfer/assets
mkdir -p assets/edge
curl -sL "$base/edge/control_edge.mp4" -o assets/edge/control_edge.mp4
curl -sL "$base/edge/prompt.json"      -o assets/edge/prompt.json
curl -sL "$base/negative_prompt.json"  -o assets/negative_prompt.json
```

```python
import json
import torch
from diffusers import Cosmos3OmniModularPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

# Downloaded into assets/ from the Cosmos cookbook (see the curl snippet above).
json_prompt = json.load(open("assets/edge/prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
control_edge = load_video("assets/edge/control_edge.mp4")

videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    control_videos={"edge": control_edge},
    num_frames=121,
    height=720,
    width=1280,
    fps=30.0,
    num_inference_steps=35,
    guidance_scale=3.0,
    control_guidance=1.5,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_transfer_edge.mp4", fps=30, macro_block_size=1)
```

### Distilled (few-step) text-to-image and image-to-video[[diffusers.Cosmos3OmniModularPipeline]]

Few-step distilled checkpoints are served by [Cosmos3DistilledModularPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3DistilledModularPipeline) (blocks:
`Cosmos3DistilledBlocks`); the base [Cosmos3OmniModularPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3OmniModularPipeline) and [Cosmos3OmniPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3OmniPipeline) do
not support them. `num_inference_steps` is fixed to the length of the `distilled_sigmas` pipeline
config (from the checkpoint's `modular_model_index.json`) and `guidance_scale` is forced to
1.0 since guidance is baked into the weights — passing any other value for either raises an error,
and `negative_prompt` is warned about and ignored.

Prompts follow the same descriptive JSON structure as the non-distilled models, so short text
must be upsampled first — use `--mode text2image` (T2I) or `--mode image2video` (I2V) as
described in [Prompt upsampling](#prompt-upsampling), then pass the JSON via `json.dumps(...)`.

```python
import json
import torch
from diffusers import Cosmos3DistilledModularPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

repo = "nvidia/Cosmos3-Super-Text2Image-4Step"
pipe = Cosmos3DistilledModularPipeline.from_pretrained(repo, dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

# text-to-image (distilled)
videos = pipe(
    prompt=json.dumps(json_prompt),
    num_frames=1,
    height=720,
    width=1280,
    output="videos",
)
videos[0].save("cosmos3_distilled_t2i.jpg", format="JPEG", quality=85)

# image-to-video (distilled) — load the I2V repo instead
# JSON-upsampled prompt (see "Prompt upsampling" above); upsampled from the source prompt
# "The right robotic hand picks up the red sphere on the shelf."
json_prompt_i2v = json.load(open("assets/example_i2v_prompt.json"))

repo_i2v = "nvidia/Cosmos3-Super-Image2Video-4Step"
pipe = Cosmos3DistilledModularPipeline.from_pretrained(repo_i2v, dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg"
)
videos = pipe(
    prompt=json.dumps(json_prompt_i2v),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    output="videos",
)
export_to_video(videos, "cosmos3_distilled_i2v.mp4", fps=24, macro_block_size=1)
```

#### diffusers.Cosmos3OmniModularPipeline[[diffusers.Cosmos3OmniModularPipeline]]

```python
diffusers.Cosmos3OmniModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/cosmos/modular_pipeline.py#L7)

A ModularPipeline for Cosmos 3 omni generation.

- all
- __call__

## Cosmos3DistilledModularPipeline[[diffusers.Cosmos3DistilledModularPipeline]]

#### diffusers.Cosmos3DistilledModularPipeline[[diffusers.Cosmos3DistilledModularPipeline]]

```python
diffusers.Cosmos3DistilledModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/cosmos/modular_pipeline.py#L121)

A ModularPipeline for distilled (few-step) Cosmos 3 omni generation.

Distilled checkpoints bake classifier-free guidance into the weights and sample on a fixed schedule read from the
pipeline's `distilled_sigmas` config (populated from `modular_model_index.json`), so `guidance_scale` and
`num_inference_steps` are fixed and `negative_prompt` is not supported.

- all
- __call__

## CosmosActionCondition[[diffusers.CosmosActionCondition]]

#### diffusers.CosmosActionCondition[[diffusers.CosmosActionCondition]]

```python
diffusers.CosmosActionCondition(mode: typing.Literal['policy', 'forward_dynamics', 'inverse_dynamics'], chunk_size: int, domain_name: str, resolution_tier: int = 480, raw_actions: typing.Optional[torch.Tensor] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, NoneType] = None, video: typing.Union[list, numpy.ndarray, torch.Tensor, NoneType] = None, view_point: str = 'ego_view')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L296)

**Parameters:**

mode (`str`) : The action task. One of `"forward_dynamics"` (roll out future video from a first frame and a given `raw_actions` sequence), `"inverse_dynamics"` (infer the actions connecting the conditioning frames), or `"policy"` (jointly roll out future video and actions from the first frame).

chunk_size (`int`) : Number of action transition steps in the chunk. The paired conditioning video spans `chunk_size + 1` frames.

domain_name (`str`) : Embodiment domain selecting the domain-aware action projection weights. Must be one of the registered Cosmos 3 embodiment domains. It also fixes the unpadded action width used to slice predicted actions, resolved internally from this name (see `_EMBODIMENT_TO_RAW_ACTION_DIM`).

resolution_tier (`int`, defaults to `480`) : Action conditioning resolution *tier* (one of `256`, `480`, `704`, `720`). The tier picks a predefined canvas whose aspect ratio is closest to the input; the input is downscaled (never upscaled) and padded into it for conditioning. This is not the output frame size, which tracks the input content. Match the tier to the input's native resolution: a lower tier discards detail, while a higher tier adds no resolution (no upscaling), wastes compute on padding, and is a train/inference mismatch that can hurt quality.

raw_actions (`torch.Tensor`, *optional*) : Raw domain action vectors of shape `[T, raw_action_dim]` driving `"forward_dynamics"`. Sequences shorter than `chunk_size` repeat the last action; longer ones are truncated. Channels beyond the model's `action_dim` are rejected, and narrower inputs are zero-padded up to `action_dim`.

image (`PIL.Image.Image`, `np.ndarray`, or `torch.Tensor`, *optional*) : Conditioning frame for `"policy"` / `"forward_dynamics"`. Mutually exclusive with `video`.

video (`list`, `np.ndarray`, or `torch.Tensor`, *optional*) : Conditioning video, required for `"inverse_dynamics"`. For `"policy"` / `"forward_dynamics"` only its first frame is used. Mutually exclusive with `image`.

view_point (`str`, defaults to `"ego_view"`) : Camera perspective label used to populate the action caption's `cinematography.framing` field. One of `"ego_view"`, `"third_person_view"`, `"wrist_view"`, or `"concat_view"`. The action model was trained on structured JSON captions that carry this viewpoint sentence; an unrecognized label drops the framing field (with a warning).

Groups every input required for a Cosmos 3 action-conditioned generation task.

Pass this to `Cosmos3OmniPipeline.__call__()` via the `action` argument instead of the top-level `image` / `height`
/ `width` arguments, which are reserved for t2v, i2v runs.

## Cosmos3OmniPipelineOutput[[diffusers.pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput]]

#### diffusers.pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput[[diffusers.pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput]]

```python
diffusers.pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput(video: typing.Any, sound: typing.Optional[torch.Tensor] = None, action: list[torch.Tensor] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py#L277)

**Parameters:**

video : The generated video. The exact type depends on `output_type` passed to the pipeline: a list of PIL frames for `"pil"` (default), an `np.ndarray` of shape `[T, H, W, C]` for `"np"`, a `torch.Tensor` of shape `[T, C, H, W]` for `"pt"`, or a raw latent tensor when `output_type="latent"`.

sound : Decoded audio waveform of shape `[C, N]`. `None` when `enable_sound=False`.

action : Predicted action tokens. `None` unless an action mode predicts actions.

Output dataclass for [Cosmos3OmniPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/cosmos3#diffusers.Cosmos3OmniPipeline).

### Sana
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/sana.md

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

# SanaPipeline

  
  

[SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://huggingface.co/papers/2410.10629) from NVIDIA and MIT HAN Lab, by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, Song Han.

The abstract from the paper is:

*We introduce Sana, a text-to-image framework that can efficiently generate images up to 4096×4096 resolution. Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU. Core designs include: (1) Deep compression autoencoder: unlike traditional AEs, which compress images only 8×, we trained an AE that can compress images 32×, effectively reducing the number of latent tokens. (2) Linear DiT: we replace all vanilla attention in DiT with linear attention, which is more efficient at high resolutions without sacrificing quality. (3) Decoder-only text encoder: we replaced T5 with modern decoder-only small LLM as the text encoder and designed complex human instruction with in-context learning to enhance the image-text alignment. (4) Efficient training and sampling: we propose Flow-DPM-Solver to reduce sampling steps, with efficient caption labeling and selection to accelerate convergence. As a result, Sana-0.6B is very competitive with modern giant diffusion model (e.g. Flux-12B), being 20 times smaller and 100+ times faster in measured throughput. Moreover, Sana-0.6B can be deployed on a 16GB laptop GPU, taking less than 1 second to generate a 1024×1024 resolution image. Sana enables content creation at low cost. Code and model will be publicly released.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [lawrence-cj](https://github.com/lawrence-cj) and [chenjy2003](https://github.com/chenjy2003). The original codebase can be found [here](https://github.com/NVlabs/Sana). The original weights can be found under [hf.co/Efficient-Large-Model](https://huggingface.co/Efficient-Large-Model).

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers) | `torch.bfloat16` |
| [`Efficient-Large-Model/Sana_1600M_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) | `torch.float16` |
| [`Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers) | `torch.float16` |
| [`Efficient-Large-Model/Sana_1600M_512px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers) | `torch.float16` |
| [`Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers) | `torch.float16` |
| [`Efficient-Large-Model/Sana_600M_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) | `torch.float16` |
| [`Efficient-Large-Model/Sana_600M_512px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers) | `torch.float16` |

Refer to [this](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e) collection for more information.

Note: The recommended dtype mentioned is for the transformer weights. The text encoder and VAE weights must stay in `torch.bfloat16` or `torch.float32` for the model to work correctly. Please refer to the inference example below to see how to load the model with the recommended dtype. 

> [!TIP]
> Make sure to pass the `variant` argument for downloaded checkpoints to use lower disk space. Set it to `"fp16"` for models with recommended dtype as `torch.float16`, and `"bf16"` for models with recommended dtype as `torch.bfloat16`. By default, `torch.float32` weights are downloaded, which use twice the amount of disk storage. Additionally, `torch.float32` weights can be downcasted on-the-fly by specifying the `dtype` argument. Read about it in the [docs](https://huggingface.co/docs/diffusers/v0.31.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained).

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [SanaPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/sana#diffusers.SanaPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaTransformer2DModel, SanaPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaTransformer2DModel.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    dtype=torch.float16,
)

pipeline = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("sana.png")
```

## SanaPipeline[[diffusers.SanaPipeline]]

#### diffusers.SanaPipeline[[diffusers.SanaPipeline]]

```python
diffusers.SanaPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderDC, transformer: SanaTransformer2DModel, scheduler: DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana.py#L189)

Pipeline for text-to-image generation using [Sana](https://huggingface.co/papers/2410.10629).

#### __call__[[diffusers.SanaPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str = '', num_inference_steps: int = 20, timesteps: list = None, sigmas: list = None, guidance_scale: float = 4.5, num_images_per_prompt: int | None = 1, height: int = 1024, width: int = 1024, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana.py#L673)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion.IFPipelineOutput` instead of a plain tuple.

attention_kwargs : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

clean_caption (`bool`, *optional*, defaults to `True`) : Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to be installed. If the dependencies are not installed, the embeddings will be created from the raw prompt.

use_resolution_binning (`bool` defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to the requested resolution. Useful for generating non-square images.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to `300`) : Maximum sequence length to use with the `prompt`.

complex_human_instruction (`list[str]`, *optional*) : Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

**Returns:** [SanaPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) or `tuple`

If `return_dict` is `True`, [SanaPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaPipeline

>>> pipe = SanaPipeline.from_pretrained(
...     "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.float32
... )
>>> pipe.to("cuda")
>>> pipe.text_encoder.to(torch.bfloat16)
>>> pipe.transformer = pipe.transformer.to(torch.bfloat16)

>>> image = pipe(prompt='a cyberpunk cat with a neon sign that says "Sana"')[0]
>>> image[0].save("output.png")
```

#### encode_prompt[[diffusers.SanaPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], do_classifier_free_guidance: bool = True, negative_prompt: str = '', num_images_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana.py#L280)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For PixArt-Alpha, this should be "".

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : whether to use classifier free guidance or not

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

Encodes the prompt into text encoder hidden states.

## SanaPAGPipeline[[diffusers.SanaPAGPipeline]]

#### diffusers.SanaPAGPipeline[[diffusers.SanaPAGPipeline]]

```python
diffusers.SanaPAGPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderDC, transformer: SanaTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, pag_applied_layers: str | list[str] = 'transformer_blocks.0')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/pag/pipeline_pag_sana.py#L147)

Pipeline for text-to-image generation using [Sana](https://huggingface.co/papers/2410.10629). This pipeline
supports the use of [Perturbed Attention Guidance
(PAG)](https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag).

#### __call__[[diffusers.SanaPAGPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str = '', num_inference_steps: int = 20, timesteps: list = None, sigmas: list = None, guidance_scale: float = 4.5, num_images_per_prompt: int | None = 1, height: int = 1024, width: int = 1024, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '], pag_scale: float = 3.0, pag_adaptive_scale: float = 0.0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/pag/pipeline_pag_sana.py#L594)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion.IFPipelineOutput` instead of a plain tuple.

clean_caption (`bool`, *optional*, defaults to `True`) : Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to be installed. If the dependencies are not installed, the embeddings will be created from the raw prompt.

use_resolution_binning (`bool` defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to the requested resolution. Useful for generating non-square images.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 300) : Maximum sequence length to use with the `prompt`.

complex_human_instruction (`list[str]`, *optional*) : Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

pag_scale (`float`, *optional*, defaults to 3.0) : The scale factor for the perturbed attention guidance. If it is set to 0.0, the perturbed attention guidance will not be used.

pag_adaptive_scale (`float`, *optional*, defaults to 0.0) : The adaptive scale factor for the perturbed attention guidance. If it is set to 0.0, `pag_scale` is used.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaPAGPipeline

>>> pipe = SanaPAGPipeline.from_pretrained(
...     "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
...     pag_applied_layers=["transformer_blocks.8"],
...     torch_dtype=torch.float32,
... )
>>> pipe.to("cuda")
>>> pipe.text_encoder.to(torch.bfloat16)
>>> pipe.transformer = pipe.transformer.to(torch.bfloat16)

>>> image = pipe(prompt='a cyberpunk cat with a neon sign that says "Sana"')[0]
>>> image[0].save("output.png")
```

#### encode_prompt[[diffusers.SanaPAGPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], do_classifier_free_guidance: bool = True, negative_prompt: str = '', num_images_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/pag/pipeline_pag_sana.py#L188)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For PixArt-Alpha, this should be "".

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : whether to use classifier free guidance or not

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

Encodes the prompt into text encoder hidden states.

## SanaPipelineOutput[[diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput]]

#### diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput[[diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput]]

```python
diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for Sana pipelines.

### Kandinsky 5.0 Video
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/kandinsky5_video.md
