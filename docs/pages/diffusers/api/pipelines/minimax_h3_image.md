# MiniMax-H3-Image

MiniMax-H3-Image is an image model by MiniMax based on [MiniMax-H3](./minimax_h3), for text-to-image generation and reference-image editing. You can find the checkpoint at [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).

Like MiniMax-H3, it is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only: [MiniMaxH3ImageBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/minimax_h3_image#diffusers.MiniMaxH3ImageBlocks) and their [MiniMaxH3ImageModularPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/minimax_h3_image#diffusers.MiniMaxH3ImageModularPipeline) are the whole integration.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3-Image", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
```

The transformer and the text encoder are about 62 GB each in bfloat16 and do not fit on an 80 GB GPU together, so the examples use auto offloading: the [ComponentsManager](/docs/diffusers/v0.43.0/en/api/modular_diffusers/pipeline_components#diffusers.ComponentsManager) moves each component onto the accelerator when a block needs it. A 4736x3552 image then peaks at about 65 GB.

Text-to-image:

```py
prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."
)
image = pipe(
    prompt=prompt, height=3552, width=4736, generator=torch.Generator().manual_seed(42), output="images"
)[0]
```

Editing a reference image:

```py
from diffusers.utils import load_image

reference = load_image("https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/examples/reference1.png")
image = pipe(
    prompt="Keep everything in <Picture 1> as it is, but remove the headphones and have the two hosts fist bump.",
    image=reference,
    height=4096,
    width=4096,
    generator=torch.Generator().manual_seed(1234),
    output="images",
)[0]
```

## Components

| Component | Class |
|---|---|
| `transformer` | [MiniMaxH3ImageTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/minimax_h3_image_transformer2d#diffusers.MiniMaxH3ImageTransformer2DModel) |
| `text_encoder` | `Qwen3VLForConditionalGeneration` |
| `vae` | [AutoencoderKLMiniMaxH3Image](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_minimax_h3_image#diffusers.AutoencoderKLMiniMaxH3Image) |
| `tokenizer`, `processor` | The text encoder's tokenizer and a `Qwen3VLProcessor` |
| `scheduler` | [MiniMaxH3Scheduler](/docs/diffusers/v0.43.0/en/api/schedulers/minimax_h3#diffusers.MiniMaxH3Scheduler) |

The text encoder is the same one [MiniMax-H3](./minimax_h3) uses. The checkpoint keeps its own copy so that the repository is self-contained; the Hub deduplicates identical files across repositories, so the copy takes no extra storage there.

## Workflows

The workflow is selected on the `image` input:

- **`text2image`** generates from `prompt`.
- **`image_conditioned`** generates from `prompt` and one or more reference images passed as `image`. The prompt refers to them in order as `<Picture 1>`, `<Picture 2>`, and so on. The references keep their own size: each is resized only when its short edge is outside 1024 to 2048 pixels, and never to the size of the generated image.

The prompt is used verbatim, with no chat template, and long, complete descriptions work best. An edit prompt typically names what to preserve, then what to change, then describes the target image in full.

## Prompt enhancement

[MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) is MiniMax's prompt expansion model: it rewrites a short request into the kind of detailed description the image model works best with, and recommends an aspect ratio. The [diffusers/minimax-h3-image-pe-blocks](https://huggingface.co/diffusers/minimax-h3-image-pe-blocks) custom blocks run it as a pipeline of its own. They need `xgrammar`, and about 18 GB of GPU memory.

```py
import torch
from diffusers import ModularPipeline

enhancer = ModularPipeline.from_pretrained("diffusers/minimax-h3-image-pe-blocks", trust_remote_code=True)
enhancer.load_components(dtype=torch.bfloat16, device_map="cuda")

state = enhancer(prompt="a cozy cabin in the snow")
prompt, aspect_ratio = state.get("prompt"), state.get("aspect_ratio")
```

Pass the reference images as `image` to expand an editing instruction instead. Then pass `prompt` to the image pipeline, with a `height` and `width` that match `aspect_ratio`.

## Image size

`height` and `width` are the size of the returned image and default to 2048. The model generates at 4 megapixels or more: a smaller request is generated on a larger canvas of the same aspect ratio and resized back, so a 1024x1024 request costs as much as a 2048x2048 one. `min_generation_area` sets that threshold in pixels; pass `1024` to always generate at the requested size. Either way the canvas sides are rounded to multiples of 32.

MiniMax recommends 4096x4096 for 1:1, 4736x3552 for 4:3, 5024x3360 for 3:2, 5472x3072 for 16:9 and 6272x2688 for 21:9, or the same sizes with height and width swapped for portrait formats.

## Sampling

- The checkpoint samples in 6 steps over its `sample_sigmas`, and the transformer has one output head per step, so there is no `num_inference_steps`.
- The model is guidance-distilled, so there is no guider, `negative_prompt` or `guidance_scale`.

## MiniMaxH3ImageModularPipeline[[diffusers.MiniMaxH3ImageModularPipeline]]

#### diffusers.MiniMaxH3ImageModularPipeline[[diffusers.MiniMaxH3ImageModularPipeline]]

```python
diffusers.MiniMaxH3ImageModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/minimax_h3_image/modular_pipeline.py#L27)

A ModularPipeline for MiniMax-H3-Image: text-to-image (`text2image`) and reference-image editing
(`image_conditioned`).

MiniMax-H3-Image runs the MiniMax-H3 transformer without audio over one packed sequence of text rows, the rows of
the reference images and the rows of the generated image, which is a single latent frame. The checkpoint is
distilled twice over: guidance is baked into the weights, so there is no guider, no `negative_prompt` and no
`guidance_scale`, and it samples on a fixed grid of six steps with one output head of the transformer per step, so
there is no `num_inference_steps` either.

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3-Image")
pipe.load_components(dtype=torch.bfloat16)
```

## MiniMaxH3ImageBlocks[[diffusers.MiniMaxH3ImageBlocks]]

#### diffusers.MiniMaxH3ImageBlocks[[diffusers.MiniMaxH3ImageBlocks]]

```python
diffusers.MiniMaxH3ImageBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/modular_pipelines/minimax_h3_image/modular_blocks_minimax_h3_image.py#L371)

Modular pipeline blocks for MiniMax-H3-Image: `text2image`, and reference-image editing (`image_conditioned`) when
`image` is passed.

Supported workflows:
- `text2image`: requires `prompt`
- `image_conditioned`: requires `prompt`, `image`

Components:
image_processor (`VaeImageProcessor`) text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
(`Qwen2Tokenizer`) processor (`Qwen3VLProcessor`) vae (`AutoencoderKLMiniMaxH3Image`) scheduler
(`MiniMaxH3Scheduler`) transformer (`MiniMaxH3ImageTransformer2DModel`)

Configs:
reference_min_short_edge (default: 1024) reference_max_short_edge (default: 2048) vision_min_short_edge
(default: 1024) vision_max_short_edge (default: 1024) sample_sigmas (default: [1.0, 0.9675236940383911,
0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]): The sampling grid the checkpoint's
per-step output heads were distilled on.

Inputs:
image (`Image | list`, *optional*):
The reference image of the edit, or a list of them. The prompt refers to them in order as `<Picture 1>`,
`<Picture 2>`, ...
prompt (`str`):
The prompt to guide generation, a single string.
vision_images (`list`, *optional*):
The reference images resized for the text encoder.
reference_images (`list`, *optional*):
The reference images resized for the VAE.
generator (`Generator`, *optional*):
The generator of the request. One posterior sample per reference is drawn from it.
height (`int`):
Height of the returned image in pixels.
width (`int`):
Width of the returned image in pixels.
min_generation_area (`int`, *optional*, defaults to 4194304):
The smallest area, in pixels, an image is generated at. A smaller `height x width` is generated on a
canvas scaled up to this area and resized back to `height x width` after decoding. Pass 1024 to always
generate at the requested size.
reference_latents (`list`, *optional*):
The conditioning latents of the reference images. Their shape is where every reference's geometry comes
from.
latents (`Tensor`):
Pre-generated noise of shape `(1, latent_channels, latent_height, latent_width)`, used instead of the
draw.
reference_noise (`list`, *optional*):
Pre-generated noise for the reference images, one tensor of the shape of each entry of
`reference_latents`, used instead of the draws.
**denoiser_input_fields (`None`, *optional*):
The structural description of the packed sequence the transformer reads by name: `token_tags`,
`position_ids` and the two row-index tensors.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
images (`list`):
Generated images.

### Kandinsky 6
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/kandinsky6.md
