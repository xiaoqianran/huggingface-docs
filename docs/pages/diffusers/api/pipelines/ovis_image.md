# Ovis-Image

![concepts](https://github.com/AIDC-AI/Ovis-Image/blob/main/docs/imgs/ovis_image_case.png)

Ovis-Image is a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints.

[Ovis-Image Technical Report](https://arxiv.org/abs/2511.22982) from Alibaba Group, by Guo-Hua Wang, Liangfu Cao, Tianyu Cui, Minghao Fu, Xiaohao Chen, Pengxin Zhan, Jianshan Zhao, Lan Li, Bowen Fu, Jiaqi Liu, Qing-Guo Chen.

The abstract from the paper is:

*We introduce Ovis-Image, a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints. Built upon our previous Ovis-U1 framework, Ovis-Image integrates a diffusion-based visual decoder with the stronger Ovis 2.5 multimodal backbone, leveraging a text-centric training pipeline that combines large-scale pre-training with carefully tailored post-training refinements. Despite its compact architecture, Ovis-Image achieves text rendering performance on par with significantly larger open models such as Qwen-Image and approaches closed-source systems like Seedream and GPT4o. Crucially, the model remains deployable on a single high-end GPU with moderate memory, narrowing the gap between frontier-level text rendering and practical deployment. Our results indicate that combining a strong multimodal backbone with a carefully designed, text-focused training recipe is sufficient to achieve reliable bilingual text rendering without resorting to oversized or proprietary models.*

**Highlights**: 

*   **Strong text rendering at a compact 7B scale**: Ovis-Image is a 7B text-to-image model that delivers text rendering quality comparable to much larger 20B-class systems such as Qwen-Image and competitive with leading closed-source models like GPT4o in text-centric scenarios, while remaining small enough to run on widely accessible hardware.
*   **High fidelity on text-heavy, layout-sensitive prompts**: The model excels on prompts that demand tight alignment between linguistic content and rendered typography (e.g., posters, banners, logos, UI mockups, infographics), producing legible, correctly spelled, and semantically consistent text across diverse fonts, sizes, and aspect ratios without compromising overall visual quality.
*   **Efficiency and deployability**: With its 7B parameter budget and streamlined architecture, Ovis-Image fits on a single high-end GPU with moderate memory, supports low-latency interactive use, and scales to batch production serving, bringing near–frontier text rendering to applications where tens-of-billions–parameter models are impractical.

This pipeline was contributed by Ovis-Image Team. The original codebase can be found [here](https://github.com/AIDC-AI/Ovis-Image).

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`AIDC-AI/Ovis-Image-7B`](https://huggingface.co/AIDC-AI/Ovis-Image-7B) | `torch.bfloat16` |

Refer to [this](https://huggingface.co/collections/AIDC-AI/ovis-image) collection for more information.

## OvisImagePipeline[[diffusers.OvisImagePipeline]]

#### diffusers.OvisImagePipeline[[diffusers.OvisImagePipeline]]

```python
diffusers.OvisImagePipeline(scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, text_encoder: Qwen3Model, tokenizer: Qwen2Tokenizer, transformer: OvisImageTransformer2DModel)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/ovis_image/pipeline_ovis_image.py#L129)

**Parameters:**

transformer ([OvisImageTransformer2DModel](/docs/diffusers/v0.42.1/en/api/models/ovisimage_transformer2d#diffusers.OvisImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.42.1/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.42.1/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen3Model`) : Text encoder of class [Qwen3Model](https://huggingface.co/docs/transformers/en/model_doc/qwen3#transformers.Qwen3Model).

tokenizer (`Qwen2TokenizerFast`) : Tokenizer of class [Qwen2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2TokenizerFast).

The Ovis-Image pipeline for text-to-image generation.

Reference: https://github.com/AIDC-AI/Ovis-Image

#### __call__[[diffusers.OvisImagePipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = '', guidance_scale: float = 5.0, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, sigmas: list[float] | None = None, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, output_type: str | None = 'pil', return_dict: bool = True, joint_attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 256)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/ovis_image/pipeline_ovis_image.py#L457)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is not greater than `1`).

guidance_scale (`float`, *optional*, defaults to 1.0) : True classifier-free guidance (guidance scale) is enabled when `guidance_scale` > 1 and `negative_prompt` is provided.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

joint_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 512) : Maximum sequence length to use with the `prompt`.

**Returns:** [OvisImagePipelineOutput](/docs/diffusers/v0.42.1/en/api/pipelines/ovis_image#diffusers.pipelines.ovis_image.OvisImagePipelineOutput) or `tuple`

[OvisImagePipelineOutput](/docs/diffusers/v0.42.1/en/api/pipelines/ovis_image#diffusers.pipelines.ovis_image.OvisImagePipelineOutput) if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import OvisImagePipeline

>>> pipe = OvisImagePipeline.from_pretrained("AIDC-AI/Ovis-Image-7B", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = 'A creative 3D artistic render where the text "OVIS-IMAGE" is written in a bold, expressive handwritten brush style using thick, wet oil paint. The paint is a mix of vibrant rainbow colors (red, blue, yellow) swirling together like toothpaste or impasto art. You can see the ridges of the brush bristles and the glossy, wet texture of the paint. The background is a clean artist's canvas. Dynamic lighting creates soft shadows behind the floating paint strokes. Colorful, expressive, tactile texture, 4k detail.'
>>> image = pipe(prompt, negative_prompt="", num_inference_steps=50, guidance_scale=5.0).images[0]
>>> image.save("ovis_image.png")
```

#### encode_prompt[[diffusers.OvisImagePipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, max_sequence_length: int = 256, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/ovis_image/pipeline_ovis_image.py#L269)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. Used only when `do_classifier_free_guidance` is `True`. If not defined, an empty string is used.

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to also encode the `negative_prompt` for classifier-free guidance.

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

max_sequence_length (`int`, *optional*, defaults to 256) : Maximum sequence length to use for the `prompt`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. If not provided, they are generated from `negative_prompt`.

## OvisImagePipelineOutput[[diffusers.pipelines.ovis_image.OvisImagePipelineOutput]]

#### diffusers.pipelines.ovis_image.OvisImagePipelineOutput[[diffusers.pipelines.ovis_image.OvisImagePipelineOutput]]

```python
diffusers.pipelines.ovis_image.OvisImagePipelineOutput(images: list)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/ovis_image/pipeline_output.py#L24)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for Ovis-Image pipelines.

### Flux3
https://huggingface.co/docs/diffusers/v0.42.1/api/pipelines/flux3.md

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

# FLUX 3 Image Klein

FLUX 3 Image Klein is a 5.6B image generation model from Black Forest Labs: a single transformer ([Flux3Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/flux3_transformer#diffusers.Flux3Transformer2DModel)) generates images from a text prompt, alone or together with one or more reference images. Text conditioning comes from Qwen3-VL, and images use the FLUX.2 packed autoencoder (`AutoencoderKLFlux2`).

FLUX 3 Image Klein ships as a [modular pipeline](../../modular_diffusers/overview). Get started with a text-to-image generation:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to("cuda")

image = pipe(
    prompt="A red fox in a snow-covered forest clearing at golden hour",
    generator=torch.Generator().manual_seed(0),
    output="images",
)[0]
image.save("fox.png")
```

`prompt` is the only required input: the pipeline carries the recommended sampling settings (50 steps, guidance scale 3.5, scheduler shift 6.93) and generates 1024 x 1024 images by default. It takes one prompt per call; use `num_images_per_prompt` for several images of the same prompt.

As you can see in the example above, loading and running differ a little from a `DiffusionPipeline`. Loading is lazy: [from_pretrained()](/docs/diffusers/v0.42.1/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.from_pretrained) only builds the pipeline, and you call [load_components()](/docs/diffusers/v0.42.1/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.load_components) to load the weights. A call returns the pipeline state, and you ask for what you need with `output`: `output="images"` here, or `output=["images", "prompt"]` to also get the caption the prompt upsampler wrote.

> [!TIP]
> If you're working with `ModularPipeline` for the first time, read the [quickstart](../../modular_diffusers/quickstart), and [Coming from DiffusionPipeline](../../modular_diffusers/coming_from_diffusion_pipeline) if you know the standard pipelines. Besides loading and outputs, a few other things differ: [guidance](#guidance-scale), [callbacks](#per-step-callbacks) and [offloading](#memory).

## Prompt upsampling

Prompting matters: FLUX 3 Image Klein takes full scene captions that describe the subject, composition, lighting, and atmosphere, not short prompts; a request sent as typed gives generic output. That is why the pipeline runs the prompt upsampler first on every request, text-to-image and editing alike. It is a separate 9B model from Black Forest Labs ([black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)) that expands `prompt` into such a caption before the text encoder, and a short prompt is all it takes.

The prompt upsampler turns a short request, together with the reference images when editing, into the full scene caption the model takes: medium and style, scene, lighting and camera, each subject with its colors, materials and position in the frame. One model covers text-to-image, editing with reference images and layout with boxes. The caption it wrote comes back as the `prompt` output. In the example above, ask for it alongside the images to see what the image model received:

```python
out = pipe(
    prompt="A red fox in a snow-covered forest clearing at golden hour",
    generator=torch.Generator().manual_seed(0),
    output=["images", "prompt"],
)
print(out["prompt"])
```

```
Full-color outdoor wildlife photograph taken from a low eye-level perspective in a snow-covered forest clearing at golden hour. In the center foreground, a red fox stands in profile facing left, its body turned toward the viewer and its head raised as it looks across the clearing. The fox has dense rust-red winter fur, a white throat and chest, black lower legs, pointed ears with dark tips, and a long bushy tail ending in white. ...
```

The full caption runs to about 240 words; `max_new_tokens` caps its length (3584 by default).

You can print the step's `doc` to see what it takes and what it writes:

```python
print(pipe.blocks.sub_blocks["prompt_upsampler"].doc)
```

```
  Inputs:
      prompt (`str`):
          The edit instruction to expand into the full caption (one per call).
      image (`Image | list`, *optional*):
          The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
      boxes (`list`, *optional*):
          Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max, x_max] in
          0-1: ...
      max_new_tokens (`int`, *optional*, defaults to 3584):
          The most tokens the upsampler may generate for the caption.

  Outputs:
      prompt (`str`):
          The full caption, replacing the request.
```

Like other blocks in the pipeline, the `prompt_upsampler` block also runs on its own. The recommended way is to pop it out of the blocks: the popped block becomes a standalone upsampler, and the remaining blocks become a pipeline that takes the expanded caption as `prompt`:

```python
blocks = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base").blocks
pup_block = blocks.sub_blocks.pop("prompt_upsampler")

pup = pup_block.init_pipeline("black-forest-labs/flux-3-image-klein-base")
pup.load_components(dtype=torch.bfloat16)
pup.to("cuda")
caption = pup(prompt="A red fox in a snow-covered forest clearing at golden hour", output="prompt")

pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.to("cuda")
image = pipe(prompt=caption, generator=torch.Generator().manual_seed(0), output="images")[0]
```

This way it can be served as a separate stage, on its own GPU or behind its own endpoint, and a UI can show the caption it wrote and let the user edit it before generating. Any other LLM can take the upsampler's place, with the same system prompts: give it `system_prompts["t2i"]` from the upsampler repository (or `"i2i"`, with the references first in the user message) as the system prompt and the request as the user message, and pass the caption it returns as `prompt` to `pipe`.

## Image-to-image

Pass one or more reference images with `image`. A single reference is kept whole: it is downscaled only above 4 megapixels, padded to the 16-pixel grid, and the generation defaults to its size, so the edit comes back at the size of the input. Several references are each fitted under 1 megapixel and cropped to the grid, and the generation defaults to the first one's size. A given `height`/`width` overrides the default. Describe the *target* image as a complete caption of the reference with your edit folded in, rather than a one-line instruction, or let the [prompt upsampler](#prompt-upsampling) write it from the instruction and the references:

```python
night = pipe(
    prompt="the same scene at night under a full moon",
    image=image,
    generator=torch.Generator().manual_seed(0),
    output=["images", "prompt"],
)
```

With several references, refer to them as the first image, the second image, ... in the order they are passed to `image`; the caption names them `<ref_image_0>`, `<ref_image_1>`, ... for the image model:

```python
combined = pipe(
    prompt="the fox from the first image sitting on the wooden bench from the second image",
    image=[image, bench],
    height=768,
    width=1360,
    output="images",
)[0]
```

Pass `height` and/or `width` to override the derived size.

## Layout control

To place content explicitly, pass `boxes`: for each element an `id`, a short `desc`, and a `bbox` of `[y_min, x_min, y_max, x_max]` in 0-1 (origin top-left), so it is independent of the output size. The upsampler writes the caption around them, citing each element as `<id>`, and appends the table of boxes, yours kept exactly:

```python
boxes = [
    {"id": "balloon_1", "desc": "a red and yellow striped hot-air balloon", "bbox": [0.08, 0.06, 0.48, 0.38]},
    {"id": "balloon_2", "desc": "a small blue hot-air balloon", "bbox": [0.20, 0.62, 0.42, 0.80]},
]
out = pipe(prompt="hot-air balloons over a green valley at sunrise", boxes=boxes, output=["images", "prompt"])
```

When editing, a box names its source with `from`: `"target"` with a `tgt_bbox` adds a new element there, `"ref_image_0"` with a `src_bbox` and a `tgt_bbox` moves an element of that reference (equal boxes keep it in place):

```python
boxes = [{"id": "cactus", "desc": "a small potted cactus", "from": "target", "src_bbox": None, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]}]
out = pipe(prompt="add a small potted cactus in the marked spot", image=mug, boxes=boxes, output=["images", "prompt"])

boxes = [{"id": "knight", "desc": "the knight figure", "from": "ref_image_0", "src_bbox": [0.50, 0.15, 0.85, 0.35], "tgt_bbox": [0.19, 0.06, 0.54, 0.26]}]
out = pipe(prompt="move the knight up and to the left, keep the rest unchanged", image=knight, boxes=boxes, output=["images", "prompt"])
```

The caption the image model reads has the elements tagged `<id>` in the text and the table appended as JSON, with the boxes on a 0-1000 canvas; print the `prompt` output to see it. A caption written in that form by hand goes to the pipeline without the step, as in [Prompt upsampling](#prompt-upsampling).

## Guidance scale

The pipeline applies classifier-free guidance through its `guider` component and does not take a `guidance_scale` argument at call time. The default guider is [ClassifierFreeGuidance](/docs/diffusers/v0.42.1/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance) with `guidance_scale=3.5`, the recommended setting for both workflows. To change it, create an updated copy of the guider:

```python
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

See the [guiders guide](../../using-diffusers/guiders) to switch to a different guidance method.

## Step-distilled model

The step-distilled checkpoint generates in 16 steps without classifier-free guidance. Its transformer is a flow map: at every step it is told the current timestep and the one the step jumps to. It loads like the base model (the repo's `modular_model_index.json` resolves to [Flux3DistilledModularPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/flux3#diffusers.Flux3DistilledModularPipeline) and [Flux3DistilledAutoBlocks](/docs/diffusers/v0.42.1/en/api/pipelines/flux3#diffusers.Flux3DistilledAutoBlocks)); the pipeline has no `guider` and does not take `negative_prompt`:

```python
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="images")[0]
```

It serves the same two workflows as the base model. `num_inference_steps` defaults to 16; the model is trained for 4 to 16 steps.

## Reproducible generation

A `generator` fixes the noise the pipeline draws (see [Reproducibility](../../using-diffusers/reusing_seeds)). To reuse noise drawn elsewhere, pass it as `latents` with shape `(batch_size, 128, height // 16, width // 16)`. For example, the [BFL implementation](https://github.com/black-forest-labs/flux-3) draws its noise as image tokens on the CPU and casts it to bf16; this reproduces its sample for a given seed:

```python
height, width = 1024, 1024
h, w = height // 16, width // 16

noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
latents = noise.T.reshape(1, 128, h, w)

image = pipe(prompt=prompt, latents=latents, height=height, width=width, output="images")[0]
```

## Workflows

The pipeline serves two workflows, `text2image` and `image_conditioned`; the one that runs is picked by the inputs (`image` selects image-to-image). [from_pretrained()](/docs/diffusers/v0.42.1/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.from_pretrained) can also build a pipeline for one workflow with the `workflow` argument, which then only expects and loads the components that workflow needs:

```python
from diffusers.modular_pipelines import Flux3AutoBlocks

print(Flux3AutoBlocks().available_workflows)
# ["text2image", "image_conditioned"]

t2i = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base", workflow="text2image")
print(t2i.blocks.doc)
```

## Per-step callbacks

To use an existing callback function, insert a block into the denoising loop. The [callbacks guide](../../using-diffusers/callback#modular-pipelines) has a simple example with the `CallbackOnStepEnd` block and the `walk_denoise_loops` helper:

```python
blocks = Flux3AutoBlocks()  # or Flux3DistilledAutoBlocks
for loop in walk_denoise_loops(blocks):
    loop.sub_blocks["callback"] = CallbackOnStepEnd()
pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")

image = pipe(prompt=caption, callback_on_step_end=my_callback, output="images")[0]
```

To stop early, raise from the callback; there is no `interrupt` flag.

## Memory

Loading everything in bf16 takes about 39 GB of weights: the transformer is 5.5B parameters, the Qwen3-VL-4B text encoder 4B and the [prompt upsampler](#prompt-upsampling) 9B (about 19 GB on its own), with the image VAE in fp32. A pipeline built without the upsampler step (see there) loads about 20 GB. For GPUs with less memory, enable component offloading; the manager moves components between the CPU and the GPU as each step needs them, the upsampler included:

```python
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base", components_manager=manager)
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
manager.enable_auto_cpu_offload(device="cuda")
```

## Single-file loading

[Flux3Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/flux3_transformer#diffusers.Flux3Transformer2DModel) loads the transformer straight from a Black Forest Labs model package with [from_single_file()](/docs/diffusers/v0.42.1/en/api/loaders/single_file#diffusers.loaders.FromOriginalModelMixin.from_single_file). The package `config.json` is not read; pass the diffusers config with `config` and `subfolder`.

```python
import torch
from diffusers import Flux3Transformer2DModel, ModularPipeline

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/model.safetensors",
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    dtype=torch.bfloat16,
)

pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein-base")
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
pipe.update_components(transformer=transformer)
pipe.to("cuda")
```

Load the other components by name first: [load_components()](/docs/diffusers/v0.42.1/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline.load_components) without `names` would also load the transformer from the pipeline repo.

Loading the FP8r and NVFP4 single-file checkpoints are also supported. Both load through the same call with a [BFLQuantizationConfig](/docs/diffusers/v0.42.1/en/api/quantization#diffusers.BFLQuantizationConfig). See the [BFL quantization guide](../../quantization/bfl) for the kernel requirements of each scheme.

```python
from diffusers import BFLQuantizationConfig

transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/variants/nvfp4/model.safetensors",
    config="black-forest-labs/flux-3-image-klein-base",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
```

When loading the step-distilled checkpoints, we need their
configs (`flow_map=True`):

```python
transformer = Flux3Transformer2DModel.from_single_file(
    "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/variants/nvfp4/model.safetensors",
    config="black-forest-labs/flux-3-image-klein",
    subfolder="transformer",
    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16,
)
```

## Flux3ModularPipeline[[diffusers.Flux3ModularPipeline]]

#### diffusers.Flux3ModularPipeline[[diffusers.Flux3ModularPipeline]]

```python
diffusers.Flux3ModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/flux3/modular_pipeline.py#L24)

A ModularPipeline for FLUX.3.

## Flux3AutoBlocks[[diffusers.Flux3AutoBlocks]]

#### diffusers.Flux3AutoBlocks[[diffusers.Flux3AutoBlocks]]

```python
diffusers.Flux3AutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/flux3/modular_blocks_flux3.py#L339)

Auto blocks that perform text-to-image and image-to-image generation using FLUX.3. `prompt` goes through the prompt
upsampler first; pop the `prompt_upsampler` step to encode it as given.

Supported workflows:
- `text2image`: requires `prompt`
- `image_conditioned`: requires `prompt`, `image`

Components:
prompt_upsampler (`Qwen3_5ForConditionalGeneration`): The prompt upsampler. prompt_upsampler_processor
(`AutoProcessor`): Its processor. text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
(`AutoProcessor`) image_processor (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`) transformer
(`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`ClassifierFreeGuidance`)

Inputs:
prompt (`str`):
The edit instruction to expand into the full caption (one per call).
image (`Image | list`, *optional*):
The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
boxes (`list`, *optional*):
Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max,
x_max] in 0-1: 'from': 'target' with `tgt_bbox` adds an element there, 'from': 'ref_image_0', ... with
`src_bbox` and `tgt_bbox` moves one from a reference image.
max_new_tokens (`int`, *optional*, defaults to 3584):
The most tokens the upsampler may generate for the caption.
negative_prompt (`str`, *optional*):
The prompt not to guide the generation
max_sequence_length (`int`, *optional*, defaults to 8192):
Maximum number of prompt tokens; longer prompts are truncated.
text_encoder_out_layers (`tuple`, *optional*, defaults to (4, 8, 12, 16, 20, 24, 28, 32)):
Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings
num_images_per_prompt (`int`, *optional*, defaults to 1):
The number of images to generate per prompt.
image_cond_latents (`list`, *optional*):
Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
per image
height (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The height in pixels of the generated image; defaults to the first reference image's own height
width (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The width in pixels of the generated image; defaults to the first reference image's own width
reference_crop (`list | NoneType`, *optional*):
Per reference, the box of the reference inside its padded canvas or None, from the preprocess step
latents (`Tensor`):
Pre-generated noisy latents for image generation.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`):
The number of denoising steps.
timesteps (`Tensor`):
Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
sigmas (`list`, *optional*):
Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
**denoiser_input_fields (`None`, *optional*):
conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
attention_kwargs (`dict`, *optional*):
Additional kwargs for attention processors.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
images (`list`):
Generated images.
latents (`Tensor`):
Denoised latents.

## Flux3DistilledAutoBlocks[[diffusers.Flux3DistilledAutoBlocks]]

#### diffusers.Flux3DistilledAutoBlocks[[diffusers.Flux3DistilledAutoBlocks]]

```python
diffusers.Flux3DistilledAutoBlocks()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/flux3/modular_blocks_flux3_distilled.py#L258)

Auto blocks that perform text-to-image and image-to-image generation using the step-distilled FLUX.3 model (16
steps, no classifier-free guidance). `prompt` goes through the prompt upsampler first; pop the `prompt_upsampler`
step to encode it as given.

Supported workflows:
- `text2image`: requires `prompt`
- `image_conditioned`: requires `prompt`, `image`

Components:
prompt_upsampler (`Qwen3_5ForConditionalGeneration`): The prompt upsampler. prompt_upsampler_processor
(`AutoProcessor`): Its processor. text_encoder (`Qwen3VLForConditionalGeneration`) tokenizer
(`AutoProcessor`) image_processor (`Flux3ImageProcessor`) vae (`AutoencoderKLFlux2`) transformer
(`Flux3Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

Inputs:
prompt (`str`):
The edit instruction to expand into the full caption (one per call).
image (`Image | list`, *optional*):
The reference image(s) of the edit, at most 10; the same ones the pipeline receives.
boxes (`list`, *optional*):
Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as [y_min, x_min, y_max,
x_max] in 0-1: 'from': 'target' with `tgt_bbox` adds an element there, 'from': 'ref_image_0', ... with
`src_bbox` and `tgt_bbox` moves one from a reference image.
max_new_tokens (`int`, *optional*, defaults to 3584):
The most tokens the upsampler may generate for the caption.
max_sequence_length (`int`, *optional*, defaults to 8192):
Maximum number of prompt tokens; longer prompts are truncated.
text_encoder_out_layers (`tuple`, *optional*, defaults to (4, 8, 12, 16, 20, 24, 28, 32)):
Indices of the text-encoder hidden layers whose states are stacked to form the text embeddings
num_images_per_prompt (`int`, *optional*, defaults to 1):
The number of images to generate per prompt.
image_cond_latents (`list`, *optional*):
Normalized packed latents of the reference images, one tensor of shape (B, C, height_i//16, width_i//16)
per image
height (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The height in pixels of the generated image; defaults to the first reference image's own height
width (`int`, *optional*, defaults to None or 1024, depending on the workflow):
The width in pixels of the generated image; defaults to the first reference image's own width
reference_crop (`list | NoneType`, *optional*):
Per reference, the box of the reference inside its padded canvas or None, from the preprocess step
latents (`Tensor`):
Pre-generated noisy latents for image generation.
generator (`Generator`, *optional*):
Torch generator for deterministic generation.
num_inference_steps (`int`):
The number of denoising steps.
timesteps (`Tensor`):
Timesteps for the denoising process. (overrides `num_inference_steps` when provided)
sigmas (`list`, *optional*):
Custom sigmas for the denoising process. (overrides `num_inference_steps` when provided)
**denoiser_input_fields (`None`, *optional*):
conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
output_type (`str`, *optional*, defaults to pil):
Output format: 'pil', 'np', 'pt'.

Outputs:
images (`list`):
Generated images.
latents (`Tensor`):
Denoised latents.

## Flux3DistilledModularPipeline[[diffusers.Flux3DistilledModularPipeline]]

#### diffusers.Flux3DistilledModularPipeline[[diffusers.Flux3DistilledModularPipeline]]

```python
diffusers.Flux3DistilledModularPipeline(blocks: diffusers.modular_pipelines.modular_pipeline.ModularPipelineBlocks | None = None, pretrained_model_name_or_path: str | os.PathLike | None = None, components_manager: diffusers.modular_pipelines.components_manager.ComponentsManager | None = None, collection: str | None = None, workflow: str | None = None, modular_config_dict: dict[str, typing.Any] | None = None, config_dict: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/flux3/modular_pipeline.py#L52)

A ModularPipeline for the step-distilled FLUX.3 model (16 steps, no classifier-free guidance).

### Latent Consistency Models
https://huggingface.co/docs/diffusers/v0.42.1/api/pipelines/latent_consistency_models.md
