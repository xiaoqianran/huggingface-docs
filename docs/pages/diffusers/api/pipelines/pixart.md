# PixArt-α

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pixart/header_collage.png)

[PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://huggingface.co/papers/2310.00426) is Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li.

The abstract from the paper is:

*The most advanced text-to-image (T2I) models require significant training costs (e.g., millions of GPU hours), seriously hindering the fundamental innovation for the AIGC community while increasing CO2 emissions. This paper introduces PIXART-α, a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost, as shown in Figure 1 and 2. To achieve this goal, three core designs are proposed: (1) Training strategy decomposition: We devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; (2) Efficient T2I Transformer: We incorporate cross-attention modules into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; (3) High-informative data: We emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. As a result, PIXART-α's training speed markedly surpasses existing large-scale T2I models, e.g., PIXART-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%. Extensive experiments demonstrate that PIXART-α excels in image quality, artistry, and semantic control. We hope PIXART-α will provide new insights to the AIGC community and startups to accelerate building their own high-quality yet low-cost generative models from scratch.*

You can find the original codebase at [PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) and all the available checkpoints at [PixArt-alpha](https://huggingface.co/PixArt-alpha).

Some notes about this pipeline:

* It uses a Transformer backbone (instead of a UNet) for denoising. As such it has a similar architecture as [DiT](./dit).
* It was trained using text conditions computed from T5. This aspect makes the pipeline better at following complex text prompts with intricate details.
* It is good at producing high-resolution images at different aspect ratios. To get the best results, the authors recommend some size brackets which can be found [here](https://github.com/PixArt-alpha/PixArt-alpha/blob/08fbbd281ec96866109bdd2cdb75f2f58fb17610/diffusion/data/datasets/utils.py).
* It rivals the quality of state-of-the-art text-to-image generation systems (as of this writing) such as Stable Diffusion XL, Imagen, and DALL-E 2, while being more efficient than them.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Inference with under 8GB GPU VRAM

Run the [PixArtAlphaPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/pixart#diffusers.PixArtAlphaPipeline) with under 8GB GPU VRAM by loading the text encoder in 8-bit precision. Let's walk through a full-fledged example.

First, install the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library:

```bash
pip install -U bitsandbytes
```

Then load the text encoder in 8-bit:

```python
from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline
import torch

text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="text_encoder",
    load_in_8bit=True,
    device_map="auto",

)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=text_encoder,
    transformer=None,
    device_map="auto"
)
```

Now, use the `pipe` to encode a prompt:

```python
with torch.no_grad():
    prompt = "cute cat"
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
```

Since text embeddings have been computed, remove the `text_encoder` and `pipe` from the memory, and free up some GPU VRAM:

```python
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()

del text_encoder
del pipe
flush()
```

Then compute the latents with the prompt embeddings as inputs:

```python
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")

latents = pipe(
    negative_prompt=None,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    num_images_per_prompt=1,
    output_type="latent",
).images

del pipe.transformer
flush()
```

> [!TIP]
> Notice that while initializing `pipe`, you're setting `text_encoder` to `None` so that it's not loaded.

Once the latents are computed, pass it off to the VAE to decode into a real image:

```python
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")[0]
image.save("cat.png")
```

By deleting components you aren't using and flushing the GPU VRAM, you should be able to run [PixArtAlphaPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/pixart#diffusers.PixArtAlphaPipeline) with under 8GB GPU VRAM.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pixart/8bits_cat.png)

If you want a report of your memory-usage, run this [script](https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e).

> [!WARNING]
> Text embeddings computed in 8-bit can impact the quality of the generated images because of the information loss in the representation space caused by the reduced precision. It's recommended to compare the outputs with and without 8-bit.

While loading the `text_encoder`, you set `load_in_8bit` to `True`. You could also specify `load_in_4bit` to bring your memory requirements down even further to under 7GB.

## PixArtAlphaPipeline[[diffusers.PixArtAlphaPipeline]]

#### diffusers.PixArtAlphaPipeline[[diffusers.PixArtAlphaPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py#L241)

Pipeline for text-to-image generation using PixArt-Alpha.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.PixArtAlphaPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py#L686[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str = ''"}, {"name": "num_inference_steps", "val": ": int = 20"}, {"name": "timesteps", "val": ": list = None"}, {"name": "sigmas", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 4.5"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback", "val": ": typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None"}, {"name": "callback_steps", "val": ": int = 1"}, {"name": "clean_caption", "val": ": bool = True"}, {"name": "use_resolution_binning", "val": ": bool = True"}, {"name": "max_sequence_length", "val": ": int = 120"}, {"name": "**kwargs", "val": ""}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **num_inference_steps** (`int`, *optional*, defaults to 100) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to 4.5) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size) --
  The height in pixels of the generated image.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size) --
  The width in pixels of the generated image.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
  applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **prompt_attention_mask** (`torch.Tensor`, *optional*) -- Pre-generated attention mask for text embeddings.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **negative_prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for negative text embeddings.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion.IFPipelineOutput` instead of a plain tuple.
- **callback** (`Callable`, *optional*) --
  A function that will be called every `callback_steps` steps during inference. The function will be
  called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
- **callback_steps** (`int`, *optional*, defaults to 1) --
  The frequency at which the `callback` function will be called. If not specified, the callback will be
  called at every step.
- **clean_caption** (`bool`, *optional*, defaults to `True`) --
  Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
  be installed. If the dependencies are not installed, the embeddings will be created from the raw
  prompt.
- **use_resolution_binning** (`bool` defaults to `True`) --
  If set to `True`, the requested height and width are first mapped to the closest resolutions using
  `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
  the requested resolution. Useful for generating non-square images.
- **max_sequence_length** (`int` defaults to 120) -- Maximum sequence length to use with the `prompt`.0[ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple`If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import PixArtAlphaPipeline

>>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
>>> pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
>>> # Enable memory optimizations.
>>> pipe.enable_model_cpu_offload()

>>> prompt = "A small cactus with a happy face in the Sahara desert."
>>> image = pipe(prompt).images[0]
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. PixArt-Alpha uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([PixArtTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/pixart_transformer2d#diffusers.PixArtTransformer2DModel)) : A text conditioned `PixArtTransformer2DModel` to denoise the encoded image latents. Initially published as [`Transformer2DModel`](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/blob/main/transformer/config.json#L2) in the config, but the mismatch can be ignored.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

`[ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) or `tuple``

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/ddim#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images
#### encode_prompt[[diffusers.PixArtAlphaPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py#L303)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For PixArt-Alpha, this should be "".

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : whether to use classifier free guidance or not

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the "" string.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 120) : Maximum sequence length to use for the prompt.

### Hunyuan Video15
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/hunyuan_video15.md

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

# HunyuanVideo-1.5

HunyuanVideo-1.5 is a lightweight yet powerful video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture with selective and sliding tile attention (SSTA), enhanced bilingual understanding through glyph-aware text encoding, progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions. Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source models.

You can find all the original HunyuanVideo checkpoints under the [Tencent](https://huggingface.co/tencent) organization.

> [!TIP]
> Click on the HunyuanVideo models in the right sidebar for more examples of video generation tasks.
>
> The examples below use a checkpoint from [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) because the weights are stored in a layout compatible with Diffusers.

The example below demonstrates how to generate a video optimized for memory or inference speed.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

```py
import torch
from diffusers import AutoModel, HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

pipeline = HunyuanVideo15Pipeline.from_pretrained(
    "HunyuanVideo-1.5-Diffusers-480p_t2v",
    torch_dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

## Notes

- HunyuanVideo1.5 use attention masks with variable-length sequences. For best performance, we recommend using an attention backend that handles padding efficiently.

    - **H100/H800:** `_flash_3_hub` or `_flash_3_varlen_hub`
    - **A100/A800/RTX 4090:** `flash_hub` or `flash_varlen_hub`
    - **Other GPUs:** `sage_hub`

Refer to the [Attention backends](../../optimization/attention_backends) guide for more details about using a different backend.

```py
pipe.transformer.set_attention_backend("flash_hub")  # or your preferred backend
```

- [HunyuanVideo15Pipeline](/docs/diffusers/v0.39.0/en/api/pipelines/hunyuan_video15#diffusers.HunyuanVideo15Pipeline) use guider and does not take `guidance_scale` parameter at runtime. 

You can check the default guider configuration using `pipe.guider`:

```py
>>> pipe.guider 
ClassifierFreeGuidance {
  "_class_name": "ClassifierFreeGuidance",
  "_diffusers_version": "0.36.0.dev0",
  "enabled": true,
  "guidance_rescale": 0.0,
  "guidance_scale": 6.0,
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
```

To update guider configuration, you can run `pipe.guider = pipe.guider.new(...)`

```py
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

Read more on Guider [here](../../using-diffusers/guiders).

## HunyuanVideo15Pipeline[[diffusers.HunyuanVideo15Pipeline]]

#### diffusers.HunyuanVideo15Pipeline[[diffusers.HunyuanVideo15Pipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L166)

Pipeline for text-to-video generation using HunyuanVideo1.5.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.HunyuanVideo15Pipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L542[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list = None"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_2", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask_2", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_2", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask_2", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead.
- **height** (`int`, *optional*) --
  The height in pixels of the generated video.
- **width** (`int`, *optional*) --
  The width in pixels of the generated video.
- **num_frames** (`int`, defaults to `121`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality video at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Pre-generated mask for prompt embeddings.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Pre-generated mask for negative prompt embeddings.
- **prompt_embeds_2** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.
- **prompt_embeds_mask_2** (`torch.Tensor`, *optional*) --
  Pre-generated mask for prompt embeddings from the second text encoder.
- **negative_prompt_embeds_2** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings from the second text encoder.
- **negative_prompt_embeds_mask_2** (`torch.Tensor`, *optional*) --
  Pre-generated mask for negative prompt embeddings from the second text encoder.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated video. Choose between "np", "pt", or "latent".
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `HunyuanVideo15PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).0`~HunyuanVideo15PipelineOutput` or `tuple`If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideo15Pipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v"
>>> pipe = HunyuanVideo15Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> output = pipe(
...     prompt="A cat walks on the grass, realistic",
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=15)
```

**Parameters:**

transformer ([HunyuanVideo15Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.HunyuanVideo15Transformer3DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLHunyuanVideo15](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_hunyuan_video15#diffusers.AutoencoderKLHunyuanVideo15)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider ([ClassifierFreeGuidance](/docs/diffusers/v0.39.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance)) : [ClassifierFreeGuidance]for classifier free guidance.

**Returns:**

``~HunyuanVideo15PipelineOutput` or `tuple``

If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.
#### encode_prompt[[diffusers.HunyuanVideo15Pipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L334)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.
#### prepare_cond_latents_and_mask[[diffusers.HunyuanVideo15Pipeline.prepare_cond_latents_and_mask]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L508)

Prepare conditional latents and mask for t2v generation.

**Parameters:**

latents : Main latents tensor (B, C, F, H, W)

**Returns:**

`tuple`

(cond_latents_concat, mask_concat) - both are zero tensors for t2v

## HunyuanVideo15ImageToVideoPipeline[[diffusers.HunyuanVideo15ImageToVideoPipeline]]

#### diffusers.HunyuanVideo15ImageToVideoPipeline[[diffusers.HunyuanVideo15ImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L193)

Pipeline for image-to-video generation using HunyuanVideo1.5.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.HunyuanVideo15ImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L650[{"name": "image", "val": ": Image"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list = None"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_2", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask_2", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_2", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask_2", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}]- **image** (`PIL.Image.Image`) --
  The input image to condition video generation on.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the video generation. If not defined, one has to pass
  `negative_prompt_embeds` instead.
- **num_frames** (`int`, defaults to `121`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality video at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Pre-generated mask for prompt embeddings.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Pre-generated mask for negative prompt embeddings.
- **prompt_embeds_2** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.
- **prompt_embeds_mask_2** (`torch.Tensor`, *optional*) --
  Pre-generated mask for prompt embeddings from the second text encoder.
- **negative_prompt_embeds_2** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings from the second text encoder.
- **negative_prompt_embeds_mask_2** (`torch.Tensor`, *optional*) --
  Pre-generated mask for negative prompt embeddings from the second text encoder.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated video. Choose between "np", "pt", or "latent".
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `HunyuanVideo15PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).0`~HunyuanVideo15PipelineOutput` or `tuple`If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideo15ImageToVideoPipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v"
>>> pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")

>>> output = pipe(
...     prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
...     image=image,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=24)
```

**Parameters:**

transformer ([HunyuanVideo15Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.HunyuanVideo15Transformer3DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLHunyuanVideo15](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_hunyuan_video15#diffusers.AutoencoderKLHunyuanVideo15)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider ([ClassifierFreeGuidance](/docs/diffusers/v0.39.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance)) : [ClassifierFreeGuidance]for classifier free guidance.

image_encoder (`SiglipVisionModel`) : [SiglipVisionModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipVisionModel) variant.

feature_extractor (`SiglipImageProcessor`) : [SiglipImageProcessor](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipImageProcessor) variant.

**Returns:**

``~HunyuanVideo15PipelineOutput` or `tuple``

If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.
#### encode_prompt[[diffusers.HunyuanVideo15ImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L422)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.
#### prepare_cond_latents_and_mask[[diffusers.HunyuanVideo15ImageToVideoPipeline.prepare_cond_latents_and_mask]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L594)

Prepare conditional latents and mask for t2v generation.

**Parameters:**

latents : Main latents tensor (B, C, F, H, W)

**Returns:**

`tuple`

(cond_latents_concat, mask_concat) - both are zero tensors for t2v

## HunyuanVideo15PipelineOutput[[diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput]]

#### diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput[[diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_output.py#L9)

Output class for HunyuanVideo1.5 pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### LLaDA2
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/llada2.md
