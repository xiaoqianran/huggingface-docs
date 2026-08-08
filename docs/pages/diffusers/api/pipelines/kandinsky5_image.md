# Kandinsky 5.0 Image

[Kandinsky 5.0](https://arxiv.org/abs/2511.14993) is a family of diffusion models for Video & Image generation. 

Kandinsky 5.0 Image Lite is a lightweight image generation model (6B parameters).

The model introduces several key innovations:
- **Latent diffusion pipeline** with **Flow Matching** for improved training stability
- **Diffusion Transformer (DiT)** as the main generative backbone with cross-attention to text embeddings
- Dual text encoding using **Qwen2.5-VL** and **CLIP** for comprehensive text understanding
- **Flux VAE** for efficient image encoding and decoding

The original codebase can be found at [kandinskylab/Kandinsky-5](https://github.com/kandinskylab/Kandinsky-5).

> [!TIP]
> Check out the [Kandinsky Lab](https://huggingface.co/kandinskylab) organization on the Hub for the official model checkpoints for text-to-video generation, including pretrained, SFT, no-CFG, and distilled variants.

## Available Models

Kandinsky 5.0 Image Lite:

| model_id | Description | Use Cases |
|------------|-------------|-----------|
| [**kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers) | 6B image Supervised Fine-Tuned model | Highest generation quality |
| [**kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers) | 6B image editing Supervised Fine-Tuned model | Highest generation quality |
| [**kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers) | 6B image Base pretrained model | Research and fine-tuning |
| [**kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers) | 6B image editing Base pretrained model | Research and fine-tuning |

## Usage Examples

### Basic Text-to-Image Generation

```python
import torch
from diffusers import Kandinsky5T2IPipeline

# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
pipe = Kandinsky5T2IPipeline.from_pretrained(model_id)
_ = pipe.to(device='cuda',dtype=torch.bfloat16)

# Generate image
prompt = "A fluffy, expressive cat wearing a bright red hat with a soft, slightly textured fabric. The hat should look cozy and well-fitted on the cat’s head. On the front of the hat, add clean, bold white text that reads “SWEET”, clearly visible and neatly centered. Ensure the overall lighting highlights the hat’s color and the cat’s fur details."

output = pipe(
    prompt=prompt,
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
).image[0]
```

### Basic Image-to-Image Generation

```python
import torch
from diffusers import Kandinsky5I2IPipeline
from diffusers.utils import load_image 
# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
pipe = Kandinsky5I2IPipeline.from_pretrained(model_id)

_ = pipe.to(device='cuda',dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()                                               # <--- Enable CPU offloading for single GPU inference

# Edit the input image
image = load_image(
    "https://huggingface.co/kandinsky-community/kandinsky-3/resolve/main/assets/title.jpg?download=true"
)

prompt = "Change the background from a winter night scene to a bright summer day. Place the character on a sandy beach with clear blue sky, soft sunlight, and gentle waves in the distance. Replace the winter clothing with a light short-sleeved T-shirt (in soft pastel colors) and casual shorts. Ensure the character’s fur reflects warm daylight instead of cold winter tones. Add small beach details such as seashells, footprints in the sand, and a few scattered beach toys nearby. Keep the oranges in the scene, but place them naturally on the sand."
negative_prompt = ""

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=3.5,
).image[0]
```

## Kandinsky5T2IPipeline[[diffusers.Kandinsky5T2IPipeline]]

#### diffusers.Kandinsky5T2IPipeline[[diffusers.Kandinsky5T2IPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py#L120)

Pipeline for text-to-image generation using Kandinsky 5.0.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.Kandinsky5T2IPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py#L534[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 1024"}, {"name": "width", "val": ": int = 1024"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 3.5"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_qwen", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_clip", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_qwen", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_clip", "val": ": torch.Tensor | None = None"}, {"name": "prompt_cu_seqlens", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_cu_seqlens", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, NoneType], diffusers.callbacks.PipelineCallback | diffusers.callbacks.MultiPipelineCallbacks]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
  instead. Ignored when not using guidance (`guidance_scale` 0`~KandinskyImagePipelineOutput` or `tuple`If `return_dict` is `True`, `KandinskyImagePipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

The call function to the pipeline for text-to-image generation.

Examples:

```python
>>> import torch
>>> from diffusers import Kandinsky5T2IPipeline

>>> # Available models:
>>> # kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers
>>> # kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers

>>> model_id = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
>>> pipe = Kandinsky5T2IPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen."

>>> output = pipe(
...     prompt=prompt,
...     negative_prompt="",
...     height=1024,
...     width=1024,
...     num_inference_steps=50,
...     guidance_scale=3.5,
... ).frames[0]
```

**Parameters:**

transformer (`Kandinsky5Transformer3DModel`) : Conditional Transformer to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder Model [black-forest-labs/FLUX.1-dev (vae)](https://huggingface.co/black-forest-labs/FLUX.1-dev) to encode and decode videos to and from latent representations.

text_encoder (`Qwen2_5_VLForConditionalGeneration`) : Frozen text-encoder [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

tokenizer (`AutoProcessor`) : Tokenizer for Qwen2.5-VL.

text_encoder_2 (`CLIPTextModel`) : Frozen [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

tokenizer_2 (`CLIPTokenizer`) : Tokenizer for CLIP.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

``~KandinskyImagePipelineOutput` or `tuple``

If `return_dict` is `True`, `KandinskyImagePipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### check_inputs[[diffusers.Kandinsky5T2IPipeline.check_inputs]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py#L380)

Validate input parameters for the pipeline.

**Parameters:**

prompt : Input prompt

negative_prompt : Negative prompt for guidance

height : Image height

width : Image width

prompt_embeds_qwen : Pre-computed Qwen prompt embeddings

prompt_embeds_clip : Pre-computed CLIP prompt embeddings

negative_prompt_embeds_qwen : Pre-computed Qwen negative prompt embeddings

negative_prompt_embeds_clip : Pre-computed CLIP negative prompt embeddings

prompt_cu_seqlens : Pre-computed cumulative sequence lengths for Qwen positive prompt

negative_prompt_cu_seqlens : Pre-computed cumulative sequence lengths for Qwen negative prompt

callback_on_step_end_tensor_inputs : Callback tensor inputs
#### encode_prompt[[diffusers.Kandinsky5T2IPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py#L289)

Encodes a single prompt (positive or negative) into text encoder hidden states.

This method combines embeddings from both Qwen2.5-VL and CLIP text encoders to create comprehensive text
representations for image generation.

**Parameters:**

prompt (`str` or `list[str]`) : Prompt to be encoded.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images to generate per prompt.

max_sequence_length (`int`, *optional*, defaults to 512) : Maximum sequence length for text encoding. Must be less than 1024

device (`torch.device`, *optional*) : Torch device.

dtype (`torch.dtype`, *optional*) : Torch dtype.

**Returns:**

`tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

- Qwen text embeddings of shape (batch_size * num_images_per_prompt, sequence_length, embedding_dim)
- CLIP pooled embeddings of shape (batch_size * num_images_per_prompt, clip_embedding_dim)
- Cumulative sequence lengths (`cu_seqlens`) for Qwen embeddings of shape (batch_size *
  num_images_per_prompt + 1,)
#### prepare_latents[[diffusers.Kandinsky5T2IPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_t2i.py#L469)

Prepare initial latent variables for text-to-image generation.

This method creates random noise latents

**Parameters:**

batch_size (int) : Number of images to generate

num_channels_latents (int) : Number of channels in latent space

height (int) : Height of generated image

width (int) : Width of generated image

dtype (torch.dtype) : Data type for latents

device (torch.device) : Device to create latents on

generator (torch.Generator) : Random number generator

latents (torch.Tensor) : Pre-existing latents to use

**Returns:**

`torch.Tensor`

Prepared latent tensor

## Kandinsky5I2IPipeline[[diffusers.Kandinsky5I2IPipeline]]

#### diffusers.Kandinsky5I2IPipeline[[diffusers.Kandinsky5I2IPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2i.py#L120)

Pipeline for image-to-image generation using Kandinsky 5.0.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.Kandinsky5I2IPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2i.py#L567[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 3.5"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_qwen", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_clip", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_qwen", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_clip", "val": ": torch.Tensor | None = None"}, {"name": "prompt_cu_seqlens", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_cu_seqlens", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, NoneType], diffusers.callbacks.PipelineCallback | diffusers.callbacks.MultiPipelineCallbacks]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 1024"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
  instead. Ignored when not using guidance (`guidance_scale` 0`~KandinskyImagePipelineOutput` or `tuple`If `return_dict` is `True`, `KandinskyImagePipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

The call function to the pipeline for image-to-image generation.

Examples:

```python
>>> import torch
>>> from diffusers import Kandinsky5I2IPipeline

>>> # Available models:
>>> # kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers
>>> # kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers

>>> model_id = "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
>>> pipe = Kandinsky5I2IPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen."

>>> output = pipe(
...     prompt=prompt,
...     negative_prompt="",
...     height=1024,
...     width=1024,
...     num_inference_steps=50,
...     guidance_scale=3.5,
... ).frames[0]
```

**Parameters:**

transformer (`Kandinsky5Transformer3DModel`) : Conditional Transformer to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder Model [black-forest-labs/FLUX.1-dev (vae)](https://huggingface.co/black-forest-labs/FLUX.1-dev) to encode and decode videos to and from latent representations.

text_encoder (`Qwen2_5_VLForConditionalGeneration`) : Frozen text-encoder [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

tokenizer (`AutoProcessor`) : Tokenizer for Qwen2.5-VL.

text_encoder_2 (`CLIPTextModel`) : Frozen [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

tokenizer_2 (`CLIPTokenizer`) : Tokenizer for CLIP.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

``~KandinskyImagePipelineOutput` or `tuple``

If `return_dict` is `True`, `KandinskyImagePipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### check_inputs[[diffusers.Kandinsky5I2IPipeline.check_inputs]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2i.py#L388)

Validate input parameters for the pipeline.

**Parameters:**

prompt : Input prompt

negative_prompt : Negative prompt for guidance

image : Input image for conditioning

height : Image height

width : Image width

prompt_embeds_qwen : Pre-computed Qwen prompt embeddings

prompt_embeds_clip : Pre-computed CLIP prompt embeddings

negative_prompt_embeds_qwen : Pre-computed Qwen negative prompt embeddings

negative_prompt_embeds_clip : Pre-computed CLIP negative prompt embeddings

prompt_cu_seqlens : Pre-computed cumulative sequence lengths for Qwen positive prompt

negative_prompt_cu_seqlens : Pre-computed cumulative sequence lengths for Qwen negative prompt

callback_on_step_end_tensor_inputs : Callback tensor inputs
#### encode_prompt[[diffusers.Kandinsky5I2IPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2i.py#L295)

Encodes a single prompt (positive or negative) into text encoder hidden states.

This method combines embeddings from both Qwen2.5-VL and CLIP text encoders to create comprehensive text
representations for image generation.

**Parameters:**

prompt (`str` or `list[str]`) : Prompt to be encoded.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images to generate per prompt.

max_sequence_length (`int`, *optional*, defaults to 1024) : Maximum sequence length for text encoding. Must be less than 1024

device (`torch.device`, *optional*) : Torch device.

dtype (`torch.dtype`, *optional*) : Torch dtype.

**Returns:**

`tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

- Qwen text embeddings of shape (batch_size * num_images_per_prompt, sequence_length, embedding_dim)
- CLIP pooled embeddings of shape (batch_size * num_images_per_prompt, clip_embedding_dim)
- Cumulative sequence lengths (`cu_seqlens`) for Qwen embeddings of shape (batch_size *
  num_images_per_prompt + 1,)
#### prepare_latents[[diffusers.Kandinsky5I2IPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/kandinsky5/pipeline_kandinsky_i2i.py#L482)

Prepare initial latent variables for image-to-image generation.

This method creates random noise latents with encoded image,

**Parameters:**

image (PipelineImageInput) : Input image to condition the generation on

batch_size (int) : Number of images to generate

num_channels_latents (int) : Number of channels in latent space

height (int) : Height of generated image

width (int) : Width of generated image

dtype (torch.dtype) : Data type for latents

device (torch.device) : Device to create latents on

generator (torch.Generator) : Random number generator

latents (torch.Tensor) : Pre-existing latents to use

**Returns:**

`torch.Tensor`

Prepared latent tensor with encoded image

## Citation
```bibtex
@misc{kandinsky2025,
    author = {Alexander Belykh and Alexander Varlamov and Alexey Letunovskiy and Anastasia Aliaskina and Anastasia Maltseva and Anastasiia Kargapoltseva and Andrey Shutkin and Anna Averchenkova and Anna Dmitrienko and Bulat Akhmatov and Denis Dimitrov and Denis Koposov and Denis Parkhomenko and Dmitrii and Ilya Vasiliev and Ivan Kirillov and Julia Agafonova and Kirill Chernyshev and Kormilitsyn Semen and Lev Novitskiy and Maria Kovaleva and Mikhail Mamaev and Mikhailov and Nikita Kiselev and Nikita Osterov and Nikolai Gerasimenko and Nikolai Vaulin and Olga Kim and Olga Vdovchenko and Polina Gavrilova and Polina Mikhailova and Tatiana Nikulina and Viacheslav Vasilev and Vladimir Arkhipkin and Vladimir Korviakov and Vladimir Polovnikov and Yury Kolabushin},
    title = {Kandinsky 5.0: A family of diffusion models for Video & Image generation},
    howpublished = {\url{https://github.com/kandinskylab/Kandinsky-5}},
    year = 2025
}
```

### Cogvideox
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/cogvideox.md

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

  
    
      
    
  

# CogVideoX

[CogVideoX](https://huggingface.co/papers/2408.06072) is a large diffusion transformer model - available in 2B and 5B parameters - designed to generate longer and more consistent videos from text. This model uses a 3D causal variational autoencoder to more efficiently process video data by reducing sequence length (and associated training compute) and preventing flickering in generated videos. An "expert" transformer with adaptive LayerNorm improves alignment between text and video, and 3D full attention helps accurately capture motion and time in generated videos.

You can find all the original CogVideoX checkpoints under the [CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) collection.

> [!TIP]
> Click on the CogVideoX models in the right sidebar for more examples of other video generation tasks.

The example below demonstrates how to generate a video optimized for memory or inference speed.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The quantized CogVideoX 5B model below requires ~16GB of VRAM.

```py
import torch
from diffusers import CogVideoXPipeline, AutoModel, TorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from torchao.quantization import Int8WeightOnlyConfig

# quantize weights to int8 with torchao
pipeline_quant_config = PipelineQuantizationConfig(
  quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig())}
)

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    transformer=transformer,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

# model-offloading
pipeline.enable_model_cpu_offload()

prompt = """
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. 
The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. 
Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, 
with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
"""

video = pipeline(
    prompt=prompt,
    guidance_scale=6,
    num_inference_steps=50
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster.

The average inference time with torch.compile on a 80GB A100 is 76.27 seconds compared to 96.89 seconds for an uncompiled model.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = """
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. 
The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. 
Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, 
with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
"""

video = pipeline(
    prompt=prompt,
    guidance_scale=6,
    num_inference_steps=50
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

## Notes

- CogVideoX supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.CogVideoXLoraLoaderMixin.load_lora_weights).

  
  Show example code

  ```py
  import torch
  from diffusers import CogVideoXPipeline
  from diffusers.hooks import apply_group_offloading
  from diffusers.utils import export_to_video

  pipeline = CogVideoXPipeline.from_pretrained(
      "THUDM/CogVideoX-5b",
      torch_dtype=torch.bfloat16
  )
  pipeline.to("cuda")

  # load LoRA weights
  pipeline.load_lora_weights("finetrainers/CogVideoX-1.5-crush-smol-v0", adapter_name="crush-lora")
  pipeline.set_adapters("crush-lora", 0.9)

  # model-offloading
  pipeline.enable_model_cpu_offload()

  prompt = """
  PIKA_CRUSH A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press.
  """
  negative_prompt = "inconsistent motion, blurry motion, worse quality, degenerate outputs, deformed outputs"

  video = pipeline(
      prompt=prompt, 
      negative_prompt=negative_prompt, 
      num_frames=81, 
      height=480,
      width=768,
      num_inference_steps=50
  ).frames[0]
  export_to_video(video, "output.mp4", fps=16)
  ```

  

- The text-to-video (T2V) checkpoints work best with a resolution of 1360x768 because that was the resolution it was pretrained on.

- The image-to-video (I2V) checkpoints work with multiple resolutions. The width can vary from 768 to 1360, but the height must be 758. Both height and width must be divisible by 16.

- Both T2V and I2V checkpoints work best with 81 and 161 frames. It is recommended to export the generated video at 16fps.

- Refer to the table below to view memory usage when various memory-saving techniques are enabled.

  | method | memory usage (enabled) | memory usage (disabled) |
  |---|---|---|
  | enable_model_cpu_offload | 19GB | 33GB |
  | enable_sequential_cpu_offload | <4GB | ~33GB (very slow inference speed) |
  | enable_tiling | 11GB (with enable_model_cpu_offload) | --- |
 
## CogVideoXPipeline[[diffusers.CogVideoXPipeline]]

#### diffusers.CogVideoXPipeline[[diffusers.CogVideoXPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L147)

Pipeline for text-to-video generation using CogVideoX.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.CogVideoXPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L505[{"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_frames", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "guidance_scale", "val": ": float = 6"}, {"name": "use_dynamic_cfg", "val": ": bool = False"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 226"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The height in pixels of the generated image. This is set to 480 by default for the best results.
- **width** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The width in pixels of the generated image. This is set to 720 by default for the best results.
- **num_frames** (`int`, defaults to `48`) --
  Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
  contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
  num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
  needs to be satisfied is that of divisibility mentioned above.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **guidance_scale** (`float`, *optional*, defaults to 7.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **use_dynamic_cfg** (`bool`, *optional*, defaults to `False`) --
  If True, dynamically adjusts the guidance scale during inference.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
  to [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.FloatTensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `226`) --
  Maximum sequence length in encoded prompt. Must be consistent with
  `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.0[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CogVideoXPipeline
>>> from diffusers.utils import export_to_video

>>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
>>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
>>> prompt = (
...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
...     "atmosphere of this unique musical performance."
... )
>>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
>>> export_to_video(video, "output.mp4", fps=8)
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CogVideoXTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) : A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple``

[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.CogVideoXPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L244)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### fuse_qkv_projections[[diffusers.CogVideoXPipeline.fuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L428)

Enables fused QKV projections.
#### unfuse_qkv_projections[[diffusers.CogVideoXPipeline.unfuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L433)

Disable QKV projection fusion if enabled.

## CogVideoXImageToVideoPipeline[[diffusers.CogVideoXImageToVideoPipeline]]

#### diffusers.CogVideoXImageToVideoPipeline[[diffusers.CogVideoXImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L160)

Pipeline for image-to-video generation using CogVideoX.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.CogVideoXImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L598[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_frames", "val": ": int = 49"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "guidance_scale", "val": ": float = 6"}, {"name": "use_dynamic_cfg", "val": ": bool = False"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 226"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The height in pixels of the generated image. This is set to 480 by default for the best results.
- **width** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The width in pixels of the generated image. This is set to 720 by default for the best results.
- **num_frames** (`int`, defaults to `48`) --
  Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
  contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
  num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
  needs to be satisfied is that of divisibility mentioned above.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **guidance_scale** (`float`, *optional*, defaults to 7.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **use_dynamic_cfg** (`bool`, *optional*, defaults to `False`) --
  If True, dynamically adjusts the guidance scale during inference.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
  to [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.FloatTensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `226`) --
  Maximum sequence length in encoded prompt. Must be consistent with
  `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.0[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import CogVideoXImageToVideoPipeline
>>> from diffusers.utils import export_to_video, load_image

>>> pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
... )
>>> video = pipe(image, prompt, use_dynamic_cfg=True)
>>> export_to_video(video.frames[0], "output.mp4", fps=8)
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CogVideoXTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) : A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple``

[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.CogVideoXImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L263)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### fuse_qkv_projections[[diffusers.CogVideoXImageToVideoPipeline.fuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L519)

Enables fused QKV projections.
#### unfuse_qkv_projections[[diffusers.CogVideoXImageToVideoPipeline.unfuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L525)

Disable QKV projection fusion if enabled.

## CogVideoXVideoToVideoPipeline[[diffusers.CogVideoXVideoToVideoPipeline]]

#### diffusers.CogVideoXVideoToVideoPipeline[[diffusers.CogVideoXVideoToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L169)

Pipeline for video-to-video generation using CogVideoX.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.CogVideoXVideoToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L575[{"name": "video", "val": ": list = None"}, {"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "strength", "val": ": float = 0.8"}, {"name": "guidance_scale", "val": ": float = 6"}, {"name": "use_dynamic_cfg", "val": ": bool = False"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 226"}]- **video** (`list[PIL.Image.Image]`) --
  The input video to condition the generation on. Must be a list of images/frames of the video.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The height in pixels of the generated image. This is set to 480 by default for the best results.
- **width** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The width in pixels of the generated image. This is set to 720 by default for the best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **strength** (`float`, *optional*, defaults to 0.8) --
  Higher strength leads to more differences between original video and generated video.
- **guidance_scale** (`float`, *optional*, defaults to 7.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **use_dynamic_cfg** (`bool`, *optional*, defaults to `False`) --
  If True, dynamically adjusts the guidance scale during inference.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
  to [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.FloatTensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `226`) --
  Maximum sequence length in encoded prompt. Must be consistent with
  `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.0[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline
>>> from diffusers.utils import export_to_video, load_video

>>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
>>> pipe = CogVideoXVideoToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

>>> input_video = load_video(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
... )
>>> prompt = (
...     "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
...     "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
...     "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
...     "moons, but the remainder of the scene is mostly realistic."
... )

>>> video = pipe(
...     video=input_video, prompt=prompt, strength=0.8, guidance_scale=6, num_inference_steps=50
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=8)
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CogVideoXTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) : A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple``

[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.CogVideoXVideoToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L269)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### fuse_qkv_projections[[diffusers.CogVideoXVideoToVideoPipeline.fuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L496)

Enables fused QKV projections.
#### unfuse_qkv_projections[[diffusers.CogVideoXVideoToVideoPipeline.unfuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py#L502)

Disable QKV projection fusion if enabled.

## CogVideoXFunControlPipeline[[diffusers.CogVideoXFunControlPipeline]]

#### diffusers.CogVideoXFunControlPipeline[[diffusers.CogVideoXFunControlPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L154)

Pipeline for controlled text-to-video generation using CogVideoX Fun.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.CogVideoXFunControlPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L551[{"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "control_video", "val": ": list[PIL.Image.Image] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "guidance_scale", "val": ": float = 6"}, {"name": "use_dynamic_cfg", "val": ": bool = False"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "control_video_latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 226"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **control_video** (`list[PIL.Image.Image]`) --
  The control video to condition the generation on. Must be a list of images/frames of the video. If not
  provided, `control_video_latents` must be provided.
- **height** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The height in pixels of the generated image. This is set to 480 by default for the best results.
- **width** (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial) --
  The width in pixels of the generated image. This is set to 720 by default for the best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **guidance_scale** (`float`, *optional*, defaults to 6.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **use_dynamic_cfg** (`bool`, *optional*, defaults to `False`) --
  If True, dynamically adjusts the guidance scale during inference.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
  to [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **control_video_latents** (`torch.Tensor`, *optional*) --
  Pre-generated control latents, sampled from a Gaussian distribution, to be used as inputs for
  controlled video generation. If not provided, `control_video` must be provided.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `226`) --
  Maximum sequence length in encoded prompt. Must be consistent with
  `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.0[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CogVideoXFunControlPipeline, DDIMScheduler
>>> from diffusers.utils import export_to_video, load_video

>>> pipe = CogVideoXFunControlPipeline.from_pretrained(
...     "alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose", torch_dtype=torch.bfloat16
... )
>>> pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
>>> pipe.to("cuda")

>>> control_video = load_video(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
... )
>>> prompt = (
...     "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
...     "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
...     "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
...     "moons, but the remainder of the scene is mostly realistic."
... )

>>> video = pipe(prompt=prompt, control_video=control_video).frames[0]
>>> export_to_video(video, "output.mp4", fps=8)
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. CogVideoX uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CogVideoXTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cogvideox_transformer3d#diffusers.CogVideoXTransformer3DModel)) : A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) or `tuple``

[CogVideoXPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogvideox#diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.CogVideoXFunControlPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L253)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### fuse_qkv_projections[[diffusers.CogVideoXFunControlPipeline.fuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L473)

Enables fused QKV projections.
#### unfuse_qkv_projections[[diffusers.CogVideoXFunControlPipeline.unfuse_qkv_projections]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py#L478)

Disable QKV projection fusion if enabled.

## CogVideoXPipelineOutput[[diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput]]

#### diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput[[diffusers.pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogvideo/pipeline_output.py#L9)

Output class for CogVideo pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Skyreels V2
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/skyreels_v2.md

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

  
    
      
    
  

# SkyReels-V2: Infinite-length Film Generative model

[SkyReels-V2](https://huggingface.co/papers/2504.13074) by the SkyReels Team from Skywork AI.

*Recent advances in video generation have been driven by diffusion models and autoregressive frameworks, yet critical challenges persist in harmonizing prompt adherence, visual quality, motion dynamics, and duration: compromises in motion dynamics to enhance temporal visual quality, constrained video duration (5-10 seconds) to prioritize resolution, and inadequate shot-aware generation stemming from general-purpose MLLMs' inability to interpret cinematic grammar, such as shot composition, actor expressions, and camera motions. These intertwined limitations hinder realistic long-form synthesis and professional film-style generation. To address these limitations, we propose SkyReels-V2, an Infinite-length Film Generative Model, that synergizes Multi-modal Large Language Model (MLLM), Multi-stage Pretraining, Reinforcement Learning, and Diffusion Forcing Framework. Firstly, we design a comprehensive structural representation of video that combines the general descriptions by the Multi-modal LLM and the detailed shot language by sub-expert models. Aided with human annotation, we then train a unified Video Captioner, named SkyCaptioner-V1, to efficiently label the video data. Secondly, we establish progressive-resolution pretraining for the fundamental video generation, followed by a four-stage post-training enhancement: Initial concept-balanced Supervised Fine-Tuning (SFT) improves baseline quality; Motion-specific Reinforcement Learning (RL) training with human-annotated and synthetic distortion data addresses dynamic artifacts; Our diffusion forcing framework with non-decreasing noise schedules enables long-video synthesis in an efficient search space; Final high-quality SFT refines visual fidelity. All the code and models are available at [this https URL](https://github.com/SkyworkAI/SkyReels-V2).*

You can find all the original SkyReels-V2 checkpoints under the [Skywork](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9) organization.

The following SkyReels-V2 models are supported in Diffusers:
- [SkyReels-V2 DF 1.3B - 540P](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)
- [SkyReels-V2 DF 14B - 540P](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P-Diffusers)
- [SkyReels-V2 DF 14B - 720P](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P-Diffusers)
- [SkyReels-V2 T2V 14B - 540P](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P-Diffusers)
- [SkyReels-V2 T2V 14B - 720P](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-720P-Diffusers)
- [SkyReels-V2 I2V 1.3B - 540P](https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers)
- [SkyReels-V2 I2V 14B - 540P](https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P-Diffusers)
- [SkyReels-V2 I2V 14B - 720P](https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P-Diffusers)

This model was contributed by [M. Tolga Cangöz](https://github.com/tolgacangoz).

> [!TIP]
> Click on the SkyReels-V2 models in the right sidebar for more examples of video generation.

### A _Visual_ Demonstration

The example below has the following parameters:

- `base_num_frames=97`
- `num_frames=97`
- `num_inference_steps=30`
- `ar_step=5`
- `causal_block_size=5`

With `vae_scale_factor_temporal=4`, expect `5` blocks of `5` frames each as calculated by:

`num_latent_frames: (97-1)//vae_scale_factor_temporal+1 = 25 frames -> 5 blocks of 5 frames each`

And the maximum context length in the latent space is calculated with `base_num_latent_frames`:

`base_num_latent_frames = (97-1)//vae_scale_factor_temporal+1 = 25 -> 25//5 = 5 blocks`

Asynchronous Processing Timeline:
```text
┌─────────────────────────────────────────────────────────────────┐
│ Steps:    1    6   11   16   21   26   31   36   41   46   50   │
│ Block 1: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                       │
│ Block 2:      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                  │
│ Block 3:           [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]             │
│ Block 4:                [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]        │
│ Block 5:                     [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]   │
└─────────────────────────────────────────────────────────────────┘
```

For Long Videos (`num_frames` > `base_num_frames`):
`base_num_frames` acts as the "sliding window size" for processing long videos.

Example: `257`-frame video with `base_num_frames=97`, `overlap_history=17`
```text
┌──── Iteration 1 (frames 1-97) ────┐
│ Processing window: 97 frames      │ → 5 blocks,
│ Generates: frames 1-97            │   async processing
└───────────────────────────────────┘
            ┌────── Iteration 2 (frames 81-177) ──────┐
            │ Processing window: 97 frames            │
            │ Overlap: 17 frames (81-97) from prev    │ → 5 blocks,
            │ Generates: frames 98-177                │   async processing
            └─────────────────────────────────────────┘
                        ┌────── Iteration 3 (frames 161-257) ──────┐
                        │ Processing window: 97 frames             │
                        │ Overlap: 17 frames (161-177) from prev   │ → 5 blocks,
                        │ Generates: frames 178-257                │   async processing
                        └──────────────────────────────────────────┘
```

Each iteration independently runs the asynchronous processing with its own `5` blocks.
`base_num_frames` controls:
1. Memory usage (larger window = more VRAM)
2. Model context length (must match training constraints)
3. Number of blocks per iteration (`base_num_latent_frames // causal_block_size`)

Each block takes `30` steps to complete denoising.
Block N starts at step: `1 + (N-1) x ar_step`
Total steps: `30 + (5-1) x 5 = 50` steps

Synchronous mode (`ar_step=0`) would process all blocks/frames simultaneously:
```text
┌──────────────────────────────────────────────┐
│ Steps:       1            ...            30  │
│ All blocks: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] │
└──────────────────────────────────────────────┘
```
Total steps: `30` steps

An example on how the step matrix is constructed for asynchronous processing:
Given the parameters: (`num_inference_steps=30, flow_shift=8, num_frames=97, ar_step=5, causal_block_size=5`)
```
- num_latent_frames = (97 frames - 1) // (4 temporal downsampling) + 1 = 25
- step_template = [999, 995, 991, 986, 980, 975, 969, 963, 956, 948,
                   941, 932, 922, 912, 901, 888, 874, 859, 841, 822,
                   799, 773, 743, 708, 666, 615, 551, 470, 363, 216]
```

The algorithm creates a `50x25` `step_matrix` where:
```
- Row 1:  [999×5, 999×5, 999×5, 999×5, 999×5]
- Row 2:  [995×5, 999×5, 999×5, 999×5, 999×5]
- Row 3:  [991×5, 999×5, 999×5, 999×5, 999×5]
- ...
- Row 7:  [969×5, 995×5, 999×5, 999×5, 999×5]
- ...
- Row 21: [799×5, 888×5, 941×5, 975×5, 999×5]
- ...
- Row 35: [  0×5, 216×5, 666×5, 822×5, 901×5]
- ...
- Row 42: [  0×5,   0×5,   0×5, 551×5, 773×5]
- ...
- Row 50: [  0×5,   0×5,   0×5,   0×5, 216×5]
```

Detailed Row `6` Analysis:
```
- step_matrix[5]:      [ 975×5,  999×5,   999×5,   999×5,   999×5]
- step_index[5]:       [   6×5,    1×5,     0×5,     0×5,     0×5]
- step_update_mask[5]: [True×5, True×5, False×5, False×5, False×5]
- valid_interval[5]:   (0, 25)
```

Key Pattern: Block `i` lags behind Block `i-1` by exactly `ar_step=5` timesteps, creating the
staggered "diffusion forcing" effect where later blocks condition on cleaner earlier blocks.

### Text-to-Video Generation

The example below demonstrates how to generate a video from text.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

From the original repo:
>You can use --ar_step 5 to enable asynchronous inference. When asynchronous inference, --causal_block_size 5 is recommended while it is not supposed to be set for synchronous generation... Asynchronous inference will take more steps to diffuse the whole sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous inference may improve the instruction following and visual consistent performance.

```py
import torch
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipeline.to("cuda")
flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

output = pipeline(
    prompt=prompt,
    num_inference_steps=30,
    height=544,  # 720 for 720P
    width=960,   # 1280 for 720P
    num_frames=97,
    base_num_frames=97,  # 121 for 720P
    ar_step=5,  # Controls asynchronous inference (0 for synchronous mode)
    causal_block_size=5,  # Number of frames in each block for asynchronous processing
    overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos; 17 for long video generations
    addnoise_condition=20,  # Improves consistency in long video generation
).frames[0]
export_to_video(output, "video.mp4", fps=24, quality=8)
```

### First-Last-Frame-to-Video Generation

The example below demonstrates how to use the image-to-video pipeline to generate a video using a text description, a starting frame, and an ending frame.

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

model_id = "Skywork/SkyReels-V2-DF-1.3B-720P-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipeline = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
flow_shift = 5.0  # 8.0 for T2V, 5.0 for I2V
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)

first_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png")
last_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png")

def aspect_ratio_resize(image, pipeline, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

def center_crop_resize(image, height, width):
    # Calculate resize ratio to match first frame dimensions
    resize_ratio = max(width / image.width, height / image.height)

    # Resize the image
    width = round(image.width * resize_ratio)
    height = round(image.height * resize_ratio)
    size = [width, height]
    image = TF.center_crop(image, size)

    return image, height, width

first_frame, height, width = aspect_ratio_resize(first_frame, pipeline)
if last_frame.size != first_frame.size:
    last_frame, _, _ = center_crop_resize(last_frame, height, width)

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

output = pipeline(
    image=first_frame, last_image=last_frame, prompt=prompt, height=height, width=width, guidance_scale=5.0
).frames[0]
export_to_video(output, "video.mp4", fps=24, quality=8)
```

### Video-to-Video Generation

`SkyReelsV2DiffusionForcingVideoToVideoPipeline` extends a given video.

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingVideoToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

model_id = "Skywork/SkyReels-V2-DF-1.3B-720P-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipeline = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
flow_shift = 5.0  # 8.0 for T2V, 5.0 for I2V
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)

video = load_video("input_video.mp4")

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

output = pipeline(
    video=video, prompt=prompt, height=720, width=1280, guidance_scale=5.0, overlap_history=17,
    num_inference_steps=30, num_frames=257, base_num_frames=121#, ar_step=5, causal_block_size=5,
).frames[0]
export_to_video(output, "video.mp4", fps=24, quality=8)
# Total frames will be the number of frames of the given video + 257
```

## Notes

- SkyReels-V2 supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.SkyReelsV2LoraLoaderMixin.load_lora_weights).

`SkyReelsV2Pipeline` and `SkyReelsV2ImageToVideoPipeline` are also available without Diffusion Forcing framework applied.

## SkyReelsV2DiffusionForcingPipeline[[diffusers.SkyReelsV2DiffusionForcingPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingPipeline[[diffusers.SkyReelsV2DiffusionForcingPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L128)

Pipeline for Text-to-Video (t2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

__call__diffusers.SkyReelsV2DiffusionForcingPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L597[{"name": "prompt", "val": ": str | list[str]"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 544"}, {"name": "width", "val": ": int = 960"}, {"name": "num_frames", "val": ": int = 97"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "overlap_history", "val": ": int | None = None"}, {"name": "addnoise_condition", "val": ": float = 0"}, {"name": "base_num_frames", "val": ": int = 97"}, {"name": "ar_step", "val": ": int = 0"}, {"name": "causal_block_size", "val": ": int | None = None"}, {"name": "fps", "val": ": int = 24"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, defaults to `544`) --
  The height of the generated video.
- **width** (`int`, defaults to `960`) --
  The width of the generated video.
- **num_frames** (`int`, defaults to `97`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `6.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `512`) --
  The maximum sequence length of the prompt.
- **overlap_history** (`int`, *optional*, defaults to `None`) --
  Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes
  short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.
- **addnoise_condition** (`float`, *optional*, defaults to `0`) --
  This is used to help smooth the long video generation by adding some noise to the clean condition. Too
  large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger
  ones, but it is recommended to not exceed 50.
- **base_num_frames** (`int`, *optional*, defaults to `97`) --
  97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)
- **ar_step** (`int`, *optional*, defaults to `0`) --
  Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous
  inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed
  to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole
  sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous
  inference may improve the instruction following and visual consistent performance.
- **causal_block_size** (`int`, *optional*, defaults to `None`) --
  The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step >
  0)
- **fps** (`int`, *optional*, defaults to `24`) --
  Frame rate of the generated video0`~SkyReelsV2PipelineOutput` or `tuple`If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import (
...     SkyReelsV2DiffusionForcingPipeline,
...     UniPCMultistepScheduler,
...     AutoencoderKLWan,
... )
>>> from diffusers.utils import export_to_video

>>> # Load the pipeline
>>> # Available models:
>>> # - Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-DF-14B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-DF-14B-720P-Diffusers
>>> vae = AutoencoderKLWan.from_pretrained(
...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
...     subfolder="vae",
...     torch_dtype=torch.float32,
... )
>>> pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
...     vae=vae,
...     torch_dtype=torch.bfloat16,
... )
>>> flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
>>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

>>> output = pipe(
...     prompt=prompt,
...     num_inference_steps=30,
...     height=544,
...     width=960,
...     guidance_scale=6.0,  # 6.0 for T2V, 5.0 for I2V
...     num_frames=97,
...     ar_step=5,  # Controls asynchronous inference (0 for synchronous mode)
...     causal_block_size=5,  # Number of frames processed together in a causal block
...     overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos
...     addnoise_condition=20,  # Improves consistency in long video generation
... ).frames[0]
>>> export_to_video(output, "video.mp4", fps=24, quality=8)
```

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~SkyReelsV2PipelineOutput` or `tuple``

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.SkyReelsV2DiffusionForcingPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L218)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### generate_timestep_matrix[[diffusers.SkyReelsV2DiffusionForcingPipeline.generate_timestep_matrix]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L417)

This function implements the core diffusion forcing algorithm that creates a coordinated denoising schedule
across temporal frames. It supports both synchronous and asynchronous generation modes:

**Synchronous Mode** (ar_step=0, causal_block_size=1):
- All frames are denoised simultaneously at each timestep
- Each frame follows the same denoising trajectory: [1000, 800, 600, ..., 0]
- Simpler but may have less temporal consistency for long videos

**Asynchronous Mode** (ar_step>0, causal_block_size>1):
- Frames are grouped into causal blocks and processed block/chunk-wise
- Each block is denoised in a staggered pattern creating a "denoising wave"
- Earlier blocks are more denoised, later blocks lag behind by ar_step timesteps
- Creates stronger temporal dependencies and better consistency

**Parameters:**

num_latent_frames (int) : Total number of latent frames to generate

step_template (torch.Tensor) : Base timestep schedule (e.g., [1000, 800, 600, ..., 0])

base_num_latent_frames (int) : Maximum frames the model can process in one forward pass

ar_step (int, optional) : Autoregressive step size for temporal lag. 0 = synchronous, >0 = asynchronous. Defaults to 5.

num_pre_ready (int, optional) : Number of frames already denoised (e.g., from prefix in a video2video task). Defaults to 0.

causal_block_size (int, optional) : Number of frames processed as a causal block. Defaults to 1.

shrink_interval_with_mask (bool, optional) : Whether to optimize processing intervals. Defaults to False.

**Returns:**

`tuple containing`

- step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
  [num_iterations, num_latent_frames]
- step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
  num_latent_frames]
- step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
  [num_iterations, num_latent_frames]
- valid_interval (list[tuple]): list of (start, end) intervals for each iteration

## SkyReelsV2DiffusionForcingImageToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L133)

Pipeline for Image-to-Video (i2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

__call__diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L643[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 544"}, {"name": "width", "val": ": int = 960"}, {"name": "num_frames", "val": ": int = 97"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 5.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "image_embeds", "val": ": torch.Tensor | None = None"}, {"name": "last_image", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "overlap_history", "val": ": int | None = None"}, {"name": "addnoise_condition", "val": ": float = 0"}, {"name": "base_num_frames", "val": ": int = 97"}, {"name": "ar_step", "val": ": int = 0"}, {"name": "causal_block_size", "val": ": int | None = None"}, {"name": "fps", "val": ": int = 24"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, defaults to `544`) --
  The height of the generated video.
- **width** (`int`, defaults to `960`) --
  The width of the generated video.
- **num_frames** (`int`, defaults to `97`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `5.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `negative_prompt` input argument.
- **image_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
  image embeddings are generated from the `image` input argument.
- **last_image** (`torch.Tensor`, *optional*) --
  Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
  image embeddings are generated from the `image` input argument.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `512`) --
  The maximum sequence length of the prompt.
- **overlap_history** (`int`, *optional*, defaults to `None`) --
  Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes
  short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.
- **addnoise_condition** (`float`, *optional*, defaults to `0`) --
  This is used to help smooth the long video generation by adding some noise to the clean condition. Too
  large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger
  ones, but it is recommended to not exceed 50.
- **base_num_frames** (`int`, *optional*, defaults to `97`) --
  97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)
- **ar_step** (`int`, *optional*, defaults to `0`) --
  Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous
  inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed
  to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole
  sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous
  inference may improve the instruction following and visual consistent performance.
- **causal_block_size** (`int`, *optional*, defaults to `None`) --
  The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step >
  0)
- **fps** (`int`, *optional*, defaults to `24`) --
  Frame rate of the generated video0`~SkyReelsV2PipelineOutput` or `tuple`If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import (
...     SkyReelsV2DiffusionForcingImageToVideoPipeline,
...     UniPCMultistepScheduler,
...     AutoencoderKLWan,
... )
>>> from diffusers.utils import export_to_video
>>> from PIL import Image

>>> # Load the pipeline
>>> # Available models:
>>> # - Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-DF-14B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-DF-14B-720P-Diffusers
>>> vae = AutoencoderKLWan.from_pretrained(
...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
...     subfolder="vae",
...     torch_dtype=torch.float32,
... )
>>> pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
...     vae=vae,
...     torch_dtype=torch.bfloat16,
... )
>>> flow_shift = 5.0  # 8.0 for T2V, 5.0 for I2V
>>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
>>> image = Image.open("path/to/image.png")

>>> output = pipe(
...     image=image,
...     prompt=prompt,
...     num_inference_steps=50,
...     height=544,
...     width=960,
...     guidance_scale=5.0,  # 6.0 for T2V, 5.0 for I2V
...     num_frames=97,
...     ar_step=0,  # Controls asynchronous inference (0 for synchronous mode)
...     overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos
...     addnoise_condition=20,  # Improves consistency in long video generation
... ).frames[0]
>>> export_to_video(output, "video.mp4", fps=24, quality=8)
```

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~SkyReelsV2PipelineOutput` or `tuple``

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L223)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### generate_timestep_matrix[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.generate_timestep_matrix]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L463)

This function implements the core diffusion forcing algorithm that creates a coordinated denoising schedule
across temporal frames. It supports both synchronous and asynchronous generation modes:

**Synchronous Mode** (ar_step=0, causal_block_size=1):
- All frames are denoised simultaneously at each timestep
- Each frame follows the same denoising trajectory: [1000, 800, 600, ..., 0]
- Simpler but may have less temporal consistency for long videos

**Asynchronous Mode** (ar_step>0, causal_block_size>1):
- Frames are grouped into causal blocks and processed block/chunk-wise
- Each block is denoised in a staggered pattern creating a "denoising wave"
- Earlier blocks are more denoised, later blocks lag behind by ar_step timesteps
- Creates stronger temporal dependencies and better consistency

**Parameters:**

num_latent_frames (int) : Total number of latent frames to generate

step_template (torch.Tensor) : Base timestep schedule (e.g., [1000, 800, 600, ..., 0])

base_num_latent_frames (int) : Maximum frames the model can process in one forward pass

ar_step (int, optional) : Autoregressive step size for temporal lag. 0 = synchronous, >0 = asynchronous. Defaults to 5.

num_pre_ready (int, optional) : Number of frames already denoised (e.g., from prefix in a video2video task). Defaults to 0.

causal_block_size (int, optional) : Number of frames processed as a causal block. Defaults to 1.

shrink_interval_with_mask (bool, optional) : Whether to optimize processing intervals. Defaults to False.

**Returns:**

`tuple containing`

- step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
  [num_iterations, num_latent_frames]
- step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
  num_latent_frames]
- step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
  [num_iterations, num_latent_frames]
- valid_interval (list[tuple]): list of (start, end) intervals for each iteration

## SkyReelsV2DiffusionForcingVideoToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L189)

Pipeline for Video-to-Video (v2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

__call__diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L681[{"name": "video", "val": ": list"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 544"}, {"name": "width", "val": ": int = 960"}, {"name": "num_frames", "val": ": int = 120"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "overlap_history", "val": ": int | None = None"}, {"name": "addnoise_condition", "val": ": float = 0"}, {"name": "base_num_frames", "val": ": int = 97"}, {"name": "ar_step", "val": ": int = 0"}, {"name": "causal_block_size", "val": ": int | None = None"}, {"name": "fps", "val": ": int = 24"}]- **video** (`list[Image.Image]`) --
  The video to guide the video generation.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the video generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, defaults to `544`) --
  The height of the generated video.
- **width** (`int`, defaults to `960`) --
  The width of the generated video.
- **num_frames** (`int`, defaults to `120`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `6.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
  `guidance_scale` is defined as `w` of equation 2. of [Imagen
  Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
  1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
  usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `512`) --
  The maximum sequence length of the prompt.
- **overlap_history** (`int`, *optional*, defaults to `None`) --
  Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes
  short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.
- **addnoise_condition** (`float`, *optional*, defaults to `0`) --
  This is used to help smooth the long video generation by adding some noise to the clean condition. Too
  large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger
  ones, but it is recommended to not exceed 50.
- **base_num_frames** (`int`, *optional*, defaults to `97`) --
  97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)
- **ar_step** (`int`, *optional*, defaults to `0`) --
  Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous
  inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed
  to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole
  sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous
  inference may improve the instruction following and visual consistent performance.
- **causal_block_size** (`int`, *optional*, defaults to `None`) --
  The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step >
  0)
- **fps** (`int`, *optional*, defaults to `24`) --
  Frame rate of the generated video0`~SkyReelsV2PipelineOutput` or `tuple`If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import (
...     SkyReelsV2DiffusionForcingVideoToVideoPipeline,
...     UniPCMultistepScheduler,
...     AutoencoderKLWan,
... )
>>> from diffusers.utils import export_to_video

>>> # Load the pipeline
>>> # Available models:
>>> # - Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-DF-14B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-DF-14B-720P-Diffusers
>>> vae = AutoencoderKLWan.from_pretrained(
...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
...     subfolder="vae",
...     torch_dtype=torch.float32,
... )
>>> pipe = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(
...     "Skywork/SkyReels-V2-DF-14B-720P-Diffusers",
...     vae=vae,
...     torch_dtype=torch.bfloat16,
... )
>>> flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
>>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

>>> output = pipe(
...     prompt=prompt,
...     num_inference_steps=50,
...     height=544,
...     width=960,
...     guidance_scale=6.0,  # 6.0 for T2V, 5.0 for I2V
...     num_frames=97,
...     ar_step=0,  # Controls asynchronous inference (0 for synchronous mode)
...     overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos
...     addnoise_condition=20,  # Improves consistency in long video generation
... ).frames[0]
>>> export_to_video(output, "video.mp4", fps=24, quality=8)
```

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~SkyReelsV2PipelineOutput` or `tuple``

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L279)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype
#### generate_timestep_matrix[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.generate_timestep_matrix]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L501)

This function implements the core diffusion forcing algorithm that creates a coordinated denoising schedule
across temporal frames. It supports both synchronous and asynchronous generation modes:

**Synchronous Mode** (ar_step=0, causal_block_size=1):
- All frames are denoised simultaneously at each timestep
- Each frame follows the same denoising trajectory: [1000, 800, 600, ..., 0]
- Simpler but may have less temporal consistency for long videos

**Asynchronous Mode** (ar_step>0, causal_block_size>1):
- Frames are grouped into causal blocks and processed block/chunk-wise
- Each block is denoised in a staggered pattern creating a "denoising wave"
- Earlier blocks are more denoised, later blocks lag behind by ar_step timesteps
- Creates stronger temporal dependencies and better consistency

**Parameters:**

num_latent_frames (int) : Total number of latent frames to generate

step_template (torch.Tensor) : Base timestep schedule (e.g., [1000, 800, 600, ..., 0])

base_num_latent_frames (int) : Maximum frames the model can process in one forward pass

ar_step (int, optional) : Autoregressive step size for temporal lag. 0 = synchronous, >0 = asynchronous. Defaults to 5.

num_pre_ready (int, optional) : Number of frames already denoised (e.g., from prefix in a video2video task). Defaults to 0.

causal_block_size (int, optional) : Number of frames processed as a causal block. Defaults to 1.

shrink_interval_with_mask (bool, optional) : Whether to optimize processing intervals. Defaults to False.

**Returns:**

`tuple containing`

- step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
  [num_iterations, num_latent_frames]
- step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
  num_latent_frames]
- step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
  [num_iterations, num_latent_frames]
- valid_interval (list[tuple]): list of (start, end) intervals for each iteration

## SkyReelsV2Pipeline[[diffusers.SkyReelsV2Pipeline]]

#### diffusers.SkyReelsV2Pipeline[[diffusers.SkyReelsV2Pipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L107)

Pipeline for Text-to-Video (t2v) generation using SkyReels-V2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.SkyReelsV2Pipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L376[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 544"}, {"name": "width", "val": ": int = 960"}, {"name": "num_frames", "val": ": int = 97"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
  not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `512`) --
  The maximum sequence length for the text encoder.0`~SkyReelsV2PipelineOutput` or `tuple`If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import (
...     SkyReelsV2Pipeline,
...     UniPCMultistepScheduler,
...     AutoencoderKLWan,
... )
>>> from diffusers.utils import export_to_video

>>> # Load the pipeline
>>> # Available models:
>>> # - Skywork/SkyReels-V2-T2V-14B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
>>> vae = AutoencoderKLWan.from_pretrained(
...     "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
...     subfolder="vae",
...     torch_dtype=torch.float32,
... )
>>> pipe = SkyReelsV2Pipeline.from_pretrained(
...     "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
...     vae=vae,
...     torch_dtype=torch.bfloat16,
... )
>>> flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
>>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

>>> output = pipe(
...     prompt=prompt,
...     num_inference_steps=50,
...     height=544,
...     width=960,
...     guidance_scale=6.0,  # 6.0 for T2V, 5.0 for I2V
...     num_frames=97,
... ).frames[0]
>>> export_to_video(output, "video.mp4", fps=24, quality=8)
```

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~SkyReelsV2PipelineOutput` or `tuple``

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.SkyReelsV2Pipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L197)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

## SkyReelsV2ImageToVideoPipeline[[diffusers.SkyReelsV2ImageToVideoPipeline]]

#### diffusers.SkyReelsV2ImageToVideoPipeline[[diffusers.SkyReelsV2ImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L127)

Pipeline for Image-to-Video (i2v) generation using SkyReels-V2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.SkyReelsV2ImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L476[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 544"}, {"name": "width", "val": ": int = 960"}, {"name": "num_frames", "val": ": int = 97"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 5.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "image_embeds", "val": ": torch.Tensor | None = None"}, {"name": "last_image", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, defaults to `544`) --
  The height of the generated video.
- **width** (`int`, defaults to `960`) --
  The width of the generated video.
- **num_frames** (`int`, defaults to `97`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `5.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `negative_prompt` input argument.
- **image_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
  image embeddings are generated from the `image` input argument.
- **last_image** (`torch.Tensor`, *optional*) --
  Optional last image for image-to-video conditioning that anchors the end of the generated video.
- **output_type** (`str`, *optional*, defaults to `"np"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `WanPipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `512`) --
  The maximum sequence length of the prompt.0`~SkyReelsV2PipelineOutput` or `tuple`If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import (
...     SkyReelsV2ImageToVideoPipeline,
...     UniPCMultistepScheduler,
...     AutoencoderKLWan,
... )
>>> from diffusers.utils import export_to_video
>>> from PIL import Image

>>> # Load the pipeline
>>> # Available models:
>>> # - Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-I2V-14B-540P-Diffusers
>>> # - Skywork/SkyReels-V2-I2V-14B-720P-Diffusers
>>> vae = AutoencoderKLWan.from_pretrained(
...     "Skywork/SkyReels-V2-I2V-14B-720P-Diffusers",
...     subfolder="vae",
...     torch_dtype=torch.float32,
... )
>>> pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
...     "Skywork/SkyReels-V2-I2V-14B-720P-Diffusers",
...     vae=vae,
...     torch_dtype=torch.bfloat16,
... )
>>> flow_shift = 5.0  # 8.0 for T2V, 5.0 for I2V
>>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
>>> pipe = pipe.to("cuda")

>>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
>>> image = Image.open("path/to/image.png")

>>> output = pipe(
...     image=image,
...     prompt=prompt,
...     num_inference_steps=50,
...     height=544,
...     width=960,
...     guidance_scale=5.0,  # 6.0 for T2V, 5.0 for I2V
...     num_frames=97,
... ).frames[0]
>>> export_to_video(output, "video.mp4", fps=24, quality=8)
```

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

image_encoder (`CLIPVisionModelWithProjection`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.39.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~SkyReelsV2PipelineOutput` or `tuple``

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.SkyReelsV2ImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L238)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

## SkyReelsV2PipelineOutput[[diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput]]

#### diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput[[diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/skyreels_v2/pipeline_output.py#L9)

Output class for SkyReelsV2 pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Latent Consistency Models
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/latent_consistency_models.md
