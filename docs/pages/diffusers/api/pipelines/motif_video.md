# Motif-Video

[Technical Report](https://arxiv.org/abs/2604.16503)

Motif-Video is a 2B parameter diffusion transformer designed for text-to-video and image-to-video generation. It features a three-stage architecture with 12 dual-stream + 16 single-stream + 8 DDT decoder layers, Shared Cross-Attention for stable text-video alignment under long video sequences, T5Gemma2 text encoder, and rectified flow matching for velocity prediction.

  

## Text-to-Video Generation

Use `MotifVideoPipeline` for text-to-video generation:

```python
import torch
from diffusers import MotifVideoPipeline
from diffusers.utils import export_to_video

pipe = MotifVideoPipeline.from_pretrained(
    "Motif-Technologies/Motif-Video-2B",
    dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1280,
    height=736,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

## Image-to-Video Generation

Use `MotifVideoImage2VideoPipeline` for image-to-video generation:

```python
import torch
from diffusers import MotifVideoImage2VideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = MotifVideoImage2VideoPipeline.from_pretrained(
    "Motif-Technologies/Motif-Video-2B",
    dtype=torch.bfloat16,
)
pipe.to("cuda")

image = load_image("input_image.png")
prompt = "A cinematic scene with vivid colors."
negative_prompt = "worst quality, blurry, jittery, distorted"

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1280,
    height=736,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "i2v_output.mp4", fps=24)
```

### Memory-efficient Inference

For GPUs with less than 30GB VRAM (e.g., RTX 4090), use model CPU offloading:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```python
import torch
from diffusers import MotifVideoPipeline
from diffusers.utils import export_to_video

pipe = MotifVideoPipeline.from_pretrained(
    "Motif-Technologies/Motif-Video-2B",
    dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1280,
    height=736,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

## MotifVideoPipeline[[diffusers.MotifVideoPipeline]]

#### diffusers.MotifVideoPipeline[[diffusers.MotifVideoPipeline]]

```python
diffusers.MotifVideoPipeline(scheduler: SchedulerMixin, vae: AutoencoderKLWan, text_encoder: T5Gemma2Encoder, tokenizer: PreTrainedTokenizerBase, transformer: MotifVideoTransformer3DModel, guider: BaseGuidance, feature_extractor: typing.Optional[transformers.models.siglip.image_processing_pil_siglip.SiglipImageProcessorPil] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_motif_video.py#L148)

**Parameters:**

transformer ([MotifVideoTransformer3DModel](/docs/diffusers/v0.40.0/en/api/models/motif_video_transformer_3d#diffusers.MotifVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents. Should be an instance of a class inheriting from `SchedulerMixin`, such as [DPMSolverMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler). If not provided, uses the scheduler attached to the pretrained model.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5Gemma2Encoder`) : Primary text encoder for encoding text prompts into embeddings.

tokenizer (`PreTrainedTokenizerBase`) : Tokenizer corresponding to the primary text encoder.

guider ([BaseGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.BaseGuidance)) : The guidance method to use. Should be an instance of a class inheriting from `BaseGuidance`, such as [ClassifierFreeGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance), [AdaptiveProjectedGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.AdaptiveProjectedGuidance), or [SkipLayerGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.SkipLayerGuidance). If not provided, defaults to `ClassifierFreeGuidance`.

Pipeline for text-to-video generation using Motif-Video.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.MotifVideoPipeline.__call__]]

```python
__call__(prompt: typing.Union[str, typing.List[str], NoneType] = None, negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, height: int = 736, width: int = 1280, num_frames: int = 121, num_inference_steps: int = 50, timesteps: typing.Optional[typing.List[int]] = None, num_videos_per_prompt: typing.Optional[int] = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: typing.Optional[str] = 'pil', return_dict: bool = True, attention_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, callback_on_step_end: typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: typing.List[str] = ['latents'], max_sequence_length: int = 512, vae_batch_size: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_motif_video.py#L492)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance.

height (`int`, defaults to `736`) : The height in pixels of the generated video.

width (`int`, defaults to `1280`) : The width in pixels of the generated video.

num_frames (`int`, defaults to `121`) : The number of video frames to generate.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `List[torch.Generator]`, *optional*) : PyTorch Generator object(s) for deterministic generation.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated video. Choose between `"pil"`, `"np"`, or `"latent"`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [~MotifVideoPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/motif_video#diffusers.MotifVideoPipelineOutput) instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : Arguments passed to the attention processor.

callback_on_step_end (`Callable`, *optional*) : A function or subclass of `PipelineCallback` or `MultiPipelineCallbacks` called at the end of each denoising step.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function.

max_sequence_length (`int`, defaults to `512`) : Maximum sequence length for the tokenizer.

vae_batch_size (`int`, *optional*) : Batch size for VAE decoding. If provided and latents batch size is larger, VAE decoding will be done in chunks.

**Returns:** [~MotifVideoPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/motif_video#diffusers.MotifVideoPipelineOutput) or `tuple`

If `return_dict` is `True`, [~MotifVideoPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/motif_video#diffusers.MotifVideoPipelineOutput) is returned, otherwise a `tuple` is returned
where the first element is a list of generated video frames.

The call function to the pipeline for text-to-video generation.

Examples:
```python
>>> import torch
>>> from diffusers import MotifVideoPipeline
>>> from diffusers.utils import export_to_video

>>> # Load the Motif-Video pipeline
>>> motif_video_model_id = "Motif-Technologies/Motif-Video-2B"
>>> pipe = MotifVideoPipeline.from_pretrained(motif_video_model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> video = pipe(
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=1280,
...     height=736,
...     num_frames=121,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=24)
```

#### encode_prompt[[diffusers.MotifVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: typing.Union[str, typing.List[str]], negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 512, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_motif_video.py#L247)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to be encoded.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos to generate per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

max_sequence_length (`int`, defaults to 512) : Maximum sequence length for the tokenizer.

device (`torch.device`, *optional*) : Device to place tensors on.

dtype (`torch.dtype`, *optional*) : Data type for tensors.

**Returns:** `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

A tuple containing:
- `prompt_embeds`: The text embeddings for the positive prompt
- `negative_prompt_embeds`: The text embeddings for the negative prompt (None if not using guidance)
- `prompt_attention_mask`: The attention mask for the positive prompt
- `negative_prompt_attention_mask`: The attention mask for the negative prompt (None if not using
  guidance)

Encodes the prompt into text encoder hidden states.

## MotifVideoImage2VideoPipeline[[diffusers.MotifVideoImage2VideoPipeline]]

#### diffusers.MotifVideoImage2VideoPipeline[[diffusers.MotifVideoImage2VideoPipeline]]

```python
diffusers.MotifVideoImage2VideoPipeline(scheduler: SchedulerMixin, vae: AutoencoderKLWan, text_encoder: T5Gemma2Encoder, tokenizer: PreTrainedTokenizerBase, transformer: MotifVideoTransformer3DModel, guider: BaseGuidance, feature_extractor: SiglipImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_motif_video_image2video.py#L157)

**Parameters:**

transformer ([MotifVideoTransformer3DModel](/docs/diffusers/v0.40.0/en/api/models/motif_video_transformer_3d#diffusers.MotifVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents. Should be an instance of a class inheriting from `SchedulerMixin`, such as [DPMSolverMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler). If not provided, uses the scheduler attached to the pretrained model.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5Gemma2Encoder`) : Primary text encoder for encoding text prompts into embeddings.

tokenizer (`PreTrainedTokenizerBase`) : Tokenizer corresponding to the primary text encoder.

feature_extractor (`SiglipImageProcessor`) : Image processor for the SigLIP vision encoder.

guider ([BaseGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.BaseGuidance)) : The guidance method to use. Should be an instance of a class inheriting from `BaseGuidance`, such as [ClassifierFreeGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance), [AdaptiveProjectedGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.AdaptiveProjectedGuidance), or [SkipLayerGuidance](/docs/diffusers/v0.40.0/en/api/modular_diffusers/guiders#diffusers.SkipLayerGuidance). If not provided, defaults to `ClassifierFreeGuidance`.

Pipeline for image-to-video generation using Motif-Video with first frame conditioning.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.MotifVideoImage2VideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: typing.Union[str, typing.List[str]], negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, height: int = 736, width: int = 1280, num_frames: int = 121, num_inference_steps: int = 50, timesteps: typing.Optional[typing.List[int]] = None, num_videos_per_prompt: typing.Optional[int] = 1, generator: typing.Union[torch.Generator, typing.List[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: typing.Optional[str] = 'pil', return_dict: bool = True, attention_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, callback_on_step_end: typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: typing.List[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_motif_video_image2video.py#L620)

**Parameters:**

image (`PipelineImageInput`) : The input image to use as the first frame for video generation.

prompt (`str` or `List[str]`) : The prompt or prompts to guide the video generation.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the video generation.

height (`int`, defaults to `736`) : The height in pixels of the generated video.

width (`int`, defaults to `1280`) : The width in pixels of the generated video.

num_frames (`int`, defaults to `121`) : The number of video frames to generate.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `List[torch.Generator]`, *optional*) : PyTorch Generator object(s) for deterministic generation.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated video.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [~MotifVideoPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/motif_video#diffusers.MotifVideoPipelineOutput) instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : Arguments passed to the attention processor.

callback_on_step_end (`Callable`, *optional*) : A function or subclass of `PipelineCallback` called at the end of each denoising step.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function.

max_sequence_length (`int`, defaults to `512`) : Maximum sequence length for the tokenizer.

**Returns:** [~MotifVideoPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/motif_video#diffusers.MotifVideoPipelineOutput) or `tuple`

If `return_dict` is `True`, [~MotifVideoPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/motif_video#diffusers.MotifVideoPipelineOutput) is returned, otherwise a `tuple` is returned
where the first element is a list of generated video frames.

The call function to the pipeline for image-to-video generation.

Examples:
```python
>>> import torch
>>> from PIL import Image
>>> from diffusers import MotifVideoImage2VideoPipeline
>>> from diffusers.utils import export_to_video, load_image

>>> # Load the Motif-Video image-to-video pipeline
>>> motif_video_model_id = "Motif-Technologies/Motif-Video-2B"
>>> pipe = MotifVideoImage2VideoPipeline.from_pretrained(motif_video_model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> # Load an image
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.png"
... )

>>> prompt = "An astronaut is walking on the moon surface, kicking up dust with each step"
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> video = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=1280,
...     height=736,
...     num_frames=121,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=24)
```

#### encode_prompt[[diffusers.MotifVideoImage2VideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: typing.Union[str, typing.List[str]], negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 512, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_motif_video_image2video.py#L259)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to be encoded.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos to generate per prompt.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

max_sequence_length (`int`, defaults to 512) : Maximum sequence length for the tokenizer.

device (`torch.device`, *optional*) : Device to place tensors on.

dtype (`torch.dtype`, *optional*) : Data type for tensors.

**Returns:** `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

A tuple containing:
- `prompt_embeds`: The text embeddings for the positive prompt
- `negative_prompt_embeds`: The text embeddings for the negative prompt (None if not using guidance)
- `prompt_attention_mask`: The attention mask for the positive prompt
- `negative_prompt_attention_mask`: The attention mask for the negative prompt (None if not using
  guidance)

Encodes the prompt into text encoder hidden states.

## MotifVideoPipelineOutput[[diffusers.MotifVideoPipelineOutput]]

#### diffusers.MotifVideoPipelineOutput[[diffusers.MotifVideoPipelineOutput]]

```python
diffusers.MotifVideoPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/motif_video/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for Motif-Video pipelines.

### Glm Image
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/glm_image.md

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

# GLM-Image

## Overview

GLM-Image is an image generation model adopts a hybrid autoregressive + diffusion decoder architecture, effectively pushing the upper bound of visual fidelity and fine-grained details. In general image generation quality, it aligns with industry-standard LDM-based approaches, while demonstrating significant advantages in knowledge-intensive image generation scenarios.

Model architecture: a hybrid autoregressive + diffusion decoder design、

+ Autoregressive generator: a 9B-parameter model initialized from [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414), with an expanded vocabulary to incorporate visual tokens. The model first generates a compact encoding of approximately 256 tokens, then expands to 1K–4K tokens, corresponding to 1K–2K high-resolution image outputs. You can check AR model in class `GlmImageForConditionalGeneration` of `transformers` library.
+ Diffusion Decoder: a 7B-parameter decoder based on a single-stream DiT architecture for latent-space image decoding. It is equipped with a Glyph Encoder text module, significantly improving accurate text rendering within images.

Post-training with decoupled reinforcement learning: the model introduces a fine-grained, modular feedback strategy using the GRPO algorithm, substantially enhancing both semantic understanding and visual detail quality.

+ Autoregressive module: provides low-frequency feedback signals focused on aesthetics and semantic alignment, improving instruction following and artistic expressiveness.
+ Decoder module: delivers high-frequency feedback targeting detail fidelity and text accuracy, resulting in highly realistic textures, lighting, and color reproduction, as well as more precise text rendering.

GLM-Image supports both text-to-image and image-to-image generation within a single model

+ Text-to-image: generates high-detail images from textual descriptions, with particularly strong performance in information-dense scenarios.
+ Image-to-image: supports a wide range of tasks, including image editing, style transfer, multi-subject consistency, and identity-preserving generation for people and objects.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The codebase can be found [here](https://huggingface.co/zai-org/GLM-Image).

## Usage examples

### Text to Image Generation

```python
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline

pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image",dtype=torch.bfloat16,device_map="cuda")
prompt = "A beautifully designed modern food magazine style dessert recipe illustration, themed around a raspberry mousse cake. The overall layout is clean and bright, divided into four main areas: the top left features a bold black title 'Raspberry Mousse Cake Recipe Guide', with a soft-lit close-up photo of the finished cake on the right, showcasing a light pink cake adorned with fresh raspberries and mint leaves; the bottom left contains an ingredient list section, titled 'Ingredients' in a simple font, listing 'Flour 150g', 'Eggs 3', 'Sugar 120g', 'Raspberry puree 200g', 'Gelatin sheets 10g', 'Whipping cream 300ml', and 'Fresh raspberries', each accompanied by minimalist line icons (like a flour bag, eggs, sugar jar, etc.); the bottom right displays four equally sized step boxes, each containing high-definition macro photos and corresponding instructions, arranged from top to bottom as follows: Step 1 shows a whisk whipping white foam (with the instruction 'Whip egg whites to stiff peaks'), Step 2 shows a red-and-white mixture being folded with a spatula (with the instruction 'Gently fold in the puree and batter'), Step 3 shows pink liquid being poured into a round mold (with the instruction 'Pour into mold and chill for 4 hours'), Step 4 shows the finished cake decorated with raspberries and mint leaves (with the instruction 'Decorate with raspberries and mint'); a light brown information bar runs along the bottom edge, with icons on the left representing 'Preparation time: 30 minutes', 'Cooking time: 20 minutes', and 'Servings: 8'. The overall color scheme is dominated by creamy white and light pink, with a subtle paper texture in the background, featuring compact and orderly text and image layout with clear information hierarchy."
image = pipe(
    prompt=prompt,
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=30,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("output_t2i.png")
```

### Image to Image Generation

```python
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image",dtype=torch.bfloat16,device_map="cuda")
image_path = "cond.jpg" 
prompt = "Replace the background of the snow forest with an underground station featuring an automatic escalator."
image = Image.open(image_path).convert("RGB")
image = pipe(
    prompt=prompt,
    image=[image], # can input multiple images for multi-image-to-image generation such as [image, image1]
    height=33 * 32,
    width=32 * 32,
    num_inference_steps=30,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("output_i2i.png")
```

+ Since the AR model used in GLM-Image is configured with `do_sample=True` and a temperature of `0.95` by default, the generated images can vary significantly across runs. We do not recommend setting do_sample=False, as this may lead to incorrect or degenerate outputs from the AR model.

## GlmImagePipeline[[diffusers.GlmImagePipeline]]

#### diffusers.GlmImagePipeline[[diffusers.GlmImagePipeline]]

```python
diffusers.GlmImagePipeline(tokenizer: ByT5Tokenizer, processor: GlmImageProcessor, text_encoder: T5EncoderModel, vision_language_encoder: GlmImageForConditionalGeneration, vae: AutoencoderKL, transformer: GlmImageTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L161)

**Parameters:**

tokenizer (`PreTrainedTokenizer`) : Tokenizer for the text encoder.

processor (`AutoProcessor`) : Processor for the AR model to handle chat templates and tokenization.

text_encoder (`T5EncoderModel`) : Frozen text-encoder for glyph embeddings.

vision_language_encoder (`GlmImageForConditionalGeneration`) : The AR model that generates image tokens from text prompts.

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

transformer ([GlmImageTransformer2DModel](/docs/diffusers/v0.40.0/en/api/models/glm_image_transformer2d#diffusers.GlmImageTransformer2DModel)) : A text conditioned transformer to denoise the encoded image latents (DiT).

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

Pipeline for text-to-image generation using GLM-Image.

This pipeline integrates both the AR (autoregressive) model for token generation and the DiT (diffusion
transformer) model for image decoding.

#### __call__[[diffusers.GlmImagePipeline.__call__]]

```python
__call__(prompt: str | list[str] | None = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, timesteps: list[int] | None = None, sigmas: list[float] | None = None, guidance_scale: float = 1.5, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prior_token_ids: typing.Optional[torch.Tensor] = None, prior_token_image_ids: list[torch.Tensor] | None = None, source_image_grid_thw: list[torch.Tensor] | None = None, crops_coords_top_left: tuple = (0, 0), output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int, dict], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 2048)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L719)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. Must contain shape info in the format 'H W' where H and W are token dimensions (d32). Example: "A beautiful sunset36 24" generates a 1152x768 image.

image : Optional condition images for image-to-image generation.

height (`int`, *optional*) : The height in pixels. If not provided, derived from prompt shape info.

width (`int`, *optional*) : The width in pixels. If not provided, derived from prompt shape info.

num_inference_steps (`int`, *optional*, defaults to `50`) : The number of denoising steps for DiT.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process. If not defined, the scheduler's default schedule for `num_inference_steps` is used.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process. If not defined, the scheduler's default schedule is used.

guidance_scale (`float`, *optional*, defaults to `1.5`) : Guidance scale for classifier-free guidance.

num_images_per_prompt (`int`, *optional*, defaults to `1`) : The number of images to generate per prompt.

generator (`torch.Generator`, *optional*) : Random generator for reproducibility.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents to be used as inputs for image generation.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, embeddings are generated from `prompt`.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Used when classifier-free guidance is enabled.

prior_token_ids (`torch.Tensor`, *optional*) : Pre-generated prior token ids from `generate_prior_tokens`. If supplied, prior generation is skipped.

prior_token_image_ids (`list[torch.Tensor]`, *optional*) : Image token ids associated with `prior_token_ids`.

source_image_grid_thw (`list[torch.Tensor]`, *optional*) : Per-sample THW grid information for the source image tokens.

crops_coords_top_left (`tuple[int, int]`, *optional*, defaults to `(0, 0)`) : The top-left coordinates of the crop used for conditioning embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : Output format: "pil", "np", or "latent".

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `GlmImagePipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function called at the end of each denoising step.

callback_on_step_end_tensor_inputs (`list[str]`, *optional*) : Tensor inputs passed to `callback_on_step_end`.

max_sequence_length (`int`, *optional*, defaults to `2048`) : Maximum sequence length for the text encoder.

**Returns:** `GlmImagePipelineOutput` or `tuple`

Generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import GlmImagePipeline

>>> pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A photo of an astronaut riding a horse on mars"
>>> image = pipe(prompt).images[0]
>>> image.save("output.png")
```

#### encode_prompt[[diffusers.GlmImagePipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], do_classifier_free_guidance: bool = True, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, max_sequence_length: int = 2048)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L545)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

max_sequence_length (`int`, defaults to `2048`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

Encodes the prompt into text encoder hidden states.

#### generate_prior_tokens[[diffusers.GlmImagePipeline.generate_prior_tokens]]

```python
generate_prior_tokens(prompt: str | list[str], height: int, width: int, image: list[list[PIL.Image.Image]] | None = None, device: typing.Optional[torch.device] = None, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L321)

**Parameters:**

prompt : Single prompt or list of prompts

height : Target image height

width : Target image width

image : Normalized image input as List[List[PIL.Image]]. Should be pre-validated using _validate_and_normalize_images() before calling this method.

device : Target device

generator : Random generator for reproducibility

**Returns:** `Tuple of`

- prior_token_ids: Tensor of shape (batch_size, num_tokens) with upsampled prior tokens
- prior_token_image_ids_per_sample: List of tensors, one per sample. Each tensor contains
  the upsampled prior token ids for all condition images in that sample. None for t2i.
- source_image_grid_thw_per_sample: List of tensors, one per sample. Each tensor has shape
  (num_condition_images, 3) with upsampled grid info. None for t2i.

Generate prior tokens for the DiT model using the AR model.

#### get_glyph_texts[[diffusers.GlmImagePipeline.get_glyph_texts]]

```python
get_glyph_texts(prompt)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L476)

Extract glyph texts from prompt(s). Returns a list of lists for batch processing.

## GlmImagePipelineOutput[[diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput]]

#### diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput[[diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput]]

```python
diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/glm_image/pipeline_output.py#L10)

**Parameters:**

images (`List[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for CogView3 pipelines.

### Cosmos3
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/cosmos3.md

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

# Cosmos 3

NVIDIA Cosmos 3 is a unified world foundation model (WFM) for Physical AI — a single omni-model that combines world generation, physical reasoning, and action generation. It replaces the separate Predict, Reason, and Transfer models from earlier Cosmos releases: whether you're building for robotics, autonomous vehicles, or smart spaces, Cosmos 3 gives you one foundation to simulate and understand the physical world.

What's shipping with this release:

- Models on the Hugging Face Hub with model cards and licensing
- Cosmos 3 Diffusers integration for generation pipelines (this page)
- Post-training scripts for fine-tuning Cosmos 3 on your own data
- Open synthetic data generation (SDG) datasets for Physical AI

## What's new in Cosmos 3

The biggest change from previous Cosmos releases is that Cosmos 3 is an *omni-model*, built on a Mixture-of-Transformers (MoT) architecture. Previously, developers worked with separate models for world generation (Predict), controlled generation (Transfer), scene understanding (Reason), and action-policy generation. Cosmos 3 unifies all of these in one model that reasons and generates across modalities in a single forward pass.

From one model you can:

- Generate physically plausible video worlds from text, images, or action inputs (image-to-video, text-to-video, action-conditioned video generation).
- Reason about physical properties like motion, causality, and spatial relationships.
- Predict future video and action sequences from the current state.
- Transfer scenes across viewpoints and conditions with structural control *(coming soon)*.

Under the hood, a single `Cosmos3OmniTransformer` runs a Qwen-style language model in parallel with a diffusion generation pathway: text tokens flow through a causal "understanding" stream while video and sound latents flow through a bi-directionally-attended "generation" stream, joined by a 3D multimodal RoPE. See the [Cosmos World Foundation Model Platform paper](https://huggingface.co/papers/2501.03575) for the architectural background.

## Available checkpoints

Two checkpoints are released on the Hub — [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) (smaller, faster) and [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (larger, higher quality). The same pipeline class supports text-to-image, text-to-video, image-to-video, and (with a sound-capable checkpoint) text+image-to-video-with-sound — pick a repo and use the per-model tab in each workflow below.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Prompt upsampling

Cosmos 3 was trained on long, highly descriptive captions. For optimal quality, short text prompts should be **upsampled into a specific JSON structure** before they are passed to the pipeline. The upsampler lives in the [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) package.

Start from a short, plain-text prompt and save it to `assets/prompt.txt`. For the text-to-video example below, the original prompt is *"A robotic arm is cleaning a plate in a kitchen"*:

```bash
mkdir -p assets
echo "A robotic arm is cleaning a plate in a kitchen" > assets/prompt.txt
```

Then install the framework and run the upsampler. The example below upsamples for text-to-video using Opus-4.6:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos-framework
pip install -e packages/cosmos-framework

export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-opus-4-6"
export PROMPT_UPSAMPLER_API_TOKEN="REDACTED"

python -m cosmos_framework.inference.prompt_upsampling \
    --input assets/prompt.txt \
    --output assets/example_t2v_prompt.json \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "16,9"
```

Switch `--mode` to match the workflow you are targeting (`text2image`, `text2video`, `image2video`). The command writes the upsampled prompt(s) to the `--output` file as a JSON array (one object per non-empty line in `--input`); pass a `.jsonl` path instead to get one JSON object per line. For `image2video`, you must also supply the conditioning image via `--image-url` (a URL or local path) or `--image-list` (one image per prompt).

A pre-upsampled positive prompt (`assets/example_t2v_prompt.json`) and negative prompt (`assets/negative_prompt.json`) are provided for convenience, and are used by the generation examples below. The examples load these JSON files and pass them to the pipeline as JSON strings via `json.dumps(...)`.

## Text-to-video

Multi-frame generation conditioned on text alone. Pick `num_frames` based on the target duration — the default `num_frames=189` produces ≈ 7.9 s at 24 FPS. The prompt and negative prompt are read from the JSON-upsampled files described in [Prompt upsampling](#prompt-upsampling).

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
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
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
