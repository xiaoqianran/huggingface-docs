# Framepack

  

[Packing Input Frame Context in Next-Frame Prediction Models for Video Generation](https://huggingface.co/papers/2504.12626) by Lvmin Zhang and Maneesh Agrawala.

*We present a neural network structure, FramePack, to train next-frame (or next-frame-section) prediction models for video generation. The FramePack compresses input frames to make the transformer context length a fixed number regardless of the video length. As a result, we are able to process a large number of frames using video diffusion with computation bottleneck similar to image diffusion. This also makes the training video batch sizes significantly higher (batch sizes become comparable to image diffusion training). We also propose an anti-drifting sampling method that generates frames in inverted temporal order with early-established endpoints to avoid exposure bias (error accumulation over iterations). Finally, we show that existing video diffusion models can be finetuned with FramePack, and their visual quality may be improved because the next-frame prediction supports more balanced diffusion schedulers with less extreme flow shift timesteps.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Available models

| Model name | Description |
|:---|:---|
- [`lllyasviel/FramePackI2V_HY`](https://huggingface.co/lllyasviel/FramePackI2V_HY) | Trained with the "inverted anti-drifting" strategy as described in the paper. Inference requires setting `sampling_type="inverted_anti_drifting"` when running the pipeline. |
- [`lllyasviel/FramePack_F1_I2V_HY_20250503`](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503) | Trained with a novel anti-drifting strategy but inference is performed in "vanilla" strategy as described in the paper. Inference requires setting `sampling_type="vanilla"` when running the pipeline. |

## Usage

Refer to the pipeline documentation for basic usage examples. The following section contains examples of offloading, different sampling methods, quantization, and more.

### First and last frame to video

The following example shows how to use Framepack with start and end image controls, using the inverted anti-drifiting sampling model.

```python
import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePackI2V_HY", dtype=torch.bfloat16
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", dtype=torch.float16
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    dtype=torch.float16,
)

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
first_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
)
last_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
)
output = pipe(
    image=first_image,
    last_image=last_image,
    prompt=prompt,
    height=512,
    width=512,
    num_frames=91,
    num_inference_steps=30,
    guidance_scale=9.0,
    generator=torch.Generator().manual_seed(0),
    sampling_type="inverted_anti_drifting",
).frames[0]
export_to_video(output, "output.mp4", fps=30)
```

### Vanilla sampling

The following example shows how to use Framepack with the F1 model trained with vanilla sampling but new regulation approach for anti-drifting.

```python
import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePack_F1_I2V_HY_20250503", dtype=torch.bfloat16
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", dtype=torch.float16
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    dtype=torch.float16,
)

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)
output = pipe(
    image=image,
    prompt="A penguin dancing in the snow",
    height=832,
    width=480,
    num_frames=91,
    num_inference_steps=30,
    guidance_scale=9.0,
    generator=torch.Generator().manual_seed(0),
    sampling_type="vanilla",
).frames[0]
export_to_video(output, "output.mp4", fps=30)
```

### Group offloading

Group offloading ([apply_group_offloading()](/docs/diffusers/v0.40.0/en/api/utilities#diffusers.hooks.apply_group_offloading)) provides aggressive memory optimizations for offloading internal parts of any model to the CPU, with possibly no additional overhead to generation time. If you have very low VRAM available, this approach may be suitable for you depending on the amount of CPU RAM available.

```python
import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePack_F1_I2V_HY_20250503", dtype=torch.bfloat16
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl", subfolder="image_encoder", dtype=torch.float16
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    dtype=torch.float16,
)

# Enable group offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
list(map(
    lambda x: apply_group_offloading(x, onload_device, offload_device, offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True),
    [pipe.text_encoder, pipe.text_encoder_2, pipe.transformer]
))
pipe.image_encoder.to(onload_device)
pipe.vae.to(onload_device)
pipe.vae.enable_tiling()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)
output = pipe(
    image=image,
    prompt="A penguin dancing in the snow",
    height=832,
    width=480,
    num_frames=91,
    num_inference_steps=30,
    guidance_scale=9.0,
    generator=torch.Generator().manual_seed(0),
    sampling_type="vanilla",
).frames[0]
print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
export_to_video(output, "output.mp4", fps=30)
```

## HunyuanVideoFramepackPipeline[[diffusers.HunyuanVideoFramepackPipeline]]

#### diffusers.HunyuanVideoFramepackPipeline[[diffusers.HunyuanVideoFramepackPipeline]]

```python
diffusers.HunyuanVideoFramepackPipeline(text_encoder: LlamaModel, tokenizer: LlamaTokenizer, transformer: HunyuanVideoFramepackTransformer3DModel, vae: AutoencoderKLHunyuanVideo, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder_2: CLIPTextModel, tokenizer_2: CLIPTokenizer, image_encoder: SiglipVisionModel, feature_extractor: SiglipImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py#L243)

**Parameters:**

text_encoder (`LlamaModel`) : [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

tokenizer (`LlamaTokenizer`) : Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

transformer ([HunyuanVideoTransformer3DModel](/docs/diffusers/v0.40.0/en/api/models/hunyuan_video_transformer_3d#diffusers.HunyuanVideoTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLHunyuanVideo](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_hunyuan_video#diffusers.AutoencoderKLHunyuanVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder_2 (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

tokenizer_2 (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

Pipeline for text-to-video generation using HunyuanVideo.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideoFramepackPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], last_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, prompt: str | list[str] = None, prompt_2: str | list[str] = None, negative_prompt: str | list[str] = None, negative_prompt_2: str | list[str] = None, height: int = 720, width: int = 1280, num_frames: int = 129, latent_window_size: int = 9, num_inference_steps: int = 50, sigmas: list = None, true_cfg_scale: float = 1.0, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, image_latents: typing.Optional[torch.Tensor] = None, last_image_latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], prompt_template: dict = {'template': '<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>', 'crop_start': 95}, max_sequence_length: int = 256, sampling_type: FramepackSamplingType = <FramepackSamplingType.INVERTED_ANTI_DRIFTING: 'inverted_anti_drifting'>)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py#L586)

**Parameters:**

image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) : The image to be used as the starting point for the video generation.

last_image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`, *optional*) : The optional last image to be used as the ending point for the video generation. This is useful for generating transitions between two images.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

negative_prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.

height (`int`, defaults to `720`) : The height in pixels of the generated image.

width (`int`, defaults to `1280`) : The width in pixels of the generated image.

num_frames (`int`, defaults to `129`) : The number of frames in the generated video.

latent_window_size (`int`, defaults to `9`) : Number of latent frames produced per Framepack sampling window.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

true_cfg_scale (`float`, *optional*, defaults to 1.0) : When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.

guidance_scale (`float`, defaults to `6.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Note that the only available HunyuanVideo model is CFG-distilled, which means that traditional guidance between unconditional and conditional latent is not applied.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

image_latents (`torch.Tensor`, *optional*) : Pre-encoded image latents. If not provided, the image will be encoded using the VAE.

last_image_latents (`torch.Tensor`, *optional*) : Pre-encoded last image latents. If not provided, the last image will be encoded using the VAE.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`. Required when `prompt_embeds` is passed directly.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`. Required when `negative_prompt_embeds` is passed directly.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `HunyuanVideoFramepackPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

prompt_template (`dict`, *optional*) : Template used to format the prompt before encoding. Defaults to the model's default template.

max_sequence_length (`int`, *optional*, defaults to 256) : Maximum sequence length to use for the prompt encoder.

sampling_type (`FramepackSamplingType`, *optional*) : The Framepack sampling strategy to use when iterating over latent windows.

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** `~HunyuanVideoFramepackPipelineOutput` or `tuple`

If `return_dict` is `True`, `HunyuanVideoFramepackPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images and the second element is a list
of `bool`s indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw)
content.

The call function to the pipeline for generation.

Examples:
##### Image-to-Video

```python
>>> import torch
>>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
>>> from diffusers.utils import export_to_video, load_image
>>> from transformers import SiglipImageProcessor, SiglipVisionModel

>>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
... )
>>> feature_extractor = SiglipImageProcessor.from_pretrained(
...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
... )
>>> image_encoder = SiglipVisionModel.from_pretrained(
...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
... )
>>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
...     "hunyuanvideo-community/HunyuanVideo",
...     transformer=transformer,
...     feature_extractor=feature_extractor,
...     image_encoder=image_encoder,
...     torch_dtype=torch.float16,
... )
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
... )
>>> output = pipe(
...     image=image,
...     prompt="A penguin dancing in the snow",
...     height=832,
...     width=480,
...     num_frames=91,
...     num_inference_steps=30,
...     guidance_scale=9.0,
...     generator=torch.Generator().manual_seed(0),
...     sampling_type="inverted_anti_drifting",
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=30)
```

##### First and Last Image-to-Video

```python
>>> import torch
>>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
>>> from diffusers.utils import export_to_video, load_image
>>> from transformers import SiglipImageProcessor, SiglipVisionModel

>>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
... )
>>> feature_extractor = SiglipImageProcessor.from_pretrained(
...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
... )
>>> image_encoder = SiglipVisionModel.from_pretrained(
...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
... )
>>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
...     "hunyuanvideo-community/HunyuanVideo",
...     transformer=transformer,
...     feature_extractor=feature_extractor,
...     image_encoder=image_encoder,
...     torch_dtype=torch.float16,
... )
>>> pipe.to("cuda")

>>> prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
>>> first_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
... )
>>> last_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
... )
>>> output = pipe(
...     image=first_image,
...     last_image=last_image,
...     prompt=prompt,
...     height=512,
...     width=512,
...     num_frames=91,
...     num_inference_steps=30,
...     guidance_scale=9.0,
...     generator=torch.Generator().manual_seed(0),
...     sampling_type="inverted_anti_drifting",
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=30)
```

## HunyuanVideoPipelineOutput[[diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput]]

#### diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput[[diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput]]

```python
diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/hunyuan_video/pipeline_output.py#L11)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for HunyuanVideo pipelines.

### Sana Sprint
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/sana_sprint.md

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

# SANA-Sprint

  

[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](https://huggingface.co/papers/2503.09641) from NVIDIA, MIT HAN Lab, and Hugging Face by Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Enze Xie, Song Han

The abstract from the paper is:

*This paper presents SANA-Sprint, an efficient diffusion model for ultra-fast text-to-image (T2I) generation. SANA-Sprint is built on a pre-trained foundation model and augmented with hybrid distillation, dramatically reducing inference steps from 20 to 1-4. We introduce three key innovations: (1) We propose a training-free approach that transforms a pre-trained flow-matching model for continuous-time consistency distillation (sCM), eliminating costly training from scratch and achieving high training efficiency. Our hybrid distillation strategy combines sCM with latent adversarial distillation (LADD): sCM ensures alignment with the teacher model, while LADD enhances single-step generation fidelity. (2) SANA-Sprint is a unified step-adaptive model that achieves high-quality generation in 1-4 steps, eliminating step-specific training and improving efficiency. (3) We integrate ControlNet with SANA-Sprint for real-time interactive image generation, enabling instant visual feedback for user interaction. SANA-Sprint establishes a new Pareto frontier in speed-quality tradeoffs, achieving state-of-the-art performance with 7.59 FID and 0.74 GenEval in only 1 step — outperforming FLUX-schnell (7.94 FID / 0.71 GenEval) while being 10× faster (0.1s vs 1.1s on H100). It also achieves 0.1s (T2I) and 0.25s (ControlNet) latency for 1024×1024 images on H100, and 0.31s (T2I) on an RTX 4090, showcasing its exceptional efficiency and potential for AI-powered consumer applications (AIPC). Code and pre-trained models will be open-sourced.*

This pipeline was contributed by [lawrence-cj](https://github.com/lawrence-cj), [shuchen Xue](https://github.com/scxue) and [Enze Xie](https://github.com/xieenze). The original codebase can be found [here](https://github.com/NVlabs/Sana). The original weights can be found under [hf.co/Efficient-Large-Model](https://huggingface.co/Efficient-Large-Model/).

Available models:

|                                                                    Model                                                                    | Recommended dtype |
|:-------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
| [`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) | `torch.bfloat16`  |
| [`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) | `torch.bfloat16`  |

Refer to [this](https://huggingface.co/collections/Efficient-Large-Model/sana-sprint-67d6810d65235085b3b17c76) collection for more information.

Note: The recommended dtype mentioned is for the transformer weights. The text encoder must stay in `torch.bfloat16` and VAE weights must stay in `torch.bfloat16` or `torch.float32` for the model to work correctly. Please refer to the inference example below to see how to load the model with the recommended dtype. 

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [SanaSprintPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/sana_sprint#diffusers.SanaSprintPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaTransformer2DModel, SanaSprintPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    dtype=torch.bfloat16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaTransformer2DModel.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    dtype=torch.bfloat16,
)

pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    dtype=torch.bfloat16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("sana.png")
```

## Setting `max_timesteps`

Users can tweak the `max_timesteps` value for experimenting with the visual quality of the generated outputs. The default `max_timesteps` value was obtained with an inference-time search process. For more details about it, check out the paper.

## Image to Image 

The [SanaSprintImg2ImgPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/sana_sprint#diffusers.SanaSprintImg2ImgPipeline) is a pipeline for image-to-image generation. It takes an input image and a prompt, and generates a new image based on the input image and the prompt.

```py
import torch
from diffusers import SanaSprintImg2ImgPipeline
from diffusers.utils.loading_utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)

pipe = SanaSprintImg2ImgPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", 
    dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    prompt="a cute pink bear", 
    image=image, 
    strength=0.5, 
    height=832, 
    width=480
).images[0]
image.save("output.png")
```

## SanaSprintPipeline[[diffusers.SanaSprintPipeline]]

#### diffusers.SanaSprintPipeline[[diffusers.SanaSprintPipeline]]

```python
diffusers.SanaSprintPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderDC, transformer: SanaTransformer2DModel, scheduler: DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py#L140)

Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641).

#### __call__[[diffusers.SanaSprintPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, num_inference_steps: int = 2, timesteps: list = None, max_timesteps: float = 1.5708, intermediate_timesteps: float = 1.3, guidance_scale: float = 4.5, num_images_per_prompt: int | None = 1, height: int = 1024, width: int = 1024, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py#L561)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

max_timesteps (`float`, *optional*, defaults to 1.57080) : The maximum timestep value used in the SCM scheduler.

intermediate_timesteps (`float`, *optional*, defaults to 1.3) : The intermediate timestep value used in SCM scheduler (only used when num_inference_steps=2).

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 4.5) : Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages a model to generate images more aligned with `prompt` at the expense of lower image quality.  Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to the [paper](https://huggingface.co/papers/2210.03142) to learn more.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

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
>>> from diffusers import SanaSprintPipeline

>>> pipe = SanaSprintPipeline.from_pretrained(
...     "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = pipe(prompt="a tiny astronaut hatching from an egg on the moon")[0]
>>> image[0].save("output.png")
```

#### encode_prompt[[diffusers.SanaSprintPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], num_images_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py#L232)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded 

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

Encodes the prompt into text encoder hidden states.

## SanaSprintImg2ImgPipeline[[diffusers.SanaSprintImg2ImgPipeline]]

#### diffusers.SanaSprintImg2ImgPipeline[[diffusers.SanaSprintImg2ImgPipeline]]

```python
diffusers.SanaSprintImg2ImgPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: AutoencoderDC, transformer: SanaTransformer2DModel, scheduler: DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py#L146)

Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641).

#### __call__[[diffusers.SanaSprintImg2ImgPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, num_inference_steps: int = 2, timesteps: list = None, max_timesteps: float = 1.5708, intermediate_timesteps: float = 1.3, guidance_scale: float = 4.5, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, strength: float = 0.6, num_images_per_prompt: int | None = 1, height: int = 1024, width: int = 1024, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py#L629)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

max_timesteps (`float`, *optional*, defaults to 1.57080) : The maximum timestep value used in the SCM scheduler.

intermediate_timesteps (`float`, *optional*, defaults to 1.3) : The intermediate timestep value used in SCM scheduler (only used when num_inference_steps=2).

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image`, numpy array or tensor representing an image batch to be used as the starting point. Can also accept image latents as `image`, but if passing latents directly it is not encoded again.

strength (`float`, *optional*, defaults to 0.6) : Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a starting point and more noise is added the higher the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of 1 essentially ignores `image`.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

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
>>> from diffusers import SanaSprintImg2ImgPipeline
>>> from diffusers.utils.loading_utils import load_image

>>> pipe = SanaSprintImg2ImgPipeline.from_pretrained(
...     "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
... )

>>> image = pipe(prompt="a cute pink bear", image=image, strength=0.5, height=832, width=480).images[0]
>>> image[0].save("output.png")
```

#### encode_prompt[[diffusers.SanaSprintImg2ImgPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], num_images_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint_img2img.py#L240)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded 

num_images_per_prompt (`int`, *optional*, defaults to 1) : number of images that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

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

### Sana Video
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/sana_video.md

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
