# Framepack

  

[Packing Input Frame Context in Next-Frame Prediction Models for Video Generation](https://huggingface.co/papers/2504.12626) by Lvmin Zhang and Maneesh Agrawala.

*We present a neural network structure, FramePack, to train next-frame (or next-frame-section) prediction models for video generation. The FramePack compresses input frames to make the transformer context length a fixed number regardless of the video length. As a result, we are able to process a large number of frames using video diffusion with computation bottleneck similar to image diffusion. This also makes the training video batch sizes significantly higher (batch sizes become comparable to image diffusion training). We also propose an anti-drifting sampling method that generates frames in inverted temporal order with early-established endpoints to avoid exposure bias (error accumulation over iterations). Finally, we show that existing video diffusion models can be finetuned with FramePack, and their visual quality may be improved because the next-frame prediction supports more balanced diffusion schedulers with less extreme flow shift timesteps.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

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

Group offloading ([apply_group_offloading()](/docs/diffusers/v0.43.0/en/api/utilities#diffusers.hooks.apply_group_offloading)) provides aggressive memory optimizations for offloading internal parts of any model to the CPU, with possibly no additional overhead to generation time. If you have very low VRAM available, this approach may be suitable for you depending on the amount of CPU RAM available.

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py#L243)

**Parameters:**

text_encoder (`LlamaModel`) : [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

tokenizer (`LlamaTokenizer`) : Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

transformer ([HunyuanVideoTransformer3DModel](/docs/diffusers/v0.43.0/en/api/models/hunyuan_video_transformer_3d#diffusers.HunyuanVideoTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLHunyuanVideo](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_hunyuan_video#diffusers.AutoencoderKLHunyuanVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder_2 (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

tokenizer_2 (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

Pipeline for text-to-video generation using HunyuanVideo.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideoFramepackPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], last_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, prompt: str | list[str] = None, prompt_2: str | list[str] = None, negative_prompt: str | list[str] = None, negative_prompt_2: str | list[str] = None, height: int = 720, width: int = 1280, num_frames: int = 129, latent_window_size: int = 9, num_inference_steps: int = 50, sigmas: list[float] = None, true_cfg_scale: float = 1.0, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, image_latents: typing.Optional[torch.Tensor] = None, last_image_latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], prompt_template: dict[str, typing.Any] = {'template': '<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>', 'crop_start': 95}, max_sequence_length: int = 256, sampling_type: FramepackSamplingType = <FramepackSamplingType.INVERTED_ANTI_DRIFTING: 'inverted_anti_drifting'>)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_framepack.py#L586)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video/pipeline_output.py#L11)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for HunyuanVideo pipelines.

### Cogview4
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/cogview4.md

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

# CogView4

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

## CogView4Pipeline[[diffusers.CogView4Pipeline]]

#### diffusers.CogView4Pipeline[[diffusers.CogView4Pipeline]]

```python
diffusers.CogView4Pipeline(tokenizer: AutoTokenizer, text_encoder: GlmModel, vae: AutoencoderKL, transformer: CogView4Transformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L137)

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`GLMModel`) : Frozen text-encoder. CogView4 uses [glm-4-9b-hf](https://huggingface.co/THUDM/glm-4-9b-hf).

tokenizer (`PreTrainedTokenizer`) : Tokenizer of class [PreTrainedTokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).

transformer ([CogView4Transformer2DModel](/docs/diffusers/v0.43.0/en/api/models/cogview4_transformer2d#diffusers.CogView4Transformer2DModel)) : A text conditioned `CogView4Transformer2DModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.43.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

Pipeline for text-to-image generation using CogView4.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.CogView4Pipeline.__call__]]

```python
__call__(prompt: str | list[str] | None = None, negative_prompt: str | list[str] | None = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, timesteps: list[int] | None = None, sigmas: list[float] | None = None, guidance_scale: float = 5.0, num_images_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, prompt_embeds: typing.Optional[torch.FloatTensor] = None, negative_prompt_embeds: typing.Optional[torch.FloatTensor] = None, original_size: tuple[int, int] | None = None, crops_coords_top_left: tuple[int, int] = (0, 0), output_type: str = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L402)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. If not provided, it is set to 1024.

width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. If not provided it is set to 1024.

num_inference_steps (`int`, *optional*, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to `5.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_images_per_prompt (`int`, *optional*, defaults to `1`) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

original_size (`tuple[int]`, *optional*, defaults to (1024, 1024)) : If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled. `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

crops_coords_top_left (`tuple[int]`, *optional*, defaults to (0, 0)) : `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, defaults to `224`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

**Returns:** `~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput` or `tuple`

`~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput` if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CogView4Pipeline

>>> pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A photo of an astronaut riding a horse on mars"
>>> image = pipe(prompt).images[0]
>>> image.save("output.png")
```

#### encode_prompt[[diffusers.CogView4Pipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_images_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, max_sequence_length: int = 1024)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L221)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

max_sequence_length (`int`, defaults to `1024`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

Encodes the prompt into text encoder hidden states.

## CogView4PipelineOutput[[diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput]]

#### diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput[[diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput]]

```python
diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/cogview4/pipeline_output.py#L10)

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

Output class for CogView3 pipelines.

### Chronoedit
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/chronoedit.md

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

  
    
      
    
  

# ChronoEdit

[ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation](https://huggingface.co/papers/2510.04290) from NVIDIA and University of Toronto, by Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, Huan Ling.

> **TL;DR:** ChronoEdit reframes image editing as a video generation task, using input and edited images as start/end frames to leverage pretrained video models with temporal consistency. A temporal reasoning stage introduces reasoning tokens to ensure physically plausible edits and visualize the editing trajectory.

*Recent advances in large generative models have greatly enhanced both image editing and in-context image generation, yet a critical gap remains in ensuring physical consistency, where edited objects must remain coherent. This capability is especially vital for world simulation related tasks. In this paper, we present ChronoEdit, a framework that reframes image editing as a video generation problem. First, ChronoEdit treats the input and edited images as the first and last frames of a video, allowing it to leverage large pretrained video generative models that capture not only object appearance but also the implicit physics of motion and interaction through learned temporal consistency. Second, ChronoEdit introduces a temporal reasoning stage that explicitly performs editing at inference time. Under this setting, target frame is jointly denoised with reasoning tokens to imagine a plausible editing trajectory that constrains the solution space to physically viable transformations. The reasoning tokens are then dropped after a few steps to avoid the high computational cost of rendering a full video. To validate ChronoEdit, we introduce PBench-Edit, a new benchmark of image-prompt pairs for contexts that require physical consistency, and demonstrate that ChronoEdit surpasses state-of-the-art baselines in both visual fidelity and physical plausibility. Project page for code and models: [this https URL](https://research.nvidia.com/labs/toronto-ai/chronoedit).*

The ChronoEdit pipeline is developed by the ChronoEdit Team. The original code is available on [GitHub](https://github.com/nv-tlabs/ChronoEdit), and pretrained models can be found in the [nvidia/ChronoEdit](https://huggingface.co/collections/nvidia/chronoedit) collection on Hugging Face.

Available Models/LoRAs:
- [nvidia/ChronoEdit-14B-Diffusers](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers)
- [nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers-Upscaler-Lora)
- [nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora)

### Image Editing

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

Optionally, enable **temporal reasoning** for improved physical consistency:
```py
output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=29,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_temporal_reasoning=True,
    num_temporal_reasoning_steps=50,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

### Inference with 8-Step Distillation Lora

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, dtype=torch.bfloat16)
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers", weight_name="lora/chronoedit_distill_lora.safetensors", adapter_name="distill")
pipe.fuse_lora(adapter_names=["distill"], lora_scale=1.0)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=8,
    guidance_scale=1.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output.png")
```

### Inference with Multiple LoRAs

```py
import torch
import numpy as np
from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image

model_id = "nvidia/ChronoEdit-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
transformer = ChronoEditTransformer3DModel.from_pretrained(model_id, subfolder="transformer", dtype=torch.bfloat16)
pipe = ChronoEditPipeline.from_pretrained(model_id, image_encoder=image_encoder, transformer=transformer, vae=vae, dtype=torch.bfloat16)
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers-Paint-Brush-Lora", weight_name="paintbrush_lora_diffusers.safetensors", adapter_name="paintbrush")
pipe.load_lora_weights("nvidia/ChronoEdit-14B-Diffusers", weight_name="lora/chronoedit_distill_lora.safetensors", adapter_name="distill")
pipe.fuse_lora(adapter_names=["paintbrush", "distill"], lora_scale=1.0)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=2.0)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

image = load_image(
    "https://raw.githubusercontent.com/nv-tlabs/ChronoEdit/refs/heads/main/assets/images/input_paintbrush.png"
)
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
print("width", width, "height", height)
image = image.resize((width, height))
prompt = (
    "Turn the pencil sketch in the image into an actual object that is consistent with the image’s content. The user wants to change the sketch to a crown and a hat."
)

output = pipe(
    image=image,
    prompt=prompt,
    height=height,
    width=width,
    num_frames=5,
    num_inference_steps=8,
    guidance_scale=1.0,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8")).save("output_1.png")
```

## ChronoEditPipeline[[diffusers.ChronoEditPipeline]]

#### diffusers.ChronoEditPipeline[[diffusers.ChronoEditPipeline]]

```python
diffusers.ChronoEditPipeline(tokenizer: AutoTokenizer, text_encoder: UMT5EncoderModel, image_encoder: CLIPVisionModel, image_processor: CLIPImageProcessorPil, transformer: ChronoEditTransformer3DModel, vae: AutoencoderKLWan, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L128)

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

image_encoder (`CLIPVisionModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.

transformer ([WanTransformer3DModel](/docs/diffusers/v0.43.0/en/api/models/wan_transformer_3d#diffusers.WanTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for image-to-video generation using Wan.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.ChronoEditPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 480, width: int = 832, num_frames: int = 81, num_inference_steps: int = 50, guidance_scale: float = 5.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, image_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int, NoneType], diffusers.callbacks.PipelineCallback | diffusers.callbacks.MultiPipelineCallbacks]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512, enable_temporal_reasoning: bool = False, num_temporal_reasoning_steps: int = 0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L470)

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, defaults to `480`) : The height of the generated video.

width (`int`, defaults to `832`) : The width of the generated video.

num_frames (`int`, defaults to `81`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `5.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `negative_prompt` input argument.

image_embeds (`torch.Tensor`, *optional*) : Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided, image embeddings are generated from the `image` input argument.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `ChronoEditPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, defaults to `512`) : The maximum sequence length of the text encoder. If the prompt is longer than this, it will be truncated. If the prompt is shorter, it will be padded to this length.

enable_temporal_reasoning (`bool`, *optional*, defaults to `False`) : Whether to enable temporal reasoning.

num_temporal_reasoning_steps (`int`, *optional*, defaults to `0`) : The number of steps to enable temporal reasoning.

**Returns:** `~ChronoEditPipelineOutput` or `tuple`

If `return_dict` is `True`, `ChronoEditPipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> import numpy as np
>>> from diffusers import AutoencoderKLWan, ChronoEditTransformer3DModel, ChronoEditPipeline
>>> from diffusers.utils import export_to_video, load_image
>>> from transformers import CLIPVisionModel

>>> # Available models: nvidia/ChronoEdit-14B-Diffusers
>>> model_id = "nvidia/ChronoEdit-14B-Diffusers"
>>> image_encoder = CLIPVisionModel.from_pretrained(
...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
... )
>>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
>>> transformer = ChronoEditTransformer3DModel.from_pretrained(
...     model_id, subfolder="transformer", torch_dtype=torch.bfloat16
... )
>>> pipe = ChronoEditPipeline.from_pretrained(
...     model_id, vae=vae, image_encoder=image_encoder, transformer=transformer, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> image = load_image("https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png")
>>> max_area = 720 * 1280
>>> aspect_ratio = image.height / image.width
>>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
>>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
>>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
>>> image = image.resize((width, height))
>>> prompt = (
...     "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
...     "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
... )

>>> output = pipe(
...     image=image,
...     prompt=prompt,
...     height=height,
...     width=width,
...     num_frames=5,
...     guidance_scale=5.0,
...     enable_temporal_reasoning=False,
...     num_temporal_reasoning_steps=0,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=16)
```

#### encode_prompt[[diffusers.ChronoEditPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/chronoedit/pipeline_chronoedit.py#L242)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

Encodes the prompt into text encoder hidden states.

## ChronoEditPipelineOutput[[diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput]]

#### diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput[[diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput]]

```python
diffusers.pipelines.chronoedit.pipeline_output.ChronoEditPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/chronoedit/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for ChronoEdit pipelines.

### Consistency Models
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/consistency_models.md
