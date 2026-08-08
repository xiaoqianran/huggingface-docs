# Cosmos

[Cosmos World Foundation Model Platform for Physical AI](https://huggingface.co/papers/2501.03575) by NVIDIA.

*Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make our platform open-source and our models open-weight with permissive licenses available via https://github.com/NVIDIA/Cosmos.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Basic usage

```python
import torch
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video

model_id = "nvidia/Cosmos-Predict2.5-2B"
pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    model_id, revision="diffusers/base/post-trained", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt = "As the red light shifts to green, the red bus at the intersection begins to move forward, its headlights cutting through the falling snow. The snowy tire tracks deepen as the vehicle inches ahead, casting fresh lines onto the slushy road. Around it, streetlights glow warmer, illuminating the drifting flakes and wet reflections on the asphalt. Other cars behind start to edge forward, their beams joining the scene. The stillness of the urban street transitions into motion as the quiet snowfall is punctuated by the slow advance of traffic through the frosty city corridor."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    image=None,
    video=None,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=93,
    generator=torch.Generator().manual_seed(1),
).frames[0]
export_to_video(output, "text2world.mp4", fps=16)
```

## Cosmos2_5_TransferPipeline[[diffusers.Cosmos2_5_TransferPipeline]]

#### diffusers.Cosmos2_5_TransferPipeline[[diffusers.Cosmos2_5_TransferPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py#L152)

Pipeline for Cosmos Transfer2.5, supporting auto-regressive inference.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.Cosmos2_5_TransferPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py#L569[{"name": "controls", "val": ": typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], typing.List[PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]]]"}, {"name": "controls_conditioning_scale", "val": ": typing.Union[float, typing.List[float]] = 1.0"}, {"name": "prompt", "val": ": typing.Union[str, typing.List[str], NoneType] = None"}, {"name": "negative_prompt", "val": ": typing.Union[str, typing.List[str]] = 'The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.'"}, {"name": "height", "val": ": int = 704"}, {"name": "width", "val": ": typing.Optional[int] = None"}, {"name": "num_frames", "val": ": typing.Optional[int] = None"}, {"name": "num_frames_per_chunk", "val": ": int = 93"}, {"name": "num_inference_steps", "val": ": int = 36"}, {"name": "guidance_scale", "val": ": float = 3.0"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": typing.Union[torch._C.Generator, typing.List[torch._C.Generator], NoneType] = None"}, {"name": "latents", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "prompt_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "negative_prompt_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_type", "val": ": typing.Optional[str] = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int, typing.Dict], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": typing.List[str] = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "conditional_frame_timestep", "val": ": float = 0.1"}, {"name": "num_ar_conditional_frames", "val": ": typing.Optional[int] = 1"}, {"name": "num_ar_latent_conditional_frames", "val": ": typing.Optional[int] = None"}]- **controls** (`PipelineImageInput`, `List[PipelineImageInput]`) --
  Control image or video input used by the ControlNet.
- **controls_conditioning_scale** (`float` or `List[float]`, *optional*, defaults to `1.0`) --
  The scale factor(s) for the ControlNet outputs. A single float is broadcast to all control blocks.
- **prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts to guide generation. Required unless `prompt_embeds` is supplied.
- **negative_prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  not greater than `1`).
- **height** (`int`, defaults to `704`) --
  The height in pixels of the generated image.
- **width** (`int`, *optional*) --
  The width in pixels of the generated image. If not provided, this will be determined based on the
  aspect ratio of the input and the provided height.
- **num_frames** (`int`, *optional*) --
  Number of output frames. Defaults to `None` to output the same number of frames as the input
  `controls`.
- **num_frames_per_chunk** (`int`, *optional*, defaults to `93`) --
  Number of frames generated per auto-regressive chunk. When the total number of frames exceeds this
  value, generation is split into multiple chunks using a sliding-window approach.
- **num_inference_steps** (`int`, defaults to `36`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `3.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `List[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs. Can be used to
  tweak the same generation with different prompts. If not provided, a latents tensor is generated by
  sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `CosmosPipelineOutput` instead of a plain tuple.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
  the prompt is shorter than this length, it will be padded.
- **conditional_frame_timestep** (`float`, *optional*, defaults to 0.1) --
  Timestep value used for the conditional frames during denoising. Must be in the `[0, 1]` interval.
- **num_ar_conditional_frames** (`int`, *optional*, defaults to `1`) --
  Number of frames to condition on subsequent inference loops in auto-regressive inference, i.e. for the
  second chunk and onwards. Only used if `num_ar_latent_conditional_frames` is `None`.

  This is only used when auto-regressive inference is performed, i.e. when the number of frames in
  controls is > num_frames_per_chunk
- **num_ar_latent_conditional_frames** (`int`, *optional*) --
  Number of latent frames to condition on subsequent inference loops in auto-regressive inference, i.e.
  for the second chunk and onwards. Only used if `num_ar_conditional_frames` is `None`.

  This is only used when auto-regressive inference is performed, i.e. when the number of frames in
  controls is > num_frames_per_chunk0`~CosmosPipelineOutput` or `tuple`If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

`controls` drive the conditioning through ControlNet. Controls are assumed to be pre-processed, e.g. edge maps
are pre-computed.

Setting `num_frames` will restrict the total number of frames output, if not provided or assigned to None
(default) then the number of output frames will match the input `controls`.

Auto-regressive inference is supported and thus a sliding window of `num_frames_per_chunk` frames are used per
denoising loop. In addition, when auto-regressive inference is performed, the previous
`num_ar_latent_conditional_frames` or `num_ar_conditional_frames` are used to condition the following denoising
inference loops.

Examples:
```python
>>> import cv2
>>> import numpy as np
>>> from PIL import Image
>>> import torch
>>> from diffusers import Cosmos2_5_TransferPipeline, AutoModel
>>> from diffusers.utils import export_to_video, load_video

>>> model_id = "nvidia/Cosmos-Transfer2.5-2B"
>>> # Load a Transfer2.5 controlnet variant (edge, depth, seg, or blur)
>>> controlnet = AutoModel.from_pretrained(
...     model_id, revision="diffusers/controlnet/general/edge", torch_dtype=torch.bfloat16
... )
>>> pipe = Cosmos2_5_TransferPipeline.from_pretrained(
...     model_id, controlnet=controlnet, revision="diffusers/general", torch_dtype=torch.bfloat16
... )
>>> pipe = pipe.to("cuda")

>>> # Video2World with edge control: Generate video guided by edge maps extracted from input video.
>>> prompt = (
...     "The video is a demonstration of robotic manipulation, likely in a laboratory or testing environment. It"
...     "features two robotic arms interacting with a piece of blue fabric. The setting is a room with a beige"
...     "couch in the background, providing a neutral backdrop for the robotic activity. The robotic arms are"
...     "positioned on either side of the fabric, which is placed on a yellow cushion. The left robotic arm is"
...     "white with a black gripper, while the right arm is black with a more complex, articulated gripper. At the"
...     "beginning, the fabric is laid out on the cushion. The left robotic arm approaches the fabric, its gripper"
...     "opening and closing as it positions itself. The right arm remains stationary initially, poised to assist."
...     "As the video progresses, the left arm grips the fabric, lifting it slightly off the cushion. The right arm"
...     "then moves in, its gripper adjusting to grasp the opposite side of the fabric. Both arms work in"
...     "coordination, lifting and holding the fabric between them. The fabric is manipulated with precision,"
...     "showcasing the dexterity and control of the robotic arms. The camera remains static throughout, focusing"
...     "on the interaction between the robotic arms and the fabric, allowing viewers to observe the detailed"
...     "movements and coordination involved in the task."
... )
>>> negative_prompt = (
...     "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
...     "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
...     "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
...     "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
...     "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
...     "Overall, the video is of poor quality."
... )
>>> input_video = load_video(
...     "https://github.com/nvidia-cosmos/cosmos-transfer2.5/raw/refs/heads/main/assets/robot_example/robot_input.mp4"
... )
>>> num_frames = 93

>>> # Extract edge maps from the input video using Canny edge detection
>>> edge_maps = [
...     cv2.Canny(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR), 100, 200)
...     for frame in input_video[:num_frames]
... ]
>>> edge_maps = np.stack(edge_maps)[None]  # (T, H, W) -> (1, T, H, W)
>>> controls = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)  # (1, T, H, W) -> (3, T, H, W)
>>> controls = [Image.fromarray(x.numpy()) for x in controls.permute(1, 2, 3, 0)]
>>> export_to_video(controls, "edge_controlled_video_edge.mp4", fps=30)

>>> # Transfer inference with controls.
>>> video = pipe(
...     controls=controls,
...     controls_conditioning_scale=1.0,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     num_frames=num_frames,
... ).frames[0]
>>> export_to_video(video, "edge_controlled_video.mp4", fps=30)
```

**Parameters:**

text_encoder (`Qwen2_5_VLForConditionalGeneration`) : Frozen text-encoder. Cosmos Transfer2.5 uses the [Qwen2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) encoder.

tokenizer (`AutoTokenizer`) : Tokenizer associated with the Qwen2.5 VL encoder.

transformer ([CosmosTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cosmos_transformer3d#diffusers.CosmosTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

controlnet (`CosmosControlNetModel`) : ControlNet used to condition generation on control inputs.

**Returns:**

``~CosmosPipelineOutput` or `tuple``

If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.Cosmos2_5_TransferPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_transfer.py#L296)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

## Cosmos2_5_PredictBasePipeline[[diffusers.Cosmos2_5_PredictBasePipeline]]

#### diffusers.Cosmos2_5_PredictBasePipeline[[diffusers.Cosmos2_5_PredictBasePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py#L185)

Pipeline for [Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) base model.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.Cosmos2_5_PredictBasePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py#L544[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "video", "val": ": list[PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]] | None = None"}, {"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 704"}, {"name": "width", "val": ": int = 1280"}, {"name": "num_frames", "val": ": int = 93"}, {"name": "num_inference_steps", "val": ": int = 36"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, NoneType], diffusers.callbacks.PipelineCallback | diffusers.callbacks.MultiPipelineCallbacks]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "conditional_frame_timestep", "val": ": float = 0.0001"}, {"name": "num_latent_conditional_frames", "val": ": int = 2"}]- **image** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*) --
  Optional single image for Image2World conditioning. Must be `None` when `video` is provided.
- **video** (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*) --
  Optional input video for Video2World conditioning. Must be `None` when `image` is provided.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide generation. Required unless `prompt_embeds` is supplied.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  not greater than `1`).
- **height** (`int`, defaults to `704`) --
  The height in pixels of the generated image.
- **width** (`int`, defaults to `1280`) --
  The width in pixels of the generated image.
- **num_frames** (`int`, defaults to `93`) --
  Number of output frames. Use `93` for world (video) generation; set to `1` to return a single frame.
- **num_inference_steps** (`int`, defaults to `35`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `7.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`.
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
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `CosmosPipelineOutput` instead of a plain tuple.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
  the prompt is shorter than this length, it will be padded.
- **num_latent_conditional_frames** (`int`, defaults to `2`) --
  Number of latent conditional frames to use for Video2World conditioning. The number of pixel frames
  extracted from the input video is calculated as `4 * (num_latent_conditional_frames - 1) + 1`. Set to 1
  for Image2World-like behavior (single frame conditioning).
- **conditional_frame_timestep** (`float`, *optional*, defaults to 0.0001) --
  Timestep value used for the conditional frames during denoising.0`~CosmosPipelineOutput` or `tuple`If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation. Supports three modes:

- **Text2World**: `image=None`, `video=None`, `prompt` provided. Generates a world clip.
- **Image2World**: `image` provided, `video=None`, `prompt` provided. Conditions on a single frame.
- **Video2World**: `video` provided, `image=None`, `prompt` provided. Conditions on an input clip.

Set `num_frames=93` (default) to produce a world video, or `num_frames=1` to produce a single image frame (the
above in "*2Image mode").

Outputs follow `output_type` (e.g., `"pil"` returns a list of `num_frames` PIL images per prompt).

Examples:
```python
>>> import torch
>>> from diffusers import Cosmos2_5_PredictBasePipeline
>>> from diffusers.utils import export_to_video, load_image, load_video

>>> model_id = "nvidia/Cosmos-Predict2.5-2B"
>>> pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
...     model_id, revision="diffusers/base/post-trained", torch_dtype=torch.bfloat16
... )
>>> pipe = pipe.to("cuda")

>>> # Common negative prompt reused across modes.
>>> negative_prompt = (
...     "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
...     "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
...     "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
...     "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
...     "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
...     "Overall, the video is of poor quality."
... )

>>> # Text2World: generate a 93-frame world video from text only.
>>> prompt = (
...     "As the red light shifts to green, the red bus at the intersection begins to move forward, its headlights "
...     "cutting through the falling snow. The snowy tire tracks deepen as the vehicle inches ahead, casting fresh "
...     "lines onto the slushy road. Around it, streetlights glow warmer, illuminating the drifting flakes and wet "
...     "reflections on the asphalt. Other cars behind start to edge forward, their beams joining the scene. "
...     "The stillness of the urban street transitions into motion as the quiet snowfall is punctuated by the slow "
...     "advance of traffic through the frosty city corridor."
... )
>>> video = pipe(
...     image=None,
...     video=None,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     num_frames=93,
...     generator=torch.Generator().manual_seed(1),
... ).frames[0]
>>> export_to_video(video, "text2world.mp4", fps=16)

>>> # Image2World: condition on a single image and generate a 93-frame world video.
>>> prompt = (
...     "A high-definition video captures the precision of robotic welding in an industrial setting. "
...     "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. "
...     "The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid "
...     "display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring "
...     "the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a "
...     "ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video "
...     "progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. "
...     "The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. "
...     "The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with "
...     "the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
... )
>>> image = load_image(
...     "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg"
... )
>>> video = pipe(
...     image=image,
...     video=None,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     num_frames=93,
...     generator=torch.Generator().manual_seed(1),
... ).frames[0]
>>> export_to_video(video, "image2world.mp4", fps=16)

>>> # Video2World: condition on an input clip and predict a 93-frame world video.
>>> prompt = (
...     "The video opens with an aerial view of a large-scale sand mining construction operation, showcasing extensive piles "
...     "of brown sand meticulously arranged in parallel rows. A central water channel, fed by a water pipe, flows through the "
...     "middle of these sand heaps, creating ripples and movement as it cascades down. The surrounding area features dense green "
...     "vegetation on the left, contrasting with the sandy terrain, while a body of water is visible in the background on the right. "
...     "As the video progresses, a piece of heavy machinery, likely a bulldozer, enters the frame from the right, moving slowly along "
...     "the edge of the sand piles. This machinery's presence indicates ongoing construction work in the operation. The final frame "
...     "captures the same scene, with the water continuing its flow and the bulldozer still in motion, maintaining the dynamic yet "
...     "steady pace of the construction activity."
... )
>>> input_video = load_video(
...     "https://github.com/nvidia-cosmos/cosmos-predict2.5/raw/refs/heads/main/assets/base/sand_mining.mp4"
... )
>>> video = pipe(
...     image=None,
...     video=input_video,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     num_frames=93,
...     generator=torch.Generator().manual_seed(1),
... ).frames[0]
>>> export_to_video(video, "video2world.mp4", fps=16)

>>> # To produce an image instead of a world (video) clip, set num_frames=1 and
>>> # save the first frame: pipe(..., num_frames=1).frames[0][0].
```

**Parameters:**

text_encoder (`Qwen2_5_VLForConditionalGeneration`) : Frozen text-encoder. Cosmos Predict2.5 uses the [Qwen2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) encoder.

tokenizer (`AutoTokenizer`) : Tokenizer associated with the Qwen2.5 VL encoder.

transformer ([CosmosTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cosmos_transformer3d#diffusers.CosmosTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~CosmosPipelineOutput` or `tuple``

If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.Cosmos2_5_PredictBasePipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py#L324)

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

## CosmosTextToWorldPipeline[[diffusers.CosmosTextToWorldPipeline]]

#### diffusers.CosmosTextToWorldPipeline[[diffusers.CosmosTextToWorldPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py#L140)

Pipeline for text-to-world generation using [Cosmos Predict1](https://github.com/nvidia-cosmos/cosmos-predict1).

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.CosmosTextToWorldPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py#L401[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 704"}, {"name": "width", "val": ": int = 1280"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "num_inference_steps", "val": ": int = 36"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "fps", "val": ": int = 30"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  not greater than `1`).
- **height** (`int`, defaults to `720`) --
  The height in pixels of the generated image.
- **width** (`int`, defaults to `1280`) --
  The width in pixels of the generated image.
- **num_frames** (`int`, defaults to `121`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `36`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `7.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`.
- **fps** (`int`, defaults to `30`) --
  The frames per second of the generated video.
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
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `CosmosPipelineOutput` instead of a plain tuple.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
  the prompt is shorter than this length, it will be padded.0`~CosmosPipelineOutput` or `tuple`If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CosmosTextToWorldPipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
>>> pipe = CosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

>>> output = pipe(prompt=prompt).frames[0]
>>> export_to_video(output, "output.mp4", fps=30)
```

**Parameters:**

text_encoder (`T5EncoderModel`) : Frozen text-encoder. Cosmos uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.

tokenizer (`T5TokenizerFast`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CosmosTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cosmos_transformer3d#diffusers.CosmosTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLCosmos](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_cosmos#diffusers.AutoencoderKLCosmos)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~CosmosPipelineOutput` or `tuple``

If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.CosmosTextToWorldPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py#L239)

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

## CosmosVideoToWorldPipeline[[diffusers.CosmosVideoToWorldPipeline]]

#### diffusers.CosmosVideoToWorldPipeline[[diffusers.CosmosVideoToWorldPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py#L183)

Pipeline for image-to-world and video-to-world generation using [Cosmos
Predict-1](https://github.com/nvidia-cosmos/cosmos-predict1).

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.CosmosVideoToWorldPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py#L513[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "video", "val": ": list = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 704"}, {"name": "width", "val": ": int = 1280"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "num_inference_steps", "val": ": int = 36"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "input_frames_guidance", "val": ": bool = False"}, {"name": "augment_sigma", "val": ": float = 0.001"}, {"name": "fps", "val": ": int = 30"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **image** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*) --
  The image to be used as a conditioning input for the video generation.
- **video** (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*) --
  The video to be used as a conditioning input for the video generation.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  not greater than `1`).
- **height** (`int`, defaults to `720`) --
  The height in pixels of the generated image.
- **width** (`int`, defaults to `1280`) --
  The width in pixels of the generated image.
- **num_frames** (`int`, defaults to `121`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `36`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `7.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`.
- **input_frames_guidance** (`bool`, *optional*, defaults to `False`) --
  Whether to apply guidance on the conditional input frames.
- **augment_sigma** (`float`, *optional*, defaults to 0.001) --
  Sigma value used to augment the conditional latents during denoising.
- **fps** (`int`, defaults to `30`) --
  The frames per second of the generated video.
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
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `CosmosPipelineOutput` instead of a plain tuple.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
  the prompt is shorter than this length, it will be padded.0`~CosmosPipelineOutput` or `tuple`If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:

Image conditioning:

```python
>>> import torch
>>> from diffusers import CosmosVideoToWorldPipeline
>>> from diffusers.utils import export_to_video, load_image

>>> model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"
>>> pipe = CosmosVideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day."
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
... )

>>> video = pipe(image=image, prompt=prompt).frames[0]
>>> export_to_video(video, "output.mp4", fps=30)
```

Video conditioning:

```python
>>> import torch
>>> from diffusers import CosmosVideoToWorldPipeline
>>> from diffusers.utils import export_to_video, load_video

>>> model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"
>>> pipe = CosmosVideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe.transformer = torch.compile(pipe.transformer)
>>> pipe.to("cuda")

>>> prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
>>> video = load_video(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
... )[
...     :21
... ]  # This example uses only the first 21 frames

>>> video = pipe(video=video, prompt=prompt).frames[0]
>>> export_to_video(video, "output.mp4", fps=30)
```

**Parameters:**

text_encoder (`T5EncoderModel`) : Frozen text-encoder. Cosmos uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.

tokenizer (`T5TokenizerFast`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CosmosTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cosmos_transformer3d#diffusers.CosmosTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLCosmos](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_cosmos#diffusers.AutoencoderKLCosmos)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~CosmosPipelineOutput` or `tuple``

If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.CosmosVideoToWorldPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_video2world.py#L285)

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

## Cosmos2TextToImagePipeline[[diffusers.Cosmos2TextToImagePipeline]]

#### diffusers.Cosmos2TextToImagePipeline[[diffusers.Cosmos2TextToImagePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py#L143)

Pipeline for text-to-image generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.Cosmos2TextToImagePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py#L417[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 768"}, {"name": "width", "val": ": int = 1360"}, {"name": "num_inference_steps", "val": ": int = 35"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  not greater than `1`).
- **height** (`int`, defaults to `768`) --
  The height in pixels of the generated image.
- **width** (`int`, defaults to `1360`) --
  The width in pixels of the generated image.
- **num_inference_steps** (`int`, defaults to `35`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `7.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor is generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `CosmosImagePipelineOutput` instead of a plain tuple.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
  the prompt is shorter than this length, it will be padded.0`~CosmosImagePipelineOutput` or `tuple`If `return_dict` is `True`, `CosmosImagePipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import Cosmos2TextToImagePipeline

>>> # Available checkpoints: nvidia/Cosmos-Predict2-2B-Text2Image, nvidia/Cosmos-Predict2-14B-Text2Image
>>> model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
>>> pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
>>> negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

>>> output = pipe(
...     prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
... ).images[0]
>>> output.save("output.png")
```

**Parameters:**

text_encoder (`T5EncoderModel`) : Frozen text-encoder. Cosmos uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.

tokenizer (`T5TokenizerFast`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CosmosTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cosmos_transformer3d#diffusers.CosmosTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~CosmosImagePipelineOutput` or `tuple``

If `return_dict` is `True`, `CosmosImagePipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.Cosmos2TextToImagePipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py#L254)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of videos that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

## Cosmos2VideoToWorldPipeline[[diffusers.Cosmos2VideoToWorldPipeline]]

#### diffusers.Cosmos2VideoToWorldPipeline[[diffusers.Cosmos2VideoToWorldPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py#L162)

Pipeline for video-to-world generation using [Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.Cosmos2VideoToWorldPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py#L485[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "video", "val": ": list = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 704"}, {"name": "width", "val": ": int = 1280"}, {"name": "num_frames", "val": ": int = 93"}, {"name": "num_inference_steps", "val": ": int = 35"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "fps", "val": ": int = 16"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "sigma_conditioning", "val": ": float = 0.0001"}]- **image** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*) --
  The image to be used as a conditioning input for the video generation.
- **video** (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*) --
  The video to be used as a conditioning input for the video generation.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  not greater than `1`).
- **height** (`int`, defaults to `704`) --
  The height in pixels of the generated image.
- **width** (`int`, defaults to `1280`) --
  The width in pixels of the generated image.
- **num_frames** (`int`, defaults to `93`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `35`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, defaults to `7.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`.
- **fps** (`int`, defaults to `16`) --
  The frames per second of the generated video.
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
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `CosmosPipelineOutput` instead of a plain tuple.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `512`) --
  The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
  the prompt is shorter than this length, it will be padded.
- **sigma_conditioning** (`float`, defaults to `0.0001`) --
  The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
  set to a small value close to zero.0`~CosmosPipelineOutput` or `tuple`If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import Cosmos2VideoToWorldPipeline
>>> from diffusers.utils import export_to_video, load_image

>>> # Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
>>> model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
>>> pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
>>> negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
... )

>>> video = pipe(
...     image=image, prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=16)
```

**Parameters:**

text_encoder (`T5EncoderModel`) : Frozen text-encoder. Cosmos uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-11b](https://huggingface.co/google-t5/t5-11b) variant.

tokenizer (`T5TokenizerFast`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CosmosTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/cosmos_transformer3d#diffusers.CosmosTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

**Returns:**

``~CosmosPipelineOutput` or `tuple``

If `return_dict` is `True`, `CosmosPipelineOutput` is returned, otherwise a `tuple` is returned where
the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.Cosmos2VideoToWorldPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py#L273)

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

## CosmosPipelineOutput[[diffusers.pipelines.cosmos.pipeline_output.CosmosPipelineOutput]]

#### diffusers.pipelines.cosmos.pipeline_output.CosmosPipelineOutput[[diffusers.pipelines.cosmos.pipeline_output.CosmosPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_output.py#L14)

Output class for Cosmos any-to-world/video pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

## CosmosImagePipelineOutput[[diffusers.pipelines.cosmos.pipeline_output.CosmosImagePipelineOutput]]

#### diffusers.pipelines.cosmos.pipeline_output.CosmosImagePipelineOutput[[diffusers.pipelines.cosmos.pipeline_output.CosmosImagePipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_output.py#L29)

Output class for Cosmos any-to-image pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Ltx2
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/ltx2.md

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

# LTX-2

  

[LTX-2](https://hf.co/papers/2601.03233) is a DiT-based foundation model designed to generate synchronized video and audio within a single model. It brings together the core building blocks of modern video generation, with open weights and a focus on practical, local execution.

You can find all the original LTX-Video checkpoints under the [Lightricks](https://huggingface.co/Lightricks) organization.

The original codebase for LTX-2 can be found [here](https://github.com/Lightricks/LTX-2).

## Two-stages Generation
Recommended pipeline to achieve production quality generation, this pipeline is composed of two stages:

- Stage 1: Generate a video at the target resolution using diffusion sampling with classifier-free guidance (CFG). This stage produces a coherent low-noise video sequence that respects the text/image conditioning.
- Stage 2: Upsample the Stage 1 output by 2 and refine details using a distilled LoRA model to improve fidelity and visual quality. Stage 2 may apply lighter CFG to preserve the structure from Stage 1 while enhancing texture and sharpness.

Sample usage of text-to-video two stages pipeline

```py
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda:0"
width = 768
height = 512

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2", torch_dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

# Stage 1 default (non-distilled) inference
frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    sigmas=None,
    guidance_scale=4.0,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "Lightricks/LTX-2",
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

# Load Stage 2 distilled LoRA
pipe.load_lora_weights(
    "Lightricks/LTX-2", adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors"
)
pipe.set_adapters("stage_2_distilled", 1.0)
# VAE tiling is usually necessary to avoid OOM error when VAE decoding
pipe.vae.enable_tiling()
# Change scheduler to use Stage 2 distilled sigmas as is
new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)
pipe.scheduler = new_scheduler
# Stage 2 inference with distilled LoRA and sigmas
video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], # renoise with first sigma value https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py#L218
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_lora_distilled_sample.mp4",
)
```

## Distilled checkpoint generation
Fastest two-stages generation pipeline using a distilled checkpoint.

```py
import torch
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2Pipeline.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload(device=device)

prompt = "A beautiful sunset over the ocean"
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], # renoise with first sigma value https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/distilled.py#L178
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_distilled_sample.mp4",
)
```

## Condition Pipeline Generation

You can use `LTX2ConditionPipeline` to specify image and/or video conditions at arbitrary latent indices. For example, we can specify both a first-frame and last-frame condition to perform first-last-frame-to-video (FLF2V) generation:

```py
import torch
from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video
from diffusers.utils import load_image

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are "
    "delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright "
    "sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, "
    "low-angle perspective."
)

first_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png",
)
last_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png",
)
first_cond = LTX2VideoCondition(frames=first_image, index=0, strength=1.0)
last_cond = LTX2VideoCondition(frames=last_image, index=-1, strength=1.0)
conditions = [first_cond, last_cond]

frame_rate = 24.0
video_latent, audio_latent = pipe(
    conditions=conditions,
    prompt=prompt,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    model_path,
    subfolder="latent_upsampler",
    torch_dtype=torch.bfloat16,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    width=width * 2,
    height=height * 2,
    num_inference_steps=3,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_distilled_flf2v.mp4",
)
```

You can use both image and video conditions:

```py
import torch
from diffusers import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image, load_video

device = "cuda"
width = 768
height = 512
random_seed = 42
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "rootonchair/LTX-2-19b-distilled"

pipe = LTX2ConditionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is "
    "divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features "
    "dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered "
    "clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, "
    "with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The "
    "landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the "
    "solitude and beauty of a winter drive through a mountainous region."
)

cond_video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
)
cond_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
)
video_cond = LTX2VideoCondition(frames=cond_video, index=0, strength=1.0)
image_cond = LTX2VideoCondition(frames=cond_image, index=8, strength=1.0)
conditions = [video_cond, image_cond]

frame_rate = 24.0
video, audio = pipe(
    conditions=conditions,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_cond_video.mp4",
)
```

Because the conditioning is done via latent frames, the 8 data space frames corresponding to the specified latent frame for an image condition will tend to be static.

## Multimodal Guidance

LTX-2.X pipelines support multimodal guidance. It is composed of three terms, all using a CFG-style update rule:

1. Classifier-Free Guidance (CFG): standard [CFG](https://huggingface.co/papers/2207.12598) where the perturbed ("weaker") output is generated using the negative prompt.
2. Spatio-Temporal Guidance (STG): [STG](https://huggingface.co/papers/2411.18664) moves away from a perturbed output created from short-cutting self-attention operations and substitutes in the attention values instead. The idea is that this creates sharper videos and better spatiotemporal consistency.
3. Modality Isolation Guidance: moves away from a perturbed output created from disabling cross-modality (audio-to-video and video-to-audio) cross attention. This guidance is more specific to [LTX-2.X](https://huggingface.co/papers/2601.03233) models, with the idea that this produces better consistency between the generated audio and video.

These are controlled by the `guidance_scale`, `stg_scale`, and `modality_scale` arguments and can be set separately for video and audio. Additionally, for STG the transformer block indices where self-attention is skipped needs to be specified via the `spatio_temporal_guidance_blocks` argument. The LTX-2.X pipelines also support [guidance rescaling](https://huggingface.co/papers/2305.08891) to help reduce over-exposure, which can be a problem when the guidance scales are set to high values.

```py
import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "diffusers/LTX-2.3-Diffusers"

pipe = LTX2ImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)
pipe.vae.enable_tiling()

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    "breath-taking, movie-like shot."
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
)

video, audio = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,  # Recommended LTX-2.3 guidance parameters
    stg_scale=1.0,  # Note that 0.0 (not 1.0) means that STG is disabled (all other guidance is disabled at 1.0)
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,  # Note that a higher CFG guidance scale is recommended for audio
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_3_i2v_stage_1.mp4",
)
```

## Prompt Enhancement

The LTX-2.X models are sensitive to prompting style. Refer to the [official prompting guide](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) for recommendations on how to write a good prompt. Using prompt enhancement, where the supplied prompts are enhanced using the pipeline's text encoder (by default a [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) model) given a system prompt, can also improve sample quality. The optional `processor` pipeline component needs to be present to use prompt enhancement. Enable prompt enhancement by supplying a `system_prompt` argument:

```py
import torch
from transformers import Gemma3Processor
from diffusers import LTX2Pipeline
from diffusers.utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, T2V_DEFAULT_SYSTEM_PROMPT

device = "cuda"
width = 768
height = 512
random_seed = 42
frame_rate = 24.0
generator = torch.Generator(device).manual_seed(random_seed)
model_path = "diffusers/LTX-2.3-Diffusers"

pipe = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()
if getattr(pipe, "processor", None) is None:
    processor = Gemma3Processor.from_pretrained("google/gemma-3-12b-it-qat-q4_0-unquantized")
    pipe.processor = processor

prompt = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    "breath-taking, movie-like shot."
)

video, audio = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],
    use_cross_timestep=True,
    system_prompt=T2V_DEFAULT_SYSTEM_PROMPT,
    generator=generator,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_3_t2v_stage_1.mp4",
)
```

## LTX2Pipeline[[diffusers.LTX2Pipeline]]

#### diffusers.LTX2Pipeline[[diffusers.LTX2Pipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L185)

Pipeline for text-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

__call__diffusers.LTX2Pipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L808[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 768"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "frame_rate", "val": ": float = 24.0"}, {"name": "num_inference_steps", "val": ": int = 40"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "timesteps", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "stg_scale", "val": ": float = 0.0"}, {"name": "modality_scale", "val": ": float = 1.0"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "audio_guidance_scale", "val": ": float | None = None"}, {"name": "audio_stg_scale", "val": ": float | None = None"}, {"name": "audio_modality_scale", "val": ": float | None = None"}, {"name": "audio_guidance_rescale", "val": ": float | None = None"}, {"name": "spatio_temporal_guidance_blocks", "val": ": list[int] | None = None"}, {"name": "noise_scale", "val": ": float = 0.0"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "audio_latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "use_cross_timestep", "val": ": bool = False"}, {"name": "system_prompt", "val": ": str | None = None"}, {"name": "prompt_max_new_tokens", "val": ": int = 512"}, {"name": "prompt_enhancement_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "prompt_enhancement_seed", "val": ": int = 10"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 1024"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is
  a separate value `audio_guidance_scale` for the audio modality).
- **stg_scale** (`float`, *optional*, defaults to `0.0`) --
  Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for
  Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate
  where we move the sample away from a weak sample from a perturbed version of the denoising model.
  Enabling STG will result in an additional denoising model forward pass; the default value of `0.0`
  means that STG is disabled.
- **modality_scale** (`float`, *optional*, defaults to `1.0`) --
  Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a
  weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio)
  cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an
  additional denoising model forward pass; the default value of `1.0` means that modality guidance is
  disabled.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR. Used for the video modality.
- **audio_guidance_scale** (`float`, *optional* defaults to `None`) --
  Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for
  video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest
  that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for
  LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value
  `guidance_scale`.
- **audio_stg_scale** (`float`, *optional*, defaults to `None`) --
  Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and
  audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the
  video value `stg_scale`.
- **audio_modality_scale** (`float`, *optional*, defaults to `None`) --
  Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule
  is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and
  audio. If `None`, defaults to the video value `modality_scale`.
- **audio_guidance_rescale** (`float`, *optional*, defaults to `None`) --
  A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value
  `guidance_rescale`.
- **spatio_temporal_guidance_blocks** (`list[int]`, *optional*, defaults to `None`) --
  The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used
  (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0
  and `[28]` is recommended for LTX-2.3.
- **noise_scale** (`float`, *optional*, defaults to `0.0`) --
  The interpolation factor between random noise and denoised latents at each timestep. Applying noise to
  the `latents` and `audio_latents` before continue denoising.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **audio_latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for text embeddings.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **negative_prompt_attention_mask** (`torch.FloatTensor`, *optional*) --
  Pre-generated attention mask for negative text embeddings.
- **decode_timestep** (`float`, defaults to `0.0`) --
  The timestep at which generated video is decoded.
- **decode_noise_scale** (`float`, defaults to `None`) --
  The interpolation factor between random noise and denoised latents at the decode timestep.
- **use_cross_timestep** (`bool` *optional*, defaults to `False`) --
  Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
  calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
  `False` is the legacy LTX-2.0 behavior.
- **system_prompt** (`str`, *optional*, defaults to `None`) --
  Optional system prompt to use for prompt enhancement. The system prompt will be used by the current
  text encoder (by default, a `Gemma3ForConditionalGeneration` model) to generate an enhanced prompt from
  the original `prompt` to condition generation. If not supplied, prompt enhancement will not be
  performed.
- **prompt_max_new_tokens** (`int`, *optional*, defaults to `512`) --
  The maximum number of new tokens to generate when performing prompt enhancement.
- **prompt_enhancement_kwargs** (`dict[str, Any]`, *optional*, defaults to `None`) --
  Keyword arguments for `self.text_encoder.generate`. If not supplied, default arguments of
  `do_sample=True` and `temperature=0.7` will be used. See
  https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate
  for more details.
- **prompt_enhancement_seed** (`int`, *optional*, default to `10`) --
  Random seed for any random operations during prompt enhancement.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*, defaults to `["latents"]`) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `1024`) --
  Maximum sequence length to use with the `prompt`.0`~pipelines.ltx.LTX2PipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2Pipeline
>>> from diffusers.utils import encode_video

>>> pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> frame_rate = 24.0
>>> video, audio = pipe(
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=40,
...     guidance_scale=4.0,
...     output_type="np",
...     return_dict=False,
... )

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

**Parameters:**

transformer ([LTXVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLLTXVideo](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

connectors (`LTX2TextConnectors`) : Text connector stack used to adapt text encoder hidden states for the video and audio branches.

**Returns:**

``~pipelines.ltx.LTX2PipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### encode_prompt[[diffusers.LTX2Pipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L337)

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
#### enhance_prompt[[diffusers.LTX2Pipeline.enhance_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2.py#L423)

Enhances the supplied `prompt` by generating a new prompt using the current text encoder (default is a
`transformers.Gemma3ForConditionalGeneration` model) from it and a system prompt.

## LTX2ImageToVideoPipeline[[diffusers.LTX2ImageToVideoPipeline]]

#### diffusers.LTX2ImageToVideoPipeline[[diffusers.LTX2ImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L205)

Pipeline for image-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

TODO

__call__diffusers.LTX2ImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L868[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 768"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "frame_rate", "val": ": float = 24.0"}, {"name": "num_inference_steps", "val": ": int = 40"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "stg_scale", "val": ": float = 0.0"}, {"name": "modality_scale", "val": ": float = 1.0"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "audio_guidance_scale", "val": ": float | None = None"}, {"name": "audio_stg_scale", "val": ": float | None = None"}, {"name": "audio_modality_scale", "val": ": float | None = None"}, {"name": "audio_guidance_rescale", "val": ": float | None = None"}, {"name": "spatio_temporal_guidance_blocks", "val": ": list[int] | None = None"}, {"name": "noise_scale", "val": ": float = 0.0"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "audio_latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "use_cross_timestep", "val": ": bool = False"}, {"name": "system_prompt", "val": ": str | None = None"}, {"name": "prompt_max_new_tokens", "val": ": int = 512"}, {"name": "prompt_enhancement_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "prompt_enhancement_seed", "val": ": int = 10"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 1024"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is
  a separate value `audio_guidance_scale` for the audio modality).
- **stg_scale** (`float`, *optional*, defaults to `0.0`) --
  Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for
  Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate
  where we move the sample away from a weak sample from a perturbed version of the denoising model.
  Enabling STG will result in an additional denoising model forward pass; the default value of `0.0`
  means that STG is disabled.
- **modality_scale** (`float`, *optional*, defaults to `1.0`) --
  Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a
  weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio)
  cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an
  additional denoising model forward pass; the default value of `1.0` means that modality guidance is
  disabled.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR. Used for the video modality.
- **audio_guidance_scale** (`float`, *optional* defaults to `None`) --
  Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for
  video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest
  that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for
  LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value
  `guidance_scale`.
- **audio_stg_scale** (`float`, *optional*, defaults to `None`) --
  Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and
  audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the
  video value `stg_scale`.
- **audio_modality_scale** (`float`, *optional*, defaults to `None`) --
  Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule
  is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and
  audio. If `None`, defaults to the video value `modality_scale`.
- **audio_guidance_rescale** (`float`, *optional*, defaults to `None`) --
  A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value
  `guidance_rescale`.
- **spatio_temporal_guidance_blocks** (`list[int]`, *optional*, defaults to `None`) --
  The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used
  (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0
  and `[28]` is recommended for LTX-2.3.
- **noise_scale** (`float`, *optional*, defaults to `0.0`) --
  The interpolation factor between random noise and denoised latents at each timestep. Applying noise to
  the `latents` and `audio_latents` before continue denoising.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **audio_latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for text embeddings.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **negative_prompt_attention_mask** (`torch.FloatTensor`, *optional*) --
  Pre-generated attention mask for negative text embeddings.
- **decode_timestep** (`float`, defaults to `0.0`) --
  The timestep at which generated video is decoded.
- **decode_noise_scale** (`float`, defaults to `None`) --
  The interpolation factor between random noise and denoised latents at the decode timestep.
- **use_cross_timestep** (`bool` *optional*, defaults to `False`) --
  Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
  calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
  `False` is the legacy LTX-2.0 behavior.
- **system_prompt** (`str`, *optional*, defaults to `None`) --
  Optional system prompt to use for prompt enhancement. The system prompt will be used by the current
  text encoder (by default, a `Gemma3ForConditionalGeneration` model) to generate an enhanced prompt from
  the original `prompt` to condition generation. If not supplied, prompt enhancement will not be
  performed.
- **prompt_max_new_tokens** (`int`, *optional*, defaults to `512`) --
  The maximum number of new tokens to generate when performing prompt enhancement.
- **prompt_enhancement_kwargs** (`dict[str, Any]`, *optional*, defaults to `None`) --
  Keyword arguments for `self.text_encoder.generate`. If not supplied, default arguments of
  `do_sample=True` and `temperature=0.7` will be used. See
  https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate
  for more details.
- **prompt_enhancement_seed** (`int`, *optional*, default to `10`) --
  Random seed for any random operations during prompt enhancement.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `1024`) --
  Maximum sequence length to use with the `prompt`.0`~pipelines.ltx.LTX2PipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2ImageToVideoPipeline
>>> from diffusers.utils import encode_video
>>> from diffusers.utils import load_image

>>> pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> image = load_image(
...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
... )
>>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> frame_rate = 24.0
>>> video, audio = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=40,
...     guidance_scale=4.0,
...     output_type="np",
...     return_dict=False,
... )

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

height (`int`, *optional*, defaults to `512`) : The height in pixels of the generated image. This is set to 480 by default for the best results.

width (`int`, *optional*, defaults to `768`) : The width in pixels of the generated image. This is set to 848 by default for the best results.

num_frames (`int`, *optional*, defaults to `121`) : The number of video frames to generate

frame_rate (`float`, *optional*, defaults to `24.0`) : The frames per second (FPS) of the generated video.

num_inference_steps (`int`, *optional*, defaults to 40) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`List[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to `4.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is a separate value `audio_guidance_scale` for the audio modality).

stg_scale (`float`, *optional*, defaults to `0.0`) : Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate where we move the sample away from a weak sample from a perturbed version of the denoising model. Enabling STG will result in an additional denoising model forward pass; the default value of `0.0` means that STG is disabled.

modality_scale (`float`, *optional*, defaults to `1.0`) : Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio) cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an additional denoising model forward pass; the default value of `1.0` means that modality guidance is disabled.

guidance_rescale (`float`, *optional*, defaults to 0.0) : Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when using zero terminal SNR. Used for the video modality.

audio_guidance_scale (`float`, *optional* defaults to `None`) : Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value `guidance_scale`.

audio_stg_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the video value `stg_scale`.

audio_modality_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and audio. If `None`, defaults to the video value `modality_scale`.

audio_guidance_rescale (`float`, *optional*, defaults to `None`) : A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value `guidance_rescale`.

spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`) : The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0 and `[28]` is recommended for LTX-2.3.

noise_scale (`float`, *optional*, defaults to `0.0`) : The interpolation factor between random noise and denoised latents at each timestep. Applying noise to the `latents` and `audio_latents` before continue denoising.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

audio_latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

use_cross_timestep (`bool` *optional*, defaults to `False`) : Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior; `False` is the legacy LTX-2.0 behavior.

system_prompt (`str`, *optional*, defaults to `None`) : Optional system prompt to use for prompt enhancement. The system prompt will be used by the current text encoder (by default, a `Gemma3ForConditionalGeneration` model) to generate an enhanced prompt from the original `prompt` to condition generation. If not supplied, prompt enhancement will not be performed.

prompt_max_new_tokens (`int`, *optional*, defaults to `512`) : The maximum number of new tokens to generate when performing prompt enhancement.

prompt_enhancement_kwargs (`dict[str, Any]`, *optional*, defaults to `None`) : Keyword arguments for `self.text_encoder.generate`. If not supplied, default arguments of `do_sample=True` and `temperature=0.7` will be used. See https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate for more details.

prompt_enhancement_seed (`int`, *optional*, default to `10`) : Random seed for any random operations during prompt enhancement.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length to use with the `prompt`.

**Returns:**

``~pipelines.ltx.LTX2PipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### encode_prompt[[diffusers.LTX2ImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L342)

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
#### enhance_prompt[[diffusers.LTX2ImageToVideoPipeline.enhance_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py#L428)

Enhances the supplied `prompt` by generating a new prompt using the current text encoder (default is a
`transformers.Gemma3ForConditionalGeneration` model) from it and a system prompt.

## LTX2ConditionPipeline[[diffusers.LTX2ConditionPipeline]]

#### diffusers.LTX2ConditionPipeline[[diffusers.LTX2ConditionPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L236)

Pipeline for video generation which allows image conditions to be inserted at arbitary parts of the video.

Reference: https://github.com/Lightricks/LTX-Video

TODO

__call__diffusers.LTX2ConditionPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L1174[{"name": "conditions", "val": ": diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition | list[diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2VideoCondition] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 768"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "frame_rate", "val": ": float = 24.0"}, {"name": "num_inference_steps", "val": ": int = 40"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "timesteps", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "stg_scale", "val": ": float = 0.0"}, {"name": "modality_scale", "val": ": float = 1.0"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "audio_guidance_scale", "val": ": float | None = None"}, {"name": "audio_stg_scale", "val": ": float | None = None"}, {"name": "audio_modality_scale", "val": ": float | None = None"}, {"name": "audio_guidance_rescale", "val": ": float | None = None"}, {"name": "spatio_temporal_guidance_blocks", "val": ": list[int] | None = None"}, {"name": "noise_scale", "val": ": float | None = None"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "audio_latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "use_cross_timestep", "val": ": bool = False"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 1024"}]- **conditions** (`List[LTXVideoCondition], *optional*`) --
  The list of frame-conditioning items for the video generation.
- **prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is
  a separate value `audio_guidance_scale` for the audio modality).
- **stg_scale** (`float`, *optional*, defaults to `0.0`) --
  Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for
  Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate
  where we move the sample away from a weak sample from a perturbed version of the denoising model.
  Enabling STG will result in an additional denoising model forward pass; the default value of `0.0`
  means that STG is disabled.
- **modality_scale** (`float`, *optional*, defaults to `1.0`) --
  Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a
  weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio)
  cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an
  additional denoising model forward pass; the default value of `1.0` means that modality guidance is
  disabled.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR. Used for the video modality.
- **audio_guidance_scale** (`float`, *optional* defaults to `None`) --
  Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for
  video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest
  that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for
  LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value
  `guidance_scale`.
- **audio_stg_scale** (`float`, *optional*, defaults to `None`) --
  Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and
  audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the
  video value `stg_scale`.
- **audio_modality_scale** (`float`, *optional*, defaults to `None`) --
  Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule
  is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and
  audio. If `None`, defaults to the video value `modality_scale`.
- **audio_guidance_rescale** (`float`, *optional*, defaults to `None`) --
  A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value
  `guidance_rescale`.
- **spatio_temporal_guidance_blocks** (`list[int]`, *optional*, defaults to `None`) --
  The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used
  (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0
  and `[28]` is recommended for LTX-2.3.
- **noise_scale** (`float`, *optional*, defaults to `None`) --
  The interpolation factor between random noise and denoised latents at each timestep. Applying noise to
  the `latents` and `audio_latents` before continue denoising. If not set, will be inferred from the
  sigma schedule.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **generator** (`torch.Generator` or `List[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **audio_latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random `generator`.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from `prompt` input argument.
- **prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for text embeddings.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **negative_prompt_attention_mask** (`torch.FloatTensor`, *optional*) --
  Pre-generated attention mask for negative text embeddings.
- **decode_timestep** (`float`, defaults to `0.0`) --
  The timestep at which generated video is decoded.
- **decode_noise_scale** (`float`, defaults to `None`) --
  The interpolation factor between random noise and denoised latents at the decode timestep.
- **use_cross_timestep** (`bool` *optional*, defaults to `False`) --
  Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
  calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
  `False` is the legacy LTX-2.0 behavior.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, *optional*, defaults to `1024`) --
  Maximum sequence length to use with the `prompt`.0`~pipelines.ltx.LTX2PipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2ConditionPipeline
>>> from diffusers.utils import encode_video
>>> from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
>>> from diffusers.utils import load_image

>>> pipe = LTX2ConditionPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> first_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
... )
>>> last_image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
... )
>>> first_cond = LTX2VideoCondition(frames=first_image, index=0, strength=1.0)
>>> last_cond = LTX2VideoCondition(frames=last_image, index=-1, strength=1.0)
>>> conditions = [first_cond, last_cond]
>>> prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, static"

>>> frame_rate = 24.0
>>> video = pipe(
...     conditions=conditions,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=40,
...     guidance_scale=4.0,
...     output_type="np",
...     return_dict=False,
... )
>>> video = (video * 255).round().astype("uint8")
>>> video = torch.from_numpy(video)

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

**Parameters:**

conditions (`List[LTXVideoCondition], *optional*`) : The list of frame-conditioning items for the video generation.

prompt (`str` or `List[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

height (`int`, *optional*, defaults to `512`) : The height in pixels of the generated image. This is set to 480 by default for the best results.

width (`int`, *optional*, defaults to `768`) : The width in pixels of the generated image. This is set to 848 by default for the best results.

num_frames (`int`, *optional*, defaults to `121`) : The number of video frames to generate

frame_rate (`float`, *optional*, defaults to `24.0`) : The frames per second (FPS) of the generated video.

num_inference_steps (`int`, *optional*, defaults to 40) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`List[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to `4.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is a separate value `audio_guidance_scale` for the audio modality).

stg_scale (`float`, *optional*, defaults to `0.0`) : Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate where we move the sample away from a weak sample from a perturbed version of the denoising model. Enabling STG will result in an additional denoising model forward pass; the default value of `0.0` means that STG is disabled.

modality_scale (`float`, *optional*, defaults to `1.0`) : Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio) cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an additional denoising model forward pass; the default value of `1.0` means that modality guidance is disabled.

guidance_rescale (`float`, *optional*, defaults to 0.0) : Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when using zero terminal SNR. Used for the video modality.

audio_guidance_scale (`float`, *optional* defaults to `None`) : Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value `guidance_scale`.

audio_stg_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the video value `stg_scale`.

audio_modality_scale (`float`, *optional*, defaults to `None`) : Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and audio. If `None`, defaults to the video value `modality_scale`.

audio_guidance_rescale (`float`, *optional*, defaults to `None`) : A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value `guidance_rescale`.

spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`) : The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0 and `[28]` is recommended for LTX-2.3.

noise_scale (`float`, *optional*, defaults to `None`) : The interpolation factor between random noise and denoised latents at each timestep. Applying noise to the `latents` and `audio_latents` before continue denoising. If not set, will be inferred from the sigma schedule.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `List[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

audio_latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.FloatTensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

use_cross_timestep (`bool` *optional*, defaults to `False`) : Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior; `False` is the legacy LTX-2.0 behavior.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTX2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `1024`) : Maximum sequence length to use with the `prompt`.

**Returns:**

``~pipelines.ltx.LTX2PipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTX2PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### apply_first_frame_conditioning[[diffusers.LTX2ConditionPipeline.apply_first_frame_conditioning]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L793)

Apply first-frame visual conditioning by overwriting tokens at the first-frame positions.

Only conditions with `latent_idx == 0` are applied here (matching `VideoConditionByLatentIndex` in the
reference implementation). Conditions at non-zero latent indices are appended as separate keyframe tokens via
`prepare_keyframe_extras` (matching `VideoConditionByKeyframeIndex`) and are skipped here.

**Parameters:**

latents (`torch.Tensor`) : Initial packed (patchified) latents of shape [batch_size, patch_seq_len, hidden_dim].

conditioning_mask (`torch.Tensor`) : Initial packed (patchified) conditioning mask of shape [batch_size, patch_seq_len, 1] with values in [0, 1] where 0 means the denoising model output will be fully used and 1 means the condition will be fully used.

**Returns:**

``Tuple[torch.Tensor, torch.Tensor, torch.Tensor]``

Returns a 3-tuple of tensors where:
1. The packed video latents with first-frame conditions applied.
2. The packed conditioning mask with first-frame strengths applied.
3. The clean conditioning latents at first-frame positions (zeros elsewhere).
#### encode_prompt[[diffusers.LTX2ConditionPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L381)

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
#### prepare_latents[[diffusers.LTX2ConditionPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L900)

Prepare noisy video latents, applying frame conditions.

First-frame conditions (`latent_idx == 0`) are applied by overwriting tokens at the first-frame positions
(`VideoConditionByLatentIndex` semantics). Non-first-frame conditions (`latent_idx > 0`) are concatenated onto
the main latent sequence with per-token `conditioning_mask = strength` (`VideoConditionByKeyframeIndex`
semantics) — the denoising loop's existing timestep formula `t * (1 - conditioning_mask)` and post-process
blend `denoised * (1 - conditioning_mask) + clean * conditioning_mask` then drive them across steps.

Returns a 4-tuple:
- `latents`: packed noisy latents (base tokens + any keyframe tokens cat'd onto the sequence dim).
- `conditioning_mask`: packed conditioning mask with values in `[0, 1]` — `1` at first-frame positions,
  `strength` at keyframe positions, `0` elsewhere.
- `clean_latents`: clean condition values at conditioned positions (zeros elsewhere); same shape as
  `latents`.
- `keyframe_coords`: `[B, 3, num_keyframe_patches, 2]` positional coordinates to append to `video_coords`,
  or `None` if there are no non-first-frame conditions.
#### preprocess_conditions[[diffusers.LTX2ConditionPipeline.preprocess_conditions]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L686)

Preprocesses the condition images/videos to torch tensors.

**Parameters:**

conditions (`LTX2VideoCondition` or `List[LTX2VideoCondition]`, *optional*, defaults to `None`) : A list of image/video condition instances.

height (`int`, *optional*, defaults to `512`) : The desired height in pixels.

width (`int`, *optional*, defaults to `768`) : The desired width in pixels.

num_frames (`int`, *optional*, defaults to `121`) : The desired number of frames in the generated video.

device (`torch.device`, *optional*, defaults to `None`) : The device on which to put the preprocessed image/video tensors.

**Returns:**

``Tuple[List[torch.Tensor], List[float], List[int], List[int]]``

Returns a 4-tuple of lists of length `len(conditions)` as follows:
1. The first list is a list of preprocessed video tensors of shape [batch_size=1, num_channels,
   num_frames, height, width].
2. The second list is a list of conditioning strengths.
3. The third list is a list of latent-space indices for each condition.
4. The fourth list is a list of (trimmed) pixel-space frame counts per condition. This is needed
   for keyframe coord semantics (single-pixel-frame keyframes have a clamped temporal extent).
#### trim_conditioning_sequence[[diffusers.LTX2ConditionPipeline.trim_conditioning_sequence]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py#L669)

Trim a conditioning sequence to the allowed number of frames.

**Parameters:**

start_frame (int) : The target frame number of the first frame in the sequence.

sequence_num_frames (int) : The number of frames in the sequence.

target_num_frames (int) : The target number of frames in the generated video.

**Returns:**

`int`

updated sequence length

## LTX2LatentUpsamplePipeline[[diffusers.LTX2LatentUpsamplePipeline]]

#### diffusers.LTX2LatentUpsamplePipeline[[diffusers.LTX2LatentUpsamplePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L104)

__call__diffusers.LTX2LatentUpsamplePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L264[{"name": "video", "val": ": list[PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 768"}, {"name": "num_frames", "val": ": int = 121"}, {"name": "spatial_patch_size", "val": ": int = 1"}, {"name": "temporal_patch_size", "val": ": int = 1"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "latents_normalized", "val": ": bool = False"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "adain_factor", "val": ": float = 0.0"}, {"name": "tone_map_compression_ratio", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}]- **video** (`list[PipelineImageInput]`, *optional*) --
  The video to be upsampled (such as a LTX 2.0 first stage output). If not supplied, `latents` should be
  supplied.
- **height** (`int`, *optional*, defaults to `512`) --
  The height in pixels of the input video (not the generated video, which will have a larger resolution).
- **width** (`int`, *optional*, defaults to `768`) --
  The width in pixels of the input video (not the generated video, which will have a larger resolution).
- **num_frames** (`int`, *optional*, defaults to `121`) --
  The number of frames in the input video.
- **spatial_patch_size** (`int`, *optional*, defaults to `1`) --
  The spatial patch size of the video latents. Used when `latents` is supplied if unpacking is necessary.
- **temporal_patch_size** (`int`, *optional*, defaults to `1`) --
  The temporal patch size of the video latents. Used when `latents` is supplied if unpacking is
  necessary.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated video latents. This can be supplied in place of the `video` argument. Can either be a
  patch sequence of shape `(batch_size, seq_len, hidden_dim)` or a video latent of shape `(batch_size,
  latent_channels, latent_frames, latent_height, latent_width)`.
- **latents_normalized** (`bool`, *optional*, defaults to `False`) --
  If `latents` are supplied, whether the `latents` are normalized using the VAE latent mean and std. If
  `True`, the `latents` will be denormalized before being supplied to the latent upsampler.
- **decode_timestep** (`float`, defaults to `0.0`) --
  The timestep at which generated video is decoded.
- **decode_noise_scale** (`float`, defaults to `None`) --
  The interpolation factor between random noise and denoised latents at the decode timestep.
- **adain_factor** (`float`, *optional*, defaults to `0.0`) --
  Adaptive Instance Normalization (AdaIN) blending factor between the upsampled and original latents.
  Should be in [-10.0, 10.0]; supplying 0.0 (the default) means that AdaIN is not performed.
- **tone_map_compression_ratio** (`float`, *optional*, defaults to `0.0`) --
  The compression strength for tone mapping, which will reduce the dynamic range of the latent values.
  This is useful for regularizing high-variance latents or for conditioning outputs during generation.
  Should be in [0, 1], where 0.0 (the default) means tone mapping is not applied and 1.0 corresponds to
  the full compression effect.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.0`~pipelines.ltx.LTXPipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is the upsampled video.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
>>> from diffusers.utils import encode_video
>>> from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
>>> from diffusers.utils import load_image

>>> pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
>>> pipe.enable_model_cpu_offload()

>>> image = load_image(
...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
... )
>>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> frame_rate = 24.0
>>> video, audio = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=121,
...     frame_rate=frame_rate,
...     num_inference_steps=40,
...     guidance_scale=4.0,
...     output_type="pil",
...     return_dict=False,
... )

>>> latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
...     "Lightricks/LTX-2", subfolder="latent_upsampler", torch_dtype=torch.bfloat16
... )
>>> upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
>>> upsample_pipe.vae.enable_tiling()
>>> upsample_pipe.to(device="cuda", dtype=torch.bfloat16)

>>> video = upsample_pipe(
...     video=video,
...     width=768,
...     height=512,
...     output_type="np",
...     return_dict=False,
... )[0]

>>> encode_video(
...     video[0],
...     fps=frame_rate,
...     audio=audio[0].float().cpu(),
...     audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
...     output_path="video.mp4",
... )
```

**Parameters:**

video (`list[PipelineImageInput]`, *optional*) : The video to be upsampled (such as a LTX 2.0 first stage output). If not supplied, `latents` should be supplied.

height (`int`, *optional*, defaults to `512`) : The height in pixels of the input video (not the generated video, which will have a larger resolution).

width (`int`, *optional*, defaults to `768`) : The width in pixels of the input video (not the generated video, which will have a larger resolution).

num_frames (`int`, *optional*, defaults to `121`) : The number of frames in the input video.

spatial_patch_size (`int`, *optional*, defaults to `1`) : The spatial patch size of the video latents. Used when `latents` is supplied if unpacking is necessary.

temporal_patch_size (`int`, *optional*, defaults to `1`) : The temporal patch size of the video latents. Used when `latents` is supplied if unpacking is necessary.

latents (`torch.Tensor`, *optional*) : Pre-generated video latents. This can be supplied in place of the `video` argument. Can either be a patch sequence of shape `(batch_size, seq_len, hidden_dim)` or a video latent of shape `(batch_size, latent_channels, latent_frames, latent_height, latent_width)`.

latents_normalized (`bool`, *optional*, defaults to `False`) : If `latents` are supplied, whether the `latents` are normalized using the VAE latent mean and std. If `True`, the `latents` will be denormalized before being supplied to the latent upsampler.

decode_timestep (`float`, defaults to `0.0`) : The timestep at which generated video is decoded.

decode_noise_scale (`float`, defaults to `None`) : The interpolation factor between random noise and denoised latents at the decode timestep.

adain_factor (`float`, *optional*, defaults to `0.0`) : Adaptive Instance Normalization (AdaIN) blending factor between the upsampled and original latents. Should be in [-10.0, 10.0]; supplying 0.0 (the default) means that AdaIN is not performed.

tone_map_compression_ratio (`float`, *optional*, defaults to `0.0`) : The compression strength for tone mapping, which will reduce the dynamic range of the latent values. This is useful for regularizing high-variance latents or for conditioning outputs during generation. Should be in [0, 1], where 0.0 (the default) means tone mapping is not applied and 1.0 corresponds to the full compression effect.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.

**Returns:**

``~pipelines.ltx.LTXPipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is the upsampled video.
#### adain_filter_latent[[diffusers.LTX2LatentUpsamplePipeline.adain_filter_latent]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L168)

Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
tensor.

**Parameters:**

latent (`torch.Tensor`) : Input latents to normalize

reference_latents (`torch.Tensor`) : The reference latents providing style statistics.

factor (`float`) : Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

**Returns:**

`torch.Tensor`

The transformed latent tensor
#### tone_map_latents[[diffusers.LTX2LatentUpsamplePipeline.tone_map_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py#L196)

Applies a non-linear tone-mapping function to latent values to reduce their dynamic range in a perceptually
smooth way using a sigmoid-based compression.

This is useful for regularizing high-variance latents or for conditioning outputs during generation, especially
when controlling dynamic behavior with a `compression` factor.

**Parameters:**

latents : torch.Tensor Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.

compression : float Compression strength in the range [0, 1]. - 0.0: No tone-mapping (identity transform) - 1.0: Full compression effect

**Returns:**

torch.Tensor
The tone-mapped latent tensor of the same shape as input.

## LTX2PipelineOutput[[diffusers.pipelines.ltx2.pipeline_output.LTX2PipelineOutput]]

#### diffusers.pipelines.ltx2.pipeline_output.LTX2PipelineOutput[[diffusers.pipelines.ltx2.pipeline_output.LTX2PipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx2/pipeline_output.py#L9)

Output class for LTX pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

audio (`torch.Tensor`, `np.ndarray`) : TODO

### Sana
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/sana.md

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
