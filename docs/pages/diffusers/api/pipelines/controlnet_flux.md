# ControlNet with Flux.1

  

FluxControlNetPipeline is an implementation of ControlNet for Flux.1.

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, eg, edges, depth, segmentation, human pose, etc, with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (<50k) and large (>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models.*

This controlnet code is implemented by [The InstantX Team](https://huggingface.co/InstantX). You can find pre-trained checkpoints for Flux-ControlNet in the table below:

| ControlNet type | Developer | Link |
| -------- | ---------- | ---- |
| Canny | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny) |
| Depth | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth) |
| Union | [The InstantX Team](https://huggingface.co/InstantX) | [Link](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union) |

XLabs ControlNets are also supported, which was contributed by the [XLabs team](https://huggingface.co/XLabs-AI).

| ControlNet type | Developer | Link |
| -------- | ---------- | ---- |
| Canny | [The XLabs Team](https://huggingface.co/XLabs-AI) | [Link](https://huggingface.co/XLabs-AI/flux-controlnet-canny-diffusers) |
| Depth | [The XLabs Team](https://huggingface.co/XLabs-AI) | [Link](https://huggingface.co/XLabs-AI/flux-controlnet-depth-diffusers) |
| HED | [The XLabs Team](https://huggingface.co/XLabs-AI) | [Link](https://huggingface.co/XLabs-AI/flux-controlnet-hed-diffusers) |

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## FluxControlNetPipeline[[diffusers.FluxControlNetPipeline]]
#### diffusers.FluxControlNetPipeline[[diffusers.FluxControlNetPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L177)

The Flux pipeline for text-to-image generation.

Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

__call__diffusers.FluxControlNetPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L678[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "prompt_2", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt_2", "val": ": str | list[str] | None = None"}, {"name": "true_cfg_scale", "val": ": float = 1.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 28"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "control_guidance_start", "val": ": float | list[float] = 0.0"}, {"name": "control_guidance_end", "val": ": float | list[float] = 1.0"}, {"name": "control_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "control_mode", "val": ": int | list[int] | None = None"}, {"name": "controlnet_conditioning_scale", "val": ": float | list[float] = 1.0"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "pooled_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "ip_adapter_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "ip_adapter_image_embeds", "val": ": list[torch.Tensor] | None = None"}, {"name": "negative_ip_adapter_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "negative_ip_adapter_image_embeds", "val": ": list[torch.Tensor] | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_pooled_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "joint_attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **prompt_2** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
  will be used instead
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **negative_prompt_2** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
  `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
  `negative_prompt` is provided.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to 7.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **control_guidance_start** (`float` or `list[float]`, *optional*, defaults to 0.0) --
  The percentage of total steps at which the ControlNet starts applying.
- **control_guidance_end** (`float` or `list[float]`, *optional*, defaults to 1.0) --
  The percentage of total steps at which the ControlNet stops applying.
- **control_image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, `list[np.ndarray]`, --
  `list[list[torch.Tensor]]`, `list[list[np.ndarray]]` or `list[list[PIL.Image.Image]]`):
  The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
  specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
  as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
  width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
  images must be passed as a list such that each element of the list can be correctly batched for input
  to a single ControlNet.
- **controlnet_conditioning_scale** (`float` or `list[float]`, *optional*, defaults to 1.0) --
  The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
  to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
  the corresponding scale as a list.
- **control_mode** (`int` or `list[int]`,, *optional*, defaults to None) --
  The control mode when applying ControlNet-Union.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **pooled_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
  If not provided, pooled text embeddings will be generated from `prompt` input argument.
- **negative_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_pooled_prompt_embeds** (`torch.FloatTensor`, *optional*) --
  Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
  input argument.
- **ip_adapter_image** -- (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
- **ip_adapter_image_embeds** (`list[torch.Tensor]`, *optional*) --
  Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
  IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
  provided, embeddings are computed from the `ip_adapter_image` input argument.
- **negative_ip_adapter_image** --
  (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
- **negative_ip_adapter_image_embeds** (`list[torch.Tensor]`, *optional*) --
  Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
  IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
  provided, embeddings are computed from the `ip_adapter_image` input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.
- **joint_attention_kwargs** (`dict`, *optional*) --
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.flux.FluxPipelineOutput` or `tuple``~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers.utils import load_image
>>> from diffusers import FluxControlNetPipeline
>>> from diffusers import FluxControlNetModel

>>> base_model = "black-forest-labs/FLUX.1-dev"
>>> controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
>>> controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
>>> pipe = FluxControlNetPipeline.from_pretrained(
...     base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
>>> prompt = "A girl in city, 25 years old, cool, futuristic"
>>> image = pipe(
...     prompt,
...     control_image=control_image,
...     control_guidance_start=0.2,
...     control_guidance_end=0.8,
...     controlnet_conditioning_scale=1.0,
...     num_inference_steps=28,
...     guidance_scale=3.5,
... ).images[0]
>>> image.save("flux.png")
```

**Parameters:**

transformer ([FluxTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/flux_transformer#diffusers.FluxTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

**Returns:**

``~pipelines.flux.FluxPipelineOutput` or `tuple``

`~pipelines.flux.FluxPipelineOutput` if `return_dict`
is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
images.
#### encode_prompt[[diffusers.FluxControlNetPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py#L342)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is used in all text-encoders

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

clip_skip (`int`, *optional*) : Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.

lora_scale (`float`, *optional*) : A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.

## FluxPipelineOutput[[diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput]]
#### diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput[[diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/flux/pipeline_output.py#L11)

Output class for Flux image generation pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be passed to the decoder.

### Consisid
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/consisid.md

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

# ConsisID

  

[Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://huggingface.co/papers/2411.17440) from Peking University & University of Rochester & etc, by Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, Li Yuan.

The abstract from the paper is:

*Identity-preserving text-to-video (IPT2V) generation aims to create high-fidelity videos with consistent human identity. It is an important task in video generation but remains an open problem for generative models. This paper pushes the technical frontier of IPT2V in two directions that have not been resolved in the literature: (1) A tuning-free pipeline without tedious case-by-case finetuning, and (2) A frequency-aware heuristic identity-preserving Diffusion Transformer (DiT)-based control scheme. To achieve these goals, we propose **ConsisID**, a tuning-free DiT-based controllable IPT2V model to keep human-**id**entity **consis**tent in the generated video. Inspired by prior findings in frequency analysis of vision/diffusion transformers, it employs identity-control signals in the frequency domain, where facial features can be decomposed into low-frequency global features (e.g., profile, proportions) and high-frequency intrinsic features (e.g., identity markers that remain unaffected by pose changes). First, from a low-frequency perspective, we introduce a global facial extractor, which encodes the reference image and facial key points into a latent space, generating features enriched with low-frequency information. These features are then integrated into the shallow layers of the network to alleviate training challenges associated with DiT. Second, from a high-frequency perspective, we design a local facial extractor to capture high-frequency details and inject them into the transformer blocks, enhancing the model's ability to preserve fine-grained features. To leverage the frequency information for identity preservation, we propose a hierarchical training strategy, transforming a vanilla pre-trained video generation model into an IPT2V model. Extensive experiments demonstrate that our frequency-aware heuristic scheme provides an optimal control solution for DiT-based models. Thanks to this scheme, our **ConsisID** achieves excellent results in generating high-quality, identity-preserving videos, making strides towards more effective IPT2V. The model weight of ConsID is publicly available at https://github.com/PKU-YuanGroup/ConsisID.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [SHYuanBest](https://github.com/SHYuanBest). The original codebase can be found [here](https://github.com/PKU-YuanGroup/ConsisID). The original weights can be found under [hf.co/BestWishYsh](https://huggingface.co/BestWishYsh).

There are two official ConsisID checkpoints for identity-preserving text-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`BestWishYsh/ConsisID-preview`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |
| [`BestWishYsh/ConsisID-1.5`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |

### Memory optimization

ConsisID requires about 44 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/SHYuanBest/bc4207c36f454f9e969adbb50eaf8258) script.

| Feature (overlay the previous) | Max Memory Allocated | Max Memory Reserved |
| :----------------------------- | :------------------- | :------------------ |
| -                              | 37 GB                | 44 GB               |
| enable_model_cpu_offload       | 22 GB                | 25 GB               |
| enable_sequential_cpu_offload  | 16 GB                | 22 GB               |
| vae.enable_slicing             | 16 GB                | 22 GB               |
| vae.enable_tiling              | 5 GB                 | 7 GB                |

## Load Model Checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [from_pretrained()](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) method.

```python
# !pip install consisid_eva_clip insightface facexlib
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from huggingface_hub import snapshot_download

# Download ckpts
snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")

# Load face helper model to preprocess input face image
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)

# Load consisid base model
pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

## Identity-Preserving Text-to-Video

For identity-preserving text-to-video, pass a text prompt and an image contain clear face (e.g., preferably half-body or full-body). By default, ConsisID generates a 720x480 video for the best results.

```python
from diffusers.utils import export_to_video

prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_input.png?download=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, "cuda", torch.bfloat16, image, is_align_face=True)

video = pipe(image=image, prompt=prompt, num_inference_steps=50, guidance_scale=6.0, use_dynamic_cfg=False, id_vit_hidden=id_vit_hidden, id_cond=id_cond, kps_cond=face_kps, generator=torch.Generator("cuda").manual_seed(42))
export_to_video(video.frames[0], "output.mp4", fps=8)
```

  
    Face Image
    Video
    Description
  
  
    
    
    The video, in a beautifully crafted animated style, features a confident woman riding a horse through a lush forest clearing. Her expression is focused yet serene as she adjusts her wide-brimmed hat with a practiced hand. She wears a flowy bohemian dress, which moves gracefully with the rhythm of the horse, the fabric flowing fluidly in the animated motion. The dappled sunlight filters through the trees, casting soft, painterly patterns on the forest floor. Her posture is poised, showing both control and elegance as she guides the horse with ease. The animation's gentle, fluid style adds a dreamlike quality to the scene, with the woman’s calm demeanor and the peaceful surroundings evoking a sense of freedom and harmony.
  
  
    
    
    The video, in a captivating animated style, shows a woman standing in the center of a snowy forest, her eyes narrowed in concentration as she extends her hand forward. She is dressed in a deep blue cloak, her breath visible in the cold air, which is rendered with soft, ethereal strokes. A faint smile plays on her lips as she summons a wisp of ice magic, watching with focus as the surrounding trees and ground begin to shimmer and freeze, covered in delicate ice crystals. The animation’s fluid motion brings the magic to life, with the frost spreading outward in intricate, sparkling patterns. The environment is painted with soft, watercolor-like hues, enhancing the magical, dreamlike atmosphere. The overall mood is serene yet powerful, with the quiet winter air amplifying the delicate beauty of the frozen scene.
  
  
    
    
    The animation features a whimsical portrait of a balloon seller standing in a gentle breeze, captured with soft, hazy brushstrokes that evoke the feel of a serene spring day. His face is framed by a gentle smile, his eyes squinting slightly against the sun, while a few wisps of hair flutter in the wind. He is dressed in a light, pastel-colored shirt, and the balloons around him sway with the wind, adding a sense of playfulness to the scene. The background blurs softly, with hints of a vibrant market or park, enhancing the light-hearted, yet tender mood of the moment.
  
  
    
    
    The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.
  
  
    
    
    The video features a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose. The baby has a determined look on their face, with eyes wide and lips pursed in concentration, as if ready to take on a challenge. The setting appears playful, with colorful toys scattered around and a soft rug underfoot, while sunlight streams through a nearby window, highlighting the fluttering cape and adding to the impression of heroism. The overall atmosphere is lighthearted and fun, with the baby's expressions capturing a mix of innocence and an adorable attempt at bravery, as if truly ready to save the day.
  

## Resources

Learn more about ConsisID with the following resources.
- A [video](https://www.youtube.com/watch?v=PhlgC-bI5SQ) demonstrating ConsisID's main features.
- The research paper, [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://hf.co/papers/2411.17440) for more details.

## ConsisIDPipeline[[diffusers.ConsisIDPipeline]]

#### diffusers.ConsisIDPipeline[[diffusers.ConsisIDPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/consisid/pipeline_consisid.py#L250)

Pipeline for image-to-video generation using ConsisID.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.ConsisIDPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/consisid/pipeline_consisid.py#L661[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 480"}, {"name": "width", "val": ": int = 720"}, {"name": "num_frames", "val": ": int = 49"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "use_dynamic_cfg", "val": ": bool = False"}, {"name": "num_videos_per_prompt", "val": ": int = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 226"}, {"name": "id_vit_hidden", "val": ": torch.Tensor | None = None"}, {"name": "id_cond", "val": ": torch.Tensor | None = None"}, {"name": "kps_cond", "val": ": torch.Tensor | None = None"}]- **image** (`PipelineImageInput`) --
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
- **num_frames** (`int`, defaults to `49`) --
  Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
  contain 1 extra frame because ConsisID is conditioned with (num_seconds * fps + 1) frames where
  num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
  needs to be satisfied is that of divisibility mentioned above.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, *optional*, defaults to 6) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **use_dynamic_cfg** (`bool`, *optional*, defaults to `False`) --
  If True, dynamically adjusts the guidance scale during inference. This allows the model to use a
  progressive guidance scale, improving the balance between text-guided generation and image quality over
  the course of the inference steps. Typically, early inference steps use a higher guidance scale for
  more faithful image generation, while later steps reduce it for more diverse and natural results.
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
  `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.
- **id_vit_hidden** (`torch.Tensor | None`, *optional*) --
  The tensor representing the hidden features extracted from the face model, which are used to condition
  the local facial extractor. This is crucial for the model to obtain high-frequency information of the
  face. If not provided, the local facial extractor will not run normally.
- **id_cond** (`torch.Tensor | None`, *optional*) --
  The tensor representing the hidden features extracted from the clip model, which are used to condition
  the local facial extractor. This is crucial for the model to edit facial features If not provided, the
  local facial extractor will not run normally.
- **kps_cond** (`torch.Tensor | None`, *optional*) --
  A tensor that determines whether the global facial extractor use keypoint information for conditioning.
  If provided, this tensor controls whether facial keypoints such as eyes, nose, and mouth landmarks are
  used during the generation process. This helps ensure the model retains more facial low-frequency
  information.0[ConsisIDPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/consisid#diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput) or `tuple`[ConsisIDPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/consisid#diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import ConsisIDPipeline
>>> from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
>>> from diffusers.utils import export_to_video
>>> from huggingface_hub import snapshot_download

>>> snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")
>>> (
...     face_helper_1,
...     face_helper_2,
...     face_clip_model,
...     face_main_model,
...     eva_transform_mean,
...     eva_transform_std,
... ) = prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)
>>> pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> # ConsisID works well with long and well-described prompts. Make sure the face in the image is clearly visible (e.g., preferably half-body or full-body).
>>> prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
>>> image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_input.png?download=true"

>>> id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
...     face_helper_1,
...     face_clip_model,
...     face_helper_2,
...     eva_transform_mean,
...     eva_transform_std,
...     face_main_model,
...     "cuda",
...     torch.bfloat16,
...     image,
...     is_align_face=True,
... )

>>> video = pipe(
...     image=image,
...     prompt=prompt,
...     num_inference_steps=50,
...     guidance_scale=6.0,
...     use_dynamic_cfg=False,
...     id_vit_hidden=id_vit_hidden,
...     id_cond=id_cond,
...     kps_cond=face_kps,
...     generator=torch.Generator("cuda").manual_seed(42),
... )
>>> export_to_video(video.frames[0], "output.mp4", fps=8)
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. ConsisID uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([ConsisIDTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/consisid_transformer3d#diffusers.ConsisIDTransformer3DModel)) : A text conditioned `ConsisIDTransformer3DModel` to denoise the encoded video latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[ConsisIDPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/consisid#diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput) or `tuple``

[ConsisIDPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/consisid#diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.ConsisIDPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/consisid/pipeline_consisid.py#L355)

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

## ConsisIDPipelineOutput[[diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput]]

#### diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput[[diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/consisid/pipeline_output.py#L9)

Output class for ConsisID pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Text-to-Video Generation with AnimateDiff
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/animatediff.md
