# ControlNet

  

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide a depth map, the ControlNet model generates an image that'll preserve the spatial information from the depth map. It is a more flexible and accurate way to control the image generation process.

The abstract from the paper is:

*We present ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models. ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with "zero convolutions" (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning. We test various conditioning controls, eg, edges, depth, segmentation, human pose, etc, with Stable Diffusion, using single or multiple conditions, with or without prompts. We show that the training of ControlNets is robust with small (<50k) and large (>1m) datasets. Extensive results show that ControlNet may facilitate wider applications to control image diffusion models.*

This pipeline was contributed by [ishan24](https://huggingface.co/ishan24). ❤️
The original codebase can be found at [NVlabs/Sana](https://github.com/NVlabs/Sana), and you can find official ControlNet checkpoints on [Efficient-Large-Model's](https://huggingface.co/Efficient-Large-Model) Hub profile.

## SanaControlNetPipeline[[diffusers.SanaControlNetPipeline]]
#### diffusers.SanaControlNetPipeline[[diffusers.SanaControlNetPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L197)

Pipeline for text-to-image generation using [Sana](https://huggingface.co/papers/2410.10629).

__call__diffusers.SanaControlNetPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L776[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str = ''"}, {"name": "num_inference_steps", "val": ": int = 20"}, {"name": "timesteps", "val": ": list = None"}, {"name": "sigmas", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 4.5"}, {"name": "control_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "controlnet_conditioning_scale", "val": ": float | list[float] = 1.0"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "height", "val": ": int = 1024"}, {"name": "width", "val": ": int = 1024"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "clean_caption", "val": ": bool = False"}, {"name": "use_resolution_binning", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 300"}, {"name": "complex_human_instruction", "val": ": list = [\"Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: ']"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **num_inference_steps** (`int`, *optional*, defaults to 20) --
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
  Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
  provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
- **negative_prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Pre-generated attention mask for negative text embeddings.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion.IFPipelineOutput` instead of a plain tuple.
- **attention_kwargs** --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **clean_caption** (`bool`, *optional*, defaults to `True`) --
  Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
  be installed. If the dependencies are not installed, the embeddings will be created from the raw
  prompt.
- **use_resolution_binning** (`bool` defaults to `True`) --
  If set to `True`, the requested height and width are first mapped to the closest resolutions using
  `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
  the requested resolution. Useful for generating non-square images.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int` defaults to `300`) --
  Maximum sequence length to use with the `prompt`.
- **complex_human_instruction** (`list[str]`, *optional*) --
  Instructions for complex human attention:
  https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.0[SanaPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) or `tuple`If `return_dict` is `True`, [SanaPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaControlNetPipeline
>>> from diffusers.utils import load_image

>>> pipe = SanaControlNetPipeline.from_pretrained(
...     "ishan24/Sana_600M_1024px_ControlNetPlus_diffusers",
...     variant="fp16",
...     torch_dtype={"default": torch.bfloat16, "controlnet": torch.float16, "transformer": torch.float16},
...     device_map="balanced",
... )
>>> cond_image = load_image(
...     "https://huggingface.co/ishan24/Sana_600M_1024px_ControlNet_diffusers/resolve/main/hed_example.png"
... )
>>> prompt = 'a cat with a neon sign that says "Sana"'
>>> image = pipe(
...     prompt,
...     control_image=cond_image,
... ).images[0]
>>> image.save("output.png")
```

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_inference_steps (`int`, *optional*, defaults to 20) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, `list[np.ndarray]`, : `list[list[torch.Tensor]]`, `list[list[np.ndarray]]` or `list[list[PIL.Image.Image]]`): The ControlNet input condition to provide guidance to the `unet` for generation. If the type is specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`, images must be passed as a list such that each element of the list can be correctly batched for input to a single ControlNet.

controlnet_conditioning_scale (`float` or `list[float]`, *optional*, defaults to 1.0) : The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set the corresponding scale as a list.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

height (`int`, *optional*, defaults to self.unet.config.sample_size) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size) : The width in pixels of the generated image.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

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

**Returns:**

`[SanaPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) or `tuple``

If `return_dict` is `True`, [SanaPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_sprint#diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images
#### disable_vae_slicing[[diffusers.SanaControlNetPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L249)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.SanaControlNetPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L276)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.SanaControlNetPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L236)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.SanaControlNetPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L262)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.SanaControlNetPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_sana_controlnet.py#L349)

Encodes the prompt into text encoder hidden states.

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

## SanaPipelineOutput[[diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput]]
#### diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput[[diffusers.pipelines.sana.pipeline_output.SanaPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana/pipeline_output.py#L10)

Output class for Sana pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Easyanimate
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/easyanimate.md

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

# EasyAnimate
[EasyAnimate](https://github.com/aigc-apps/EasyAnimate) by Alibaba PAI.

The description from it's GitHub page:
*EasyAnimate is a pipeline based on the transformer architecture, designed for generating AI images and videos, and for training baseline models and Lora models for Diffusion Transformer. We support direct prediction from pre-trained EasyAnimate models, allowing for the generation of videos with various resolutions, approximately 6 seconds in length, at 8fps (EasyAnimateV5.1, 1 to 49 frames). Additionally, users can train their own baseline and Lora models for specific style transformations.*

This pipeline was contributed by [bubbliiiing](https://github.com/bubbliiiing). The original codebase can be found [here](https://huggingface.co/alibaba-pai). The original weights can be found under [hf.co/alibaba-pai](https://huggingface.co/alibaba-pai).

There are two official EasyAnimate checkpoints for text-to-video and video-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/EasyAnimateV5.1-12b-zh`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh) | torch.float16 |
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-InP`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | torch.float16 |

There is one official EasyAnimate checkpoints available for image-to-video and video-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-InP`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | torch.float16 |

There are two official EasyAnimate checkpoints available for control-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-Control`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control) | torch.float16 |
| [`alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera`](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera) | torch.float16 |

For the EasyAnimateV5.1 series:
- Text-to-video (T2V) and Image-to-video (I2V) works for multiple resolutions. The width and height can vary from 256 to 1024.
- Both T2V and I2V models support generation with 1~49 frames and work best at this value. Exporting videos at 8 FPS is recommended.

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [EasyAnimatePipeline](/docs/diffusers/v0.39.0/en/api/pipelines/easyanimate#diffusers.EasyAnimatePipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, EasyAnimateTransformer3DModel, EasyAnimatePipeline
from diffusers.utils import export_to_video

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = EasyAnimateTransformer3DModel.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = EasyAnimatePipeline.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh",
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A cat walks on the grass, realistic style."
negative_prompt = "bad detailed"
video = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_frames=49, num_inference_steps=30).frames[0]
export_to_video(video, "cat.mp4", fps=8)
```

## EasyAnimatePipeline[[diffusers.EasyAnimatePipeline]]

#### diffusers.EasyAnimatePipeline[[diffusers.EasyAnimatePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py#L186)

Pipeline for text-to-video generation using EasyAnimate.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

EasyAnimate uses one text encoder [qwen2 vl](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) in V5.1.

__call__diffusers.EasyAnimatePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py#L524[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "num_frames", "val": ": int | None = 49"}, {"name": "height", "val": ": int | None = 512"}, {"name": "width", "val": ": int | None = 512"}, {"name": "num_inference_steps", "val": ": int | None = 50"}, {"name": "guidance_scale", "val": ": float | None = 5.0"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "eta", "val": ": float | None = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "guidance_rescale", "val": ": float = 0.0"}]- **prompt** (`str` or `list[str]`, *optional*) --
  Text prompts to guide the image or video generation. If not provided, use `prompt_embeds` instead.
- **num_frames** (`int`, *optional*) --
  Length of the generated video (in frames).
- **height** (`int`, *optional*) --
  Height of the generated image in pixels.
- **width** (`int`, *optional*) --
  Width of the generated image in pixels.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  Number of denoising steps during generation. More steps generally yield higher quality images but slow
  down inference.
- **guidance_scale** (`float`, *optional*, defaults to 5.0) --
  Encourages the model to align outputs with prompts. A higher value may decrease image quality.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  Prompts indicating what to exclude in generation. If not specified, use `negative_prompt_embeds`.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  Number of images to generate for each prompt.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Applies to DDIM scheduling. Controlled by the eta parameter from the related literature.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A generator to ensure reproducibility in image generation.
- **latents** (`torch.Tensor`, *optional*) --
  Predefined latent tensors to condition generation.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Text embeddings for the prompts. Overrides prompt string inputs for more flexibility.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Embeddings for negative prompts. Overrides string inputs if defined.
- **prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Attention mask for the primary prompt embeddings.
- **negative_prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Attention mask for negative prompt embeddings.
- **output_type** (`str`, *optional*, defaults to "latent") --
  Format of the generated output, either as a PIL image or as a NumPy array.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  If `True`, returns a structured output. Otherwise returns a simple tuple.
- **callback_on_step_end** (`Callable`, *optional*) --
  Functions called at the end of each denoising step.
- **callback_on_step_end_tensor_inputs** (`list[str]`, *optional*) --
  Tensor names to be included in callback function calls.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Adjusts noise levels based on guidance scale.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process. If not defined, the scheduler's default schedule for
  `num_inference_steps` is used.0[StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple`If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images and the
second element is a list of `bool`s indicating whether the corresponding generated image contains
"not-safe-for-work" (nsfw) content.

Generates images or video using the EasyAnimate pipeline based on the provided prompts.

Examples:
```python
>>> import torch
>>> from diffusers import EasyAnimatePipeline
>>> from diffusers.utils import export_to_video

>>> # Models: "alibaba-pai/EasyAnimateV5.1-12b-zh"
>>> pipe = EasyAnimatePipeline.from_pretrained(
...     "alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers", torch_dtype=torch.float16
... ).to("cuda")
>>> prompt = (
...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
...     "atmosphere of this unique musical performance."
... )
>>> sample_size = (512, 512)
>>> video = pipe(
...     prompt=prompt,
...     guidance_scale=6,
...     negative_prompt="bad detailed",
...     height=sample_size[0],
...     width=sample_size[1],
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=8)
```

**Parameters:**

vae ([AutoencoderKLMagvit](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_magvit#diffusers.AutoencoderKLMagvit)) : Variational Auto-Encoder (VAE) Model to encode and decode video to and from latent representations.

text_encoder (`~transformers.Qwen2VLForConditionalGeneration`, `~transformers.BertModel` | None) : EasyAnimate uses [qwen2 vl](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) in V5.1.

tokenizer (`~transformers.Qwen2Tokenizer`, `~transformers.BertTokenizer` | None) : A `Qwen2Tokenizer` or `BertTokenizer` to tokenize text.

transformer ([EasyAnimateTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/easyanimate_transformer3d#diffusers.EasyAnimateTransformer3DModel)) : The EasyAnimate model designed by EasyAnimate Team.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with EasyAnimate to denoise the encoded image latents.

**Returns:**

`[StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple``

If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images and the
second element is a list of `bool`s indicating whether the corresponding generated image contains
"not-safe-for-work" (nsfw) content.
#### encode_prompt[[diffusers.EasyAnimatePipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py#L241)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

dtype (`torch.dtype`) : torch dtype

num_images_per_prompt (`int`) : number of images that should be generated per prompt

do_classifier_free_guidance (`bool`) : whether to use classifier free guidance or not

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Attention mask for the prompt. Required when `prompt_embeds` is passed directly.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.

max_sequence_length (`int`, *optional*) : maximum sequence length to use for the prompt.

## EasyAnimatePipelineOutput[[diffusers.pipelines.easyanimate.pipeline_output.EasyAnimatePipelineOutput]]

#### diffusers.pipelines.easyanimate.pipeline_output.EasyAnimatePipelineOutput[[diffusers.pipelines.easyanimate.pipeline_output.EasyAnimatePipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/easyanimate/pipeline_output.py#L9)

Output class for EasyAnimate pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Lumina2
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/lumina2.md

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
