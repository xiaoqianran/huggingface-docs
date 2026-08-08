# LongCat-AudioDiT

LongCat-AudioDiT is a text-to-audio diffusion model from Meituan LongCat. The diffusers integration exposes a standard [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) interface for text-conditioned audio generation.

This pipeline was adapted from the LongCat-AudioDiT reference implementation: https://github.com/meituan-longcat/LongCat-AudioDiT

This pipeline supports loading from a local directory or Hugging Face Hub repository in diffusers format (containing `text_encoder/`, `transformer/`, `vae/`, `tokenizer/`, and `scheduler/` subfolders).

## Usage

```py
import soundfile as sf
import torch
from diffusers import LongCatAudioDiTPipeline

pipeline = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

prompt = "A calm ocean wave ambience with soft wind in the background."
audio = pipeline(
    prompt,
    audio_duration_s=5.0,
    num_inference_steps=16,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
).audios[0, 0]

sf.write("longcat.wav", audio, pipeline.sample_rate)
```

## Tips

- `audio_duration_s` is the most direct way to control output duration.
- Use `generator=torch.Generator("cuda").manual_seed(42)` to make generation reproducible.
- Output shape is `(batch, channels, samples)` - use `.audios[0, 0]` to get a single audio sample.
- The pipeline outputs mono audio (1 channel). If you need stereo, you can duplicate the channel: `audio.unsqueeze(0).repeat(1, 2, 1)`.

## LongCatAudioDiTPipeline[[diffusers.LongCatAudioDiTPipeline]]

#### diffusers.LongCatAudioDiTPipeline[[diffusers.LongCatAudioDiTPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/longcat_audio_dit/pipeline_longcat_audio_dit.py#L99)

__call__diffusers.LongCatAudioDiTPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/longcat_audio_dit/pipeline_longcat_audio_dit.py#L219[{"name": "prompt", "val": ": str | list[str]"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "audio_duration_s", "val": ": float | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "num_inference_steps", "val": ": int = 16"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "output_type", "val": ": str = 'np'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}]- **prompt** (`str` or `list[str]`) -- Prompt or prompts that guide audio generation.
- **negative_prompt** (`str` or `list[str]`, *optional*) -- Negative prompt(s) for classifier-free guidance.
- **audio_duration_s** (`float`, *optional*) --
  Target audio duration in seconds. Ignored when `latents` is provided.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents of shape `(batch_size, duration, latent_dim)`.
- **num_inference_steps** (`int`, defaults to 16) -- Number of denoising steps.
- **guidance_scale** (`float`, defaults to 4.0) -- Guidance scale for classifier-free guidance.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) -- Random generator(s).
- **output_type** (`str`, defaults to `"np"`) -- Output format: `"np"`, `"pt"`, or `"latent"`.
- **return_dict** (`bool`, defaults to `True`) -- Whether to return `AudioPipelineOutput`.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function called at the end of each denoising step with the pipeline, step index, timestep, and tensor
  inputs specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, defaults to `["latents"]`) --
  Tensor inputs passed to `callback_on_step_end`.0

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import soundfile as sf
>>> import torch
>>> from diffusers import LongCatAudioDiTPipeline

>>> pipe = LongCatAudioDiTPipeline.from_pretrained("ruixiangma/LongCat-AudioDiT-1B-Diffusers")
>>> pipe.to("cuda")

>>> prompt = "A calm ocean wave ambience with soft wind in the background."
>>> audio = pipe(
...     prompt,
...     audio_duration_s=5.0,
...     num_inference_steps=20,
...     guidance_scale=4.0,
...     generator=torch.Generator("cuda").manual_seed(42),
... ).audios[0, 0]
>>> sf.write("output.wav", audio, pipe.sample_rate)
```

**Parameters:**

prompt (`str` or `list[str]`) : Prompt or prompts that guide audio generation.

negative_prompt (`str` or `list[str]`, *optional*) : Negative prompt(s) for classifier-free guidance.

audio_duration_s (`float`, *optional*) : Target audio duration in seconds. Ignored when `latents` is provided.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents of shape `(batch_size, duration, latent_dim)`.

num_inference_steps (`int`, defaults to 16) : Number of denoising steps.

guidance_scale (`float`, defaults to 4.0) : Guidance scale for classifier-free guidance.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : Random generator(s).

output_type (`str`, defaults to `"np"`) : Output format: `"np"`, `"pt"`, or `"latent"`.

return_dict (`bool`, defaults to `True`) : Whether to return `AudioPipelineOutput`.

callback_on_step_end (`Callable`, *optional*) : A function called at the end of each denoising step with the pipeline, step index, timestep, and tensor inputs specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, defaults to `["latents"]`) : Tensor inputs passed to `callback_on_step_end`.
#### from_pretrained[[diffusers.LongCatAudioDiTPipeline.from_pretrained]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/pipeline_utils.py#L617)

Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

The pipeline is set in evaluation mode (`model.eval()`) by default.

If you get the error message below, you need to finetune the weights for your downstream task:

```
Some weights of UNet2DConditionModel were not initialized from the model checkpoint at stable-diffusion-v1-5/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
- conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

> [!TIP] > To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in
with `hf > auth login`.

Examples:

```py
>>> from diffusers import DiffusionPipeline

>>> # Download pipeline from huggingface.co and cache.
>>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

>>> # Download pipeline that requires an authorization token
>>> # For more information on access tokens, please refer to this section
>>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
>>> pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

>>> # Use a different scheduler
>>> from diffusers import LMSDiscreteScheduler

>>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
>>> pipeline.scheduler = scheduler
```

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*) : Can be either:  - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline hosted on the Hub. - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights saved using [save_pretrained()](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.save_pretrained). - A path to a *directory* (for example `./my_pipeline_directory/`) containing a dduf file

torch_dtype (`torch.dtype` or `dict[str, Union[str, torch.dtype]]`, *optional*) : Override the default `torch.dtype` and load the model with another dtype. To load submodels with different dtype pass a `dict` (for example `{'transformer': torch.bfloat16, 'vae': torch.float16}`). Set the default dtype for unspecified components with `default` (for example `{'transformer': torch.bfloat16, 'default': torch.float16}`). If a component is not specified and no default is set, `torch.float32` is used.

custom_pipeline (`str`, *optional*) : > [!WARNING] > 🧪 This is an experimental feature and may change in the future.  Can be either:  - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines the custom pipeline. - A string, the *file name* of a community pipeline hosted on GitHub under [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file names must match the file name and not the pipeline script (`clip_guided_stable_diffusion` instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the current main branch of GitHub. - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory must contain a file called `pipeline.py` that defines the custom pipeline.  For more information on how to load and create custom pipelines, please have a look at [Loading and Adding Custom Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.

cache_dir (`Union[str, os.PathLike]`, *optional*) : Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used. 

proxies (`Dict[str, str]`, *optional*) : A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.

output_loading_info(`bool`, *optional*, defaults to `False`) : Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.

local_files_only (`bool`, *optional*, defaults to `False`) : Whether to only load local model weights and configuration files or not. If set to `True`, the model won't be downloaded from the Hub.

token (`str` or *bool*, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from `diffusers-cli login` (stored in `~/.huggingface`) is used.

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier allowed by Git.

custom_revision (`str`, *optional*) : The specific model version to use. It can be a branch name, a tag name, or a commit id similar to `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers version.

mirror (`str`, *optional*) : Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not guarantee the timeliness or safety of the source, and you should refer to the mirror site for more information.

device_map (`str`, *optional*) : Strategy that dictates how the different components of a pipeline should be placed on available devices. Currently, only "balanced" `device_map` is supported. Check out [this](https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement) to know more.

max_memory (`Dict`, *optional*) : A dictionary device identifier for the maximum memory. Will default to the maximum memory available for each GPU and the available CPU RAM if unset.

offload_folder (`str` or `os.PathLike`, *optional*) : The path to offload weights if device_map contains the value `"disk"`.

offload_state_dict (`bool`, *optional*) : If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True` when there is some disk offload.

low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`) : Speed up model loading only loading the pretrained weights and not initializing the weights. This also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this argument to `True` will raise an error.

use_safetensors (`bool`, *optional*, defaults to `None`) : If set to `None`, the safetensors weights are downloaded if they're available **and** if the safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors weights. If set to `False`, safetensors weights are not loaded.

use_onnx (`bool`, *optional*, defaults to `None`) : If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending with `.onnx` and `.pb`.

kwargs (remaining dictionary of keyword arguments, *optional*) : Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline class). The overwritten components are passed directly to the pipelines `__init__` method. See example below for more information.

variant (`str`, *optional*) : Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when loading `from_flax`.

dduf_file(`str`, *optional*) : Load weights from the specified dduf file.

disable_mmap ('bool', *optional*, defaults to 'False') : Whether to disable mmap when loading a Safetensors model. This option can perform better when the model is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

### Cogview3
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/cogview3.md

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

# CogView3Plus

[CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion](https://huggingface.co/papers/2403.05121) from Tsinghua University & ZhipuAI, by Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, Jie Tang.

The abstract from the paper is:

*Recent advancements in text-to-image generative systems have been largely driven by diffusion models. However, single-stage text-to-image diffusion models still face challenges, in terms of computational efficiency and the refinement of image details. To tackle the issue, we propose CogView3, an innovative cascaded framework that enhances the performance of text-to-image diffusion. CogView3 is the first model implementing relay diffusion in the realm of text-to-image generation, executing the task by first creating low-resolution images and subsequently applying relay-based super-resolution. This methodology not only results in competitive text-to-image outputs but also greatly reduces both training and inference costs. Our experimental results demonstrate that CogView3 outperforms SDXL, the current state-of-the-art open-source text-to-image diffusion model, by 77.0% in human evaluations, all while requiring only about 1/2 of the inference time. The distilled variant of CogView3 achieves comparable performance while only utilizing 1/10 of the inference time by SDXL.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

## CogView3PlusPipeline[[diffusers.CogView3PlusPipeline]]

#### diffusers.CogView3PlusPipeline[[diffusers.CogView3PlusPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py#L118)

Pipeline for text-to-image generation using CogView3Plus.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.CogView3PlusPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py#L407[{"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "guidance_scale", "val": ": float = 5.0"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "original_size", "val": ": tuple[int, int] | None = None"}, {"name": "crops_coords_top_left", "val": ": tuple = (0, 0)"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 224"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **height** (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. If not provided, it is set to 1024.
- **width** (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. If not provided it is set to 1024.
- **num_inference_steps** (`int`, *optional*, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **guidance_scale** (`float`, *optional*, defaults to `5.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **num_images_per_prompt** (`int`, *optional*, defaults to `1`) --
  The number of images to generate per prompt.
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
- **original_size** (`tuple[int]`, *optional*, defaults to (1024, 1024)) --
  If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
  `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
  explained in section 2.2 of
  [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
- **crops_coords_top_left** (`tuple[int]`, *optional*, defaults to (0, 0)) --
  `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
  `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
  `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
  [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int`, defaults to `224`) --
  Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.0[CogView3PipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogview3#diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput) or `tuple`[CogView3PipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogview3#diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import CogView3PlusPipeline

>>> pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A photo of an astronaut riding a horse on mars"
>>> image = pipe(prompt).images[0]
>>> image.save("output.png")
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : Frozen text-encoder. CogView3Plus uses [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.

tokenizer (`T5Tokenizer`) : Tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).

transformer ([CogView3PlusTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/cogview3plus_transformer2d#diffusers.CogView3PlusTransformer2DModel)) : A text conditioned `CogView3PlusTransformer2DModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

`[CogView3PipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogview3#diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput) or `tuple``

[CogView3PipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/cogview3#diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput) if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.CogView3PlusPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py#L210)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

max_sequence_length (`int`, defaults to `224`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

## CogView3PipelineOutput[[diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput]]

#### diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput[[diffusers.pipelines.cogview3.pipeline_output.CogView3PipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview3/pipeline_output.py#L10)

Output class for CogView3 pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Prx
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/prx.md

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

# PRX

PRX generates high-quality images from text using a simplified MMDIT architecture where text tokens don't update through transformer blocks. It employs flow matching with discrete scheduling for efficient sampling and uses Google's T5Gemma-2B-2B-UL2 model for multi-language text encoding. The ~1.3B parameter transformer delivers fast inference without sacrificing quality. You can choose between Flux VAE (8x compression, 16 latent channels) for balanced quality and speed or DC-AE (32x compression, 32 latent channels) for latent compression and faster processing.

## Available models

PRX offers multiple variants with different VAE configurations, each optimized for specific resolutions. Base models excel with detailed prompts, capturing complex compositions and subtle details. Fine-tuned models trained on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) improve aesthetic quality, especially with simpler prompts.

| Model | Resolution | Fine-tuned | Distilled | Description | Suggested prompts | Suggested parameters | Recommended dtype |
|:-----:|:-----------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [`Photoroom/prx-256-t2i`](https://huggingface.co/Photoroom/prx-256-t2i)| 256 | No | No | Base model pre-trained at 256 with Flux VAE|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-256-t2i-sft`](https://huggingface.co/Photoroom/prx-256-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i`](https://huggingface.co/Photoroom/prx-512-t2i)| 512 | No | No | Base model pre-trained at 512 with Flux VAE |Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-sft`](https://huggingface.co/Photoroom/prx-512-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/prx-512-t2i-sft`](https://huggingface.co/Photoroom/prx-512-t2i-sft) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae)| 512 | No | No | Base model pre-trained at 512 with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae)|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae-sft`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae) | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/prx-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft-distilled) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |s

Refer to [this](https://huggingface.co/collections/Photoroom/prx-models-68e66254c202ebfab99ad38e) collection for more information.

## Loading the pipeline

Load the pipeline with [from_pretrained()](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained).

```py
from diffusers.pipelines.prx import PRXPipeline

# Load pipeline - VAE and text encoder will be loaded from HuggingFace
pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("prx_output.png")
```

### Manual Component Loading

Load components individually to customize the pipeline for instance to use quantized models.

```py
import torch
from diffusers.pipelines.prx import PRXPipeline
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5GemmaModel, GemmaTokenizerFast
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
# Load transformer
transformer = PRXTransformer2DModel.from_pretrained(
    "checkpoints/prx-512-t2i-sft",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

# Load scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "checkpoints/prx-512-t2i-sft", subfolder="scheduler"
)

# Load T5Gemma text encoder
t5gemma_model = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2",
                                            quantization_config=quant_config,
                                            torch_dtype=torch.bfloat16)
text_encoder = t5gemma_model.encoder.to(dtype=torch.bfloat16)
tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")
tokenizer.model_max_length = 256

# Load VAE - choose either Flux VAE or DC-AE
# Flux VAE
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev",
                                    subfolder="vae",
                                    quantization_config=quant_config,
                                    torch_dtype=torch.bfloat16)

pipe = PRXPipeline(
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    vae=vae
)
pipe.to("cuda")
```

## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers.pipelines.prx import PRXPipeline

pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Offload components to CPU when not in use

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PRXPipeline[[diffusers.PRXPipeline]]

#### diffusers.PRXPipeline[[diffusers.PRXPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx.py#L257)

Pipeline for text-to-image generation using PRX Transformer.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.PRXPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx.py#L554[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str = ''"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 28"}, {"name": "timesteps", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 4.0"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.BoolTensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.BoolTensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "use_resolution_binning", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "tokenizer_max_length", "val": ": int | None = None"}, {"name": "skip_text_cleaning", "val": ": bool = False"}]- **prompt** (*str* or *list[str]*, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass *prompt_embeds*
  instead.
- **negative_prompt** (*str*, *optional*, defaults to *""*) --
  The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
  if *guidance_scale* is less than *1*).
- **height** (*int*, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image.
- **width** (*int*, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image.
- **num_inference_steps** (*int*, *optional*, defaults to 28) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **timesteps** (*list[int]*, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a *timesteps* argument
  in their *set_timesteps* method. If not defined, the default behavior when *num_inference_steps* is
  passed will be used. Must be in descending order.
- **guidance_scale** (*float*, *optional*, defaults to 4.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). *guidance_scale* is defined as *w* of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  *guidance_scale > 1*. Higher guidance scale encourages to generate images that are closely linked to
  the text *prompt*, usually at the expense of lower image quality.
- **num_images_per_prompt** (*int*, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (*torch.Generator* or *list[torch.Generator]*, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (*torch.Tensor*, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will be generated by sampling using the supplied random *generator*.
- **prompt_embeds** (*torch.FloatTensor*, *optional*) --
  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
  provided, text embeddings will be generated from *prompt* input argument.
- **negative_prompt_embeds** (*torch.FloatTensor*, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided and *guidance_scale > 1*, negative embeddings will be generated from an
  empty string.
- **prompt_attention_mask** (*torch.BoolTensor*, *optional*) --
  Pre-generated attention mask for *prompt_embeds*. If not provided, attention mask will be generated
  from *prompt* input argument.
- **negative_prompt_attention_mask** (*torch.BoolTensor*, *optional*) --
  Pre-generated attention mask for *negative_prompt_embeds*. If not provided and *guidance_scale > 1*,
  attention mask will be generated from an empty string.
- **output_type** (*str*, *optional*, defaults to *"pil"*) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): *PIL.Image.Image* or *np.array*.
- **tokenizer_max_length** (*int*, *optional*) --
  Override the maximum number of tokens used when tokenizing the prompt. Defaults to the tokenizer's own
  `model_max_length` when not set.
- **skip_text_cleaning** (*bool*, *optional*, defaults to *False*) --
  If *True*, uses only light prompt cleaning (fix encoding + unescape HTML) instead of the full DeepFloyd
  cleaning pipeline.
- **return_dict** (*bool*, *optional*, defaults to *True*) --
  Whether or not to return a [*~pipelines.prx.PRXPipelineOutput*] instead of a plain tuple.
- **use_resolution_binning** (*bool*, *optional*, defaults to *True*) --
  If set to *True*, the requested height and width are first mapped to the closest resolutions using
  predefined aspect ratio bins. After the produced latents are decoded into images, they are resized back
  to the requested resolution. Useful for generating non-square images at optimal resolutions.
- **callback_on_step_end** (*Callable*, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: *callback_on_step_end(self, step, timestep, callback_kwargs)*.
  *callback_kwargs* will include a list of all tensors as specified by
  *callback_on_step_end_tensor_inputs*.
- **callback_on_step_end_tensor_inputs** (*list*, *optional*) --
  The list of tensor inputs for the *callback_on_step_end* function. The tensors specified in the list
  will be passed as *callback_kwargs* argument. You will only be able to include tensors that are listed
  in the *._callback_tensor_inputs* attribute.0[*~pipelines.prx.PRXPipelineOutput*] or *tuple`[*~pipelines.prx.PRXPipelineOutput*] if *return_dict* is
True, otherwise a *tuple. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import PRXPipeline

>>> # Load pipeline with from_pretrained
>>> pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft")
>>> pipe.to("cuda")

>>> prompt = "A digital painting of a rusty, vintage tram on a sandy beach"
>>> image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
>>> image.save("prx_output.png")
```

**Parameters:**

transformer (`PRXTransformer2DModel`) : The PRX transformer model to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

text_encoder (`T5GemmaEncoder`) : Text encoder model for encoding prompts.

tokenizer ([`T5TokenizerFast` or `GemmaTokenizerFast`]) : Tokenizer for the text encoder.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL) or [AutoencoderDC](/docs/diffusers/v0.39.0/en/api/models/autoencoder_dc#diffusers.AutoencoderDC)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. Supports both AutoencoderKL (8x compression) and AutoencoderDC (32x compression).

**Returns:**

`[*~pipelines.prx.PRXPipelineOutput*] or *tuple``

[*~pipelines.prx.PRXPipelineOutput*] if *return_dict* is
True, otherwise a *tuple. When returning a tuple, the first element is a list with the generated images.
#### check_inputs[[diffusers.PRXPipeline.check_inputs]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx.py#L500)

Check that all inputs are in correct format.
#### encode_prompt[[diffusers.PRXPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx.py#L376)

Encode text prompt using standard text encoder and tokenizer, or use precomputed embeddings.
#### get_default_resolution[[diffusers.PRXPipeline.get_default_resolution]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx.py#L340)

Determine the default resolution based on the loaded VAE and config.

**Returns:**

`int`

The default sample size (height/width) to use for generation.
#### prepare_latents[[diffusers.PRXPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_prx.py#L352)

Prepare initial latents for the diffusion process.

## PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

#### diffusers.pipelines.prx.PRXPipelineOutput[[diffusers.pipelines.prx.PRXPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/prx/pipeline_output.py#L24)

Output class for PRX pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Chroma
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/chroma.md
