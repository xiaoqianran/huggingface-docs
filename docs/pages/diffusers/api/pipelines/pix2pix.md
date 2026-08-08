# InstructPix2Pix

  

[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://huggingface.co/papers/2211.09800) is by Tim Brooks, Aleksander Holynski and Alexei A. Efros.

The abstract from the paper is:

*We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models -- a language model (GPT-3) and a text-to-image model (Stable Diffusion) -- to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions.*

You can find additional information about InstructPix2Pix on the [project page](https://www.timothybrooks.com/instruct-pix2pix), [original codebase](https://github.com/timothybrooks/instruct-pix2pix), and try it out in a [demo](https://huggingface.co/spaces/timbrooks/instruct-pix2pix).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## StableDiffusionInstructPix2PixPipeline[[diffusers.StableDiffusionInstructPix2PixPipeline]]
#### diffusers.StableDiffusionInstructPix2PixPipeline[[diffusers.StableDiffusionInstructPix2PixPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L83)

Pipeline for pixel-level image editing by following text instructions (based on Stable Diffusion).

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

The pipeline also inherits the following loading methods:
- [load_textual_inversion()](/docs/diffusers/v0.39.0/en/api/loaders/textual_inversion#diffusers.loaders.TextualInversionLoaderMixin.load_textual_inversion) for loading textual inversion embeddings
- [load_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_weights) for loading LoRA weights
- [save_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.save_lora_weights) for saving LoRA weights
- [load_ip_adapter()](/docs/diffusers/v0.39.0/en/api/loaders/ip_adapter#diffusers.loaders.IPAdapterMixin.load_ip_adapter) for loading IP Adapters

__call__diffusers.StableDiffusionInstructPix2PixPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L172[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "num_inference_steps", "val": ": int = 100"}, {"name": "guidance_scale", "val": ": float = 7.5"}, {"name": "image_guidance_scale", "val": ": float = 1.5"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "ip_adapter_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "ip_adapter_image_embeds", "val": ": list[torch.Tensor] | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "cross_attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "**kwargs", "val": ""}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
- **image** (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
  image latents as `image`, but if passing latents directly it is not encoded again.
- **num_inference_steps** (`int`, *optional*, defaults to 100) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **guidance_scale** (`float`, *optional*, defaults to 7.5) --
  A higher guidance scale value encourages the model to generate images closely linked to the text
  `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
- **image_guidance_scale** (`float`, *optional*, defaults to 1.5) --
  Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
  `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
  linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
  value of at least `1`.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide what to not include in image generation. If not defined, you need to
  pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale 0[StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple`If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images and the
second element is a list of `bool`s indicating whether the corresponding generated image contains
"not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:

```py
>>> import PIL
>>> import requests
>>> import torch
>>> from io import BytesIO

>>> from diffusers import StableDiffusionInstructPix2PixPipeline

>>> def download_image(url):
...     response = requests.get(url)
...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")

>>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

>>> image = download_image(img_url).resize((512, 512))

>>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
... )
>>> pipe = pipe.to("cuda")

>>> prompt = "make the mountains snowy"
>>> image = pipe(prompt=prompt, image=image).images[0]
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.

text_encoder ([CLIPTextModel](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/clip#transformers.CLIPTextModel)) : Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).

tokenizer ([CLIPTokenizer](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/clip#transformers.CLIPTokenizer)) : A `CLIPTokenizer` to tokenize text.

unet ([UNet2DConditionModel](/docs/diffusers/v0.39.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : A `UNet2DConditionModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

safety_checker (`StableDiffusionSafetyChecker`) : Classification module that estimates whether generated images could be considered offensive or harmful. Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for more details about a model's potential harms.

feature_extractor ([CLIPImageProcessor](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/clip#transformers.CLIPImageProcessor)) : A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.

**Returns:**

`[StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple``

If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated images and the
second element is a list of `bool`s indicating whether the corresponding generated image contains
"not-safe-for-work" (nsfw) content.
#### load_textual_inversion[[diffusers.StableDiffusionInstructPix2PixPipeline.load_textual_inversion]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/loaders/textual_inversion.py#L271)

Load Textual Inversion embeddings into the text encoder of [StableDiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) (both 🤗 Diffusers and
Automatic1111 formats are supported).

Example:

To load a Textual Inversion embedding vector in 🤗 Diffusers format:

```py
from diffusers import StableDiffusionPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("sd-concepts-library/cat-toy")

prompt = "A  backpack"

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("cat-backpack.png")
```

To load a Textual Inversion embedding vector in Automatic1111 format, make sure to download the vector first
(for example from [civitAI](https://civitai.com/models/3036?modelVersionId=9857)) and then load the vector

locally:

```py
from diffusers import StableDiffusionPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")

prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("character.png")
```

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike` or `list[str or os.PathLike]` or `Dict` or `list[Dict]`) : Can be either one of the following or a list of them:  - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a pretrained model hosted on the Hub. - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual inversion weights. - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights. - A [torch state dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict). 

token (`str` or `list[str]`, *optional*) : Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a list, then `token` must also be a list of equal length.

text_encoder ([CLIPTextModel](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/clip#transformers.CLIPTextModel), *optional*) : Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)). If not specified, function will take self.tokenizer.

tokenizer ([CLIPTokenizer](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/clip#transformers.CLIPTokenizer), *optional*) : A `CLIPTokenizer` to tokenize text. If not specified, function will take self.tokenizer.

weight_name (`str`, *optional*) : Name of a custom weight file. This should be used when:  - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight name such as `text_inv.bin`. - The saved textual inversion file is in the Automatic1111 format.

cache_dir (`str | os.PathLike`, *optional*) : Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. 

proxies (`dict[str, str]`, *optional*) : A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.

local_files_only (`bool`, *optional*, defaults to `False`) : Whether to only load local model weights and configuration files or not. If set to `True`, the model won't be downloaded from the Hub.

hf_token (`str` or *bool*, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from `diffusers-cli login` (stored in `~/.huggingface`) is used.

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier allowed by Git.

subfolder (`str`, *optional*, defaults to `""`) : The subfolder location of a model file within a larger model repository on the Hub or locally.

mirror (`str`, *optional*) : Mirror source to resolve accessibility issues if you're downloading a model in China. We do not guarantee the timeliness or safety of the source, and you should refer to the mirror site for more information.
#### load_lora_weights[[diffusers.StableDiffusionInstructPix2PixPipeline.load_lora_weights]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/loaders/lora_pipeline.py#L146)

Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
`self.text_encoder`.

All kwargs are forwarded to `self.lora_state_dict`.

See [lora_state_dict()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.lora_state_dict) for more details on how the state dict is
loaded.

See [load_lora_into_unet()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet) for more details on how the state dict is
loaded into `self.unet`.

See [load_lora_into_text_encoder()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder) for more details on how the state
dict is loaded into `self.text_encoder`.

**Parameters:**

pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`) : See [lora_state_dict()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.lora_state_dict).

adapter_name (`str`, *optional*) : Adapter name to be used for referencing the loaded adapter model. If not specified, it will use `default_{i}` where i is the total number of adapters being loaded.

low_cpu_mem_usage (`bool`, *optional*) : Speed up model loading by only loading the pretrained LoRA weights and not initializing the random weights.

hotswap (`bool`, *optional*) : Defaults to `False`. Whether to substitute an existing (LoRA) adapter with the newly loaded adapter in-place. This means that, instead of loading an additional adapter, this will take the existing adapter weights and replace them with the weights of the new adapter. This can be faster and more memory efficient. However, the main advantage of hotswapping is that when the model is compiled with torch.compile, loading the new adapter does not require recompilation of the model. When using hotswapping, the passed `adapter_name` should be the name of an already loaded adapter.  If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need to call an additional method before loading the adapter:  ```py pipeline = ...  # load diffusers pipeline max_rank = ...  # the highest rank among all LoRAs that you want to load # call *before* compiling and loading the LoRA adapter pipeline.enable_lora_hotswap(target_rank=max_rank) pipeline.load_lora_weights(file_name) # optionally compile the model now ```  Note that hotswapping adapters of the text encoder is not yet supported. There are some further limitations to this technique, which are documented here: https://huggingface.co/docs/peft/main/en/package_reference/hotswap

kwargs (`dict`, *optional*) : See [lora_state_dict()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.lora_state_dict).
#### save_lora_weights[[diffusers.StableDiffusionInstructPix2PixPipeline.save_lora_weights]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/loaders/lora_pipeline.py#L477)

Save the LoRA parameters corresponding to the UNet and text encoder.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory to save LoRA parameters to. Will be created if it doesn't exist.

unet_lora_layers (`dict[str, torch.nn.Module]` or `dict[str, torch.Tensor]`) : State dict of the LoRA layers corresponding to the `unet`.

text_encoder_lora_layers (`dict[str, torch.nn.Module]` or `dict[str, torch.Tensor]`) : State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text encoder LoRA state dict because it comes from 🤗 Transformers.

is_main_process (`bool`, *optional*, defaults to `True`) : Whether the process calling this is the main process or not. Useful during distributed training and you need to call this function on all processes. In this case, set `is_main_process=True` only on the main process to avoid race conditions.

save_function (`Callable`) : The function to use to save the state dictionary. Useful during distributed training when you need to replace `torch.save` with another method. Can be configured with the environment variable `DIFFUSERS_SAVE_MODE`.

safe_serialization (`bool`, *optional*, defaults to `True`) : Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.

unet_lora_adapter_metadata : LoRA adapter metadata associated with the unet to be serialized with the state dict.

text_encoder_lora_adapter_metadata : LoRA adapter metadata associated with the text encoder to be serialized with the state dict.

## StableDiffusionXLInstructPix2PixPipeline[[diffusers.StableDiffusionXLInstructPix2PixPipeline]]
#### diffusers.StableDiffusionXLInstructPix2PixPipeline[[diffusers.StableDiffusionXLInstructPix2PixPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L113)

Pipeline for pixel-level image editing by following text instructions. Based on Stable Diffusion XL.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

The pipeline also inherits the following loading methods:
- [load_textual_inversion()](/docs/diffusers/v0.39.0/en/api/loaders/textual_inversion#diffusers.loaders.TextualInversionLoaderMixin.load_textual_inversion) for loading textual inversion embeddings
- [from_single_file()](/docs/diffusers/v0.39.0/en/api/loaders/single_file#diffusers.loaders.FromSingleFileMixin.from_single_file) for loading `.ckpt` files
- [load_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights) for loading LoRA weights
- [save_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights) for saving LoRA weights

__call__diffusers.StableDiffusionXLInstructPix2PixPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L595[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "prompt_2", "val": ": str | list[str] | None = None"}, {"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 100"}, {"name": "denoising_end", "val": ": float | None = None"}, {"name": "guidance_scale", "val": ": float = 5.0"}, {"name": "image_guidance_scale", "val": ": float = 1.5"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt_2", "val": ": str | list[str] | None = None"}, {"name": "num_images_per_prompt", "val": ": int | None = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "pooled_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_pooled_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback", "val": ": typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None"}, {"name": "callback_steps", "val": ": int = 1"}, {"name": "cross_attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "original_size", "val": ": tuple = None"}, {"name": "crops_coords_top_left", "val": ": tuple = (0, 0)"}, {"name": "target_size", "val": ": tuple = None"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **prompt_2** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
  used in both text-encoders
- **image** (`torch.Tensor` or `PIL.Image.Image` or `np.ndarray` or `list[torch.Tensor]` or `list[PIL.Image.Image]` or `list[np.ndarray]`) --
  The image(s) to modify with the pipeline.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **denoising_end** (`float`, *optional*) --
  When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
  completed before it is intentionally prematurely terminated. As a result, the returned sample will
  still retain a substantial amount of noise as determined by the discrete timesteps selected by the
  scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
  "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
  Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
- **guidance_scale** (`float`, *optional*, defaults to 5.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **image_guidance_scale** (`float`, *optional*, defaults to 1.5) --
  Image guidance scale is to push the generated image towards the initial image `image`. Image guidance
  scale is enabled by setting `image_guidance_scale > 1`. Higher image guidance scale encourages to
  generate images that are closely linked to the source image `image`, usually at the expense of lower
  image quality. This pipeline requires a value of at least `1`.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **negative_prompt_2** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
  `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **pooled_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
  If not provided, pooled text embeddings will be generated from `prompt` input argument.
- **negative_pooled_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
  input argument.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput` instead of a
  plain tuple.
- **callback** (`Callable`, *optional*) --
  A function that will be called every `callback_steps` steps during inference. The function will be
  called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
- **callback_steps** (`int`, *optional*, defaults to 1) --
  The frequency at which the `callback` function will be called. If not specified, the callback will be
  called at every step.
- **cross_attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR.
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
- **target_size** (`tuple[int]`, *optional*, defaults to (1024, 1024)) --
  For most cases, `target_size` should be set to the desired height and width of the generated image. If
  not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
  section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).0`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` or `tuple``~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import StableDiffusionXLInstructPix2PixPipeline
>>> from diffusers.utils import load_image

>>> resolution = 768
>>> image = load_image(
...     "https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
... ).resize((resolution, resolution))
>>> edit_instruction = "Turn sky into a cloudy one"

>>> pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
...     "diffusers/sdxl-instructpix2pix-768", torch_dtype=torch.float16
... ).to("cuda")

>>> edited_image = pipe(
...     prompt=edit_instruction,
...     image=image,
...     height=resolution,
...     width=resolution,
...     guidance_scale=3.0,
...     image_guidance_scale=1.5,
...     num_inference_steps=30,
... ).images[0]
>>> edited_image
```

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : Frozen text-encoder. Stable Diffusion XL uses the text portion of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (` CLIPTextModelWithProjection`) : Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection), specifically the [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`CLIPTokenizer`) : Second Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

unet ([UNet2DConditionModel](/docs/diffusers/v0.39.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`) : Whether the `unet` requires a aesthetic_score condition to be passed during inference. Also see the config of `stabilityai/stable-diffusion-xl-refiner-1-0`.

force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`) : Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of `stabilityai/stable-diffusion-xl-base-1-0`.

add_watermarker (`bool`, *optional*) : Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to watermark output images. If not defined, it will default to True if the package is installed, otherwise no watermarker will be used.

is_cosxl_edit (`bool`, *optional*) : When set the image latents are scaled.

**Returns:**

``~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` or `tuple``

`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.StableDiffusionXLInstructPix2PixPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L213)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is used in both text-encoders

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

do_classifier_free_guidance (`bool`) : whether to use classifier free guidance or not

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

negative_prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

negative_pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt` input argument.

lora_scale (`float`, *optional*) : A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.

### Qwenimage
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/qwenimage.md

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

# QwenImage

  

Qwen-Image from the Qwen team is an image generation foundation model in the Qwen series that achieves significant advances in complex text rendering and precise image editing. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.

Qwen-Image comes in the following variants:

| model type | model id |
|:----------:|:--------:|
| Qwen-Image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-Edit | [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| Qwen-Image-Edit Plus | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |

> [!TIP]
> See the [Caching](../../optimization/cache) guide to speed up inference by storing and reusing intermediate outputs.

## LoRA for faster inference

Use a LoRA from `lightx2v/Qwen-Image-Lightning` to speed up inference by reducing the
number of steps. Refer to the code snippet below:

Code

```py
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch 
import math

ckpt_id = "Qwen/Qwen-Image"

# From
# https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    ckpt_id, scheduler=scheduler, torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
)

prompt = "a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition."
negative_prompt = " "
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=8,
    true_cfg_scale=1.0,
    generator=torch.manual_seed(0),
).images[0]
image.save("qwen_fewsteps.png")
```

> [!TIP]
> The `guidance_scale` parameter in the pipeline is there to support future guidance-distilled models when they come up. Note that passing `guidance_scale` to the pipeline is ineffective. To enable classifier-free guidance, please pass `true_cfg_scale` and `negative_prompt` (even an empty negative prompt like " ") should enable classifier-free guidance computations.

## Multi-image reference with QwenImageEditPlusPipeline

With [QwenImageEditPlusPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/qwenimage#diffusers.QwenImageEditPlusPipeline), one can provide multiple images as input reference.

```py
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
).to("cuda")

image_1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/grumpy.jpg")
image_2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peng.png")
image = pipe(
    image=[image_1, image_2],
    prompt='''put the penguin and the cat at a game show called "Qwen Edit Plus Games"''',
    num_inference_steps=50
).images[0]
```

## Performance

### torch.compile

Using `torch.compile` on the transformer provides ~2.4x speedup (A100 80GB: 4.70s → 1.93s):

```python
import torch
from diffusers import QwenImagePipeline

pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer = torch.compile(pipe.transformer)

# First call triggers compilation (~7s overhead)
# Subsequent calls run at ~2.4x faster
image = pipe("a cat", num_inference_steps=50).images[0]
```

### Batched Inference with Variable-Length Prompts

When using classifier-free guidance (CFG) with prompts of different lengths, the pipeline properly handles padding through attention masking. This ensures padding tokens do not influence the generated output.

```python
# CFG with different prompt lengths works correctly
image = pipe(
    prompt="A cat",
    negative_prompt="blurry, low quality, distorted",
    true_cfg_scale=3.5,
    num_inference_steps=50,
).images[0]
```

For detailed benchmark scripts and results, see [this gist](https://gist.github.com/cdutr/bea337e4680268168550292d7819dc2f).

## QwenImagePipeline[[diffusers.QwenImagePipeline]]

#### diffusers.QwenImagePipeline[[diffusers.QwenImagePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L132)

The QwenImage pipeline for text-to-image generation.

__call__diffusers.QwenImagePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L461[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by
  setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to
  generate images that are closely linked to the text `prompt`, usually at the expense of lower image
  quality.
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
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImagePipeline

>>> pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, num_inference_steps=50).images[0]
>>> image.save("qwenimage.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### disable_vae_slicing[[diffusers.QwenImagePipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L369)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.QwenImagePipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L396)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.QwenImagePipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L356)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.QwenImagePipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L382)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.QwenImagePipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L225)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageImg2ImgPipeline[[diffusers.QwenImageImg2ImgPipeline]]

#### diffusers.QwenImageImg2ImgPipeline[[diffusers.QwenImageImg2ImgPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L134)

The QwenImage pipeline for text-to-image generation.

__call__diffusers.QwenImageImg2ImgPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L535[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "strength", "val": ": float = 0.6"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by
  setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to
  generate images that are closely linked to the text `prompt`, usually at the expense of lower image
  quality.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **strength** (`float`, *optional*, defaults to 1.0) --
  Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
  starting point and more noise is added the higher the `strength`. The number of denoising steps depends
  on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
  process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
  essentially ignores `image`.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImageImg2ImgPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageImg2ImgPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
>>> pipe = pipe.to("cuda")
>>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
>>> init_image = load_image(url).resize((1024, 1024))
>>> prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney"
>>> images = pipe(prompt=prompt, negative_prompt=" ", image=init_image, strength=0.95).images[0]
>>> images.save("qwenimage_img2img.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### disable_vae_slicing[[diffusers.QwenImageImg2ImgPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L418)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.QwenImageImg2ImgPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L445)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.QwenImageImg2ImgPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L405)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.QwenImageImg2ImgPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L431)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.QwenImageImg2ImgPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L268)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageInpaintPipeline[[diffusers.QwenImageInpaintPipeline]]

#### diffusers.QwenImageInpaintPipeline[[diffusers.QwenImageInpaintPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L137)

The QwenImage pipeline for text-to-image generation.

__call__diffusers.QwenImageInpaintPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L645[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "mask_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "masked_image_latents", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "padding_mask_crop", "val": ": int | None = None"}, {"name": "strength", "val": ": float = 0.6"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by
  setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to
  generate images that are closely linked to the text `prompt`, usually at the expense of lower image
  quality.
- **mask_image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
  are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
  single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
  color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
  H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
  1)`, or `(H, W)`.
- **masked_image_latents** (`torch.Tensor`, `list[torch.Tensor]`) --
  `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
  latents tensor will be generated by `mask_image`.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **padding_mask_crop** (`int`, *optional*, defaults to `None`) --
  The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
  image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region
  with the same aspect ration of the image and contains all masked area, and then expand that area based
  on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before
  resizing to the original image size for inpainting. This is useful when the masked area is small while
  the image is large and contain information irrelevant for inpainting, such as background.
- **strength** (`float`, *optional*, defaults to 1.0) --
  Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
  starting point and more noise is added the higher the `strength`. The number of denoising steps depends
  on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
  process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
  essentially ignores `image`.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import QwenImageInpaintPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageInpaintPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
>>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
>>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
>>> source = load_image(img_url)
>>> mask = load_image(mask_url)
>>> image = pipe(prompt=prompt, negative_prompt=" ", image=source, mask_image=mask, strength=0.85).images[0]
>>> image.save("qwenimage_inpainting.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### disable_vae_slicing[[diffusers.QwenImageInpaintPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L445)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.QwenImageInpaintPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L472)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.QwenImageInpaintPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L432)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.QwenImageInpaintPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L458)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.QwenImageInpaintPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_inpaint.py#L279)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageEditPipeline[[diffusers.QwenImageEditPipeline]]

#### diffusers.QwenImageEditPipeline[[diffusers.QwenImageEditPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L165)

The Qwen-Image-Edit pipeline for image editing.

__call__diffusers.QwenImageEditPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L557[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
  Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
  equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
  enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
  encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
  lower image quality.
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
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageEditPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGB")
>>> prompt = (
...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(image, prompt, num_inference_steps=50).images[0]
>>> image.save("qwenimage_edit.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### disable_vae_slicing[[diffusers.QwenImageEditPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L442)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.QwenImageEditPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L469)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.QwenImageEditPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L429)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.QwenImageEditPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L455)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.QwenImageEditPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py#L272)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

image (`torch.Tensor`, *optional*) : image to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageEditInpaintPipeline[[diffusers.QwenImageEditInpaintPipeline]]

#### diffusers.QwenImageEditInpaintPipeline[[diffusers.QwenImageEditInpaintPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L167)

The Qwen-Image-Edit pipeline for image editing.

__call__diffusers.QwenImageEditInpaintPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L690[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "mask_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "masked_image_latents", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "padding_mask_crop", "val": ": int | None = None"}, {"name": "strength", "val": ": float = 0.6"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
  Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
  equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
  enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
  encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
  lower image quality.
- **mask_image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
  are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
  single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
  color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
  H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
  1)`, or `(H, W)`.
- **masked_image_latents** (`torch.Tensor`, `list[torch.Tensor]`) --
  `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
  latents tensor will ge generated by `mask_image`.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **padding_mask_crop** (`int`, *optional*, defaults to `None`) --
  The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
  image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region
  with the same aspect ration of the image and contains all masked area, and then expand that area based
  on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before
  resizing to the original image size for inpainting. This is useful when the masked area is small while
  the image is large and contain information irrelevant for inpainting, such as background.
- **strength** (`float`, *optional*, defaults to 1.0) --
  Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
  starting point and more noise is added the higher the `strength`. The number of denoising steps depends
  on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
  process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
  essentially ignores `image`.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageEditInpaintPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditInpaintPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

>>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
>>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
>>> source = load_image(img_url)
>>> mask = load_image(mask_url)
>>> image = pipe(
...     prompt=prompt, negative_prompt=" ", image=source, mask_image=mask, strength=1.0, num_inference_steps=50
... ).images[0]
>>> image.save("qwenimage_inpainting.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### disable_vae_slicing[[diffusers.QwenImageEditInpaintPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L488)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.QwenImageEditInpaintPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L515)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.QwenImageEditInpaintPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L475)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.QwenImageEditInpaintPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L501)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.QwenImageEditInpaintPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py#L284)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

image (`torch.Tensor`, *optional*) : image to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageControlNetPipeline[[diffusers.QwenImageControlNetPipeline]]

#### diffusers.QwenImageControlNetPipeline[[diffusers.QwenImageControlNetPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L192)

The QwenImage pipeline for text-to-image generation.

__call__diffusers.QwenImageControlNetPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L564[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "control_guidance_start", "val": ": float | list[float] = 0.0"}, {"name": "control_guidance_end", "val": ": float | list[float] = 1.0"}, {"name": "control_image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "controlnet_conditioning_scale", "val": ": float | list[float] = 1.0"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is enabled by
  setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale encourages to
  generate images that are closely linked to the text `prompt`, usually at the expense of lower image
  quality.
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
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **control_guidance_start** (`float` or `list[float]`, *optional*, defaults to 0.0) --
  The percentage of total steps at which the ControlNet starts applying.
- **control_guidance_end** (`float` or `list[float]`, *optional*, defaults to 1.0) --
  The percentage of total steps at which the ControlNet stops applying.
- **control_image** (`PipelineImageInput`, *optional*) --
  The ControlNet input condition to provide guidance for the generation.
- **controlnet_conditioning_scale** (`float` or `list[float]`, *optional*, defaults to 1.0) --
  The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
  to the residual in the original `transformer`.
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers.utils import load_image
>>> from diffusers import QwenImageControlNetModel, QwenImageMultiControlNetModel, QwenImageControlNetPipeline

>>> # QwenImageControlNetModel
>>> controlnet = QwenImageControlNetModel.from_pretrained(
...     "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch.bfloat16
... )
>>> pipe = QwenImageControlNetPipeline.from_pretrained(
...     "Qwen/Qwen-Image", controlnet=controlnet, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation."
>>> negative_prompt = " "
>>> control_image = load_image(
...     "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/canny.png"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     control_image=control_image,
...     controlnet_conditioning_scale=1.0,
...     num_inference_steps=30,
...     true_cfg_scale=4.0,
... ).images[0]
>>> image.save("qwenimage_cn_union.png")

>>> # QwenImageMultiControlNetModel
>>> controlnet = QwenImageControlNetModel.from_pretrained(
...     "InstantX/Qwen-Image-ControlNet-Union", torch_dtype=torch.bfloat16
... )
>>> controlnet = QwenImageMultiControlNetModel([controlnet])
>>> pipe = QwenImageControlNetPipeline.from_pretrained(
...     "Qwen/Qwen-Image", controlnet=controlnet, torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")
>>> prompt = "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation."
>>> negative_prompt = " "
>>> control_image = load_image(
...     "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/conds/canny.png"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     control_image=[control_image, control_image],
...     controlnet_conditioning_scale=[0.5, 0.5],
...     num_inference_steps=30,
...     true_cfg_scale=4.0,
... ).images[0]
>>> image.save("qwenimage_cn_union_multi.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### disable_vae_slicing[[diffusers.QwenImageControlNetPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L436)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.QwenImageControlNetPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L463)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.QwenImageControlNetPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L423)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.QwenImageControlNetPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L449)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### encode_prompt[[diffusers.QwenImageControlNetPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet.py#L290)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageEditPlusPipeline[[diffusers.QwenImageEditPlusPipeline]]

#### diffusers.QwenImageEditPlusPipeline[[diffusers.QwenImageEditPlusPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L168)

The Qwen-Image-Edit pipeline for image editing.

__call__diffusers.QwenImageEditPlusPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L526[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}]- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
  Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
  equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
  enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
  encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
  lower image quality.
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
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageEditPlusPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGB")
>>> prompt = (
...     "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
... )
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(image, prompt, num_inference_steps=50).images[0]
>>> image.save("qwenimage_edit_plus.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.QwenImageEditPlusPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L286)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

image (`torch.Tensor`, *optional*) : image to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImageLayeredPipeline[[diffusers.QwenImageLayeredPipeline]]

#### diffusers.QwenImageLayeredPipeline[[diffusers.QwenImageLayeredPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py#L174)

The Qwen-Image-Layered pipeline for image decomposing.

__call__diffusers.QwenImageLayeredPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py#L537[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | None = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "true_cfg_scale", "val": ": float = 4.0"}, {"name": "layers", "val": ": int | None = 4"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float | None = None"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 512"}, {"name": "resolution", "val": ": int = 640"}, {"name": "cfg_normalize", "val": ": bool = False"}, {"name": "use_en_prompt", "val": ": bool = False"}]- **image** (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) --
  `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
  numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
  or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
  list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
  latents as `image`, but if passing latents directly it is not encoded again.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
  Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
  equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
  enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
  encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
  lower image quality.
- **layers** (`int`, *optional*, defaults to 4) --
  Number of latent layers to generate for the layered output.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to None) --
  A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
  where the guidance scale is applied during inference through noise prediction rescaling, guidance
  distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
  scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
  that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
  parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
  ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
  please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
  enable classifier-free guidance computations).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
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
- **prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
  argument.
- **negative_prompt_embeds_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.qwenimage.QwenImagePipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to 512) -- Maximum sequence length to use with the `prompt`.
- **resolution** (`int`, *optional*, defaults to 640) --
  using different bucket in (640, 1024) to determin the condition and output resolution
- **cfg_normalize** (`bool`, *optional*, defaults to `False`) --
  whether enable cfg normalization.
- **use_en_prompt** (`bool`, *optional*, defaults to `False`) --
  automatic caption language if user does not provide caption0`~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import QwenImageLayeredPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGBA")
>>> prompt = ""
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> images = pipe(
...     image,
...     prompt,
...     num_inference_steps=50,
...     true_cfg_scale=4.0,
...     layers=4,
...     resolution=640,
...     cfg_normalize=False,
...     use_en_prompt=True,
... ).images[0]
>>> for i, image in enumerate(images):
...     image.save(f"{i}.out.png")
```

**Parameters:**

transformer ([QwenImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/qwenimage_transformer2d#diffusers.QwenImageTransformer2DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2Tokenizer

**Returns:**

``~pipelines.qwenimage.QwenImagePipelineOutput` or `tuple``

`~pipelines.qwenimage.QwenImagePipelineOutput` if `return_dict` is True, otherwise a `tuple`. When
returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.QwenImageLayeredPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_layered.py#L291)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

## QwenImagePipelineOutput[[diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput]]

#### diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput[[diffusers.pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/qwenimage/pipeline_output.py#L10)

Output class for Stable Diffusion pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### PixArt-α
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/pixart.md
