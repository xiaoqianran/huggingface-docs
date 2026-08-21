# InstructPix2Pix

  

[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://huggingface.co/papers/2211.09800) is by Tim Brooks, Aleksander Holynski and Alexei A. Efros.

The abstract from the paper is:

*We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models -- a language model (GPT-3) and a text-to-image model (Stable Diffusion) -- to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions.*

You can find additional information about InstructPix2Pix on the [project page](https://www.timothybrooks.com/instruct-pix2pix), [original codebase](https://github.com/timothybrooks/instruct-pix2pix), and try it out in a [demo](https://huggingface.co/spaces/timbrooks/instruct-pix2pix).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## StableDiffusionInstructPix2PixPipeline[[diffusers.StableDiffusionInstructPix2PixPipeline]]

#### diffusers.StableDiffusionInstructPix2PixPipeline[[diffusers.StableDiffusionInstructPix2PixPipeline]]

```python
diffusers.StableDiffusionInstructPix2PixPipeline(vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessorPil, image_encoder: transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection | None = None, requires_safety_checker: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L83)

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.

text_encoder ([CLIPTextModel](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip#transformers.CLIPTextModel)) : Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).

tokenizer ([CLIPTokenizer](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip#transformers.CLIPTokenizer)) : A `CLIPTokenizer` to tokenize text.

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : A `UNet2DConditionModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

safety_checker (`StableDiffusionSafetyChecker`) : Classification module that estimates whether generated images could be considered offensive or harmful. Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for more details about a model's potential harms.

feature_extractor ([CLIPImageProcessor](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip#transformers.CLIPImageProcessor)) : A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.

Pipeline for pixel-level image editing by following text instructions (based on Stable Diffusion).

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

The pipeline also inherits the following loading methods:
- [load_textual_inversion()](/docs/diffusers/v0.40.0/en/api/loaders/textual_inversion#diffusers.loaders.TextualInversionLoaderMixin.load_textual_inversion) for loading textual inversion embeddings
- [load_lora_weights()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_weights) for loading LoRA weights
- [save_lora_weights()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.save_lora_weights) for saving LoRA weights
- [load_ip_adapter()](/docs/diffusers/v0.40.0/en/api/loaders/ip_adapter#diffusers.loaders.IPAdapterMixin.load_ip_adapter) for loading IP Adapters

#### __call__[[diffusers.StableDiffusionInstructPix2PixPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, num_inference_steps: int = 100, guidance_scale: float = 7.5, image_guidance_scale: float = 1.5, negative_prompt: str | list[str] | None = None, num_images_per_prompt: int | None = 1, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, ip_adapter_image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None, ip_adapter_image_embeds: list[torch.Tensor] | None = None, output_type: str | None = 'pil', return_dict: bool = True, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], cross_attention_kwargs: dict[str, typing.Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py#L172)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.

image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`) : `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept image latents as `image`, but if passing latents directly it is not encoded again.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 7.5) : A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.

image_guidance_scale (`float`, *optional*, defaults to 1.5) : Push the generated image towards the initial `image`. Image guidance scale is enabled by setting `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a value of at least `1`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide what to not include in image generation. If not defined, you need to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the [DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

ip_adapter_image : (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.

ip_adapter_image_embeds (`list[torch.Tensor]`, *optional*) : Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [StableDiffusionPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) instead of a plain tuple.

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

cross_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined in [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

**Returns:** [StableDiffusionPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple`

If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/image_variation#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
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

#### load_textual_inversion[[diffusers.StableDiffusionInstructPix2PixPipeline.load_textual_inversion]]

```python
load_textual_inversion(pretrained_model_name_or_path: str | list[str] | dict[str, torch.Tensor] | list[dict[str, torch.Tensor]], token: str | list[str] | None = None, tokenizer: 'PreTrainedTokenizer' | None = None, text_encoder: 'PreTrainedModel' | None = None, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/loaders/textual_inversion.py#L271)

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike` or `list[str or os.PathLike]` or `Dict` or `list[Dict]`) : Can be either one of the following or a list of them:  - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a pretrained model hosted on the Hub. - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual inversion weights. - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights. - A [torch state dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict). 

token (`str` or `list[str]`, *optional*) : Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a list, then `token` must also be a list of equal length.

text_encoder ([CLIPTextModel](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip#transformers.CLIPTextModel), *optional*) : Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)). If not specified, function will take self.tokenizer.

tokenizer ([CLIPTokenizer](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/clip#transformers.CLIPTokenizer), *optional*) : A `CLIPTokenizer` to tokenize text. If not specified, function will take self.tokenizer.

weight_name (`str`, *optional*) : Name of a custom weight file. This should be used when:  - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight name such as `text_inv.bin`. - The saved textual inversion file is in the Automatic1111 format.

cache_dir (`str | os.PathLike`, *optional*) : Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. 

proxies (`dict[str, str]`, *optional*) : A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.

local_files_only (`bool`, *optional*, defaults to `False`) : Whether to only load local model weights and configuration files or not. If set to `True`, the model won't be downloaded from the Hub.

hf_token (`str` or *bool*, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from `diffusers-cli login` (stored in `~/.huggingface`) is used.

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier allowed by Git.

subfolder (`str`, *optional*, defaults to `""`) : The subfolder location of a model file within a larger model repository on the Hub or locally.

mirror (`str`, *optional*) : Mirror source to resolve accessibility issues if you're downloading a model in China. We do not guarantee the timeliness or safety of the source, and you should refer to the mirror site for more information.

Load Textual Inversion embeddings into the text encoder of [StableDiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline) (both 🤗 Diffusers and
Automatic1111 formats are supported).

Example:

To load a Textual Inversion embedding vector in 🤗 Diffusers format:

```py
from diffusers import StableDiffusionPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("sd-concepts-library/cat-toy")

prompt = "A <cat-toy> backpack"

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

pipe.load_textual_inversion("./charturnerv2.pt", token="REDACTED")

prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

image = pipe(prompt, num_inference_steps=50).images[0]
image.save("character.png")
```

#### load_lora_weights[[diffusers.StableDiffusionInstructPix2PixPipeline.load_lora_weights]]

```python
load_lora_weights(pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor], adapter_name: str | None = None, hotswap: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/loaders/lora_pipeline.py#L143)

**Parameters:**

pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`) : See [lora_state_dict()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.lora_state_dict).

adapter_name (`str`, *optional*) : Adapter name to be used for referencing the loaded adapter model. If not specified, it will use `default_{i}` where i is the total number of adapters being loaded.

low_cpu_mem_usage (`bool`, *optional*) : Speed up model loading by only loading the pretrained LoRA weights and not initializing the random weights.

hotswap (`bool`, *optional*) : Defaults to `False`. Whether to substitute an existing (LoRA) adapter with the newly loaded adapter in-place. This means that, instead of loading an additional adapter, this will take the existing adapter weights and replace them with the weights of the new adapter. This can be faster and more memory efficient. However, the main advantage of hotswapping is that when the model is compiled with torch.compile, loading the new adapter does not require recompilation of the model. When using hotswapping, the passed `adapter_name` should be the name of an already loaded adapter.  If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need to call an additional method before loading the adapter:  ```py pipeline = ...  # load diffusers pipeline max_rank = ...  # the highest rank among all LoRAs that you want to load # call *before* compiling and loading the LoRA adapter pipeline.enable_lora_hotswap(target_rank=max_rank) pipeline.load_lora_weights(file_name) # optionally compile the model now ```  Note that hotswapping adapters of the text encoder is not yet supported. There are some further limitations to this technique, which are documented here: https://huggingface.co/docs/peft/main/en/package_reference/hotswap

kwargs (`dict`, *optional*) : See [lora_state_dict()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.lora_state_dict).

Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
`self.text_encoder`.

All kwargs are forwarded to `self.lora_state_dict`.

See [lora_state_dict()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.lora_state_dict) for more details on how the state dict is
loaded.

See [load_lora_into_unet()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet) for more details on how the state dict is
loaded into `self.unet`.

See [load_lora_into_text_encoder()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder) for more details on how the state
dict is loaded into `self.text_encoder`.

#### save_lora_weights[[diffusers.StableDiffusionInstructPix2PixPipeline.save_lora_weights]]

```python
save_lora_weights(save_directory: str | os.PathLike, unet_lora_layers: dict = None, text_encoder_lora_layers: dict = None, is_main_process: bool = True, weight_name: str = None, save_function: typing.Callable = None, safe_serialization: bool = True, unet_lora_adapter_metadata = None, text_encoder_lora_adapter_metadata = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/loaders/lora_pipeline.py#L461)

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory to save LoRA parameters to. Will be created if it doesn't exist.

unet_lora_layers (`dict[str, torch.nn.Module]` or `dict[str, torch.Tensor]`) : State dict of the LoRA layers corresponding to the `unet`.

text_encoder_lora_layers (`dict[str, torch.nn.Module]` or `dict[str, torch.Tensor]`) : State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text encoder LoRA state dict because it comes from 🤗 Transformers.

is_main_process (`bool`, *optional*, defaults to `True`) : Whether the process calling this is the main process or not. Useful during distributed training and you need to call this function on all processes. In this case, set `is_main_process=True` only on the main process to avoid race conditions.

save_function (`Callable`) : The function to use to save the state dictionary. Useful during distributed training when you need to replace `torch.save` with another method. Can be configured with the environment variable `DIFFUSERS_SAVE_MODE`.

safe_serialization (`bool`, *optional*, defaults to `True`) : Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.

unet_lora_adapter_metadata : LoRA adapter metadata associated with the unet to be serialized with the state dict.

text_encoder_lora_adapter_metadata : LoRA adapter metadata associated with the text encoder to be serialized with the state dict.

Save the LoRA parameters corresponding to the UNet and text encoder.

## StableDiffusionXLInstructPix2PixPipeline[[diffusers.StableDiffusionXLInstructPix2PixPipeline]]

#### diffusers.StableDiffusionXLInstructPix2PixPipeline[[diffusers.StableDiffusionXLInstructPix2PixPipeline]]

```python
diffusers.StableDiffusionXLInstructPix2PixPipeline(vae: AutoencoderKL, text_encoder: CLIPTextModel, text_encoder_2: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer, tokenizer_2: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, force_zeros_for_empty_prompt: bool = True, add_watermarker: bool | None = None, is_cosxl_edit: bool | None = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L113)

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.40.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`CLIPTextModel`) : Frozen text-encoder. Stable Diffusion XL uses the text portion of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

text_encoder_2 (` CLIPTextModelWithProjection`) : Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection), specifically the [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer_2 (`CLIPTokenizer`) : Second Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).

unet ([UNet2DConditionModel](/docs/diffusers/v0.40.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : Conditional U-Net architecture to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.40.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`) : Whether the `unet` requires a aesthetic_score condition to be passed during inference. Also see the config of `stabilityai/stable-diffusion-xl-refiner-1-0`.

force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`) : Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of `stabilityai/stable-diffusion-xl-base-1-0`.

add_watermarker (`bool`, *optional*) : Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to watermark output images. If not defined, it will default to True if the package is installed, otherwise no watermarker will be used.

is_cosxl_edit (`bool`, *optional*) : When set the image latents are scaled.

Pipeline for pixel-level image editing by following text instructions. Based on Stable Diffusion XL.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

The pipeline also inherits the following loading methods:
- [load_textual_inversion()](/docs/diffusers/v0.40.0/en/api/loaders/textual_inversion#diffusers.loaders.TextualInversionLoaderMixin.load_textual_inversion) for loading textual inversion embeddings
- [from_single_file()](/docs/diffusers/v0.40.0/en/api/loaders/single_file#diffusers.loaders.FromSingleFileMixin.from_single_file) for loading `.ckpt` files
- [load_lora_weights()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights) for loading LoRA weights
- [save_lora_weights()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights) for saving LoRA weights

#### __call__[[diffusers.StableDiffusionXLInstructPix2PixPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, prompt_2: str | list[str] | None = None, image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]] = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 100, denoising_end: float | None = None, guidance_scale: float = 5.0, image_guidance_scale: float = 1.5, negative_prompt: str | list[str] | None = None, negative_prompt_2: str | list[str] | None = None, num_images_per_prompt: int | None = 1, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, callback: typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None, callback_steps: int = 1, cross_attention_kwargs: dict[str, typing.Any] | None = None, guidance_rescale: float = 0.0, original_size: tuple = None, crops_coords_top_left: tuple = (0, 0), target_size: tuple = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L595)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is used in both text-encoders

image (`torch.Tensor` or `PIL.Image.Image` or `np.ndarray` or `list[torch.Tensor]` or `list[PIL.Image.Image]` or `list[np.ndarray]`) : The image(s) to modify with the pipeline.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

denoising_end (`float`, *optional*) : When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be completed before it is intentionally prematurely terminated. As a result, the returned sample will still retain a substantial amount of noise as determined by the discrete timesteps selected by the scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)

guidance_scale (`float`, *optional*, defaults to 5.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

image_guidance_scale (`float`, *optional*, defaults to 1.5) : Image guidance scale is to push the generated image towards the initial image `image`. Image guidance scale is enabled by setting `image_guidance_scale > 1`. Higher image guidance scale encourages to generate images that are closely linked to the source image `image`, usually at the expense of lower image quality. This pipeline requires a value of at least `1`.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

negative_prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

negative_pooled_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput` instead of a plain tuple.

callback (`Callable`, *optional*) : A function that will be called every `callback_steps` steps during inference. The function will be called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.

callback_steps (`int`, *optional*, defaults to 1) : The frequency at which the `callback` function will be called. If not specified, the callback will be called at every step.

cross_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

guidance_rescale (`float`, *optional*, defaults to 0.0) : Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when using zero terminal SNR.

original_size (`tuple[int]`, *optional*, defaults to (1024, 1024)) : If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled. `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

crops_coords_top_left (`tuple[int]`, *optional*, defaults to (0, 0)) : `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

target_size (`tuple[int]`, *optional*, defaults to (1024, 1024)) : For most cases, `target_size` should be set to the desired height and width of the generated image. If not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

**Returns:** `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` or `tuple`

`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` if `return_dict` is True, otherwise a
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

#### encode_prompt[[diffusers.StableDiffusionXLInstructPix2PixPipeline.encode_prompt]]

```python
encode_prompt(prompt: str, prompt_2: str | None = None, device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, do_classifier_free_guidance: bool = True, negative_prompt: str | None = None, negative_prompt_2: str | None = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py#L213)

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

Encodes the prompt into text encoder hidden states.

### Skyreels V2
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/skyreels_v2.md

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
vae = AutoModel.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)

pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    model_id,
    vae=vae,
    dtype=torch.bfloat16,
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
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
pipeline = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, dtype=torch.bfloat16
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
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
pipeline = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, dtype=torch.bfloat16
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

- SkyReels-V2 supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.40.0/en/api/loaders/lora#diffusers.loaders.SkyReelsV2LoraLoaderMixin.load_lora_weights).

`SkyReelsV2Pipeline` and `SkyReelsV2ImageToVideoPipeline` are also available without Diffusion Forcing framework applied.

## SkyReelsV2DiffusionForcingPipeline[[diffusers.SkyReelsV2DiffusionForcingPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingPipeline[[diffusers.SkyReelsV2DiffusionForcingPipeline]]

```python
diffusers.SkyReelsV2DiffusionForcingPipeline(tokenizer: AutoTokenizer, text_encoder: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.umt5.modeling_umt5.UMT5EncoderModel, transformer: SkyReelsV2Transformer3DModel, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L128)

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Text-to-Video (t2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

#### __call__[[diffusers.SkyReelsV2DiffusionForcingPipeline.__call__]]

```python
__call__(prompt: str | list[str], negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512, overlap_history: int | None = None, addnoise_condition: float = 0, base_num_frames: int = 97, ar_step: int = 0, causal_block_size: int | None = None, fps: int = 24)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L597)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, defaults to `544`) : The height of the generated video.

width (`int`, defaults to `960`) : The width of the generated video.

num_frames (`int`, defaults to `97`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `6.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `512`) : The maximum sequence length of the prompt.

overlap_history (`int`, *optional*, defaults to `None`) : Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.

addnoise_condition (`float`, *optional*, defaults to `0`) : This is used to help smooth the long video generation by adding some noise to the clean condition. Too large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger ones, but it is recommended to not exceed 50.

base_num_frames (`int`, *optional*, defaults to `97`) : 97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)

ar_step (`int`, *optional*, defaults to `0`) : Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous inference may improve the instruction following and visual consistent performance.

causal_block_size (`int`, *optional*, defaults to `None`) : The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step > 0)

fps (`int`, *optional*, defaults to `24`) : Frame rate of the generated video

**Returns:** `~SkyReelsV2PipelineOutput` or `tuple`

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
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

#### encode_prompt[[diffusers.SkyReelsV2DiffusionForcingPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L218)

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

#### generate_timestep_matrix[[diffusers.SkyReelsV2DiffusionForcingPipeline.generate_timestep_matrix]]

```python
generate_timestep_matrix(num_latent_frames: int, step_template: Tensor, base_num_latent_frames: int, ar_step: int = 5, num_pre_ready: int = 0, causal_block_size: int = 1, shrink_interval_with_mask: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L417)

**Parameters:**

num_latent_frames (int) : Total number of latent frames to generate

step_template (torch.Tensor) : Base timestep schedule (e.g., [1000, 800, 600, ..., 0])

base_num_latent_frames (int) : Maximum frames the model can process in one forward pass

ar_step (int, optional) : Autoregressive step size for temporal lag. 0 = synchronous, >0 = asynchronous. Defaults to 5.

num_pre_ready (int, optional) : Number of frames already denoised (e.g., from prefix in a video2video task). Defaults to 0.

causal_block_size (int, optional) : Number of frames processed as a causal block. Defaults to 1.

shrink_interval_with_mask (bool, optional) : Whether to optimize processing intervals. Defaults to False.

**Returns:** `tuple containing`

- step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
  [num_iterations, num_latent_frames]
- step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
  num_latent_frames]
- step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
  [num_iterations, num_latent_frames]
- valid_interval (list[tuple]): list of (start, end) intervals for each iteration

**Raises:** ``ValueError``

- ``ValueError`` -- If ar_step is too small for the given configuration

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

## SkyReelsV2DiffusionForcingImageToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline]]

```python
diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline(tokenizer: AutoTokenizer, text_encoder: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.umt5.modeling_umt5.UMT5EncoderModel, transformer: SkyReelsV2Transformer3DModel, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L133)

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Image-to-Video (i2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

#### __call__[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 5.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, image_embeds: typing.Optional[torch.Tensor] = None, last_image: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512, overlap_history: int | None = None, addnoise_condition: float = 0, base_num_frames: int = 97, ar_step: int = 0, causal_block_size: int | None = None, fps: int = 24)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L643)

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, defaults to `544`) : The height of the generated video.

width (`int`, defaults to `960`) : The width of the generated video.

num_frames (`int`, defaults to `97`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `5.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `negative_prompt` input argument.

image_embeds (`torch.Tensor`, *optional*) : Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided, image embeddings are generated from the `image` input argument.

last_image (`torch.Tensor`, *optional*) : Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided, image embeddings are generated from the `image` input argument.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `512`) : The maximum sequence length of the prompt.

overlap_history (`int`, *optional*, defaults to `None`) : Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.

addnoise_condition (`float`, *optional*, defaults to `0`) : This is used to help smooth the long video generation by adding some noise to the clean condition. Too large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger ones, but it is recommended to not exceed 50.

base_num_frames (`int`, *optional*, defaults to `97`) : 97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)

ar_step (`int`, *optional*, defaults to `0`) : Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous inference may improve the instruction following and visual consistent performance.

causal_block_size (`int`, *optional*, defaults to `None`) : The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step > 0)

fps (`int`, *optional*, defaults to `24`) : Frame rate of the generated video

**Returns:** `~SkyReelsV2PipelineOutput` or `tuple`

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
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

#### encode_prompt[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L223)

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

#### generate_timestep_matrix[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.generate_timestep_matrix]]

```python
generate_timestep_matrix(num_latent_frames: int, step_template: Tensor, base_num_latent_frames: int, ar_step: int = 5, num_pre_ready: int = 0, causal_block_size: int = 1, shrink_interval_with_mask: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L463)

**Parameters:**

num_latent_frames (int) : Total number of latent frames to generate

step_template (torch.Tensor) : Base timestep schedule (e.g., [1000, 800, 600, ..., 0])

base_num_latent_frames (int) : Maximum frames the model can process in one forward pass

ar_step (int, optional) : Autoregressive step size for temporal lag. 0 = synchronous, >0 = asynchronous. Defaults to 5.

num_pre_ready (int, optional) : Number of frames already denoised (e.g., from prefix in a video2video task). Defaults to 0.

causal_block_size (int, optional) : Number of frames processed as a causal block. Defaults to 1.

shrink_interval_with_mask (bool, optional) : Whether to optimize processing intervals. Defaults to False.

**Returns:** `tuple containing`

- step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
  [num_iterations, num_latent_frames]
- step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
  num_latent_frames]
- step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
  [num_iterations, num_latent_frames]
- valid_interval (list[tuple]): list of (start, end) intervals for each iteration

**Raises:** ``ValueError``

- ``ValueError`` -- If ar_step is too small for the given configuration

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

## SkyReelsV2DiffusionForcingVideoToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline]]

```python
diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline(tokenizer: AutoTokenizer, text_encoder: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.umt5.modeling_umt5.UMT5EncoderModel, transformer: SkyReelsV2Transformer3DModel, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L189)

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Video-to-Video (v2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

#### __call__[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.__call__]]

```python
__call__(video: list, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 120, num_inference_steps: int = 50, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512, overlap_history: int | None = None, addnoise_condition: float = 0, base_num_frames: int = 97, ar_step: int = 0, causal_block_size: int | None = None, fps: int = 24)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L681)

**Parameters:**

video (`list[Image.Image]`) : The video to guide the video generation.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, defaults to `544`) : The height of the generated video.

width (`int`, defaults to `960`) : The width of the generated video.

num_frames (`int`, defaults to `120`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `6.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. (**6.0 for T2V**, **5.0 for I2V**)

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `512`) : The maximum sequence length of the prompt.

overlap_history (`int`, *optional*, defaults to `None`) : Number of frames to overlap for smooth transitions in long videos. If `None`, the pipeline assumes short video generation mode, and no overlap is applied. 17 and 37 are recommended to set.

addnoise_condition (`float`, *optional*, defaults to `0`) : This is used to help smooth the long video generation by adding some noise to the clean condition. Too large noise can cause the inconsistency as well. 20 is a recommended value, and you may try larger ones, but it is recommended to not exceed 50.

base_num_frames (`int`, *optional*, defaults to `97`) : 97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)

ar_step (`int`, *optional*, defaults to `0`) : Controls asynchronous inference (0 for synchronous mode) You can set `ar_step=5` to enable asynchronous inference. When asynchronous inference, `causal_block_size=5` is recommended while it is not supposed to be set for synchronous generation. Asynchronous inference will take more steps to diffuse the whole sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous inference may improve the instruction following and visual consistent performance.

causal_block_size (`int`, *optional*, defaults to `None`) : The number of frames in each block/chunk. Recommended when using asynchronous inference (when ar_step > 0)

fps (`int`, *optional*, defaults to `24`) : Frame rate of the generated video

**Returns:** `~SkyReelsV2PipelineOutput` or `tuple`

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
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

#### encode_prompt[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L279)

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

#### generate_timestep_matrix[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.generate_timestep_matrix]]

```python
generate_timestep_matrix(num_latent_frames: int, step_template: Tensor, base_num_latent_frames: int, ar_step: int = 5, num_pre_ready: int = 0, causal_block_size: int = 1, shrink_interval_with_mask: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L501)

**Parameters:**

num_latent_frames (int) : Total number of latent frames to generate

step_template (torch.Tensor) : Base timestep schedule (e.g., [1000, 800, 600, ..., 0])

base_num_latent_frames (int) : Maximum frames the model can process in one forward pass

ar_step (int, optional) : Autoregressive step size for temporal lag. 0 = synchronous, >0 = asynchronous. Defaults to 5.

num_pre_ready (int, optional) : Number of frames already denoised (e.g., from prefix in a video2video task). Defaults to 0.

causal_block_size (int, optional) : Number of frames processed as a causal block. Defaults to 1.

shrink_interval_with_mask (bool, optional) : Whether to optimize processing intervals. Defaults to False.

**Returns:** `tuple containing`

- step_matrix (torch.Tensor): Matrix of timesteps for each frame at each iteration Shape:
  [num_iterations, num_latent_frames]
- step_index (torch.Tensor): Index matrix for timestep lookup Shape: [num_iterations,
  num_latent_frames]
- step_update_mask (torch.Tensor): Boolean mask indicating which frames to update Shape:
  [num_iterations, num_latent_frames]
- valid_interval (list[tuple]): list of (start, end) intervals for each iteration

**Raises:** ``ValueError``

- ``ValueError`` -- If ar_step is too small for the given configuration

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

## SkyReelsV2Pipeline[[diffusers.SkyReelsV2Pipeline]]

#### diffusers.SkyReelsV2Pipeline[[diffusers.SkyReelsV2Pipeline]]

```python
diffusers.SkyReelsV2Pipeline(tokenizer: AutoTokenizer, text_encoder: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.umt5.modeling_umt5.UMT5EncoderModel, transformer: SkyReelsV2Transformer3DModel, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L107)

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Text-to-Video (t2v) generation using SkyReels-V2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.SkyReelsV2Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L376)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).

height (`int`, defaults to `544`) : The height in pixels of the generated image.

width (`int`, defaults to `960`) : The width in pixels of the generated image.

num_frames (`int`, defaults to `97`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `6.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `SkyReelsV2PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `512`) : The maximum sequence length for the text encoder.

**Returns:** `~SkyReelsV2PipelineOutput` or `tuple`

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
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

#### encode_prompt[[diffusers.SkyReelsV2Pipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L197)

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

## SkyReelsV2ImageToVideoPipeline[[diffusers.SkyReelsV2ImageToVideoPipeline]]

#### diffusers.SkyReelsV2ImageToVideoPipeline[[diffusers.SkyReelsV2ImageToVideoPipeline]]

```python
diffusers.SkyReelsV2ImageToVideoPipeline(tokenizer: AutoTokenizer, text_encoder: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.umt5.modeling_umt5.UMT5EncoderModel, image_encoder: CLIPVisionModelWithProjection, image_processor: CLIPProcessor, transformer: SkyReelsV2Transformer3DModel, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L127)

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

image_encoder (`CLIPVisionModelWithProjection`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.40.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.40.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.40.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Image-to-Video (i2v) generation using SkyReels-V2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.SkyReelsV2ImageToVideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 5.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, image_embeds: typing.Optional[torch.Tensor] = None, last_image: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L476)

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

height (`int`, defaults to `544`) : The height of the generated video.

width (`int`, defaults to `960`) : The width of the generated video.

num_frames (`int`, defaults to `97`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, defaults to `5.0`) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `negative_prompt` input argument.

image_embeds (`torch.Tensor`, *optional*) : Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided, image embeddings are generated from the `image` input argument.

last_image (`torch.Tensor`, *optional*) : Optional last image for image-to-video conditioning that anchors the end of the generated video.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `WanPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int`, *optional*, defaults to `512`) : The maximum sequence length of the prompt.

**Returns:** `~SkyReelsV2PipelineOutput` or `tuple`

If `return_dict` is `True`, `SkyReelsV2PipelineOutput` is returned, otherwise a `tuple` is returned
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

#### encode_prompt[[diffusers.SkyReelsV2ImageToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], negative_prompt: str | list[str] | None = None, do_classifier_free_guidance: bool = True, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, max_sequence_length: int = 226, device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L238)

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

## SkyReelsV2PipelineOutput[[diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput]]

#### diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput[[diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput]]

```python
diffusers.pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/skyreels_v2/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for SkyReelsV2 pipelines.

### JoyAI-Image-Edit
https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/joyimage_edit.md
