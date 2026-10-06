# Bria Fibo Edit

Fibo Edit is an 8B parameter image-to-image model that introduces a new paradigm of structured control, operating on JSON inputs paired with source images to enable deterministic and repeatable editing workflows.
Featuring native masking for granular precision, it moves beyond simple prompt-based diffusion to offer explicit, interpretable control optimized for production environments.
Its lightweight architecture is designed for deep customization, empowering researchers to build specialized "Edit" models for domain-specific tasks while delivering top-tier aesthetic quality

Refer to the Bria Fibo Edit Hugging Face [page](https://huggingface.co/briaai/Fibo-Edit-1.5-base) to learn more. A distilled checkpoint is available at [Fibo-Edit-1.5-turbo](https://huggingface.co/briaai/Fibo-Edit-1.5-turbo).

## Usage

_As the model is gated, before using it with diffusers you first need to go to the [Bria Fibo Edit Hugging Face page](https://huggingface.co/briaai/Fibo-Edit-1.5-base), fill in the form and accept the gate. Once you are in, you need to login so that your system knows you’ve accepted the gate._

Use the command below to log in:

```bash
hf auth login
```

## BriaFiboEditPipeline[[diffusers.BriaFiboEditPipeline]]

#### diffusers.BriaFiboEditPipeline[[diffusers.BriaFiboEditPipeline]]

```python
diffusers.BriaFiboEditPipeline(transformer: BriaFiboTransformer2DModel, scheduler: typing.Union[diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler, diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers], vae: AutoencoderKLWan, text_encoder: SmolLM3ForCausalLM, tokenizer: AutoTokenizer)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L240)

**Parameters:**

transformer (`BriaFiboTransformer2DModel`) : The transformer model for 2D diffusion modeling.

scheduler (`FlowMatchEulerDiscreteScheduler` or `KarrasDiffusionSchedulers`) : Scheduler to be used with `transformer` to denoise the encoded latents.

vae (`AutoencoderKLWan`) : Variational Auto-Encoder for encoding and decoding images to and from latent representations.

text_encoder (`SmolLM3ForCausalLM`) : Text encoder for processing input prompts.

tokenizer (`AutoTokenizer`) : Tokenizer used for processing the input text prompts for the text_encoder.

#### __call__[[diffusers.BriaFiboEditPipeline.__call__]]

```python
__call__(prompt: typing.Union[str, typing.List[str]] = None, image: typing.Union[PIL.Image.Image, typing.List[PIL.Image.Image], NoneType] = None, mask: typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[PIL.Image.Image], typing.List[torch.FloatTensor], numpy.ndarray, typing.List[numpy.ndarray], NoneType] = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 30, timesteps: typing.List[int] = None, seed: int | None = None, guidance_scale: float = 5, negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, num_images_per_prompt: typing.Optional[int] = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.FloatTensor] = None, output_type: str = 'pil', return_dict: bool = True, joint_attention_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int, typing.Dict], NoneType]] = None, callback_on_step_end_tensor_inputs: typing.List[str] = ['latents'], max_sequence_length: int = 3000, do_patching = False, _auto_resize: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L599)

**Parameters:**

prompt (`str` or `List[str]`) : The prompt or prompts to guide the image generation.

image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*) : One or more reference images to guide the image generation. A list is interpreted as multiple references (not a batch): each reference is VAE-encoded at its own aspect ratio and placed on its own RoPE time plane 1, 2, ... . If not defined, the pipeline generates an image from scratch.

mask (`PipelineMaskInput`, *optional*) : Optional mask defining the region of `image` to be edited. Pixels covered by the mask are regenerated while the rest of the image is preserved.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 30) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

seed (`int`, *optional*) : A seed used to make generation deterministic.

timesteps (`List[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 5.0) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `List[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.FloatTensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead of a plain tuple.

joint_attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to 3000) : Maximum sequence length to use with the `prompt`.

do_patching (`bool`, *optional*, defaults to `False`) : Whether to use patching.

_auto_resize (`bool`, *optional*, defaults to `True`) : Whether to snap the default output resolution (taken from the first reference image) to the preferred resolutions.

**Returns:** `~pipelines.flux.BriaFiboPipelineOutput` or `tuple`

`~pipelines.flux.BriaFiboPipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.

Function invoked when calling the pipeline for generation.

Example:
```python
import torch
from PIL import Image

from diffusers import BriaFiboEditPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks

# This prompt-to-JSON block calls Gemini and needs GEMINI_API_KEY in the environment.
vlm_pipe = ModularPipelineBlocks.from_pretrained("briaai/FIBO-edit-gemini-prompt-to-JSON", trust_remote_code=True)
vlm_pipe = vlm_pipe.init_pipeline()

pipe = BriaFiboEditPipeline.from_pretrained(
    "briaai/Fibo-Edit-1.5-base",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = Image.open("owl.png")
json_prompt = vlm_pipe(image=image, prompt="Make the owl into a cat").values["json_prompt"]

result = pipe(prompt=json_prompt, image=image, num_inference_steps=30, guidance_scale=5)

# Multiple reference images: pass a list. Each reference conditions the edit at its
# own aspect ratio; the output resolution follows the first reference.
owl, forest = Image.open("owl.png"), Image.open("forest.png")
json_prompt = vlm_pipe(
    image=[owl, forest], prompt="Place the owl from the first image in the forest from the second image"
).values["json_prompt"]
result = pipe(
    prompt=json_prompt,
    image=[owl, forest],
    num_inference_steps=30,
    guidance_scale=5,
)

# The distilled Turbo checkpoint edits in 4 steps without classifier-free guidance.
pipe = BriaFiboEditPipeline.from_pretrained("briaai/Fibo-Edit-1.5-turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")
result = pipe(prompt=json_prompt, image=[owl, forest], num_inference_steps=4, guidance_scale=1)
```

#### encode_prompt[[diffusers.BriaFiboEditPipeline.encode_prompt]]

```python
encode_prompt(prompt: typing.Union[str, typing.List[str]], device: typing.Optional[torch.device] = None, num_images_per_prompt: int = 1, guidance_scale: float = 5, negative_prompt: typing.Union[str, typing.List[str], NoneType] = None, max_sequence_length: int = 3000, lora_scale: bool | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L365)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

guidance_scale (`float`) : Guidance scale for classifier free guidance.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

#### prepare_reference_latents[[diffusers.BriaFiboEditPipeline.prepare_reference_latents]]

```python
prepare_reference_latents(image: Image, num_channels_latents: int, dtype: dtype, device: device, do_patching: bool = False, reference_index: int = 1)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L992)

VAE-encode one PIL reference at its own size and pack it as an edit-context token stream.

### Flux2
https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/flux2.md
