# Bria Fibo Edit

Fibo Edit is an 8B parameter image-to-image model that introduces a new paradigm of structured control, operating on JSON inputs paired with source images to enable deterministic and repeatable editing workflows.
Featuring native masking for granular precision, it moves beyond simple prompt-based diffusion to offer explicit, interpretable control optimized for production environments.
Its lightweight architecture is designed for deep customization, empowering researchers to build specialized "Edit" models for domain-specific tasks while delivering top-tier aesthetic quality

## Usage
_As the model is gated, before using it with diffusers you first need to go to the [Bria Fibo Hugging Face page](https://huggingface.co/briaai/Fibo-Edit), fill in the form and accept the gate. Once you are in, you need to login so that your system knows you’ve accepted the gate._

Use the command below to log in:

```bash
hf auth login
```

## BriaFiboEditPipeline[[diffusers.BriaFiboEditPipeline]]

#### diffusers.BriaFiboEditPipeline[[diffusers.BriaFiboEditPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L234)

__call__diffusers.BriaFiboEditPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L603[{"name": "prompt", "val": ": typing.Union[str, typing.List[str]] = None"}, {"name": "image", "val": ": typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], NoneType] = None"}, {"name": "mask", "val": ": typing.Union[torch.FloatTensor, PIL.Image.Image, typing.List[PIL.Image.Image], typing.List[torch.FloatTensor], numpy.ndarray, typing.List[numpy.ndarray], NoneType] = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 30"}, {"name": "timesteps", "val": ": typing.List[int] = None"}, {"name": "seed", "val": ": int | None = None"}, {"name": "guidance_scale", "val": ": float = 5"}, {"name": "negative_prompt", "val": ": typing.Union[str, typing.List[str], NoneType] = None"}, {"name": "num_images_per_prompt", "val": ": typing.Optional[int] = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "joint_attention_kwargs", "val": ": typing.Optional[typing.Dict[str, typing.Any]] = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int, typing.Dict], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": typing.List[str] = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 3000"}, {"name": "do_patching", "val": " = False"}, {"name": "_auto_resize", "val": ": bool = True"}]- **prompt** (`str` or `List[str]`) --
  The prompt or prompts to guide the image generation.
- **image** (`PIL.Image.Image` or `torch.FloatTensor`, *optional*) --
  The image to guide the image generation. If not defined, the pipeline will generate an image from
  scratch.
- **mask** (`PipelineMaskInput`, *optional*) --
  Optional mask defining the region of `image` to be edited. Pixels covered by the mask are regenerated
  while the rest of the image is preserved.
- **height** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The height in pixels of the generated image. This is set to 1024 by default for the best results.
- **width** (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) --
  The width in pixels of the generated image. This is set to 1024 by default for the best results.
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **seed** (`int`, *optional*) --
  A seed used to make generation deterministic.
- **timesteps** (`List[int]`, *optional*) --
  Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
  in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
  passed will be used. Must be in descending order.
- **guidance_scale** (`float`, *optional*, defaults to 5.0) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **negative_prompt** (`str` or `List[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
  if `guidance_scale` is less than `1`).
- **num_images_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator` or `List[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.FloatTensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
  tensor will ge generated by sampling using the supplied random `generator`.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput` instead
  of a plain tuple.
- **joint_attention_kwargs** (`dict`, *optional*) --
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
- **max_sequence_length** (`int` defaults to 3000) -- Maximum sequence length to use with the `prompt`.
- **do_patching** (`bool`, *optional*, defaults to `False`) -- Whether to use patching.
- **_auto_resize** (`bool`, *optional*, defaults to `True`) --
  Whether to automatically resize the input image to the preferred resolutions.0`~pipelines.flux.BriaFiboPipelineOutput` or `tuple``~pipelines.flux.BriaFiboPipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.

Function invoked when calling the pipeline for generation.

Example:
```python
import torch
from diffusers import BriaFiboEditPipeline
from diffusers.modular_pipelines import ModularPipeline

torch.set_grad_enabled(False)
vlm_pipe = ModularPipelineBlocks.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)
vlm_pipe = vlm_pipe.init_pipeline()

pipe = BriaFiboEditPipeline.from_pretrained(
    "briaai/fibo-edit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

output = vlm_pipe(
    prompt="A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."
)
json_prompt_generate = json.loads(output.values["json_prompt"])

image = Image.open("image_generate.png")

edit_prompt = "Make the owl to be a cat"

json_prompt_generate["edit_instruction"] = edit_prompt

results_generate = pipe(
    prompt=json_prompt_generate, num_inference_steps=50, guidance_scale=3.5, image=image, output_type="np"
)
```

**Parameters:**

transformer (`BriaFiboTransformer2DModel`) : The transformer model for 2D diffusion modeling.

scheduler (`FlowMatchEulerDiscreteScheduler` or `KarrasDiffusionSchedulers`) : Scheduler to be used with `transformer` to denoise the encoded latents.

vae (`AutoencoderKLWan`) : Variational Auto-Encoder for encoding and decoding images to and from latent representations.

text_encoder (`SmolLM3ForCausalLM`) : Text encoder for processing input prompts.

tokenizer (`AutoTokenizer`) : Tokenizer used for processing the input text prompts for the text_encoder.

**Returns:**

``~pipelines.flux.BriaFiboPipelineOutput` or `tuple``

`~pipelines.flux.BriaFiboPipelineOutput` if
`return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
generated images.
#### encode_prompt[[diffusers.BriaFiboEditPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py#L359)

**Parameters:**

prompt (`str` or `List[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

num_images_per_prompt (`int`) : number of images that should be generated per prompt

guidance_scale (`float`) : Guidance scale for classifier free guidance.

negative_prompt (`str` or `List[str]`, *optional*) : The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

### Glm Image
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/glm_image.md

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

pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image",torch_dtype=torch.bfloat16,device_map="cuda")
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

pipe = GlmImagePipeline.from_pretrained("zai-org/GLM-Image",torch_dtype=torch.bfloat16,device_map="cuda")
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

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L161)

Pipeline for text-to-image generation using GLM-Image.

This pipeline integrates both the AR (autoregressive) model for token generation and the DiT (diffusion
transformer) model for image decoding.

__call__diffusers.GlmImagePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L719[{"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "image", "val": ": torch.Tensor | PIL.Image.Image | numpy.ndarray | list[torch.Tensor] | list[PIL.Image.Image] | list[numpy.ndarray] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float = 1.5"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prior_token_ids", "val": ": torch.Tensor | None = None"}, {"name": "prior_token_image_ids", "val": ": list[torch.Tensor] | None = None"}, {"name": "source_image_grid_thw", "val": ": list[torch.Tensor] | None = None"}, {"name": "crops_coords_top_left", "val": ": tuple = (0, 0)"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int, dict], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 2048"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. Must contain shape info in the format 'H
  W' where H and W are token dimensions (d32). Example: "A beautiful sunset36 24"
  generates a 1152x768 image.
- **image** -- Optional condition images for image-to-image generation.
- **height** (`int`, *optional*) --
  The height in pixels. If not provided, derived from prompt shape info.
- **width** (`int`, *optional*) --
  The width in pixels. If not provided, derived from prompt shape info.
- **num_inference_steps** (`int`, *optional*, defaults to `50`) --
  The number of denoising steps for DiT.
- **timesteps** (`list[int]`, *optional*) --
  Custom timesteps to use for the denoising process. If not defined, the scheduler's default schedule for
  `num_inference_steps` is used.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process. If not defined, the scheduler's default schedule is
  used.
- **guidance_scale** (`float`, *optional*, defaults to `1.5`) --
  Guidance scale for classifier-free guidance.
- **num_images_per_prompt** (`int`, *optional*, defaults to `1`) --
  The number of images to generate per prompt.
- **generator** (`torch.Generator`, *optional*) --
  Random generator for reproducibility.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents to be used as inputs for image generation.
- **prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated text embeddings. If not provided, embeddings are generated from `prompt`.
- **negative_prompt_embeds** (`torch.Tensor`, *optional*) --
  Pre-generated negative text embeddings. Used when classifier-free guidance is enabled.
- **prior_token_ids** (`torch.Tensor`, *optional*) --
  Pre-generated prior token ids from `generate_prior_tokens`. If supplied, prior generation is skipped.
- **prior_token_image_ids** (`list[torch.Tensor]`, *optional*) --
  Image token ids associated with `prior_token_ids`.
- **source_image_grid_thw** (`list[torch.Tensor]`, *optional*) --
  Per-sample THW grid information for the source image tokens.
- **crops_coords_top_left** (`tuple[int, int]`, *optional*, defaults to `(0, 0)`) --
  The top-left coordinates of the crop used for conditioning embeddings.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  Output format: "pil", "np", or "latent".
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `GlmImagePipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function called at the end of each denoising step.
- **callback_on_step_end_tensor_inputs** (`list[str]`, *optional*) --
  Tensor inputs passed to `callback_on_step_end`.
- **max_sequence_length** (`int`, *optional*, defaults to `2048`) --
  Maximum sequence length for the text encoder.0`GlmImagePipelineOutput` or `tuple`Generated images.

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

**Parameters:**

tokenizer (`PreTrainedTokenizer`) : Tokenizer for the text encoder.

processor (`AutoProcessor`) : Processor for the AR model to handle chat templates and tokenization.

text_encoder (`T5EncoderModel`) : Frozen text-encoder for glyph embeddings.

vision_language_encoder (`GlmImageForConditionalGeneration`) : The AR model that generates image tokens from text prompts.

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

transformer ([GlmImageTransformer2DModel](/docs/diffusers/v0.39.0/en/api/models/glm_image_transformer2d#diffusers.GlmImageTransformer2DModel)) : A text conditioned transformer to denoise the encoded image latents (DiT).

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

``GlmImagePipelineOutput` or `tuple``

Generated images.
#### encode_prompt[[diffusers.GlmImagePipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L545)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : Whether to use classifier free guidance or not.

num_images_per_prompt (`int`, *optional*, defaults to 1) : Number of images that should be generated per prompt. torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

device : (`torch.device`, *optional*): torch device

dtype : (`torch.dtype`, *optional*): torch dtype

max_sequence_length (`int`, defaults to `2048`) : Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.
#### generate_prior_tokens[[diffusers.GlmImagePipeline.generate_prior_tokens]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L321)

Generate prior tokens for the DiT model using the AR model.

**Parameters:**

prompt : Single prompt or list of prompts

height : Target image height

width : Target image width

image : Normalized image input as List[List[PIL.Image]]. Should be pre-validated using _validate_and_normalize_images() before calling this method.

device : Target device

generator : Random generator for reproducibility

**Returns:**

`Tuple of`

- prior_token_ids: Tensor of shape (batch_size, num_tokens) with upsampled prior tokens
- prior_token_image_ids_per_sample: List of tensors, one per sample. Each tensor contains
  the upsampled prior token ids for all condition images in that sample. None for t2i.
- source_image_grid_thw_per_sample: List of tensors, one per sample. Each tensor has shape
  (num_condition_images, 3) with upsampled grid info. None for t2i.
#### get_glyph_texts[[diffusers.GlmImagePipeline.get_glyph_texts]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L476)

Extract glyph texts from prompt(s). Returns a list of lists for batch processing.

## GlmImagePipelineOutput[[diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput]]

#### diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput[[diffusers.pipelines.glm_image.pipeline_output.GlmImagePipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/glm_image/pipeline_output.py#L10)

Output class for CogView3 pipelines.

**Parameters:**

images (`List[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Perturbed-Attention Guidance
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/pag.md
