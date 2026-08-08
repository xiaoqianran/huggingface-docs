# Stable Audio

Stable Audio was proposed in [Stable Audio Open](https://huggingface.co/papers/2407.14358) by Zach Evans et al. . it takes a text prompt as input and predicts the corresponding sound or music sample.

Stable Audio Open generates variable-length (up to 47s) stereo audio at 44.1kHz from text prompts. It comprises three components: an autoencoder that compresses waveforms into a manageable sequence length, a T5-based text embedding for text conditioning, and a transformer-based diffusion (DiT) model that operates in the latent space of the autoencoder.

Stable Audio is trained on a corpus of around 48k audio recordings, where around 47k are from Freesound and the rest are from the Free Music Archive (FMA). All audio files are licensed under CC0, CC BY, or CC Sampling+. This data is used to train the autoencoder and the DiT.

The abstract of the paper is the following:
*Open generative models are vitally important for the community, allowing for fine-tunes and serving as baselines when presenting new models. However, most current text-to-audio models are private and not accessible for artists and researchers to build upon. Here we describe the architecture and training process of a new open-weights text-to-audio model trained with Creative Commons data. Our evaluation shows that the model's performance is competitive with the state-of-the-art across various metrics. Notably, the reported FDopenl3 results (measuring the realism of the generations) showcase its potential for high-quality stereo sound synthesis at 44.1kHz.*

This pipeline was contributed by [Yoach Lacombe](https://huggingface.co/ylacombe). The original codebase can be found at [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).

## Tips

When constructing a prompt, keep in mind:

* Descriptive prompt inputs work best; use adjectives to describe the sound (for example, "high quality" or "clear") and make the prompt context specific where possible (e.g. "melodic techno with a fast beat and synths" works better than "techno").
* Using a *negative prompt* can significantly improve the quality of the generated audio. Try using a negative prompt of "low quality, average quality".

During inference:

* The _quality_ of the generated audio sample can be controlled by the `num_inference_steps` argument; higher steps give higher quality audio at the expense of slower inference.
* Multiple waveforms can be generated in one go: set `num_waveforms_per_prompt` to a value greater than 1 to enable. Automatic scoring will be performed between the generated waveforms and prompt text, and the audios ranked from best to worst accordingly.

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [StableAudioPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/stable_audio#diffusers.StableAudioPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, StableAudioDiTModel, StableAudioPipeline
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = StableAudioDiTModel.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "The sound of a hammer hitting a wooden surface."
negative_prompt = "Low quality."
audio = pipeline(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
sf.write("hammer.wav", output, pipeline.vae.sampling_rate)
```

## StableAudioPipeline[[diffusers.StableAudioPipeline]]
#### diffusers.StableAudioPipeline[[diffusers.StableAudioPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py#L78)

Pipeline for text-to-audio generation using StableAudio.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.StableAudioPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py#L490[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "audio_end_in_s", "val": ": float | None = None"}, {"name": "audio_start_in_s", "val": ": float | None = 0.0"}, {"name": "num_inference_steps", "val": ": int = 100"}, {"name": "guidance_scale", "val": ": float = 7.0"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "num_waveforms_per_prompt", "val": ": int | None = 1"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "initial_audio_waveforms", "val": ": torch.Tensor | None = None"}, {"name": "initial_audio_sampling_rate", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "attention_mask", "val": ": torch.LongTensor | None = None"}, {"name": "negative_attention_mask", "val": ": torch.LongTensor | None = None"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "callback", "val": ": typing.Optional[typing.Callable[[int, int, torch.Tensor], NoneType]] = None"}, {"name": "callback_steps", "val": ": int | None = 1"}, {"name": "output_type", "val": ": str | None = 'pt'"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
- **audio_end_in_s** (`float`, *optional*, defaults to 47.55) --
  Audio end index in seconds.
- **audio_start_in_s** (`float`, *optional*, defaults to 0) --
  Audio start index in seconds.
- **num_inference_steps** (`int`, *optional*, defaults to 100) --
  The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
  expense of slower inference.
- **guidance_scale** (`float`, *optional*, defaults to 7.0) --
  A higher guidance scale value encourages the model to generate audio that is closely linked to the text
  `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
  pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale 0[StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple`If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated audio.

The call function to the pipeline for generation.

Examples:
```py
>>> import scipy
>>> import torch
>>> import soundfile as sf
>>> from diffusers import StableAudioPipeline

>>> repo_id = "stabilityai/stable-audio-open-1.0"
>>> pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")

>>> # define the prompts
>>> prompt = "The sound of a hammer hitting a wooden surface."
>>> negative_prompt = "Low quality."

>>> # set the seed for generator
>>> generator = torch.Generator("cuda").manual_seed(0)

>>> # run the generation
>>> audio = pipe(
...     prompt,
...     negative_prompt=negative_prompt,
...     num_inference_steps=200,
...     audio_end_in_s=10.0,
...     num_waveforms_per_prompt=3,
...     generator=generator,
... ).audios

>>> output = audio[0].T.float().cpu().numpy()
>>> sf.write("hammer.wav", output, pipe.vae.sampling_rate)
```

**Parameters:**

vae ([AutoencoderOobleck](/docs/diffusers/v0.39.0/en/api/models/autoencoder_oobleck#diffusers.AutoencoderOobleck)) : Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.

text_encoder ([T5EncoderModel](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/t5#transformers.T5EncoderModel)) : Frozen text-encoder. StableAudio uses the encoder of [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) variant.

projection_model (`StableAudioProjectionModel`) : A trained model used to linearly project the hidden-states from the text encoder model and the start and end seconds. The projected hidden-states from the encoder and the conditional seconds are concatenated to give the input to the transformer model.

tokenizer ([T5Tokenizer](https://huggingface.co/docs/transformers/v5.12.1/en/model_doc/t5#transformers.T5Tokenizer)) : Tokenizer to tokenize text for the frozen text-encoder.

transformer ([StableAudioDiTModel](/docs/diffusers/v0.39.0/en/api/models/stable_audio_transformer#diffusers.StableAudioDiTModel)) : A `StableAudioDiTModel` to denoise the encoded audio latents.

scheduler ([EDMDPMSolverMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/edm_multistep_dpm_solver#diffusers.EDMDPMSolverMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded audio latents.

**Returns:**

`[StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) or `tuple``

If `return_dict` is `True`, [StableDiffusionPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/stable_diffusion/inpaint#diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput) is returned,
otherwise a `tuple` is returned where the first element is a list with the generated audio.
#### disable_vae_slicing[[diffusers.StableAudioPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py#L142)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.StableAudioPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py#L128)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.

### Cogview4
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/cogview4.md

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
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

## CogView4Pipeline[[diffusers.CogView4Pipeline]]

#### diffusers.CogView4Pipeline[[diffusers.CogView4Pipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L137)

Pipeline for text-to-image generation using CogView4.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

__call__diffusers.CogView4Pipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L402[{"name": "prompt", "val": ": str | list[str] | None = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int | None = None"}, {"name": "width", "val": ": int | None = None"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list[int] | None = None"}, {"name": "sigmas", "val": ": list[float] | None = None"}, {"name": "guidance_scale", "val": ": float = 5.0"}, {"name": "num_images_per_prompt", "val": ": int = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.FloatTensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.FloatTensor | None = None"}, {"name": "original_size", "val": ": tuple[int, int] | None = None"}, {"name": "crops_coords_top_left", "val": ": tuple = (0, 0)"}, {"name": "output_type", "val": ": str = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 1024"}]- **prompt** (`str` or `list[str]`, *optional*) --
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
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **guidance_scale** (`float`, *optional*, defaults to `5.0`) --
  Guidance scale as defined in [Classifier-Free Diffusion
  Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
  of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
  `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **num_images_per_prompt** (`int`, *optional*, defaults to `1`) --
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
- **max_sequence_length** (`int`, defaults to `224`) --
  Maximum sequence length in encoded prompt. Can be set to other values but may lead to poorer results.0`~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput` or `tuple``~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput` if `return_dict` is True, otherwise a
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

**Parameters:**

vae ([AutoencoderKL](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`GLMModel`) : Frozen text-encoder. CogView4 uses [glm-4-9b-hf](https://huggingface.co/THUDM/glm-4-9b-hf).

tokenizer (`PreTrainedTokenizer`) : Tokenizer of class [PreTrainedTokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).

transformer ([CogView4Transformer2DModel](/docs/diffusers/v0.39.0/en/api/models/cogview4_transformer2d#diffusers.CogView4Transformer2DModel)) : A text conditioned `CogView4Transformer2DModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.39.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

**Returns:**

``~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput` or `tuple``

`~pipelines.cogview4.pipeline_CogView4.CogView4PipelineOutput` if `return_dict` is True, otherwise a
`tuple`. When returning a tuple, the first element is a list with the generated images.
#### encode_prompt[[diffusers.CogView4Pipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L221)

Encodes the prompt into text encoder hidden states.

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

## CogView4PipelineOutput[[diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput]]

#### diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput[[diffusers.pipelines.cogview4.pipeline_output.CogView4PipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cogview4/pipeline_output.py#L10)

Output class for CogView3 pipelines.

**Parameters:**

images (`list[PIL.Image.Image]` or `np.ndarray`) : list of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.

### Ltx Video
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/ltx_video.md

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

  
    
      
    
    
  

# LTX-Video

[LTX-Video](https://huggingface.co/Lightricks/LTX-Video) is a diffusion transformer designed for fast and real-time generation of high-resolution videos from text and images. The main feature of LTX-Video is the Video-VAE. The Video-VAE has a higher pixel to latent compression ratio (1:192) which enables more efficient video data processing and faster generation speed. To support and prevent finer details from being lost during generation, the Video-VAE decoder performs the latent to pixel conversion *and* the last denoising step.

You can find all the original LTX-Video checkpoints under the [Lightricks](https://huggingface.co/Lightricks) organization.

> [!TIP]
> Click on the LTX-Video models in the right sidebar for more examples of other video generation tasks.

The example below demonstrates how to generate a video optimized for memory or inference speed.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The LTX-Video model below requires ~10GB of VRAM.

```py
import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipeline.vae, onload_device=onload_device, offload_type="leaf_level")

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair.
The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.
The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster. [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

```py
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
)

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair.
The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.
The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

## Notes

- Refer to the following recommended settings for generation from the [LTX-Video](https://github.com/Lightricks/LTX-Video) repository.

  - The recommended dtype for the transformer, VAE, and text encoder is `torch.bfloat16`. The VAE and text encoder can also be `torch.float32` or `torch.float16`.
  - For guidance-distilled variants of LTX-Video, set `guidance_scale` to `1.0`. The `guidance_scale` for any other model should be set higher, like `5.0`, for good generation quality.
  - For timestep-aware VAE variants (LTX-Video 0.9.1 and above), set `decode_timestep` to `0.05` and `image_cond_noise_scale` to `0.025`.
  - For variants that support interpolation between multiple conditioning images and videos (LTX-Video 0.9.5 and above), use similar images and videos for the best results. Divergence from the conditioning inputs may lead to abrupt transitions in the generated video.

- LTX-Video 0.9.7 includes a spatial latent upscaler and a 13B parameter transformer. During inference, a low resolution video is quickly generated first and then upscaled and refined.

  
  Show example code

  ```py
  import torch
  from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
  from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
  from diffusers.utils import export_to_video, load_video

  pipeline = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16)
  pipeline_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipeline.vae, torch_dtype=torch.bfloat16)
  pipeline.to("cuda")
  pipe_upsample.to("cuda")
  pipeline.vae.enable_tiling()

  def round_to_nearest_resolution_acceptable_by_vae(height, width):
      height = height - (height % pipeline.vae_temporal_compression_ratio)
      width = width - (width % pipeline.vae_temporal_compression_ratio)
      return height, width

  video = load_video(
      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
  )[:21]  # only use the first 21 frames as conditioning
  condition1 = LTXVideoCondition(video=video, frame_index=0)

  prompt = """
  The video depicts a winding mountain road covered in snow, with a single vehicle 
  traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. 
  The landscape is characterized by rugged terrain and a river visible in the distance. 
  The scene captures the solitude and beauty of a winter drive through a mountainous region.
  """
  negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
  expected_height, expected_width = 768, 1152
  downscale_factor = 2 / 3
  num_frames = 161

  # 1. Generate video at smaller resolution
  # Text-only conditioning is also supported without the need to pass `conditions`
  downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
  downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
  latents = pipeline(
      conditions=[condition1],
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=downscaled_width,
      height=downscaled_height,
      num_frames=num_frames,
      num_inference_steps=30,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=5.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="latent",
  ).frames

  # 2. Upscale generated video using latent upsampler with fewer inference steps
  # The available latent upsampler upscales the height/width by 2x
  upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
  upscaled_latents = pipe_upsample(
      latents=latents,
      output_type="latent"
  ).frames

  # 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
  video = pipeline(
      conditions=[condition1],
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=upscaled_width,
      height=upscaled_height,
      num_frames=num_frames,
      denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
      num_inference_steps=10,
      latents=upscaled_latents,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=5.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="pil",
  ).frames[0]

  # 4. Downscale the video to the expected resolution
  video = [frame.resize((expected_width, expected_height)) for frame in video]

  export_to_video(video, "output.mp4", fps=24)
  ```

  

- LTX-Video 0.9.7 distilled model is guidance and timestep-distilled to speedup generation. It requires `guidance_scale` to be set to `1.0` and `num_inference_steps` should be set between `4` and `10` for good generation quality. You should also use the following custom timesteps for the best results.

  - Base model inference to prepare for upscaling: `[1000, 993, 987, 981, 975, 909, 725, 0.03]`.
  - Upscaling: `[1000, 909, 725, 421, 0]`.

  
  Show example code

  ```py
  import torch
  from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
  from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
  from diffusers.utils import export_to_video, load_video

  pipeline = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.bfloat16)
  pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipeline.vae, torch_dtype=torch.bfloat16)
  pipeline.to("cuda")
  pipe_upsample.to("cuda")
  pipeline.vae.enable_tiling()

  def round_to_nearest_resolution_acceptable_by_vae(height, width):
      height = height - (height % pipeline.vae_spatial_compression_ratio)
      width = width - (width % pipeline.vae_spatial_compression_ratio)
      return height, width

  prompt = """
  artistic anatomical 3d render, utlra quality, human half full male body with transparent 
  skin revealing structure instead of organs, muscular, intricate creative patterns, 
  monochromatic with backlighting, lightning mesh, scientific concept art, blending biology 
  with botany, surreal and ethereal quality, unreal engine 5, ray tracing, ultra realistic, 
  16K UHD, rich details. camera zooms out in a rotating fashion
  """
  negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
  expected_height, expected_width = 768, 1152
  downscale_factor = 2 / 3
  num_frames = 161

  # 1. Generate video at smaller resolution
  downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
  downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
  latents = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=downscaled_width,
      height=downscaled_height,
      num_frames=num_frames,
      timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="latent",
  ).frames

  # 2. Upscale generated video using latent upsampler with fewer inference steps
  # The available latent upsampler upscales the height/width by 2x
  upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
  upscaled_latents = pipe_upsample(
      latents=latents,
      adain_factor=1.0,
      output_type="latent"
  ).frames

  # 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
  video = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=upscaled_width,
      height=upscaled_height,
      num_frames=num_frames,
      denoise_strength=0.999,  # Effectively, 4 inference steps out of 5
      timesteps=[1000, 909, 725, 421, 0],
      latents=upscaled_latents,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="pil",
  ).frames[0]

  # 4. Downscale the video to the expected resolution
  video = [frame.resize((expected_width, expected_height)) for frame in video]

  export_to_video(video, "output.mp4", fps=24)
  ```

  

- LTX-Video 0.9.8 distilled model is similar to the 0.9.7 variant. It is guidance and timestep-distilled, and similar inference code can be used as above. An improvement of this version is that it supports generating very long videos. Additionally, it supports using tone mapping to improve the quality of the generated video using the `tone_map_compression_ratio` parameter. The default value of `0.6` is recommended.

  
  Show example code

  ```python
  import torch
  from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
  from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
  from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
  from diffusers.utils import export_to_video, load_video

  pipeline = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.8-13B-distilled", torch_dtype=torch.bfloat16)
  # TODO: Update the checkpoint here once updated in LTX org
  upsampler = LTXLatentUpsamplerModel.from_pretrained("a-r-r-o-w/LTX-0.9.8-Latent-Upsampler", torch_dtype=torch.bfloat16)
  pipe_upsample = LTXLatentUpsamplePipeline(vae=pipeline.vae, latent_upsampler=upsampler).to(torch.bfloat16)
  pipeline.to("cuda")
  pipe_upsample.to("cuda")
  pipeline.vae.enable_tiling()

  def round_to_nearest_resolution_acceptable_by_vae(height, width):
      height = height - (height % pipeline.vae_spatial_compression_ratio)
      width = width - (width % pipeline.vae_spatial_compression_ratio)
      return height, width

  prompt = """The camera pans over a snow-covered mountain range, revealing a vast expanse of snow-capped peaks and valleys.The mountains are covered in a thick layer of snow, with some areas appearing almost white while others have a slightly darker, almost grayish hue. The peaks are jagged and irregular, with some rising sharply into the sky while others are more rounded. The valleys are deep and narrow, with steep slopes that are also covered in snow. The trees in the foreground are mostly bare, with only a few leaves remaining on their branches. The sky is overcast, with thick clouds obscuring the sun. The overall impression is one of peace and tranquility, with the snow-covered mountains standing as a testament to the power and beauty of nature."""
  # prompt = """A woman walks away from a white Jeep parked on a city street at night, then ascends a staircase and knocks on a door. The woman, wearing a dark jacket and jeans, walks away from the Jeep parked on the left side of the street, her back to the camera; she walks at a steady pace, her arms swinging slightly by her sides; the street is dimly lit, with streetlights casting pools of light on the wet pavement; a man in a dark jacket and jeans walks past the Jeep in the opposite direction; the camera follows the woman from behind as she walks up a set of stairs towards a building with a green door; she reaches the top of the stairs and turns left, continuing to walk towards the building; she reaches the door and knocks on it with her right hand; the camera remains stationary, focused on the doorway; the scene is captured in real-life footage."""
  negative_prompt = "bright colors, symbols, graffiti, watermarks, worst quality, inconsistent motion, blurry, jittery, distorted"
  expected_height, expected_width = 480, 832
  downscale_factor = 2 / 3
  # num_frames = 161
  num_frames = 361

  # 1. Generate video at smaller resolution
  downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
  downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
  latents = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=downscaled_width,
      height=downscaled_height,
      num_frames=num_frames,
      timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="latent",
  ).frames

  # 2. Upscale generated video using latent upsampler with fewer inference steps
  # The available latent upsampler upscales the height/width by 2x
  upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
  upscaled_latents = pipe_upsample(
      latents=latents,
      adain_factor=1.0,
      tone_map_compression_ratio=0.6,
      output_type="latent"
  ).frames

  # 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
  video = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=upscaled_width,
      height=upscaled_height,
      num_frames=num_frames,
      denoise_strength=0.999,  # Effectively, 4 inference steps out of 5
      timesteps=[1000, 909, 725, 421, 0],
      latents=upscaled_latents,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="pil",
  ).frames[0]

  # 4. Downscale the video to the expected resolution
  video = [frame.resize((expected_width, expected_height)) for frame in video]

  export_to_video(video, "output.mp4", fps=24)
  ```

  

- LTX-Video supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.LTXVideoLoraLoaderMixin.load_lora_weights).

  
  Show example code

  ```py
  import torch
  from diffusers import LTXConditionPipeline
  from diffusers.utils import export_to_video, load_image

  pipeline = LTXConditionPipeline.from_pretrained(
      "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
  )

  pipeline.load_lora_weights("Lightricks/LTX-Video-Cakeify-LoRA", adapter_name="cakeify")
  pipeline.set_adapters("cakeify")

  # use "CAKEIFY" to trigger the LoRA
  prompt = "CAKEIFY a person using a knife to cut a cake shaped like a Pikachu plushie"
  image = load_image("https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA/resolve/main/assets/images/pikachu.png")

  video = pipeline(
      prompt=prompt,
      image=image,
      width=576,
      height=576,
      num_frames=161,
      decode_timestep=0.03,
      decode_noise_scale=0.025,
      num_inference_steps=50,
  ).frames[0]
  export_to_video(video, "output.mp4", fps=26)
  ```

  

- LTX-Video supports loading from single files, such as [GGUF checkpoints](../../quantization/gguf), with [loaders.FromOriginalModelMixin.from_single_file()](/docs/diffusers/v0.39.0/en/api/loaders/single_file#diffusers.loaders.FromOriginalModelMixin.from_single_file) or [loaders.FromSingleFileMixin.from_single_file()](/docs/diffusers/v0.39.0/en/api/loaders/single_file#diffusers.loaders.FromSingleFileMixin.from_single_file).

  
  Show example code

  ```py
  import torch
  from diffusers.utils import export_to_video
  from diffusers import LTXPipeline, AutoModel, GGUFQuantizationConfig

  transformer = AutoModel.from_single_file(
      "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3_K_S.gguf"
      quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
      torch_dtype=torch.bfloat16
  )
  pipeline = LTXPipeline.from_pretrained(
      "Lightricks/LTX-Video",
      transformer=transformer,
      torch_dtype=torch.bfloat16
  )
  ```

  

## LTXI2VLongMultiPromptPipeline[[diffusers.LTXI2VLongMultiPromptPipeline]]

#### diffusers.LTXI2VLongMultiPromptPipeline[[diffusers.LTXI2VLongMultiPromptPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py#L389)

Long-duration I2V (image-to-video) multi-prompt pipeline with ComfyUI parity.

Key features:
- Temporal sliding-window sampling only (no spatial H/W sharding); autoregressive fusion across windows.
- Multi-prompt segmentation per window with smooth transitions at window heads.
- First-frame hard conditioning via per-token mask for I2V.
- VRAM control via temporal windowing and VAE tiled decoding.

Reference: https://github.com/Lightricks/LTX-Video

__call__diffusers.LTXI2VLongMultiPromptPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py#L935[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "prompt_segments", "val": ": list[dict[str, Any]] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 704"}, {"name": "num_frames", "val": ": int = 161"}, {"name": "frame_rate", "val": ": float = 25"}, {"name": "guidance_scale", "val": ": float = 1.0"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "num_inference_steps", "val": ": int | None = 8"}, {"name": "sigmas", "val": ": list[float, torch.Tensor] | None = None"}, {"name": "generator", "val": ": torch.Generator | list[torch.Generator] | None = None"}, {"name": "seed", "val": ": int | None = 0"}, {"name": "cond_image", "val": ": 'PIL.Image.Image' | torch.Tensor | None = None"}, {"name": "cond_strength", "val": ": float = 0.5"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "temporal_tile_size", "val": ": int = 80"}, {"name": "temporal_overlap", "val": ": int = 24"}, {"name": "temporal_overlap_cond_strength", "val": ": float = 0.5"}, {"name": "adain_factor", "val": ": float = 0.25"}, {"name": "guidance_latents", "val": ": torch.Tensor | None = None"}, {"name": "guiding_strength", "val": ": float = 1.0"}, {"name": "negative_index_latents", "val": ": torch.Tensor | None = None"}, {"name": "negative_index_strength", "val": ": float = 1.0"}, {"name": "skip_steps_sigma_threshold", "val": ": float | None = 1"}, {"name": "decode_timestep", "val": ": float | None = 0.05"}, {"name": "decode_noise_scale", "val": ": float | None = 0.025"}, {"name": "decode_horizontal_tiles", "val": ": int = 4"}, {"name": "decode_vertical_tiles", "val": ": int = 4"}, {"name": "decode_overlap", "val": ": int = 3"}, {"name": "output_type", "val": ": str | None = 'latent'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, Any] | None = None"}, {"name": "callback_on_step_end", "val": ": Callable[[int, int], None] | None = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list[str] = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 128"}]- **prompt** (`str` or `list[str]`, *optional*) --
  Positive text prompt(s) per window. If a single string contains '|', parts are split by bars.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  Negative prompt(s) to suppress undesired content.
- **prompt_segments** (`list[dict]`, *optional*) --
  Segment mapping with {"start_window", "end_window", "text"} to override prompts per window.
- **height** (`int`, defaults to `512`) --
  Output image height in pixels; must be divisible by 32.
- **width** (`int`, defaults to `704`) --
  Output image width in pixels; must be divisible by 32.
- **num_frames** (`int`, defaults to `161`) --
  Number of output frames (in decoded pixel space).
- **frame_rate** (`float`, defaults to `25`) --
  Frames-per-second; used to normalize temporal coordinates in `video_coords`.
- **guidance_scale** (`float`, defaults to `1.0`) --
  CFG scale; values > 1 enable classifier-free guidance.
- **guidance_rescale** (`float`, defaults to `0.0`) --
  Optional rescale to mitigate overexposure under CFG (see `rescale_noise_cfg`).
- **num_inference_steps** (`int`, *optional*, defaults to `8`) --
  Denoising steps per window. Ignored if `sigmas` is provided.
- **sigmas** (`list[float]` or `torch.Tensor`, *optional*) --
  Explicit sigma schedule per window; if set, overrides `num_inference_steps`.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  Controls stochasticity; list accepted but first element is used (batch=1).
- **seed** (`int`, *optional*, defaults to `0`) --
  If provided, seeds the shared generator for global latents and derives a window-local generator with
  `seed + w_start` per temporal window.
- **cond_image** (`PIL.Image.Image` or `torch.Tensor`, *optional*) --
  Conditioning image; fixes frame 0 via per-token mask when `cond_strength > 0`.
- **cond_strength** (`float`, defaults to `0.5`) --
  Strength of first-frame hard conditioning (smaller cond_mask ⇒ stronger preservation).
- **latents** (`torch.Tensor`, *optional*) --
  Initial latents [B, C_lat, F_lat, H_lat, W_lat]; if None, sampled with `randn_tensor`.
- **temporal_tile_size** (`int`, defaults to `80`) --
  Temporal window size (in decoded frames); internally scaled by VAE temporal compression.
- **temporal_overlap** (`int`, defaults to `24`) --
  Overlap between consecutive windows (in decoded frames); internally scaled by compression.
- **temporal_overlap_cond_strength** (`float`, defaults to `0.5`) --
  Strength for injecting previous window tail latents at new window head.
- **adain_factor** (`float`, defaults to `0.25`) --
  AdaIN normalization strength for cross-window consistency (0 disables).
- **guidance_latents** (`torch.Tensor`, *optional*) --
  Reference latents injected at window head; length trimmed by overlap for subsequent windows.
- **guiding_strength** (`float`, defaults to `1.0`) --
  Injection strength for `guidance_latents`.
- **negative_index_latents** (`torch.Tensor`, *optional*) --
  A single-frame latent appended at window head for "negative index" semantics.
- **negative_index_strength** (`float`, defaults to `1.0`) --
  Injection strength for `negative_index_latents`.
- **skip_steps_sigma_threshold** (`float`, *optional*, defaults to `1`) --
  Skip steps whose sigma exceeds this threshold.
- **decode_timestep** (`float`, *optional*, defaults to `0.05`) --
  Decode-time timestep (if VAE supports timestep_conditioning).
- **decode_noise_scale** (`float`, *optional*, defaults to `0.025`) --
  Decode-time noise mix scale (if VAE supports timestep_conditioning).
- **decode_horizontal_tiles** (`int`, defaults to `4`) --
  Number of horizontal tiles during VAE decoding.
- **decode_vertical_tiles** (`int`, defaults to `4`) --
  Number of vertical tiles during VAE decoding.
- **decode_overlap** (`int`, defaults to `3`) --
  Overlap (in latent pixels) between tiles during VAE decoding.
- **output_type** (`str`, *optional*, defaults to `"latent"`) --
  The output format of the generated video. Choose between "latent", "pt", "np", or "pil". If "latent",
  returns latents without decoding.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  Extra attention parameters forwarded to the transformer.
- **callback_on_step_end** (`PipelineCallback` or `MultiPipelineCallbacks`, *optional*) --
  Per-step callback hook.
- **callback_on_step_end_tensor_inputs** (`list[str]`, defaults to `["latents"]`) --
  Keys from locals() to pass into the callback.
- **max_sequence_length** (`int`, defaults to `128`) --
  Tokenizer max length for prompt encoding.0`~pipelines.ltx.LTXPipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated frames. The output format depends on
`output_type`:
- "latent"/"pt": `torch.Tensor` [B, C, F, H, W]; "latent" is in normalized latent space, "pt" is VAE
  output space.
- "np": `np.ndarray` post-processed.
- "pil": `list[PIL.Image.Image]` list of PIL images.

Generate an image-to-video sequence via temporal sliding windows and multi-prompt scheduling.

Examples:
```py
>>> import torch
>>> from diffusers import LTXEulerAncestralRFScheduler, LTXI2VLongMultiPromptPipeline

>>> pipe = LTXI2VLongMultiPromptPipeline.from_pretrained("LTX-Video-0.9.8-13B-distilled")
>>> # For ComfyUI parity, swap in the RF scheduler (keeps the original config).
>>> pipe.scheduler = LTXEulerAncestralRFScheduler.from_config(pipe.scheduler.config)
>>> pipe = pipe.to("cuda").to(dtype=torch.bfloat16)
>>> # Example A: get decoded frames (PIL)
>>> out = pipe(
...     prompt="a chimpanzee walks | a chimpanzee eats",
...     num_frames=161,
...     height=512,
...     width=704,
...     temporal_tile_size=80,
...     temporal_overlap=24,
...     output_type="pil",
...     return_dict=True,
... )
>>> frames = out.frames[0]  # list of PIL.Image.Image
>>> # Example B: get latent video and decode later (saves VRAM during sampling)
>>> out_latent = pipe(prompt="a chimpanzee walking", output_type="latent", return_dict=True).frames
>>> frames = pipe.vae_decode_tiled(out_latent, output_type="pil")[0]
```

Shapes:
Latent sizes (when auto-generated):
- F_lat = (num_frames - 1) // vae_temporal_compression_ratio + 1
- H_lat = height // vae_spatial_compression_ratio
- W_lat = width // vae_spatial_compression_ratio

Notes:
- Seeding: when `seed` is provided, each temporal window uses a local generator seeded with `seed +
  w_start`, while the shared generator is seeded once for global latents if no generator is passed;
  otherwise the passed-in generator is reused.
- CFG: unified `noise_pred = uncond + w * (text - uncond)` with optional `guidance_rescale`.
- Memory: denoising performs full-frame predictions (no spatial tiling); decoding can be tiled to avoid
  OOM.

**Parameters:**

transformer ([LTXVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler) or `LTXEulerAncestralRFScheduler`) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLLTXVideo](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`T5TokenizerFast`) : Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

**Returns:**

``~pipelines.ltx.LTXPipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated frames. The output format depends on
`output_type`:
- "latent"/"pt": `torch.Tensor` [B, C, F, H, W]; "latent" is in normalized latent space, "pt" is VAE
  output space.
- "np": `np.ndarray` post-processed.
- "pil": `list[PIL.Image.Image]` list of PIL images.
#### encode_prompt[[diffusers.LTXI2VLongMultiPromptPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py#L552)

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
#### prepare_latents[[diffusers.LTXI2VLongMultiPromptPipeline.prepare_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py#L694)

Prepare base latents and optionally inject first-frame conditioning latents.

**Returns:**

latents, negative_index_latents, latent_num_frames, latent_height, latent_width
#### vae_decode_tiled[[diffusers.LTXI2VLongMultiPromptPipeline.vae_decode_tiled]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py#L738)

VAE-based spatial tiled decoding (ComfyUI parity) implemented in Diffusers style.
- Linearly feather and blend overlapping tiles to avoid seams.
- Optional last_frame_fix: duplicate the last latent frame before decoding, then drop time_scale_factor frames
  at the end.
- Supports timestep_conditioning and decode_noise_scale injection.
- By default, "normalized latents" (the denoising output) are de-normalized internally (auto_denormalize=True).
- Tile fusion is computed in compute_dtype (float32 by default) to reduce blur and color shifts.

**Parameters:**

latents : [B, C_latent, F_latent, H_latent, W_latent]

decode_timestep : Optional decode timestep (effective only if VAE supports timestep_conditioning)

decode_noise_scale : Optional decode noise interpolation (effective only if VAE supports timestep_conditioning)

horizontal_tiles, vertical_tiles : Number of tiles horizontally/vertically (>= 1)

overlap : Overlap in latent space (in latent pixels, >= 0)

last_frame_fix : Whether to enable the "repeat last frame" fix

generator : Random generator (used for decode_noise_scale noise)

output_type : "latent" | "pt" | "np" | "pil" - "latent": return latents unchanged (useful for downstream processing) - "pt": return tensor in VAE output space - "np"/"pil": post-processed outputs via VideoProcessor.postprocess_video

auto_denormalize : If True, apply LTX de-normalization to `latents` internally (recommended)

compute_dtype : Precision used during tile fusion (float32 default; significantly reduces seam blur)

enable_vae_tiling : If True, delegate tiling to VAE's built-in `tiled_decode` (sets `vae.use_tiling`).

**Returns:**

`- If output_type="latent"`

returns input `latents` unchanged
- If output_type="pt": returns [B, C, F, H, W] (values roughly in [-1, 1])
- If output_type="np"/"pil": returns post-processed outputs via postprocess_video

## LTXPipeline[[diffusers.LTXPipeline]]

#### diffusers.LTXPipeline[[diffusers.LTXPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx.py#L170)

Pipeline for text-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

__call__diffusers.LTXPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx.py#L535[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 704"}, {"name": "num_frames", "val": ": int = 161"}, {"name": "frame_rate", "val": ": int = 25"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 3"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 128"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
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
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to `128 `) --
  Maximum sequence length to use with the `prompt`.0`~pipelines.ltx.LTXPipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTXPipeline
>>> from diffusers.utils import export_to_video

>>> pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> video = pipe(
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=704,
...     height=480,
...     num_frames=161,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=24)
```

**Parameters:**

transformer ([LTXVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLLTXVideo](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

**Returns:**

``~pipelines.ltx.LTXPipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### encode_prompt[[diffusers.LTXPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx.py#L283)

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

## LTXImageToVideoPipeline[[diffusers.LTXImageToVideoPipeline]]

#### diffusers.LTXImageToVideoPipeline[[diffusers.LTXImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py#L189)

Pipeline for image-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

__call__diffusers.LTXImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py#L596[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] = None"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 704"}, {"name": "num_frames", "val": ": int = 161"}, {"name": "frame_rate", "val": ": int = 25"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 3"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 128"}]- **image** (`PipelineImageInput`) --
  The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
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
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to `128 `) --
  Maximum sequence length to use with the `prompt`.0`~pipelines.ltx.LTXPipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import LTXImageToVideoPipeline
>>> from diffusers.utils import export_to_video, load_image

>>> pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> image = load_image(
...     "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
... )
>>> prompt = "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, creating a poignant and emotionally charged scene."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> video = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=704,
...     height=480,
...     num_frames=161,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(video, "output.mp4", fps=24)
```

**Parameters:**

transformer ([LTXVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLLTXVideo](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

**Returns:**

``~pipelines.ltx.LTXPipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### encode_prompt[[diffusers.LTXImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py#L306)

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

## LTXConditionPipeline[[diffusers.LTXConditionPipeline]]

#### diffusers.LTXConditionPipeline[[diffusers.LTXConditionPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py#L252)

Pipeline for text/image/video-to-video generation.

Reference: https://github.com/Lightricks/LTX-Video

__call__diffusers.LTXConditionPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py#L848[{"name": "conditions", "val": ": diffusers.pipelines.ltx.pipeline_ltx_condition.LTXVideoCondition | list[diffusers.pipelines.ltx.pipeline_ltx_condition.LTXVideoCondition] = None"}, {"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor] | list[PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]] = None"}, {"name": "video", "val": ": list = None"}, {"name": "frame_index", "val": ": int | list[int] = 0"}, {"name": "strength", "val": ": float | list[float] = 1.0"}, {"name": "denoise_strength", "val": ": float = 1.0"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 704"}, {"name": "num_frames", "val": ": int = 161"}, {"name": "frame_rate", "val": ": int = 25"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 3"}, {"name": "guidance_rescale", "val": ": float = 0.0"}, {"name": "image_cond_noise_scale", "val": ": float = 0.15"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 256"}]- **conditions** (`list[LTXVideoCondition], *optional*`) --
  The list of frame-conditioning items for the video generation.If not provided, conditions will be
  created using `image`, `video`, `frame_index` and `strength`.
- **image** (`PipelineImageInput` or `list[PipelineImageInput]`, *optional*) --
  The image or images to condition the video generation. If not provided, one has to pass `video` or
  `conditions`.
- **video** (`list[PipelineImageInput]`, *optional*) --
  The video to condition the video generation. If not provided, one has to pass `image` or `conditions`.
- **frame_index** (`int` or `list[int]`, *optional*) --
  The frame index or frame indices at which the image or video will conditionally effect the video
  generation. If not provided, one has to pass `conditions`.
- **strength** (`float` or `list[float]`, *optional*) --
  The strength or strengths of the conditioning effect. If not provided, one has to pass `conditions`.
- **denoise_strength** (`float`, defaults to `1.0`) --
  The strength of the noise added to the latents for editing. Higher strength leads to more noise added
  to the latents, therefore leading to more differences between original video and generated video. This
  is useful for video-to-video editing.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale  1`. Higher guidance scale encourages to generate images that are closely linked to
  the text `prompt`, usually at the expense of lower image quality.
- **guidance_rescale** (`float`, *optional*, defaults to 0.0) --
  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
  [Common Diffusion Noise Schedules and Sample Steps are
  Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
  using zero terminal SNR.
- **image_cond_noise_scale** (`float`, defaults to `0.15`) --
  Scale of noise added to the conditioning image latents.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
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
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generate image. Choose between
  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
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
- **max_sequence_length** (`int` defaults to `128 `) --
  Maximum sequence length to use with the `prompt`.0`~pipelines.ltx.LTXPipelineOutput` or `tuple`If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
>>> from diffusers.utils import export_to_video, load_video, load_image

>>> pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> # Load input image and video
>>> video = load_video(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
... )
>>> image = load_image(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
... )

>>> # Create conditioning objects
>>> condition1 = LTXVideoCondition(
...     image=image,
...     frame_index=0,
... )
>>> condition2 = LTXVideoCondition(
...     video=video,
...     frame_index=80,
... )

>>> prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
>>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

>>> # Generate video
>>> generator = torch.Generator("cuda").manual_seed(0)
>>> # Text-only conditioning is also supported without the need to pass `conditions`
>>> video = pipe(
...     conditions=[condition1, condition2],
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     width=768,
...     height=512,
...     num_frames=161,
...     num_inference_steps=40,
...     generator=generator,
... ).frames[0]

>>> export_to_video(video, "output.mp4", fps=24)
```

**Parameters:**

transformer ([LTXVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/ltx_video_transformer3d#diffusers.LTXVideoTransformer3DModel)) : Conditional Transformer architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLLTXVideo](/docs/diffusers/v0.39.0/en/api/models/autoencoderkl_ltx_video#diffusers.AutoencoderKLLTXVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.

tokenizer (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

tokenizer (`T5TokenizerFast`) : Second Tokenizer of class [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).

**Returns:**

``~pipelines.ltx.LTXPipelineOutput` or `tuple``

If `return_dict` is `True`, `~pipelines.ltx.LTXPipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.
#### add_noise_to_image_conditioning_latents[[diffusers.LTXConditionPipeline.add_noise_to_image_conditioning_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py#L646)

Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
when conditioned on a single frame.
#### encode_prompt[[diffusers.LTXConditionPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py#L369)

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
#### trim_conditioning_sequence[[diffusers.LTXConditionPipeline.trim_conditioning_sequence]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py#L629)

Trim a conditioning sequence to the allowed number of frames.

**Parameters:**

start_frame (int) : The target frame number of the first frame in the sequence.

sequence_num_frames (int) : The number of frames in the sequence.

target_num_frames (int) : The target number of frames in the generated video.

**Returns:**

`int`

updated sequence length

## LTXLatentUpsamplePipeline[[diffusers.LTXLatentUpsamplePipeline]]

#### diffusers.LTXLatentUpsamplePipeline[[diffusers.LTXLatentUpsamplePipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L44)

__call__diffusers.LTXLatentUpsamplePipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L241[{"name": "video", "val": ": list[PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]] | None = None"}, {"name": "height", "val": ": int = 512"}, {"name": "width", "val": ": int = 704"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "decode_timestep", "val": ": float | list[float] = 0.0"}, {"name": "decode_noise_scale", "val": ": float | list[float] | None = None"}, {"name": "adain_factor", "val": ": float = 0.0"}, {"name": "tone_map_compression_ratio", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}]- **video** (`list[PipelineImageInput]`, *optional*) --
  The input video frames to upsample. Mutually exclusive with `latents`.
- **height** (`int`, defaults to `512`) --
  The height in pixels of the upsampled output.
- **width** (`int`, defaults to `704`) --
  The width in pixels of the upsampled output.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-encoded video latents to upsample. Mutually exclusive with `video`.
- **decode_timestep** (`float` or `list[float]`, defaults to `0.0`) --
  The timestep at which the upsampled latents are decoded.
- **decode_noise_scale** (`float` or `list[float]`, *optional*) --
  Interpolation factor between random noise and denoised latents at the decode timestep.
- **adain_factor** (`float`, defaults to `0.0`) --
  Strength of AdaIN statistical matching applied to the upsampled latents.
- **tone_map_compression_ratio** (`float`, defaults to `0.0`) --
  Compression ratio used for tone mapping the upsampled latents. Must be in the range [0, 1].
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
  generation deterministic.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated video. Choose between `PIL.Image`, `np.array`, or `latent`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.0

Function invoked when calling the pipeline for latent upsampling.

**Parameters:**

video (`list[PipelineImageInput]`, *optional*) : The input video frames to upsample. Mutually exclusive with `latents`.

height (`int`, defaults to `512`) : The height in pixels of the upsampled output.

width (`int`, defaults to `704`) : The width in pixels of the upsampled output.

latents (`torch.Tensor`, *optional*) : Pre-encoded video latents to upsample. Mutually exclusive with `video`.

decode_timestep (`float` or `list[float]`, defaults to `0.0`) : The timestep at which the upsampled latents are decoded.

decode_noise_scale (`float` or `list[float]`, *optional*) : Interpolation factor between random noise and denoised latents at the decode timestep.

adain_factor (`float`, defaults to `0.0`) : Strength of AdaIN statistical matching applied to the upsampled latents.

tone_map_compression_ratio (`float`, defaults to `0.0`) : Compression ratio used for tone mapping the upsampled latents. Must be in the range [0, 1].

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated video. Choose between `PIL.Image`, `np.array`, or `latent`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.ltx.LTXPipelineOutput` instead of a plain tuple.
#### adain_filter_latent[[diffusers.LTXLatentUpsamplePipeline.adain_filter_latent]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L94)

Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
tensor.

**Parameters:**

latent (`torch.Tensor`) : Input latents to normalize

reference_latents (`torch.Tensor`) : The reference latents providing style statistics.

factor (`float`) : Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

**Returns:**

`torch.Tensor`

The transformed latent tensor
#### disable_vae_slicing[[diffusers.LTXLatentUpsamplePipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L189)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.LTXLatentUpsamplePipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L216)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.LTXLatentUpsamplePipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L176)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.LTXLatentUpsamplePipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L202)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.
#### tone_map_latents[[diffusers.LTXLatentUpsamplePipeline.tone_map_latents]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py#L122)

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

## LTXPipelineOutput[[diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput]]

#### diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput[[diffusers.pipelines.ltx.pipeline_output.LTXPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ltx/pipeline_output.py#L9)

Output class for LTX pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Hunyuan Video
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/hunyuan_video.md

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

  
    
      
    
  

# HunyuanVideo

[HunyuanVideo](https://huggingface.co/papers/2412.03603) is a 13B parameter diffusion transformer model designed to be competitive with closed-source video foundation models and enable wider community access. This model uses a "dual-stream to single-stream" architecture to separately process the video and text tokens first, before concatenating and feeding them to the transformer to fuse the multimodal information. A pretrained multimodal large language model (MLLM) is used as the encoder because it has better image-text alignment, better image detail description and reasoning, and it can be used as a zero-shot learner if system instructions are added to user prompts. Finally, HunyuanVideo uses a 3D causal variational autoencoder to more efficiently process video data at the original resolution and frame rate.

You can find all the original HunyuanVideo checkpoints under the [Tencent](https://huggingface.co/tencent) organization.

> [!TIP]
> Click on the HunyuanVideo models in the right sidebar for more examples of video generation tasks.
>
> The examples below use a checkpoint from [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) because the weights are stored in a layout compatible with Diffusers.

The example below demonstrates how to generate a video optimized for memory or inference speed.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The quantized HunyuanVideo model below requires ~14GB of VRAM.

```py
import torch
from diffusers import AutoModel, HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

# quantize weights to int4 with bitsandbytes
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
      "load_in_4bit": True,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": torch.bfloat16
      },
    components_to_quantize="transformer"
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster.

```py
import torch
from diffusers import AutoModel, HunyuanVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

# quantize weights to int4 with bitsandbytes
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
      "load_in_4bit": True,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": torch.bfloat16
      },
    components_to_quantize="transformer"
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

## Notes

- HunyuanVideo supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.39.0/en/api/loaders/lora#diffusers.loaders.HunyuanVideoLoraLoaderMixin.load_lora_weights).

  
  Show example code

  ```py
  import torch
  from diffusers import AutoModel, HunyuanVideoPipeline
  from diffusers.quantizers import PipelineQuantizationConfig
  from diffusers.utils import export_to_video

  # quantize weights to int4 with bitsandbytes
  pipeline_quant_config = PipelineQuantizationConfig(
      quant_backend="bitsandbytes_4bit",
      quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16
        },
      components_to_quantize="transformer"
  )

  pipeline = HunyuanVideoPipeline.from_pretrained(
      "hunyuanvideo-community/HunyuanVideo",
      quantization_config=pipeline_quant_config,
      torch_dtype=torch.bfloat16,
  )

  # load LoRA weights
  pipeline.load_lora_weights("https://huggingface.co/lucataco/hunyuan-steamboat-willie-10", adapter_name="steamboat-willie")
  pipeline.set_adapters("steamboat-willie", 0.9)

  # model-offloading and tiling
  pipeline.enable_model_cpu_offload()
  pipeline.vae.enable_tiling()

  # use "In the style of SWR" to trigger the LoRA
  prompt = """
  In the style of SWR. A black and white animated scene featuring a fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys.
  """
  video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
  export_to_video(video, "output.mp4", fps=15)
  ```

  

- Refer to the table below for recommended inference values.

  | parameter | recommended value |
  |---|---|
  | text encoder dtype | `torch.float16` |
  | transformer dtype | `torch.bfloat16` |
  | vae dtype | `torch.float16` |
  | `num_frames (k)` | 4 * `k` + 1 |

- Try lower `shift` values (`2.0` to `5.0`) for lower resolution videos and higher `shift` values (`7.0` to `12.0`) for higher resolution images.

## HunyuanVideoPipeline[[diffusers.HunyuanVideoPipeline]]

#### diffusers.HunyuanVideoPipeline[[diffusers.HunyuanVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L144)

Pipeline for text-to-video generation using HunyuanVideo.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.HunyuanVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L491[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "prompt_2", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt_2", "val": ": str | list[str] = None"}, {"name": "height", "val": ": int = 720"}, {"name": "width", "val": ": int = 1280"}, {"name": "num_frames", "val": ": int = 129"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "sigmas", "val": ": list = None"}, {"name": "true_cfg_scale", "val": ": float = 1.0"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "pooled_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_pooled_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "prompt_template", "val": ": dict = {'template': 'system\\n\\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:user\\n\\n{}', 'crop_start': 95}"}, {"name": "max_sequence_length", "val": ": int = 256"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **prompt_2** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
  will be used instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
  not greater than `1`).
- **negative_prompt_2** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
  `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
- **height** (`int`, defaults to `720`) --
  The height in pixels of the generated image.
- **width** (`int`, defaults to `1280`) --
  The width in pixels of the generated image.
- **num_frames** (`int`, defaults to `129`) --
  The number of frames in the generated video.
- **num_inference_steps** (`int`, defaults to `50`) --
  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
  expense of slower inference.
- **sigmas** (`list[float]`, *optional*) --
  Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
  their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
  will be used.
- **true_cfg_scale** (`float`, *optional*, defaults to 1.0) --
  True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
  `negative_prompt` is provided.
- **guidance_scale** (`float`, defaults to `6.0`) --
  Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
  a model to generate images more aligned with `prompt` at the expense of lower image quality.

  Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
  the [paper](https://huggingface.co/papers/2210.03142) to learn more.
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
  Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
  provided, text embeddings are generated from the `prompt` input argument.
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
- **prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `prompt_embeds`. Required when `prompt_embeds` is passed directly.
- **negative_prompt_attention_mask** (`torch.Tensor`, *optional*) --
  Attention mask for `negative_prompt_embeds`. Required when `negative_prompt_embeds` is passed directly.
- **output_type** (`str`, *optional*, defaults to `"pil"`) --
  The output format of the generated image. Choose between `PIL.Image` or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `HunyuanVideoPipelineOutput` instead of a plain tuple.
- **attention_kwargs** (`dict`, *optional*) --
  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
  `self.processor` in
  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
- **prompt_template** (`dict`, *optional*) --
  Template used to format the prompt before encoding. Defaults to the model's default template.
- **max_sequence_length** (`int`, *optional*, defaults to 256) --
  Maximum sequence length to use for the prompt encoder.
- **callback_on_step_end** (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) --
  A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
  each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
  DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
  list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`list`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.0`~HunyuanVideoPipelineOutput` or `tuple`If `return_dict` is `True`, `HunyuanVideoPipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo"
>>> transformer = HunyuanVideoTransformer3DModel.from_pretrained(
...     model_id, subfolder="transformer", torch_dtype=torch.bfloat16
... )
>>> pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> output = pipe(
...     prompt="A cat walks on the grass, realistic",
...     height=320,
...     width=512,
...     num_frames=61,
...     num_inference_steps=30,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=15)
```

**Parameters:**

text_encoder (`LlamaModel`) : [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

tokenizer (`LlamaTokenizer`) : Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

transformer ([HunyuanVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/hunyuan_video_transformer_3d#diffusers.HunyuanVideoTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLHunyuanVideo](/docs/diffusers/v0.39.0/en/api/models/autoencoder_kl_hunyuan_video#diffusers.AutoencoderKLHunyuanVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder_2 (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

tokenizer_2 (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

**Returns:**

``~HunyuanVideoPipelineOutput` or `tuple``

If `return_dict` is `True`, `HunyuanVideoPipelineOutput` is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images and the second element is a list of `bool`s
indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
#### disable_vae_slicing[[diffusers.HunyuanVideoPipeline.disable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L431)

Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
computing decoding in one step.
#### disable_vae_tiling[[diffusers.HunyuanVideoPipeline.disable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L458)

Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
computing decoding in one step.
#### enable_vae_slicing[[diffusers.HunyuanVideoPipeline.enable_vae_slicing]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L418)

Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
#### enable_vae_tiling[[diffusers.HunyuanVideoPipeline.enable_vae_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L444)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.

## HunyuanVideoPipelineOutput[[diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput]]

#### diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput[[diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/hunyuan_video/pipeline_output.py#L11)

Output class for HunyuanVideo pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Sana Video
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/sana_video.md

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

# Sana-Video

  
  

[SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer](https://huggingface.co/papers/2509.24695) from NVIDIA and MIT HAN Lab, by Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, Enze Xie.

The abstract from the paper is:

*We introduce SANA-Video, a small diffusion model that can efficiently generate videos up to 720x1280 resolution and minute-length duration. SANA-Video synthesizes high-resolution, high-quality and long videos with strong text-video alignment at a remarkably fast speed, deployable on RTX 5090 GPU. Two core designs ensure our efficient, effective and long video generation: (1) Linear DiT: We leverage linear attention as the core operation, which is more efficient than vanilla attention given the large number of tokens processed in video generation. (2) Constant-Memory KV cache for Block Linear Attention: we design block-wise autoregressive approach for long video generation by employing a constant-memory state, derived from the cumulative properties of linear attention. This KV cache provides the Linear DiT with global context at a fixed memory cost, eliminating the need for a traditional KV cache and enabling efficient, minute-long video generation. In addition, we explore effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs, which is only 1% of the cost of MovieGen. Given its low cost, SANA-Video achieves competitive performance compared to modern state-of-the-art small diffusion models (e.g., Wan 2.1-1.3B and SkyReel-V2-1.3B) while being 16x faster in measured latency. Moreover, SANA-Video can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (2.4x speedup). In summary, SANA-Video enables low-cost, high-quality video generation. [this https URL](https://github.com/NVlabs/SANA).*

This pipeline was contributed by SANA Team. The original codebase can be found [here](https://github.com/NVlabs/Sana). The original weights can be found under [hf.co/Efficient-Large-Model](https://hf.co/collections/Efficient-Large-Model/sana-video).

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`](https://huggingface.co/Efficient-Large-Model/ANA-Video_2B_480p_diffusers) | `torch.bfloat16` |

Refer to [this](https://huggingface.co/collections/Efficient-Large-Model/sana-video) collection for more information.

Note: The recommended dtype mentioned is for the transformer weights. The text encoder and VAE weights must stay in `torch.bfloat16` or `torch.float32` for the model to work correctly. Please refer to the inference example below to see how to load the model with the recommended dtype. 

## Generation Pipelines

`

The example below demonstrates how to use the text-to-video pipeline to generate a video using a text description.

```python
pipe = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers", 
    torch_dtype=torch.bfloat16,
)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_scale = 30
motion_prompt = f" motion score: {motion_scale}."
prompt = prompt + motion_prompt

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]

export_to_video(video, "sana_video.mp4", fps=16)
```

The example below demonstrates how to use the image-to-video pipeline to generate a video using a text description and a starting frame.

```python
pipe = SanaImageToVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")

image = load_image("https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/samples/i2v-1.png")
prompt = "A woman stands against a stunning sunset backdrop, her long, wavy brown hair gently blowing in the breeze. She wears a sleeveless, light-colored blouse with a deep V-neckline, which accentuates her graceful posture. The warm hues of the setting sun cast a golden glow across her face and hair, creating a serene and ethereal atmosphere. The background features a blurred landscape with soft, rolling hills and scattered clouds, adding depth to the scene. The camera remains steady, capturing the tranquil moment from a medium close-up angle."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_scale = 30
motion_prompt = f" motion score: {motion_scale}."
prompt = prompt + motion_prompt

motion_scale = 30.0

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]

export_to_video(video, "sana-i2v.mp4", fps=16)
```

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [SanaVideoPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.SanaVideoPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaVideoTransformer3DModel, SanaVideoPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaVideoTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

model_score = 30
prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_prompt = f" motion score: {model_score}."
prompt = prompt + motion_prompt

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=6.0,
    num_inference_steps=50
).frames[0]
export_to_video(output, "sana-video-output.mp4", fps=16)
```

## SanaVideoPipeline[[diffusers.SanaVideoPipeline]]

#### diffusers.SanaVideoPipeline[[diffusers.SanaVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py#L186)

Pipeline for text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). This model inherits
from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all
pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.SanaVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py#L712[{"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str = ''"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list = None"}, {"name": "sigmas", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "height", "val": ": int = 480"}, {"name": "width", "val": ": int = 832"}, {"name": "frames", "val": ": int = 81"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "clean_caption", "val": ": bool = False"}, {"name": "use_resolution_binning", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 300"}, {"name": "complex_human_instruction", "val": ": list = [\"Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:\", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: ']"}]- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the video generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality video at the
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
  `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to
  the text `prompt`, usually at the expense of lower video quality.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **height** (`int`, *optional*, defaults to 480) --
  The height in pixels of the generated video.
- **width** (`int`, *optional*, defaults to 832) --
  The width in pixels of the generated video.
- **frames** (`int`, *optional*, defaults to 81) --
  The number of frames in the generated video.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
  applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
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
  The output format of the generated video. Choose between mp4 or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `SanaVideoPipelineOutput` instead of a plain tuple.
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
  `ASPECT_RATIO_480_BIN` or `ASPECT_RATIO_720_BIN`. After the produced latents are decoded into videos,
  they are resized back to the requested resolution. Useful for generating non-square videos.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int` defaults to `300`) --
  Maximum sequence length to use with the `prompt`.
- **complex_human_instruction** (`list[str]`, *optional*) --
  Instructions for complex human attention:
  https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.0[SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) or `tuple`If `return_dict` is `True`, [SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) is
returned, otherwise a `tuple` is returned where the first element is a list with the generated videos

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaVideoPipeline
>>> from diffusers.utils import export_to_video

>>> pipe = SanaVideoPipeline.from_pretrained("Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
>>> pipe.transformer.to(torch.bfloat16)
>>> pipe.text_encoder.to(torch.bfloat16)
>>> pipe.vae.to(torch.float32)
>>> pipe.to("cuda")
>>> motion_score = 30

>>> prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
>>> negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
>>> motion_prompt = f" motion score: {motion_score}."
>>> prompt = prompt + motion_prompt

>>> output = pipe(
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     height=480,
...     width=832,
...     frames=81,
...     guidance_scale=6,
...     num_inference_steps=50,
...     generator=torch.Generator(device="cuda").manual_seed(42),
... ).frames[0]

>>> export_to_video(output, "sana-video-output.mp4", fps=16)
```

**Parameters:**

tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`) : The tokenizer used to tokenize the prompt.

text_encoder (`Gemma2PreTrainedModel`) : Text encoder model to encode the input prompts.

vae ([`AutoencoderKLWan`, `AutoencoderDC`, or `AutoencoderKLLTX2Video`]) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

transformer ([SanaVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/sana_video_transformer3d#diffusers.SanaVideoTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([DPMSolverMultistepScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) or `tuple``

If `return_dict` is `True`, [SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) is
returned, otherwise a `tuple` is returned where the first element is a list with the generated videos
#### encode_prompt[[diffusers.SanaVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py#L303)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For PixArt-Alpha, this should be "".

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : whether to use classifier free guidance or not

num_videos_per_prompt (`int`, *optional*, defaults to 1) : number of videos that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

## SanaImageToVideoPipeline[[diffusers.SanaImageToVideoPipeline]]

#### diffusers.SanaImageToVideoPipeline[[diffusers.SanaImageToVideoPipeline]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py#L176)

Pipeline for image/text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). This model
inherits from [DiffusionPipeline](/docs/diffusers/v0.39.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all
pipelines (downloading, saving, running on a particular device, etc.).

__call__diffusers.SanaImageToVideoPipeline.__call__https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py#L739[{"name": "image", "val": ": PIL.Image.Image | numpy.ndarray | torch.Tensor | list[PIL.Image.Image] | list[numpy.ndarray] | list[torch.Tensor]"}, {"name": "prompt", "val": ": str | list[str] = None"}, {"name": "negative_prompt", "val": ": str = ''"}, {"name": "num_inference_steps", "val": ": int = 50"}, {"name": "timesteps", "val": ": list = None"}, {"name": "sigmas", "val": ": list = None"}, {"name": "guidance_scale", "val": ": float = 6.0"}, {"name": "num_videos_per_prompt", "val": ": int | None = 1"}, {"name": "height", "val": ": int = 480"}, {"name": "width", "val": ": int = 832"}, {"name": "frames", "val": ": int = 81"}, {"name": "eta", "val": ": float = 0.0"}, {"name": "generator", "val": ": torch._C.Generator | list[torch._C.Generator] | None = None"}, {"name": "latents", "val": ": torch.Tensor | None = None"}, {"name": "prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_embeds", "val": ": torch.Tensor | None = None"}, {"name": "negative_prompt_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "output_type", "val": ": str | None = 'pil'"}, {"name": "return_dict", "val": ": bool = True"}, {"name": "clean_caption", "val": ": bool = False"}, {"name": "use_resolution_binning", "val": ": bool = True"}, {"name": "attention_kwargs", "val": ": dict[str, typing.Any] | None = None"}, {"name": "callback_on_step_end", "val": ": typing.Optional[typing.Callable[[int, int], NoneType]] = None"}, {"name": "callback_on_step_end_tensor_inputs", "val": ": list = ['latents']"}, {"name": "max_sequence_length", "val": ": int = 300"}, {"name": "complex_human_instruction", "val": ": list = [\"Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:\", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: ']"}]- **image** (`PipelineImageInput`) --
  The input image to condition the video generation on. The first frame of the generated video will be
  conditioned on this image.
- **prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
  instead.
- **negative_prompt** (`str` or `list[str]`, *optional*) --
  The prompt or prompts not to guide the video generation. If not defined, one has to pass
  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
  less than `1`).
- **num_inference_steps** (`int`, *optional*, defaults to 50) --
  The number of denoising steps. More denoising steps usually lead to a higher quality video at the
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
  `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to
  the text `prompt`, usually at the expense of lower video quality.
- **num_videos_per_prompt** (`int`, *optional*, defaults to 1) --
  The number of videos to generate per prompt.
- **height** (`int`, *optional*, defaults to 480) --
  The height in pixels of the generated video.
- **width** (`int`, *optional*, defaults to 832) --
  The width in pixels of the generated video.
- **frames** (`int`, *optional*, defaults to 81) --
  The number of frames in the generated video.
- **eta** (`float`, *optional*, defaults to 0.0) --
  Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
  applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.
- **generator** (`torch.Generator` or `list[torch.Generator]`, *optional*) --
  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
  to make generation deterministic.
- **latents** (`torch.Tensor`, *optional*) --
  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
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
  The output format of the generated video. Choose between mp4 or `np.array`.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `SanaVideoPipelineOutput` instead of a plain tuple.
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
  `ASPECT_RATIO_480_BIN` or `ASPECT_RATIO_720_BIN`. After the produced latents are decoded into videos,
  they are resized back to the requested resolution. Useful for generating non-square videos.
- **callback_on_step_end** (`Callable`, *optional*) --
  A function that calls at the end of each denoising steps during the inference. The function is called
  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
  callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
  `callback_on_step_end_tensor_inputs`.
- **callback_on_step_end_tensor_inputs** (`List`, *optional*) --
  The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
  will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
  `._callback_tensor_inputs` attribute of your pipeline class.
- **max_sequence_length** (`int` defaults to `300`) --
  Maximum sequence length to use with the `prompt`.
- **complex_human_instruction** (`list[str]`, *optional*) --
  Instructions for complex human attention:
  https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.0[SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) or `tuple`If `return_dict` is `True`, [SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) is
returned, otherwise a `tuple` is returned where the first element is a list with the generated videos

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import SanaImageToVideoPipeline
>>> from diffusers.utils import export_to_video, load_image

>>> pipe = SanaImageToVideoPipeline.from_pretrained("Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
>>> pipe.transformer.to(torch.bfloat16)
>>> pipe.text_encoder.to(torch.bfloat16)
>>> pipe.vae.to(torch.float32)
>>> pipe.to("cuda")
>>> motion_score = 30

>>> prompt = "A woman stands against a stunning sunset backdrop, her long, wavy brown hair gently blowing in the breeze. She wears a sleeveless, light-colored blouse with a deep V-neckline, which accentuates her graceful posture. The warm hues of the setting sun cast a golden glow across her face and hair, creating a serene and ethereal atmosphere. The background features a blurred landscape with soft, rolling hills and scattered clouds, adding depth to the scene. The camera remains steady, capturing the tranquil moment from a medium close-up angle."
>>> negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
>>> motion_prompt = f" motion score: {motion_score}."
>>> prompt = prompt + motion_prompt
>>> image = load_image("https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/samples/i2v-1.png")

>>> output = pipe(
...     image=image,
...     prompt=prompt,
...     negative_prompt=negative_prompt,
...     height=480,
...     width=832,
...     frames=81,
...     guidance_scale=6,
...     num_inference_steps=50,
...     generator=torch.Generator(device="cuda").manual_seed(42),
... ).frames[0]

>>> export_to_video(output, "sana-ti2v-output.mp4", fps=16)
```

**Parameters:**

tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`) : The tokenizer used to tokenize the prompt.

text_encoder (`Gemma2PreTrainedModel`) : Text encoder model to encode the input prompts.

vae ([`AutoencoderKLWan`, `AutoencoderDC`, or `AutoencoderKLLTX2Video`]) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

transformer ([SanaVideoTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/sana_video_transformer3d#diffusers.SanaVideoTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.39.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

**Returns:**

`[SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) or `tuple``

If `return_dict` is `True`, [SanaVideoPipelineOutput](/docs/diffusers/v0.39.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) is
returned, otherwise a `tuple` is returned where the first element is a list with the generated videos
#### encode_prompt[[diffusers.SanaImageToVideoPipeline.encode_prompt]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py#L301)

Encodes the prompt into text encoder hidden states.

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

negative_prompt (`str` or `list[str]`, *optional*) : The prompt not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For PixArt-Alpha, this should be "".

do_classifier_free_guidance (`bool`, *optional*, defaults to `True`) : whether to use classifier free guidance or not

num_videos_per_prompt (`int`, *optional*, defaults to 1) : number of videos that should be generated per prompt

device : (`torch.device`, *optional*): torch device to place the resulting embeddings on

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For Sana, it's should be the embeddings of the "" string.

clean_caption (`bool`, defaults to `False`) : If `True`, the function will preprocess and clean the provided caption before encoding.

max_sequence_length (`int`, defaults to 300) : Maximum sequence length to use for the prompt.

complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`) : If `complex_human_instruction` is not empty, the function will use the complex Human instruction for the prompt.

## SanaVideoPipelineOutput[[diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput]]

#### diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput[[diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/sana_video/pipeline_output.py#L9)

Output class for Sana-Video pipelines.

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

### Kandinsky 2.2
https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/kandinsky_v22.md
