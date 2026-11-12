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
    dtype=torch.bfloat16,
)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

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
    dtype=torch.bfloat16,
)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

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

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [SanaVideoPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/sana_video#diffusers.SanaVideoPipeline) for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaVideoTransformer3DModel, SanaVideoPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaVideoTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    dtype=torch.float16,
)

pipeline = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    dtype=torch.float16,
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

```python
diffusers.SanaVideoPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC | diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video | diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan, transformer: SanaVideoTransformer3DModel, scheduler: DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py#L186)

**Parameters:**

tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`) : The tokenizer used to tokenize the prompt.

text_encoder (`Gemma2PreTrainedModel`) : Text encoder model to encode the input prompts.

vae ([`AutoencoderKLWan`, `AutoencoderDC`, or `AutoencoderKLLTX2Video`]) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

transformer ([SanaVideoTransformer3DModel](/docs/diffusers/v0.43.0/en/api/models/sana_video_transformer3d#diffusers.SanaVideoTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([DPMSolverMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

Pipeline for text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). This model inherits
from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all
pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.SanaVideoPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str = '', num_inference_steps: int = 50, timesteps: list[int] = None, sigmas: list[float] = None, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, height: int = 480, width: int = 832, frames: int = 81, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list[str] = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py#L712)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`, usually at the expense of lower video quality.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

height (`int`, *optional*, defaults to 480) : The height in pixels of the generated video.

width (`int`, *optional*, defaults to 832) : The width in pixels of the generated video.

frames (`int`, *optional*, defaults to 81) : The number of frames in the generated video.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated video. Choose between mp4 or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `SanaVideoPipelineOutput` instead of a plain tuple.

attention_kwargs : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

clean_caption (`bool`, *optional*, defaults to `True`) : Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to be installed. If the dependencies are not installed, the embeddings will be created from the raw prompt.

use_resolution_binning (`bool` defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using `ASPECT_RATIO_480_BIN` or `ASPECT_RATIO_720_BIN`. After the produced latents are decoded into videos, they are resized back to the requested resolution. Useful for generating non-square videos.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to `300`) : Maximum sequence length to use with the `prompt`.

complex_human_instruction (`list[str]`, *optional*) : Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

**Returns:** [SanaVideoPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) or `tuple`

If `return_dict` is `True`, [SanaVideoPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) is
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

#### encode_prompt[[diffusers.SanaVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], do_classifier_free_guidance: bool = True, negative_prompt: str = '', num_videos_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_sana_video.py#L303)

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

Encodes the prompt into text encoder hidden states.

## SanaImageToVideoPipeline[[diffusers.SanaImageToVideoPipeline]]

#### diffusers.SanaImageToVideoPipeline[[diffusers.SanaImageToVideoPipeline]]

```python
diffusers.SanaImageToVideoPipeline(tokenizer: GemmaTokenizer, text_encoder: Gemma2PreTrainedModel, vae: diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC | diffusers.models.autoencoders.autoencoder_kl_ltx2.AutoencoderKLLTX2Video | diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan, transformer: SanaVideoTransformer3DModel, scheduler: FlowMatchEulerDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py#L176)

**Parameters:**

tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`) : The tokenizer used to tokenize the prompt.

text_encoder (`Gemma2PreTrainedModel`) : Text encoder model to encode the input prompts.

vae ([`AutoencoderKLWan`, `AutoencoderDC`, or `AutoencoderKLLTX2Video`]) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

transformer ([SanaVideoTransformer3DModel](/docs/diffusers/v0.43.0/en/api/models/sana_video_transformer3d#diffusers.SanaVideoTransformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

Pipeline for image/text-to-video generation using [Sana](https://huggingface.co/papers/2509.24695). This model
inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods implemented for all
pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.SanaImageToVideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str = '', num_inference_steps: int = 50, timesteps: list[int] = None, sigmas: list[float] = None, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, height: int = 480, width: int = 832, frames: int = 81, eta: float = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, clean_caption: bool = False, use_resolution_binning: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 300, complex_human_instruction: list[str] = ["Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:", '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: '])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py#L739)

**Parameters:**

image (`PipelineImageInput`) : The input image to condition the video generation on. The first frame of the generated video will be conditioned on this image.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`. instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

guidance_scale (`float`, *optional*, defaults to 4.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`, usually at the expense of lower video quality.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

height (`int`, *optional*, defaults to 480) : The height in pixels of the generated video.

width (`int`, *optional*, defaults to 832) : The width in pixels of the generated video.

frames (`int`, *optional*, defaults to 81) : The number of frames in the generated video.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to [schedulers.DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), will be ignored for others.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for text embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Pre-generated attention mask for negative text embeddings.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated video. Choose between mp4 or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `SanaVideoPipelineOutput` instead of a plain tuple.

attention_kwargs : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

clean_caption (`bool`, *optional*, defaults to `True`) : Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to be installed. If the dependencies are not installed, the embeddings will be created from the raw prompt.

use_resolution_binning (`bool` defaults to `True`) : If set to `True`, the requested height and width are first mapped to the closest resolutions using `ASPECT_RATIO_480_BIN` or `ASPECT_RATIO_720_BIN`. After the produced latents are decoded into videos, they are resized back to the requested resolution. Useful for generating non-square videos.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`List`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

max_sequence_length (`int` defaults to `300`) : Maximum sequence length to use with the `prompt`.

complex_human_instruction (`list[str]`, *optional*) : Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

**Returns:** [SanaVideoPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) or `tuple`

If `return_dict` is `True`, [SanaVideoPipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/sana_video#diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput) is
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

#### encode_prompt[[diffusers.SanaImageToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], do_classifier_free_guidance: bool = True, negative_prompt: str = '', num_videos_per_prompt: int = 1, device: typing.Optional[torch.device] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, clean_caption: bool = False, max_sequence_length: int = 300, complex_human_instruction: list[str] | None = None, lora_scale: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py#L301)

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

Encodes the prompt into text encoder hidden states.

## SanaVideoPipelineOutput[[diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput]]

#### diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput[[diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput]]

```python
diffusers.pipelines.sana_video.pipeline_output.SanaVideoPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/sana_video/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for Sana-Video pipelines.

### Omnigen
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/omnigen.md

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

# OmniGen

[OmniGen: Unified Image Generation](https://huggingface.co/papers/2409.11340) from BAAI, by Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, Zheng Liu.

The abstract from the paper is:

*The emergence of Large Language Models (LLMs) has unified language  generation tasks and revolutionized human-machine interaction.  However, in the realm of image generation, a unified model capable of handling various tasks within a single framework remains largely unexplored. In this work, we introduce OmniGen, a new diffusion model for unified image generation. OmniGen is characterized by the following features: 1) Unification: OmniGen not only demonstrates text-to-image generation capabilities but also inherently supports various downstream tasks, such as image editing, subject-driven generation, and visual conditional generation. 2) Simplicity: The architecture of OmniGen is highly simplified, eliminating the need for additional plugins. Moreover, compared to existing diffusion models, it is more user-friendly and can complete complex tasks end-to-end through instructions without the need for extra intermediate steps, greatly simplifying the image generation workflow. 3) Knowledge Transfer: Benefit from learning in a unified format, OmniGen effectively transfers knowledge across different tasks, manages unseen tasks and domains, and exhibits novel capabilities. We also explore the model’s reasoning capabilities and potential applications of the chain-of-thought mechanism.  This work represents the first attempt at a general-purpose image generation model,  and we will release our resources at https://github.com/VectorSpaceLab/OmniGen to foster future advancements.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [staoxiao](https://github.com/staoxiao). The original codebase can be found [here](https://github.com/VectorSpaceLab/OmniGen). The original weights can be found under [hf.co/shitao](https://huggingface.co/Shitao/OmniGen-v1).

## Load model checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [from_pretrained()](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) method.

```python
import torch
from diffusers import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1-diffusers", dtype=torch.bfloat16)
```

## Text-to-image

For text-to-image, pass a text prompt. By default, OmniGen generates a 1024x1024 image. 
You can try setting the `height` and `width` parameters to generate images with different size.

```python
import torch
from diffusers import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt = "Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3,
    generator=torch.Generator(device="cpu").manual_seed(111),
).images[0]
image.save("output.png")
```

    

## Image edit

OmniGen supports multimodal inputs. 
When the input includes an image, you need to add a placeholder `<img><|image_1|></img>` in the text prompt to represent the image. 
It is recommended to enable `use_input_image_size_as_output` to keep the edited image the same size as the original image.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt="<img><|image_1|></img> Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/t2i_woman_with_book.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(222)
).images[0]
image.save("output.png")
```

  
    
    original image
  
  
    
    edited image
  

OmniGen has some interesting features, such as visual reasoning, as shown in the example below.

```python
prompt="If the woman is thirsty, what should she take? Find it in the image and highlight it in blue. <img><|image_1|></img>"
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]
image.save("output.png")
```

    

## Controllable generation

OmniGen can handle several classic computer vision tasks. As shown below, OmniGen can detect human skeletons in input images, which can be used as control conditions to generate new images.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt="Detect the skeleton of human in this image: <img><|image_1|></img>"
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image1 = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(333)
).images[0]
image1.save("image1.png")

prompt="Generate a new photo using the following picture and text as conditions: <img><|image_1|></img>\n A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/skeletal.png")]
image2 = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(333)
).images[0]
image2.save("image2.png")
```

  
    
    original image
  
  
    
    detected skeleton
  
  
    
    skeleton to image
  

OmniGen can also directly use relevant information from input images to generate new images.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt="Following the pose of this image <img><|image_1|></img>, generate a new photo: A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him."
input_images=[load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/edit.png")]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    guidance_scale=2, 
    img_guidance_scale=1.6,
    use_input_image_size_as_output=True,
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]
image.save("output.png")
```

  
    
    generated image
  

## ID and object preserving

OmniGen can generate multiple images based on the people and objects in the input image and supports inputting multiple images simultaneously. 
Additionally, OmniGen can extract desired objects from an image containing multiple objects based on instructions.

```python
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt="A man and a woman are sitting at a classroom desk. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>"
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/3.png")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/4.png")
input_images=[input_image_1, input_image_2]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    height=1024,
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator(device="cpu").manual_seed(666)
).images[0]
image.save("output.png")
```

  
    
    input_image_1
  
  
    
    input_image_2
  
  
    
    generated image
  

```py
import torch
from diffusers import OmniGenPipeline
from diffusers.utils import load_image 

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1-diffusers",
    dtype=torch.bfloat16
)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt="A woman is walking down the street, wearing a white long-sleeve blouse with lace details on the sleeves, paired with a blue pleated skirt. The woman is <img><|image_1|></img>. The long-sleeve blouse and a pleated skirt are <img><|image_2|></img>."
input_image_1 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/emma.jpeg")
input_image_2 = load_image("https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/imgs/docs_img/dress.jpg")
input_images=[input_image_1, input_image_2]
image = pipe(
    prompt=prompt, 
    input_images=input_images, 
    height=1024,
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    generator=torch.Generator(device="cpu").manual_seed(666)
).images[0]
image.save("output.png")
```

  
    
    person image
  
  
    
    clothe image
  
  
    
    generated image
  

## Optimization when using multiple images 

For text-to-image task, OmniGen requires minimal memory and time costs (9GB memory and 31s for a 1024x1024 image on A800 GPU). 
However, when using input images, the computational cost increases. 

Here are some guidelines to help you reduce computational costs when using multiple images. The experiments are conducted on an A800 GPU with two input images.

Like other pipelines, you can reduce memory usage by offloading the model: `pipe.enable_model_cpu_offload()` or `pipe.enable_sequential_cpu_offload() `. 
In OmniGen, you can also decrease computational overhead by reducing the `max_input_image_size`. 
The memory consumption for different image sizes is shown in the table below:

| Method                    | Memory Usage |
|---------------------------|--------------|
| max_input_image_size=1024 | 40GB         |
| max_input_image_size=512  | 17GB         |
| max_input_image_size=256  | 14GB         |

## OmniGenPipeline[[diffusers.OmniGenPipeline]]

#### diffusers.OmniGenPipeline[[diffusers.OmniGenPipeline]]

```python
diffusers.OmniGenPipeline(transformer: OmniGenTransformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, tokenizer: LlamaTokenizer)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/omnigen/pipeline_omnigen.py#L119)

**Parameters:**

transformer ([OmniGenTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/omnigen_transformer#diffusers.OmniGenTransformer2DModel)) : Autoregressive Transformer architecture for OmniGen.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKL](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl#diffusers.AutoencoderKL)) : Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

tokenizer (`LlamaTokenizer`) : Text tokenizer of class. [LlamaTokenizer](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaTokenizer).

The OmniGen pipeline for multimodal-to-image generation.

Reference: https://huggingface.co/papers/2409.11340

#### __call__[[diffusers.OmniGenPipeline.__call__]]

```python
__call__(prompt: str | list[str], input_images: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor], list[typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]]] = None, height: int | None = None, width: int | None = None, num_inference_steps: int = 50, max_input_image_size: int = 1024, timesteps: list[int] = None, guidance_scale: float = 2.5, img_guidance_scale: float = 1.6, use_input_image_size_as_output: bool = False, num_images_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, callback_on_step_end: typing.Optional[typing.Callable[[int, int], NoneType]] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/omnigen/pipeline_omnigen.py#L275)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If the input includes images, need to add placeholders `<img><|image_i|></img>` in the prompt to indicate the position of the i-th images.

input_images (`PipelineImageInput` or `list[PipelineImageInput]`, *optional*) : The list of input images. We will replace the "<|image_i|>" in prompt with the i-th image in list.

height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The height in pixels of the generated image. This is set to 1024 by default for the best results.

width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor) : The width in pixels of the generated image. This is set to 1024 by default for the best results.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

max_input_image_size (`int`, *optional*, defaults to 1024) : the maximum size of input image, which will be used to crop the input image to the maximum size

timesteps (`list[int]`, *optional*) : Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used. Must be in descending order.

guidance_scale (`float`, *optional*, defaults to 2.5) : Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.

img_guidance_scale (`float`, *optional*, defaults to 1.6) : Defined as equation 3 in [Instrucpix2pix](https://huggingface.co/papers/2211.09800).

use_input_image_size_as_output (bool, defaults to False) : whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task

num_images_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generate image. Choose between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~pipelines.flux.FluxPipelineOutput` instead of a plain tuple.

callback_on_step_end (`Callable`, *optional*) : A function that calls at the end of each denoising steps during the inference. The function is called with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) or `tuple`

[ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) if `return_dict` is True,
otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.

Function invoked when calling the pipeline for generation.

Examples:
```py
>>> import torch
>>> from diffusers import OmniGenPipeline

>>> pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1-diffusers", torch_dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> prompt = "A cat holding a sign that says hello world"
>>> # Depending on the variant being used, the pipeline call will slightly vary.
>>> # Refer to the pipeline documentation for more details.
>>> image = pipe(prompt, num_inference_steps=50, guidance_scale=2.5).images[0]
>>> image.save("output.png")
```

Returns: [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) or `tuple`:
If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is returned
where the first element is a list with the generated images.

#### encode_input_images[[diffusers.OmniGenPipeline.encode_input_images]]

```python
encode_input_images(input_pixel_values: list[torch.Tensor], device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/omnigen/pipeline_omnigen.py#L169)

**Parameters:**

input_pixel_values : normalized pixel of input images

device --

get the continue embedding of input images by VAE

Returns: torch.Tensor

### Hunyuan Video15
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/hunyuan_video15.md

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

# HunyuanVideo-1.5

HunyuanVideo-1.5 is a lightweight yet powerful video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture with selective and sliding tile attention (SSTA), enhanced bilingual understanding through glyph-aware text encoding, progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions. Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source models.

You can find all the original HunyuanVideo checkpoints under the [Tencent](https://huggingface.co/tencent) organization.

> [!TIP]
> Click on the HunyuanVideo models in the right sidebar for more examples of video generation tasks.
>
> The examples below use a checkpoint from [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) because the weights are stored in a layout compatible with Diffusers.

The example below demonstrates how to generate a video optimized for memory or inference speed.

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

```py
import torch
from diffusers import AutoModel, HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

pipeline = HunyuanVideo15Pipeline.from_pretrained(
    "HunyuanVideo-1.5-Diffusers-480p_t2v",
    dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

## Notes

- HunyuanVideo1.5 use attention masks with variable-length sequences. For best performance, we recommend using an attention backend that handles padding efficiently.

    - **H100/H800:** `_flash_3_hub` or `_flash_3_varlen_hub`
    - **A100/A800/RTX 4090:** `flash_hub` or `flash_varlen_hub`
    - **Other GPUs:** `sage_hub`

Refer to the [Attention backends](../../optimization/attention_backends) guide for more details about using a different backend.

```py
pipe.transformer.set_attention_backend("flash_hub")  # or your preferred backend
```

- [HunyuanVideo15Pipeline](/docs/diffusers/v0.43.0/en/api/pipelines/hunyuan_video15#diffusers.HunyuanVideo15Pipeline) use guider and does not take `guidance_scale` parameter at runtime. 

You can check the default guider configuration using `pipe.guider`:

```py
>>> pipe.guider 
ClassifierFreeGuidance {
  "_class_name": "ClassifierFreeGuidance",
  "_diffusers_version": "0.36.0.dev0",
  "enabled": true,
  "guidance_rescale": 0.0,
  "guidance_scale": 6.0,
  "start": 0.0,
  "stop": 1.0,
  "use_original_formulation": false
}

State:
  step: None
  num_inference_steps: None
  timestep: None
  count_prepared: 0
  enabled: True
  num_conditions: 2
```

To update guider configuration, you can run `pipe.guider = pipe.guider.new(...)`

```py
pipe.guider = pipe.guider.new(guidance_scale=5.0)
```

Read more on Guider [here](../../using-diffusers/guiders).

## HunyuanVideo15Pipeline[[diffusers.HunyuanVideo15Pipeline]]

#### diffusers.HunyuanVideo15Pipeline[[diffusers.HunyuanVideo15Pipeline]]

```python
diffusers.HunyuanVideo15Pipeline(text_encoder: Qwen2_5_VLTextModel, tokenizer: Qwen2Tokenizer, transformer: HunyuanVideo15Transformer3DModel, vae: AutoencoderKLHunyuanVideo15, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder_2: T5EncoderModel, tokenizer_2: ByT5Tokenizer, guider: ClassifierFreeGuidance)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L166)

**Parameters:**

transformer ([HunyuanVideo15Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.HunyuanVideo15Transformer3DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLHunyuanVideo15](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_hunyuan_video15#diffusers.AutoencoderKLHunyuanVideo15)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider ([ClassifierFreeGuidance](/docs/diffusers/v0.43.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance)) : [ClassifierFreeGuidance]for classifier free guidance.

Pipeline for text-to-video generation using HunyuanVideo1.5.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideo15Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int | None = None, width: int | None = None, num_frames: int = 121, num_inference_steps: int = 50, sigmas: list[float] = None, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L542)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead.

height (`int`, *optional*) : The height in pixels of the generated video.

width (`int`, *optional*) : The width in pixels of the generated video.

num_frames (`int`, defaults to `121`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings from the second text encoder.

negative_prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings from the second text encoder.

negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings from the second text encoder.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated video. Choose between "np", "pt", or "latent".

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `HunyuanVideo15PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

**Returns:** `~HunyuanVideo15PipelineOutput` or `tuple`

If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideo15Pipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v"
>>> pipe = HunyuanVideo15Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> output = pipe(
...     prompt="A cat walks on the grass, realistic",
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=15)
```

#### encode_prompt[[diffusers.HunyuanVideo15Pipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, batch_size: int = 1, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L334)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_videos_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

#### prepare_cond_latents_and_mask[[diffusers.HunyuanVideo15Pipeline.prepare_cond_latents_and_mask]]

```python
prepare_cond_latents_and_mask(latents, dtype: typing.Optional[torch.dtype], device: typing.Optional[torch.device])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L508)

**Parameters:**

latents : Main latents tensor (B, C, F, H, W)

**Returns:** `tuple`

(cond_latents_concat, mask_concat) - both are zero tensors for t2v

Prepare conditional latents and mask for t2v generation.

## HunyuanVideo15ImageToVideoPipeline[[diffusers.HunyuanVideo15ImageToVideoPipeline]]

#### diffusers.HunyuanVideo15ImageToVideoPipeline[[diffusers.HunyuanVideo15ImageToVideoPipeline]]

```python
diffusers.HunyuanVideo15ImageToVideoPipeline(text_encoder: Qwen2_5_VLTextModel, tokenizer: Qwen2Tokenizer, transformer: HunyuanVideo15Transformer3DModel, vae: AutoencoderKLHunyuanVideo15, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder_2: T5EncoderModel, tokenizer_2: ByT5Tokenizer, guider: ClassifierFreeGuidance, image_encoder: SiglipVisionModel, feature_extractor: SiglipImageProcessorPil)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L193)

**Parameters:**

transformer ([HunyuanVideo15Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.HunyuanVideo15Transformer3DModel)) : Conditional Transformer (MMDiT) architecture to denoise the encoded video latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded video latents.

vae ([AutoencoderKLHunyuanVideo15](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_hunyuan_video15#diffusers.AutoencoderKLHunyuanVideo15)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder (`Qwen2.5-VL-7B-Instruct`) : [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.

tokenizer (`Qwen2Tokenizer`) : Tokenizer of class [Qwen2Tokenizer].

text_encoder_2 (`T5EncoderModel`) : [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel) variant.

tokenizer_2 (`ByT5Tokenizer`) : Tokenizer of class [ByT5Tokenizer]

guider ([ClassifierFreeGuidance](/docs/diffusers/v0.43.0/en/api/modular_diffusers/guiders#diffusers.ClassifierFreeGuidance)) : [ClassifierFreeGuidance]for classifier free guidance.

image_encoder (`SiglipVisionModel`) : [SiglipVisionModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipVisionModel) variant.

feature_extractor (`SiglipImageProcessor`) : [SiglipImageProcessor](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipImageProcessor) variant.

Pipeline for image-to-video generation using HunyuanVideo1.5.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideo15ImageToVideoPipeline.__call__]]

```python
__call__(image: Image, prompt: str | list[str] = None, negative_prompt: str | list[str] = None, num_frames: int = 121, num_inference_steps: int = 50, sigmas: list[float] = None, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_2: typing.Optional[torch.Tensor] = None, negative_prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L650)

**Parameters:**

image (`PIL.Image.Image`) : The input image to condition video generation on.

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds` instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead.

num_frames (`int`, defaults to `121`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of videos to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings.

negative_prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated text embeddings from the second text encoder. Can be used to easily tweak text inputs.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for prompt embeddings from the second text encoder.

negative_prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated negative text embeddings from the second text encoder.

negative_prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated mask for negative prompt embeddings from the second text encoder.

output_type (`str`, *optional*, defaults to `"np"`) : The output format of the generated video. Choose between "np", "pt", or "latent".

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `HunyuanVideo15PipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

**Returns:** `~HunyuanVideo15PipelineOutput` or `tuple`

If `return_dict` is `True`, `HunyuanVideo15PipelineOutput` is returned, otherwise a `tuple` is
returned where the first element is a list with the generated videos.

The call function to the pipeline for generation.

Examples:
```python
>>> import torch
>>> from diffusers import HunyuanVideo15ImageToVideoPipeline
>>> from diffusers.utils import export_to_video

>>> model_id = "hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v"
>>> pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
>>> pipe.vae.enable_tiling()
>>> pipe.to("cuda")

>>> image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")

>>> output = pipe(
...     prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
...     image=image,
...     num_inference_steps=50,
... ).frames[0]
>>> export_to_video(output, "output.mp4", fps=24)
```

#### encode_prompt[[diffusers.HunyuanVideo15ImageToVideoPipeline.encode_prompt]]

```python
encode_prompt(prompt: str | list[str], device: typing.Optional[torch.device] = None, dtype: typing.Optional[torch.dtype] = None, batch_size: int = 1, num_videos_per_prompt: int = 1, prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_embeds_mask: typing.Optional[torch.Tensor] = None, prompt_embeds_2: typing.Optional[torch.Tensor] = None, prompt_embeds_mask_2: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L422)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : prompt to be encoded

device : (`torch.device`): torch device

batch_size (`int`) : batch size of prompts, defaults to 1

num_videos_per_prompt (`int`) : number of images that should be generated per prompt

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. If not provided, text embeddings will be generated from `prompt` input argument.

prompt_embeds_mask (`torch.Tensor`, *optional*) : Pre-generated text mask. If not provided, text mask will be generated from `prompt` input argument.

prompt_embeds_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text embeddings from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

prompt_embeds_mask_2 (`torch.Tensor`, *optional*) : Pre-generated glyph text mask from ByT5. If not provided, will be generated from `prompt` input argument using self.tokenizer_2 and self.text_encoder_2.

#### prepare_cond_latents_and_mask[[diffusers.HunyuanVideo15ImageToVideoPipeline.prepare_cond_latents_and_mask]]

```python
prepare_cond_latents_and_mask(latents: Tensor, image: Image, batch_size: int, height: int, width: int, dtype: dtype, device: device)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5_image2video.py#L594)

**Parameters:**

latents : Main latents tensor (B, C, F, H, W)

**Returns:** `tuple`

(cond_latents_concat, mask_concat) - both are zero tensors for t2v

Prepare conditional latents and mask for t2v generation.

## HunyuanVideo15PipelineOutput[[diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput]]

#### diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput[[diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput]]

```python
diffusers.pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video1_5/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for HunyuanVideo1.5 pipelines.

### LEDITS++
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/ledits_pp.md
