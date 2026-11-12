# Latent Diffusion

Latent Diffusion was proposed in [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.

The abstract from the paper is:

*By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.*

The original codebase can be found at [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

## LDMTextToImagePipeline[[diffusers.LDMTextToImagePipeline]]

#### diffusers.LDMTextToImagePipeline[[diffusers.LDMTextToImagePipeline]]

```python
diffusers.LDMTextToImagePipeline(vqvae: diffusers.models.autoencoders.vq_model.VQModel | diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL, bert: PreTrainedModel, tokenizer: PythonBackend, unet: diffusers.models.unets.unet_2d.UNet2DModel | diffusers.models.unets.unet_2d_condition.UNet2DConditionModel, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_pndm.PNDMScheduler | diffusers.utils.dummy_torch_and_scipy_objects.LMSDiscreteScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L39)

**Parameters:**

vqvae ([VQModel](/docs/diffusers/v0.43.0/en/api/models/vq#diffusers.VQModel)) : Vector-quantized (VQ) model to encode and decode images to and from latent representations.

bert (`LDMBertModel`) : Text-encoder model based on `BERT`.

tokenizer ([BertTokenizer](https://huggingface.co/docs/transformers/v5.22.2/en/model_doc/lxmert#transformers.BertTokenizer)) : A `BertTokenizer` to tokenize text.

unet ([UNet2DConditionModel](/docs/diffusers/v0.43.0/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel)) : A `UNet2DConditionModel` to denoise the encoded image latents.

scheduler ([SchedulerMixin](/docs/diffusers/v0.43.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of [DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), or [PNDMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

Pipeline for text-to-image generation using latent diffusion.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.LDMTextToImagePipeline.__call__]]

```python
__call__(prompt: str | list[str], height: int | None = None, width: int | None = None, num_inference_steps: int | None = 50, guidance_scale: float | None = 1.0, eta: float | None = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L74)

**Parameters:**

prompt (`str` or `list[str]`) : The prompt or prompts to guide the image generation.

height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`) : The height in pixels of the generated image.

width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`) : The width in pixels of the generated image.

num_inference_steps (`int`, *optional*, defaults to 50) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

guidance_scale (`float`, *optional*, defaults to 1.0) : A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the [DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `"pil"` (`PIL.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) or `tuple`

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images.

The call function to the pipeline for generation.

Example:

```py
>>> from diffusers import DiffusionPipeline

>>> # load model and scheduler
>>> ldm = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

>>> # run pipeline in inference (sample random noise and denoise)
>>> prompt = "A painting of a squirrel eating a burger"
>>> images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

>>> # save images
>>> for idx, image in enumerate(images):
...     image.save(f"squirrel-{idx}.png")
```

## LDMSuperResolutionPipeline[[diffusers.LDMSuperResolutionPipeline]]

#### diffusers.LDMSuperResolutionPipeline[[diffusers.LDMSuperResolutionPipeline]]

```python
diffusers.LDMSuperResolutionPipeline(vqvae: VQModel, unet: UNet2DModel, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_pndm.PNDMScheduler | diffusers.utils.dummy_torch_and_scipy_objects.LMSDiscreteScheduler | diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler | diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler | diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py#L39)

**Parameters:**

vqvae ([VQModel](/docs/diffusers/v0.43.0/en/api/models/vq#diffusers.VQModel)) : Vector-quantized (VQ) model to encode and decode images to and from latent representations.

unet ([UNet2DModel](/docs/diffusers/v0.43.0/en/api/models/unet2d#diffusers.UNet2DModel)) : A `UNet2DModel` to denoise the encoded image.

scheduler ([SchedulerMixin](/docs/diffusers/v0.43.0/en/api/schedulers/overview#diffusers.SchedulerMixin)) : A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of [DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), [LMSDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), [EulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/euler#diffusers.EulerDiscreteScheduler), [EulerAncestralDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/euler_ancestral#diffusers.EulerAncestralDiscreteScheduler), [DPMSolverMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler), or [PNDMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/pndm#diffusers.PNDMScheduler).

A pipeline for image super-resolution using latent diffusion.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.LDMSuperResolutionPipeline.__call__]]

```python
__call__(image: typing.Union[torch.Tensor, PIL.Image.Image] = None, batch_size: int | None = 1, num_inference_steps: int | None = 100, eta: float | None = 0.0, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str | None = 'pil', return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py#L71)

**Parameters:**

image (`torch.Tensor` or `PIL.Image.Image`) : `Image` or tensor representing an image batch to be used as the starting point for the process.

batch_size (`int`, *optional*, defaults to 1) : Number of images to generate.

num_inference_steps (`int`, *optional*, defaults to 100) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

eta (`float`, *optional*, defaults to 0.0) : Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies to the [DDIMScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/ddim#diffusers.DDIMScheduler), and is ignored in other schedulers.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `"pil"` (`PIL.Image`), `"np"` (`np.array`) or `"pt"` (`torch.Tensor`).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) instead of a plain tuple.

**Returns:** [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) or `tuple`

If `return_dict` is `True`, [ImagePipelineOutput](/docs/diffusers/v0.43.0/en/api/pipelines/consistency_models#diffusers.ImagePipelineOutput) is returned, otherwise a `tuple` is
returned where the first element is a list with the generated images

The call function to the pipeline for generation.

Example:

```py
>>> import requests
>>> from PIL import Image
>>> from io import BytesIO
>>> from diffusers import LDMSuperResolutionPipeline
>>> import torch

>>> # load model and scheduler
>>> pipeline = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
>>> pipeline = pipeline.to("cuda")

>>> # let's download an  image
>>> url = (
...     "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
... )
>>> response = requests.get(url)
>>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
>>> low_res_img = low_res_img.resize((128, 128))

>>> # run pipeline in inference (sample random noise and denoise)
>>> upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
>>> # save image
>>> upscaled_image.save("ldm_generated_image.png")
```

## ImagePipelineOutput[[diffusers.ImagePipelineOutput]]

#### diffusers.ImagePipelineOutput[[diffusers.ImagePipelineOutput]]

```python
diffusers.ImagePipelineOutput(images: list[PIL.Image.Image] | numpy.ndarray)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/pipeline_utils.py#L135)

**Parameters:**

images (`List[PIL.Image.Image]` or `np.ndarray`) : List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width, num_channels)`.

Output class for image pipelines.

### Consisid
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/consisid.md

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
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

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

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [from_pretrained()](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.from_pretrained) method.

```python
# !pip install consisid_eva_clip insightface facexlib
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from huggingface_hub import snapshot_download

# Download ckpts
snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")

# Load face helper model to preprocess input face image
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)  # or "mps", "xpu", "cpu"

# Load consisid base model
pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", dtype=torch.bfloat16)
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

```python
diffusers.ConsisIDPipeline(*args, **kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/utils/dummy_torch_and_transformers_and_opencv_objects.py#L5)

#### __call__[[diffusers.ConsisIDPipeline.__call__]]

```python
__call__(*args, **kwargs)
```

Call self as a function.

## ConsisIDPipelineOutput[[diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput]]

#### diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput[[diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput]]

```python
diffusers.pipelines.consisid.pipeline_output.ConsisIDPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/consisid/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for ConsisID pipelines.

### Skyreels V2
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/skyreels_v2.md

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
pipeline.to("cuda")  # or "mps", "xpu", "cpu"
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
pipeline.to("cuda")  # or "mps", "xpu", "cpu"
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
pipeline.to("cuda")  # or "mps", "xpu", "cpu"
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

- SkyReels-V2 supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.43.0/en/api/loaders/lora#diffusers.loaders.SkyReelsV2LoraLoaderMixin.load_lora_weights).

`SkyReelsV2Pipeline` and `SkyReelsV2ImageToVideoPipeline` are also available without Diffusion Forcing framework applied.

## SkyReelsV2DiffusionForcingPipeline[[diffusers.SkyReelsV2DiffusionForcingPipeline]]

#### diffusers.SkyReelsV2DiffusionForcingPipeline[[diffusers.SkyReelsV2DiffusionForcingPipeline]]

```python
diffusers.SkyReelsV2DiffusionForcingPipeline(tokenizer: AutoTokenizer, text_encoder: transformers.models.t5.modeling_t5.T5EncoderModel | transformers.models.umt5.modeling_umt5.UMT5EncoderModel, transformer: SkyReelsV2Transformer3DModel, vae: AutoencoderKLWan, scheduler: UniPCMultistepScheduler)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L128)

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Text-to-Video (t2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

#### __call__[[diffusers.SkyReelsV2DiffusionForcingPipeline.__call__]]

```python
__call__(prompt: str | list[str], negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512, overlap_history: int | None = None, addnoise_condition: float = 0, base_num_frames: int = 97, ar_step: int = 0, causal_block_size: int | None = None, fps: int = 24)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L598)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L219)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing.py#L418)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L133)

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Image-to-Video (i2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

#### __call__[[diffusers.SkyReelsV2DiffusionForcingImageToVideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 5.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, image_embeds: typing.Optional[torch.Tensor] = None, last_image: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512, overlap_history: int | None = None, addnoise_condition: float = 0, base_num_frames: int = 97, ar_step: int = 0, causal_block_size: int | None = None, fps: int = 24)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L644)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L224)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_i2v.py#L464)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L189)

**Parameters:**

tokenizer (`AutoTokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`UMT5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Video-to-Video (v2v) generation using SkyReels-V2 with diffusion forcing.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a specific device, etc.).

#### __call__[[diffusers.SkyReelsV2DiffusionForcingVideoToVideoPipeline.__call__]]

```python
__call__(video: list[PIL.Image.Image], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 120, num_inference_steps: int = 50, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512, overlap_history: int | None = None, addnoise_condition: float = 0, base_num_frames: int = 97, ar_step: int = 0, causal_block_size: int | None = None, fps: int = 24)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L684)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L280)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_diffusion_forcing_v2v.py#L504)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L107)

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Text-to-Video (t2v) generation using SkyReels-V2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.SkyReelsV2Pipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L377)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2.py#L198)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L127)

**Parameters:**

tokenizer (`T5Tokenizer`) : Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

text_encoder (`T5EncoderModel`) : [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.

image_encoder (`CLIPVisionModelWithProjection`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection), specifically the [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large) variant.

transformer ([SkyReelsV2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/skyreels_v2_transformer_3d#diffusers.SkyReelsV2Transformer3DModel)) : Conditional Transformer to denoise the input latents.

scheduler ([UniPCMultistepScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/unipc#diffusers.UniPCMultistepScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLWan](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_wan#diffusers.AutoencoderKLWan)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

Pipeline for Image-to-Video (i2v) generation using SkyReels-V2.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.SkyReelsV2ImageToVideoPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], prompt: str | list[str] = None, negative_prompt: str | list[str] = None, height: int = 544, width: int = 960, num_frames: int = 97, num_inference_steps: int = 50, guidance_scale: float = 5.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, image_embeds: typing.Optional[torch.Tensor] = None, last_image: typing.Optional[torch.Tensor] = None, output_type: str | None = 'np', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], max_sequence_length: int = 512)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L479)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_skyreels_v2_i2v.py#L241)

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

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/skyreels_v2/pipeline_output.py#L9)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for SkyReelsV2 pipelines.

### Allegro
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/allegro.md
