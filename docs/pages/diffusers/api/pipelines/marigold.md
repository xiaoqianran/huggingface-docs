# Marigold Computer Vision

![marigold](https://marigoldmonodepth.github.io/images/teaser_collage_compressed.jpg)

Marigold was proposed in 
[Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://huggingface.co/papers/2312.02145), 
a CVPR 2024 Oral paper by 
[Bingxin Ke](http://www.kebingxin.com/), 
[Anton Obukhov](https://www.obukhov.ai/), 
[Shengyu Huang](https://shengyuh.github.io/), 
[Nando Metzger](https://nandometzger.github.io/), 
[Rodrigo Caye Daudt](https://rcdaudt.github.io/), and 
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en).
The core idea is to **repurpose the generative prior of Text-to-Image Latent Diffusion Models (LDMs) for traditional 
computer vision tasks**.
This approach was explored by fine-tuning Stable Diffusion for **Monocular Depth Estimation**, as demonstrated in the 
teaser above.

Marigold was later extended in the follow-up paper, 
[Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis](https://huggingface.co/papers/2312.02145), 
authored by 
[Bingxin Ke](http://www.kebingxin.com/), 
[Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/?locale=en_US), 
[Tianfu Wang](https://tianfwang.github.io/), 
[Nando Metzger](https://nandometzger.github.io/), 
[Shengyu Huang](https://shengyuh.github.io/), 
[Bo Li](https://www.linkedin.com/in/bobboli0202/), 
[Anton Obukhov](https://www.obukhov.ai/), and 
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en).
This work expanded Marigold to support new modalities such as **Surface Normals** and **Intrinsic Image Decomposition** 
(IID), introduced a training protocol for **Latent Consistency Models** (LCM), and demonstrated **High-Resolution** (HR) 
processing capability.

> [!TIP]
> The early Marigold models (`v1-0` and earlier) were optimized for best results with at least 10 inference steps.
> LCM models were later developed to enable high-quality inference in just 1 to 4 steps.
> Marigold models `v1-1` and later use the DDIM scheduler to achieve optimal 
> results in as few as 1 to 4 steps.

## Available Pipelines

Each pipeline is tailored for a specific computer vision task, processing an input RGB image and generating a 
corresponding prediction.
Currently, the following computer vision tasks are implemented:

| Pipeline                                                                                                                                          | Recommended Model Checkpoints                                                                                                                                                                           |                              Spaces (Interactive Apps)                               | Predicted Modalities                                                                                                                                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MarigoldDepthPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py)           | [prs-eth/marigold-depth-v1-1](https://huggingface.co/prs-eth/marigold-depth-v1-1)                                                                                                                       |          [Depth Estimation](https://huggingface.co/spaces/prs-eth/marigold)          | [Depth](https://en.wikipedia.org/wiki/Depth_map), [Disparity](https://en.wikipedia.org/wiki/Binocular_disparity)                                                                   |
| [MarigoldNormalsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py)       | [prs-eth/marigold-normals-v1-1](https://huggingface.co/prs-eth/marigold-normals-v1-1)                                                                                                                   | [Surface Normals Estimation](https://huggingface.co/spaces/prs-eth/marigold-normals) | [Surface normals](https://en.wikipedia.org/wiki/Normal_mapping)                                                                                                                    |
| [MarigoldIntrinsicsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py) | [prs-eth/marigold-iid-appearance-v1-1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1),[prs-eth/marigold-iid-lighting-v1-1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) | [Intrinsic Image Decomposition](https://huggingface.co/spaces/prs-eth/marigold-iid)  | [Albedo](https://en.wikipedia.org/wiki/Albedo), [Materials](https://www.n.aiq3d.com/wiki/roughnessmetalnessao-map), [Lighting](https://en.wikipedia.org/wiki/Diffuse_reflection)   |

## Available Checkpoints

All original checkpoints are available under the [PRS-ETH](https://huggingface.co/prs-eth/) organization on Hugging Face.
They are designed for use with diffusers pipelines and the [original codebase](https://github.com/prs-eth/marigold), which can also be used to train 
new model checkpoints.
The following is a summary of the recommended checkpoints, all of which produce reliable results with 1 to 4 steps. 

| Checkpoint                                                                                          | Modality     | Comment                                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [prs-eth/marigold-depth-v1-1](https://huggingface.co/prs-eth/marigold-depth-v1-1)                   | Depth        | Affine-invariant depth prediction assigns each pixel a value between 0 (near plane) and 1 (far plane), with both planes determined by the model during inference.                    |
| [prs-eth/marigold-normals-v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1)               | Normals      | The surface normals predictions are unit-length 3D vectors in the screen space camera, with values in the range from -1 to 1.                                                        |
| [prs-eth/marigold-iid-appearance-v1-1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) | Intrinsics   | InteriorVerse decomposition is comprised of Albedo and two BRDF material properties: Roughness and Metallicity.                                                                      |
| [prs-eth/marigold-iid-lighting-v1-1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1)     | Intrinsics   | HyperSim decomposition of an image $I$ is comprised of Albedo $A$, Diffuse shading $S$, and Non-diffuse residual $R$: $I = A*S+R$. |

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff
> between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to
> efficiently load the same components into multiple pipelines.
> Also, to know more about reducing the memory usage of this pipeline, refer to the ["Reduce memory usage"] section
> [here](./stable_diffusion/svd#reduce-memory-usage).

> [!WARNING]
> Marigold pipelines were designed and tested with the scheduler embedded in the model checkpoint.
> The optimal number of inference steps varies by scheduler, with no universal value that works best across all cases.
> To accommodate this, the `num_inference_steps` parameter in the pipeline's `__call__` method defaults to `None` (see the 
> API reference).
> Unless set explicitly, it inherits the value from the `default_denoising_steps` field in the checkpoint configuration 
> file (`model_index.json`).
> This ensures high-quality predictions when invoking the pipeline with only the `image` argument.

The examples below are mostly given for depth prediction, but they can be universally applied to other supported 
modalities.
We showcase the predictions using the same input image of Albert Einstein generated by Midjourney.
This makes it easier to compare visualizations of the predictions across various modalities and checkpoints.

  
    
    
      Example input image for all Marigold pipelines
    
  

## Depth Prediction

To get a depth prediction, load the `prs-eth/marigold-depth-v1-1` checkpoint into [MarigoldDepthPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.MarigoldDepthPipeline), 
put the image through the pipeline, and save the predictions:

```python
import diffusers
import torch

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

depth = pipe(image)

vis = pipe.image_processor.visualize_depth(depth.prediction)
vis[0].save("einstein_depth.png")

depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
depth_16bit[0].save("einstein_depth_16bit.png")
```

The [visualize_depth()](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_depth) function applies one of 
[matplotlib's colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html) (`Spectral` by default) to map the predicted pixel values from a single-channel `[0, 1]` 
depth range into an RGB image.
With the `Spectral` colormap, pixels with near depth are painted red, and far pixels are blue.
The 16-bit PNG file stores the single channel values mapped linearly from the `[0, 1]` range into `[0, 65535]`.
Below are the raw and the visualized predictions. The darker and closer areas (mustache) are easier to distinguish in 
the visualization.

  
    
    
      Predicted depth (16-bit PNG)
    
  
  
    
    
      Predicted depth visualization (Spectral)
    
  

## Surface Normals Estimation

Load the `prs-eth/marigold-normals-v1-1` checkpoint into [MarigoldNormalsPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.MarigoldNormalsPipeline), put the image through the 
pipeline, and save the predictions:

```python
import diffusers
import torch

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-v1-1", variant="fp16", dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals.prediction)
vis[0].save("einstein_normals.png")
```

The [visualize_normals()](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_normals) maps the three-dimensional 
prediction with pixel values in the range `[-1, 1]` into an RGB image.
The visualization function supports flipping surface normals axes to make the visualization compatible with other 
choices of the frame of reference.
Conceptually, each pixel is painted according to the surface normal vector in the frame of reference, where `X` axis 
points right, `Y` axis points up, and `Z` axis points at the viewer.
Below is the visualized prediction:

  
    
    
      Predicted surface normals visualization
    
  

In this example, the nose tip almost certainly has a point on the surface, in which the surface normal vector points 
straight at the viewer, meaning that its coordinates are `[0, 0, 1]`.
This vector maps to the RGB `[128, 128, 255]`, which corresponds to the violet-blue color.
Similarly, a surface normal on the cheek in the right part of the image has a large `X` component, which increases the 
red hue.
Points on the shoulders pointing up with a large `Y` promote green color.

## Intrinsic Image Decomposition

Marigold provides two models for Intrinsic Image Decomposition (IID): "Appearance" and "Lighting". 
Each model produces Albedo maps, derived from InteriorVerse and Hypersim annotations, respectively.

- The "Appearance" model also estimates Material properties: Roughness and Metallicity.
- The "Lighting" model generates Diffuse Shading and Non-diffuse Residual.

Here is the sample code saving predictions made by the "Appearance" model:

```python
import diffusers
import torch

pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
    "prs-eth/marigold-iid-appearance-v1-1", variant="fp16", dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

intrinsics = pipe(image)

vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)
vis[0]["albedo"].save("einstein_albedo.png")
vis[0]["roughness"].save("einstein_roughness.png")
vis[0]["metallicity"].save("einstein_metallicity.png")
```

Another example demonstrating the predictions made by the "Lighting" model:

```python
import diffusers
import torch

pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
    "prs-eth/marigold-iid-lighting-v1-1", variant="fp16", dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

intrinsics = pipe(image)

vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)
vis[0]["albedo"].save("einstein_albedo.png")
vis[0]["shading"].save("einstein_shading.png")
vis[0]["residual"].save("einstein_residual.png")
```

Both models share the same pipeline while supporting different decomposition types.
The exact decomposition parameterization (e.g., sRGB vs. linear space) is stored in the 
`pipe.target_properties` dictionary, which is passed into the 
[visualize_intrinsics()](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_intrinsics) function.

Below are some examples showcasing the predicted decomposition outputs. 
All modalities can be inspected in the 
[Intrinsic Image Decomposition](https://huggingface.co/spaces/prs-eth/marigold-iid) Space.

  
    
    
      Predicted albedo ("Appearance" model)
    
  
  
    
    
      Predicted diffuse shading ("Lighting" model)
    
  

## Speeding up inference

The above quick start snippets are already optimized for quality and speed, loading the checkpoint, utilizing the 
`fp16` variant of weights and computation, and performing the default number (4) of denoising diffusion steps.
The first step to accelerate inference, at the expense of prediction quality, is to reduce the denoising diffusion 
steps to the minimum:

```diff
  import diffusers
  import torch

  pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
  ).to("cuda")  # or "mps", "xpu", "cpu"

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
  
- depth = pipe(image)
+ depth = pipe(image, num_inference_steps=1)
```

With this change, the `pipe` call completes in 280ms on RTX 3090 GPU.
Internally, the input image is first encoded using the Stable Diffusion VAE encoder, followed by a single denoising 
step performed by the U-Net. 
Finally, the prediction latent is decoded with the VAE decoder into pixel space.
In this setup, two out of three module calls are dedicated to converting between the pixel and latent spaces of the LDM.
Since Marigold's latent space is compatible with Stable Diffusion 2.0, inference can be accelerated by more than 3x, 
reducing the call time to 85ms on an RTX 3090, by using a [lightweight replacement of the SD VAE](../models/autoencoder_tiny). 
Note that using a lightweight VAE may slightly reduce the visual quality of the predictions.

```diff
  import diffusers
  import torch

  pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
  ).to("cuda")  # or "mps", "xpu", "cpu"

+ pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
+     "madebyollin/taesd", dtype=torch.float16
+ ).cuda()

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

  depth = pipe(image, num_inference_steps=1)
```

So far, we have optimized the number of diffusion steps and model components. Self-attention operations account for a 
significant portion of computations. 
Speeding them up can be achieved by using a more efficient attention processor:

```diff
  import diffusers
  import torch
+ from diffusers.models.attention_processor import AttnProcessor2_0

  pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
  ).to("cuda")  # or "mps", "xpu", "cpu"

+ pipe.vae.set_attn_processor(AttnProcessor2_0()) 
+ pipe.unet.set_attn_processor(AttnProcessor2_0())

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

  depth = pipe(image, num_inference_steps=1)
```

Finally, as suggested in [Optimizations](../../optimization/fp16#torchcompile), enabling `torch.compile` can further enhance performance depending on 
the target hardware.
However, compilation incurs a significant overhead during the first pipeline invocation, making it beneficial only when 
the same pipeline instance is called repeatedly, such as within a loop.

```diff
  import diffusers
  import torch
  from diffusers.models.attention_processor import AttnProcessor2_0

  pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
  ).to("cuda")  # or "mps", "xpu", "cpu"

  pipe.vae.set_attn_processor(AttnProcessor2_0()) 
  pipe.unet.set_attn_processor(AttnProcessor2_0())

+ pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

  depth = pipe(image, num_inference_steps=1)
```

## Maximizing Precision and Ensembling

Marigold pipelines have a built-in ensembling mechanism combining multiple predictions from different random latents.
This is a brute-force way of improving the precision of predictions, capitalizing on the generative nature of diffusion.
The ensembling path is activated automatically when the `ensemble_size` argument is set greater or equal than `3`.
When aiming for maximum precision, it makes sense to adjust `num_inference_steps` simultaneously with `ensemble_size`.
The recommended values vary across checkpoints but primarily depend on the scheduler type.
The effect of ensembling is particularly well-seen with surface normals:

```diff
  import diffusers

  pipe = diffusers.MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1").to("cuda")  # or "mps", "xpu", "cpu"

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

- depth = pipe(image)
+ depth = pipe(image, num_inference_steps=10, ensemble_size=5)

  vis = pipe.image_processor.visualize_normals(depth.prediction)
  vis[0].save("einstein_normals.png")
```

  
    
    
      Surface normals, no ensembling
    
  
  
    
    
      Surface normals, with ensembling
    
  

As can be seen, all areas with fine-grained structurers, such as hair, got more conservative and on average more 
correct predictions.
Such a result is more suitable for precision-sensitive downstream tasks, such as 3D reconstruction.

## Frame-by-frame Video Processing with Temporal Consistency

Due to Marigold's generative nature, each prediction is unique and defined by the random noise sampled for the latent 
initialization.
This becomes an obvious drawback compared to traditional end-to-end dense regression networks, as exemplified in the 
following videos:

  
    
    Input video
  
  
    
    Marigold Depth applied to input video frames independently
  

To address this issue, it is possible to pass `latents` argument to the pipelines, which defines the starting point of 
diffusion.
Empirically, we found that a convex combination of the very same starting point noise latent and the latent 
corresponding to the previous frame prediction give sufficiently smooth results, as implemented in the snippet below:

```python
import imageio
import diffusers
import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from tqdm import tqdm

device = "cuda"  # or "mps", "xpu", "cpu"
path_in = "https://huggingface.co/spaces/prs-eth/marigold-lcm/resolve/c7adb5427947d2680944f898cd91d386bf0d4924/files/video/obama.mp4"
path_out = "obama_depth.gif"

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
).to(device)
pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", dtype=torch.float16
).to(device)
pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.set_progress_bar_config(disable=True)

with imageio.get_reader(path_in) as reader:
    size = reader.get_meta_data()['size']
    last_frame_latent = None
    latent_common = torch.randn(
        (1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))
    ).to(device=device, dtype=torch.float16)

    out = []
    for frame_id, frame in tqdm(enumerate(reader), desc="Processing Video"):
        frame = Image.fromarray(frame)
        latents = latent_common
        if last_frame_latent is not None:
            latents = 0.9 * latents + 0.1 * last_frame_latent

        depth = pipe(
            frame,
            num_inference_steps=1,
            match_input_resolution=False, 
            latents=latents, 
            output_latent=True,
        )
        last_frame_latent = depth.latent
        out.append(pipe.image_processor.visualize_depth(depth.prediction)[0])

    diffusers.utils.export_to_gif(out, path_out, fps=reader.get_meta_data()['fps'])
```

Here, the diffusion process starts from the given computed latent.
The pipeline sets `output_latent=True` to access `out.latent` and computes its contribution to the next frame's latent 
initialization.
The result is much more stable now:

  
    
    Marigold Depth applied to input video frames independently
  
  
    
    Marigold Depth with forced latents initialization
  

## Marigold for ControlNet

A very common application for depth prediction with diffusion models comes in conjunction with ControlNet.
Depth crispness plays a crucial role in obtaining high-quality results from ControlNet.
As seen in comparisons with other methods above, Marigold excels at that task.
The snippet below demonstrates how to load an image, compute depth, and pass it into ControlNet in a compatible format:

```python
import torch
import diffusers

device = "cuda"  # or "mps", "xpu", "cpu"
generator = torch.Generator(device=device).manual_seed(2024)
image = diffusers.utils.load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"
)

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", dtype=torch.float16, variant="fp16"
).to(device)

depth_image = pipe(image, generator=generator).prediction
depth_image = pipe.image_processor.visualize_depth(depth_image, color_map="binary")
depth_image[0].save("motorcycle_controlnet_depth.png")

controlnet = diffusers.ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", dtype=torch.float16, variant="fp16"
).to(device)
pipe = diffusers.StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", dtype=torch.float16, variant="fp16", controlnet=controlnet
).to(device)
pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

controlnet_out = pipe(
    prompt="high quality photo of a sports bike, city",
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    image=depth_image,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.7,
    generator=generator,
).images
controlnet_out[0].save("motorcycle_controlnet_out.png")
```

  
    
    
      Input image
    
  
  
    
    
      Depth in the format compatible with ControlNet
    
  
  
    
    
      ControlNet generation, conditioned on depth and prompt: "high quality photo of a sports bike, city"
    
  

## Quantitative Evaluation

To evaluate Marigold quantitatively in standard leaderboards and benchmarks (such as NYU, KITTI, and other datasets), 
follow the evaluation protocol outlined in the paper: load the full precision fp32 model and use appropriate values 
for `num_inference_steps` and `ensemble_size`.
Optionally seed randomness to ensure reproducibility. 
Maximizing `batch_size` will deliver maximum device utilization.

```python
import diffusers
import torch

device = "cuda"  # or "mps", "xpu", "cpu"
seed = 2024

generator = torch.Generator(device=device).manual_seed(seed)
pipe = diffusers.MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1").to(device)

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

depth = pipe(
    image, 
    num_inference_steps=4,  # set according to the evaluation protocol from the paper
    ensemble_size=10,       # set according to the evaluation protocol from the paper
    generator=generator,
)

# evaluate metrics
```

## Using Predictive Uncertainty

The ensembling mechanism built into Marigold pipelines combines multiple predictions obtained from different random 
latents.
As a side effect, it can be used to quantify epistemic (model) uncertainty; simply specify `ensemble_size` greater 
or equal than 3 and set `output_uncertainty=True`.
The resulting uncertainty will be available in the `uncertainty` field of the output.
It can be visualized as follows:

```python
import diffusers
import torch

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

depth = pipe(
	image,
	ensemble_size=10,  # any number >= 3
	output_uncertainty=True,
)

uncertainty = pipe.image_processor.visualize_uncertainty(depth.uncertainty)
uncertainty[0].save("einstein_depth_uncertainty.png")
```

  
    
    
      Depth uncertainty
    
  
  
    
    
      Surface normals uncertainty
    
  
  
    
    
      Albedo uncertainty
    
  

The interpretation of uncertainty is easy: higher values (white) correspond to pixels, where the model struggles to 
make consistent predictions.
- The depth model exhibits the most uncertainty around discontinuities, where object depth changes abruptly.
- The surface normals model is least confident in fine-grained structures like hair and in dark regions such as the 
collar area.
- Albedo uncertainty is represented as an RGB image, as it captures uncertainty independently for each color channel, 
unlike depth and surface normals. It is also higher in shaded regions and at discontinuities.

## Marigold Depth Prediction API[[diffusers.MarigoldDepthPipeline]]

#### diffusers.MarigoldDepthPipeline[[diffusers.MarigoldDepthPipeline]]

```python
diffusers.MarigoldDepthPipeline(unet: UNet2DConditionModel, vae: AutoencoderKL, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_lcm.LCMScheduler, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, prediction_type: str | None = None, scale_invariant: bool | None = True, shift_invariant: bool | None = True, default_denoising_steps: int | None = None, default_processing_resolution: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py#L104)

**Parameters:**

unet (`UNet2DConditionModel`) : Conditional U-Net to denoise the depth latent, conditioned on image latent.

vae (`AutoencoderKL`) : Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent representations.

scheduler (`DDIMScheduler` or `LCMScheduler`) : A scheduler to be used in combination with `unet` to denoise the encoded image latents.

text_encoder (`CLIPTextModel`) : Text-encoder, for empty text embedding.

tokenizer (`CLIPTokenizer`) : CLIP tokenizer.

prediction_type (`str`, *optional*) : Type of predictions made by the model.

scale_invariant (`bool`, *optional*) : A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in the model config. When used together with the `shift_invariant=True` flag, the model is also called "affine-invariant". NB: overriding this value is not supported.

shift_invariant (`bool`, *optional*) : A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in the model config. When used together with the `scale_invariant=True` flag, the model is also called "affine-invariant". NB: overriding this value is not supported.

default_denoising_steps (`int`, *optional*) : The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable quality with the given model. This value must be set in the model config. When the pipeline is called without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure reasonable results with various model flavors compatible with the pipeline, such as those relying on very short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).

default_processing_resolution (`int`, *optional*) : The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in the model config. When the pipeline is called without explicitly setting `processing_resolution`, the default value is used. This is required to ensure reasonable results with various model flavors trained with varying optimal processing resolution values.

Pipeline for monocular depth estimation using the Marigold method: https://marigoldmonodepth.github.io.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.MarigoldDepthPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], num_inference_steps: int | None = None, ensemble_size: int = 1, processing_resolution: int | None = None, match_input_resolution: bool = True, resample_method_input: str = 'bilinear', resample_method_output: str = 'bilinear', batch_size: int = 1, ensembling_kwargs: dict[str, typing.Any] | None = None, latents: typing.Union[torch.Tensor, list[torch.Tensor], NoneType] = None, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str = 'np', output_uncertainty: bool = False, output_latent: bool = False, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py#L347)

**Parameters:**

image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`), : `list[torch.Tensor]`: An input image or images used as an input for the depth estimation task. For arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the same width and height.

num_inference_steps (`int`, *optional*, defaults to `None`) : Number of denoising diffusion steps during inference. The default value `None` results in automatic selection.

ensemble_size (`int`, defaults to `1`) : Number of ensemble predictions. Higher values result in measurable improvements and visual degradation.

processing_resolution (`int`, *optional*, defaults to `None`) : Effective processing resolution. When set to `0`, matches the larger input image dimension. This produces crisper predictions, but may also lead to the overall loss of global context. The default value `None` resolves to the optimal value from the model config.

match_input_resolution (`bool`, *optional*, defaults to `True`) : When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer side of the output will equal to `processing_resolution`.

resample_method_input (`str`, *optional*, defaults to `"bilinear"`) : Resampling method used to resize input images to `processing_resolution`. The accepted values are: `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.

resample_method_output (`str`, *optional*, defaults to `"bilinear"`) : Resampling method used to resize output predictions to match the input resolution. The accepted values are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.

batch_size (`int`, *optional*, defaults to `1`) : Batch size; only matters when setting `ensemble_size` or passing a tensor of images.

ensembling_kwargs (`dict`, *optional*, defaults to `None`) : Extra dictionary with arguments for precise ensembling control. The following options are available: - reduction (`str`, *optional*, defaults to `"median"`): Defines the ensembling function applied in every pixel location, can be either `"median"` or `"mean"`. - regularizer_strength (`float`, *optional*, defaults to `0.02`): Strength of the regularizer that pulls the aligned predictions to the unit range from 0 to 1. - max_iter (`int`, *optional*, defaults to `2`): Maximum number of the alignment solver steps. Refer to `scipy.optimize.minimize` function, `options` argument. - tol (`float`, *optional*, defaults to `1e-3`): Alignment solver tolerance. The solver stops when the tolerance is reached. - max_res (`int`, *optional*, defaults to `None`): Resolution at which the alignment is performed; `None` matches the `processing_resolution`.

latents (`torch.Tensor`, or `list[torch.Tensor]`, *optional*, defaults to `None`) : Latent noise tensors to replace the random initialization. These can be taken from the previous function call's output.

generator (`torch.Generator`, or `list[torch.Generator]`, *optional*, defaults to `None`) : Random number generator object to ensure reproducibility.

output_type (`str`, *optional*, defaults to `"np"`) : Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted values are: `"np"` (numpy array) or `"pt"` (torch tensor).

output_uncertainty (`bool`, *optional*, defaults to `False`) : When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that the `ensemble_size` argument is set to a value above 2.

output_latent (`bool`, *optional*, defaults to `False`) : When enabled, the output's `latent` field contains the latent codes corresponding to the predictions within the ensemble. These codes can be saved, modified, and used for subsequent calls with the `latents` argument.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [MarigoldDepthOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldDepthOutput) instead of a plain tuple.

**Returns:** [MarigoldDepthOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldDepthOutput) or `tuple`

If `return_dict` is `True`, [MarigoldDepthOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldDepthOutput) is returned, otherwise a
`tuple` is returned where the first element is the prediction, the second element is the uncertainty
(or `None`), and the third is the latent (or `None`).

Function invoked when calling the pipeline.

Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
...     "prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> depth = pipe(image)

>>> vis = pipe.image_processor.visualize_depth(depth.prediction)
>>> vis[0].save("einstein_depth.png")

>>> depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
>>> depth_16bit[0].save("einstein_depth_16bit.png")
```

#### diffusers.pipelines.marigold.MarigoldDepthOutput[[diffusers.pipelines.marigold.MarigoldDepthOutput]]

```python
diffusers.pipelines.marigold.MarigoldDepthOutput(prediction: typing.Union[numpy.ndarray, torch.Tensor], uncertainty: typing.Union[NoneType, numpy.ndarray, torch.Tensor], latent: typing.Optional[torch.Tensor])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py#L83)

**Parameters:**

prediction (`np.ndarray`, `torch.Tensor`) : Predicted depth maps with values in the range [0, 1]. The shape is `numimages × 1 × height × width` for `torch.Tensor` or `numimages × height × width × 1` for `np.ndarray`.

uncertainty (`None`, `np.ndarray`, `torch.Tensor`) : Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is `numimages × 1 × height × width` for `torch.Tensor` or `numimages × height × width × 1` for `np.ndarray`.

latent (`None`, `torch.Tensor`) : Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline. The shape is `numimages * numensemble × 4 × latentheight × latentwidth`.

Output class for Marigold monocular depth prediction pipeline.

#### diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_depth[[diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_depth]]

```python
diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_depth(depth: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], val_min: float = 0.0, val_max: float = 1.0, color_map: str = 'Spectral')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/marigold_image_processing.py#L387)

**Parameters:**

depth (`PIL.Image.Image | np.ndarray | torch.Tensor | list[PIL.Image.Image, list[np.ndarray], : list[torch.Tensor]]`): Depth maps.

val_min (`float`, *optional*, defaults to `0.0`) : Minimum value of the visualized depth range.

val_max (`float`, *optional*, defaults to `1.0`) : Maximum value of the visualized depth range.

color_map (`str`, *optional*, defaults to `"Spectral"`) : Color map used to convert a single-channel depth prediction into colored representation.

Visualizes depth maps, such as predictions of the `MarigoldDepthPipeline`.

Returns: `list[PIL.Image.Image]` with depth maps visualization.

## Marigold Normals Estimation API[[diffusers.MarigoldNormalsPipeline]]

#### diffusers.MarigoldNormalsPipeline[[diffusers.MarigoldNormalsPipeline]]

```python
diffusers.MarigoldNormalsPipeline(unet: UNet2DConditionModel, vae: AutoencoderKL, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_lcm.LCMScheduler, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, prediction_type: str | None = None, use_full_z_range: bool | None = True, default_denoising_steps: int | None = None, default_processing_resolution: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py#L99)

**Parameters:**

unet (`UNet2DConditionModel`) : Conditional U-Net to denoise the normals latent, conditioned on image latent.

vae (`AutoencoderKL`) : Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent representations.

scheduler (`DDIMScheduler` or `LCMScheduler`) : A scheduler to be used in combination with `unet` to denoise the encoded image latents.

text_encoder (`CLIPTextModel`) : Text-encoder, for empty text embedding.

tokenizer (`CLIPTokenizer`) : CLIP tokenizer.

prediction_type (`str`, *optional*) : Type of predictions made by the model.

use_full_z_range (`bool`, *optional*) : Whether the normals predicted by this model utilize the full range of the Z dimension, or only its positive half.

default_denoising_steps (`int`, *optional*) : The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable quality with the given model. This value must be set in the model config. When the pipeline is called without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure reasonable results with various model flavors compatible with the pipeline, such as those relying on very short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).

default_processing_resolution (`int`, *optional*) : The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in the model config. When the pipeline is called without explicitly setting `processing_resolution`, the default value is used. This is required to ensure reasonable results with various model flavors trained with varying optimal processing resolution values.

Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.github.io.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.MarigoldNormalsPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], num_inference_steps: int | None = None, ensemble_size: int = 1, processing_resolution: int | None = None, match_input_resolution: bool = True, resample_method_input: str = 'bilinear', resample_method_output: str = 'bilinear', batch_size: int = 1, ensembling_kwargs: dict[str, typing.Any] | None = None, latents: typing.Union[torch.Tensor, list[torch.Tensor], NoneType] = None, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str = 'np', output_uncertainty: bool = False, output_latent: bool = False, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py#L332)

**Parameters:**

image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`), : `list[torch.Tensor]`: An input image or images used as an input for the normals estimation task. For arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the same width and height.

num_inference_steps (`int`, *optional*, defaults to `None`) : Number of denoising diffusion steps during inference. The default value `None` results in automatic selection.

ensemble_size (`int`, defaults to `1`) : Number of ensemble predictions. Higher values result in measurable improvements and visual degradation.

processing_resolution (`int`, *optional*, defaults to `None`) : Effective processing resolution. When set to `0`, matches the larger input image dimension. This produces crisper predictions, but may also lead to the overall loss of global context. The default value `None` resolves to the optimal value from the model config.

match_input_resolution (`bool`, *optional*, defaults to `True`) : When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer side of the output will equal to `processing_resolution`.

resample_method_input (`str`, *optional*, defaults to `"bilinear"`) : Resampling method used to resize input images to `processing_resolution`. The accepted values are: `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.

resample_method_output (`str`, *optional*, defaults to `"bilinear"`) : Resampling method used to resize output predictions to match the input resolution. The accepted values are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.

batch_size (`int`, *optional*, defaults to `1`) : Batch size; only matters when setting `ensemble_size` or passing a tensor of images.

ensembling_kwargs (`dict`, *optional*, defaults to `None`) : Extra dictionary with arguments for precise ensembling control. The following options are available: - reduction (`str`, *optional*, defaults to `"closest"`): Defines the ensembling function applied in every pixel location, can be either `"closest"` or `"mean"`.

latents (`torch.Tensor`, *optional*, defaults to `None`) : Latent noise tensors to replace the random initialization. These can be taken from the previous function call's output.

generator (`torch.Generator`, or `list[torch.Generator]`, *optional*, defaults to `None`) : Random number generator object to ensure reproducibility.

output_type (`str`, *optional*, defaults to `"np"`) : Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted values are: `"np"` (numpy array) or `"pt"` (torch tensor).

output_uncertainty (`bool`, *optional*, defaults to `False`) : When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that the `ensemble_size` argument is set to a value above 2.

output_latent (`bool`, *optional*, defaults to `False`) : When enabled, the output's `latent` field contains the latent codes corresponding to the predictions within the ensemble. These codes can be saved, modified, and used for subsequent calls with the `latents` argument.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [MarigoldNormalsOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldNormalsOutput) instead of a plain tuple.

**Returns:** [MarigoldNormalsOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldNormalsOutput) or `tuple`

If `return_dict` is `True`, [MarigoldNormalsOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldNormalsOutput) is returned, otherwise a
`tuple` is returned where the first element is the prediction, the second element is the uncertainty
(or `None`), and the third is the latent (or `None`).

Function invoked when calling the pipeline.

Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
...     "prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> normals = pipe(image)

>>> vis = pipe.image_processor.visualize_normals(normals.prediction)
>>> vis[0].save("einstein_normals.png")
```

#### diffusers.pipelines.marigold.MarigoldNormalsOutput[[diffusers.pipelines.marigold.MarigoldNormalsOutput]]

```python
diffusers.pipelines.marigold.MarigoldNormalsOutput(prediction: typing.Union[numpy.ndarray, torch.Tensor], uncertainty: typing.Union[NoneType, numpy.ndarray, torch.Tensor], latent: typing.Optional[torch.Tensor])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py#L78)

**Parameters:**

prediction (`np.ndarray`, `torch.Tensor`) : Predicted normals with values in the range [-1, 1]. The shape is `numimages × 3 × height × width` for `torch.Tensor` or `numimages × height × width × 3` for `np.ndarray`.

uncertainty (`None`, `np.ndarray`, `torch.Tensor`) : Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is `numimages × 1 × height × width` for `torch.Tensor` or `numimages × height × width × 1` for `np.ndarray`.

latent (`None`, `torch.Tensor`) : Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline. The shape is `numimages * numensemble × 4 × latentheight × latentwidth`.

Output class for Marigold monocular normals prediction pipeline.

#### diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_normals[[diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_normals]]

```python
diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_normals(normals: typing.Union[numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor]], flip_x: bool = False, flip_y: bool = False, flip_z: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/marigold_image_processing.py#L486)

**Parameters:**

normals (`np.ndarray | torch.Tensor | list[np.ndarray, list[torch.Tensor]]`) : Surface normals.

flip_x (`bool`, *optional*, defaults to `False`) : Flips the X axis of the normals frame of reference. Default direction is right.

flip_y (`bool`, *optional*, defaults to `False`) : Flips the Y axis of the normals frame of reference. Default direction is top.

flip_z (`bool`, *optional*, defaults to `False`) : Flips the Z axis of the normals frame of reference. Default direction is facing the observer.

Visualizes surface normals, such as predictions of the `MarigoldNormalsPipeline`.

Returns: `list[PIL.Image.Image]` with surface normals visualization.

## Marigold Intrinsic Image Decomposition API[[diffusers.MarigoldIntrinsicsPipeline]]

#### diffusers.MarigoldIntrinsicsPipeline[[diffusers.MarigoldIntrinsicsPipeline]]

```python
diffusers.MarigoldIntrinsicsPipeline(unet: UNet2DConditionModel, vae: AutoencoderKL, scheduler: diffusers.schedulers.scheduling_ddim.DDIMScheduler | diffusers.schedulers.scheduling_lcm.LCMScheduler, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, prediction_type: str | None = None, target_properties: dict[str, typing.Any] | None = None, default_denoising_steps: int | None = None, default_processing_resolution: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py#L120)

**Parameters:**

unet (`UNet2DConditionModel`) : Conditional U-Net to denoise the targets latent, conditioned on image latent.

vae (`AutoencoderKL`) : Variational Auto-Encoder (VAE) Model to encode and decode images and predictions to and from latent representations.

scheduler (`DDIMScheduler` or `LCMScheduler`) : A scheduler to be used in combination with `unet` to denoise the encoded image latents.

text_encoder (`CLIPTextModel`) : Text-encoder, for empty text embedding.

tokenizer (`CLIPTokenizer`) : CLIP tokenizer.

prediction_type (`str`, *optional*) : Type of predictions made by the model.

target_properties (`dict[str, Any]`, *optional*) : Properties of the predicted modalities, such as `target_names`, a `list[str]` used to define the number, order and names of the predicted modalities, and any other metadata that may be required to interpret the predictions.

default_denoising_steps (`int`, *optional*) : The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable quality with the given model. This value must be set in the model config. When the pipeline is called without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure reasonable results with various model flavors compatible with the pipeline, such as those relying on very short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).

default_processing_resolution (`int`, *optional*) : The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in the model config. When the pipeline is called without explicitly setting `processing_resolution`, the default value is used. This is required to ensure reasonable results with various model flavors trained with varying optimal processing resolution values.

Pipeline for Intrinsic Image Decomposition (IID) using the Marigold method:
https://marigoldcomputervision.github.io.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods the
library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

#### __call__[[diffusers.MarigoldIntrinsicsPipeline.__call__]]

```python
__call__(image: typing.Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]], num_inference_steps: int | None = None, ensemble_size: int = 1, processing_resolution: int | None = None, match_input_resolution: bool = True, resample_method_input: str = 'bilinear', resample_method_output: str = 'bilinear', batch_size: int = 1, ensembling_kwargs: dict[str, typing.Any] | None = None, latents: typing.Union[torch.Tensor, list[torch.Tensor], NoneType] = None, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, output_type: str = 'np', output_uncertainty: bool = False, output_latent: bool = False, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py#L359)

**Parameters:**

image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`), : `list[torch.Tensor]`: An input image or images used as an input for the intrinsic decomposition task. For arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the same width and height.

num_inference_steps (`int`, *optional*, defaults to `None`) : Number of denoising diffusion steps during inference. The default value `None` results in automatic selection.

ensemble_size (`int`, defaults to `1`) : Number of ensemble predictions. Higher values result in measurable improvements and visual degradation.

processing_resolution (`int`, *optional*, defaults to `None`) : Effective processing resolution. When set to `0`, matches the larger input image dimension. This produces crisper predictions, but may also lead to the overall loss of global context. The default value `None` resolves to the optimal value from the model config.

match_input_resolution (`bool`, *optional*, defaults to `True`) : When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer side of the output will equal to `processing_resolution`.

resample_method_input (`str`, *optional*, defaults to `"bilinear"`) : Resampling method used to resize input images to `processing_resolution`. The accepted values are: `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.

resample_method_output (`str`, *optional*, defaults to `"bilinear"`) : Resampling method used to resize output predictions to match the input resolution. The accepted values are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.

batch_size (`int`, *optional*, defaults to `1`) : Batch size; only matters when setting `ensemble_size` or passing a tensor of images.

ensembling_kwargs (`dict`, *optional*, defaults to `None`) : Extra dictionary with arguments for precise ensembling control. The following options are available: - reduction (`str`, *optional*, defaults to `"median"`): Defines the ensembling function applied in every pixel location, can be either `"median"` or `"mean"`.

latents (`torch.Tensor`, *optional*, defaults to `None`) : Latent noise tensors to replace the random initialization. These can be taken from the previous function call's output.

generator (`torch.Generator`, or `list[torch.Generator]`, *optional*, defaults to `None`) : Random number generator object to ensure reproducibility.

output_type (`str`, *optional*, defaults to `"np"`) : Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted values are: `"np"` (numpy array) or `"pt"` (torch tensor).

output_uncertainty (`bool`, *optional*, defaults to `False`) : When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that the `ensemble_size` argument is set to a value above 2.

output_latent (`bool`, *optional*, defaults to `False`) : When enabled, the output's `latent` field contains the latent codes corresponding to the predictions within the ensemble. These codes can be saved, modified, and used for subsequent calls with the `latents` argument.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [MarigoldIntrinsicsOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldIntrinsicsOutput) instead of a plain tuple.

**Returns:** [MarigoldIntrinsicsOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldIntrinsicsOutput) or `tuple`

If `return_dict` is `True`, [MarigoldIntrinsicsOutput](/docs/diffusers/v0.43.0/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldIntrinsicsOutput) is returned, otherwise a
`tuple` is returned where the first element is the prediction, the second element is the uncertainty
(or `None`), and the third is the latent (or `None`).

Function invoked when calling the pipeline.

Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
...     "prs-eth/marigold-iid-appearance-v1-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> intrinsics = pipe(image)

>>> vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)
>>> vis[0]["albedo"].save("einstein_albedo.png")
>>> vis[0]["roughness"].save("einstein_roughness.png")
>>> vis[0]["metallicity"].save("einstein_metallicity.png")
```

```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
...     "prs-eth/marigold-iid-lighting-v1-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> intrinsics = pipe(image)

>>> vis = pipe.image_processor.visualize_intrinsics(intrinsics.prediction, pipe.target_properties)
>>> vis[0]["albedo"].save("einstein_albedo.png")
>>> vis[0]["shading"].save("einstein_shading.png")
>>> vis[0]["residual"].save("einstein_residual.png")
```

#### diffusers.pipelines.marigold.MarigoldIntrinsicsOutput[[diffusers.pipelines.marigold.MarigoldIntrinsicsOutput]]

```python
diffusers.pipelines.marigold.MarigoldIntrinsicsOutput(prediction: typing.Union[numpy.ndarray, torch.Tensor], uncertainty: typing.Union[NoneType, numpy.ndarray, torch.Tensor], latent: typing.Optional[torch.Tensor])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py#L96)

**Parameters:**

prediction (`np.ndarray`, `torch.Tensor`) : Predicted image intrinsics with values in the range [0, 1]. The shape is `(numimages * numtargets) × 3 × height × width` for `torch.Tensor` or `(numimages * numtargets) × height × width × 3` for `np.ndarray`, where `numtargets` corresponds to the number of predicted target modalities of the intrinsic image decomposition.

uncertainty (`None`, `np.ndarray`, `torch.Tensor`) : Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is `(numimages * numtargets) × 3 × height × width` for `torch.Tensor` or `(numimages * numtargets) × height × width × 3` for `np.ndarray`.

latent (`None`, `torch.Tensor`) : Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline. The shape is `(numimages * numensemble) × (numtargets * 4) × latentheight × latentwidth`.

Output class for Marigold Intrinsic Image Decomposition pipeline.

#### diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_intrinsics[[diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_intrinsics]]

```python
diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_intrinsics(prediction: typing.Union[numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor]], target_properties: dict[str, typing.Any], color_map: str | dict[str, str] = 'binary')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/marigold/marigold_image_processing.py#L542)

**Parameters:**

prediction (`np.ndarray | torch.Tensor | list[np.ndarray, list[torch.Tensor]]`) : Intrinsic image decomposition.

target_properties (`dict[str, Any]`) : Decomposition properties. Expected entries: `target_names: list[str]` and a dictionary with keys `prediction_space: str`, `sub_target_names: list[str | Null]` (must have 3 entries, null for missing modalities), `up_to_scale: bool`, one for each target and sub-target.

color_map (`str | dict[str, str]`, *optional*, defaults to `"Spectral"`) : Color map used to convert a single-channel predictions into colored representations. When a dictionary is passed, each modality can be colored with its own color map.

Visualizes intrinsic image decomposition, such as predictions of the `MarigoldIntrinsicsPipeline`.

Returns: `list[dict[str, PIL.Image.Image]]` with intrinsic image decomposition visualization.

### Hunyuan Video
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/hunyuan_video.md

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
    dtype=torch.bfloat16,
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
    dtype=torch.bfloat16,
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

- HunyuanVideo supports LoRAs with [load_lora_weights()](/docs/diffusers/v0.43.0/en/api/loaders/lora#diffusers.loaders.HunyuanVideoLoraLoaderMixin.load_lora_weights).

  
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
      dtype=torch.bfloat16,
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

```python
diffusers.HunyuanVideoPipeline(text_encoder: LlamaModel, tokenizer: LlamaTokenizer, transformer: HunyuanVideoTransformer3DModel, vae: AutoencoderKLHunyuanVideo, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder_2: CLIPTextModel, tokenizer_2: CLIPTokenizer)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L144)

**Parameters:**

text_encoder (`LlamaModel`) : [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

tokenizer (`LlamaTokenizer`) : Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).

transformer ([HunyuanVideoTransformer3DModel](/docs/diffusers/v0.43.0/en/api/models/hunyuan_video_transformer_3d#diffusers.HunyuanVideoTransformer3DModel)) : Conditional Transformer to denoise the encoded image latents.

scheduler ([FlowMatchEulerDiscreteScheduler](/docs/diffusers/v0.43.0/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler)) : A scheduler to be used in combination with `transformer` to denoise the encoded image latents.

vae ([AutoencoderKLHunyuanVideo](/docs/diffusers/v0.43.0/en/api/models/autoencoder_kl_hunyuan_video#diffusers.AutoencoderKLHunyuanVideo)) : Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.

text_encoder_2 (`CLIPTextModel`) : [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.

tokenizer_2 (`CLIPTokenizer`) : Tokenizer of class [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).

Pipeline for text-to-video generation using HunyuanVideo.

This model inherits from [DiffusionPipeline](/docs/diffusers/v0.43.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Check the superclass documentation for the generic methods
implemented for all pipelines (downloading, saving, running on a particular device, etc.).

#### __call__[[diffusers.HunyuanVideoPipeline.__call__]]

```python
__call__(prompt: str | list[str] = None, prompt_2: str | list[str] = None, negative_prompt: str | list[str] = None, negative_prompt_2: str | list[str] = None, height: int = 720, width: int = 1280, num_frames: int = 129, num_inference_steps: int = 50, sigmas: list[float] = None, true_cfg_scale: float = 1.0, guidance_scale: float = 6.0, num_videos_per_prompt: int | None = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, latents: typing.Optional[torch.Tensor] = None, prompt_embeds: typing.Optional[torch.Tensor] = None, pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, prompt_attention_mask: typing.Optional[torch.Tensor] = None, negative_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: typing.Optional[torch.Tensor] = None, negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None, output_type: str | None = 'pil', return_dict: bool = True, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: typing.Union[typing.Callable[[int, int], NoneType], diffusers.callbacks.PipelineCallback, diffusers.callbacks.MultiPipelineCallbacks, NoneType] = None, callback_on_step_end_tensor_inputs: list[str] = ['latents'], prompt_template: dict[str, typing.Any] = {'template': '<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>', 'crop_start': 95}, max_sequence_length: int = 256)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video.py#L438)

**Parameters:**

prompt (`str` or `list[str]`, *optional*) : The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`. instead.

prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is will be used instead.

negative_prompt (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is not greater than `1`).

negative_prompt_2 (`str` or `list[str]`, *optional*) : The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.

height (`int`, defaults to `720`) : The height in pixels of the generated image.

width (`int`, defaults to `1280`) : The width in pixels of the generated image.

num_frames (`int`, defaults to `129`) : The number of frames in the generated video.

num_inference_steps (`int`, defaults to `50`) : The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.

sigmas (`list[float]`, *optional*) : Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed will be used.

true_cfg_scale (`float`, *optional*, defaults to 1.0) : True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and `negative_prompt` is provided.

guidance_scale (`float`, defaults to `6.0`) : Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages a model to generate images more aligned with `prompt` at the expense of lower image quality.  Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to the [paper](https://huggingface.co/papers/2210.03142) to learn more.

num_videos_per_prompt (`int`, *optional*, defaults to 1) : The number of images to generate per prompt.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.

latents (`torch.Tensor`, *optional*) : Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.

prompt_embeds (`torch.Tensor`, *optional*) : Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.

pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled text embeddings will be generated from `prompt` input argument.

negative_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.

negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*) : Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt` input argument.

prompt_attention_mask (`torch.Tensor`, *optional*) : Attention mask for `prompt_embeds`. Required when `prompt_embeds` is passed directly.

negative_prompt_attention_mask (`torch.Tensor`, *optional*) : Attention mask for `negative_prompt_embeds`. Required when `negative_prompt_embeds` is passed directly.

output_type (`str`, *optional*, defaults to `"pil"`) : The output format of the generated image. Choose between `PIL.Image` or `np.array`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `HunyuanVideoPipelineOutput` instead of a plain tuple.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

prompt_template (`dict`, *optional*) : Template used to format the prompt before encoding. Defaults to the model's default template.

max_sequence_length (`int`, *optional*, defaults to 256) : Maximum sequence length to use for the prompt encoder.

callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*) : A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of each denoising step during the inference. with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.

callback_on_step_end_tensor_inputs (`list`, *optional*) : The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.

**Returns:** `~HunyuanVideoPipelineOutput` or `tuple`

If `return_dict` is `True`, `HunyuanVideoPipelineOutput` is returned, otherwise a `tuple` is returned
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

## HunyuanVideoPipelineOutput[[diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput]]

#### diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput[[diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput]]

```python
diffusers.pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput(frames: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/pipelines/hunyuan_video/pipeline_output.py#L11)

**Parameters:**

frames (`torch.Tensor`, `np.ndarray`, or list[list[PIL.Image.Image]]) : list of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape `(batch_size, num_frames, channels, height, width)`.

Output class for HunyuanVideo pipelines.

### InstructPix2Pix
https://huggingface.co/docs/diffusers/v0.43.0/api/pipelines/pix2pix.md
