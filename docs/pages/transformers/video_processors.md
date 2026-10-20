# Video Processor

A **Video Processor** is a utility responsible for preparing input features for video models, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch.

The video processor extends the functionality of image processors by allowing the models to handle videos with a distinct set of arguments compared to images. It serves as the bridge between raw video data and the model, ensuring that input features are optimized for the VLM.

Use [from_pretrained()](/docs/transformers/v5.20.2/en/main_classes/video_processor#transformers.VideoProcessorMixin.from_pretrained) to load a video processors configuration (image size, whether to normalize and rescale, etc.) from a video model on the Hugging Face [Hub](https://hf.co) or local directory. The configuration for each pretrained model should be saved in a [video_preprocessor_config.json] file but older models might have the config saved in [preprocessor_config.json](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/preprocessor_config.json) file. Note that the latter is less preferred and will be removed in the future.

## Usage Example

Here's an example of how to load a video processor with [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) model:

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

Currently, if using base image processor for videos, it processes video data by treating each frame as an individual image and applying transformations frame-by-frame. While functional, this approach is not highly efficient. Using `AutoVideoProcessor` allows us to take advantage of **fast video processors**, leveraging the [torchvision](https://pytorch.org/vision/stable/index.html) library. Fast processors handle the whole batch of videos at once, without iterating over each video or frame. These updates introduce GPU acceleration and significantly enhance processing speed, especially for tasks requiring high throughput.

Fast video processors are available for all models and are loaded by default when an `AutoVideoProcessor` is initialized. When using a fast video processor, you can also set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise. For even more speed improvement, we can compile the processor when using an accelerator as device.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device=device)
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```

## Frame sampling

A video processor decodes a video and picks the frames the model actually sees. Set `do_sample_frames=True` to turn sampling on, then pick the frames in one of two ways.

- `num_frames` asks for a fixed count spread uniformly across the video.
- `fps` asks for a rate in frames per second.

Pass a path or URL and the processor decodes the video for you with [torchcodec](https://meta-pytorch.org/torchcodec/stable/index.html), the default decoder. If torchcodec isn't available and you're on an older torchvision version, decoding falls back to torchvision.

Add `return_metadata=True` to get back the sampled frame indices, original dimensions, duration, and fps.

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
inputs = processor(videos=["video.mp4"], do_sample_frames=True, fps=1, return_metadata=True, return_tensors="pt")

print(inputs.pixel_values_videos.shape, inputs.video_grid_thw)
print(inputs["video_metadata"][0].total_num_frames)
```

Sampling defaults are per model, and a request for `num_frames` or `fps` is a starting point rather than a guarantee. Qwen3-VL samples at `fps=2` and clamps the result to between `min_frames=4` and `max_frames=768`, so a one-second clip still yields 4 frames and a long recording is capped well below its real frame count.

Models with a `temporal_patch_size` add a second constraint where frames are grouped into patches of that many frames along the time axis, and the final frame is repeated until the count divides evenly.

If you pass an already decoded video array and still want model-specific sampling, provide `video_metadata` as well. Without it, the sampler doesn't know the original duration or fps, and sampling by `fps` warns and assumes the video was recorded at 24 fps.

```python
import torch
from transformers import AutoVideoProcessor
from transformers.video_utils import VideoMetadata

processor = AutoVideoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
video = torch.randint(0, 255, size=(100, 3, 1280, 1280))  # short video of 100 frames
video_metadata = VideoMetadata(total_num_frames=100, fps=24, duration=4.1)

inputs = processor(
    videos=[video], video_metadata=[video_metadata], do_sample_frames=True, num_frames=16, return_tensors="pt"
)
```

### Transformers
https://huggingface.co/docs/transformers/v5.20.2/index.md

# Transformers

    

Transformers acts as the model-definition framework for state-of-the-art machine learning models in text, computer
vision, audio, video, and multimodal models, for both inference and training.

It centralizes the model definition so that this definition is agreed upon across the ecosystem. `transformers` is the
pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training
frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference engines (vLLM, SGLang, TGI, ...),
and adjacent modeling libraries (llama.cpp, mlx, ...) which leverage the model definition from `transformers`.

We pledge to help support new state-of-the-art models and democratize their usage by having their model definition be
simple, customizable, and efficient.

There are over 1M+ Transformers [model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) on the [Hugging Face Hub](https://huggingface.com/models) you can use.

Explore the [Hub](https://huggingface.com/) today to find a model and use Transformers to help you get started right away.

Explore the [Models Timeline](./models_timeline) to discover the latest text, vision, audio and multimodal model architectures in Transformers.

## Features

Transformers provides everything you need for inference or training with state-of-the-art pretrained models. Some of the main features include:

- [Pipeline](./pipeline_tutorial): Simple and optimized inference class for many machine learning tasks like text generation, image segmentation, automatic speech recognition, document question answering, and more.
- [Trainer](./trainer): A comprehensive trainer that supports features such as mixed precision, torch.compile, and FlashAttention for training and distributed training for PyTorch models.
- [generate](./llm_tutorial): Fast text generation with large language models (LLMs) and vision language models (VLMs), including support for streaming and multiple decoding strategies.

## Design

> [!TIP]
> Read our [Philosophy](./philosophy) to learn more about Transformers' design principles.

Transformers is designed for developers and machine learning engineers and researchers. Its main design principles are:

1. Fast and easy to use: Every model is implemented from only three main classes (configuration, model, and preprocessor) and can be quickly used for inference or training with [Pipeline](/docs/transformers/v5.20.2/en/main_classes/pipelines#transformers.Pipeline) or [Trainer](/docs/transformers/v5.20.2/en/main_classes/trainer#transformers.Trainer).
2. Pretrained models: Reduce your carbon footprint, compute cost and time by using a pretrained model instead of training an entirely new one. Each pretrained model is reproduced as closely as possible to the original model and offers state-of-the-art performance.

  
      
  

## Learn

If you're new to Transformers or want to learn more about transformer models, we recommend starting with the [LLM course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt). This comprehensive course covers everything from the fundamentals of how transformer models work to practical applications across various tasks. You'll learn the complete workflow, from curating high-quality datasets to fine-tuning large language models and implementing reasoning capabilities. The course contains both theoretical and hands-on exercises to build a solid foundational knowledge of transformer models as you learn.

### Generation features
https://huggingface.co/docs/transformers/v5.20.2/generation_features.md
