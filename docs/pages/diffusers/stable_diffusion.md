# Basic performance

Diffusion is a random process that is computationally demanding. You may need to run the [DiffusionPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) several times before getting a desired output. That's why it's important to carefully balance generation speed and memory usage in order to iterate faster,

This guide recommends some basic performance tips for using the [DiffusionPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline). Refer to the Inference Optimization section docs such as [Accelerate inference](./optimization/fp16) or [Reduce memory usage](./optimization/memory) for more detailed performance guides.

## Memory usage

Reducing the amount of memory used indirectly speeds up generation and can help a model fit on device.

The [enable_model_cpu_offload()](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_model_cpu_offload) method moves a model to the CPU when it is not in use to save GPU memory.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.bfloat16,
  device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.enable_model_cpu_offload()

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

## Inference speed

Denoising is the most computationally demanding process during diffusion. Methods that optimizes this process accelerates inference speed. Try the following methods for a speed up.

- Add `device_map="cuda"` to place the pipeline on a GPU. Placing a model on an accelerator, like a GPU, increases speed because it performs computations in parallel.
- Set `dtype=torch.bfloat16` to execute the pipeline in half-precision. Reducing the data type precision increases speed because it takes less time to perform computations in a lower precision.

```py
import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.bfloat16,
  device_map="cuda
)
```

- Use a faster scheduler, such as [DPMSolverMultistepScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/multistep_dpm_solver#diffusers.DPMSolverMultistepScheduler), which only requires ~20-25 steps.
- Set `num_inference_steps` to a lower value. Reducing the number of inference steps reduces the overall number of computations. However, this can result in lower generation quality.

```py
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

start_time = time.perf_counter()
image = pipeline(prompt).images[0]
end_time = time.perf_counter()

print(f"Image generation took {end_time - start_time:.3f} seconds")
```

## Generation quality

Many modern diffusion models deliver high-quality images out-of-the-box. However, you can still improve generation quality by trying the following.

- Try a more detailed and descriptive prompt. Include details such as the image medium, subject, style, and aesthetic. A negative prompt may also help by guiding a model away from undesirable features by using words like low quality or blurry.

    ```py
    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.bfloat16,
        device_map="cuda"  # or "mps", "xpu", "cpu"
    )

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt).images[0]
    ```

    For more details about creating better prompts, take a look at the [Prompt techniques](./using-diffusers/weighted_prompts) doc.

- Try a different scheduler, like [HeunDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/heun#diffusers.HeunDiscreteScheduler) or [LMSDiscreteScheduler](/docs/diffusers/v0.41.0/en/api/schedulers/lms_discrete#diffusers.LMSDiscreteScheduler), that gives up generation speed for quality.

    ```py
    import torch
    from diffusers import DiffusionPipeline, HeunDiscreteScheduler

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.bfloat16,
        device_map="cuda"  # or "mps", "xpu", "cpu"
    )
    pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt).images[0]
    ```

## Next steps

Diffusers offers more advanced and powerful optimizations such as [group-offloading](./optimization/memory#group-offloading) and [regional compilation](./optimization/fp16#regional-compilation). To learn more about how to maximize performance, take a look at the Inference Optimization section.

### Diffusers
https://huggingface.co/docs/diffusers/v0.41.0/index.md

# Diffusers

Diffusers provides pretrained diffusion models and the building blocks for custom image, video, and audio workflows.

It has two main paths.

- [DiffusionPipeline](/docs/diffusers/v0.41.0/en/api/pipelines/overview#diffusers.DiffusionPipeline) supports few-line inference with pretrained checkpoints, plus adapters like LoRA. This is the easy path for generation.
- [Modular Diffusers](./modular_diffusers/overview) enables composable blocks and [ModularPipeline](/docs/diffusers/v0.41.0/en/api/modular_diffusers/pipeline#diffusers.ModularPipeline) for custom pipelines when you need more control.

Optimizations such as offloading and quantization keep large models runnable on memory-constrained devices. If memory is not an issue, Diffusers also supports `torch.compile` for faster inference.

Browse trending Diffusers models on the [Hub](https://huggingface.co/models?library=diffusers&sort=trending) now.

## Learn

If you're a beginner, start with the [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1). It covers diffusion theory and how to generate images, fine-tune models, and more with Diffusers.

The [Quickstart](./quicktour) also includes a copyable agent setup prompt for inference.

## Where next

- [Inference](./using-diffusers/loading) — load pipelines and run generation
- [Optimize and scale](./stable_diffusion) — memory, speed, quantization, and serving
- [Modular Diffusers](./modular_diffusers/overview) — build custom pipelines from blocks
- [Train and fine-tune](./training/overview) — train diffusion models and adapters
- [CLI](./using-diffusers/cli) - run and package pipelines from the command line

### LoRA
https://huggingface.co/docs/diffusers/v0.41.0/tutorials/using_peft_for_inference.md
