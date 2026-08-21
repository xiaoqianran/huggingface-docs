# Metal Performance Shaders (MPS)

> [!TIP]
> Pipelines with a  badge indicate a model can take advantage of the MPS backend on Apple silicon devices for faster inference. Feel free to open a [Pull Request](https://github.com/huggingface/diffusers/compare) to add this badge to pipelines that are missing it.

🤗 Diffusers is compatible with Apple silicon (M1/M2 chips) using the PyTorch [`mps`](https://pytorch.org/docs/stable/notes/mps.html) device, which uses the Metal framework to leverage the GPU on MacOS devices. You'll need to have:

- macOS computer with Apple silicon (M1/M2) hardware
- macOS 12.6 or later (13.0 or later recommended)
- arm64 version of Python
- [PyTorch 2.0](https://pytorch.org/get-started/locally/) (recommended) or 1.13 (minimum version supported for `mps`)

The `mps` backend uses PyTorch's `.to()` interface to move the Stable Diffusion pipeline on to your M1 or M2 device:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image
```

> [!WARNING]
> The PyTorch [mps](https://pytorch.org/docs/stable/notes/mps.html) backend does not support NDArray sizes greater than `2**32`. Please open an [Issue](https://github.com/huggingface/diffusers/issues/new/choose) if you encounter this problem so we can investigate.

If you're using **PyTorch 1.13**, you need to "prime" the pipeline with an additional one-time pass through it. This is a temporary workaround for an issue where the first inference pass produces slightly different results than subsequent ones. You only need to do this pass once, and after just one inference step you can discard the result.

```diff
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to("mps")
  pipe.enable_attention_slicing()

  prompt = "a photo of an astronaut riding a horse on mars"
  # First-time "warmup" pass if PyTorch version is 1.13
+ _ = pipe(prompt, num_inference_steps=1)

  # Results match those from the CPU device after the warmup pass.
  image = pipe(prompt).images[0]
```

## Troubleshoot

This section lists some common issues with using the `mps` backend and how to solve them.

### Attention slicing

M1/M2 performance is very sensitive to memory pressure. When this occurs, the system automatically swaps if it needs to which significantly degrades performance.

To prevent this from happening, we recommend *attention slicing* to reduce memory pressure during inference and prevent swapping. This is especially relevant if your computer has less than 64GB of system RAM, or if you generate images at non-standard resolutions larger than 512×512 pixels. Call the [enable_attention_slicing()](/docs/diffusers/v0.40.0/en/api/pipelines/overview#diffusers.DiffusionPipeline.enable_attention_slicing) function on your pipeline:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipeline.enable_attention_slicing()
```

Attention slicing performs the costly attention operation in multiple steps instead of all at once. It usually improves performance by ~20% in computers without universal memory, but we've observed *better performance* in most Apple silicon computers unless you have 64GB of RAM or more.

### Batch inference

Generating multiple prompts in a batch can crash or fail to work reliably. If this is the case, try iterating instead of batching.

### CacheDiT
https://huggingface.co/docs/diffusers/v0.40.0/optimization/cache_dit.md

## CacheDiT  

CacheDiT is a unified, flexible, and training-free cache acceleration framework designed to support nearly all Diffusers' DiT-based pipelines. It provides a unified cache API that supports automatic block adapter, DBCache, and more.

To learn more, refer to the [CacheDiT](https://github.com/vipshop/cache-dit) repository.

Install a stable release of CacheDiT from PyPI or you can install the latest version from GitHub.

```bash
pip3 install -U cache-dit
```

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```

Run the command below to view supported DiT pipelines.

```python
>>> import cache_dit
>>> cache_dit.supported_pipelines()
(30, ['Flux*', 'Mochi*', 'CogVideoX*', 'Wan*', 'HunyuanVideo*', 'QwenImage*', 'LTX*', 'Allegro*',
'CogView3Plus*', 'CogView4*', 'Cosmos*', 'EasyAnimate*', 'SkyReelsV2*', 'StableDiffusion3*',
'ConsisID*', 'DiT*', 'Amused*', 'Bria*', 'Lumina*', 'OmniGen*', 'PixArt*', 'Sana*', 'StableAudio*',
'VisualCloze*', 'AuraFlow*', 'Chroma*', 'ShapE*', 'HiDream*', 'HunyuanDiT*', 'HunyuanDiTPAG*'])
```

For a complete benchmark, please refer to [Benchmarks](https://github.com/vipshop/cache-dit/blob/main/bench/).

## Unified Cache API

CacheDiT works by matching specific input/output patterns as shown below.

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns-v1.png)

Call the `enable_cache()` function on a pipeline to enable cache acceleration. This function is the entry point to many of CacheDiT's features.

```python
import cache_dit
from diffusers import DiffusionPipeline 

<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# Can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# One-line code with default cache options.
cache_dit.enable_cache(pipe) 

# Just call the pipe as normal.
output = pipe(...)

# Disable cache and run original pipe.
cache_dit.disable_cache(pipe)
```

## Automatic Block Adapter

For custom or modified pipelines or transformers not included in Diffusers, use the `BlockAdapter` in `auto` mode or via manual configuration. Please check the [BlockAdapter](https://github.com/vipshop/cache-dit/blob/main/docs/User_Guide.md#automatic-block-adapter) docs for more details. Refer to [Qwen-Image w/ BlockAdapter](https://github.com/vipshop/cache-dit/blob/main/examples/adapter/run_qwen_image_adapter.py) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter

# Use 🔥BlockAdapter with `auto` mode.
cache_dit.enable_cache(
    BlockAdapter(
        # Any DiffusionPipeline, Qwen-Image, etc.  
        pipe=pipe, auto=True,
        # Check `📚Forward Pattern Matching` documentation and hack the code of
        # of Qwen-Image, you will find that it has satisfied `FORWARD_PATTERN_1`.
        forward_pattern=ForwardPattern.Pattern_1,
    ),   
)

# Or, manually setup transformer configurations.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # Qwen-Image, etc.
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
    ), 
)
```

Sometimes, a Transformer class will contain more than one transformer `blocks`. For example, FLUX.1 (HiDream, Chroma, etc) contains `transformer_blocks` and `single_transformer_blocks` (with different forward patterns). The BlockAdapter is able to detect this hybrid pattern type as well. 
Refer to [FLUX.1](https://github.com/vipshop/cache-dit/blob/main/examples/adapter/run_flux_adapter.py) as an example.

```python
# For diffusers <= 0.34.0, FLUX.1 transformer_blocks and 
# single_transformer_blocks have different forward patterns.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # FLUX.1, etc.
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    ),
)
```

This also works if there is more than one transformer (namely `transformer` and `transformer_2`) in its structure. Refer to [Wan 2.2 MoE](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_wan_2.2.py) as an example.

## Patch Functor

For any pattern not included in CacheDiT, use the Patch Functor to convert the pattern into a known pattern. You need to subclass the Patch Functor and may also need to fuse the operations within the blocks for loop into block `forward`. After implementing a Patch Functor, set the `patch_functor` property in `BlockAdapter`.

![](https://github.com/vipshop/cache-dit/raw/main/assets/patch-functor.png)

Some Patch Functors are already provided in CacheDiT, [HiDreamPatchFunctor](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/cache_factory/patch_functors/functor_hidream.py), [ChromaPatchFunctor](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/cache_factory/patch_functors/functor_chroma.py), etc.

```python
@BlockAdapterRegistry.register("HiDream")
def hidream_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HiDreamImageTransformer2DModel
    from cache_dit.cache_factory.patch_functors import HiDreamPatchFunctor

    assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.double_stream_blocks,
            pipe.transformer.single_stream_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_0,
            ForwardPattern.Pattern_3,
        ],
        # NOTE: Setup your custom patch functor here.
        patch_functor=HiDreamPatchFunctor(),
        **kwargs,
    )
```

Finally, you can call the `cache_dit.summary()` function on a pipeline after its completed inference to get the cache acceleration details.

```python
stats = cache_dit.summary(pipe)
```

```python
⚡️Cache Steps and Residual Diffs Statistics: QwenImagePipeline

| Cache Steps | Diffs Min | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| 23          | 0.045     | 0.084     | 0.114     | 0.147     | 0.241     | 0.297     |
```

## DBCache: Dual Block Cache  

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

DBCache (Dual Block Caching) supports different configurations of compute blocks (F8B12, etc.) to enable a balanced trade-off between performance and precision.
- Fn_compute_blocks: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- Bn_compute_blocks: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

```python
import cache_dit
from diffusers import FluxPipeline

pipe_or_adapter = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B0, 8 warmup steps, and unlimited cached 
# steps for good balance between performance and precision
cache_dit.enable_cache(pipe_or_adapter)

# Custom options, F8B8, higher precision
from cache_dit import BasicCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=BasicCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
)
```  
Check the [DBCache](https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md) and [User Guide](https://github.com/vipshop/cache-dit/blob/main/docs/User_Guide.md#dbcache) docs for more design details.

## TaylorSeer Calibrator

The [TaylorSeers](https://huggingface.co/papers/2503.06923) algorithm further improves the precision of DBCache in cases where the cached steps are large (Hybrid TaylorSeer + DBCache). At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

TaylorSeer employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in CacheDiT supports both hidden states and residual cache types. F_pred can be a residual cache or a hidden-state cache.

```python
from cache_dit import BasicCacheConfig, TaylorSeerCalibratorConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    # Basic DBCache w/ FnBn configurations
    cache_config=BasicCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
    # Then, you can use the TaylorSeer Calibrator to approximate 
    # the values in cached steps, taylorseer_order default is 1.
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,
    ),
)
``` 

> [!TIP]  
> The `Bn_compute_blocks` parameter of DBCache can be set to `0` if you use TaylorSeer as the calibrator for approximate hidden states. DBCache's `Bn_compute_blocks` also acts as a calibrator, so you can choose either `Bn_compute_blocks` > 0 or TaylorSeer. We recommend using the configuration scheme of TaylorSeer + DBCache FnB0.

## Hybrid Cache CFG

CacheDiT supports caching for CFG (classifier-free guidance). For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG in the forward step, please set `enable_separate_cfg` parameter  to `False (default, None)`. Otherwise, set it to `True`. 

```python
from cache_dit import BasicCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=BasicCacheConfig(
        ...,
        # For example, set it as True for Wan 2.1, Qwen-Image 
        # and set it as False for FLUX.1, HunyuanVideo, etc.
        enable_separate_cfg=True,
    ),
)
```

## torch.compile

CacheDiT is designed to work with torch.compile for even better performance. Call `torch.compile` after enabling the cache.

```python
cache_dit.enable_cache(pipe)

# Compile the Transformer module
pipe.transformer = torch.compile(pipe.transformer)
```

If you're using CacheDiT with dynamic input shapes, consider increasing the `recompile_limit` of `torch._dynamo`. Otherwise, the `recompile_limit` error may be triggered, causing the module to fall back to eager mode. 

```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```

Please check [perf.py](https://github.com/vipshop/cache-dit/blob/main/bench/perf.py) for more details.

### T-GATE
https://huggingface.co/docs/diffusers/v0.40.0/optimization/tgate.md
