# Caching

Caching accelerates inference by storing and reusing intermediate outputs of different layers, such as attention and feedforward layers, instead of performing the entire computation at each inference step. It significantly improves generation speed at the expense of more memory and doesn't require additional training.

This guide shows you how to use the caching methods supported in Diffusers.

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast (PAB)](https://huggingface.co/papers/2408.12588) is based on the observation that attention outputs aren't that different between successive timesteps of the generation process. The attention differences are smallest in the cross attention layers and are generally cached over a longer timestep range. This is followed by temporal attention and spatial attention layers.

> [!TIP]
> Not all video models have three types of attention (cross, temporal, and spatial)!

PAB can be combined with other techniques like sequence parallelism and classifier-free guidance parallelism (data parallelism) for near real-time video generation.

Set up and pass a [PyramidAttentionBroadcastConfig](/docs/diffusers/v0.39.0/en/api/cache#diffusers.PyramidAttentionBroadcastConfig) to a pipeline's transformer to enable it. The `spatial_attention_block_skip_range` controls how often to skip attention calculations in the spatial attention blocks and the `spatial_attention_timestep_skip_range` is the range of timesteps to skip. Take care to choose an appropriate range because a smaller interval can lead to slower inference speeds and a larger interval can result in lower generation quality.

```python
import torch
from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe.current_timestep,
)
pipeline.transformer.enable_cache(config)
```

## FasterCache

[FasterCache](https://huggingface.co/papers/2410.19355) caches and reuses attention features similar to [PAB](#pyramid-attention-broadcast) since output differences are small for each successive timestep.

This method may also choose to skip the unconditional branch prediction, when using classifier-free guidance for sampling (common in most base models), and estimate it from the conditional branch prediction if there is significant redundancy in the predicted latent outputs between successive timesteps.

Set up and pass a [FasterCacheConfig](/docs/diffusers/v0.39.0/en/api/cache#diffusers.FasterCacheConfig) to a pipeline's transformer to enable it.

```python
import torch
from diffusers import CogVideoXPipeline, FasterCacheConfig

pipe line= CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

config = FasterCacheConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(-1, 681),
    current_timestep_callback=lambda: pipe.current_timestep,
    attention_weight_callback=lambda _: 0.3,
    unconditional_batch_skip_range=5,
    unconditional_batch_timestep_skip_range=(-1, 781),
    tensor_format="BFCHW",
)
pipeline.transformer.enable_cache(config)
```

## FirstBlockCache

[FirstBlock Cache](https://huggingface.co/docs/diffusers/main/en/api/cache#diffusers.FirstBlockCacheConfig) checks how much the early layers of the denoiser changes from one timestep to the next. If the change is small, the model skips the expensive later layers and reuses the previous output.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16
)
apply_first_block_cache(pipeline.transformer, FirstBlockCacheConfig(threshold=0.2))
```
## TaylorSeer Cache

[TaylorSeer Cache](https://huggingface.co/papers/2403.06923) accelerates diffusion inference by using Taylor series expansions to approximate and cache intermediate activations across denoising steps. The method predicts future outputs based on past computations, reusing them at specified intervals to reduce redundant calculations.

This caching mechanism delivers strong results with minimal additional memory overhead. For detailed performance analysis, see [our findings here](https://github.com/huggingface/diffusers/pull/12648#issuecomment-3610615080).

To enable TaylorSeer Cache, create a [TaylorSeerCacheConfig](/docs/diffusers/v0.39.0/en/api/cache#diffusers.TaylorSeerCacheConfig) and pass it to your pipeline's transformer:

- `cache_interval`: Number of steps to reuse cached outputs before performing a full forward pass
- `disable_cache_before_step`: Initial steps that use full computations to gather data for approximations
- `max_order`: Approximation accuracy (in theory, higher values improve quality but increase memory usage but we recommend it should be set to `1`)

```python
import torch
from diffusers import FluxPipeline, TaylorSeerCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

config = TaylorSeerCacheConfig(
    cache_interval=5,
    max_order=1,
    disable_cache_before_step=10,
    taylor_factors_dtype=torch.bfloat16,
)
pipe.transformer.enable_cache(config)
```

## MagCache

[MagCache](https://github.com/Zehong-Ma/MagCache) accelerates inference by skipping transformer blocks based on the magnitude of the residual update. It observes that the magnitude of updates (Output - Input) decays predictably over the diffusion process. By accumulating an "error budget" based on pre-computed magnitude ratios, it dynamically decides when to skip computation and reuse the previous residual.

MagCache relies on **Magnitude Ratios** (`mag_ratios`), which describe this decay curve. These ratios are specific to the model checkpoint and scheduler.

To use MagCache, you typically follow a two-step process: **Calibration** and **Inference**.

1.  **Calibration**: Run inference once with `calibrate=True`. The hook will measure the residual magnitudes and print the calculated ratios to the console.
2.  **Inference**: Pass these ratios to `MagCacheConfig` to enable acceleration.

```python
import torch
from diffusers import FluxPipeline, MagCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda")

# 1. Calibration Step
# Run full inference to measure model behavior.
calib_config = MagCacheConfig(calibrate=True, num_inference_steps=4)
pipe.transformer.enable_cache(calib_config)

# Run a prompt to trigger calibration
pipe("A cat playing chess", num_inference_steps=4)
# Logs will print something like: "MagCache Calibration Results: [1.0, 1.37, 0.97, 0.87]"

# 2. Inference Step
# Apply the specific ratios obtained from calibration for optimized speed.
# Note: For Flux models, you can also import defaults: 
# from diffusers.hooks.mag_cache import FLUX_MAG_RATIOS
mag_config = MagCacheConfig(
    mag_ratios=[1.0, 1.37, 0.97, 0.87],
    num_inference_steps=4
)

pipe.transformer.enable_cache(mag_config) 

image = pipe("A cat playing chess", num_inference_steps=4).images[0]
```

> [!NOTE]
> `mag_ratios` represent the model's intrinsic magnitude decay curve. Ratios calibrated for a high number of steps (e.g., 50) can be reused for lower step counts (e.g., 20). The implementation uses interpolation to map the curve to the current number of inference steps.

> [!TIP]
> For pipelines that run Classifier-Free Guidance sequentially (like Kandinsky 5.0), the calibration log might print two arrays: one for the Conditional pass and one for the Unconditional pass. In most cases, you should use the first array (Conditional).

> [!TIP]
> For pipelines that run Classifier-Free Guidance in a **batched** manner (like SDXL or Flux), the `hidden_states` processed by the model contain both conditional and unconditional branches concatenated together. The calibration process automatically accounts for this, producing a single array of ratios that represents the joint behavior. You can use this resulting array directly without modification.

### CacheDiT
https://huggingface.co/docs/diffusers/v0.39.0/optimization/cache_dit.md

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
    torch_dtype=torch.bfloat16,
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

### OpenVINO
https://huggingface.co/docs/diffusers/v0.39.0/optimization/open_vino.md
