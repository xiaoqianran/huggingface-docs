# Kandinsky 6 Transformers

Kandinsky 6 uses a multimodal diffusion transformer that denoises video and audio latents together for
text/image-to-video-and-audio generation, and a text-free diffusion transformer for video super-resolution.

## Kandinsky6Transformer3DModel[[diffusers.Kandinsky6Transformer3DModel]]

The multimodal transformer used by [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.0/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline).

```python
import torch
from diffusers import Kandinsky6Transformer3DModel

transformer = Kandinsky6Transformer3DModel.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)
```

#### diffusers.Kandinsky6Transformer3DModel[[diffusers.Kandinsky6Transformer3DModel]]

```python
diffusers.Kandinsky6Transformer3DModel(in_visual_dim: int = 16, out_visual_dim: int = 16, in_text_dim: int = 3584, in_text_dim2: int = 768, time_dim: int = 1024, patch_size: tuple = (1, 2, 2), model_dim: int = 4096, ff_dim: int = 16384, num_text_blocks: int = 4, num_visual_blocks: int = 60, axes_dims: tuple = (32, 48, 48), visual_cond: bool = True, in_audio_dim: int = 20, out_audio_dim: int = 20, model_dim_a: int | None = None, time_dim_a: int | None = None, ff_dim_a: int | None = None, axes_dims_a: tuple | None = None, audio_freqs_scaling: float = 1.0, scale_factor: tuple | list[float] = (1.0, 2.0, 2.0), text_token_padding: bool = False, ca_rope: bool = False, cross_gates: bool = False, fix_modulation: bool = False, visual_token_type_num_embeddings: int = 0)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_kandinsky6.py#L591)

**Parameters:**

in_visual_dim (*int*, *optional*, defaults to 16) : Number of input video latent channels.

out_visual_dim (*int*, *optional*, defaults to 16) : Number of output video latent channels.

in_text_dim (*int*, *optional*, defaults to 3584) : Text token embedding dimension.

in_text_dim2 (*int*, *optional*, defaults to 768) : Pooled text embedding dimension.

time_dim (*int*, *optional*, defaults to 1024) : Time embedding dimension.

patch_size (*tuple[int, int, int]*, *optional*, defaults to `(1, 2, 2)`) : Video patch size.

model_dim (*int*, *optional*, defaults to 4096) : Video transformer hidden dimension.

ff_dim (*int*, *optional*, defaults to 16384) : Video feed-forward hidden dimension.

num_text_blocks (*int*, *optional*, defaults to 4) : Number of text blocks per modality.

num_visual_blocks (*int*, *optional*, defaults to 60) : Number of fused video/audio blocks.

axes_dims (*tuple[int, int, int]*, *optional*, defaults to `(32, 48, 48)`) : RoPE dimensions for video.

visual_cond (*bool*, *optional*, defaults to True) : Whether video conditioning channels are present.

in_audio_dim (*int*, *optional*, defaults to 20) : Number of input audio latent channels.

out_audio_dim (*int*, *optional*, defaults to 20) : Number of output audio latent channels.

model_dim_a (*int*, *optional*) : Audio transformer hidden dimension. Defaults to *model_dim*.

time_dim_a (*int*, *optional*) : Audio time embedding dimension. Defaults to *time_dim*.

ff_dim_a (*int*, *optional*) : Audio feed-forward hidden dimension. Defaults to *ff_dim*.

axes_dims_a (*tuple[int, int, int]*, *optional*) : Audio RoPE dimensions. Defaults to *axes_dims*.

audio_freqs_scaling (*float*, *optional*, defaults to 1.0) : Audio RoPE frequency scaling.

scale_factor (*tuple[float, float, float]*, *optional*, defaults to *(1.0, 2.0, 2.0)*) : Per-axis *(t, h, w)* RoPE frequency scaling applied to the video positions.

text_token_padding (*bool*, *optional*, defaults to False) : Checkpoint metadata recording whether the reference model's text sequences are padded. *forward* always accepts an optional *encoder_attention_mask* regardless of this flag; whether one is actually passed is entirely up to the caller.

ca_rope (*bool*, *optional*, defaults to False) : Whether to use cross-modal audio RoPE.

cross_gates (*bool*, *optional*, defaults to False) : Whether to use cross-modal residual gates.

fix_modulation (*bool*, *optional*, defaults to False) : Whether to use the fixed modulation variant.

visual_token_type_num_embeddings (*int*, *optional*, defaults to 0) : Number of visual token type embeddings.

Kandinsky 6 multimodal transformer for text/image-to-video-and-audio generation.

Video and audio are denoised together through fused, cross-modal transformer blocks, each conditioned on its own
text branch (Qwen2.5-VL tokens + CLIP pooled embedding). Passing no *audio_hidden_states* denoises video alone
while still running the fused block's video self-attention, cross-attention, and feed-forward stages. Rotary
embeddings are computed inside *forward* from the token grid.

Released checkpoints set *text_token_padding*, *ca_rope*, *cross_gates*, and *fix_modulation* to *True* (see each
checkpoint's *transformer/config.json*); the *False* defaults only describe an architecture variant this repo does
not ship weights for.

#### forward[[diffusers.Kandinsky6Transformer3DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, pooled_projections: Tensor, timestep: Tensor, audio_hidden_states: Tensor | None = None, visual_rope_pos: tuple[Tensor, Tensor, Tensor] | None = None, encoder_attention_mask: Tensor | None = None, visual_token_type_ids: Tensor | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_kandinsky6.py#L739)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, height, width, in_channels)`) : Video latents in the `(B, T, H, W, C)` layout. With `visual_cond=True`, `in_channels` is `2 * in_visual_dim + 1`: the noisy latent, the conditioning latent and a conditioning mask.

encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, in_text_dim)`) : Text token embeddings, shared by the video and audio text branches.

pooled_projections (`torch.Tensor` of shape `(batch_size, in_text_dim2)`) : Pooled text embedding added to the timestep embedding of both branches.

timestep (`torch.Tensor` of shape `(batch_size,)`) : Diffusion timestep on the `[0, num_train_timesteps]` scale, shared by both modalities.

audio_hidden_states (`torch.Tensor` of shape `(batch_size, audio_length, in_audio_dim)`, *optional*) : Audio latents. When omitted the fused blocks run their video path only.

visual_rope_pos (`tuple[torch.Tensor, torch.Tensor, torch.Tensor]`, *optional*) : Per-axis `(t, h, w)` rotary position indices of the patchified video tokens. Defaults to `arange` over each axis. The pipeline passes an explicit temporal index when it appends a reference frame that reuses position `0`.

encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Boolean padding mask for `encoder_hidden_states`. Pass `None` when no prompt is padded.

visual_token_type_ids (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*) : Per-frame token type ids, embedded through `visual_token_type_embeddings`. Requires `visual_token_type_num_embeddings > 0`.

return_dict (`bool`, defaults to `True`) : Whether to return a `Kandinsky6TransformerOutput` instead of a plain tuple.

**Returns:** `Kandinsky6TransformerOutput` or `tuple`

The predicted video velocity in the `(B, T, H, W, out_visual_dim)` layout and, when
`audio_hidden_states` was given, the predicted audio velocity of shape `(B, audio_length,
out_audio_dim)`.

## Kandinsky6SRTransformer3DModel[[diffusers.Kandinsky6SRTransformer3DModel]]

The text-free transformer used by [Kandinsky6SRPipeline](/docs/diffusers/v0.42.0/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) to refine one tile of the upscaled video at a time.

```python
import torch
from diffusers import Kandinsky6SRTransformer3DModel

transformer = Kandinsky6SRTransformer3DModel.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)
# The transformer always runs NABLA sparse attention (`nabla_threshold`, 0.8 by default) on the `flex` backend.
# Compile it, otherwise flex falls back to an eager implementation that needs far more memory at video resolutions.
transformer.compile_repeated_blocks(fullgraph=True)
```

#### diffusers.Kandinsky6SRTransformer3DModel[[diffusers.Kandinsky6SRTransformer3DModel]]

```python
diffusers.Kandinsky6SRTransformer3DModel(in_visual_dim: int = 64, out_visual_dim: int = 640, time_dim: int = 512, patch_size: tuple[int, int, int] = (1, 1, 1), model_dim: int = 1792, ff_dim: int = 7168, num_visual_blocks: int = 32, axes_dims: tuple[int, int, int] = (16, 24, 24), scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0), nabla_threshold: float = 0.8, nabla_window: tuple[int, int, int] = (11, 7, 7), tile_sizes: tuple[tuple[int, int], ...] = ((512, 512), (512, 768), (768, 512)))
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_kandinsky6_sr.py#L412)

**Parameters:**

in_visual_dim (`int`, defaults to `64`) : Number of latent channels of the K-VAE.

out_visual_dim (`int`, defaults to `640`) : Number of output channels.

time_dim (`int`, defaults to `512`) : Dimension of the timestep embedding.

patch_size (`tuple[int, int, int]`, defaults to `(1, 1, 1)`) : Patch size as `(temporal, height, width)`.

model_dim (`int`, defaults to `1792`) : Hidden dimension of the transformer.

ff_dim (`int`, defaults to `7168`) : Inner dimension of the feed-forward networks.

num_visual_blocks (`int`, defaults to `32`) : Number of transformer blocks.

axes_dims (`tuple[int, int, int]`, defaults to `(16, 24, 24)`) : RoPE dimensions per `(t, h, w)` axis; their sum is the attention head dimension.

scale_factor (`tuple[float, float, float]`, defaults to `(1.0, 2.0, 2.0)`) : Per-axis RoPE frequency scaling applied to the token positions.

nabla_threshold (`float`, defaults to `0.8`) : Cumulative-attention threshold of the NABLA block selection.

nabla_window (`tuple[int, int, int]`, defaults to `(11, 7, 7)`) : Odd `(t, h, w)` extents of the sliding-tile prior that every 8x8 token block always attends to.

tile_sizes (`tuple[tuple[int, int], ...]`, defaults to `((512, 512), (512, 768), (768, 512))`) : Pixel `(height, width)` sizes of the video tiles the model was trained on. [Kandinsky6SRPipeline](/docs/diffusers/v0.42.0/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) refines every tile at the size whose aspect ratio is closest to the input video's.

Text-free diffusion transformer for Kandinsky 6 video super-resolution.

The model denoises tiles of the K-VAE latent video. Its input concatenates the noisy latent with the anchor latent
and the anchor mask that condition the super-resolution (`2 * in_visual_dim + 1` channels), and its output holds
`out_visual_dim` channels: `in_visual_dim` for a plain velocity checkpoint, or a widened `n_grid * in_visual_dim`
for the [PiflowScheduler](/docs/diffusers/v0.42.0/en/api/schedulers/piflow#diffusers.PiflowScheduler) distilled checkpoints. Video self-attention runs through the NABLA sparse block pattern
on the `flex` attention backend, which needs the token grid (`height` and `width` after `patch_size`) to be
divisible by 8.

#### forward[[diffusers.Kandinsky6SRTransformer3DModel.forward]]

```python
forward(hidden_states: Tensor, timestep: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_kandinsky6_sr.py#L496)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, num_frames, height, width, 2 * in_visual_dim + 1)`) : Latent tiles in the `(B, T, H, W, C)` layout: the noisy latent, the anchor latent and the anchor mask concatenated along the channel axis.

timestep (`torch.Tensor` of shape `(batch_size,)`) : Diffusion timestep on the `[0, num_train_timesteps]` scale.

return_dict (`bool`, defaults to `True`) : Whether to return a [Transformer2DModelOutput](/docs/diffusers/v0.42.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) instead of a plain tuple.

**Returns:** [Transformer2DModelOutput](/docs/diffusers/v0.42.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) or `tuple`

The prediction of shape `(batch_size, num_frames, height, width, out_visual_dim)`.

### LatteTransformer3DModel
https://huggingface.co/docs/diffusers/v0.42.0/api/models/latte_transformer3d.md

## LatteTransformer3DModel

A Diffusion Transformer model for 3D data from [Latte](https://github.com/Vchitect/Latte).

## LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

#### diffusers.LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

```python
diffusers.LatteTransformer3DModel(num_attention_heads: int = 16, attention_head_dim: int = 88, in_channels: int | None = None, out_channels: int | None = None, num_layers: int = 1, dropout: float = 0.0, cross_attention_dim: int | None = None, attention_bias: bool = False, sample_size: int = 64, patch_size: int | None = None, activation_fn: str = 'geglu', num_embeds_ada_norm: int | None = None, norm_type: str = 'layer_norm', norm_elementwise_affine: bool = True, norm_eps: float = 1e-05, caption_channels: int = None, video_length: int = 16)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/latte_transformer_3d.py#L27)

#### forward[[diffusers.LatteTransformer3DModel.forward]]

```python
forward(hidden_states: Tensor, timestep: typing.Optional[torch.LongTensor] = None, encoder_hidden_states: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.Tensor] = None, enable_temporal_attentions: bool = True, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/latte_transformer_3d.py#L166)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch size, channel, num_frame, height, width)`) : Input `hidden_states`.

timestep ( `torch.LongTensor`, *optional*) : Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.

encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*) : Conditional embeddings for cross attention layer. If not given, cross-attention defaults to self-attention.

encoder_attention_mask ( `torch.Tensor`, *optional*) : Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:  * Mask `(batcheight, sequence_length)` True = keep, False = discard. * Bias `(batcheight, 1, sequence_length)` 0 = keep, -10000 = discard.  If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format above. This bias will be added to the cross-attention scores.

enable_temporal_attentions : (`bool`, *optional*, defaults to `True`): Whether to enable temporal attentions.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~models.unet_2d_condition.UNet2DConditionOutput` instead of a plain tuple.

**Returns:**

If `return_dict` is True, an `~models.transformer_2d.Transformer2DModelOutput` is returned, otherwise a
`tuple` where the first element is the sample tensor.

The [LatteTransformer3DModel](/docs/diffusers/v0.42.0/en/api/models/latte_transformer3d#diffusers.LatteTransformer3DModel) forward method.

### StableAudio3DiTModel
https://huggingface.co/docs/diffusers/v0.42.0/api/models/stable_audio_3_transformer.md
