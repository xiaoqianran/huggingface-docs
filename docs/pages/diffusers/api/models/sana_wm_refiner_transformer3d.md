# SanaWMLTX2RefinerTransformer3DModel

The chunk-causal autoregressive refiner DiT used as stage 2 of [SanaWMPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/sana_wm#diffusers.SanaWMPipeline), driven by
[SanaWMLTX2Refiner](/docs/diffusers/v0.42.1/en/api/pipelines/sana_wm#diffusers.SanaWMLTX2Refiner).

It is architecturally identical to [LTX2VideoTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/ltx2_video_transformer3d#diffusers.LTX2VideoTransformer3DModel) — same config arguments, same submodules, same
parameter names — so a released LTX-2 checkpoint loads into it unchanged. The forward pass differs:

* only the video stream is run (the audio and audio/video cross-attention branches are skipped),
* self-attention runs against an explicit sliding-window KV cache (`SanaWMRefinerKVCache`) holding the attention
  sink plus the recent refined history, so per-block compute is bounded and total refinement cost scales linearly
  with video length,
* the caller supplies the video RoPE, which lets each autoregressive window keep every frame's absolute index in the
  source video (see
  [SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions()](/docs/diffusers/v0.42.1/en/api/models/sana_wm_refiner_transformer3d#diffusers.SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions)).

The model can be loaded with:

```python
import torch
from diffusers import SanaWMLTX2RefinerTransformer3DModel

transformer = SanaWMLTX2RefinerTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
    subfolder="refiner/transformer",
    torch_dtype=torch.bfloat16,
)
```

## SanaWMLTX2RefinerTransformer3DModel[[diffusers.SanaWMLTX2RefinerTransformer3DModel]]

#### diffusers.SanaWMLTX2RefinerTransformer3DModel[[diffusers.SanaWMLTX2RefinerTransformer3DModel]]

```python
diffusers.SanaWMLTX2RefinerTransformer3DModel(in_channels: int = 128, out_channels: int | None = 128, patch_size: int = 1, patch_size_t: int = 1, num_attention_heads: int = 32, attention_head_dim: int = 128, cross_attention_dim: int = 4096, vae_scale_factors: tuple = (8, 32, 32), pos_embed_max_pos: int = 20, base_height: int = 2048, base_width: int = 2048, gated_attn: bool = False, cross_attn_mod: bool = False, audio_in_channels: int = 128, audio_out_channels: int | None = 128, audio_patch_size: int = 1, audio_patch_size_t: int = 1, audio_num_attention_heads: int = 32, audio_attention_head_dim: int = 64, audio_cross_attention_dim: int = 2048, audio_scale_factor: int = 4, audio_pos_embed_max_pos: int = 20, audio_sampling_rate: int = 16000, audio_hop_length: int = 160, audio_gated_attn: bool = False, audio_cross_attn_mod: bool = False, num_layers: int = 48, activation_fn: str = 'gelu-approximate', qk_norm: str = 'rms_norm_across_heads', norm_elementwise_affine: bool = False, norm_eps: float = 1e-06, caption_channels: int = 3840, attention_bias: bool = True, attention_out_bias: bool = True, rope_theta: float = 10000.0, rope_double_precision: bool = True, causal_offset: int = 1, timestep_scale_multiplier: int = 1000, cross_attn_timestep_scale_multiplier: int = 1000, rope_type: str = 'interleaved', use_prompt_embeddings = True, perturbed_attn: bool = False, ff_bias: bool = True, audio_ff_bias: bool = True, use_prompt_adaln_single: bool = True, use_keyframes_abs_pos_embedding: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L1122)

**Parameters:**

in_channels (`int`, defaults to `128`) : The number of channels in the input.

out_channels (`int`, defaults to `128`) : The number of channels in the output.

patch_size (`int`, defaults to `1`) : The size of the spatial patches to use in the patch embedding layer.

patch_size_t (`int`, defaults to `1`) : The size of the temporal patches to use in the patch embedding layer.

num_attention_heads (`int`, defaults to `32`) : The number of heads to use for multi-head attention.

attention_head_dim (`int`, defaults to `128`) : The number of channels in each head.

cross_attention_dim (`int`, defaults to `4096`) : The number of channels for cross attention heads.

num_layers (`int`, defaults to `48`) : The number of layers of Transformer blocks to use.

activation_fn (`str`, defaults to `"gelu-approximate"`) : Activation function to use in feed-forward.

qk_norm (`str`, defaults to `"rms_norm_across_heads"`) : The normalization layer to use.

rope_type (`str`, defaults to `"interleaved"`) : Which RoPE application to use (`"interleaved"` or `"split"`).

The chunk-causal autoregressive refiner transformer used as SANA-WM stage 2.

Architecturally identical to [LTX2VideoTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/ltx2_video_transformer3d#diffusers.LTX2VideoTransformer3DModel) — same config arguments, same submodules, same
parameter names — so a released LTX-2 checkpoint loads into it unchanged. What differs is the forward pass:

* only the video stream is run (the audio and audio/video cross-attention branches are skipped),
* self-attention runs against an explicit sliding-window KV cache (`SanaWMRefinerKVCache`) holding the attention
  sink plus recent refined history, so refinement cost is bounded per AR block and scales linearly with video
  length,
* the caller supplies the video RoPE, which lets each AR window keep every frame's absolute index in the source
  video (see [build_rotary_emb_for_absolute_positions()](/docs/diffusers/v0.42.1/en/api/models/sana_wm_refiner_transformer3d#diffusers.SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions)).

The remaining arguments mirror [LTX2VideoTransformer3DModel](/docs/diffusers/v0.42.1/en/api/models/ltx2_video_transformer3d#diffusers.LTX2VideoTransformer3DModel) one-for-one. The audio-side arguments and submodules
are kept purely so the checkpoint's audio weights round-trip; they are not used by the refiner forward.

#### build_rotary_emb_for_absolute_positions[[diffusers.SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions]]

```python
build_rotary_emb_for_absolute_positions(batch_size: int, frame_positions: list, height: int, width: int, device: device, fps: float)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L1388)

**Parameters:**

batch_size (`int`) : Batch size to broadcast the coordinates to.

frame_positions (`list[int]`) : Absolute latent-frame indices covered by this window.

height (`int`), width (`int`) : Latent spatial resolution.

device (`torch.device`) : Device to build the coordinates on.

fps (`float`) : Video frame rate, which drives LTX-2's temporal RoPE scaling.

**Returns:** `tuple[torch.Tensor, torch.Tensor]`

the `(cos, sin)` RoPE tuple.

Build the video RoPE for an explicit list of absolute latent-frame indices.

`SanaWMLTX2AudioVideoRotaryPosEmbed.prepare_video_coords` assumes a contiguous `torch.arange(num_frames)`,
which is fine for bidirectional inference. The sliding-window AR refiner instead needs to keep each frame's
absolute index in the source video, so RoPE captures the correct temporal phase across the `[sink + recent +
active]` window.

#### forward[[diffusers.SanaWMLTX2RefinerTransformer3DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: Tensor, video_rotary_emb: tuple, encoder_attention_mask: typing.Optional[torch.Tensor] = None, kv_cache: diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVCache | None = None, kv_cache_mode: str | None = None, attention_kwargs: dict | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L1445)

**Parameters:**

hidden_states (`torch.Tensor`) : Patchified video latents of the active block, of shape `(batch_size, num_video_tokens, in_channels)`.

encoder_hidden_states (`torch.Tensor`) : Text embeddings of shape `(batch_size, text_seq_len, caption_channels)`.

timestep (`torch.Tensor`) : Timestep of shape `(batch_size, num_video_tokens)`, already scaled by `self.config.timestep_scale_multiplier`.

video_rotary_emb (`tuple[torch.Tensor, torch.Tensor]`) : The `(cos, sin)` RoPE for the active block's absolute frame positions, as returned by [build_rotary_emb_for_absolute_positions()](/docs/diffusers/v0.42.1/en/api/models/sana_wm_refiner_transformer3d#diffusers.SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions).

encoder_attention_mask (`torch.Tensor`, *optional*) : Multiplicative text attention mask of shape `(batch_size, text_seq_len)`.

kv_cache (`SanaWMRefinerKVCache`, *optional*) : Sliding-window KV cache holding the per-layer attention sink and recent refined history, plus the shared `sink_pe` RoPE. Required whenever `kv_cache_mode` is set.

kv_cache_mode (`str`, *optional*) : One of:  - `"inject"`: attend to `[sink + history + current]` K/V (the denoising steps). - `"capture_pre_rope"`: no prefix; record the pre-RoPE K/V of this forward into the cache (used once to seed the attention sink from the raw stage-1 latents). - `"inject_and_capture_post_rope"`: attend to `[sink + history + current]` K/V and record this block's post-RoPE K/V into the cache so it can be appended to the history.  When `None`, the block runs plain full self-attention over the current tokens only.

attention_kwargs (`dict`, *optional*) : Optional kwargs forwarded to the LoRA scale handling.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [Transformer2DModelOutput](/docs/diffusers/v0.42.1/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) instead of a plain tuple.

**Returns:** [Transformer2DModelOutput](/docs/diffusers/v0.42.1/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) or `tuple`

the predicted velocity for the active
block, of shape `(batch_size, num_video_tokens, out_channels)`.

Video-only forward pass over a single AR block.

## SanaWMRefinerKVCache[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVCache]]

#### diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVCache[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVCache]]

```python
diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVCache(num_layers: int)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L757)

**Parameters:**

num_layers (*int*) : Number of transformer blocks to allocate a per-layer cache for.

Container holding one [*SanaWMRefinerKVLayerCache*] per transformer block, plus the shared sink RoPE.

This implements the `rf_shifted_sink` KV-cache contract the SANA-WM stage-2 refiner was trained with. Refinement
is chunk-causal: *block_size* latent frames are denoised at a time while attending to a bounded window of
*[attention sink + recent history + active block]* K/V.

- `sink_pe`: the *(cos, sin)* RoPE tuple for the sink frames, rebuilt per AR window at the sliding
  `sink_rope_offset` so the sink sits immediately before the bounded working cache. Shared across layers because
  RoPE does not depend on the layer.

## SanaWMRefinerKVLayerCache[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache]]

#### diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache]]

```python
diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L668)

Per-layer KV cache for the SANA-WM stage-2 chunk-causal AR refiner.

Holds the two halves of the sliding-window prefix that the refiner's self-attention attends to, plus a slot for
reading back the K/V that the last forward captured. All tensors are *(batch_size, num_tokens, inner_dim)* (i.e.
before the per-head unflatten), matching the layout the refiner's self-attention concatenates in.

- `sink_k_pre` / `sink_v`: **pre**-RoPE K/V of the attention-sink frames, captured once from the raw stage-1
  latents. They are stored pre-RoPE so each AR window can re-apply RoPE at its own shifted sink offset
  (`SanaWMRefinerKVCache.sink_pe`).
- `history_k` / `history_v`: **post**-RoPE K/V of the already refined recent frames, ready to be concatenated
  as-is.
- `captured_k_pre` / `captured_v_pre` and `captured_k_post` / `captured_v_post`: readback slots written by
  the capture `kv_cache_mode`s.

#### get_captured_post_rope[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.get_captured_post_rope]]

```python
get_captured_post_rope()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L736)

Pop the post-RoPE K/V captured by the last forward.

#### get_captured_pre_rope[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.get_captured_pre_rope]]

```python
get_captured_pre_rope()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L722)

Pop the pre-RoPE K/V captured by the last forward.

#### get_history[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.get_history]]

```python
get_history()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L711)

Return the post-RoPE recent-history K/V, or `None` if empty.

#### get_sink[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.get_sink]]

```python
get_sink()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L700)

Return the pre-RoPE sink K/V, or `None` if it has not been captured (or is empty).

#### store_captured_post_rope[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.store_captured_post_rope]]

```python
store_captured_post_rope(key: Tensor, value: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L731)

Store the post-RoPE K/V produced by the current forward.

#### store_captured_pre_rope[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.store_captured_pre_rope]]

```python
store_captured_pre_rope(key: Tensor, value: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L717)

Store the pre-RoPE K/V produced by the current forward.

#### store_history[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.store_history]]

```python
store_history(history_k: Tensor, history_v: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L706)

Store the post-RoPE recent-history K/V.

#### store_sink[[diffusers.models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache.store_sink]]

```python
store_sink(sink_k_pre: Tensor, sink_v: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_wm_refiner.py#L695)

Store the pre-RoPE sink K/V.

## Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

#### diffusers.models.modeling_outputs.Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

```python
diffusers.models.modeling_outputs.Transformer2DModelOutput(sample: torch.Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/modeling_outputs.py#L21)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/transformer2d#diffusers.Transformer2DModel) is discrete) : The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability distributions for the unnoised latent pixels.

The output of [Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/transformer2d#diffusers.Transformer2DModel).

### WanTransformer3DModel
https://huggingface.co/docs/diffusers/v0.42.1/api/models/wan_transformer_3d.md
