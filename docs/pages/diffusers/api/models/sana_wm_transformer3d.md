# SanaWMTransformer3DModel

A 3D Diffusion Transformer (1.6B parameters) for camera-controlled image-to-video generation, used as the stage-1
sampler of [SanaWMPipeline](/docs/diffusers/v0.42.0/en/api/pipelines/sana_wm#diffusers.SanaWMPipeline). The transformer combines:

* a bidirectional GDN-Triton linear-attention main branch (depth 20, hidden 2240, 20 heads),
* a UCPE (Unified Camera Pose Embedding) camera-control branch that consumes a raymap + Plücker representation of
  the requested trajectory, and
* a Wan-style 3D rotary position embedding plus periodic softmax-attention blocks injected every `softmax_every_n`
  layers.

The model can be loaded with:

```python
import torch
from diffusers import SanaWMTransformer3DModel

transformer = SanaWMTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```

## SanaWMTransformer3DModel[[diffusers.SanaWMTransformer3DModel]]

#### diffusers.SanaWMTransformer3DModel[[diffusers.SanaWMTransformer3DModel]]

```python
diffusers.SanaWMTransformer3DModel(in_channels: int = 128, num_layers: int = 20, hidden_size: int = 2240, num_attention_heads: int = 20, patch_size: tuple[int, int, int] = (1, 1, 1), softmax_every_n: int = 4, linear_head_dim: int = 112, ffn_type: str = 'GLUMBConvTemp', t_kernel_size: int = 3, conv_kernel_size: int = 4, k_conv_only: bool = True, pos_embed_type: str = 'wan_rope', qk_norm: bool = True, cross_norm: bool = True, y_norm: bool = True, cam_attn_compress: int = 1, init_cam_from_base: bool = True, use_chunk_plucker_post_attn: bool = True, chunk_plucker_channels: int = 48, chunk_plucker_post_attn_blocks: int = 20, fp32_attention: bool = True, image_size: int = 720, caption_channels: int = 2304, mlp_ratio: float = 3.0, mlp_acts: tuple = ('silu', 'silu', None), use_pe: bool = True, learn_sigma: bool = False, pred_sigma: bool = False, mixed_precision: str = 'bf16')
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_sana_wm.py#L2663)

**Parameters:**

in_channels (*int*, defaults to 128) : VAE latent channels (LTX-2).

softmax_every_n (*int*, defaults to 4) : Use a softmax attention block every N blocks.

linear_head_dim (*int*, defaults to 112) : GDN head dimension.

ffn_type (*str*, defaults to `"GLUMBConvTemp"`) : FFN.

t_kernel_size (*int*, defaults to 3) : Temporal conv kernel.

conv_kernel_size (*int*, defaults to 4) : Spatial conv kernel inside attention.

k_conv_only (*bool*, defaults to True) : Apply conv only on K.

pos_embed_type (*str*, defaults to `"wan_rope"`) : Position embedding.

qk_norm (*bool*, defaults to True) : RMSNorm on Q/K.

cross_norm (*bool*, defaults to True) : RMSNorm on cross-attention K.

y_norm (*bool*, defaults to True) : Apply `attention_y_norm` to text embeddings.

init_cam_from_base (*bool*, defaults to True) : Unused; the camera branch is loaded from the checkpoint. Kept so released *config.json* files load.

use_chunk_plucker_post_attn (*bool*, defaults to True). --

chunk_plucker_channels (*int*, defaults to 48) : `6 dims * temporal_stride 8`.

chunk_plucker_post_attn_blocks (*int*, defaults to 20) : All blocks.

fp32_attention (*bool*, defaults to True) : Unused; attention always runs in fp32. Kept so released *config.json* files load.

image_size (*int*, defaults to 720) : Nominal image size.

caption_channels (*int*, defaults to 2304) : Gemma-2 hidden size.

SANA-WM 1600M bidirectional camera-controlled DiT.

A single-class DiT (depth=20, hidden_size=2240, patch_size=(1,1,1), num_heads=20 — i.e. the public
`Efficient-Large-Model/SANA-WM_bidirectional` release). `save_pretrained` / `from_pretrained` work out of the
box via [ConfigMixin](/docs/diffusers/v0.42.0/en/api/configuration#diffusers.ConfigMixin).

Every block runs a camera-conditioned (UCPE) attention: [*BidirectionalGDNUCPESinglePathLiteLA*], except every
`softmax_every_n`-th block which runs its softmax counterpart [*_SoftmaxUCPESinglePathLiteLA*].

The state-dict is identical to the public sana checkpoint apart from the intentionally-removed `pos_embed`
buffer.

#### forward[[diffusers.SanaWMTransformer3DModel.forward]]

```python
forward(hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor | None = None, mask: torch.Tensor | None = None, return_dict: bool = True, data_info: Optional[dict] = None, camera_conditions: torch.Tensor | None = None, chunk_plucker: torch.Tensor | None = None, cam_pos_embeds: Optional[dict] = None, pos_embeds: torch.Tensor | None = None, raymats: torch.Tensor | None = None, frame_valid_mask: torch.Tensor | None = None, chunk_size: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_sana_wm.py#L2896)

**Parameters:**

hidden_states : `(B, C, T, H, W)` latents.

timestep : `(B, 1, T)` per-frame diffusion timesteps (LTX style).

encoder_hidden_states : `(B, 1, L, D_caption)` text embeddings.

encoder_attention_mask : `(B, L)` text attention mask (diffusers convention).

mask : Alias for `encoder_attention_mask` matching the sana DiT's kwarg name. If both are passed, `mask` takes precedence.

return_dict : If `True` (default), returns a `Transformer2DModelOutput`; otherwise returns a one-tuple `(sample,)`.

data_info : Extra conditioning; `data_info["image_vae_embeds"]` is concatenated to the latents along the channel axis when present.

camera_conditions : `(B, T, 20)` raw camera conditions driving the camera-control (UCPE) branch.

chunk_plucker : Plucker ray embeddings `(B, C, T, H, W)`, consumed when the model is configured with `use_chunk_plucker_input` / `use_chunk_plucker_post_attn`.

cam_pos_embeds : Optional pre-computed camera positional embeddings (`"absmap"` / `"P"` entries) reused instead of recomputing them.

pos_embeds : Optional pre-computed rotary position embeddings; when `None` they are built from the latent shape.

raymats : Optional pre-computed UCPE ray matrices (used only when `cam_pos_embeds` does not carry `"P"`).

frame_valid_mask : Optional per-frame validity mask, shaped `(B, 1, T, 1, 1)`, `(B, 1, T)` or `(B, T)`.

chunk_size : Chunk length override forwarded to the blocks; falls back to each block's configured `chunk_size`.

**Returns:**

`Transformer2DModelOutput` with `sample` of shape `(B, C, T, H, W)`.

Run the SANA-WM DiT.

#### unpatchify[[diffusers.SanaWMTransformer3DModel.unpatchify]]

```python
unpatchify(x, post_patch_num_frames, post_patch_height, post_patch_width)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/transformers/transformer_sana_wm.py#L3099)

x: (N, T, patch_size**2 * C) imgs: (N, H, W, C)

## Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

#### diffusers.models.modeling_outputs.Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

```python
diffusers.models.modeling_outputs.Transformer2DModelOutput(sample: torch.Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/modeling_outputs.py#L21)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [Transformer2DModel](/docs/diffusers/v0.42.0/en/api/models/transformer2d#diffusers.Transformer2DModel) is discrete) : The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability distributions for the unnoised latent pixels.

The output of [Transformer2DModel](/docs/diffusers/v0.42.0/en/api/models/transformer2d#diffusers.Transformer2DModel).

### LongCatImageTransformer2DModel
https://huggingface.co/docs/diffusers/v0.42.0/api/models/longcat_image_transformer2d.md
