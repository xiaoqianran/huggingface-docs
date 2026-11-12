# MiniMaxH3ImageTransformer2DModel

A Diffusion Transformer model for image generation and editing, introduced in [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by MiniMax.

It is the [MiniMaxH3Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/minimax_h3_transformer3d#diffusers.MiniMaxH3Transformer3DModel) without its audio stream: a single stack of blocks runs over **one packed 1-D sequence** that holds the text conditioning, the rows of the reference images and the rows of the generated image. Building the packed layout is the caller's job, which is why the forward signature takes the layout apart from the latents: the `(t, h, w)` position grid, the per-row modality tags, the per-row timestep indices and the two index tensors that address the image and text rows. [MiniMaxH3ImageBlocks](/docs/diffusers/v0.43.0/en/api/pipelines/minimax_h3_image#diffusers.MiniMaxH3ImageBlocks) builds all of it.

The checkpoint is distilled to six sampling steps with Parallel Decoding Distillation (PDD), which keeps one output head per step. `forward` therefore takes `pdd_step_index`, the step it is running, and projects with that step's head.

```python
import torch
from diffusers import MiniMaxH3ImageTransformer2DModel

transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3-Image", subfolder="transformer_diffusers", dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"
```

The checkpoint is mixed precision: the input patch projection, the timestep MLP and the output heads are float32 while the block stack is bfloat16. `from_pretrained` keeps that layout through `_keep_in_fp32_modules`, so pass `dtype=torch.bfloat16` and let it place the float32 modules rather than casting the model with `.to(torch.bfloat16)` afterwards.

## MiniMaxH3ImageTransformer2DModel[[diffusers.MiniMaxH3ImageTransformer2DModel]]

#### diffusers.MiniMaxH3ImageTransformer2DModel[[diffusers.MiniMaxH3ImageTransformer2DModel]]

```python
diffusers.MiniMaxH3ImageTransformer2DModel(num_attention_heads: int = 56, attention_head_dim: int = 128, hidden_size: int = 5376, num_layers: int = 50, num_refiner_layers: int = 2, ffn_dim: int = 14336, in_channels: int = 24, patch_size: tuple[int, int, int] = (1, 2, 2), text_dim: int = 5120, freq_dim: int = 256, time_embed_hidden_dim: int = 5376, time_embed_dim: int = 2688, rope_freq_dim: int = 16, rope_theta: float = 10000.0, norm_eps: float = 1e-05, qk_norm_eps: float = 1e-05, final_norm_eps: float = 1e-05, num_pdd_heads: int = 6)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_minimax_h3_image.py#L365)

**Parameters:**

num_attention_heads (`int`, defaults to `56`) : The number of heads to use for multi-head attention.

attention_head_dim (`int`, defaults to `128`) : The number of channels in each attention head. Note that `num_attention_heads * attention_head_dim` is *larger* than `hidden_size`.

hidden_size (`int`, defaults to `5376`) : The number of channels of the packed sequence (the residual stream).

num_layers (`int`, defaults to `50`) : The number of transformer blocks.

num_refiner_layers (`int`, defaults to `2`) : The number of token refiner blocks applied to the projected text stream.

ffn_dim (`int`, defaults to `14336`) : The inner dimension of the SwiGLU feed-forward layers.

in_channels (`int`, defaults to `24`) : The number of channels of the image latents.

patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`) : The `(t, h, w)` patch used to pack the latents into rows.

text_dim (`int`, defaults to `5120`) : The number of channels of the text conditioning produced by the text encoder.

freq_dim (`int`, defaults to `256`) : The dimension of the sinusoidal timestep embedding. Timesteps are consumed unscaled in `[0, 1]`.

time_embed_hidden_dim (`int`, defaults to `5376`) : The inner dimension of the timestep MLP.

time_embed_dim (`int`, defaults to `2688`) : The output dimension of the timestep MLP, i.e. the input of every AdaLN projection.

rope_freq_dim (`int`, defaults to `16`) : The number of rotary frequencies per axis. The `(t, h, w)` axes share one `inv_freq` buffer of this length and `2 * 3 * rope_freq_dim` of the `attention_head_dim` channels are rotated.

rope_theta (`float`, defaults to `10000.0`) : The base of the rotary frequency schedule the `rope.inv_freq` buffer is computed from.

norm_eps (`float`, defaults to `1e-5`) : Epsilon of the pre-attention and pre-feed-forward norms.

qk_norm_eps (`float`, defaults to `1e-5`) : Epsilon of the per-head query/key norms.

final_norm_eps (`float`, defaults to `1e-5`) : Epsilon of the token refiner output norm and of `norm_out`.

num_pdd_heads (`int`, defaults to `6`) : The number of sampling steps the checkpoint was distilled to, each with its own output head.

A Transformer model for image generation and editing, introduced in MiniMax-H3-Image.

It is the MiniMax-H3 transformer without its audio stream: a single stack of blocks runs over **one packed 1-D
sequence** that holds the text condition, the rows of the reference images and the rows of the generated image.
Attention is full self-attention over that sequence; there is no cross-attention. The checkpoint is distilled to a
fixed number of sampling steps with Parallel Decoding Distillation (PDD), which keeps one output head per step, so
`forward` takes the index of the step it is running.

The caller is responsible for building the packed layout: patchifying the latents, ordering the rows, and producing
the `(t, h, w)` position grid, the per-row modality tags and the per-row timestep indices. The sequence carries no
padding, so attention runs unmasked over one document and every attention backend stays available.

The batch axis is a pure replication axis: the structural arguments (`timestep`, `timestep_indices`, `token_tags`,
`position_ids` and the two index tensors) describe one packed layout that every batch item shares.

#### forward[[diffusers.MiniMaxH3ImageTransformer2DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: Tensor, timestep_indices: Tensor, token_tags: Tensor, position_ids: Tensor, image_indices: Tensor, text_indices: Tensor, pdd_step_index: int, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_minimax_h3_image.py#L545)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, num_image_tokens, in_channels * prod(patch_size))`) : Patchified latent rows — the rows of the reference images and the rows of the generated image — ordered as they appear in the packed sequence, i.e. matching `image_indices`.

encoder_hidden_states (`torch.Tensor` of shape `(batch_size, num_text_tokens, text_dim)`) : Text conditioning, ordered to match `text_indices`.

timestep (`torch.Tensor` of shape `(num_timesteps,)`) : The *distinct* timestep values present in the packed sequence, in `[0, 1]` and unscaled. One forward serves rows at different noise levels (generated rows, reference rows).

timestep_indices (`torch.Tensor` of shape `(seq_len,)`) : For every row of the packed sequence, the index of its timestep in `timestep`.

token_tags (`torch.Tensor` of shape `(seq_len,)`) : For every row of the packed sequence, its modality: `0` image, `1` text.

position_ids (`torch.Tensor` of shape `(seq_len, 3)`) : The `(t, h, w)` rotary coordinates of every row of the packed sequence.

image_indices (`torch.Tensor` of shape `(num_image_tokens,)`) : Positions of the image rows in the packed sequence.

text_indices (`torch.Tensor` of shape `(num_text_tokens,)`) : Positions of the text rows in the packed sequence.

pdd_step_index (`int`) : The sampling step, which selects the output head.

return_dict (`bool`, defaults to `True`) : Whether to return a [Transformer2DModelOutput](/docs/diffusers/v0.43.0/en/api/models/cogview3plus_transformer2d#diffusers.models.modeling_outputs.Transformer2DModelOutput) instead of a plain tuple.

**Returns:** [Transformer2DModelOutput](/docs/diffusers/v0.43.0/en/api/models/cogview3plus_transformer2d#diffusers.models.modeling_outputs.Transformer2DModelOutput) or `tuple`

The velocity of every image row, of shape `(batch_size, num_image_tokens, in_channels *
prod(patch_size))`, in the row order of `image_indices`.

### WanAnimate2Transformer3DModel
https://huggingface.co/docs/diffusers/v0.43.0/api/models/wan_animate_2_transformer_3d.md
