# PAN2Transformer3DModel

The diffusion transformer of PAN 2. It runs joint attention over the video tokens and the text tokens, with separate projections and feed-forward networks for each stream.

The video input stacks the noisy latents, the conditioning latents and a one-channel conditioning mask, so the same model serves text-to-video and image-to-video.

```python
import torch
from diffusers import PAN2Transformer3DModel

transformer = PAN2Transformer3DModel.from_pretrained("IFM/PAN2", subfolder="transformer", dtype=torch.bfloat16)
```

## PAN2Transformer3DModel[[diffusers.PAN2Transformer3DModel]]

#### diffusers.PAN2Transformer3DModel[[diffusers.PAN2Transformer3DModel]]

```python
diffusers.PAN2Transformer3DModel(patch_size: tuple[int, int, int] = (1, 2, 2), in_channels: int = 97, out_channels: int = 48, num_attention_heads: int = 32, attention_head_dim: int = 128, num_layers: int = 44, num_refiner_layers: int = 4, mlp_ratio: float = 8.0, context_mlp_ratio: float = 4.0, refiner_mlp_ratio: float = 4.0, text_embed_dim: int = 4096, rope_axes_dim: tuple[int, int, int] = (16, 56, 56), rope_theta: float = 256.0, rope_temporal_scale: float = 0.6666666666666666)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_pan2.py#L325)

**Parameters:**

patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`) : The `(t, h, w)` patch size of the patch embedding.

in_channels (`int`, defaults to `97`) : The number of input channels: noisy latents, conditioning latents and one conditioning mask channel.

out_channels (`int`, defaults to `48`) : The number of output channels.

num_attention_heads (`int`, defaults to `32`) : The number of attention heads.

attention_head_dim (`int`, defaults to `128`) : The number of channels in each attention head.

num_layers (`int`, defaults to `44`) : The number of joint-attention blocks.

num_refiner_layers (`int`, defaults to `4`) : The number of token refiner blocks applied to the text embeddings.

mlp_ratio (`float`, defaults to `8.0`) : The ratio of the video feed-forward inner dimension to the hidden dimension.

context_mlp_ratio (`float`, defaults to `4.0`) : The ratio of the text feed-forward inner dimension to the hidden dimension.

refiner_mlp_ratio (`float`, defaults to `4.0`) : The ratio of the token refiner feed-forward inner dimension to the hidden dimension.

text_embed_dim (`int`, defaults to `4096`) : The dimension of the text encoder hidden states.

rope_axes_dim (`tuple[int, int, int]`, defaults to `(16, 56, 56)`) : The number of channels of each `(t, h, w)` axis in the rotary embedding.

rope_theta (`float`, defaults to `256.0`) : The base of the rotary embedding.

rope_temporal_scale (`float`, defaults to `16 / 24`) : The factor the frame positions of the rotary embedding are multiplied with: the reference frame rate of the rotary embedding over the frame rate of the videos (16 over 24 fps).

The diffusion transformer of PAN 2.

A stack of joint-attention blocks over the concatenation of the patchified video latents and the refined text
tokens. The video input carries the noisy latents, the conditioning latents and a conditioning mask stacked along
the channel axis, so the same model serves text-to-video and image-to-video.

#### forward[[diffusers.PAN2Transformer3DModel.forward]]

```python
forward(hidden_states: Tensor, timestep: Tensor, encoder_hidden_states: Tensor, attention_kwargs: dict[str, typing.Any] | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_pan2.py#L438)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames, height, width)`) : The noisy latents, the conditioning latents and the conditioning mask, concatenated along channels.

timestep (`torch.Tensor` of shape `(batch_size,)`) : The denoising timestep in `[0, 1000]`.

encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, text_embed_dim)`) : The text encoder hidden states.

attention_kwargs (`dict`, *optional*) : A kwargs dictionary passed along to the `AttentionProcessor`; only used for the LoRA scale.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [Transformer2DModelOutput](/docs/diffusers/v0.43.0/en/api/models/cogview3plus_transformer2d#diffusers.models.modeling_outputs.Transformer2DModelOutput) instead of a plain tuple.

**Returns:** [Transformer2DModelOutput](/docs/diffusers/v0.43.0/en/api/models/cogview3plus_transformer2d#diffusers.models.modeling_outputs.Transformer2DModelOutput) or `tuple`

The predicted velocity of shape `(batch_size, out_channels, num_frames, height, width)`.

The [PAN2Transformer3DModel](/docs/diffusers/v0.43.0/en/api/models/pan2_transformer3d#diffusers.PAN2Transformer3DModel) forward method.

### AutoencoderSAME
https://huggingface.co/docs/diffusers/v0.43.0/api/models/autoencoder_same.md
