# CogView3PlusTransformer2DModel

A Diffusion Transformer model for 2D data from [CogView3Plus](https://github.com/THUDM/CogView3) was introduced in [CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion](https://huggingface.co/papers/2403.05121) by Tsinghua University & ZhipuAI.

The model can be loaded with the following code snippet.

```python
from diffusers import CogView3PlusTransformer2DModel

transformer = CogView3PlusTransformer2DModel.from_pretrained("THUDM/CogView3Plus-3b", subfolder="transformer", dtype=torch.bfloat16).to("cuda")  # or "mps", "xpu", "cpu"
```

## CogView3PlusTransformer2DModel[[diffusers.CogView3PlusTransformer2DModel]]

#### diffusers.CogView3PlusTransformer2DModel[[diffusers.CogView3PlusTransformer2DModel]]

```python
diffusers.CogView3PlusTransformer2DModel(patch_size: int = 2, in_channels: int = 16, num_layers: int = 30, attention_head_dim: int = 40, num_attention_heads: int = 64, out_channels: int = 16, text_embed_dim: int = 4096, time_embed_dim: int = 512, condition_dim: int = 256, pos_embed_max_size: int = 128, sample_size: int = 128)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_cogview3plus.py#L126)

**Parameters:**

patch_size (`int`, defaults to `2`) : The size of the patches to use in the patch embedding layer.

in_channels (`int`, defaults to `16`) : The number of channels in the input.

num_layers (`int`, defaults to `30`) : The number of layers of Transformer blocks to use.

attention_head_dim (`int`, defaults to `40`) : The number of channels in each head.

num_attention_heads (`int`, defaults to `64`) : The number of heads to use for multi-head attention.

out_channels (`int`, defaults to `16`) : The number of channels in the output.

text_embed_dim (`int`, defaults to `4096`) : Input dimension of text embeddings from the text encoder.

time_embed_dim (`int`, defaults to `512`) : Output dimension of timestep embeddings.

condition_dim (`int`, defaults to `256`) : The embedding dimension of the input SDXL-style resolution conditions (original_size, target_size, crop_coords).

pos_embed_max_size (`int`, defaults to `128`) : The maximum resolution of the positional embeddings, from which slices of shape `H x W` are taken and added to input patched latents, where `H` and `W` are the latent height and width respectively. A value of 128 means that the maximum supported height and width for image generation is `128 * vae_scale_factor * patch_size => 128 * 8 * 2 => 2048`.

sample_size (`int`, defaults to `128`) : The base resolution of input latents. If height/width is not provided during generation, this value is used to determine the resolution as `sample_size * vae_scale_factor => 128 * 8 => 1024`

The Transformer model introduced in [CogView3: Finer and Faster Text-to-Image Generation via Relay
Diffusion](https://huggingface.co/papers/2403.05121).

#### forward[[diffusers.CogView3PlusTransformer2DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: LongTensor, original_size: Tensor, target_size: Tensor, crop_coords: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_cogview3plus.py#L225)

**Parameters:**

hidden_states (`torch.Tensor`) : Input `hidden_states` of shape `(batch size, channel, height, width)`.

encoder_hidden_states (`torch.Tensor`) : Conditional embeddings (embeddings computed from the input conditions such as prompts) of shape `(batch_size, sequence_len, text_embed_dim)`

timestep (`torch.LongTensor`) : Used to indicate denoising step.

original_size (`torch.Tensor`) : CogView3 uses SDXL-like micro-conditioning for original image size as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

target_size (`torch.Tensor`) : CogView3 uses SDXL-like micro-conditioning for target image size as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

crop_coords (`torch.Tensor`) : CogView3 uses SDXL-like micro-conditioning for crop coordinates as explained in section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~models.transformer_2d.Transformer2DModelOutput` instead of a plain tuple.

**Returns:** `torch.Tensor` or `~models.transformer_2d.Transformer2DModelOutput`

The denoised latents using provided inputs as conditioning.

The [CogView3PlusTransformer2DModel](/docs/diffusers/v0.43.0/en/api/models/cogview3plus_transformer2d#diffusers.CogView3PlusTransformer2DModel) forward method.

## Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

#### diffusers.models.modeling_outputs.Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

```python
diffusers.models.modeling_outputs.Transformer2DModelOutput(sample: torch.Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/modeling_outputs.py#L21)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [Transformer2DModel](/docs/diffusers/v0.43.0/en/api/models/transformer2d#diffusers.Transformer2DModel) is discrete) : The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability distributions for the unnoised latent pixels.

The output of [Transformer2DModel](/docs/diffusers/v0.43.0/en/api/models/transformer2d#diffusers.Transformer2DModel).

### LatteTransformer3DModel
https://huggingface.co/docs/diffusers/v0.43.0/api/models/latte_transformer3d.md

## LatteTransformer3DModel

A Diffusion Transformer model for 3D data from [Latte](https://github.com/Vchitect/Latte).

## LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

#### diffusers.LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

```python
diffusers.LatteTransformer3DModel(num_attention_heads: int = 16, attention_head_dim: int = 88, in_channels: int | None = None, out_channels: int | None = None, num_layers: int = 1, dropout: float = 0.0, cross_attention_dim: int | None = None, attention_bias: bool = False, sample_size: int = 64, patch_size: int | None = None, activation_fn: str = 'geglu', num_embeds_ada_norm: int | None = None, norm_type: str = 'layer_norm', norm_elementwise_affine: bool = True, norm_eps: float = 1e-05, caption_channels: int = None, video_length: int = 16)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/latte_transformer_3d.py#L27)

#### forward[[diffusers.LatteTransformer3DModel.forward]]

```python
forward(hidden_states: Tensor, timestep: typing.Optional[torch.LongTensor] = None, encoder_hidden_states: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.Tensor] = None, enable_temporal_attentions: bool = True, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/latte_transformer_3d.py#L166)

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

The [LatteTransformer3DModel](/docs/diffusers/v0.43.0/en/api/models/latte_transformer3d#diffusers.LatteTransformer3DModel) forward method.

### BriaTransformer2DModel
https://huggingface.co/docs/diffusers/v0.43.0/api/models/bria_transformer.md
