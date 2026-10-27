# BriaFibo2Transformer2DModel

The single-stream transformer of [Bria](https://huggingface.co/briaai)'s Fibo 2.0, built on the Z-Image block. Its text conditioning comes from Qwen3-VL in two ways: a Perceiver resampler turns the text encoder's hidden states into gist tokens that share the stream with the image tokens, and five blocks add a gated cross-attention to their own bundle of Qwen3-VL layers. To edit, the latents of the images to edit join the stream as clean tokens, each image on its own RoPE plane.

## BriaFibo2Transformer2DModel[[diffusers.BriaFibo2Transformer2DModel]]

#### diffusers.BriaFibo2Transformer2DModel[[diffusers.BriaFibo2Transformer2DModel]]

```python
diffusers.BriaFibo2Transformer2DModel(in_channels: int = 32, patch_size: int = 2, dim: int = 3584, n_layers: int = 44, n_refiner_layers: int = 2, n_heads: int = 28, norm_eps: float = 1e-05, cap_feat_dim: int = 7680, injection_layer_ids: tuple = (6, 14, 22, 30, 38), perceiver_num_layers: int = 2, min_num_gist_tokens: int = 256, max_num_gist_tokens: int = 512, gist_step: int = 64, gist_min_text_len: int = 256, gist_max_text_len: int = 2048, rope_theta: float = 256.0, t_scale: float = 1000.0, axes_dims: tuple = (32, 48, 48), axes_lens: tuple = (2048, 512, 512))
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_bria_fibo2.py#L430)

#### forward[[diffusers.BriaFibo2Transformer2DModel.forward]]

```python
forward(hidden_states: Tensor, timestep: Tensor, encoder_hidden_states: Tensor, encoder_attention_mask: typing.Optional[torch.Tensor] = None, context_latents: list[torch.Tensor] | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_bria_fibo2.py#L500)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, height, width)`) : Normalized VAE latents of the image being generated.

timestep (`torch.Tensor` of shape `(batch_size,)`) : Noise level in `[0, 1]`: 1 is pure noise, 0 is a clean image.

encoder_hidden_states (`torch.Tensor` of shape `(batch_size, 1 + len(injection_layer_ids), text_len, cap_feat_dim)`) : Text encoder features: the Perceiver's bundle first, then one bundle per injection block.

encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_len)`, *optional*) : Bool mask of the real text tokens, for a batch of prompts padded to the same length. Leave it as `None` when nothing is padded.

context_latents (`list[torch.Tensor]`, *optional*) : Normalized VAE latents of the images to edit, one tensor of shape `(batch_size, in_channels, height, width)` per image, in order. Each image can have its own size. Their tokens condition the generation as clean tokens and are not predicted.

return_dict (`bool`, defaults to `True`) : Whether to return a [Transformer2DModelOutput](/docs/diffusers/v0.42.1/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) instead of a tuple.

**Returns:** [Transformer2DModelOutput](/docs/diffusers/v0.42.1/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) or `tuple`

The predicted velocity, with the shape of
`hidden_states`. If `return_dict` is `False`, a tuple whose first element is that tensor.

### SanaWMTransformer3DModel
https://huggingface.co/docs/diffusers/v0.42.1/api/models/sana_wm_transformer3d.md
