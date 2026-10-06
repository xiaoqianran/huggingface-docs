# QwenImage21Transformer2DModel

The single-stream transformer used by Qwen-Image 2.1. Text and image latents share one sequence, and a single shared
`modulation` projection feeds every block.

Two behaviours distinguish 2.1 from earlier QwenImage transformers:

- **Block-causal attention** — attention follows `(q_idx >= kv_idx) or same_image_block`, so the joint sequence is
  causal while each image block stays internally bidirectional. `QwenImage21AttnProcessor` implements it as one
  attention call per prefix segment and is the default. `QwenImage21FlexAttnProcessor` implements it as a single
  `flex_attention` call driven by a `BlockMask`, which is faster once the model is compiled. Both produce the same
  results.
- `causal_condition` — text and condition-image tokens are modulated from `t = 0` rather than the sampled timestep.
  Their activations are independent of the denoising step, so the keys and values of that prefix are cacheable
  across steps via the `kv_cache` argument.

Load it with:

```python
import torch
from diffusers import QwenImage21Transformer2DModel

transformer = QwenImage21Transformer2DModel.from_pretrained(
    "Qwen/Qwen-Image-2.1", subfolder="transformer", dtype=torch.bfloat16
)
```

## QwenImage21Transformer2DModel[[diffusers.QwenImage21Transformer2DModel]]

#### diffusers.QwenImage21Transformer2DModel[[diffusers.QwenImage21Transformer2DModel]]

```python
diffusers.QwenImage21Transformer2DModel(patch_size: int = 1, in_channels: int = 64, out_channels: int | None = 64, num_layers: int = 32, attention_head_dim: int = 128, num_attention_heads: int = 32, context_in_dim: int = 4096, mlp_ratio: int = 3, axes_dims_rope: tuple = (16, 56, 56), eps: float = 1e-06, causal_condition: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/transformers/transformer_qwenimage21.py#L713)

**Parameters:**

patch_size (`int`, defaults to `1`) : Side length of the latent patch folded into the channel dim. 2.1 consumes latents unpatched.

in_channels (`int`, defaults to `64`) : Latent channels of the input.

out_channels (`int`, *optional*, defaults to `64`) : Latent channels of the output. Falls back to `in_channels`.

num_layers (`int`, defaults to `32`) : Number of single-stream blocks.

attention_head_dim (`int`, defaults to `128`) : Channels per attention head.

num_attention_heads (`int`, defaults to `32`) : Number of attention heads.

context_in_dim (`int`, defaults to `4096`) : Channel dim of `encoder_hidden_states`.

mlp_ratio (`int`, defaults to `3`) : Feed-forward expansion factor.

axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`) : Rotary dims for the frame, height and width axes.

eps (`float`, defaults to `1e-6`) : Epsilon for the norm layers.

causal_condition (`bool`, defaults to `True`) : Modulate text and condition-image tokens from `t = 0`. Required for KV caching.

The single-stream Transformer used by Qwen-Image 2.1.

Text and image latents share one sequence: condition-image tokens are substituted into the text stream at the
positions the vision-language encoder reserved for them, and the target image's tokens are appended. A single
shared `modulation` projection feeds every block, so blocks hold no modulation parameters of their own.

Two behaviours distinguish 2.1:

- **Block-causal attention** — attention follows `(q_idx >= kv_idx) or same_image_block`, so the sequence is causal
  while each image block stays internally bidirectional. The `flex` attention backend gives efficient single-pass
  attention; without it the model uses an exact multi-pass SDPA prefill that processes each block separately.
- `causal_condition` — text and condition-image tokens are modulated from `t = 0` instead of the sampled timestep,
  which also makes their activations timestep-independent and so cacheable across denoising steps.

#### build_token_metadata[[diffusers.QwenImage21Transformer2DModel.build_token_metadata]]

```python
build_token_metadata(image_pad_mask: Tensor, img_shapes: list)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/transformers/transformer_qwenimage21.py#L807)

**Parameters:**

image_pad_mask (`torch.Tensor`) : `(seq_len,)` bool, `True` at image-token positions.

img_shapes (`list[tuple[int, int, int]]`) : Per-image `(frame, height, width)` in latent tokens, condition images first and the target image last.

**Returns:** `tuple[torch.Tensor, torch.Tensor]`

`image_ids` `(seq_len,)` with `-1` at text positions and a unique id
per image block, and `target_token_mask` `(seq_len,)` marking the target image's tokens.

Label every token of the joint sequence with the image block it belongs to.

Block boundaries come from the token counts in `img_shapes`, not from runs of `True` in `image_pad_mask`: two
condition images that happen to sit next to each other with no text between them form one run but must stay
separate blocks, otherwise they would attend to each other bidirectionally.

#### forward[[diffusers.QwenImage21Transformer2DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: Tensor, img_shapes: list, img_mask: Tensor, encoder_hidden_states_mask: typing.Optional[torch.Tensor] = None, attention_kwargs: dict[str, typing.Any] | None = None, kv_cache: diffusers.models.transformers.transformer_qwenimage21.QwenImage21KVCache | None = None, kv_cache_mode: str | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/transformers/transformer_qwenimage21.py#L846)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`) : Packed latents, condition images first and the target image last.

encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, context_in_dim)`) : Text embeddings from the vision-language encoder.

timestep (`torch.Tensor`) : Current denoising step, scaled to `[0, 1]`.

img_shapes (`list[list[tuple[int, int, int]]]`) : Per-sample list of `(frame, height, width)` in latent tokens, condition images first and the target image last. All samples must share a layout.

img_mask (`torch.Tensor` of shape `(batch_size, vlm_sequence_length)`) : `True` at the vision-language encoder's image slots, each standing for a `2x2` group of latent tokens.

encoder_hidden_states_mask (`torch.Tensor`, *optional*) : `(batch_size, text_sequence_length)` bool marking valid text tokens. Padded positions are excluded from attention.

kv_cache (`QwenImage21KVCache`, *optional*) : Cache container. Pass together with `kv_cache_mode` to enable prefix KV caching.

kv_cache_mode (`str`, *optional*) : `"extract"` to prefill the cache (first denoising step), `"cached"` to decode from it (later steps). Requires `causal_condition=True`.

attention_kwargs (`dict`, *optional*) : Forwarded to the attention processors, and carries `scale` for the LoRA layers.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [Transformer2DModelOutput](/docs/diffusers/v0.41.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) instead of a plain tuple.

**Returns:** [Transformer2DModelOutput](/docs/diffusers/v0.41.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) or `tuple`

[Transformer2DModelOutput](/docs/diffusers/v0.41.0/en/api/models/hunyuan_video15_transformer_3d#diffusers.models.modeling_outputs.Transformer2DModelOutput) if `return_dict` is True, otherwise a plain
`tuple` whose first element is the denoised latents.

## Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

#### diffusers.models.modeling_outputs.Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

```python
diffusers.models.modeling_outputs.Transformer2DModelOutput(sample: torch.Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/modeling_outputs.py#L21)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [Transformer2DModel](/docs/diffusers/v0.41.0/en/api/models/transformer2d#diffusers.Transformer2DModel) is discrete) : The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability distributions for the unnoised latent pixels.

The output of [Transformer2DModel](/docs/diffusers/v0.41.0/en/api/models/transformer2d#diffusers.Transformer2DModel).

### JoyImageEditTransformer3DModel
https://huggingface.co/docs/diffusers/v0.41.0/api/models/transformer_joyimage.md
