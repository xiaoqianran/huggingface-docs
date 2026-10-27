# Flux3Transformer2DModel

A Transformer model for image-like data from [FLUX.3](https://bfl.ai/), for text-to-image and image-to-image generation. Three token modes share one joint sequence: the noisy image tokens, the VAE-encoded reference image tokens and the text context. Each mode first passes through its own per-mode transformer blocks, then the modes are concatenated for the joint single blocks, and the image tokens pass through the output layers. Timesteps are per token: the image tokens carry the flow timestep, the reference tokens and the text context sit at `0`.

```python
import torch
from diffusers import Flux3Transformer2DModel

transformer = Flux3Transformer2DModel.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", subfolder="transformer", dtype=torch.bfloat16
)
```

## Flux3Transformer2DModel[[diffusers.Flux3Transformer2DModel]]

#### diffusers.Flux3Transformer2DModel[[diffusers.Flux3Transformer2DModel]]

```python
diffusers.Flux3Transformer2DModel(in_channels: int = 128, cond_in_channels: int = 128, num_layers: int = 5, num_single_layers: int = 28, attention_head_dim: int = 128, num_attention_heads: int = 24, joint_attention_dim: int = 20480, timestep_channels: int = 256, mlp_ratio: float = 3.0, axes_dims_rope: tuple = (32, 32, 32, 32), rope_theta: int = 10000, eps: float = 1e-06, flow_map: bool = False)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_flux3.py#L341)

**Parameters:**

in_channels (`int`, defaults to `128`) : Channel width of the image tokens (the packed VAE latents, `latent_channels * 4`).

cond_in_channels (`int`, defaults to `128`) : Channel width of the reference image tokens.

num_layers (`int`, defaults to `5`) : The number of per-mode transformer blocks.

num_single_layers (`int`, defaults to `28`) : The number of joint single transformer blocks.

attention_head_dim (`int`, defaults to `128`) : The number of dimensions to use for each attention head.

num_attention_heads (`int`, defaults to `24`) : The number of attention heads to use.

joint_attention_dim (`int`, defaults to `20480`) : The embedding/channel dimension of `encoder_hidden_states` (stacked Qwen3-VL hidden states).

timestep_channels (`int`, defaults to `256`) : The number of channels of the sinusoidal timestep embedding.

mlp_ratio (`float`, defaults to `3.0`) : The expansion ratio of the feedforward sub-blocks.

axes_dims_rope (`tuple[int]`, defaults to `(32, 32, 32, 32)`) : The dimensions to use for the rotary positional embeddings, per `(t, h, w, l)` position axis.

rope_theta (`int`, defaults to `10000`) : The base frequency of the rotary positional embeddings.

eps (`float`, defaults to `1e-6`) : The epsilon of the layer norms and the QK norms.

flow_map (`bool`, defaults to `False`) : Whether the model is a flow map (step-distilled): adds the `target_timestep_embedder` and requires `target_timestep` in the forward pass.

The Transformer model introduced in FLUX.3 for text-to-image and image-to-image generation.

Three token modes share one joint sequence: the noisy image tokens (`hidden_states`), the optional VAE-encoded
reference image tokens (`image_cond_hidden_states`) and the text context (`encoder_hidden_states`). Each mode first
passes through its own per-mode transformer blocks (`mode_blocks` for the image modes, `txt_mode_blocks` for the
text context), then the text context and the active image modes are concatenated for the joint single blocks, and
only the image tokens pass through the output layers. The modulation is per mode: the image tokens are modulated by
the flow timestep, the reference image tokens and the text context by timestep `0`. A flow-map checkpoint
(`flow_map=True`, the step-distilled models) additionally embeds the timestep each denoising step jumps to and adds
it to every mode's modulation.

#### forward[[diffusers.Flux3Transformer2DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: Tensor, img_ids: Tensor, txt_ids: Tensor, image_cond_hidden_states: typing.Optional[torch.Tensor] = None, image_cond_ids: typing.Optional[torch.Tensor] = None, target_timestep: typing.Optional[torch.Tensor] = None, attention_kwargs: dict[str, typing.Any] | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_flux3.py#L494)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(batch_size, image_seq_len, in_channels)`) : Noisy image latent tokens.

encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, joint_attention_dim)`) : Text context embeddings (stacked Qwen3-VL hidden states).

timestep (`torch.Tensor` of shape `(batch_size,)`) : Flow timestep in `[0, 1]` of the image tokens. The reference image tokens and the text context sit at timestep `0`.

img_ids (`torch.Tensor` of shape `(image_seq_len, 4)`) : Position ids `(t, h, w, l)` of the image tokens, used for the rotary positional embeddings and shared by all samples in the batch.

txt_ids (`torch.Tensor` of shape `(text_seq_len, 4)`) : Position ids of the text context tokens.

image_cond_hidden_states (`torch.Tensor` of shape `(batch_size, cond_seq_len, cond_in_channels)`, *optional*) : VAE-encoded reference image tokens (clean conditioning content, for image-to-image generation).

image_cond_ids (`torch.Tensor` of shape `(cond_seq_len, 4)`, *optional*) : Position ids of the reference image tokens. Required with `image_cond_hidden_states`.

target_timestep (`torch.Tensor` of shape `(batch_size,)`, *optional*) : Flow timestep in `[0, 1]` that the current denoising step jumps to. Required by, and only by, a flow-map model (`flow_map=True`).

attention_kwargs (`dict`, *optional*) : A kwargs dictionary whose `scale` entry, if specified, scales the LoRA layers for this forward pass.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~models.transformer_2d.Transformer2DModelOutput` instead of a plain tuple.

**Returns:**

If `return_dict` is True, an `~models.transformer_2d.Transformer2DModelOutput` is returned, otherwise a
`tuple` where the first element is the predicted velocity of the image tokens, of the same shape as
`hidden_states`.

The [Flux3Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/flux3_transformer#diffusers.Flux3Transformer2DModel) forward method.

## Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

#### diffusers.models.modeling_outputs.Transformer2DModelOutput[[diffusers.models.modeling_outputs.Transformer2DModelOutput]]

```python
diffusers.models.modeling_outputs.Transformer2DModelOutput(sample: torch.Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/modeling_outputs.py#L21)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/transformer2d#diffusers.Transformer2DModel) is discrete) : The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability distributions for the unnoised latent pixels.

The output of [Transformer2DModel](/docs/diffusers/v0.42.1/en/api/models/transformer2d#diffusers.Transformer2DModel).

### ZImageTransformer2DModel
https://huggingface.co/docs/diffusers/v0.42.1/api/models/z_image_transformer2d.md
