# AutoencoderKLWan

The 3D variational autoencoder (VAE) model with KL loss used in [Wan 2.1](https://github.com/Wan-Video/Wan2.1) by the Alibaba Wan Team.

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLWan

vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
```

## AutoencoderKLWan[[diffusers.AutoencoderKLWan]]

#### diffusers.AutoencoderKLWan[[diffusers.AutoencoderKLWan]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L960)

A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
Introduced in [Wan 2.1].

This model inherits from [ModelMixin](/docs/diffusers/v0.39.0/en/api/models/overview#diffusers.ModelMixin). Check the superclass documentation for it's generic methods implemented
for all models (such as downloading or saving).

wrapperdiffusers.AutoencoderKLWan.decodehttps://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/utils/accelerate_utils.py#L43[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]
#### enable_tiling[[diffusers.AutoencoderKLWan.enable_tiling]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1093)

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.

**Parameters:**

tile_sample_min_height (`int`, *optional*) : The minimum height required for a sample to be separated into tiles across the height dimension.

tile_sample_min_width (`int`, *optional*) : The minimum width required for a sample to be separated into tiles across the width dimension.

tile_sample_stride_height (`int`, *optional*) : The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are no tiling artifacts produced across the height dimension.

tile_sample_stride_width (`int`, *optional*) : The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling artifacts produced across the width dimension.
#### forward[[diffusers.AutoencoderKLWan.forward]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1409)

**Parameters:**

sample (`torch.Tensor`) : Input sample.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample from the posterior.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `DecoderOutput` instead of a plain tuple.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling deterministic.

**Returns:**

``~models.vae.DecoderOutput` or `tuple``

If `return_dict` is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.
#### tiled_decode[[diffusers.AutoencoderKLWan.tiled_decode]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1331)

Decode a batch of images using a tiled decoder.

**Parameters:**

z (`torch.Tensor`) : Input batch of latent vectors.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~models.vae.DecoderOutput` instead of a plain tuple.

**Returns:**

``~models.vae.DecoderOutput` or `tuple``

If return_dict is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.
#### tiled_encode[[diffusers.AutoencoderKLWan.tiled_encode]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1259)

Encode a batch of images using a tiled encoder.

**Parameters:**

x (`torch.Tensor`) : Input batch of videos.

**Returns:**

``torch.Tensor``

The latent representation of the encoded videos.

## DecoderOutput[[diffusers.models.autoencoders.vae.DecoderOutput]]

#### diffusers.models.autoencoders.vae.DecoderOutput[[diffusers.models.autoencoders.vae.DecoderOutput]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/vae.py#L46)

Output of decoding method.

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`) : The decoded output sample from the last layer of the model.

### LatteTransformer3DModel
https://huggingface.co/docs/diffusers/v0.39.0/api/models/latte_transformer3d.md

## LatteTransformer3DModel

A Diffusion Transformer model for 3D data from [Latte](https://github.com/Vchitect/Latte).

## LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

#### diffusers.LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

[Source](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/latte_transformer_3d.py#L27)

forwarddiffusers.LatteTransformer3DModel.forwardhttps://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/latte_transformer_3d.py#L166[{"name": "hidden_states", "val": ": Tensor"}, {"name": "timestep", "val": ": torch.LongTensor | None = None"}, {"name": "encoder_hidden_states", "val": ": torch.Tensor | None = None"}, {"name": "encoder_attention_mask", "val": ": torch.Tensor | None = None"}, {"name": "enable_temporal_attentions", "val": ": bool = True"}, {"name": "return_dict", "val": ": bool = True"}]- **hidden_states** (`torch.Tensor` of shape `(batch size, channel, num_frame, height, width)`) --
  Input `hidden_states`.
- **timestep** ( `torch.LongTensor`, *optional*) --
  Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
- **encoder_hidden_states** ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*) --
  Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
  self-attention.
- **encoder_attention_mask** ( `torch.Tensor`, *optional*) --
  Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

  * Mask `(batcheight, sequence_length)` True = keep, False = discard.
  * Bias `(batcheight, 1, sequence_length)` 0 = keep, -10000 = discard.

  If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
  above. This bias will be added to the cross-attention scores.
- **enable_temporal_attentions** --
  (`bool`, *optional*, defaults to `True`): Whether to enable temporal attentions.
- **return_dict** (`bool`, *optional*, defaults to `True`) --
  Whether or not to return a `~models.unet_2d_condition.UNet2DConditionOutput` instead of a plain
  tuple.0If `return_dict` is True, an `~models.transformer_2d.Transformer2DModelOutput` is returned, otherwise a
`tuple` where the first element is the sample tensor.

The [LatteTransformer3DModel](/docs/diffusers/v0.39.0/en/api/models/latte_transformer3d#diffusers.LatteTransformer3DModel) forward method.

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

### ChromaTransformer2DModel
https://huggingface.co/docs/diffusers/v0.39.0/api/models/chroma_transformer.md
