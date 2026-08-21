# AutoencoderKLHunyuanImageRefiner

The 3D variational autoencoder (VAE) model with KL loss used in [HunyuanImage2.1](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) for its refiner pipeline.

The model can be loaded with the following code snippet.

```python
from diffusers import AutoencoderKLHunyuanImageRefiner

vae = AutoencoderKLHunyuanImageRefiner.from_pretrained("hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers", subfolder="vae", dtype=torch.bfloat16)
```

## AutoencoderKLHunyuanImageRefiner[[diffusers.AutoencoderKLHunyuanImageRefiner]]

#### diffusers.AutoencoderKLHunyuanImageRefiner[[diffusers.AutoencoderKLHunyuanImageRefiner]]

```python
diffusers.AutoencoderKLHunyuanImageRefiner(in_channels: int = 3, out_channels: int = 3, latent_channels: int = 32, block_out_channels: tuple = (128, 256, 512, 1024, 1024), layers_per_block: int = 2, spatial_compression_ratio: int = 16, temporal_compression_ratio: int = 4, downsample_match_channel: bool = True, upsample_match_channel: bool = True, scaling_factor: float = 1.03682)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L593)

A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos. Used for
HunyuanImage-2.1 Refiner.

This model inherits from [ModelMixin](/docs/diffusers/v0.40.0/en/api/models/overview#diffusers.ModelMixin). Check the superclass documentation for it's generic methods implemented
for all models (such as downloading or saving).

#### decode[[diffusers.AutoencoderKLHunyuanImageRefiner.decode]]

```python
decode(z: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L743)

**Parameters:**

z (`torch.Tensor`) : Input batch of latent vectors.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.vae.DecoderOutput` instead of a plain tuple.

**Returns:** `~models.vae.DecoderOutput` or `tuple`

If return_dict is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.

Decode a batch of images.

#### enable_tiling[[diffusers.AutoencoderKLHunyuanImageRefiner.enable_tiling]]

```python
enable_tiling(tile_sample_min_height: int | None = None, tile_sample_min_width: int | None = None, tile_sample_stride_height: float | None = None, tile_sample_stride_width: float | None = None, tile_overlap_factor: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L662)

**Parameters:**

tile_sample_min_height (`int`, *optional*) : The minimum height required for a sample to be separated into tiles across the height dimension.

tile_sample_min_width (`int`, *optional*) : The minimum width required for a sample to be separated into tiles across the width dimension.

tile_sample_stride_height (`int`, *optional*) : The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are no tiling artifacts produced across the height dimension.

tile_sample_stride_width (`int`, *optional*) : The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling artifacts produced across the width dimension.

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.

#### encode[[diffusers.AutoencoderKLHunyuanImageRefiner.encode]]

```python
encode(x: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L703)

**Parameters:**

x (`torch.Tensor`) : Input batch of images.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.autoencoder_kl.AutoencoderKLOutput` instead of a plain tuple.

**Returns:**

The latent representations of the encoded videos. If `return_dict` is True, a
`~models.autoencoder_kl.AutoencoderKLOutput` is returned, otherwise a plain `tuple` is returned.

Encode a batch of images into latents.

#### forward[[diffusers.AutoencoderKLHunyuanImageRefiner.forward]]

```python
forward(sample: Tensor, sample_posterior: bool = False, return_dict: bool = True, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L897)

**Parameters:**

sample (`torch.Tensor`) : Input sample.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample from the posterior.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `DecoderOutput` instead of a plain tuple.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling deterministic.

**Returns:** `~models.vae.DecoderOutput` or `tuple`

If `return_dict` is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.

#### tiled_decode[[diffusers.AutoencoderKLHunyuanImageRefiner.tiled_decode]]

```python
tiled_decode(z: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L843)

**Parameters:**

z (`torch.Tensor`) : Input batch of latent vectors.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~models.vae.DecoderOutput` instead of a plain tuple.

**Returns:** `~models.vae.DecoderOutput` or `tuple`

If return_dict is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.

Decode a batch of images using a tiled decoder.

#### tiled_encode[[diffusers.AutoencoderKLHunyuanImageRefiner.tiled_encode]]

```python
tiled_encode(x: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py#L793)

**Parameters:**

x (`torch.Tensor`) : Input batch of videos.

**Returns:** `torch.Tensor`

The latent representation of the encoded videos.

Encode a batch of images using a tiled encoder.

## DecoderOutput[[diffusers.models.autoencoders.vae.DecoderOutput]]

#### diffusers.models.autoencoders.vae.DecoderOutput[[diffusers.models.autoencoders.vae.DecoderOutput]]

```python
diffusers.models.autoencoders.vae.DecoderOutput(sample: Tensor, commit_loss: typing.Optional[torch.FloatTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/vae.py#L46)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`) : The decoded output sample from the last layer of the model.

Output of decoding method.

### LatteTransformer3DModel
https://huggingface.co/docs/diffusers/v0.40.0/api/models/latte_transformer3d.md

## LatteTransformer3DModel

A Diffusion Transformer model for 3D data from [Latte](https://github.com/Vchitect/Latte).

## LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

#### diffusers.LatteTransformer3DModel[[diffusers.LatteTransformer3DModel]]

```python
diffusers.LatteTransformer3DModel(num_attention_heads: int = 16, attention_head_dim: int = 88, in_channels: int | None = None, out_channels: int | None = None, num_layers: int = 1, dropout: float = 0.0, cross_attention_dim: int | None = None, attention_bias: bool = False, sample_size: int = 64, patch_size: int | None = None, activation_fn: str = 'geglu', num_embeds_ada_norm: int | None = None, norm_type: str = 'layer_norm', norm_elementwise_affine: bool = True, norm_eps: float = 1e-05, caption_channels: int = None, video_length: int = 16)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/latte_transformer_3d.py#L27)

#### forward[[diffusers.LatteTransformer3DModel.forward]]

```python
forward(hidden_states: Tensor, timestep: typing.Optional[torch.LongTensor] = None, encoder_hidden_states: typing.Optional[torch.Tensor] = None, encoder_attention_mask: typing.Optional[torch.Tensor] = None, enable_temporal_attentions: bool = True, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/latte_transformer_3d.py#L166)

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

The [LatteTransformer3DModel](/docs/diffusers/v0.40.0/en/api/models/latte_transformer3d#diffusers.LatteTransformer3DModel) forward method.

### StableAudio3DiTModel
https://huggingface.co/docs/diffusers/v0.40.0/api/models/stable_audio_3_transformer.md
