# AutoencoderKLQwenImage21

The 64-channel variational auto-encoder used by Qwen-Image 2.1. It compresses 16x spatially, and its per-channel
`latents_mean` / `latents_std` are part of the config rather than a single scaling factor.

```python
import torch
from diffusers import AutoencoderKLQwenImage21

vae = AutoencoderKLQwenImage21.from_pretrained("Qwen/Qwen-Image-2.1", subfolder="vae", dtype=torch.bfloat16)
```

## AutoencoderKLQwenImage21[[diffusers.AutoencoderKLQwenImage21]]

#### diffusers.AutoencoderKLQwenImage21[[diffusers.AutoencoderKLQwenImage21]]

```python
diffusers.AutoencoderKLQwenImage21(base_dim: int = 96, decoder_base_dim: int | None = 144, z_dim: int = 64, dim_mult: list = [1, 2, 4, 8, 8], num_res_blocks: int = 2, attn_scales: list = [], temperal_downsample: list = [False, True, True, True], dropout: float = 0.0, latents_mean: list = [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304], latents_std: list = [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161], is_residual: bool = True, in_channels: int = 4, out_channels: int = 4, patch_size: int | None = None, scale_factor_temporal: int | None = 8, scale_factor_spatial: int | None = 16)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L977)

A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
Introduced in [Qwen Image 2].

This model inherits from [ModelMixin](/docs/diffusers/v0.41.0/en/api/models/overview#diffusers.ModelMixin). Check the superclass documentation for it's generic methods implemented
for all models (such as downloading or saving).

#### decode[[diffusers.AutoencoderKLQwenImage21.decode]]

```python
decode(z: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1336)

**Parameters:**

z (`torch.Tensor`) : Input batch of latent vectors.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.vae.DecoderOutput` instead of a plain tuple.

**Returns:** `~models.vae.DecoderOutput` or `tuple`

If return_dict is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.

Decode a batch of images.

#### encode[[diffusers.AutoencoderKLQwenImage21.encode]]

```python
encode(x: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1276)

**Parameters:**

x (`torch.Tensor`) : Input batch of images.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.autoencoder_kl.AutoencoderKLOutput` instead of a plain tuple.

**Returns:**

The latent representations of the encoded videos. If `return_dict` is True, a
`~models.autoencoder_kl.AutoencoderKLOutput` is returned, otherwise a plain `tuple` is returned.

Encode a batch of images into latents.

#### enable_tiling[[diffusers.AutoencoderKLQwenImage21.enable_tiling]]

```python
enable_tiling(tile_sample_min_height: int | None = None, tile_sample_min_width: int | None = None, tile_sample_stride_height: float | None = None, tile_sample_stride_width: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1207)

**Parameters:**

tile_sample_min_height (`int`, *optional*) : The minimum height required for a sample to be separated into tiles across the height dimension.

tile_sample_min_width (`int`, *optional*) : The minimum width required for a sample to be separated into tiles across the width dimension.

tile_sample_stride_height (`int`, *optional*) : The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are no tiling artifacts produced across the height dimension.

tile_sample_stride_width (`int`, *optional*) : The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling artifacts produced across the width dimension.

Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
processing larger images.

#### forward[[diffusers.AutoencoderKLQwenImage21.forward]]

```python
forward(sample: Tensor, sample_posterior: bool = False, return_dict: bool = True, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1532)

**Parameters:**

sample (`torch.Tensor`) : Input sample.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample from the posterior instead of taking its mode.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `DecoderOutput` instead of a plain tuple.

generator (`torch.Generator`, *optional*) : Generator used when `sample_posterior` is `True`.

**Returns:** [DecoderOutput](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) or `tuple`

[DecoderOutput](/docs/diffusers/v0.41.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) if `return_dict` is True, otherwise a plain `tuple`.

#### tiled_decode[[diffusers.AutoencoderKLQwenImage21.tiled_decode]]

```python
tiled_decode(z: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1454)

**Parameters:**

z (`torch.Tensor`) : Input batch of latent vectors.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a `~models.vae.DecoderOutput` instead of a plain tuple.

**Returns:** `~models.vae.DecoderOutput` or `tuple`

If return_dict is True, a `~models.vae.DecoderOutput` is returned, otherwise a plain `tuple` is
returned.

Decode a batch of images using a tiled decoder.

#### tiled_encode[[diffusers.AutoencoderKLQwenImage21.tiled_encode]]

```python
tiled_encode(x: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py#L1381)

**Parameters:**

x (`torch.Tensor`) : Input batch of videos.

**Returns:** `torch.Tensor`

The latent representation of the encoded videos.

Encode a batch of images using a tiled encoder.

## AutoencoderKLOutput[[diffusers.models.modeling_outputs.AutoencoderKLOutput]]

#### diffusers.models.modeling_outputs.AutoencoderKLOutput[[diffusers.models.modeling_outputs.AutoencoderKLOutput]]

```python
diffusers.models.modeling_outputs.AutoencoderKLOutput(latent_dist: DiagonalGaussianDistribution)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/modeling_outputs.py#L7)

**Parameters:**

latent_dist (`DiagonalGaussianDistribution`) : Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`. `DiagonalGaussianDistribution` allows for sampling latents from the distribution.

Output of AutoencoderKL encoding method.

## DecoderOutput[[diffusers.models.autoencoders.vae.DecoderOutput]]

#### diffusers.models.autoencoders.vae.DecoderOutput[[diffusers.models.autoencoders.vae.DecoderOutput]]

```python
diffusers.models.autoencoders.vae.DecoderOutput(sample: Tensor, commit_loss: typing.Optional[torch.FloatTensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/vae.py#L49)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`) : The decoded output sample from the last layer of the model.

Output of decoding method.

### HeliosTransformer3DModel
https://huggingface.co/docs/diffusers/v0.41.0/api/models/helios_transformer3d.md
