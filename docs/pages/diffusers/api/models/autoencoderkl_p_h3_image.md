# AutoencoderKLPH3Image

The image VAE of [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image) by Pruna AI. It uses the encoder and latent space of MiniMax-H3's [video VAE](./autoencoderkl_minimax_h3), with a ViT decoder that decodes a single latent frame into an image.

It follows the conventions of [AutoencoderKLMiniMaxH3](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_minimax_h3#diffusers.AutoencoderKLMiniMaxH3):

- Images are passed as single-frame videos, `(batch_size, channels, 1, height, width)`, with 16x spatial compression.
- Latents are normalized per channel with `latents_mean` / `latents_std` instead of a `scaling_factor`.
- Pixels are ImageNet-normalized RGB over `[0, 1]`: `encode` expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns that space.
- Spatial tiling is on by default, with 256 pixel tiles and at least 64 pixels of overlap.

```python
import torch
from diffusers import AutoencoderKLPH3Image

vae = AutoencoderKLPH3Image.from_pretrained("PrunaAI/P-H3-Image", subfolder="vae", dtype=torch.bfloat16).to(
    "cuda"
)  # or "mps", "xpu", "cpu"
```

The encoder and the decoder's output projection stay in float32 whatever `dtype` the model is loaded in.

## AutoencoderKLPH3Image[[diffusers.AutoencoderKLPH3Image]]

#### diffusers.AutoencoderKLPH3Image[[diffusers.AutoencoderKLPH3Image]]

```python
diffusers.AutoencoderKLPH3Image(in_channels: int = 3, out_channels: int = 3, latent_channels: int = 24, block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024), layers_per_block: int = 2, spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1), temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1), norm_num_groups: int = 32, norm_eps: float = 1e-06, spatial_padding_mode: str = 'reflect', decoder_num_layers: int = 36, decoder_num_attention_heads: int = 32, decoder_attention_head_dim: int = 64, decoder_num_register_tokens: int = 4, decoder_ffn_mult: int = 4, decoder_rope_theta: float = 100.0, decoder_rope_dim_ratio: float = 0.75, decoder_norm_eps: float = 1e-05, latents_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), latents_std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L493)

The image VAE of [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image): MiniMax-H3's VAE encoder and latent
space, with a `PH3ImageViTDecoder` that decodes a single latent frame.

Images are passed as single-frame videos, `(batch_size, channels, 1, height, width)`. As in
[AutoencoderKLMiniMaxH3](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_minimax_h3#diffusers.AutoencoderKLMiniMaxH3), pixels are ImageNet-normalized RGB over `[0, 1]` (mean `(0.485, 0.456, 0.406)`, std
`(0.229, 0.224, 0.225)`) and spatial tiling is on by default.

#### encode[[diffusers.AutoencoderKLPH3Image.encode]]

```python
encode(x: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L728)

**Parameters:**

x (`torch.Tensor`) : Input batch of images as single-frame videos, shape `(batch_size, in_channels, 1, height, width)`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [AutoencoderKLOutput](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_qwenimage#diffusers.models.modeling_outputs.AutoencoderKLOutput) instead of a plain tuple.

**Returns:**

The latent distribution of the encoded images. Note that the latents are normalized with `latents_mean` /
`latents_std` afterwards.

Encode a batch of images into latents.

#### decode[[diffusers.AutoencoderKLPH3Image.decode]]

```python
decode(z: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L757)

**Parameters:**

z (`torch.Tensor`) : Input batch of latents, shape `(batch_size, latent_channels, 1, height, width)`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [DecoderOutput](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_qwenimage#diffusers.models.autoencoders.vae.DecoderOutput) instead of a plain tuple.

**Returns:** [DecoderOutput](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_qwenimage#diffusers.models.autoencoders.vae.DecoderOutput) or `tuple`

The decoded images, shape `(batch_size, out_channels, 1, height, width)`.

Decode a batch of latent images.

#### enable_tiling[[diffusers.AutoencoderKLPH3Image.enable_tiling]]

```python
enable_tiling(tile_sample_min_height: int | None = None, tile_sample_min_width: int | None = None, tile_sample_min_overlap_height: int | None = None, tile_sample_min_overlap_width: int | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L575)

**Parameters:**

tile_sample_min_height (`int`, *optional*) : The tile height in pixel space. Frames taller than this are split along the height dimension.

tile_sample_min_width (`int`, *optional*) : The tile width in pixel space. Frames wider than this are split along the width dimension.

tile_sample_min_overlap_height (`int`, *optional*) : The minimum overlap, in pixels, between two consecutive vertical tiles.

tile_sample_min_overlap_width (`int`, *optional*) : The minimum overlap, in pixels, between two consecutive horizontal tiles.

Enable tiled VAE encoding/decoding. When this option is enabled, the VAE splits the frames into tiles, encodes
or decodes each tile separately and linearly blends the overlaps back together. This lowers the memory
requirement and allows processing larger frames.

#### forward[[diffusers.AutoencoderKLPH3Image.forward]]

```python
forward(sample: Tensor, sample_posterior: bool = False, generator: typing.Optional[torch.Generator] = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L784)

**Parameters:**

sample (`torch.Tensor`) : Input batch of images as single-frame videos, shape `(batch_size, in_channels, 1, height, width)`.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample the posterior instead of taking its mode.

generator (`torch.Generator`, *optional*) : Generator used when `sample_posterior=True`.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [DecoderOutput](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_qwenimage#diffusers.models.autoencoders.vae.DecoderOutput) instead of a plain tuple.

**Returns:** [DecoderOutput](/docs/diffusers/v0.43.0/en/api/models/autoencoderkl_qwenimage#diffusers.models.autoencoders.vae.DecoderOutput) or `tuple`

The round-tripped images, shape `(batch_size, out_channels, 1, height, width)`.

Encode then decode a batch of images.

### AutoencoderKLQwenImage
https://huggingface.co/docs/diffusers/v0.43.0/api/models/autoencoderkl_qwenimage.md
