# Kandinsky 6 VAEs

Kandinsky 6 uses a causal 3D K-VAE for video super-resolution and the MMAudio mel-spectrogram VAE, paired with a
separate BigVGAN [MMAudioVocoder](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVocoder), for synchronized audio generation.

## Kandinsky6SRVAE[[diffusers.Kandinsky6SRVAE]]

The causal 3D K-VAE used by [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline). It processes arbitrarily long videos in bounded-memory
segments while reproducing the exact output of a single, non-segmented pass.

```python
import torch
from diffusers import Kandinsky6SRVAE

vae = Kandinsky6SRVAE.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
)
```

#### diffusers.Kandinsky6SRVAE[[diffusers.Kandinsky6SRVAE]]

```python
diffusers.Kandinsky6SRVAE(in_channels: int = 3, out_channels: int = 3, latent_channels: int = 64, encoder_block_out_channels: tuple[int, ...] = (16, 128, 256, 512, 1024), decoder_block_out_channels: tuple[int, ...] = (16, 256, 512, 1024, 2048), layers_per_block: int = 2, temporal_compression_ratio: int = 4, temporal_compression_start_level: int = 1, scaling_factor: float = 0.910344004631042)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_kandinsky6_sr.py#L516)

**Parameters:**

in_channels (`int`, defaults to `3`) : Number of pixel channels.

out_channels (`int`, defaults to `3`) : Number of reconstructed pixel channels.

latent_channels (`int`, defaults to `64`) : Number of latent channels.

encoder_block_out_channels (`tuple[int, ...]`, defaults to `(16, 128, 256, 512, 1024)`) : Output width of the residual blocks at each encoder level; every level but the last halves the spatial size and doubles the width on the way to the next level.

decoder_block_out_channels (`tuple[int, ...]`, defaults to `(16, 256, 512, 1024, 2048)`) : Output width of the residual blocks at each decoder level.

layers_per_block (`int`, defaults to `2`) : Number of residual blocks per encoder level; the decoder uses one more per level.

temporal_compression_ratio (`int`, defaults to `4`) : Temporal compression factor; `log2` of it consecutive levels also compress time.

temporal_compression_start_level (`int`, defaults to `1`) : First level that compresses time.

scaling_factor (`float`, defaults to `0.910344`) : Scale applied to the latents before they enter the diffusion transformer.

Causal 3D K-VAE used by [Kandinsky6SRPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6SRPipeline) to encode and decode video.

Videos are processed in temporal segments of 16 pixel frames (plus the leading frame). The causal convolutions
carry their padding state between segments, so the segmentation only bounds peak memory and does not change the
result.

#### encode[[diffusers.Kandinsky6SRVAE.encode]]

```python
encode(x: torch.Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_kandinsky6_sr.py#L585)

**Parameters:**

x (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`) : Pixel video in `[-1, 1]`. `num_frames` should be `1 + k * temporal_compression_ratio`.

return_dict (`bool`, defaults to `True`) : Whether to return an [AutoencoderKLOutput](/docs/diffusers/v0.42.1/en/api/models/autoencoderkl_qwenimage21#diffusers.models.modeling_outputs.AutoencoderKLOutput) instead of a plain tuple.

Encode a video into its latent distribution.

#### decode[[diffusers.Kandinsky6SRVAE.decode]]

```python
decode(z: torch.Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_kandinsky6_sr.py#L611)

**Parameters:**

z (`torch.Tensor` of shape `(batch_size, latent_channels, num_latent_frames, height, width)`) : Latents, already divided by `scaling_factor`.

return_dict (`bool`, defaults to `True`) : Whether to return a `~models.autoencoder_kl.DecoderOutput` instead of a plain tuple.

Decode latents into a video.

#### forward[[diffusers.Kandinsky6SRVAE.forward]]

```python
forward(sample: torch.Tensor, sample_posterior: bool = False, return_dict: bool = True, generator: torch.Generator | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_kandinsky6_sr.py#L642)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`) : Pixel video in `[-1, 1]`. `num_frames` should be `1 + k * temporal_compression_ratio`.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample from the latent posterior instead of using its mode.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.autoencoder_kl.DecoderOutput` instead of a plain tuple.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling deterministic.

**Returns:** `~models.autoencoder_kl.DecoderOutput` or `tuple`

If `return_dict` is True, a `~models.autoencoder_kl.DecoderOutput` is returned, otherwise a plain
`tuple` is returned. Its `sample` is the reconstructed video.

## MMAudioVAE[[diffusers.MMAudioVAE]]

The mel-spectrogram VAE used by [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline) when `sample_audio=True`. Its `decode` output is a mel
spectrogram; pass it through [MMAudioVocoder](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVocoder) to get a waveform.

The reference implementation can be found at [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) (MIT
license).

```python
import torch
from diffusers import MMAudioVAE

audio_vae = MMAudioVAE.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", subfolder="audio_vae", torch_dtype=torch.bfloat16
)
```

#### diffusers.MMAudioVAE[[diffusers.MMAudioVAE]]

```python
diffusers.MMAudioVAE(mel_bins: int = 128, latent_channels: int = 40, hidden_channels: int = 512, channel_multipliers: tuple[int, ...] = (1, 2, 4), layers_per_block: int = 2, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512, scaling_factor: float = 0.417)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_mmaudio.py#L350)

**Parameters:**

mel_bins (`int`, defaults to `128`) : Number of mel bins.

latent_channels (`int`, defaults to `40`) : Number of latent channels.

hidden_channels (`int`, defaults to `512`) : Base width of the autoencoder.

channel_multipliers (`tuple[int, ...]`, defaults to `(1, 2, 4)`) : Width multipliers of the autoencoder levels.

layers_per_block (`int`, defaults to `2`) : Residual blocks per encoder level; the decoder uses one more per level.

sample_rate (`int`, defaults to `44100`) : Waveform sample rate.

n_fft (`int`, defaults to `2048`) : FFT size of the mel front end.

hop_length (`int`, defaults to `512`) : Hop length of the mel front end. Must match the total upsampling factor of the [MMAudioVocoder](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVocoder) this VAE is paired with.

scaling_factor (`float`, defaults to `0.417`) : Scale applied to the latents before they enter the diffusion transformer.

Audio VAE of [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline): a magnitude-preserving autoencoder over log-mel spectrograms (MMAudio,
https://arxiv.org/abs/2412.15322).

`encode` turns a waveform into a latent distribution; `decode` turns latents back into a mel spectrogram, which
[MMAudioVocoder](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVocoder) then turns into a waveform. One latent frame covers `hop_length * 2` samples.

#### encode[[diffusers.MMAudioVAE.encode]]

```python
encode(audio: torch.Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_mmaudio.py#L403)

**Parameters:**

audio (`torch.Tensor` of shape `(batch_size, num_samples)`) : Mono waveform in `[-1, 1]` at `sample_rate`.

return_dict (`bool`, defaults to `True`) : Whether to return an [AutoencoderKLOutput](/docs/diffusers/v0.42.1/en/api/models/autoencoderkl_qwenimage21#diffusers.models.modeling_outputs.AutoencoderKLOutput) instead of a plain tuple.

Encode a waveform into its latent distribution.

#### decode[[diffusers.MMAudioVAE.decode]]

```python
decode(z: torch.Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_mmaudio.py#L420)

**Parameters:**

z (`torch.Tensor` of shape `(batch_size, latent_channels, num_latent_frames)`) : Latents, already divided by `scaling_factor`.

return_dict (`bool`, defaults to `True`) : Whether to return a `~models.autoencoder_kl.DecoderOutput` instead of a plain tuple.

**Returns:**

The mel spectrogram of shape `(batch_size, mel_bins, num_mel_frames)`, ready for [MMAudioVocoder](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVocoder).

Decode latents into a mel spectrogram.

#### forward[[diffusers.MMAudioVAE.forward]]

```python
forward(sample: torch.Tensor, sample_posterior: bool = False, return_dict: bool = True, generator: torch.Generator | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_mmaudio.py#L439)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_samples)`) : Mono waveform in `[-1, 1]` at `sample_rate` to encode and reconstruct as a mel spectrogram.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample from the latent posterior instead of using its mode.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.autoencoder_kl.DecoderOutput` instead of a plain tuple.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling deterministic.

**Returns:** `~models.autoencoder_kl.DecoderOutput` or `tuple`

If `return_dict` is True, a `~models.autoencoder_kl.DecoderOutput` is returned, otherwise a plain
`tuple` is returned. Its `sample` is the reconstructed mel spectrogram of shape `(batch_size, mel_bins,
num_mel_frames)`.

## MMAudioVocoder[[diffusers.MMAudioVocoder]]

Adapted from the BigVGAN-v2 vocoder MMAudio bundles, itself from
[NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN) (MIT license), with the anti-aliased Snake activations of
[alias-free-torch](https://github.com/junjun3518/alias-free-torch) (Apache License 2.0).

#### diffusers.MMAudioVocoder[[diffusers.MMAudioVocoder]]

```python
diffusers.MMAudioVocoder(num_mels: int = 128, upsample_initial_channel: int = 1536, upsample_rates: tuple[int, ...] = (8, 4, 2, 2, 2, 2), upsample_kernel_sizes: tuple[int, ...] = (16, 8, 4, 4, 4, 4), resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11), resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)))
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/mmaudio_vocoder.py#L154)

**Parameters:**

num_mels (`int`, defaults to `128`) : Number of mel bins of the input spectrogram. Must match the paired [MMAudioVAE](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVAE)'s `mel_bins`.

upsample_initial_channel (`int`, defaults to `1536`) : Width of the first layer.

upsample_rates (`tuple[int, ...]`, defaults to `(8, 4, 2, 2, 2, 2)`) : Upsampling factors of the vocoder stages. Their product is the total upsampling factor and must match the paired [MMAudioVAE](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVAE)'s `hop_length`.

upsample_kernel_sizes (`tuple[int, ...]`, defaults to `(16, 8, 4, 4, 4, 4)`) : Transposed-convolution kernel sizes of the vocoder stages.

resblock_kernel_sizes (`tuple[int, ...]`, defaults to `(3, 7, 11)`) : Kernel sizes of the residual blocks.

resblock_dilation_sizes (`tuple[tuple[int, ...], ...]`, defaults to `((1, 3, 5), (1, 3, 5), (1, 3, 5))`) : Dilations of the residual blocks.

BigVGAN-v2 vocoder (https://github.com/NVIDIA/BigVGAN, MIT license) with the anti-aliased Snake activations of
https://github.com/junjun3518/alias-free-torch (Apache 2.0), turning the mel spectrograms [MMAudioVAE](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVAE) decodes
into waveforms for [Kandinsky6TI2VAPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/kandinsky6#diffusers.Kandinsky6TI2VAPipeline).

#### forward[[diffusers.MMAudioVocoder.forward]]

```python
forward(mel: torch.Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/mmaudio_vocoder.py#L213)

**Parameters:**

mel (`torch.Tensor` of shape `(batch_size, num_mels, num_mel_frames)`) : Mel spectrogram, as decoded by [MMAudioVAE](/docs/diffusers/v0.42.1/en/api/models/kandinsky6_vae#diffusers.MMAudioVAE).

return_dict (`bool`, defaults to `True`) : Whether to return a `~models.autoencoder_kl.DecoderOutput` instead of a plain tuple.

**Returns:**

The waveform of shape `(batch_size, 1, num_samples)` in `[-1, 1]`.

### AsymmetricAutoencoderKL
https://huggingface.co/docs/diffusers/v0.42.1/api/models/asymmetricautoencoderkl.md
