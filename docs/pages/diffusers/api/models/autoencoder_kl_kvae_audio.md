# AutoencoderKLKVAEAudio

A 1D convolutional variational autoencoder (VAE) with KL loss for audio, introduced by the Kandinsky Lab
Team in [KVAE-Audio](https://huggingface.co/kandinskylab/KVAE-Audio). It compresses raw, full-band (48 kHz)
waveforms into compact continuous latents and reconstructs them with high fidelity across speech, music, and
general audio.

The model can be loaded with the following code snippet.

```python
import torch
from diffusers import AutoencoderKLKVAEAudio

vae = AutoencoderKLKVAEAudio.from_pretrained("kandinskylab/KVAE-Audio", subfolder="diffusers", dtype=torch.float32)
```

## AutoencoderKLKVAEAudio[[diffusers.AutoencoderKLKVAEAudio]]

#### diffusers.AutoencoderKLKVAEAudio[[diffusers.AutoencoderKLKVAEAudio]]

```python
diffusers.AutoencoderKLKVAEAudio(encoder_dim: int = 64, encoder_rates: list = [2, 3, 4, 5, 8], latent_dim: int = 2048, codebook_dim: int = 64, decoder_dim: int = 1536, decoder_rates: list = [8, 5, 4, 3, 2], sample_rate: int = 48000, num_channels: int = 1, attn_num_heads: int = 8)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_kl_kvae_audio.py#L249)

**Parameters:**

encoder_dim (`int`, *optional*, defaults to 64) : Base channel dimension for the encoder; doubled at every downsampling stage.

encoder_rates (`list[int]`, *optional*, defaults to `[2, 3, 4, 5, 8]`) : Strides for downsampling in the encoder.

latent_dim (`int`, *optional*, defaults to 2048) : Channel dimension of the encoder output (before the `in_proj` bottleneck).

codebook_dim (`int`, *optional*, defaults to 64) : Channel dimension of the posterior distribution's latent space (after the `in_proj` bottleneck).

decoder_dim (`int`, *optional*, defaults to 1536) : Base channel dimension for the decoder; halved at every upsampling stage.

decoder_rates (`list[int]`, *optional*, defaults to `[8, 5, 4, 3, 2]`) : Strides for upsampling in the decoder.

sample_rate (`int`, *optional*, defaults to 48000) : The sampling rate, in Hz, that the model was trained on.

num_channels (`int`, *optional*, defaults to 1) : Number of channels in the audio data (1 for mono).

attn_num_heads (`int`, *optional*, defaults to 8) : Number of heads in the self-attention block applied to the encoder output.

A 1D convolutional autoencoder for encoding raw audio waveforms into continuous latents and decoding them back into
waveforms. Introduced in [KVAE-Audio](https://huggingface.co/kandinskylab/KVAE-Audio).

This model inherits from [ModelMixin](/docs/diffusers/v0.42.0/en/api/models/overview#diffusers.ModelMixin). Check the superclass documentation for it's generic methods implemented
for all models (such as downloading or saving).

#### decode[[diffusers.AutoencoderKLKVAEAudio.decode]]

```python
decode(z: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_kl_kvae_audio.py#L354)

**Parameters:**

z (`torch.Tensor`) : Input batch of latent vectors.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a [DecoderOutput](/docs/diffusers/v0.42.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) instead of a plain tuple.

**Returns:** [DecoderOutput](/docs/diffusers/v0.42.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) or `tuple`

If return_dict is True, a [DecoderOutput](/docs/diffusers/v0.42.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) is returned, otherwise a plain
`tuple` is returned.

Decode a batch of latents into audio waveforms.

#### encode[[diffusers.AutoencoderKLKVAEAudio.encode]]

```python
encode(audio_data: Tensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_kl_kvae_audio.py#L321)

**Parameters:**

audio_data (`torch.Tensor` of shape `(batch_size, num_channels, num_samples)`) : Input batch of raw audio waveforms.

return_dict (`bool`, *optional*, defaults to `True`) : Whether to return a `~models.autoencoder_kl.AutoencoderKLOutput` instead of a plain tuple.

**Returns:**

The latent representations of the encoded audio. If `return_dict` is True, a
`~models.autoencoder_kl.AutoencoderKLOutput` is returned, otherwise a plain `tuple` is returned.

Encode a batch of audio waveforms into latents.

#### forward[[diffusers.AutoencoderKLKVAEAudio.forward]]

```python
forward(sample: Tensor, sample_posterior: bool = False, return_dict: bool = True, generator: typing.Optional[torch.Generator] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_kl_kvae_audio.py#L379)

**Parameters:**

sample (`torch.Tensor` of shape `(batch_size, num_channels, num_samples)`) : Input batch of raw audio waveforms.

sample_posterior (`bool`, *optional*, defaults to `False`) : Whether to sample from the posterior.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not to return a [DecoderOutput](/docs/diffusers/v0.42.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) instead of a plain tuple.

generator (`torch.Generator`, *optional*) : A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make sampling deterministic.

**Returns:** [DecoderOutput](/docs/diffusers/v0.42.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) or `tuple`

If `return_dict` is True, a [DecoderOutput](/docs/diffusers/v0.42.0/en/api/models/autoencoderkl_qwenimage21#diffusers.models.autoencoders.vae.DecoderOutput) is returned, otherwise a plain
`tuple` is returned.

### HunyuanVideoTransformer3DModel
https://huggingface.co/docs/diffusers/v0.42.0/api/models/hunyuan_video_transformer_3d.md
