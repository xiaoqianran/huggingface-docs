# RFD4ProteinaAutoencoder

The autoencoder of [RFD4-Proteína](../pipelines/rfd4_proteina). `encode` maps atom37 coordinates to per-token local
latents, and `decode` turns latents and backbone positions back into atom37 structure and residue identities. It is
registered in the pipeline as `vae`.

```py
from diffusers import RFD4ProteinaAutoencoder

vae = RFD4ProteinaAutoencoder.from_pretrained("nvidia/RFD4-Proteina-Diffusers", subfolder="vae")
```

## RFD4ProteinaAutoencoder[[diffusers.RFD4ProteinaAutoencoder]]

#### diffusers.RFD4ProteinaAutoencoder[[diffusers.RFD4ProteinaAutoencoder]]

```python
diffusers.RFD4ProteinaAutoencoder(atom37_mask: Sequence, allowed_residue_types: Sequence, latent_dim: int = 32, n_residue_types: int = 147, num_token_types: int = 7, nlayers: int = 12, token_dim: int = 768, nheads: int = 12, dim_cond: int = 128, pair_repr_dim: int = 256, token_type_emb_dim: int = 32, encoder_feats_seq: tuple = ('token_type', 'x1_a37coors_nm', 'x1_a37coors_nm_rel', 'chain_idx_seq', 'incident_bond_type_counts', 'atom37_element_charge_embedding'), decoder_feats_seq: tuple = ('ca_coors_nm', 'z_latent_seq', 'token_type', 'cond_residue_type'))
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_rfd4_proteina.py#L500)

**Parameters:**

latent_dim (`int`, defaults to `32`) : Width of the per-token local latent.

n_residue_types (`int`, defaults to `147`) : Residue vocabulary size.

num_token_types (`int`, defaults to `7`) : Size of the token-type vocabulary: protein, RNA, DNA, atomized, unknown, glycan and non-canonical amino acid.

allowed_residue_types (`list[list[bool]]`) : `(num_token_types, n_residue_types)` table of the residue classes each token type may decode to.

nlayers (`int`, defaults to `12`) : Trunk layers in each of the encoder and decoder.

token_dim (`int`, defaults to `768`) : Token representation width.

nheads (`int`, defaults to `12`) : Attention heads per layer.

dim_cond (`int`, defaults to `128`) : Width of the (unused, zero) conditioning vector.

pair_repr_dim (`int`, defaults to `256`) : Pair width the trunk layers are constructed with. The released autoencoder uses no pair bias, so this holds no parameters.

token_type_emb_dim (`int`, defaults to `32`) : Token-type embedding width.

encoder_feats_seq, decoder_feats_seq (`list[str]`) : Features projected into the encoder and decoder inputs.

atom37_mask (`list[list[bool]]`) : Valid atom37 slots for each residue type, shaped `(n_residue_types, 37)`.

Paired molecular encoder and atom37 decoder.

Coordinates supplied to and returned by this model are in nanometers.

#### encode[[diffusers.RFD4ProteinaAutoencoder.encode]]

```python
encode(coords_nm: Tensor, padding_mask: Tensor, token_type: typing.Optional[torch.Tensor] = None, atom37_mask: typing.Optional[torch.Tensor] = None, conditioning: dict[str, typing.Any] | None = None, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_rfd4_proteina.py#L583)

**Parameters:**

coords_nm (`torch.Tensor`) : `(batch, tokens, 37, 3)` atom37 coordinates in nanometers.

padding_mask (`torch.Tensor`) : `(batch, tokens)`, true where a token is real rather than padding.

token_type (`torch.Tensor`, *optional*) : `(batch, tokens)` token-type codes.

atom37_mask (`torch.Tensor`, *optional*) : `(batch, tokens, 37)`, true where an atom slot is used.

conditioning (`dict[str, Any]`, *optional*) : The remaining topology and conditioning tensors, under the keys the preprocessor emits.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One generator per batch row for independent stochastic latent samples. Without one, sampling draws from the global stream.

return_dict (`bool`, defaults to `True`) : Return a typed output, or a tuple of its fields.

**Returns:** `RFD4ProteinaEncoderOutput` or `tuple`

Latent, mean and log-scale tensors shaped `(batch, tokens,
latent_dim)`.

Encode prepared coordinates and conditioning into sampled per-token local latents.

#### decode[[diffusers.RFD4ProteinaAutoencoder.decode]]

```python
decode(z_latent: Tensor, ca_coors_nm: Tensor, padding_mask: Tensor, token_type: typing.Optional[torch.Tensor] = None, atom37_mask: typing.Optional[torch.Tensor] = None, conditioning: dict[str, typing.Any] | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_rfd4_proteina.py#L624)

**Parameters:**

z_latent (`torch.Tensor`) : `(batch, tokens, latent_dim)` per-token local latents.

ca_coors_nm (`torch.Tensor`) : `(batch, tokens, 3)` backbone CA positions in nanometers.

padding_mask (`torch.Tensor`) : `(batch, tokens)`, true where a token is real rather than padding.

token_type (`torch.Tensor`, *optional*) : `(batch, tokens)` token-type codes.

atom37_mask (`torch.Tensor`, *optional*) : `(batch, tokens, 37)`, true where an atom slot is used.

conditioning (`dict[str, Any]`, *optional*) : The remaining topology and conditioning tensors, under the keys the preprocessor emits. The sequence condition is read from here.

return_dict (`bool`, defaults to `True`) : Return a typed output, or a tuple of its fields.

**Returns:** `RFD4ProteinaDecoderOutput` or `tuple`

Coordinates `(batch, tokens, 37, 3)` in nanometers, residue types
`(batch, tokens)`, emission mask `(batch, tokens)` and atom mask `(batch, tokens, 37)`. Non-emitted atom
coordinates are zero.

Decode per-token latents and backbone positions into atom37 structure.

#### forward[[diffusers.RFD4ProteinaAutoencoder.forward]]

```python
forward(z_latent: Tensor, ca_coors_nm: Tensor, padding_mask: Tensor, token_type: typing.Optional[torch.Tensor] = None, atom37_mask: typing.Optional[torch.Tensor] = None, conditioning: dict[str, typing.Any] | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_rfd4_proteina.py#L666)

**Parameters:**

z_latent (`torch.Tensor`) : `(batch, tokens, latent_dim)` per-token local latents.

ca_coors_nm (`torch.Tensor`) : `(batch, tokens, 3)` backbone CA positions in nanometers.

padding_mask (`torch.Tensor`) : `(batch, tokens)`, true where a token is real rather than padding.

token_type (`torch.Tensor`, *optional*) : `(batch, tokens)` token-type codes.

atom37_mask (`torch.Tensor`, *optional*) : `(batch, tokens, 37)`, true where an atom slot is used.

conditioning (`dict[str, Any]`, *optional*) : The remaining topology and conditioning tensors.

return_dict (`bool`, defaults to `True`) : Return a typed output, or a tuple of its fields.

**Returns:** `RFD4ProteinaDecoderOutput` or `tuple`

Decoded coordinates, residue types and atom/emit masks.

Decode per-token latents and backbone positions; equivalent to calling [decode()](/docs/diffusers/v0.42.0/en/api/models/rfd4_proteina_autoencoder#diffusers.RFD4ProteinaAutoencoder.decode).

## RFD4ProteinaEncoderOutput[[diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaEncoderOutput]]

#### diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaEncoderOutput[[diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaEncoderOutput]]

```python
diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaEncoderOutput(latent: Tensor, mean: typing.Optional[torch.Tensor] = None, log_scale: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_rfd4_proteina.py#L469)

**Parameters:**

latent (`torch.Tensor`) : Sample `mean + noise * exp(log_scale)`, zero outside emitted latent tokens.

mean (`torch.Tensor`) : Gaussian mean, zero outside emitted latent tokens.

log_scale (`torch.Tensor`) : Log standard deviation, zero outside emitted latent tokens.

Sampled latent and Gaussian parameters, each shaped `(batch, tokens, latent_dim)`.

## RFD4ProteinaDecoderOutput[[diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaDecoderOutput]]

#### diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaDecoderOutput[[diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaDecoderOutput]]

```python
diffusers.models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaDecoderOutput(coordinates: Tensor, residue_type: typing.Optional[torch.Tensor] = None, emit_mask: typing.Optional[torch.Tensor] = None, atom_mask: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/models/autoencoders/autoencoder_rfd4_proteina.py#L484)

**Parameters:**

coordinates (`torch.Tensor`) : Coordinates `(batch, tokens, 37, 3)` in nanometers; non-emitted atoms are zero.

residue_type (`torch.Tensor`) : Residue vocabulary indices shaped `(batch, tokens)`.

emit_mask (`torch.Tensor`) : Boolean emitted-token mask shaped `(batch, tokens)`.

atom_mask (`torch.Tensor`) : Boolean valid-atom mask shaped `(batch, tokens, 37)`.

Decoded molecular coordinates and discrete atom37 topology.

### FluxControlNetModel
https://huggingface.co/docs/diffusers/v0.42.0/api/models/controlnet_flux.md
