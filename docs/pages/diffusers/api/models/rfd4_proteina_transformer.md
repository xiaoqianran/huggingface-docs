# RFD4ProteinaTransformer

The denoiser of [RFD4-Proteína](../pipelines/rfd4_proteina). It predicts velocities for the two modalities the
sampler integrates — backbone CA positions and per-token local latents — from prepared molecular tensors. Inputs are
named tensors plus a conditioning mapping whose contents the checkpoint configuration selects; structure parsing is
outside this class.

```py
from diffusers import RFD4ProteinaTransformer

transformer = RFD4ProteinaTransformer.from_pretrained(
    "nvidia/RFD4-Proteina-Diffusers", subfolder="transformer"
)
```

## RFD4ProteinaTransformer[[diffusers.RFD4ProteinaTransformer]]

#### diffusers.RFD4ProteinaTransformer[[diffusers.RFD4ProteinaTransformer]]

```python
diffusers.RFD4ProteinaTransformer(nlayers: int = 14, token_dim: int = 768, nheads: int = 12, dim_cond: int = 256, pair_repr_dim: int = 384, latent_dim: int = 32, token_type_emb_dim: int = 32, cath_emb_dim: int = 32, t_emb_dim: int = 256, seq_sep_dim: int = 64, xt_pair_dist_dim: int = 30, xt_pair_dist_min: float = 0.1, xt_pair_dist_max: float = 3.0, feats_seq: tuple = ('xt_bb_ca', 'xt_local_latents', 'token_type', 'sequence_probabilities', 'cond_a37coors_nm', 'cond_a37coors_internal_nm', 'rasa_binned', 'secondary_structure', 'hbond_modernized', 'pistack', 'pication', 'saltbridge', 'hydrophobic', 'metal_coord', 'na_ss_state', 'hotspot', 'atom_charge', 'atomic_element', 'coord_group_seq', 'chirality', 'chain_centroid_xyz'), feats_cond_seq: tuple = ('time_emb_bb_ca', 'time_emb_local_latents', 'cath_code', 'recycle_remaining'), feats_pair_repr: tuple = ('xt_bb_ca_pair_dists', 'bond_mask', 'bond_order', 'is_aromatic', 'index_group_pair', 'molecule_idx_pair', 'molecule_entity_pair', 'token_bond_path_length', 'canonical_polymer_ordering', 'na_ss_pair'), feats_pair_cond: tuple = ('time_emb_bb_ca', 'time_emb_local_latents'), update_pair_repr_every_n: int = 2, tri_mult_c: int = 256, num_sequence_classes: int = 147, num_token_types: int = 7, recycle_conditioning_dim: int = 1)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_rfd4_proteina.py#L906)

**Parameters:**

nlayers (`int`, defaults to `14`) : Number of trunk layers.

token_dim (`int`, defaults to `768`) : Token representation width.

nheads (`int`, defaults to `12`) : Attention heads per layer.

dim_cond (`int`, defaults to `256`) : Width of the conditioning vector (time, recycle index, CATH label).

pair_repr_dim (`int`, defaults to `384`) : Pair representation width.

latent_dim (`int`, defaults to `32`) : Width of the per-token local latent.

token_type_emb_dim (`int`, defaults to `32`) : Token-type embedding width.

cath_emb_dim (`int`, defaults to `32`) : Width of each CATH class/architecture/topology embedding.

t_emb_dim (`int`, defaults to `256`) : Sinusoidal time embedding width.

seq_sep_dim (`int`, defaults to `64`) : Number of bond-path-length bins.

xt_pair_dist_dim (`int`, defaults to `30`) : Number of backbone pair-distance bins.

xt_pair_dist_min (`float`, defaults to `0.1`), xt_pair_dist_max (`float`, defaults to `3.0`) : Pair-distance range in nanometers.

feats_seq, feats_cond_seq, feats_pair_repr, feats_pair_cond (`list[str]`) : Names of the features projected into the token, conditioning, pair and pair-conditioning inputs.

update_pair_repr_every_n (`int`, defaults to `2`) : How often the pair representation is updated from the tokens between layers.

tri_mult_c (`int`, defaults to `256`) : Hidden width of the triangle multiplication in those updates.

num_token_types (`int`, defaults to `7`) : Size of the token-type vocabulary: protein, RNA, DNA, atomized, unknown, glycan and non-canonical amino acid.

recycle_conditioning_dim (`int`, defaults to `1`) : Width of the `recycle_remaining` one-hot, when that feature is in `feats_cond_seq`.

num_sequence_classes (`int`, defaults to `147`) : Size of the residue vocabulary the sequence readout predicts.

RFD4 velocity denoiser with paired backbone and local-latent modalities.

The model runs in its own dtype: load it with `dtype=torch.bfloat16` for reduced-precision inference. The sampling
state passed to `forward` stays in float32; feature embeddings are computed in float32 and cast at the input
projection.

#### forward[[diffusers.RFD4ProteinaTransformer.forward]]

```python
forward(bb_ca: Tensor, local_latents: Tensor, bb_ca_time: Tensor, local_latents_time: Tensor, padding_mask: Tensor, token_type: typing.Optional[torch.Tensor] = None, atom37_mask: typing.Optional[torch.Tensor] = None, conditioning: dict[str, typing.Any] | None = None, recycle_depth: int = 0, attention_kwargs: dict[str, typing.Any] | None = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_rfd4_proteina.py#L1062)

**Parameters:**

bb_ca (`torch.Tensor`) : Noisy backbone coordinates `(batch, tokens, 3)` in nanometers.

local_latents (`torch.Tensor`) : Noisy local latents `(batch, tokens, latent_dim)`.

bb_ca_time (`torch.Tensor`) : `(batch,)` backbone time in [0, 1].

local_latents_time (`torch.Tensor`) : `(batch,)` local-latent time in [0, 1].

padding_mask (`torch.Tensor`) : `(batch, tokens)`, true where a token is real rather than padding.

token_type (`torch.Tensor`, *optional*) : `(batch, tokens)` token-type codes.

atom37_mask (`torch.Tensor`, *optional*) : `(batch, tokens, 37)`, true where an atom slot is used.

conditioning (`dict[str, Any]`, *optional*) : The remaining topology and conditioning tensors, under the keys the preprocessor emits. Which of them are read is fixed by the checkpoint's `feats_seq`, `feats_cond_seq`, `feats_pair_repr` and `feats_pair_cond`; a key a configured feature asks for but that is absent leaves that feature unconditioned rather than raising.

recycle_depth (`int`, defaults to `0`) : Number of trunk passes; zero still executes one pass.

attention_kwargs (`dict`, *optional*) : Only `scale` is read, as the LoRA scale for this call.

return_dict (`bool`, defaults to `True`) : Return a typed output, or a tuple of non-None fields.

**Returns:** `RFD4ProteinaTransformerOutput` or `tuple`

Backbone/local-latent velocities and optional
sequence/insertion logits, before scheduler integration.

Predict velocities from prepared molecular tensors without advancing the sampler.

## RFD4ProteinaTransformerOutput[[diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaTransformerOutput]]

#### diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaTransformerOutput[[diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaTransformerOutput]]

```python
diffusers.models.transformers.transformer_rfd4_proteina.RFD4ProteinaTransformerOutput(backbone: Tensor, local_latents: typing.Optional[torch.Tensor] = None, sequence_logits: typing.Optional[torch.Tensor] = None, insertion_logits: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_rfd4_proteina.py#L890)

**Parameters:**

backbone (`torch.Tensor`) : Backbone velocity shaped `(batch, tokens, 3)`, in nanometers per unit time.

local_latents (`torch.Tensor`) : Latent velocity shaped `(batch, tokens, latent_dim)`.

sequence_logits (`torch.Tensor`, *optional*) : Residue logits shaped `(batch, tokens, 147)`.

insertion_logits (`torch.Tensor`, *optional*) : Insertion assignment logits shaped `(batch, tokens, tokens)`.

Denoiser predictions before scheduler integration.

### AutoencoderRAE
https://huggingface.co/docs/diffusers/v0.42.1/api/models/autoencoder_rae.md
