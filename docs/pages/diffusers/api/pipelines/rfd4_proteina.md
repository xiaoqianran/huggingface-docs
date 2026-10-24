# RFD4-Proteína

RFD4-Proteína is a flow-matching model from the Baker Lab and NVIDIA for biomolecular design. It generates protein
sequences and all-atom structures together, conditioned on the molecular context to preserve: a binding target, an
enzyme motif, a ligand, hotspots or secondary structure. A paired transformer and autoencoder are integrated by
two [RFD4ProteinaScheduler](/docs/diffusers/v0.42.0/en/api/schedulers/rfd4_proteina#diffusers.RFD4ProteinaScheduler), one for the backbone and one for the local latents.

The pipeline consumes prepared feature tensors. Turning an annotated CIF file into those tensors, and the generated
tensors back into a CIF file, is domain-specific preprocessing that lives in the
[`rfd4-proteina-prep`](https://huggingface.co/nvidia/RFD4-Proteina-prep) package.

## Usage example

Install the preparation package at a fixed revision. It decides the feature tensors the model sees, so an unpinned
install changes generated structures when the package is updated.

```bash
pip install diffusers git+https://huggingface.co/nvidia/RFD4-Proteina-prep@c9959efdebd75ac7fe9129fbbf0e291a6e0487d5
```

```py
import torch
from huggingface_hub import hf_hub_download
from diffusers import RFD4ProteinaPipeline
from rfd4_proteina_prep import prepare_cif, write_cif

pipe = RFD4ProteinaPipeline.from_pretrained("nvidia/RFD4-Proteina-Diffusers", dtype=torch.bfloat16).to("cuda")

# An annotated CIF specifies what to hold fixed and what to design: here a PD-L1 target and a 50-120 residue binder.
input_cif = hf_hub_download("nvidia/RFD4-Proteina-Diffusers", "examples/protein_binder/input.cif")
features, atoms = prepare_cif(input_cif, seed=42, device=pipe.device)

# The masks and molecule tensors the pipeline always needs are arguments; the annotation tensors go in one mapping.
required = ["padding_mask", "is_in_context_conditioning", "supports_local_latent", "token_type", "atom37_mask"]
conditioning = {name: value for name, value in features.items() if name not in required}

output = pipe(
    padding_mask=features["padding_mask"],
    is_in_context_conditioning=features["is_in_context_conditioning"],
    supports_local_latent=features["supports_local_latent"],
    token_type=features["token_type"],
    atom37_mask=features["atom37_mask"],
    conditioning_features=conditioning,
    generator=torch.Generator("cuda").manual_seed(42),
)
write_cif(output, features, atoms, "design.cif")
```

## Tips

* The model repository ships an annotated input and a generated design for each task under `examples/`: protein binder,
  antibody, VHH, enzyme, motif scaffolding, small-molecule binder, unconditional, inverse folding and partial diffusion.
* Pass `partial_t_bb_ca` and `partial_t_local_latents` with start times in `[0, 1)` to redesign an input from a noised
  start, or `inverse_folding=True` with both start times at `0.0` to keep the backbone and design the sequence. Both
  encode the input, so they need clean `coords_nm` and reject coordinates that are not finite, which are usually
  unresolved residues.
* The number of steps is the length of the `recycle_depths` pipeline config, 60 for the released recipe (180 trunk
  passes). Change the schedule by passing a different `recycle_depths` list when constructing the pipeline.
* To generate several designs of each input, set `num_designs_per_prompt` and pass one distinct generator per
  design. The output has `batch * num_designs_per_prompt` rows, and row `r` is design `r % num_designs_per_prompt` of
  input row `r // num_designs_per_prompt`.

  ```py
  generators = [torch.Generator("cuda").manual_seed(seed) for seed in range(8)]
  output = pipe(
      padding_mask=features["padding_mask"],
      is_in_context_conditioning=features["is_in_context_conditioning"],
      supports_local_latent=features["supports_local_latent"],
      token_type=features["token_type"],
      atom37_mask=features["atom37_mask"],
      conditioning_features=conditioning,
      num_designs_per_prompt=8,
      generator=generators,
  )
  ```

  `write_cif` reads the features of the row it writes, so repeat them the same way, with `repeat_interleave` rather
  than `repeat`:

  ```py
  repeated = {k: v.repeat_interleave(8, dim=0) if torch.is_tensor(v) else v for k, v in features.items()}
  for row in range(8):
      write_cif(output, repeated, atoms, f"design_{row}.cif", row=row)
  ```

* Load in `torch.bfloat16` for speed. The sampler state and the feature embeddings stay in float32. A different dtype,
  kernel or batch size can end in a different design for the same seed.
* A generated structure is not a validated design. Refold and score it for the intended task.

## LoRA

The released model is a rank-32 LoRA fine-tune of the pretrained base, merged into the weights. The base and the adapter
are on the `pretrain-base` revision of the model repository, and the pipeline loads the adapter with
`load_lora_weights` through `RFD4ProteinaLoraLoaderMixin`:

```py
pipe = RFD4ProteinaPipeline.from_pretrained(
    "nvidia/RFD4-Proteina-Diffusers", revision="pretrain-base", dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(
    "nvidia/RFD4-Proteina-Diffusers",
    revision="pretrain-base",
    subfolder="lora",
    weight_name="pytorch_lora_weights.safetensors",
)
```

## RFD4ProteinaPipeline[[diffusers.RFD4ProteinaPipeline]]

#### diffusers.RFD4ProteinaPipeline[[diffusers.RFD4ProteinaPipeline]]

```python
diffusers.RFD4ProteinaPipeline(transformer: RFD4ProteinaTransformer, vae: RFD4ProteinaAutoencoder, bb_ca_scheduler: RFD4ProteinaScheduler, local_latents_scheduler: RFD4ProteinaScheduler, recycle_depths: list[int] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/rfd4_proteina/pipeline_rfd4_proteina.py#L68)

**Parameters:**

transformer ([RFD4ProteinaTransformer](/docs/diffusers/v0.42.0/en/api/models/rfd4_proteina_transformer#diffusers.RFD4ProteinaTransformer)) : Paired backbone and local-latent velocity denoiser.

vae ([RFD4ProteinaAutoencoder](/docs/diffusers/v0.42.0/en/api/models/rfd4_proteina_autoencoder#diffusers.RFD4ProteinaAutoencoder)) : The autoencoder paired with the denoiser checkpoint.

bb_ca_scheduler ([RFD4ProteinaScheduler](/docs/diffusers/v0.42.0/en/api/schedulers/rfd4_proteina#diffusers.RFD4ProteinaScheduler)) : Integrates the backbone coordinates.

local_latents_scheduler ([RFD4ProteinaScheduler](/docs/diffusers/v0.42.0/en/api/schedulers/rfd4_proteina#diffusers.RFD4ProteinaScheduler)) : Integrates the local latents.

recycle_depths (`list[int]`, *optional*) : Trunk recycling passes for each denoising step. Its length is the default number of inference steps. Defaults to the released schedule of 60 steps and 180 passes.

Generate molecular structures and sequences from prepared molecular tensors.

The pipeline consumes the feature tensors produced by the RFD4 input preparation (see the `rfd4-proteina-prep`
package for annotated CIF files) and returns decoded atom37 coordinates and residue identities. Coordinates are in
nanometers.

#### __call__[[diffusers.RFD4ProteinaPipeline.__call__]]

```python
__call__(padding_mask: Tensor, is_in_context_conditioning: Tensor, supports_local_latent: Tensor, token_type: typing.Optional[torch.Tensor] = None, atom37_mask: typing.Optional[torch.Tensor] = None, coords_nm: typing.Optional[torch.Tensor] = None, conditioning_features: dict[str, typing.Any] | None = None, num_designs_per_prompt: int = 1, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, partial_t_bb_ca: float | None = None, partial_t_local_latents: float | None = None, inverse_folding: bool = False, latents_bb_ca: typing.Optional[torch.Tensor] = None, latents_local_latents: typing.Optional[torch.Tensor] = None, attention_kwargs: dict[str, typing.Any] | None = None, callback_on_step_end: collections.abc.Callable[[diffusers.pipelines.pipeline_utils.DiffusionPipeline, int, torch.Tensor, dict], dict | None] | None = None, callback_on_step_end_tensor_inputs: list = ['bb_ca', 'local_latents'], output_type: str = 'pt', return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/rfd4_proteina/pipeline_rfd4_proteina.py#L287)

**Parameters:**

padding_mask (`torch.Tensor`) : Boolean `(batch, tokens)`, true where a token is real rather than padding.

is_in_context_conditioning (`torch.Tensor`) : Boolean `(batch, tokens)`, true for tokens that are given as context and left unchanged.

supports_local_latent (`torch.Tensor`) : Boolean `(batch, tokens)`, true for tokens that carry a local latent. All three masks share one shape and device.

token_type (`torch.Tensor`, *optional*) : `(batch, tokens)` token-type codes.

atom37_mask (`torch.Tensor`, *optional*) : `(batch, tokens, 37)`, true where an atom slot is used.

coords_nm (`torch.Tensor`, *optional*) : Clean `(batch, tokens, 37, 3)` atom37 coordinates in nanometers, needed by partial diffusion and inverse folding.

conditioning_features (`dict[str, Any]`, *optional*) : The remaining topology and conditioning tensors under the keys the preprocessor emits. It must hold the sequence condition, `condition_sequence_1_residue`, and `molecule_iid`. Which of the rest are read is fixed by the checkpoint's feature configuration. It is one mapping because the preprocessor emits about 30 optional tensors under fixed names, and a signature with that many arguments would tie the pipeline to the preprocessor's naming.

num_designs_per_prompt (`int`, defaults to `1`) : Number of designs to generate for each input row. The inputs are repeated along the batch axis, so the output has `batch * num_designs_per_prompt` rows and row `r` is design `r % num_designs_per_prompt` of input row `r // num_designs_per_prompt`.

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : One distinct generator per output row, for reproducible designs. The generators advance and the global PyTorch random stream is left untouched. Omit it to draw from the global stream.

partial_t_bb_ca (`float`, *optional*) : Start time in [0, 1) for the backbone, to start from a noised version of the input instead of pure noise. Give it together with `partial_t_local_latents`; requires `coords_nm` and `atom37_mask`.

partial_t_local_latents (`float`, *optional*) : Start time in [0, 1) for the local latents; see `partial_t_bb_ca`.

inverse_folding (`bool`, defaults to `False`) : Preserve the input backbone and design the sequence. Requires zero for both partial start times.

latents_bb_ca (`torch.Tensor`, *optional*) : Pre-generated starting backbone state `(batch * num_designs_per_prompt, tokens, 3)`, to replace the one this call would otherwise sample. The schedule is unchanged. Give it together with `latents_local_latents`.

latents_local_latents (`torch.Tensor`, *optional*) : Pre-generated starting local latents `(batch * num_designs_per_prompt, tokens, latent_dim)`.

attention_kwargs (`dict`, *optional*) : Passed to the transformer; `scale` sets the LoRA scale.

callback_on_step_end (`Callable`, *optional*) : Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` after each step. `callback_kwargs` holds the tensors named in `callback_on_step_end_tensor_inputs`; a returned dict may replace `"bb_ca"` and `"local_latents"`.

output_type (`str`, defaults to `"pt"`) : `"pt"` returns torch tensors and `"np"` returns numpy arrays.

callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["bb_ca", "local_latents"]`) : Tensors passed to the callback: `"bb_ca"`, `"local_latents"`, `"last_clean_bb_ca"`, `"last_clean_local_latents"`, `"sequence_logits"`, `"insertion_logits"`.

return_dict (`bool`, defaults to `True`) : Return a typed output, or a tuple of its non-None fields.

**Returns:** `RFD4ProteinaPipelineOutput` or `tuple`

Decoded atom37 tensors (`coords_nm` in nanometers, `residue_type`, `emit_mask`, `atom_mask`), the final
noisy state, the last clean estimate, and the last sequence and insertion logits.

Generate structures and sequences from prepared molecular tensors.

Examples:
```py
>>> import torch
>>> from diffusers import RFD4ProteinaPipeline
>>> from rfd4_proteina_prep import prepare_cif, write_cif

>>> pipe = RFD4ProteinaPipeline.from_pretrained("nvidia/RFD4-Proteina-Diffusers", dtype=torch.bfloat16)
>>> pipe.to("cuda")

>>> features, atoms = prepare_cif("input.cif", seed=42, device=pipe.device)
>>> required = ["padding_mask", "is_in_context_conditioning", "supports_local_latent", "token_type"]
>>> required += ["atom37_mask"]
>>> output = pipe(
...     padding_mask=features["padding_mask"],
...     is_in_context_conditioning=features["is_in_context_conditioning"],
...     supports_local_latent=features["supports_local_latent"],
...     token_type=features["token_type"],
...     atom37_mask=features["atom37_mask"],
...     conditioning_features={name: value for name, value in features.items() if name not in required},
...     generator=torch.Generator("cuda").manual_seed(42),
... )
>>> write_cif(output, features, atoms, "design.cif")
```

#### check_inputs[[diffusers.RFD4ProteinaPipeline.check_inputs]]

```python
check_inputs(padding_mask, is_in_context_conditioning, supports_local_latent, atom37_mask, coords_nm, conditioning_features, num_designs_per_prompt, generator, partial_t, inverse_folding, latents, callback_on_step_end_tensor_inputs, output_type)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/rfd4_proteina/pipeline_rfd4_proteina.py#L130)

Validate molecular masks, per-design generators, partial diffusion and callbacks.

#### noise_backbone[[diffusers.RFD4ProteinaPipeline.noise_backbone]]

```python
noise_backbone(target_backbone, bb_mask, conditioning, generators, index)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/rfd4_proteina/pipeline_rfd4_proteina.py#L278)

Noise the preserved backbone to the scheduled time of step `index`, for inverse folding.

#### prepare_latents[[diffusers.RFD4ProteinaPipeline.prepare_latents]]

```python
prepare_latents(bb_mask: Tensor, latent_mask: Tensor, conditioning: dict, generators: list[torch.Generator] | None, clean_bb: typing.Optional[torch.Tensor] = None, clean_latents: typing.Optional[torch.Tensor] = None, start_times: tuple[float, float] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/rfd4_proteina/pipeline_rfd4_proteina.py#L243)

Draw the starting state: pure noise, or the clean input noised to `start_times` (backbone, latents).
`clean_latents` can be left out when the latent start time is zero.

The draws follow the native sampler's order on every row generator: for a full generation the backbone noise
and its stochastic offsets come before the latent noise, while partial diffusion draws both noises before the
offsets.

## RFD4ProteinaPipelineOutput[[diffusers.pipelines.rfd4_proteina.pipeline_output.RFD4ProteinaPipelineOutput]]

#### diffusers.pipelines.rfd4_proteina.pipeline_output.RFD4ProteinaPipelineOutput[[diffusers.pipelines.rfd4_proteina.pipeline_output.RFD4ProteinaPipelineOutput]]

```python
diffusers.pipelines.rfd4_proteina.pipeline_output.RFD4ProteinaPipelineOutput(coords_nm: Tensor, residue_type: Tensor, emit_mask: Tensor, atom_mask: Tensor, bb_ca: Tensor, local_latents: Tensor, clean_bb_ca: Tensor, clean_local_latents: Tensor, sequence_logits: Tensor, insertion_logits: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/pipelines/rfd4_proteina/pipeline_output.py#L12)

**Parameters:**

coords_nm (`torch.Tensor`) : Decoded atom37 coordinates `(B, N, 37, 3)`.

residue_type (`torch.Tensor`) : Residue vocabulary indices `(B, N)`; -1 where no residue is emitted.

emit_mask (`torch.Tensor`) : Boolean mask `(B, N)` of the tokens that were designed.

atom_mask (`torch.Tensor`) : Boolean mask `(B, N, 37)` of the valid atoms.

bb_ca (`torch.Tensor`) : Final backbone state `(B, N, 3)` in nanometers.

local_latents (`torch.Tensor`) : Final local-latent state `(B, N, D)`.

clean_bb_ca (`torch.Tensor`) : Backbone prediction of the clean state at the last denoising step.

clean_local_latents (`torch.Tensor`) : Local-latent prediction of the clean state at the last denoising step.

sequence_logits (`torch.Tensor`) : Final per-token residue logits `(B, N, 147)`.

insertion_logits (`torch.Tensor`) : Final insertion predictions `(B, N, N)` for unindexed motifs.

Molecular inference results. Coordinates are in nanometers.

### Kandinsky 5.0 Video
https://huggingface.co/docs/diffusers/v0.42.0/api/pipelines/kandinsky5_video.md
