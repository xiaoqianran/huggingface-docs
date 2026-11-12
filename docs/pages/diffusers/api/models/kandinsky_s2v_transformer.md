# KandinskyS2VTransformer3DModel

The Diffusion Transformer (DiT) of Kandinsky S2V 1.0, conditioned on a reference
frame and a speech track.

## KandinskyS2VTransformer3DModel[[diffusers.KandinskyS2VTransformer3DModel]]

#### diffusers.KandinskyS2VTransformer3DModel[[diffusers.KandinskyS2VTransformer3DModel]]

```python
diffusers.KandinskyS2VTransformer3DModel(in_visual_dim: int = 16, out_visual_dim: int = 16, in_text_dim: int = 3584, in_text_dim2: int = 768, in_audio_dim: int = 3584, time_dim: int = 512, patch_size: tuple[int, int, int] = (1, 2, 2), frame_buckets: tuple[int, int, int] = (2, 4, 24), enable_framepack: bool = True, index_rope_ref_image: int = 40, patch_proj1x_coef: tuple[int, int, int] = (1, 1, 1), patch_proj2x_coef: tuple[int, int, int] = (2, 2, 2), patch_proj4x_coef: tuple[int, int, int] = (4, 4, 4), model_dim: int = 1792, ff_dim: int = 7168, num_text_blocks: int = 2, num_audio_blocks: int = 2, num_visual_blocks: int = 32, axes_dims: tuple[int, int, int] = (16, 24, 24), audio_inject_layers: tuple[int, ...] = (), text_rope_max_pos: int = 4096)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_kandinsky_s2v.py#L487)

Speech-to-video diffusion transformer.

Parameter names match the training checkpoints, so a converted *.safetensors* loads with no key remapping.
Inference is batch size 1: the sequence is packed and the speech cross-attention is frame-aligned, so *forward*
takes a leading batch axis of 1 and squeezes it.

The visual stream is one packed sequence `[memory | ref | visual]`:

- `memory` — FramePack context from the previous chunk at 4x / 2x / 1x temporal downsampling, placed at negative
  temporal RoPE positions;
- `ref` — the reference frame, pinned to temporal RoPE index `index_rope_ref_image`;
- `visual` — the noisy latent being denoised.

`memory` and `ref` are clean, so they are modulated with `t = 0` while `visual` uses the real timestep; the
modulation holds both rows and an index selects per token.

#### forward[[diffusers.KandinskyS2VTransformer3DModel.forward]]

```python
forward(hidden_states: Tensor, encoder_hidden_states: Tensor, audio_hidden_states: Tensor, reference_hidden_states: Tensor, memory_hidden_states: tuple[Tensor, Tensor, Tensor], timestep: Tensor, pooled_projections: Tensor, visual_rope_pos: list[Tensor], reference_rope_pos: list[Tensor], memory_rope_pos: tuple[list[Tensor], list[Tensor], list[Tensor]], text_rope_pos: Tensor, audio_rope_pos: Tensor, scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0), return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/models/transformers/transformer_kandinsky_s2v.py#L632)

**Parameters:**

hidden_states : noisy latent, `(1, T, H, W, C)`.

encoder_hidden_states : text embeddings, `(1, S_text, in_text_dim)`.

audio_hidden_states : speech features, `(1, S_audio, in_audio_dim)`.

reference_hidden_states : reference frame latent, `(1, 1, H, W, C)`.

memory_hidden_states : FramePack memory at 4x / 2x / 1x, each `(1, T_i, H_i, W_i, C)`.

timestep : current timestep, `(1,)`.

pooled_projections : pooled CLIP text embedding, `(1, in_text_dim2)`.

visual_rope_pos : per-axis RoPE positions of the noisy latent.

reference_rope_pos : per-axis RoPE positions of the reference frame.

memory_rope_pos : per-axis RoPE positions of each FramePack memory scale.

text_rope_pos : 1-D RoPE positions of the text tokens.

audio_rope_pos : latent frame each speech token belongs to.

scale_factor : RoPE scale per axis, set by the resolution bucket.

return_dict : return a `Transformer2DModelOutput` instead of a tuple.

**Returns:** [*~models.modeling_outputs.Transformer2DModelOutput*] or *tuple*

the predicted velocity, shaped like
`hidden_states`.

Denoise one chunk.

Batch size must be 1: the sequence is packed and the speech cross-attention is frame-aligned.

### CogView4Transformer2DModel
https://huggingface.co/docs/diffusers/v0.43.0/api/models/cogview4_transformer2d.md
