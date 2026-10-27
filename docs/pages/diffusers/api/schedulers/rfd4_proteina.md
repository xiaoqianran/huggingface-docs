# RFD4ProteinaScheduler

`RFD4ProteinaScheduler` is the temporal-score-rescaled flow sampler of RFD4-Proteína for one modality. The pipeline uses
two instances, one for the backbone (`bb_ca`, coordinates in nanometers) and one for the per-token local latents
(`local_latents`), each with its own time schedule, noise schedule and rescaling. See [RFD4ProteinaPipeline](/docs/diffusers/v0.42.1/en/api/pipelines/rfd4_proteina#diffusers.RFD4ProteinaPipeline) for the
sampling loop.

## RFD4ProteinaScheduler[[diffusers.RFD4ProteinaScheduler]]

#### diffusers.RFD4ProteinaScheduler[[diffusers.RFD4ProteinaScheduler]]

```python
diffusers.RFD4ProteinaScheduler(schedule_mode: str, noise_mode: str, tsr_k: float, tsr_sigma: float, tail_steps: int = 10, tail_step_multiplier: int = 2, t_lim_ode: float = 0.95, t_lim_ode_below: float = 0.02)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/schedulers/scheduling_rfd4_proteina.py#L76)

**Parameters:**

schedule_mode (`str`) : Spacing of the times between noise and data: `"log"` or `"power"`.

noise_mode (`str`) : Shape of the noise injected in the stochastic window: `"1/t"` or `"tan"`.

tsr_k (`float`) : How low the sampling temperature goes; higher is lower.

tsr_sigma (`float`) : How fast the low temperature is reached; higher is faster.

tail_steps (`int`, defaults to `10`) : Number of final intervals that are finer than the rest by `tail_step_multiplier`.

tail_step_multiplier (`int`, defaults to `2`) : How much finer the final intervals are.

t_lim_ode (`float`, defaults to `0.95`) : Above this time the step is deterministic.

t_lim_ode_below (`float`, defaults to `0.02`) : Below this time the step is deterministic.

Temporal-score-rescaled flow sampler for one modality of RFD4-Proteína.

The pipeline integrates two modalities, the backbone and the per-token local latents, with one scheduler each. They
differ in their time schedule, noise schedule and rescaling.

Call `set_timesteps` before sampling. Times increase from noise toward clean data. A scheduler tracks the current
interval and should not be shared by concurrent generations.

#### set_timesteps[[diffusers.RFD4ProteinaScheduler.set_timesteps]]

```python
set_timesteps(num_inference_steps: int, device: typing.Union[str, torch.device, NoneType] = None, start_time: float | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/schedulers/scheduling_rfd4_proteina.py#L133)

**Parameters:**

num_inference_steps (`int`) : Number of integration intervals.

device (`str` or `torch.device`, *optional*) : Device for `timesteps`. The interval lengths and noise coefficients stay on the CPU.

start_time (`float`, *optional*) : Partial-diffusion start time in `[0, 1)`. Omit for full generation.

Resolve the schedule and reset the integration cursor.

#### add_noise[[diffusers.RFD4ProteinaScheduler.add_noise]]

```python
add_noise(original_samples: Tensor, noise: Tensor, timesteps: typing.Union[float, torch.Tensor])
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/schedulers/scheduling_rfd4_proteina.py#L194)

**Parameters:**

original_samples (`torch.Tensor`) : Clean state `(batch, tokens, D)`.

noise (`torch.Tensor`) : Noise of the same shape, drawn by the caller.

timesteps (`float` or `torch.Tensor`) : Time in `[0, 1)`, a scalar or `(batch,)`.

Sample the forward marginal `(1 - t) * noise + t * clean`.

#### step[[diffusers.RFD4ProteinaScheduler.step]]

```python
step(model_output: Tensor, timestep: Tensor, sample: Tensor, emit_mask: Tensor, generator: typing.Union[torch.Generator, list[torch.Generator], NoneType] = None, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/schedulers/scheduling_rfd4_proteina.py#L212)

**Parameters:**

model_output (`torch.Tensor`) : Predicted velocity, shaped like `sample`.

timestep (`torch.Tensor`) : The next element of `timesteps`. Out-of-order steps are rejected.

sample (`torch.Tensor`) : Current state `(B, N, D)`.

emit_mask (`torch.Tensor`) : Boolean `(B, N)`, true for the tokens that are updated (not context or padding).

generator (`torch.Generator` or `list[torch.Generator]`, *optional*) : A generator for a single row or distinct generators for each batch row. Omitting them uses the global stream.

return_dict (`bool`, defaults to `True`) : Return `RFD4ProteinaSchedulerOutput`; otherwise a one-item tuple holding the state.

**Returns:** `RFD4ProteinaSchedulerOutput` or `tuple`

The updated state.

Advance one integration interval and update `step_index`.

## RFD4ProteinaSchedulerOutput[[diffusers.schedulers.scheduling_rfd4_proteina.RFD4ProteinaSchedulerOutput]]

#### diffusers.schedulers.scheduling_rfd4_proteina.RFD4ProteinaSchedulerOutput[[diffusers.schedulers.scheduling_rfd4_proteina.RFD4ProteinaSchedulerOutput]]

```python
diffusers.schedulers.scheduling_rfd4_proteina.RFD4ProteinaSchedulerOutput(prev_sample: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/schedulers/scheduling_rfd4_proteina.py#L19)

**Parameters:**

prev_sample (`torch.Tensor`) : The state after the step, shaped `(B, N, D)`: `D` is 3 for the backbone (coordinates in nanometers) and the latent width for the local latents.

Updated state of one modality.

### EDMDPMSolverMultistepScheduler
https://huggingface.co/docs/diffusers/v0.42.1/api/schedulers/edm_multistep_dpm_solver.md
