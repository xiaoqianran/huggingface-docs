# PiflowScheduler

`PiflowScheduler` is the few-step scheduler of the distilled Kandinsky 6 checkpoints, both the
text/image-to-video-and-audio model and the video super-resolution model. It implements
[π-Flow](https://huggingface.co/papers/2510.14974): the transformer predicts `n_grid` denoised estimates per latent
channel at a small number of grid points, and the scheduler integrates a network-free policy between them.

The reference implementation can be found at [Lakonik/LakonLab](https://github.com/Lakonik/LakonLab).

## PiflowScheduler[[diffusers.PiflowScheduler]]

#### diffusers.PiflowScheduler[[diffusers.PiflowScheduler]]

```python
diffusers.PiflowScheduler(num_train_timesteps: int = 1000, shift: float = 5.0, n_grid: int = 10, eps: float = 1e-06, final_step_size_scale: float = 0.5, num_policy_substeps: int = 128)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/schedulers/scheduling_piflow.py#L144)

**Parameters:**

num_train_timesteps (`int`, *optional*, defaults to 1000) : Number of training diffusion steps.

shift (`float`, *optional*, defaults to 5.0) : Flow-matching timestep shift.

n_grid (`int`, *optional*, defaults to 10) : Number of predictions in the widened model output.

eps (`float`, *optional*, defaults to 1e-6) : Minimum timestep and policy denominator.

final_step_size_scale (`float`, *optional*, defaults to 0.5) : Relative size of the final raw-timestep segment.

num_policy_substeps (`int`, *optional*, defaults to 128) : Maximum policy integration substeps per raw-timestep unit.

Few-step PiFlow scheduler for widened-output diffusion transformers.

PiFlow evaluates the denoising model at a small number of grid points and integrates a network-free policy between
those evaluations. The scheduler is intended for distilled Kandinsky 6 checkpoints, including the main video/audio
model and the video super-resolution model. Their model output contains `n_grid` predictions per sample channel.

This scheduler inherits from [SchedulerMixin](/docs/diffusers/v0.42.0/en/api/schedulers/overview#diffusers.SchedulerMixin) and [ConfigMixin](/docs/diffusers/v0.42.0/en/api/configuration#diffusers.ConfigMixin). Check the superclass documentation for the
generic methods implemented for all schedulers (loading, saving, etc.).

#### set_timesteps[[diffusers.PiflowScheduler.set_timesteps]]

```python
set_timesteps(num_inference_steps: int | None = None, device: typing.Union[str, torch.device, NoneType] = None, sigmas: list[float] | None = None, mu: float | None = None, timesteps: list[float] | None = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/schedulers/scheduling_piflow.py#L223)

**Parameters:**

num_inference_steps (`int`) : Number of model evaluations.

device (`str` or `torch.device`, *optional*) : Device for the schedule.

sigmas (`list[float]`, *optional*) : Unsupported custom sigma schedule.

mu (`float`, *optional*) : Unsupported dynamic-shift parameter.

timesteps (`list[float]`, *optional*) : Unsupported custom timestep schedule.

Set the distilled PiFlow timestep schedule.

#### step[[diffusers.PiflowScheduler.step]]

```python
step(model_output: FloatTensor, timestep: typing.Union[float, torch.FloatTensor], sample: FloatTensor, return_dict: bool = True)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/schedulers/scheduling_piflow.py#L324)

**Parameters:**

model_output (*torch.FloatTensor*) : Widened model output containing `n_grid` predictions per sample channel.

timestep (*float* or *torch.FloatTensor*) : Current scheduler timestep.

sample (*torch.FloatTensor*) : Current noisy sample.

return_dict (*bool*, *optional*, defaults to True) : Whether to return a [*PiflowSchedulerOutput*].

**Returns:** [*PiflowSchedulerOutput*] or *tuple*

Updated sample.

Advance one step by integrating the PiFlow policy.

## PiflowSchedulerOutput[[diffusers.schedulers.scheduling_piflow.PiflowSchedulerOutput]]

#### diffusers.schedulers.scheduling_piflow.PiflowSchedulerOutput[[diffusers.schedulers.scheduling_piflow.PiflowSchedulerOutput]]

```python
diffusers.schedulers.scheduling_piflow.PiflowSchedulerOutput(prev_sample: Tensor)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.0/src/diffusers/schedulers/scheduling_piflow.py#L30)

**Parameters:**

prev_sample (`torch.Tensor`) : Computed sample at the next PiFlow grid point. Should be used as the next denoising input.

Output class for the scheduler's `step` function output.

### DPMSolverSinglestepScheduler
https://huggingface.co/docs/diffusers/v0.42.0/api/schedulers/singlestep_dpm_solver.md
