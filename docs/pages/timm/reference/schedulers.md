# Learning Rate Schedulers

This page contains the API reference documentation for the learning rate schedulers included in `timm`, and an
overview of how they are created and stepped.

## Usage

Schedulers are created by name with
[`create_scheduler_v2`](#timm.scheduler.scheduler_factory.create_scheduler_v2), which returns the scheduler and the
total number of epochs to train (including any warmup prefix, cycles and cooldown, see below).
Unlike `torch.optim.lr_scheduler`, `timm` schedulers do not count steps themselves. They are designed to be
stateless: the lr is a function of the schedule configuration and an explicit epoch or update count, so resuming a
run only needs the epoch, and no scheduler state is stored in checkpoints. The training loop passes the epoch or
update count in explicitly:

* `scheduler.step(epoch)` at the end of each epoch, with the index of the next epoch (`epoch + 1`)
* `scheduler.step_update(num_updates)` after each optimizer update, with the number of updates so far

A scheduler is stepped either per epoch or per update (`step_on_epochs`), the other call is a no-op, so a training
loop can make both calls and support either.

```python
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

model = timm.create_model('resnet50')
optimizer = create_optimizer_v2(model, opt='adamw', lr=1e-3, weight_decay=0.05)
updates_per_epoch = len(loader)  # optimizer updates per epoch

# Cosine decay w/ 5 epochs of linear warmup, stepped per update for a smooth schedule.
scheduler, num_epochs = create_scheduler_v2(
    optimizer,
    sched='cosine',
    num_epochs=100,
    warmup_epochs=5,
    min_lr=1e-6,
    step_on_epochs=False,
    updates_per_epoch=updates_per_epoch,
)

num_updates = 0
for epoch in range(num_epochs):
    for batch in loader:
        ...  # forward, backward, optimizer.step()
        num_updates += 1
        scheduler.step_update(num_updates=num_updates)
    scheduler.step(epoch + 1)  # pass the eval metric too for 'plateau', scheduler.step(epoch + 1, metric)
```

Durations (`num_epochs`, `warmup_epochs`, `decay_epochs`, ...) are always given in epochs, the factory converts them
to updates when `step_on_epochs=False`. Schedulers can also be constructed directly from their classes, durations are
then in the unit they are stepped in (`t_in_epochs`).

The training script (`train.py`) exposes the factory arguments through these flags:

- **`--sched`** selects the schedule.
- **`--epochs`** sets the schedule length.
- **`--sched-on-updates`** steps per update instead of per epoch.
- **`--warmup-epochs`** sets the warmup duration.
- **`--warmup-lr`** sets the initial warmup learning rate.
- **`--warmup-prefix`** adds warmup before the schedule instead of overlapping its start.
- **`--min-lr`** sets the minimum learning rate.
- **`--decay-epochs`** sets the step-decay interval or the final decay duration for WSD / rsqrt.
- **`--decay-milestones`** sets the decay epochs for the multistep schedule.
- **`--decay-rate`** sets the decay factor, or the power for the polynomial schedule.
- **`--cooldown-epochs`** adds epochs at the minimum learning rate after the schedule ends.
- **`--patience-epochs`** sets the plateau scheduler's patience.
- **`--lr-noise`** selects the part of the schedule with learning rate noise.
- **`--lr-noise-pct`** limits the noise magnitude.
- **`--lr-noise-std`** sets the noise standard deviation.
- **`--lr-cycle-mul`** sets the cycle length multiplier.
- **`--lr-cycle-decay`** sets the peak learning rate multiplier between cycles.
- **`--lr-cycle-limit`** sets the number of cycles.
- **`--lr-k-decay`** sets k-decay for cosine / polynomial schedules.
- **`--lr-decay-fn`** selects the final decay function for WSD / rsqrt.
- **`--lr-stable-timescale`** sets the rsqrt stable-phase timescale.

### Schedules

| `sched` | Class | Schedule |
|---|---|---|
| `cosine` | [`CosineLRScheduler`](#timm.scheduler.cosine_lr.CosineLRScheduler) | Cosine annealing from the base lr to `min_lr` ([SGDR](https://arxiv.org/abs/1608.03983)), optional restarts (cycles) and k-decay |
| `tanh` | [`TanhLRScheduler`](#timm.scheduler.tanh_lr.TanhLRScheduler) | Hyperbolic tangent decay ([paper](https://arxiv.org/abs/1806.01593)), optional restarts |
| `poly` | [`PolyLRScheduler`](#timm.scheduler.poly_lr.PolyLRScheduler) | Polynomial decay to `min_lr`, the power is given by `decay_rate`, optional restarts and k-decay |
| `step` | [`StepLRScheduler`](#timm.scheduler.step_lr.StepLRScheduler) | lr multiplied by `decay_rate` every `decay_epochs` |
| `multistep` | [`MultiStepLRScheduler`](#timm.scheduler.multistep_lr.MultiStepLRScheduler) | lr multiplied by `decay_rate` at each of the `decay_milestones` epochs |
| `plateau` | [`PlateauLRScheduler`](#timm.scheduler.plateau_lr.PlateauLRScheduler) | **Deprecated.** lr multiplied by `decay_rate` when the eval metric hasn't improved for `patience_epochs`, per epoch stepping only |
| `wsd` | [`WSDLRScheduler`](#timm.scheduler.wsd_lr.WSDLRScheduler) | Warmup-stable-decay ([MiniCPM](https://arxiv.org/abs/2404.06395)), constant lr then a decay to `min_lr` over the final `decay_epochs`, `1-sqrt` decay by default ([paper](https://arxiv.org/abs/2405.18392)) |
| `rsqrt` | [`WSDLRScheduler`](#timm.scheduler.wsd_lr.WSDLRScheduler) | Reciprocal square root decay w/ a linear cooldown over the final `decay_epochs` ([Big Vision](https://arxiv.org/abs/2106.04560)) |

### Common options

- **Warmup.** `warmup_epochs` of linear warmup from `warmup_lr` to the base lr, supported by all schedules.
  - **`warmup_prefix`**: by default the warmup overlaps the start of the schedule (the schedule is computed from
    epoch 0, warmup replaces its first epochs), so `num_epochs` is the total length. With `warmup_prefix=True` the
    schedule starts after the warmup and the warmup is added to the total.
  - For `wsd` and `rsqrt` the stable phase always starts after the warmup, w/o `warmup_prefix` the stable and decay
    phases are fitted into `num_epochs - warmup_epochs` so `num_epochs` remains the total.
- **`min_lr`**: the lr the decaying schedules (`cosine`, `tanh`, `poly`, `wsd`, `rsqrt`) reach at the end, and a
  lower bound for `plateau`.

The `plateau` schedule is deprecated and will be removed in a future release. It never fit the stateless design
above: its lr depends on the history of the eval metric rather than on the epoch, and that history (best metric,
patience and cooldown counters) is not saved in checkpoints, so a resumed run restarts the plateau logic. Checkpoint
support will not be added. A validation plateau is also a weak signal for when to anneal; a fixed schedule such as
`cosine` or `wsd` is reproducible and resumes exactly, and a WSD sweep
(see [Hyperparameter Sweeps](../training_script#hyperparameter-sweeps)) compares cooldowns at several budgets from one
stable run.
- **`cooldown_epochs`**: epochs held at `min_lr` after the schedule ends (`cosine`, `tanh`, `poly`, `wsd`, `rsqrt`),
  added to the returned `num_epochs`. As the schedule reaches `min_lr` one step after its last step, a cooldown
  epoch is also a way to train at `min_lr` when stepping per epoch.
- **Cycles** (`cosine`, `tanh`, `poly`): `cycle_limit > 1` restarts the schedule, each cycle `cycle_mul` times
  longer and w/ its peak lr multiplied by `cycle_decay` relative to the previous. The returned `num_epochs` covers
  all cycles.
- **`k_decay`** (`cosine`, `poly`): [k-decay](https://arxiv.org/abs/2004.05909) changes the shape of the decay
  (`t ** k`), `k > 1` keeps the lr high for longer.
- **Noise**: `noise` adds random multiplicative lr noise over a range of the schedule, given as fractions of
  `num_epochs` (a start, or a start and an end), limited by `noise_pct`.

### WSD and rsqrt

The [`WSDLRScheduler`](#timm.scheduler.wsd_lr.WSDLRScheduler) has three phases: a linear warmup, a stable phase and
a decay to `min_lr`.

- **`decay_epochs`** sets the final decay duration. A value in (0, 1) is a fraction of the stable + decay length,
  20% by default.
- **`decay_fn`** (`--lr-decay-fn`) selects `1-sqrt` (default for `wsd`), `linear` (default for `rsqrt`), `cosine`,
  `power` or `exp`.
- **`stable_timescale`** (`--lr-stable-timescale`, in epochs) controls the `rsqrt` stable phase, which decays as
  `(1 + t / timescale) ** -0.5` from the end of warmup. It defaults to the warmup length and must be set when there
  is no warmup.

The class also supports a generalized `power` stable phase and custom decay functions.

The final anneal is where WSD schedules get most of their benefit, so step per update (`--sched-on-updates`), or add
a cooldown epoch, when the decay phase is only a few epochs long. Stepped per epoch, the last epoch of a short decay
is noticeably above `min_lr` (~10% of the base lr for the last of 5 decay epochs w/ `1-sqrt`).

```python
# 300 epochs, 5 epoch warmup, constant lr, then a 1-sqrt decay over the final 20%
scheduler, num_epochs = create_scheduler_v2(
    optimizer, sched='wsd', num_epochs=300, warmup_epochs=5, decay_epochs=0.2,
    step_on_epochs=False, updates_per_epoch=updates_per_epoch,
)
```

### Parameter group lr scales

Param groups w/ an `lr_scale` field (e.g. layer-wise lr decay groups from
[`param_groups_layer_decay`](optimizers#timm.optim.param_groups_layer_decay)) get the scheduled lr multiplied by
their scale. The base lr of each group is recorded as `initial_lr` in the group when the scheduler is created.

### Resuming

As the step counts are passed in, a scheduler for a resumed run is set up by creating it as for the original run
and stepping it to the resume point, `scheduler.step(start_epoch)` or
`scheduler.step_update(start_epoch * updates_per_epoch)`. `state_dict()` / `load_state_dict()` save and restore the
scheduler's attributes.

## API

### Factory functions[[timm.scheduler.create_scheduler]]

#### timm.scheduler.create_scheduler[[timm.scheduler.create_scheduler]]

```python
timm.scheduler.create_scheduler(args, optimizer: Optimizer, updates_per_epoch: int = 0)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/scheduler_factory.py#L54)

#### timm.scheduler.create_scheduler_v2[[timm.scheduler.create_scheduler_v2]]

```python
timm.scheduler.create_scheduler_v2(optimizer: Optimizer, sched: str = 'cosine', num_epochs: int = 300, decay_epochs: typing.Optional[float] = None, decay_milestones: typing.List[int] = (90, 180, 270), cooldown_epochs: int = 0, patience_epochs: int = 10, decay_rate: float = 0.1, min_lr: float = 0.0, warmup_lr: float = 1e-05, warmup_epochs: int = 0, warmup_prefix: bool = False, noise: typing.Union[float, typing.List[float]] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, cycle_mul: float = 1.0, cycle_decay: float = 0.1, cycle_limit: int = 1, k_decay: float = 1.0, decay_fn: typing.Optional[str] = None, stable_timescale: typing.Optional[float] = None, plateau_mode: str = 'max', step_on_epochs: bool = True, updates_per_epoch: int = 0)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/scheduler_factory.py#L66)

**Parameters:**

optimizer : optimizer to schedule

sched : scheduler name, one of 'cosine', 'tanh', 'step', 'multistep', 'poly', 'wsd', 'rsqrt', 'constant', 'plateau' (deprecated), or 'none' for no scheduler (returns None, the optimizer lr is left untouched). 'constant' holds the lr after warmup w/ param group lr_scale (layer-wise lr decay) applied.

num_epochs : number of epochs in the schedule (excluding warmup if warmup_prefix, excluding cooldown)

decay_epochs : lr decay interval (step), or the length of the decay phase (wsd / rsqrt) in epochs, a value in (0, 1) is a fraction of the schedule length. None = 90 (step), 0.2 (wsd / rsqrt)

decay_milestones : epochs to decay the lr at (multistep)

cooldown_epochs : epochs to hold the lr at min_lr after the schedule ends

patience_epochs : epochs w/o improvement before decaying the lr (plateau, deprecated)

decay_rate : lr decay rate (step, multistep, plateau), the polynomial power (poly)

min_lr : lr lower bound for schedulers that decay to it

warmup_lr : lr at the start of warmup

warmup_epochs : epochs of linear warmup

warmup_prefix : exclude the warmup period from num_epochs (added to the total), wsd / rsqrt are always warmup prefixed, with warmup_prefix=False their stable + decay phases fit within num_epochs

noise : lr noise on / off points as fractions of the schedule length

noise_pct : lr noise limit percent

noise_std : lr noise std-dev

noise_seed : lr noise seed

cycle_mul : cycle length multiplier (cosine, tanh, poly)

cycle_decay : lr decay per cycle (cosine, tanh, poly)

cycle_limit : number of cycles (cosine, tanh, poly)

k_decay : k-decay (cosine, poly)

decay_fn : decay phase function (wsd / rsqrt), None = '1-sqrt' for wsd, 'linear' for rsqrt

stable_timescale : rsqrt timescale in epochs (rsqrt), None = warmup_epochs

plateau_mode : 'max' or 'min', the metric direction (plateau)

step_on_epochs : step the schedule per epoch, otherwise per update (requires updates_per_epoch)

updates_per_epoch : number of optimizer updates per epoch (step_on_epochs=False)

**Returns:**

(scheduler, num_epochs) w/ num_epochs the total number of epochs to train including warmup prefix and cooldown

Create a learning rate scheduler by name.

### Scheduler Classes[[timm.scheduler.CosineLRScheduler]]

#### timm.scheduler.CosineLRScheduler[[timm.scheduler.CosineLRScheduler]]

```python
timm.scheduler.CosineLRScheduler(optimizer: Optimizer, t_initial: int, lr_min: float = 0.0, cycle_mul: float = 1.0, cycle_decay: float = 1.0, cycle_limit: int = 1, warmup_t: int = 0, warmup_lr_init: float = 0.0, warmup_prefix: bool = False, t_in_epochs: bool = True, noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, k_decay: float = 1.0, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/cosine_lr.py#L19)

Cosine decay with restarts.
This is described in the paper https://arxiv.org/abs/1608.03983.

Inspiration from
https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909

#### timm.scheduler.MultiStepLRScheduler[[timm.scheduler.MultiStepLRScheduler]]

```python
timm.scheduler.MultiStepLRScheduler(optimizer: Optimizer, decay_t: typing.List[int], decay_rate: float = 1.0, warmup_t: int = 0, warmup_lr_init: float = 0.0, warmup_prefix: bool = True, t_in_epochs: bool = True, noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/multistep_lr.py#L10)

Multi-step learning rate schedule with warmup and noise.

Outside warmup, decay milestones are compared against `t + 1`, where
`t` excludes warmup when `warmup_prefix=True`. For example, without
warmup, a milestone of 10 applies at schedule index 9.

This longstanding convention differs from PyTorch's MultiStepLR and is
retained for compatibility with existing training schedules.

#### timm.scheduler.PlateauLRScheduler[[timm.scheduler.PlateauLRScheduler]]

```python
timm.scheduler.PlateauLRScheduler(optimizer: Optimizer, decay_rate: float = 0.1, patience_t: int = 10, threshold: float = 0.0001, cooldown_t: int = 0, warmup_t: int = 0, warmup_lr_init: float = 0.0, lr_min: float = 0.0, mode: str = 'max', noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_type: str = 'normal', noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: typing.Optional[int] = None, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/plateau_lr.py#L15)

Decay the LR by a factor every time the validation loss plateaus.

Deprecated: this scheduler will be removed in a future release. timm schedulers are designed to be stateless
functions of the epoch or update count, which this metric-driven schedule never was. Its state (best metric,
patience and cooldown counters) is not saved in training checkpoints, so a resumed run restarts the plateau logic,
and checkpoint support will not be added. Prefer a fixed schedule such as `cosine` or `wsd`, the WSD cooldown
branching in `sweep.py` compares annealing at several budgets with reproducible schedules.

#### timm.scheduler.PolyLRScheduler[[timm.scheduler.PolyLRScheduler]]

```python
timm.scheduler.PolyLRScheduler(optimizer: Optimizer, t_initial: int, power: float = 0.5, lr_min: float = 0.0, cycle_mul: float = 1.0, cycle_decay: float = 1.0, cycle_limit: int = 1, warmup_t: int = 0, warmup_lr_init: float = 0.0, warmup_prefix: bool = False, t_in_epochs: bool = True, noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, k_decay: float = 1.0, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/poly_lr.py#L19)

Polynomial LR Scheduler w/ warmup, noise, and k-decay

k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909

#### timm.scheduler.StepLRScheduler[[timm.scheduler.StepLRScheduler]]

```python
timm.scheduler.StepLRScheduler(optimizer: Optimizer, decay_t: float, decay_rate: float = 1.0, warmup_t: int = 0, warmup_lr_init: float = 0.0, warmup_prefix: bool = True, t_in_epochs: bool = True, noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/step_lr.py#L15)

#### timm.scheduler.TanhLRScheduler[[timm.scheduler.TanhLRScheduler]]

```python
timm.scheduler.TanhLRScheduler(optimizer: Optimizer, t_initial: int, lb: float = -7.0, ub: float = 3.0, lr_min: float = 0.0, cycle_mul: float = 1.0, cycle_decay: float = 1.0, cycle_limit: int = 1, warmup_t: int = 0, warmup_lr_init: float = 0.0, warmup_prefix: bool = False, t_in_epochs: bool = True, noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/tanh_lr.py#L19)

Hyberbolic-Tangent decay with restarts.
This is described in the paper https://arxiv.org/abs/1806.01593

#### timm.scheduler.WSDLRScheduler[[timm.scheduler.WSDLRScheduler]]

```python
timm.scheduler.WSDLRScheduler(optimizer: Optimizer, t_initial: int, decay_t: int, lr_min: float = 0.0, stable_fn: str = 'constant', stable_timescale: typing.Optional[float] = None, stable_power: float = 0.5, decay_fn: typing.Union[str, typing.Callable[[float], float]] = '1-sqrt', decay_power: float = 0.5, decay_ratio: float = 0.1, warmup_t: int = 0, warmup_lr_init: float = 0.0, t_in_epochs: bool = True, noise_range_t: typing.Union[typing.List[int], typing.Tuple[int, int], int, NoneType] = None, noise_pct: float = 0.67, noise_std: float = 1.0, noise_seed: int = 42, initialize: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/scheduler/wsd_lr.py#L26)

**Parameters:**

optimizer : optimizer to schedule

t_initial : length of the stable + decay phases (epochs or updates, see `t_in_epochs`)

decay_t : length of the decay phase, the final `decay_t` steps of `t_initial`

lr_min : lr at the end of the decay phase

stable_fn : stable phase function, one of 'constant', 'rsqrt', 'power'

stable_timescale : timescale of the 'rsqrt' / 'power' stable functions, defaults to `warmup_t`

stable_power : exponent of the 'power' stable function

decay_fn : decay phase function, one of 'linear', '1-sqrt', 'cosine', 'power', 'exp' or a callable

decay_power : exponent of the 'power' decay function

decay_ratio : final fraction (before normalization) of the 'exp' decay function

warmup_t : warmup length, a prefix to `t_initial`

warmup_lr_init : lr at the start of warmup

t_in_epochs : `t` is in epochs (`step()`), otherwise updates (`step_update()`)

noise_range_t : lr noise range, see `Scheduler`

noise_pct : lr noise limit percent

noise_std : lr noise std-dev

noise_seed : lr noise seed

initialize : initialize the `initial_lr` param group field from the current lr

Warmup-Stable-Decay (WSD) LR schedule w/ selectable stable and decay functions.

Three phases: a linear warmup over `warmup_t` steps, a stable phase, and a decay to `lr_min` over the final
`decay_t` steps. `t_initial` is the length of the stable + decay phases, the warmup is always a prefix so the
schedule has a total length of `warmup_t + t_initial` (see `get_cycle_length()`), after which `lr_min` is held.

NOTE: as for the other schedulers, `lr_min` is reached at `t = warmup_t + t_initial`, one step past the last
scheduled step, so the last scheduled step is one decay step short of it. With a short decay phase stepped per
epoch this is significant (e.g. the last of 5 decay epochs w/ '1-sqrt' is at ~10% of the stable lr). The final
anneal matters for WSD, so step per update (`t_in_epochs=False`, `--sched-on-updates` in train.py) or add a
cooldown epoch (held at `lr_min`, `--cooldown-epochs`).

The stable function is evaluated on the time since the end of warmup (`t_s`) and equals 1 at `t_s = 0`, so the
warmup always hands off at the base lr. The decay function maps the decay progress `frac` in [0, 1] to a factor
from 1 to 0 and multiplies the stable value, so a decaying stable function keeps decaying through the decay
phase (as in the Big Vision rsqrt + linear cooldown schedule).

lr(t) = lr_min + (base_lr * stable(t_s) - lr_min) * decay(frac)

Stable functions:
* `constant`: 1, the WSD schedule
* `rsqrt`: (1 + t_s / timescale) ** -0.5, the Big Vision reciprocal square root schedule when
  `stable_timescale == warmup_t` (the default)
* `power`: (1 + t_s / timescale) ** -stable_power, rsqrt generalized (Power scheduler)

Decay functions:
* `1-sqrt`: 1 - sqrt(frac), the best performing cooldown in https://arxiv.org/abs/2405.18392 (default)
* `linear`: 1 - frac, the 'trapezoidal' schedule and the Big Vision cooldown
* `cosine`: 0.5 * (1 + cos(pi * frac))
* `power`: 1 - frac ** decay_power, 0.5 == '1-sqrt', 1.0 == 'linear', 2.0 is a 'square' decay
* `exp`: exponential decay to a fraction `decay_ratio` of the stable value, normalized to reach lr_min
* a callable `fn(frac) -> factor` for anything else
