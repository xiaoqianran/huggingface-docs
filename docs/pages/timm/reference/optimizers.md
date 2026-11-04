# Optimization

This page contains the API reference documentation for the optimizers included in `timm`, and an overview of how
they are selected and configured through the optimizer factory.

## Usage

Optimizers are created by name with [`create_optimizer_v2`](#timm.optim.create_optimizer_v2). Given a model, the
factory builds the parameter groups (weight decay exclusions, optional layer-wise learning rate decay) before
constructing the optimizer. Optimizer specific arguments pass through as keyword arguments.

```python
import timm
from timm.optim import create_optimizer_v2, list_optimizers

model = timm.create_model('vit_base_patch16_224')

# AdamW. 1D params (biases, norm scales) and params in model.no_weight_decay() are excluded from weight decay.
optimizer = create_optimizer_v2(model, opt='adamw', lr=1e-3, weight_decay=0.05)

# Layer-wise learning rate decay (fine-tuning), per-layer lr scales from the model's group_matcher().
optimizer = create_optimizer_v2(model, opt='adamw', lr=1e-3, weight_decay=0.05, layer_decay=0.75)

# Optimizer specific arguments are passed through.
optimizer = create_optimizer_v2(model, opt='lamb', lr=5e-3, weight_decay=0.02, trust_clip=True)

# A list of params or param groups can be passed instead of a model, weight decay grouping is then up to the caller.
optimizer = create_optimizer_v2(model.parameters(), opt='sgd', lr=0.1, momentum=0.9)

# All registered names, optionally with a description.
print(list_optimizers(with_description=True))
```

The training script (`train.py`) exposes the same through these flags:

- **`--opt`** selects the optimizer.
- **`--lr`** sets the learning rate.
- **`--weight-decay`** sets weight decay.
- **`--momentum`** sets momentum.
- **`--opt-eps`** sets epsilon (`eps`).
- **`--opt-betas`** sets the beta coefficients (`betas`).
- **`--layer-decay`** sets layer-wise learning rate decay.
- **`--opt-kwargs key=value ...`** passes any other optimizer arguments.

### Naming conventions

Registered names are lower case, a base optimizer name plus modifiers:

| Modifier | Meaning | Examples |
|---|---|---|
| `c` prefix | Cautious variant, the update is masked where its sign disagrees with the gradient ([Cautious Optimizers](https://arxiv.org/abs/2411.16085)) | `cadamw`, `clion`, `cmars`, `cadopt` |
| `w` suffix | Decoupled (AdamW style) weight decay, for optimizers that default to coupled L2 decay | `sgdw`, `lambw`, `adoptw`, `novogradw`, `rmsproptfw` |
| `c` suffix | Corrected weight decay, the decay is scaled by `lr² / max_lr` instead of `lr` ([Why Gradients Rapidly Increase Near the End of Training](https://arxiv.org/abs/2506.02285)) | `adamc`, `nadamc`, `sgdc`, `lionc`, `adoptc`, `kronc` |
| `n` prefix | Nesterov momentum (SGD, LARS and Muon families) | `nesterov`, `nlars`, `nmuon`, `nadamuon` |
| `lookahead_` prefix | Wrap the optimizer in [`Lookahead`](#timm.optim.lookahead.Lookahead) | `lookahead_adamw` |
| `fused` prefix | NVIDIA Apex fused implementation (requires `apex`) | `fusedadamw`, `fusedlamb` |
| `bnb` prefix | bitsandbytes implementation, `8bit` suffix for 8-bit state (requires `bitsandbytes`) | `bnbadamw`, `bnbadamw8bit` |

A few names predate the conventions: `lambc` and `larc` are LAMB / LARS with trust ratio clipping (the corrected
decay LAMB variant is `lambcd`), `momentum` is SGD with classical momentum and `sgd` is SGD with Nesterov momentum.
The `legacy` implementations in the table below predate their inclusion in PyTorch.

### Optimizer families

| Names | Implementation | Notes |
|---|---|---|
| `adamw`, `adam`, `nadam`, `radam`, `radamw`, `adamax`, `adadelta`, `adagrad`, `rmsprop`, `sgd`, `momentum`, `nesterov` | `torch.optim` | |
| `adamwlegacy`, `adamc`, `cadamw`, `cadamc` | [`AdamWLegacy`](#timm.optim.adamw.AdamWLegacy) | AdamW with caution, corrected decay, update clipping, `capturable` |
| `nadamw`, `nadamc`, `cnadamw`, `cnadamc` | [`NAdamW`](#timm.optim.nadamw.NAdamW) | AdamW with Nesterov momentum (MLCommons algorithmic-efficiency form) |
| `adopt`, `adoptw`, `adoptc`, `cadopt`, `cadoptw`, `cadoptc` | [`Adopt`](#timm.optim.adopt.Adopt) | Adam variant that converges for any β2 |
| `adan`, `adanw`, `cadan`, `cadanw` | [`Adan`](#timm.optim.adan.Adan) | Adaptive Nesterov momentum |
| `adabelief`, `radabelief` | [`AdaBelief`](#timm.optim.adabelief.AdaBelief) | `radabelief` adds RAdam style rectification |
| `adafactor`, `cadafactor` | [`Adafactor`](#timm.optim.adafactor.Adafactor) | fairseq Adafactor, relative step size when `lr=None` |
| `adafactorbv`, `adafactorbvc`, `cadafactorbv`, `cadafactorbvc` | [`AdafactorBigVision`](#timm.optim.adafactor_bv.AdafactorBigVision) | Big Vision Adafactor, factored second moments, low precision momentum, update clipping |
| `adamp`, `cadamp`, `sgdp`, `csgdp` | [`AdamP`](#timm.optim.adamp.AdamP), [`SGDP`](#timm.optim.sgdp.SGDP) | Projection of the update for scale invariant weights |
| `adahessian` | [`Adahessian`](#timm.optim.adahessian.Adahessian) | Second order, requires a closure (`is_second_order`) |
| `kron`, `kronw`, `kronc` | [`Kron`](#timm.optim.kron.Kron) | PSGD with a Kronecker-factored preconditioner |
| `lamb`, `lambc`, `lambw`, `lambcw`, `lambcd`, `clamb`, `clambw` | [`Lamb`](#timm.optim.lamb.Lamb) | Layer-wise adaptive moments, `trust_clip` variants (`lambc`, `lambcw`) |
| `laprop`, `lapropc`, `claprop` | [`LaProp`](#timm.optim.laprop.LaProp) | Momentum and adaptivity separated |
| `lars`, `larc`, `nlars`, `nlarc` | [`Lars`](#timm.optim.lars.Lars) | Layer-wise adaptive rate scaling, `trust_clip` variants (`larc`, `nlarc`) |
| `lion`, `lionc`, `clion`, `clionc` | [`Lion`](#timm.optim.lion.Lion) | Evolved sign momentum |
| `madgrad`, `madgradw` | [`MADGRAD`](#timm.optim.madgrad.MADGRAD) | Momentum based dual averaging |
| `mars`, `cmars` | [`Mars`](#timm.optim.mars.Mars) | Variance reduction, AdamW or Lion style update |
| `muon`, `nmuon`, `adamuon`, `nadamuon` | [`Muon`](#timm.optim.muon.Muon) | Newton-Schulz orthogonalized momentum for 2D+ weights, AdamW / NAdamW fallback for 1D params |
| `novograd`, `novogradw`, `novogradc` | [`NvNovoGrad`](#timm.optim.nvnovograd.NvNovoGrad) | Layer-wise normalized moments |
| `rmsproptf`, `rmsproptfw`, `rmsproptfc`, `crmsproptf` | [`RMSpropTF`](#timm.optim.rmsprop_tf.RMSpropTF) | TensorFlow ordering of ops and eps handling, lr folded into the momentum buffer |
| `sgdw`, `nesterovw`, `sgdc`, `csgdw`, `csgdc` | [`SGDW`](#timm.optim.sgdw.SGDW) | SGD with decoupled weight decay |
| `nadamlegacy`, `radamlegacy` | [`NAdamLegacy`](#timm.optim.nadam.NAdamLegacy), [`RAdamLegacy`](#timm.optim.radam.RAdamLegacy) | Reference implementations, superseded by `torch.optim` |
| `fused*`, `bnb*` | NVIDIA Apex, bitsandbytes | Registered when the package is installed |

### Common options

Most `timm` optimizers share a set of options beyond the usual `lr`, `weight_decay`, `betas` and `eps`. Which
optimizers accept which can be checked in the class signatures below, the factory raises on unsupported arguments.

- **`caution`** applies the cautious update mask, equivalent to the `c` prefixed registry names.
- **`corrected_weight_decay`** scales decoupled weight decay by `lr² / max_lr` instead of `lr` (the `c` suffixed
  names), so the effective decay follows the learning rate schedule. `max_lr` is the learning rate the optimizer
  was constructed with, construct it with the peak schedule lr (as `train.py` does). It is kept in each param group
  (`max_lr_snapshot`) and saved / restored with the optimizer state.
- **`decoupled_decay`** (or `decoupled`) switches optimizers that default to coupled L2 decay to AdamW style decay
  (the `w` suffixed names).
- **`foreach`** selects the multi-tensor implementation. By default (`None`) it is used on CUDA and other devices
  with foreach kernels, matching `torch.optim`, unless unsupported by the configuration (a tensor `lr` without
  `capturable`, or `caution` on a PyTorch version without the required `_foreach_maximum_` overload). The
  multi-tensor step is run in chunks of parameters sized so each chunk's working set stays in the GPU L2 cache,
  which is faster than streaming every parameter through memory once per op for large models. The chunk size is
  derived from the device L2 cache size (PyTorch >= 2.5) and chunking is skipped on devices where the resulting
  chunk would be below 4M elements. `TIMM_FOREACH_CHUNK_SIZE=<elements>` overrides the chunk size, `0` disables
  chunking.
- **`capturable`** (AdamWLegacy, NAdamW, Adopt, Muon) keeps the step count on the device so the step can be captured
  in a CUDA graph. All optimizers accept an `lr` given as a 0-dim tensor, updated in place by a schedule, for use
  with `torch.compile` and CUDA graphs.
- **`clipping_threshold`** (AdamWLegacy, NAdamW, AdafactorBigVision) clips the update by its RMS,
  `update / max(1, rms / threshold)` with `rms` the RMS of the second-moment normalized gradient, per tensor
  ([StableAdamW](https://arxiv.org/abs/2304.13013), Adafactor). `1.0` as in the papers. As in Adafactor, the
  (decoupled) weight decay is not clipped, StableAdamW also scales the decay by the clipped learning rate.
- **`maximize`** maximizes the objective instead of minimizing.

Low precision parameters (`bfloat16`, `float16`) are supported throughout:

- **AdaBelief, RAdamLegacy and Adafactor** keep their state in `float32` for low precision params.
- **AdafactorBigVision** keeps its momentum in a reduced precision (`momentum_dtype`).
- **Kron** has separate `mu_dtype` / `precond_dtype` options.

These state dtypes are preserved by `load_state_dict` (PyTorch's default loader casts state to the param dtype).
The step functions avoid host syncs (data dependent Python branches on device tensors), AdamP / SGDP's projection
check is the exception.

### Parameter groups

When given a model, `create_optimizer_v2` groups parameters with
[`param_groups_weight_decay`](#timm.optim.param_groups_weight_decay) (no decay for 1D params and the model's
`no_weight_decay()` set) or, with `layer_decay` set,
[`param_groups_layer_decay`](#timm.optim.param_groups_layer_decay), which assigns a learning rate scale of
`layer_decay ** (num_layers - layer_id)` per layer using the model's `group_matcher()`. Models without a
`group_matcher()` fall back to [`auto_group_layers`](#timm.optim.auto_group_layers), which needs the model's
`pretrained_cfg['classifier']` to separate the head. For hybrid optimizers (Muon) the `fallback_list` and
`fallback_no_weight_decay` arguments route parameters to the fallback (AdamW) optimizer.

## API

### Factory functions[[timm.optim.create_optimizer_v2]]

#### timm.optim.create_optimizer_v2[[timm.optim.create_optimizer_v2]]

```python
timm.optim.create_optimizer_v2(model_or_params: typing.Union[torch.nn.Module, torch.optim.optimizer.ParamsT], opt: str = 'sgd', lr: typing.Optional[float] = None, weight_decay: float = 0.0, momentum: float = 0.9, foreach: typing.Optional[bool] = None, filter_bias_and_bn: bool = True, fallback_list: typing.Collection[str] = (), fallback_no_weight_decay: bool = False, layer_decay: typing.Optional[float] = None, layer_decay_min_scale: float = 0.0, layer_decay_no_opt_scale: typing.Optional[float] = None, param_group_fn: typing.Optional[typing.Callable[[torch.nn.Module], torch.optim.optimizer.ParamsT]] = None, **kwargs: typing.Any)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_optim_factory.py#L1242)

**Parameters:**

model_or_params : A PyTorch model or an iterable of parameters/parameter groups. If a model is provided, parameters will be automatically extracted and grouped based on the other arguments.

opt : Name of the optimizer to create (e.g., 'adam', 'adamw', 'sgd'). Use list_optimizers() to see available options.

lr : Learning rate. If None, will use the optimizer's default.

weight_decay : Weight decay factor. Will be used to create param groups if model_or_params is a model.

momentum : Momentum factor for optimizers that support it. Only used if the chosen optimizer accepts a momentum parameter.

foreach : Enable/disable foreach (multi-tensor) implementation if available. If None, will use optimizer-specific defaults.

filter_bias_and_bn : If True, bias, norm layer parameters (all 1d params) will not have weight decay applied. Only used when model_or_params is a model and weight_decay > 0.

fallback_list : Collection of parameter name patterns to use fallback optimizer for hybrid optimizers (e.g., AdamW for Muon). Supports wildcard matching.

fallback_no_weight_decay : If True, params in model's no_weight_decay() list will use fallback optimizer for hybrid optimizers (e.g., AdamW for Muon).

layer_decay : Optional layer-wise learning rate decay factor. If provided, learning rates will be scaled by layer_decay^(max_depth - layer_depth). Only used when model_or_params is a model.

param_group_fn : Optional function to create custom parameter groups. If provided, other parameter grouping options will be ignored.

- ****kwargs** : Additional optimizer-specific arguments (e.g., betas for Adam).

**Returns:**

Configured optimizer instance.

Create an optimizer instance via timm registry.

Creates and configures an optimizer with appropriate parameter groups and settings.
Supports automatic parameter group creation for weight decay and layer-wise learning
rates, as well as custom parameter grouping.

Examples:

AdamW from a model, the parameter groups are created automatically:

```python
optimizer = create_optimizer_v2(model, 'adamw', lr=1e-3)
```

SGD with momentum and weight decay:

```python
optimizer = create_optimizer_v2(model, 'sgd', lr=0.1, momentum=0.9, weight_decay=1e-4)
```

Adam with layer-wise learning rate decay:

```python
optimizer = create_optimizer_v2(model, 'adam', lr=1e-3, layer_decay=0.7)
```

Custom parameter groups:

```python
def group_fn(model):
    return [
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-3},
    ]

optimizer = create_optimizer_v2(model, 'sgd', param_group_fn=group_fn)
```

Note:
Parameter group handling precedence:
1. If param_group_fn is provided, it will be used exclusively
2. If layer_decay is provided, layer-wise groups will be created
3. If weight_decay > 0 and filter_bias_and_bn is True, weight decay groups will be created
4. Otherwise, all parameters will be in a single group

#### timm.optim.list_optimizers[[timm.optim.list_optimizers]]

```python
timm.optim.list_optimizers(filter: typing.Union[str, typing.List[str]] = '', exclude_filters: typing.Optional[typing.List[str]] = None, with_description: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_optim_factory.py#L1137)

**Parameters:**

filter : Wildcard style filter string or list of filter strings (e.g., 'adam*' for all Adam variants, or ['adam*', '*8bit'] for Adam variants and 8-bit optimizers). Empty string means no filtering.

exclude_filters : Optional list of wildcard patterns to exclude. For example, ['*8bit', 'fused*'] would exclude 8-bit and fused implementations.

with_description : If True, returns tuples of (name, description) instead of just names. Descriptions provide brief explanations of optimizer characteristics.

**Returns:** `If with_description is False`

List of optimizer names as strings (e.g., ['adam', 'adamw', ...])
If with_description is True:
List of tuples of (name, description) (e.g., [('adam', 'Adaptive Moment...'), ...])

List available optimizer names, optionally filtered.

List all registered optimizers, with optional filtering using wildcard patterns.
Optimizers can be filtered using include and exclude patterns, and can optionally
return descriptions with each optimizer name.

Examples:

```python
>>> list_optimizers()
['adam', 'adamw', 'sgd', ...]

>>> list_optimizers(['la*', 'nla*'])  # List lamb & lars
['lamb', 'lambc', 'larc', 'lars', 'nlarc', 'nlars']

>>> list_optimizers('*adam*', exclude_filters=['bnb*', 'fused*'])  # Exclude bnb & apex adam optimizers
['adam', 'adamax', 'adamp', 'adamw', 'nadam', 'nadamw', 'radam']

>>> list_optimizers(with_description=True)  # Get descriptions
[('adabelief', 'Adapts learning rate based on gradient prediction error'),
 ('adadelta', 'torch.optim Adadelta, Adapts learning rates based on running windows of gradients'),
 ('adafactor', 'Memory-efficient implementation of Adam with factored gradients'),
...]
```

#### timm.optim.get_optimizer_class[[timm.optim.get_optimizer_class]]

```python
timm.optim.get_optimizer_class(name: str, bind_defaults: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_optim_factory.py#L1200)

**Parameters:**

name : Name of the optimizer to retrieve (e.g., 'adam', 'sgd')

bind_defaults : If True, returns a partial function with default arguments from OptimInfo bound. If False, returns the raw optimizer class.

**Returns:** `If bind_defaults is False`

The optimizer class (e.g., torch.optim.Adam)
If bind_defaults is True:
A partial function with default arguments bound

**Raises:** ``ValueError``

- ``ValueError`` -- If optimizer name is not found in registry

Get optimizer class by name with option to bind default arguments.

Retrieves the optimizer class or a partial function with default arguments bound.
This allows direct instantiation of optimizers with their default configurations
without going through the full factory.

Examples:

With the registry defaults bound, `sgd` has nesterov momentum enabled:

```python
SGD = get_optimizer_class('sgd')
opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
```

The raw optimizer class:

```python
SGD = get_optimizer_class('sgd', bind_defaults=False)
opt = SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
```

#### timm.optim.get_optimizer_info[[timm.optim.get_optimizer_info]]

```python
timm.optim.get_optimizer_info(name: str)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_optim_factory.py#L1185)

**Parameters:**

name : Name of the optimizer

**Returns:**

OptimInfo configuration

**Raises:** ``ValueError``

- ``ValueError`` -- If optimizer is not found

Get the OptimInfo for an optimizer.

### Parameter group functions[[timm.optim.param_groups_weight_decay]]

#### timm.optim.param_groups_weight_decay[[timm.optim.param_groups_weight_decay]]

```python
timm.optim.param_groups_weight_decay(model: Module, weight_decay: typing.Optional[float] = 1e-05, weight_decay_exclude_1d: bool = True, no_weight_decay_list: typing.Collection[str] = (), fallback_list: typing.Collection[str] = (), fallback_no_weight_decay: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_param_groups.py#L19)

#### timm.optim.param_groups_layer_decay[[timm.optim.param_groups_layer_decay]]

```python
timm.optim.param_groups_layer_decay(model: Module, weight_decay: float = 0.05, no_weight_decay_list: typing.Collection[str] = (), fallback_list: typing.Collection[str] = (), fallback_no_weight_decay: bool = False, weight_decay_exclude_1d: bool = True, layer_decay: float = 0.75, min_scale: float = 0.0, no_opt_scale: typing.Optional[float] = None, verbose: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_param_groups.py#L113)

Parameter groups for layer-wise lr decay & weight decay
Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58

#### timm.optim.auto_group_layers[[timm.optim.auto_group_layers]]

```python
timm.optim.auto_group_layers(model, layers_per_group = 12, num_groups = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/_param_groups.py#L84)

### Optimizer Classes[[timm.optim.AdaBelief]]

#### timm.optim.AdaBelief[[timm.optim.AdaBelief]]

```python
timm.optim.AdaBelief(params, lr = 0.001, betas = (0.9, 0.999), eps = 1e-16, weight_decay = 0, amsgrad = False, decoupled_decay = True, fixed_decay = False, rectify = True, degenerated_to_sgd = True, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adabelief.py#L13)

**Parameters:**

params (iterable) : iterable of parameters to optimize or dicts defining parameter groups

lr (float, optional) : learning rate (default: 1e-3)

betas (Tuple[float, float], optional) : coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))

eps (float, optional) : term added to the denominator to improve numerical stability (default: 1e-16)

weight_decay (float, optional) : weight decay (L2 penalty) (default: 0)

amsgrad (boolean, optional) : whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_ (default: False)

decoupled_decay (boolean, optional) : (default: True) If set as True, then the optimizer uses decoupled weight decay as in AdamW

fixed_decay (boolean, optional) : (default: False) This is used when weight_decouple is set as True. When fixed_decay == True, the weight decay is performed as $W_{new} = W_{old} - W_{old} \times decay$. When fixed_decay == False, the weight decay is performed as $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the weight decay ratio decreases with learning rate (lr).

rectify (boolean, optional) : (default: True) If set as True, then perform the rectified update similar to RAdam

degenerated_to_sgd (boolean, optional) (default --True) If set as True, then perform SGD update when variance of gradient is high

foreach (boolean, optional) : use the foreach (multi-tensor) impl, by default (None) used unless unsupported

Implements AdaBelief algorithm. Modified from Adam in PyTorch

reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020

For a complete table of recommended hyperparameters, see https://github.com/juntang-zhuang/Adabelief-Optimizer'
For example train/args for EfficientNet see these gists
- link to train_script: https://gist.github.com/juntang-zhuang/0a501dd51c02278d952cf159bc233037
- link to args.yaml: https://gist.github.com/juntang-zhuang/517ce3c27022b908bb93f78e4f786dc3

#### step[[timm.optim.AdaBelief.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adabelief.py#L121)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.Adafactor[[timm.optim.Adafactor]]

```python
timm.optim.Adafactor(params: ParamsT, lr: typing.Optional[float] = None, eps: float = 1e-30, eps_scale: float = 0.001, clip_threshold: float = 1.0, decay_rate: float = -0.8, betas: typing.Optional[typing.Tuple[float, float]] = None, weight_decay: float = 0.0, scale_parameter: bool = True, warmup_init: bool = False, min_dim_size_to_factor: int = 16, caution: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adafactor.py#L22)

Implements Adafactor algorithm.

This implementation is based on: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
(see https://arxiv.org/abs/1804.04235)

Note that this optimizer internally adjusts the learning rate depending on the
*scale_parameter*, *relative_step* and *warmup_init* options.

A manual (external) learning rate schedule is used when `lr` is set (`relative_step` is then False).
Note that in this implementation `scale_parameter` and `warmup_init` only apply to the relative step
learning rate (`lr=None`), unlike fairseq / HF Transformers, the parameter RMS scaling is not applied
to an external learning rate.

Ags:
params: iterable of parameters to optimize or dicts defining parameter groups
lr: external learning rate
eps: regularization constants for square gradient and parameter scale respectively
eps_scale: regularization constants for parameter scale respectively
clip_threshold: threshold of root-mean-square of final gradient update
decay_rate: coefficient used to compute running averages of square gradient
beta1: coefficient used for computing running averages of gradient
weight_decay: weight decay
scale_parameter: if True, relative step learning rate is scaled by root-mean-square of parameter
warmup_init: time-dependent learning rate computation depends on whether warm-up initialization is being used

#### step[[timm.optim.Adafactor.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adafactor.py#L140)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.AdafactorBigVision[[timm.optim.AdafactorBigVision]]

```python
timm.optim.AdafactorBigVision(params: ParamsT, lr: float = 1.0, min_dim_size_to_factor: int = 16, decay_rate: float = 0.8, decay_offset: int = 0, beta2_cap: float = 0.999, momentum: typing.Optional[float] = 0.9, momentum_dtype: typing.Union[str, torch.dtype] = torch.bfloat16, eps: typing.Optional[float] = None, weight_decay: float = 0.0, clipping_threshold: typing.Optional[float] = None, unscaled_wd: bool = False, caution: bool = False, corrected_weight_decay: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adafactor_bv.py#L49)

PyTorch implementation of BigVision's Adafactor variant with both single and multi tensor implementations.

Adapted from https://github.com/google-research/big_vision by Ross Wightman

#### timm.optim.Adahessian[[timm.optim.Adahessian]]

```python
timm.optim.Adahessian(params, lr = 0.1, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.0, hessian_power = 1.0, update_each = 1, n_samples = 1, avg_conv_kernel = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adahessian.py#L9)

**Parameters:**

params (iterable) : iterable of parameters to optimize or dicts defining parameter groups

lr (float, optional) : learning rate (default: 0.1)

betas ((float, float), optional) : coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))

eps (float, optional) : term added to the denominator to improve numerical stability (default: 1e-8)

weight_decay (float, optional) : weight decay (L2 penalty) (default: 0.0)

hessian_power (float, optional) : exponent of the hessian trace (default: 1.0)

update_each (int, optional) : compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)

n_samples (int, optional) : how many times to sample `z` for the approximation of the hessian trace (default: 1)

Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"

#### get_params[[timm.optim.Adahessian.get_params]]

```python
get_params()
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adahessian.py#L74)

Gets all parameters in all param_groups with gradients

#### set_hessian[[timm.optim.Adahessian.set_hessian]]

```python
set_hessian()
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adahessian.py#L90)

Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.

#### step[[timm.optim.Adahessian.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adahessian.py#L119)

**Parameters:**

closure (callable, optional) : a closure that reevaluates the model and returns the loss (default -- None)

Performs a single optimization step.

#### zero_hessian[[timm.optim.Adahessian.zero_hessian]]

```python
zero_hessian()
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adahessian.py#L81)

Zeros out the accumulated hessian traces.

#### timm.optim.AdamP[[timm.optim.AdamP]]

```python
timm.optim.AdamP(params, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0, delta = 0.1, wd_ratio = 0.1, nesterov = False, caution = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adamp.py#L64)

#### timm.optim.AdamWLegacy[[timm.optim.AdamWLegacy]]

```python
timm.optim.AdamWLegacy(params: ParamsT, lr: float = 0.001, betas: typing.Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, caution: bool = False, corrected_weight_decay: bool = False, maximize: bool = False, foreach: typing.Optional[bool] = None, capturable: bool = False, clipping_threshold: typing.Optional[float] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adamw.py#L23)

**Parameters:**

params : iterable of parameters to optimize or dicts defining parameter groups

lr : learning rate

betas : coefficients used for computing running averages of gradient and its square

eps : term added to the denominator to improve numerical stability

weight_decay : weight decay coefficient

amsgrad : whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`

caution : apply caution when using AdamW

corrected_weight_decay : apply corrected weight decay (lr**2 / max_lr)

maximize : maximize the params based on the objective, instead of minimizing

foreach : whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will try to use foreach over for-loop implementation on CUDA, since it is faster in general.

capturable : whether this instance is safe to capture in a CUDA graph. Passing True can impair ungraphed performance, so if you don't intend to graph capture this instance, leave it False

clipping_threshold : RMS update clipping threshold (StableAdamW, https://arxiv.org/abs/2304.13013), the update is scaled by 1 / max(1, rms / clipping_threshold) w/ rms the RMS of grad / (sqrt(v_hat) + eps). Damps the update of params w/ a sudden gradient increase relative to their second moment estimate. 1.0 as per the paper, None (default) disables. As in Adafactor, the decoupled weight decay is not clipped (StableAdamW also scales the decay by the clipped lr).

Implements AdamW algorithm.

NOTE: This impl has been deprecated in favour of torch.optim.AdamW and remains as a reference

References:
- Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
- Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
- On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ

#### step[[timm.optim.AdamWLegacy.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adamw.py#L116)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.Adan[[timm.optim.Adan]]

```python
timm.optim.Adan(params, lr: float = 0.001, betas: typing.Tuple[float, float, float] = (0.98, 0.92, 0.99), eps: float = 1e-08, weight_decay: float = 0.0, no_prox: bool = False, caution: bool = False, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adan.py#L48)

**Parameters:**

params : Iterable of parameters to optimize or dicts defining parameter groups.

lr : Learning rate.

betas : Coefficients used for first- and second-order moments.

eps : Term added to the denominator to improve numerical stability.

weight_decay : Decoupled weight decay (L2 penalty)

no_prox : How to perform the weight decay

caution : Enable caution from 'Cautious Optimizers'

foreach : If True would use torch._foreach implementation. Faster but uses slightly more memory.

Implements a pytorch variant of Adan.

Adan was proposed in Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models
https://arxiv.org/abs/2208.06677

#### step[[timm.optim.Adan.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adan.py#L119)

Performs a single optimization step.

#### timm.optim.Adopt[[timm.optim.Adopt]]

```python
timm.optim.Adopt(params: ParamsT, lr: typing.Union[float, torch.Tensor] = 0.001, betas: typing.Tuple[float, float] = (0.9, 0.9999), eps: float = 1e-06, clip_exp: typing.Optional[float] = 0.333, weight_decay: float = 0.0, decoupled: bool = False, corrected_weight_decay: bool = False, caution: bool = False, foreach: typing.Optional[bool] = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adopt.py#L60)

ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate: https://arxiv.org/abs/2411.02853

#### step[[timm.optim.Adopt.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/adopt.py#L197)

**Parameters:**

closure (Callable, optional) : A closure that reevaluates the model and returns the loss.

Perform a single optimization step.

#### timm.optim.kron.Kron[[timm.optim.kron.Kron]]

```python
timm.optim.kron.Kron(params: ParamsT, lr: float = 0.001, momentum: float = 0.9, weight_decay: float = 0.0, preconditioner_update_probability: typing.Union[typing.Callable, float, NoneType] = None, max_size_triangular: int = 2048, min_ndim_triangular: int = 2, memory_save_mode: typing.Optional[str] = None, momentum_into_precond_update: bool = True, precond_lr: float = 0.1, precond_init_scale: float = 1.0, mu_dtype: typing.Optional[torch.dtype] = None, precond_dtype: typing.Optional[torch.dtype] = None, decoupled_decay: bool = False, corrected_weight_decay: bool = False, flatten: bool = False, flatten_start_dim: int = 2, flatten_end_dim: int = -1, stochastic_weight_decay: bool = False, deterministic: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/kron.py#L82)

**Parameters:**

params : Iterable of parameters to optimize or dicts defining parameter groups.

lr : Learning rate.

momentum : Momentum parameter.

weight_decay : Weight decay.

preconditioner_update_probability : Probability of updating the preconditioner. If None, defaults to a schedule that anneals from 1.0 to 0.03 by 4000 steps.

max_size_triangular : Max size for dim's preconditioner to be triangular.

min_ndim_triangular : Minimum number of dimensions a layer needs to have triangular preconditioners.

memory_save_mode : 'one_diag', 'smart_one_diag', or 'all_diag', None is default to set all preconditioners to be triangular, 'one_diag' sets the largest or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners to be diagonal.

momentum_into_precond_update : whether to send momentum into preconditioner update instead of raw gradients.

mu_dtype : Dtype of the momentum accumulator.

precond_dtype : Dtype of the preconditioner.

decoupled_decay : AdamW style decoupled weight decay

corrected_weight_decay : apply corrected weight decay when using decoupled_decay (lr**2 / max_lr)

flatten : Flatten dimensions instead of fully relying on expressions for higher rank params

flatten_start_dim : Start of flatten range, defaults to 2. Seems good tradeoff for ConvNets.

flatten_end_dim : End of flatten range, defaults to -1.

stochastic_weight_decay : Enable random modulation of weight decay

deterministic : Deterministic behaviour across save / load (resume). FIXME slow, needs work

Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

#### timm.optim.Lamb[[timm.optim.Lamb]]

```python
timm.optim.Lamb(params: ParamsT, lr: float = 0.001, bias_correction: bool = True, betas: typing.Tuple[float, float] = (0.9, 0.999), eps: float = 1e-06, weight_decay: float = 0.01, grad_averaging: bool = True, max_grad_norm: typing.Optional[float] = 1.0, trust_clip: bool = False, always_adapt: bool = False, caution: bool = False, decoupled_decay: bool = False, corrected_weight_decay: bool = False, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lamb.py#L71)

**Parameters:**

params : Iterable of parameters to optimize or dicts defining parameter groups.

lr : Learning rate

betas : Coefficients used for computing running averages of gradient and its norm.

eps : Term added to the denominator to improve numerical stability.

weight_decay : Weight decay

grad_averaging : Whether apply (1-beta2) to grad when calculating running averages of gradient.

max_grad_norm : Value used to clip global grad norm.

trust_clip : Enable LAMBC trust ratio clipping.

always_adapt : Apply adaptive learning rate to 0.0 weight decay parameter.

caution : Apply caution.

decoupled : apply decoupled weight decay

corrected_weight_decay : apply corrected weight decay (lr**2 / max_lr) when using decoupled_decay

foreach : use the foreach (multi-tensor) impl, by default (None) used unless unsupported

Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

LAMB was proposed in:
- Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:  https://arxiv.org/abs/1904.00962
- On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ

#### step[[timm.optim.Lamb.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lamb.py#L169)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.LaProp[[timm.optim.LaProp]]

```python
timm.optim.LaProp(params: ParamsT, lr: float = 0.0004, betas: typing.Tuple[float, float] = (0.9, 0.999), eps: float = 1e-15, weight_decay: float = 0.0, caution: bool = False, corrected_weight_decay: bool = False, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/laprop.py#L32)

**Parameters:**

foreach : use the foreach (multi-tensor) impl, by default (None) used unless unsupported (see laprop())

LaProp Optimizer

Paper: LaProp: Separating Momentum and Adaptivity in Adam, https://arxiv.org/abs/2002.04839

#### step[[timm.optim.LaProp.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/laprop.py#L106)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.Lars[[timm.optim.Lars]]

```python
timm.optim.Lars(params, lr = 1.0, momentum = 0, dampening = 0, weight_decay = 0, nesterov = False, trust_coeff = 0.001, eps = 1e-08, trust_clip = False, always_adapt = False, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lars.py#L22)

**Parameters:**

params (iterable) : iterable of parameters to optimize or dicts defining parameter groups.

lr (float, optional) : learning rate (default: 1.0).

momentum (float, optional) : momentum factor (default: 0)

weight_decay (float, optional) : weight decay (L2 penalty) (default: 0)

dampening (float, optional) : dampening for momentum (default: 0)

nesterov (bool, optional) : enables Nesterov momentum (default: False)

trust_coeff (float) : trust coefficient for computing adaptive lr / trust_ratio (default: 0.001)

eps (float) : eps for division denominator (default: 1e-8)

trust_clip (bool) : enable LARC trust ratio clipping (default: False)

always_adapt (bool) : always apply LARS LR adapt, otherwise only when group weight_decay != 0 (default: False)

foreach (bool, optional) : use the foreach (multi-tensor) impl, by default (None) used unless unsupported

LARS for PyTorch

Paper: `Large batch training of Convolutional Networks` - https://arxiv.org/pdf/1708.03888.pdf

#### step[[timm.optim.Lars.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lars.py#L84)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.Lion[[timm.optim.Lion]]

```python
timm.optim.Lion(params: ParamsT, lr: float = 0.0001, betas: typing.Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, caution: bool = False, corrected_weight_decay: bool = False, maximize: bool = False, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lion.py#L34)

Implements Lion algorithm.

#### step[[timm.optim.Lion.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lion.py#L86)

**Parameters:**

closure : A closure that reevaluates the model and returns the loss.

**Returns:**

the loss.

Performs a single optimization step.

#### timm.optim.Lookahead[[timm.optim.Lookahead]]

```python
timm.optim.Lookahead(base_optimizer, alpha = 0.5, k = 6)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/lookahead.py#L15)

#### timm.optim.MADGRAD[[timm.optim.MADGRAD]]

```python
timm.optim.MADGRAD(params: typing.Any, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0, eps: float = 1e-06, decoupled_decay: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/madgrad.py#L24)

**Parameters:**

params (iterable) : Iterable of parameters to optimize or dicts defining parameter groups.

lr (float) : Learning rate (default: 1e-2).

momentum (float) : Momentum value in  the range [0,1) (default: 0.9).

weight_decay (float) : Weight decay, i.e. a L2 penalty (default: 0).

eps (float) : Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).

MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
Optimization.

.. _MADGRAD: https://arxiv.org/abs/2101.11075

MADGRAD is a general purpose optimizer that can be used in place of SGD or
Adam may converge faster and generalize better. Currently GPU-only.
Typically, the same learning rate schedule that is used for SGD or Adam may
be used. The overall learning rate is not comparable to either method and
should be determined by a hyper-parameter sweep.

MADGRAD requires less weight decay than other methods, often as little as
zero. Momentum values used for SGD or Adam's beta1 should work here also.

On sparse problems both weight_decay and momentum should be set to 0.

#### step[[timm.optim.MADGRAD.step]]

```python
step(closure: typing.Optional[typing.Callable[[], float]] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/madgrad.py#L90)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.Mars[[timm.optim.Mars]]

```python
timm.optim.Mars(params: ParamsT, lr: float = 0.003, betas: typing.Tuple[float, float] = (0.9, 0.99), eps: float = 1e-08, weight_decay: float = 0.0, gamma: float = 0.025, mars_type: str = 'adamw', optimize_1d: bool = False, lr_1d_factor: float = 1.0, betas_1d: typing.Optional[typing.Tuple[float, float]] = None, caution: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/mars.py#L90)

MARS Optimizer

Paper: MARS: Unleashing the Power of Variance Reduction for Training Large Models
https://arxiv.org/abs/2411.10438

#### step[[timm.optim.Mars.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/mars.py#L140)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.Muon[[timm.optim.Muon]]

```python
timm.optim.Muon(params: ParamsT, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95, nesterov: bool = False, ns_steps: int = 5, ns_coefficients: typing.Union[str, typing.Tuple[float, float, float], typing.List[typing.Tuple[float, float, float]]] = 'quintic', eps: float = 1e-07, safety_factor: float = 1.0, adjust_lr_fn: typing.Optional[str] = 'match_rms_adamw', conv_mode: str = 'flatten', normalize_spatial: bool = True, adamw_lr: typing.Optional[float] = None, fallback_lr_scale: float = 1.0, betas: typing.Tuple[float, float] = (0.9, 0.95), algo: str = 'muon', scale_eps: bool = False, capturable: bool = False, verbose: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/muon.py#L654)

Muon - MomentUm Orthogonalized by Newton-schulz

Combines Muon for 2D+ parameters (weight matrices) with AdamW for 1D parameters (biases, norms) and
parameter groups with 'use_fallback=True' set (or 'use_muon=False' for compatibility).

Supports two algorithms:
- "muon": Standard Muon algorithm with momentum + orthogonalization
- "adamuon": AdaMuon algorithm that adds element-wise second moment estimation
  to orthogonalized directions for Adam-like adaptive scaling

#### step[[timm.optim.Muon.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/muon.py#L798)

Performs a single optimization step.

#### timm.optim.NAdamLegacy[[timm.optim.NAdamLegacy]]

```python
timm.optim.NAdamLegacy(params, lr = 0.002, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0, schedule_decay = 0.004)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/nadam.py#L7)

**Parameters:**

params (iterable) : iterable of parameters to optimize or dicts defining parameter groups

lr (float, optional) : learning rate (default: 2e-3)

betas (Tuple[float, float], optional) : coefficients used for computing running averages of gradient and its square

eps (float, optional) : term added to the denominator to improve numerical stability (default: 1e-8)

weight_decay (float, optional) : weight decay (L2 penalty) (default: 0)

schedule_decay (float, optional) : momentum schedule decay (default: 4e-3)

Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

NOTE: This impl has been deprecated in favour of torch.optim.NAdam and remains as a reference

It has been proposed in `Incorporating Nesterov Momentum into Adam`__.

__ http://cs229.stanford.edu/proj2015/054_report.pdf
__ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

Originally taken from: https://github.com/pytorch/pytorch/pull/1408
NOTE: Has potential issues but does work well on some problems.

#### step[[timm.optim.NAdamLegacy.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/nadam.py#L52)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.NAdamW[[timm.optim.NAdamW]]

```python
timm.optim.NAdamW(params: ParamsT, lr: float = 0.001, betas: typing.Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, caution: bool = False, corrected_weight_decay: bool = False, maximize: bool = False, foreach: typing.Optional[bool] = None, capturable: bool = False, clipping_threshold: typing.Optional[float] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/nadamw.py#L25)

**Parameters:**

params : iterable of parameters to optimize or dicts defining parameter groups

lr : learning rate

betas : coefficients used for computing running averages of gradient and its square

eps : term added to the denominator to improve numerical stability

weight_decay : weight decay coefficient

caution : enable caution

corrected_weight_decay : apply corrected weight decay (lr**2 / max_lr)

maximize : maximize the params based on the objective, instead of minimizing

foreach : whether foreach implementation of optimizer is used. If unspecified (None), the foreach implementation is used on CUDA (devices w/ foreach kernels) since it is faster in general.

capturable : whether this instance is safe to capture in a CUDA graph

clipping_threshold : RMS update clipping threshold (StableAdamW, https://arxiv.org/abs/2304.13013), the update is scaled by 1 / max(1, rms / clipping_threshold) w/ rms the RMS of grad / (sqrt(v_hat) + eps). Damps the update of params w/ a sudden gradient increase relative to their second moment estimate. 1.0 as per the paper, None (default) disables. As in Adafactor, the decoupled weight decay is not clipped (StableAdamW also scales the decay by the clipped lr).

Implements NAdamW algorithm.

See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
the NAdam algorithm (there is also a comment in the code which highlights
the only difference of NAdamW and AdamW).

For further details regarding the algorithm we refer to
- Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
- On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ

#### step[[timm.optim.NAdamW.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/nadamw.py#L113)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.NvNovoGrad[[timm.optim.NvNovoGrad]]

```python
timm.optim.NvNovoGrad(params, lr = 0.001, betas = (0.95, 0.98), eps = 1e-08, weight_decay = 0, grad_averaging = False, amsgrad = False, decoupled_decay = False, corrected_weight_decay = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/nvnovograd.py#L17)

**Parameters:**

params (iterable) : iterable of parameters to optimize or dicts defining parameter groups

lr (float, optional) : learning rate (default: 1e-3)

betas (Tuple[float, float], optional) : coefficients used for computing running averages of gradient and its square (default: (0.95, 0.98))

eps (float, optional) : term added to the denominator to improve numerical stability (default: 1e-8)

weight_decay (float, optional) : weight decay (L2 penalty) (default: 0)

grad_averaging : gradient averaging

amsgrad (boolean, optional) : whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_ (default: False)

decoupled_decay : apply weight decay directly to the weights (lr * weight_decay per step) instead of adding it to the normalized gradient before the momentum update (as per the paper)

corrected_weight_decay : apply corrected weight decay (lr**2 / max_lr) when decoupled_decay is True

Implements Novograd algorithm.

#### step[[timm.optim.NvNovoGrad.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/nvnovograd.py#L90)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model

and returns the loss. --

Performs a single optimization step.

#### timm.optim.RAdamLegacy[[timm.optim.RAdamLegacy]]

```python
timm.optim.RAdamLegacy(params, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/radam.py#L16)

PyTorch RAdam optimizer

NOTE: This impl has been deprecated in favour of torch.optim.AdamW and remains as a reference

#### timm.optim.RMSpropTF[[timm.optim.RMSpropTF]]

```python
timm.optim.RMSpropTF(params: ParamsT, lr: float = 0.01, alpha: float = 0.9, eps: float = 1e-10, weight_decay: float = 0, momentum: float = 0.0, centered: bool = False, decoupled_decay: bool = False, corrected_weight_decay: bool = False, lr_in_momentum: bool = True, caution: bool = False, foreach: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/rmsprop_tf.py#L27)

**Parameters:**

params : iterable of parameters to optimize or dicts defining parameter groups

lr : learning rate

momentum : momentum factor

alpha : smoothing (decay) constant

eps : term added to the denominator to improve numerical stability

centered : if `True`, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

weight_decay : weight decay (L2 penalty) (default: 0)

decoupled_decay : decoupled weight decay as per https://arxiv.org/abs/1711.05101

corrected_weight_decay : apply corrected weight decay (lr**2 / max_lr) when decoupled_decay is True

lr_in_momentum : learning rate scaling is included in the momentum buffer update as per defaults in Tensorflow

caution : apply caution

foreach : use the foreach (multi-tensor) impl, by default (None) used unless unsupported (see rmsprop_tf())

Implements RMSprop algorithm (TensorFlow style epsilon)

NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
and a few other modifications to closer match Tensorflow for matching hyper-params.

Noteworthy changes include:
1. Epsilon applied inside square-root
2. square_avg initialized to ones
3. LR scaling of update accumulated in momentum buffer

Proposed by G. Hinton in his
[course](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

The centered version first appears in [Generating Sequences
With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf).

#### step[[timm.optim.RMSpropTF.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/rmsprop_tf.py#L112)

**Parameters:**

closure (callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

#### timm.optim.SGDP[[timm.optim.SGDP]]

```python
timm.optim.SGDP(params, lr = torch.optim.optimizer.required, momentum = 0, dampening = 0, weight_decay = 0, nesterov = False, eps = 1e-08, delta = 0.1, wd_ratio = 0.1, caution = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/sgdp.py#L22)

#### timm.optim.SGDW[[timm.optim.SGDW]]

```python
timm.optim.SGDW(params: ParamsT, lr: float = 0.001, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False, caution: bool = False, corrected_weight_decay: bool = False, maximize: bool = False, foreach: typing.Optional[bool] = None, differentiable: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/sgdw.py#L26)

#### step[[timm.optim.SGDW.step]]

```python
step(closure = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/optim/sgdw.py#L98)

**Parameters:**

closure (Callable, optional) : A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

### Learning Rate Schedulers
https://huggingface.co/docs/timm/v1.0.31/reference/schedulers.md
