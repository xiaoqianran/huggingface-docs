# Losses

This page contains the API reference documentation for the classification losses included in `timm`, and an
overview of how they are created and configured through the loss factory.

## Usage

Losses are created by name with [`create_classification_loss`](#timm.loss.create_classification_loss).
`multi_label` selects between single-label targets (class indices, or dense soft distributions from Mixup/CutMix)
and multi-label targets (dense independent binary labels shaped `(B, C)`), with `loss_type=None` the factory picks
`ce` or `bce` accordingly. Loss specific arguments are prefixed with the loss name (`asl_*`, `poly_*`, `db_*`, ...)
and ignored by the other losses. The factory raises on combinations a loss doesn't support, e.g. class weights for
losses other than `bce` / multi-label `asl`, `bce_*` or `class_counts` arguments for other losses, or a multi-label
only loss without `multi_label` (or a single-label only loss with it).

```python
import torch
import timm
from timm.data import Mixup
from timm.loss import PolyCrossEntropy, create_classification_loss

# Single-label, class index targets. Label smoothing selects LabelSmoothingCrossEntropy.
criterion = create_classification_loss(smoothing=0.1)
loss = criterion(torch.randn(8, 1000), torch.randint(0, 1000, (8,)))

# Mixup/CutMix output dense soft targets with the smoothing already applied, SoftTargetCrossEntropy.
mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=1000)
criterion = create_classification_loss(smoothing=0.1, soft_targets=True)
images, target = mixup_fn(torch.randn(8, 3, 224, 224), torch.randint(0, 1000, (8,)))
loss = criterion(torch.randn(8, 1000), target)

# Multi-label, dense (B, C) binary targets. BCE by default, other losses by name.
criterion = create_classification_loss(multi_label=True)
criterion = create_classification_loss('asl', multi_label=True, asl_gamma_neg=4., asl_gamma_pos=1., asl_clip=0.05)
loss = criterion(torch.randn(8, 6), torch.randint(0, 2, (8, 6)).float())

# The training tasks create their criterion with the factory, multi_label is set by the task.
from timm.task import MultiLabelClassificationTask
model = timm.create_model('resnet18', num_classes=6)
task = MultiLabelClassificationTask(model, criterion_kwargs=dict(loss_type='zlpr'))

# Losses can also be constructed directly, note the class defaults can differ from the factory's (below).
criterion = PolyCrossEntropy(epsilon=2.)
```

The training script (`train.py`) maps its options to the loss factory as follows:

| CLI option or condition | Factory argument or effect |
|---|---|
| `--loss` | `loss_type` |
| `--task multilabel` | `multi_label=True` |
| `--smoothing` | `smoothing`, default 0.1 for single-label and 0 for multi-label training |
| Mixup or CutMix enabled | `soft_targets=True` |
| `--asl-*` | Corresponding `asl_*` arguments |
| `--poly-*` | Corresponding `poly_*` arguments |
| `--twoway-*` | Corresponding `twoway_*` arguments |
| `--db-*` | Corresponding `db_*` arguments |
| `--bce-sum` | `bce_sum` |
| `--bce-target-thresh` | `bce_target_thresh` |
| `--aug-splits` | `jsd_splits` |
| `--class-stats` | Class counts for the class weight functions below, and `class_counts` / `num_samples` for `db` |
| `--loss-pos-weight` | `pos_weight`, resolved by the class weight functions below |
| `--loss-class-weight` | `class_weight`, resolved by the class weight functions below |
| `--loss-weight-*` | Options for the class weight functions below |

See the [training docs](../training_script#multi-label-classification) for the command line usage.

### Loss types

`LOSS_TYPES` holds the names accepted by the factory and `MULTI_LABEL_LOSS_TYPES` the ones usable with
`multi_label=True`.

| `loss_type` | Class | Targets | Notes |
|---|---|---|---|
| `ce` | `nn.CrossEntropyLoss`, [`LabelSmoothingCrossEntropy`](#timm.loss.cross_entropy.LabelSmoothingCrossEntropy), [`SoftTargetCrossEntropy`](#timm.loss.cross_entropy.SoftTargetCrossEntropy) | Single-label | Softmax cross-entropy, the class depends on `smoothing` and `soft_targets`. Single-label default |
| `bce` | [`BinaryCrossEntropy`](#timm.loss.binary_cross_entropy.BinaryCrossEntropy) | Both | Sigmoid BCE, multi-label default. Class index targets are one-hot encoded, `bce_target_thresh` binarizes soft targets, `bce_sum` sums over classes before averaging over the batch |
| `asl` | [`AsymmetricLossMultiLabel`](#timm.loss.asymmetric_loss.AsymmetricLossMultiLabel), [`AsymmetricLossSingleLabel`](#timm.loss.asymmetric_loss.AsymmetricLossSingleLabel) | Both | [Asymmetric Loss](https://arxiv.org/abs/2009.14119), separate focusing for positives (`asl_gamma_pos`) and negatives (`asl_gamma_neg`), a probability margin on negatives (`asl_clip`, multi-label only). The single-label (softmax) variant needs class index targets, no Mixup/CutMix |
| `poly` | [`PolyBinaryCrossEntropy`](#timm.loss.poly_loss.PolyBinaryCrossEntropy), [`PolyCrossEntropy`](#timm.loss.poly_loss.PolyCrossEntropy) | Both | [Poly-1](https://arxiv.org/abs/2204.12511), adds `poly_epsilon * (1 - p_t)` to BCE / CE, `0` reduces to BCE / CE. Optional focal (`poly_gamma`) and balance (`poly_alpha`) terms for multi-label |
| `jsd` | [`JsdCrossEntropy`](#timm.loss.jsd.JsdCrossEntropy) | Single-label | CE on the clean split + Jensen-Shannon divergence across `jsd_splits` augmentation splits ([AugMix](https://arxiv.org/abs/1912.02781)) |
| `twoway` | [`TwoWayLoss`](#timm.loss.two_way_loss.TwoWayLoss) | Multi-label | [Two-way Multi-Label Loss](https://staff.aist.go.jp/takumi.kobayashi/publication/2023/CVPR2023.pdf), temperatures `twoway_tp` / `twoway_tn`, binary targets only |
| `zlpr` | [`ZlprLoss`](#timm.loss.zlpr_loss.ZlprLoss) | Multi-label | [ZLPR](https://arxiv.org/abs/2208.02955), multi-label extension of softmax CE w/o hyperparameters, binary targets only |
| `db` | [`DistributionBalancedLoss`](#timm.loss.distribution_balanced_loss.DistributionBalancedLoss) | Multi-label | [Distribution-Balanced Loss](https://arxiv.org/abs/2007.09654) for long-tailed data, needs `class_counts` and `num_samples` |

### Label smoothing and soft targets

`smoothing` defaults to 0 in the factory. For single-label targets the target class gets `1 - s + s / C` and the
others `s / C` (`LabelSmoothingCrossEntropy` is equivalent to `nn.CrossEntropyLoss(label_smoothing=s)`), `bce` uses
the same one-hot smoothing for class index targets. For multi-label targets each binary label is moved towards 0.5,
`y * (1 - s) + 0.5 * s`.

`timm.data.Mixup` applies its `label_smoothing` while mixing and outputs dense targets (with `multi_label=True` it mixes
and smooths dense multi-label targets). Create the loss with `soft_targets=True` in this case:

- **`ce`** becomes `SoftTargetCrossEntropy`.
- **`bce`, multi-label `asl`, `poly` and `db`** don't smooth the targets again.
- **Single-label `asl`** raises with `soft_targets`.
- **`twoway` / `zlpr`** raise with either `soft_targets` or `smoothing` as they binarize targets at 0.5.

The classes have their own defaults when constructed directly, `BinaryCrossEntropy`, `LabelSmoothingCrossEntropy`
and `JsdCrossEntropy` default to `smoothing=0.1` and `AsymmetricLossSingleLabel` to `eps=0.1` (its label smoothing).

### Class weighting

`pos_weight` multiplies the positive term of each class, balancing positives against negatives within a class.
`class_weight` multiplies the whole loss of each class, balancing the classes against each other. Both take a single
value or one value per class and are supported by `bce` and multi-label `asl`. They can be given explicitly or derived
from per-class positive counts with [`compute_class_weights`](#timm.loss.compute_class_weights), which returns the loss
argument the weights are for and the weights. Classes without positives are counted as having one, and `max_weight` caps
the final weights.

| `method` | Argument | Weights for a class w/ `n` positives |
|---|---|---|
| `neg_pos` | `pos_weight` | `((num_samples - n) / n) ** power` |
| `inv_freq` | `class_weight` | `(num_samples / n) ** power`, normalized to a mean of 1 |
| `effective_num` | `class_weight` | `(1 - beta) / (1 - beta ** n)` ([class-balanced](https://arxiv.org/abs/1901.05555)), normalized to a mean of 1 |

The methods are in `CLASS_WEIGHT_METHODS`, and `CLASS_WEIGHT_METHOD_KINDS` maps each to its loss argument.

- [`load_class_stats`](#timm.loss.load_class_stats) reads the counts from a JSON file with `pos_counts` and
  `num_samples` entries, as written by `class_weights.py` in the repository root.
- [`load_class_weights`](#timm.loss.load_class_weights) parses explicit weights (a number, a comma separated string,
  a list, or a JSON / text file).
- [`resolve_class_weights`](#timm.loss.resolve_class_weights) accepts either a method name or explicit weights,
  as `train.py` does for its flags.

The `db` loss takes the counts directly.

```python
import torch
from timm.loss import compute_class_weights, create_classification_loss, resolve_class_weights

pos_counts = [120, 3400, 560, 45, 980, 2100]  # positives per class
num_samples = 5000

kind, weights = compute_class_weights(pos_counts, num_samples, method='neg_pos', power=0.5)  # kind == 'pos_weight'
criterion = create_classification_loss(multi_label=True, pos_weight=weights)

# Method names (w/ class stats) or explicit weights, class_stats can be loaded w/ load_class_stats('stats.json')
class_stats = (torch.tensor(pos_counts), num_samples)
class_weight = resolve_class_weights('effective_num', 'class_weight', class_stats=class_stats, beta=0.999)
pos_weight = resolve_class_weights('2.0', 'pos_weight')
criterion = create_classification_loss('asl', multi_label=True, class_weight=class_weight, pos_weight=pos_weight)

criterion = create_classification_loss('db', multi_label=True, class_counts=pos_counts, num_samples=num_samples)
```

### Loss notes

- **Precision.** The multi-label ASL, Poly-1, DB, Two-way and ZLPR losses compute in `float32` regardless of the
  logit dtype, low precision (`float16`) ASL probabilities would otherwise underflow its log clamp.
- **Reduction.** Multi-label ASL supports these `asl_reduction` values:

  | Value | Over classes | Over the batch | Notes |
  |---|---|---|---|
  | `batchmean` | Sum | Mean | Default |
  | `sum` | Sum | Sum | Original ASL reduction; loss scales with batch size |
  | `mean` | Mean | Mean | Like BCE |

  Single-label ASL sums over classes and averages over the batch, the factory only accepts `batchmean` for it.
  `bce`, multi-label `poly` and `db` average over batch and classes (`bce_sum` sums over classes for `bce`).

- **Two-way** contrasts the hardest negative logit with the hardest positive logit across the classes of each sample
  and across the samples of each class, only samples / classes with a positive contribute. Its logits are ranking
  scores, not calibrated sigmoid probabilities, so thresholded metrics (F1) are not meaningful, use mAP.
- **ZLPR** pushes every positive logit above every negative one with 0 as the decision threshold, so logits > 0
  (sigmoid > 0.5) predict positives.
- **DB** weights each (sample, class) term by the ratio of class-level to instance-level sampling probability,
  accounting for label co-occurrence, shifts the logits by a class prior (`db_init_bias`) and scales the negative
  logits (`db_neg_scale`). Counts are clamped to `[1, num_samples - 1]`. The re-balancing map parameters and the
  focal balance are class arguments, not exposed by the factory.
- **JSD** expects the batch as `jsd_splits` equal splits with the clean images first (as `fast_collate` arranges
  `AugMixDataset` samples), the CE term uses only the clean split. The JSD weight (`alpha=12`) is a class argument.

## API

### Factory functions[[timm.loss.create_classification_loss]]

#### timm.loss.create_classification_loss[[timm.loss.create_classification_loss]]

```python
timm.loss.create_classification_loss(loss_type: typing.Optional[str] = None, multi_label: bool = False, smoothing: float = 0.0, soft_targets: bool = False, class_weight: typing.Union[torch.Tensor, float, typing.Sequence[float], NoneType] = None, pos_weight: typing.Union[torch.Tensor, float, typing.Sequence[float], NoneType] = None, bce_target_thresh: typing.Optional[float] = None, bce_sum: bool = False, asl_gamma_pos: float = 1.0, asl_gamma_neg: float = 4.0, asl_clip: typing.Optional[float] = 0.05, asl_reduction: str = 'batchmean', poly_epsilon: float = 2.0, poly_gamma: float = 0.0, poly_alpha: typing.Optional[float] = None, jsd_splits: int = 0, twoway_tp: float = 4.0, twoway_tn: float = 1.0, class_counts: typing.Union[torch.Tensor, typing.Sequence[float], NoneType] = None, num_samples: typing.Optional[int] = None, db_neg_scale: float = 2.0, db_init_bias: float = 0.05, db_focal_gamma: float = 2.0)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/_loss_factory.py#L31)

**Parameters:**

loss_type : One of LOSS_TYPES, None selects 'bce' for multi-label and 'ce' for single-label targets. 'ce': softmax cross-entropy (label smoothing / soft target variants as needed), 'bce': binary cross-entropy (bce_* args), 'asl': Asymmetric Loss, AsymmetricLossMultiLabel for multi-label targets, else the softmax AsymmetricLossSingleLabel, class index targets only, no Mixup/CutMix (asl_* args), 'poly': Poly-1 loss, PolyBinaryCrossEntropy (sigmoid) for multi-label targets, else the softmax PolyCrossEntropy (poly_* args), 'jsd': Jensen-Shannon divergence + CE over augmentation splits, single-label (jsd_splits), 'twoway': Two-way loss, binary multi-label targets only (twoway_* args), 'zlpr': ZLPR loss, binary multi-label targets only, 'db': Distribution-Balanced loss, multi-label (class_counts, num_samples, db_* args).

multi_label : Targets are dense independent binary labels shaped (B, C).

smoothing : Label smoothing factor. Ignored when soft_targets is set, since Mixup/CutMix already apply it while producing the soft targets.

soft_targets : Targets arrive as dense soft distributions from Mixup/CutMix rather than class indices.

class_weight : Per-class loss weight, a single value or one per class ('bce' and multi-label 'asl').

pos_weight : Positive term weight, a single value or one per class ('bce' and multi-label 'asl').

bce_target_thresh : Binarize soft targets above this value ('bce' only).

bce_sum : Sum the loss over classes before averaging over the batch ('bce' only).

asl_gamma_pos : Focusing parameter for positive targets ('asl' only).

asl_gamma_neg : Focusing parameter for negative targets ('asl' only).

asl_clip : Probability margin shifted onto negatives, None or 0 disables (multi-label 'asl' only).

asl_reduction : 'batchmean' (sum over classes, mean over batch), 'sum' (original ASL), or 'mean' (multi-label 'asl' only, single-label ASL always uses 'batchmean').

poly_epsilon : Poly-1 coefficient, 0 reduces to CE / BCE ('poly' only).

poly_gamma : Focusing parameter, 0 disables the focal term (multi-label 'poly' only).

poly_alpha : Positive / negative balance weight, None disables it (multi-label 'poly' only).

jsd_splits : Number of augmentation splits, must be > 1 ('jsd' only).

twoway_tp : Positive logit temperature ('twoway' only).

twoway_tn : Negative logit temperature ('twoway' only).

class_counts : Positive training sample count per class ('db' only).

num_samples : Number of training samples class_counts were taken over ('db' only).

db_neg_scale : Negative-tolerant logit scale, 1 disables ('db' only).

db_init_bias : Class prior logit shift factor, 0 disables ('db' only).

db_focal_gamma : Focusing parameter, 0 disables the focal term ('db' only).

Create the training criterion for a classification task.

### Class weight functions[[timm.loss.compute_class_weights]]

#### timm.loss.compute_class_weights[[timm.loss.compute_class_weights]]

```python
timm.loss.compute_class_weights(pos_counts: typing.Union[torch.Tensor, typing.Sequence[float]], num_samples: int, method: str = 'neg_pos', power: float = 1.0, beta: float = 0.999, max_weight: typing.Optional[float] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/class_weights.py#L19)

**Parameters:**

pos_counts : Number of positive samples per class (the class frequency for single-label targets).

num_samples : Number of samples the counts were taken over.

method : One of 'neg_pos': pos_weight = (negatives / positives) ** power, balances positive and negative terms of each class, 'inv_freq': class_weight proportional to (num_samples / positives) ** power, mean 1, 'effective_num': class-balanced class_weight from the effective number of samples (1 - beta ** n) / (1 - beta), https://arxiv.org/abs/1901.05555, mean 1.

power : Exponent for 'neg_pos' and 'inv_freq', 0.5 gives a softer (square root) re-weighting.

beta : Effective number hyperparameter for 'effective_num'.

max_weight : Cap applied to the final weights.

**Returns:**

The loss argument the weights are for ('pos_weight' or 'class_weight') and the float32 weights.

Compute per-class loss weights from positive label counts.

Classes without any positives are treated as having one, and every weight is capped at max_weight if set.

#### timm.loss.load_class_stats[[timm.loss.load_class_stats]]

```python
timm.loss.load_class_stats(path: typing.Optional[str])
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/class_weights.py#L71)

**Parameters:**

path : JSON file with 'pos_counts' and 'num_samples' entries. None or an empty string returns None.

**Returns:**

The float32 counts and the number of samples they were taken over.

Load per-class positive counts and the sample count from a class_weights.py stats file.

#### timm.loss.load_class_weights[[timm.loss.load_class_weights]]

```python
timm.loss.load_class_weights(value: typing.Union[str, float, typing.Sequence[float], torch.Tensor, NoneType], key: str = 'class_weight')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/class_weights.py#L94)

**Parameters:**

value : Weight value, list, or path. None or an empty string returns None.

key : Entry to read from JSON objects, e.g. 'pos_weight' or 'class_weight'.

**Returns:**

A float for a single value (broadcast over classes), else a 1-D float32 tensor.

Parse loss weights from a number, a comma separated list, or a file.

Files are JSON (a list, or an object holding the list under `key`, as written by class_weights.py) or text
with comma / whitespace separated numbers.

#### timm.loss.resolve_class_weights[[timm.loss.resolve_class_weights]]

```python
timm.loss.resolve_class_weights(value: typing.Union[str, float, typing.Sequence[float], torch.Tensor, NoneType], kind: str = 'class_weight', class_stats: typing.Optional[typing.Tuple[torch.Tensor, int]] = None, power: float = 1.0, beta: float = 0.999, max_weight: typing.Optional[float] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/class_weights.py#L140)

**Parameters:**

value : A method in CLASS_WEIGHT_METHODS, or explicit weights for load_class_weights().

kind : The loss argument being resolved, 'pos_weight' or 'class_weight'.

class_stats : (counts, num_samples) from load_class_stats(), required for method names.

power : See compute_class_weights().

beta : See compute_class_weights().

max_weight : See compute_class_weights().

Resolve loss weights given as a method name (computed from class stats) or as explicit weights.

### Single-label Loss Classes[[timm.loss.LabelSmoothingCrossEntropy]]

#### timm.loss.LabelSmoothingCrossEntropy[[timm.loss.LabelSmoothingCrossEntropy]]

```python
timm.loss.LabelSmoothingCrossEntropy(smoothing = 0.1)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/cross_entropy.py#L11)

NLL loss with label smoothing.

#### timm.loss.SoftTargetCrossEntropy[[timm.loss.SoftTargetCrossEntropy]]

```python
timm.loss.SoftTargetCrossEntropy()
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/cross_entropy.py#L29)

#### timm.loss.AsymmetricLossSingleLabel[[timm.loss.AsymmetricLossSingleLabel]]

```python
timm.loss.AsymmetricLossSingleLabel(gamma_pos = 1, gamma_neg = 4, eps: float = 0.1, reduction = 'mean')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/asymmetric_loss.py#L102)

#### forward[[timm.loss.AsymmetricLossSingleLabel.forward]]

```python
forward(inputs, target, reduction = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/asymmetric_loss.py#L113)

"
Parameters
----------
x: input logits
y: targets (1-hot vector)

#### timm.loss.JsdCrossEntropy[[timm.loss.JsdCrossEntropy]]

```python
timm.loss.JsdCrossEntropy(num_splits = 3, alpha = 12, smoothing = 0.1)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/jsd.py#L8)

Jensen-Shannon Divergence + Cross-Entropy Loss

Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
https://arxiv.org/abs/1912.02781

Hacked together by / Copyright 2020 Ross Wightman

#### timm.loss.PolyCrossEntropy[[timm.loss.PolyCrossEntropy]]

```python
timm.loss.PolyCrossEntropy(epsilon: float = 2.0, smoothing: float = 0.0, reduction: str = 'mean')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/poly_loss.py#L19)

**Parameters:**

epsilon : Poly-1 coefficient, 0 is plain cross-entropy.

smoothing : Label smoothing applied to class index targets. Dense targets are assumed to be softened upstream (e.g. by Mixup/CutMix).

reduction : 'mean' or 'sum' over the batch, or 'none' for per-sample losses.

Poly-1 softmax cross-entropy, CE + epsilon * (1 - p_t), for single-label classification.

p_t is the target-weighted softmax probability sum_c(t_c * p_c), so label smoothing and soft (Mixup/CutMix)
targets apply to both terms, as in the reference.

#### forward[[timm.loss.PolyCrossEntropy.forward]]

```python
forward(x: Tensor, target: Tensor)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/poly_loss.py#L41)

**Parameters:**

x : Input logits (batch_size, num_classes).

target : Class indices (batch_size,) or dense target distributions (batch_size, num_classes).

### Multi-label Loss Classes[[timm.loss.BinaryCrossEntropy]]

#### timm.loss.BinaryCrossEntropy[[timm.loss.BinaryCrossEntropy]]

```python
timm.loss.BinaryCrossEntropy(smoothing = 0.1, target_threshold: typing.Optional[float] = None, weight: typing.Optional[torch.Tensor] = None, reduction: str = 'mean', sum_classes: bool = False, pos_weight: typing.Union[torch.Tensor, float, NoneType] = None, smooth_dense: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/binary_cross_entropy.py#L12)

**Parameters:**

smoothing : Label smoothing applied while one-hot encoding index targets. Dense targets are assumed to be softened upstream (e.g. by Mixup/CutMix) unless smooth_dense is set.

target_threshold : Binarize (soft) targets above this value.

weight : Per-class rescaling weight.

reduction : Loss reduction.

sum_classes : Sum the loss over classes before averaging over the batch.

pos_weight : Positive class weight.

smooth_dense : Also apply smoothing to targets that already match the logit shape (multi-label), moving each binary target towards 0.5.

BCE with optional one-hot from dense targets, label smoothing, thresholding
NOTE for experiments comparing CE to BCE /w label smoothing, may remove

#### timm.loss.AsymmetricLossMultiLabel[[timm.loss.AsymmetricLossMultiLabel]]

```python
timm.loss.AsymmetricLossMultiLabel(gamma_neg: float = 4, gamma_pos: float = 1, clip: typing.Optional[float] = 0.05, eps: float = 1e-08, disable_torch_grad_focal_loss: bool = False, smoothing: float = 0.0, reduction: str = 'batchmean', weight: typing.Optional[torch.Tensor] = None, pos_weight: typing.Union[torch.Tensor, float, NoneType] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/asymmetric_loss.py#L7)

**Parameters:**

gamma_neg : Focusing parameter for negative targets.

gamma_pos : Focusing parameter for positive targets.

clip : Probability margin shifted onto negatives (asymmetric clipping), disabled if None or 0.

eps : Minimum probability clamped to before the log.

disable_torch_grad_focal_loss : Don't backprop through the focusing weights.

smoothing : Label smoothing, moves each (dense, independent binary) target towards 0.5.

reduction : 'batchmean' sums over classes and averages over the batch, 'sum' sums over batch and classes (the original ASL reduction, scales with batch size), 'mean' averages over batch and classes, 'none' keeps the per-element loss.

weight : Per-class rescaling weight.

pos_weight : Weight of the positive term, a single value or one per class.

Asymmetric Loss (ASL) for multi-label classification, https://arxiv.org/abs/2009.14119

#### forward[[timm.loss.AsymmetricLossMultiLabel.forward]]

```python
forward(x: Tensor, y: Tensor)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/asymmetric_loss.py#L52)

**Parameters:**

x : Input logits (batch_size, num_classes).

y : Dense (multi-hot or soft) targets (batch_size, num_classes).

#### timm.loss.PolyBinaryCrossEntropy[[timm.loss.PolyBinaryCrossEntropy]]

```python
timm.loss.PolyBinaryCrossEntropy(epsilon: float = 2.0, gamma: float = 0.0, alpha: typing.Optional[float] = None, smoothing: float = 0.0, reduction: str = 'mean')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/poly_loss.py#L64)

**Parameters:**

epsilon : Poly-1 coefficient, 0 is plain BCE (or focal loss).

gamma : Focusing parameter, 0 disables the focal term.

alpha : Positive / negative balance weight, None disables it.

smoothing : Label smoothing, moves each (dense, independent binary) target towards 0.5.

reduction : 'mean' averages over batch and classes (as BCE), 'batchmean' sums over classes and averages over the batch, 'sum' sums over both, 'none' keeps the per-element loss.

Poly-1 sigmoid (focal) loss, FL + epsilon * (1 - p_t) ** (gamma + 1), for multi-label classification.

With gamma = 0 this is BCE + epsilon * (1 - p_t) per class. With alpha set, positives are weighted by alpha
and negatives by 1 - alpha in both terms, as in the reference.

#### forward[[timm.loss.PolyBinaryCrossEntropy.forward]]

```python
forward(x: Tensor, target: Tensor)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/poly_loss.py#L98)

**Parameters:**

x : Input logits (batch_size, num_classes).

target : Dense (multi-hot or soft) targets (batch_size, num_classes).

#### timm.loss.DistributionBalancedLoss[[timm.loss.DistributionBalancedLoss]]

```python
timm.loss.DistributionBalancedLoss(class_counts: typing.Union[torch.Tensor, typing.Sequence[float]], num_samples: int, map_alpha: float = 0.1, map_beta: float = 10.0, map_mu: float = 0.2, neg_scale: float = 2.0, init_bias: float = 0.05, focal_gamma: float = 2.0, focal_balance: float = 2.0, smoothing: float = 0.0, reduction: str = 'mean')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/distribution_balanced_loss.py#L16)

**Parameters:**

class_counts : Positive sample count per class over the training set.

num_samples : Number of training samples the counts were taken over.

map_alpha : Minimum weight of the re-balancing map.

map_beta : Scale of the re-balancing map.

map_mu : Threshold of the re-balancing map.

neg_scale : Negative logit scale (lambda), 1 disables the negative-tolerant scaling.

init_bias : Class prior logit shift factor (kappa), 0 disables it.

focal_gamma : Focusing parameter, 0 disables the focal term (and focal_balance).

focal_balance : Constant scale of the focal loss.

smoothing : Label smoothing, moves each target towards 0.5.

reduction : 'mean' averages over batch and classes (the reference scale), 'batchmean' sums over classes and averages over the batch, 'sum' sums over both, 'none' keeps the per-element loss.

Distribution-Balanced (DB) loss for long-tailed multi-label classification.

Sigmoid BCE with two additions:
* Re-balanced weighting: each (sample, class) term is weighted by the ratio of the class-level sampling
  probability 1 / n_i to the sample's instance-level one sum_j(y_j / n_j), which accounts for the label
  co-occurrence of each sample, smoothed as alpha + sigmoid(beta * (r - mu)).
* Negative-tolerant regularization: logits are shifted by a class prior bias, and negative terms use a
  scaled logit, softplus(neg_scale * z) / neg_scale, so the many negatives of rare classes are not
  over-suppressed.
An optional focal term (1 - p_t) ** gamma follows the reference DB configuration.

Soft targets (Mixup/CutMix) weight the positive and negative terms, the re-balancing weights use the
targets as given. Both match the reference for binary targets.

#### forward[[timm.loss.DistributionBalancedLoss.forward]]

```python
forward(x: Tensor, target: Tensor)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/distribution_balanced_loss.py#L84)

**Parameters:**

x : Input logits (batch_size, num_classes).

target : Dense (multi-hot or soft) targets (batch_size, num_classes).

#### timm.loss.TwoWayLoss[[timm.loss.TwoWayLoss]]

```python
timm.loss.TwoWayLoss(tp: float = 4.0, tn: float = 1.0)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/two_way_loss.py#L17)

**Parameters:**

tp : Temperature of the soft-min over positive logits.

tn : Temperature of the soft-max over negative logits.

Two-way multi-label loss.

Contrasts the hardest (soft-max) negative logit with the hardest (soft-min) positive logit in two directions,
across the classes of each sample and across the samples of each class in the batch. Only samples / classes
with at least one positive contribute to the respective direction.

Targets are binarized, entries above 0.5 are positives. Logits are ranking scores, they are not calibrated
as independent sigmoid probabilities.

#### forward[[timm.loss.TwoWayLoss.forward]]

```python
forward(x: Tensor, target: Tensor)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/two_way_loss.py#L38)

**Parameters:**

x : Input logits (batch_size, num_classes).

target : Dense binary targets (batch_size, num_classes).

#### timm.loss.ZlprLoss[[timm.loss.ZlprLoss]]

```python
timm.loss.ZlprLoss(*args, **kwargs)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/zlpr_loss.py#L9)

ZLPR loss, the multi-label extension of softmax cross-entropy.

Per sample, log(1 + sum_neg exp(s_j)) + log(1 + sum_pos exp(-s_i)). Every positive logit is pushed above
every negative one and 0 acts as the decision threshold, so logits > 0 (sigmoid > 0.5) predict positives.
No hyperparameters, and the loss does not grow with the number of negative classes like per-class BCE.

Targets are binarized, entries above 0.5 are positives.

#### forward[[timm.loss.ZlprLoss.forward]]

```python
forward(x: Tensor, target: Tensor)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/loss/zlpr_loss.py#L19)

**Parameters:**

x : Input logits (batch_size, num_classes).

target : Dense binary targets (batch_size, num_classes).

### Optimization
https://huggingface.co/docs/timm/v1.0.31/reference/optimizers.md
