# Models

This page contains the API reference documentation for creating and discovering `timm` models, their pretrained
configs and the checkpoint / Hub helpers, and an overview of the interface shared by `timm` models. The individual
architectures are covered in the [model summaries](../models) and the per-model pages.

## Usage

Models are created by name with [`create_model`](#timm.create_model). The name selects an architecture registered in
`timm`, optionally w/ a pretrained tag after a `.` (`resnet50.a1_in1k`), or a model on the Hugging Face Hub or in a
local folder via a source prefix. Keyword arguments that `create_model` doesn't use itself pass through the model's
entrypoint function to the model class constructor, so any model specific argument can be set.

```python
import torch
import timm

# Random init (pretrained=False is the default). The tag selects the pretrained cfg, and the weights loaded with
# pretrained=True. Without a tag the architecture's default tag is used.
model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k')
print(model.pretrained_cfg['mean'], model.pretrained_cfg['crop_pct'])  # (0.5, 0.5, 0.5) 0.9

# Model arguments pass through to the model class, a 10 class head, single channel input and regularization.
model = timm.create_model('resnet50', num_classes=10, in_chans=1, drop_rate=0.2, drop_path_rate=0.05)

# Discover models, filters are fnmatch wildcards.
timm.list_models('convnext*')  # architecture names
timm.list_pretrained('resnet50.*')  # architecture.tag names w/ pretrained weights
timm.list_models(module='vision_transformer', pretrained=True)

# Image preprocessing to match the model's pretrained cfg (input size, interpolation, crop, mean / std).
model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k')
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

x = torch.randn(2, 3, 224, 224)
features = model.forward_features(x)  # unpooled features, (2, 197, 192)
pre_logits = model.forward_head(features, pre_logits=True)  # pooled features, (2, 192)
logits = model.forward_head(features)  # same as model(x), (2, 1000)

model.reset_classifier(5)  # replace the classifier w/ a new 5 class one
model.set_input_size(img_size=(384, 384))  # resample the position embedding for a new resolution
logits = model(torch.randn(2, 3, 384, 384))  # (2, 5)
```

The training script (`train.py`) exposes the common arguments through these flags:

- **`--model`** selects the model.
- **`--pretrained`** loads pretrained weights.
- **`--num-classes`** sets the classifier size (`num_classes`).
- **`--in-chans`** sets the input channels (`in_chans`).
- **`--gp`** sets the global pooling type (`global_pool`).
- **`--drop`** sets classifier dropout (`drop_rate`).
- **`--drop-path`** sets stochastic depth (`drop_path_rate`).
- **`--pretrained-path`** loads pretrained weights from a local file with classifier / input conv adaptation
  (`pretrained_cfg_overlay=dict(file=...)`).
- **`--initial-checkpoint`** loads a checkpoint after creating the model, without adaptation (`checkpoint_path`).
- **`--model-kwargs key=value ...`** passes any other model arguments.

See the [quickstart](../quickstart) for an introduction.

### Model names

| Name | Source |
|---|---|
| `resnet50` | Architecture in the `timm` registry, w/ its default pretrained cfg (here `a1_in1k`) |
| `resnet50.a1_in1k` | Architecture + pretrained tag, an unknown tag raises an error |
| `hf-hub:timm/resnet50.a1_in1k` | Hugging Face Hub repo, the architecture, pretrained cfg and model args come from its `config.json`, `@revision` selects a revision |
| `local-dir:/path/to/folder` | Local folder w/ a `config.json` and weights, as written by [`save_for_hf`](#timm.models.save_for_hf) |

The Hub and local sources read the config even w/ `pretrained=False`, weights are only loaded w/ `pretrained=True`.
A name w/o a source prefix that contains `/`, `\` or `:` raises a `ValueError` instead of falling back to a registry
model of the same name. `hf_hub:` is a deprecated alias of `hf-hub:`.

### Common arguments

- **`pretrained`** loads the weights of the pretrained cfg, from `state_dict` or `file` if set, otherwise from the
  Hub (`hf_hub_id`) when `huggingface_hub` is installed, or the `url`. Classifier / input conv adaptation below lets
  a pretrained model be created for fine-tuning in one call.
- **`num_classes`** sets the classifier size, `0` removes the classifier. The default comes from the pretrained cfg.
  When loading pretrained weights, a different class count discards the classifier weights (left at random init).
- **`global_pool`** sets the pooling type, the options are model specific (e.g. `avg`, `max`, `avgmax`, `catavgmax`
  for most convnets, `token`, `avg`, `map` for ViT), `''` disables pooling.
- **`in_chans`** sets the input channels. The default comes from the pretrained cfg. When loading pretrained weights,
  the first conv weights are adapted where supported if this differs from the source checkpoint's channel count
  (`input_size[0]` in the pretrained cfg).
- **`drop_rate`** sets classifier dropout, supported by most models.
- **`drop_path_rate`** sets stochastic depth, supported by most models.
- **`img_size`** is an argument of ViT style and other models w/ a fixed input size, they're built for the pretrained
  cfg `input_size` unless it's set.
- **`dynamic_img_size=True`** in the ViT and EVA families resamples the position embedding to the input size on every
  forward pass.
- **`features_only`** (w/ `out_indices`, `output_stride`, `feature_cls`) wraps the model as a feature backbone
  returning feature maps, see [feature extraction](../feature_extraction).
- **`pretrained_cfg`** replaces the pretrained cfg (a tag, dict or [`PretrainedCfg`](#timm.models.PretrainedCfg)).
- **`pretrained_cfg_overlay`** overrides some of its fields, e.g. `pretrained_cfg_overlay=dict(file='weights.pth')`
  loads `pretrained=True` weights from a local file, w/ the classifier / input conv adaptation above (as
  `--pretrained-path` does in `train.py`).
- **`checkpoint_path`** loads a checkpoint w/ [`load_checkpoint`](#timm.models.load_checkpoint) after the model is
  created. It's a strict load w/o adaptation, EMA weights are used if the checkpoint has them.
- **`cache_dir`** overrides the download cache for pretrained weights and Hub configs.
- **`device`** is passed to the model class, params are created on the device directly. `device='meta'` creates a
  model w/o allocating memory for its weights.
- **`dtype`** is passed to the model class, params are created in the dtype directly.
- **`scriptable`** sets layer config flags for TorchScript scripting.
- **`exportable`** sets layer config flags for export / tracing.
- **`no_jit`** sets layer config flags to disable jit scripted layers.

Arguments set to `None` are dropped before reaching the model, so the model defaults apply. Arguments the model class
doesn't accept raise a `TypeError` in most models, a few ignore unknown arguments.

### Pretrained config

Architectures have one or more pretrained cfgs, one per tag, registered w/ the model. A cfg describes the weights
(location, license, origin) and how the model was trained and should be evaluated w/ them. Models created by
`create_model` have their cfg as a dict in `model.pretrained_cfg` (`model.default_cfg` is an alias), the registry
versions are returned as [`PretrainedCfg`](#timm.models.PretrainedCfg) instances by
[`get_pretrained_cfg`](#timm.get_pretrained_cfg).

| Field | Description |
|---|---|
| `architecture`, `tag` | Model architecture and pretrained tag |
| `hf_hub_id`, `url`, `file`, `state_dict` | Weight sources (see `pretrained` above) |
| `input_size`, `test_input_size`, `fixed_input_size` | Train (and default eval) input size, eval size used w/ `use_test_size=True` (if different), whether the model has a fixed input size (`img_size` argument) |
| `interpolation`, `crop_pct`, `test_crop_pct`, `crop_mode` | Eval resize interpolation, crop fraction (w/ the default `center` crop mode the image is resized to `input_size / crop_pct`, then center cropped) and crop mode |
| `mean`, `std` | Input normalization |
| `num_classes`, `label_names`, `label_descriptions` | Classifier size and labels (when known) |
| `classifier`, `first_conv` | Module names of the classifier and first conv, used to adapt the weights on load |
| `license`, `origin_url`, `paper_ids`, `paper_name`, `notes` | Weight meta-data |

The input fields are what [`resolve_data_config`](data#timm.data.resolve_data_config) /
[`resolve_model_data_config`](data#timm.data.resolve_model_data_config) resolve to a data config for the dataset
transforms and loader. The cfg describes the source checkpoint and isn't updated when the model is created or
modified differently (`num_classes`, `in_chans`, `img_size`, `set_input_size()`). The model's current input
channels, size and fixed size constraint are tracked separately:
[`get_model_input_config`](#timm.models.get_model_input_config) returns them, and
[`resolve_input_data_config`](data#timm.data.resolve_input_data_config) resolves a data config against them (as the
training, validation and inference scripts do). [`get_model_args`](#timm.models.get_model_args) returns the arguments
to rebuild the model as it is now, which [`save_for_hf`](#timm.models.save_for_hf) writes to the exported config. See
[adapt and export models](../model_adaptation) for details.

### Model interface

`timm` models share a common interface beyond `forward()`: the `forward_features()` / `forward_head()` split,
classifier reset, intermediate features, input resizing and training hooks such as gradient checkpointing and
parameter grouping. It is described in [the timm Model API](../model_api), including which model families support
which parts.

### Checkpoints and the Hub

- [`load_checkpoint`](#timm.models.load_checkpoint) loads a checkpoint file (PyTorch or `.safetensors`) into a model.
- [`load_state_dict`](#timm.models.load_state_dict) reads a checkpoint file and returns its state dict.
- [`resume_checkpoint`](#timm.models.resume_checkpoint) restores the model, optimizer and loss scaler state of a
  `train.py` checkpoint and returns the epoch to resume from.

`load_checkpoint` and `load_state_dict` pick the EMA weights of a `train.py` checkpoint by default (`use_ema`)
and remove `module.` (DDP) and `_orig_mod.` (`torch.compile`) key prefixes.

- [`save_for_hf`](#timm.models.save_for_hf) writes the weights and a `config.json` (architecture, pretrained cfg,
  `model_args`, `label_names`) to a folder.
- [`push_to_hf_hub`](#timm.models.push_to_hf_hub) uploads the weights and config to a Hub repo w/ a generated model
  card if the repo has none.

The model is then created w/ `hf-hub:` or `local-dir:`, see
[sharing and loading models](../hf_hub).

```python
import timm
from timm.models import save_for_hf

model = timm.create_model('resnet18', num_classes=4)
# ... fine-tune
save_for_hf(model, 'my_resnet18', model_config=dict(label_names=['a', 'b', 'c', 'd']), safe_serialization=True)
model = timm.create_model('local-dir:my_resnet18', pretrained=True)  # 4 classes, weights and labels restored
print(model.pretrained_cfg['label_names'])
```

### Custom models

A model entrypoint function decorated w/ [`register_model`](#timm.models.register_model) is added to the registry, so
it can be created w/ `create_model` and found w/ `list_models`. Entrypoints in `timm` build their model w/
[`build_model_with_cfg`](#timm.models.build_model_with_cfg), which handles the pretrained cfg, pretrained weight
loading and `features_only`.

## API

### Model creation[[timm.create_model]]

#### timm.create_model[[timm.create_model]]

```python
timm.create_model(model_name: str, pretrained: bool = False, pretrained_cfg: typing.Union[str, typing.Dict[str, typing.Any], timm.models._pretrained.PretrainedCfg, NoneType] = None, pretrained_cfg_overlay: typing.Optional[typing.Dict[str, typing.Any]] = None, checkpoint_path: typing.Union[str, pathlib.Path, NoneType] = None, cache_dir: typing.Union[str, pathlib.Path, NoneType] = None, scriptable: typing.Optional[bool] = None, exportable: typing.Optional[bool] = None, no_jit: typing.Optional[bool] = None, **kwargs: typing.Any)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_factory.py#L162)

**Parameters:**

model_name : Name of model to instantiate.

pretrained : If set to *True*, load pretrained ImageNet-1k weights.

pretrained_cfg : Pass in an external pretrained_cfg for model.

pretrained_cfg_overlay : Replace key-values in base pretrained_cfg with these.

checkpoint_path : Path of checkpoint to load _after_ the model is initialized.

cache_dir : Override model cache dir for Hugging Face Hub and Torch checkpoints.

scriptable : Set layer config so that model is jit scriptable (not working for all models yet).

exportable : Set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet).

no_jit : Set layer config so that model doesn't utilize jit scripted layers (so far activations only).

Create a model.

Lookup model's entrypoint function and pass relevant args to create a new model.

Tip:
**kwargs will be passed through entrypoint fn to `timm.models.build_model_with_cfg()`
and then the model class __init__(). kwargs values set to None are pruned before passing.

Keyword Args:
drop_rate (float): Classifier dropout rate for training.
drop_path_rate (float): Stochastic depth drop rate for training.
global_pool (str): Classifier global pooling type.

Examples:

```python
>>> from timm import create_model

>>> # Create a MobileNetV3-Large model with no pretrained weights.
>>> model = create_model('mobilenetv3_large_100')

>>> # Create a MobileNetV3-Large model with pretrained weights.
>>> model = create_model('mobilenetv3_large_100', pretrained=True)
>>> model.num_classes
1000

>>> # Create a MobileNetV3-Large model with pretrained weights and a new head with 10 classes.
>>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
>>> model.num_classes
10

>>> # Create a Dinov2 small model with pretrained weights and save weights in a custom directory.
>>> model = create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, cache_dir="/data/my-models")
>>> # Data will be stored at */data/my-models/models--timm--vit_small_patch14_dinov2.lvd142m/*
```

### Model discovery[[timm.list_models]]

#### timm.list_models[[timm.list_models]]

```python
timm.list_models(filter: typing.Union[str, typing.Iterable[str]] = '', module: typing.Union[str, typing.Iterable[str]] = '', pretrained: bool = False, exclude_filters: typing.Union[str, typing.Iterable[str]] = '', name_matches_cfg: bool = False, include_tags: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L254)

**Parameters:**

filter : Wildcard filter(s) that work with fnmatch.

module : Limit model selection to specific submodule(s) (ie 'vision_transformer').

pretrained : Include only models with valid pretrained weights if True.

exclude_filters : Wildcard filter(s) to exclude models after including them with filter.

name_matches_cfg : Include only models w/ model_name matching default_cfg name (excludes some aliases).

include_tags : Include pretrained tags in model names (model.tag). If None, set to the value of pretrained.

**Returns:**

The sorted list of model names.

Return list of available model names, sorted naturally.

Example:
list_models('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
list_models('*resnext*', 'resnet') -- returns all models with 'resnext' in 'resnet' module

#### timm.list_pretrained[[timm.list_pretrained]]

```python
timm.list_pretrained(filter: typing.Union[str, typing.Iterable[str]] = '', exclude_filters: typing.Union[str, typing.Iterable[str]] = '')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L329)

**Parameters:**

filter : Wildcard filter(s) that work with fnmatch.

exclude_filters : Wildcard filter(s) to exclude models after including them with filter.

**Returns:**

The sorted list of model names.

Return list of pretrained model names (w/ tags), sorted naturally.

#### timm.list_modules[[timm.list_modules]]

```python
timm.list_modules()
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L385)

Return sorted list of module names that contain model entrypoints.

#### timm.is_model[[timm.is_model]]

```python
timm.is_model(model_name: str)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L363)

Check if a model architecture is registered, ignoring any pretrained tag.

#### timm.is_model_pretrained[[timm.is_model_pretrained]]

```python
timm.is_model_pretrained(model_name: str)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L404)

Check if a model name (w/ or w/o tag) has pretrained weights registered.

#### timm.model_entrypoint[[timm.model_entrypoint]]

```python
timm.model_entrypoint(model_name: str, module_filter: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L369)

**Parameters:**

model_name : Model name, with or without a pretrained tag.

module_filter : If set, raise if the model isn't in this submodule.

**Returns:**

The model entrypoint fn.

Fetch the model entrypoint fn for a model name, ignoring any pretrained tag.

### Pretrained config[[timm.get_pretrained_cfg]]

#### timm.get_pretrained_cfg[[timm.get_pretrained_cfg]]

```python
timm.get_pretrained_cfg(model_name: str, allow_unregistered: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L409)

**Parameters:**

model_name : Model name, with or without a pretrained tag.

allow_unregistered : Return None instead of raising if the architecture has no pretrained cfgs.

**Returns:**

The pretrained cfg, or None if unregistered and allowed.

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If the tag is invalid for the architecture, or it has no cfgs and allow_unregistered is False.

Get a copy of the pretrained cfg for a model name, the default tag is used if none specified.

#### timm.get_pretrained_cfg_value[[timm.get_pretrained_cfg_value]]

```python
timm.get_pretrained_cfg_value(model_name: str, cfg_key: str)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L434)

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- If the model has no pretrained cfg registered or the tag is invalid.

Get a specific pretrained cfg value by key for a model name, None if the key doesn't exist.

#### timm.models.get_arch_pretrained_cfgs[[timm.models.get_arch_pretrained_cfgs]]

```python
timm.models.get_arch_pretrained_cfgs(model_name: str)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L444)

**Parameters:**

model_name : Model name, any pretrained tag is ignored.

**Returns:**

Dict of pretrained cfgs keyed by model name w/ tag, empty if the architecture is unknown.

Get copies of all pretrained cfgs for a model architecture.

#### timm.models.PretrainedCfg[[timm.models.PretrainedCfg]]

```python
timm.models.PretrainedCfg(url: typing.Union[str, typing.Tuple[str, str], NoneType] = None, file: typing.Optional[str] = None, state_dict: typing.Optional[typing.Dict[str, typing.Any]] = None, hf_hub_id: typing.Optional[str] = None, hf_hub_filename: typing.Optional[str] = None, source: typing.Optional[str] = None, architecture: typing.Optional[str] = None, tag: typing.Optional[str] = None, custom_load: bool = False, input_size: typing.Tuple[int, int, int] = (3, 224, 224), test_input_size: typing.Optional[typing.Tuple[int, int, int]] = None, min_input_size: typing.Optional[typing.Tuple[int, int, int]] = None, fixed_input_size: bool = False, interpolation: str = 'bicubic', crop_pct: float = 0.875, test_crop_pct: typing.Optional[float] = None, crop_mode: str = 'center', mean: typing.Tuple[float, ...] = (0.485, 0.456, 0.406), std: typing.Tuple[float, ...] = (0.229, 0.224, 0.225), num_classes: int = 1000, label_offset: typing.Optional[int] = None, label_names: typing.Optional[typing.Tuple[str]] = None, label_descriptions: typing.Optional[typing.Dict[str, str]] = None, pool_size: typing.Optional[typing.Tuple[int, ...]] = None, test_pool_size: typing.Optional[typing.Tuple[int, ...]] = None, first_conv: typing.Optional[str] = None, classifier: typing.Optional[str] = None, license: typing.Optional[str] = None, description: typing.Optional[str] = None, origin_url: typing.Optional[str] = None, paper_name: typing.Optional[str] = None, paper_ids: typing.Union[str, typing.Tuple[str], NoneType] = None, notes: typing.Optional[typing.Tuple[str]] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_pretrained.py#L11)

### Current model input and export[[timm.models.get_model_input_config]]

#### timm.models.get_model_input_config[[timm.models.get_model_input_config]]

```python
timm.models.get_model_input_config(model: Module)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_input.py#L113)

Return the current image channels, nominal size, and fixed-size constraint.

The nominal size is a recommendation for flexible models. Source checkpoint
metadata remains in `model.pretrained_cfg` and is never modified here.

#### timm.models.get_model_traits[[timm.models.get_model_traits]]

```python
timm.models.get_model_traits(model: Module)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_input.py#L101)

Return a copy of the model's traits, augmented with current channel and class counts.

Existing `in_chans` and `num_classes` attributes remain authoritative;
they are read here instead of being duplicated in the stored traits.
Attributes absent from a feature wrapper are omitted.

#### timm.models.get_model_args[[timm.models.get_model_args]]

```python
timm.models.get_model_args(model: Module, overrides: typing.Optional[typing.Dict[str, typing.Any]] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_input.py#L170)

Return reconstruction arguments, including supported changes made after construction.

Runtime arguments are excluded by the factory. Drop rates are omitted unless
explicitly supplied in overrides. Non-serializable architecture arguments
require an explicit serializable replacement when exporting.

#### timm.models.resolve_model_input_args[[timm.models.resolve_model_input_args]]

```python
timm.models.resolve_model_input_args(model_name: str, args: typing.Optional[typing.Dict[str, typing.Any]] = None, **model_kwargs)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_factory.py#L123)

Resolve script input requests into arguments for `create_model`.

Saved model arguments are defaults. Explicit channel/size requests select the
target without overwriting metadata describing the source checkpoint.

#### timm.models.adapt_input_conv[[timm.models.adapt_input_conv]]

```python
timm.models.adapt_input_conv(in_chans: int, conv_weight: Tensor, base_chans: int = 3)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_manipulate.py#L293)

**Parameters:**

in_chans : Number of input channels for the target model.

conv_weight : Pretrained Conv2d weights in OIHW layout.

base_chans : Number of physical input channels used for pretraining. This can differ from the weight's input dimension for space-to-depth stems.

**Returns:**

Weights with repeated/truncated channels scaled by `base_chans / in_chans`.
Single-channel inputs instead sum the source channels, preserving separate
spatial groups for space-to-depth stems. Matching channel counts are unchanged.

**Raises:** `ValueError` or `NotImplementedError`

- `ValueError` -- If either channel count is not positive.
- `NotImplementedError` -- If the weight format cannot be adapted.

Adapt pretrained input convolution weights to a different channel count.

### Checkpoints and Hub[[timm.models.load_checkpoint]]

#### timm.models.load_checkpoint[[timm.models.load_checkpoint]]

```python
timm.models.load_checkpoint(model: Module, checkpoint_path: str, use_ema: bool = True, device: typing.Union[str, torch.device] = 'cpu', strict: bool = True, remap: bool = False, filter_fn: typing.Optional[typing.Callable] = None, weights_only: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_helpers.py#L186)

**Parameters:**

model : Model to load checkpoint into.

checkpoint_path : Path to checkpoint file.

use_ema : Whether to use EMA weights if available.

device : Device to load checkpoint to.

strict : Whether to strictly enforce state_dict keys match.

remap : Whether to remap state dict keys by order.

filter_fn : Optional function to filter state dict.

weights_only : Whether to load only weights (torch.load parameter).

**Returns:**

Incompatible keys from model.load_state_dict().

Load checkpoint into model.

Native PyTorch checkpoints are loaded as state dictionaries. NumPy
`.npz`/``.npy` checkpoints are foreign, model-specific formats and are delegated to the model's `load_pretrained`` helper.

#### timm.models.load_state_dict[[timm.models.load_state_dict]]

```python
timm.models.load_state_dict(checkpoint_path: str, use_ema: bool = True, device: typing.Union[str, torch.device] = 'cpu', weights_only: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_helpers.py#L143)

**Parameters:**

checkpoint_path : Path to checkpoint file.

use_ema : Whether to use EMA weights if available.

device : Device to load checkpoint to.

weights_only : Whether to load only weights (torch.load parameter).

**Returns:**

State dictionary loaded from checkpoint.

Load state dictionary from checkpoint file.

#### timm.models.resume_checkpoint[[timm.models.resume_checkpoint]]

```python
timm.models.resume_checkpoint(model: Module, checkpoint_path: str, optimizer: typing.Optional[torch.optim.Optimizer] = None, loss_scaler: typing.Optional[typing.Any] = None, log_info: bool = True, weights_only: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_helpers.py#L261)

**Parameters:**

model : Model to load checkpoint into.

checkpoint_path : Path to checkpoint file.

optimizer : Optional optimizer to restore state.

loss_scaler : Optional AMP loss scaler to restore state.

log_info : Whether to log loading info.

weights_only : Whether to load only weights via torch.load.

**Returns:**

Resume epoch number if available, else None.

Resume training from checkpoint.

#### timm.models.save_for_hf[[timm.models.save_for_hf]]

```python
timm.models.save_for_hf(model: Module, save_directory: str, model_config: typing.Optional[dict] = None, model_args: typing.Optional[dict] = None, safe_serialization: typing.Union[bool, typing.Literal['both']] = False, data_config: typing.Optional[dict] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_hub.py#L449)

Save native weights and their current model and preprocessing configuration.

#### timm.models.push_to_hf_hub[[timm.models.push_to_hf_hub]]

```python
timm.models.push_to_hf_hub(model: Module, repo_id: str, commit_message: str = 'Add model', token: typing.Optional[str] = None, revision: typing.Optional[str] = None, private: bool = False, create_pr: bool = False, model_config: typing.Optional[dict] = None, model_card: typing.Optional[dict] = None, model_args: typing.Optional[dict] = None, task_name: str = 'image-classification', safe_serialization: typing.Union[bool, typing.Literal['both']] = 'both', data_config: typing.Optional[dict] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_hub.py#L480)

**Parameters:**

(...) --

model_args : Explicit reconstruction arguments supplementing those recorded by the factory.

data_config : Preprocessing used with the exported checkpoint, including input size and normalization.

safe_serialization (`bool` or `"both"`, *optional*, defaults to `False`) : Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`). Can be set to `"both"` in order to push both safe and unsafe weights.

### Custom models[[timm.models.register_model]]

#### timm.models.register_model[[timm.models.register_model]]

```python
timm.models.register_model(fn: typing.Callable[..., typing.Any])
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_registry.py#L114)

**Parameters:**

fn : Model entrypoint fn, registered under its __name__.

**Returns:**

The unmodified entrypoint fn.

Decorator that registers a model entrypoint fn and any pretrained cfgs in its module's default_cfgs.

Registering a name that already exists overwrites the previous registration and its pretrained cfgs.

#### timm.models.build_model_with_cfg[[timm.models.build_model_with_cfg]]

```python
timm.models.build_model_with_cfg(model_cls: typing.Union[typing.Type[~ModelT], typing.Callable[..., ~ModelT]], variant: str, pretrained: bool, pretrained_cfg: typing.Optional[typing.Dict] = None, pretrained_cfg_overlay: typing.Optional[typing.Dict] = None, model_cfg: typing.Optional[typing.Any] = None, feature_cfg: typing.Optional[typing.Dict] = None, pretrained_strict: bool = True, pretrained_filter_fn: typing.Optional[typing.Callable] = None, cache_dir: typing.Union[str, pathlib.Path, NoneType] = None, kwargs_filter: typing.Optional[typing.Tuple[str]] = None, pretrained_model_args: typing.Optional[typing.Dict] = None, **kwargs)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/models/_builder.py#L427)

**Parameters:**

model_cls : Model class

variant : Model variant name

pretrained : Load the pretrained weights

pretrained_cfg : Model's pretrained weight/task config

pretrained_cfg_overlay : Entries that will override those in pretrained_cfg

model_cfg : Model's architecture config

feature_cfg : Feature extraction adapter config

pretrained_strict : Load pretrained weights strictly

pretrained_filter_fn : Filter callable for pretrained weights

cache_dir : Override model cache dir for Hugging Face Hub and Torch checkpoints

kwargs_filter : Kwargs keys to filter (remove) before passing to model

pretrained_model_args : Reconstruction arguments describing the source checkpoint.

- ****kwargs** : Model args passed through to model __init__

Build model with specified default_cfg and optional model_cfg

This helper fn aids in the construction of a model including:
* handling default_cfg and associated pretrained weight loading
* passing through optional model_cfg for models with config based arch spec
* features_only model adaptation
* pruning config / model adaptation

### Data
https://huggingface.co/docs/timm/v1.0.31/reference/data.md
