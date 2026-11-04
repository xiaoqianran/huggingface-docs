# Data

This page contains the API reference documentation for the `timm.data` sub-package, and an overview of the input
pipeline it provides: resolving a model's preprocessing config, building transforms, creating datasets and loaders,
Mixup / CutMix, and the NaFlex variable resolution loader.

## Usage

A pipeline is built in four steps: [`resolve_model_data_config`](#timm.data.resolve_model_data_config) reads the
model's preprocessing config, [`create_transform`](#timm.data.create_transform) builds the train or eval transform,
[`create_dataset`](#timm.data.create_dataset) creates a dataset by name and root, and
[`create_loader`](#timm.data.create_loader) builds the transform and a data loader for a dataset.

```python
import os
import tempfile

import torch
from PIL import Image

import timm
from timm.data import create_dataset, create_loader, create_transform, resolve_model_data_config

model = timm.create_model('resnet50', pretrained=False).eval()

# Preprocessing config from the model's pretrained_cfg, the default pretrained tag's config is attached to the model
# with or without pretrained weights.
data_config = resolve_model_data_config(model)
print(data_config)
# {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406),
#  'std': (0.229, 0.224, 0.225), 'crop_pct': 0.95, 'crop_mode': 'center'}

# Eval transform for a PIL image, resize to 224 / crop_pct, center crop 224, to tensor and normalize.
transform = create_transform(**data_config, is_training=False)
img = Image.new('RGB', (320, 240), color=(200, 120, 40))
with torch.no_grad():
    output = model(transform(img).unsqueeze(0))  # (1, 1000)

# A folder dataset, <root>/<split>/<class>/<image>, two classes w/ four generated images each.
root = tempfile.mkdtemp()
for i, cls in enumerate(('cat', 'dog')):
    os.makedirs(os.path.join(root, 'validation', cls))
    for j in range(4):
        path = os.path.join(root, 'validation', cls, f'{j}.jpg')
        Image.new('RGB', (300, 200), color=(60 * i, 40 * j, 128)).save(path)

dataset = create_dataset('', root=root, split='validation')  # finds the 'validation' sub-folder
print(len(dataset), dataset.reader.class_to_idx)  # 8 {'cat': 0, 'dog': 1}

# The loader creates the transform from the same config and assigns it to the dataset.
loader = create_loader(dataset, **data_config, batch_size=4, use_prefetcher=False, num_workers=0)
with torch.no_grad():
    for images, targets in loader:
        output = model(images)  # images (4, 3, 224, 224), targets (4,)
```

The [quickstart](../quickstart#image-augmentation) walks through the config to transform steps for inference.

The training script (`train.py`) exposes these options for each part of the pipeline.

Input config overrides (`resolve_input_data_config`):

- **`--input-size`** sets the input shape `(C, H, W)`.
- **`--img-size`** is the square shortcut for `--input-size`, keeping the model's channel count.
- **`--in-chans`** sets the input channels.
- **`--interpolation`** sets the resize interpolation.
- **`--mean`** sets the normalization mean.
- **`--std`** sets the normalization standard deviation.
- **`--crop-pct`** sets the evaluation crop fraction.

Dataset creation (`create_dataset`) and target setup:

- **`--dataset`** selects the dataset and reader.
- **`--data-dir`** sets the dataset root.
- **`--train-split`** selects the training split.
- **`--val-split`** selects the validation split.
- **`--class-map`** supplies the class mapping.
- **`--input-img-mode`** sets the image color conversion mode.
- **`--input-key`** selects the image field.
- **`--target-key`** selects the label field or fields.
- **`--target-format`** selects class indices or dense multi-hot targets.

Loader and augmentation options (`create_loader`) include:

- **`--aa`** selects the auto-augmentation policy.
- **`--reprob`** sets the random erasing probability.
- **`--remode`** selects the random erasing fill mode.
- **`--scale`** sets the random resized crop scale range.
- **`--ratio`** sets the random resized crop aspect ratio range.
- **`--hflip`** sets the horizontal flip probability.
- **`--color-jitter`** sets color jitter strength.
- **`--train-interpolation`** sets the training resize interpolation.
- **`--workers`** sets the number of loader workers.
- **`--no-prefetcher`** disables the prefetcher.

Mixup and CutMix ([`Mixup`](#timm.data.Mixup)):

- **`--mixup`** sets the Mixup alpha.
- **`--cutmix`** sets the CutMix alpha.
- **`--cutmix-minmax`** sets the CutMix box size range.
- **`--mixup-prob`** sets the probability of mixing.
- **`--mixup-switch-prob`** sets the probability of choosing CutMix when both methods are enabled.
- **`--mixup-mode`** selects batch, pair or element-wise mixing.
- **`--smoothing`** sets label smoothing.

See [Scripts](../training_script) for examples.

### Data config

[`resolve_data_config`](#timm.data.resolve_data_config) returns a dict with the keys below, which match the
arguments of `create_transform` and `create_loader`. Values set in `args` (a dict, e.g. `vars(args)` of parsed
command line args) take precedence over the `pretrained_cfg` argument, which defaults to `model.pretrained_cfg`.
[`resolve_model_data_config`](#timm.data.resolve_model_data_config) is the same with the model as first argument.

Both read the input size and channels from the pretrained cfg, which describes the source checkpoint, not a model
created or changed to use a different `img_size` or `in_chans`.
[`resolve_input_data_config`](#timm.data.resolve_input_data_config), used by the training, validation and inference
scripts, resolves against the model's current input instead: a fixed input size model always gets its current size,
explicit `args` that conflict with the model's channels or fixed size raise an error, and a mean / std that doesn't
match the channel count is replaced by its average (with a warning).

| Key | `args` keys | Description | Default |
|---|---|---|---|
| `input_size` | `input_size`, `img_size`, `in_chans` | `(C, H, W)` input size. With `use_test_size=True` the config's `test_input_size` is used if it has one | `(3, 224, 224)` |
| `interpolation` | `interpolation` | Resize interpolation, `bilinear`, `bicubic`, ... | `bicubic` |
| `mean`, `std` | `mean`, `std` | Normalization, a single value is repeated for each channel | ImageNet mean / std |
| `crop_pct` | `crop_pct` | Eval crop, the image is resized to `size / crop_pct` before cropping to `size` (`test_crop_pct` with `use_test_size=True`) | `0.875` |
| `crop_mode` | `crop_mode` | Eval crop mode, `center`, `squash` or `border` | `center` |

The eval crop modes, for an output `size` and `scale_size = floor(size / crop_pct)`:

| `crop_mode` | Behaviour |
|---|---|
| `center` | Resize the shortest edge to `scale_size` (aspect ratio preserved), center crop to `size` |
| `squash` | Resize both edges to `scale_size` (aspect ratio not preserved), center crop to `size` |
| `border` | Resize the longest edge to `scale_size`, center crop and pad with the mean color, the whole image is kept at `crop_pct=1.0` |

### Transforms

[`create_transform`](#timm.data.create_transform) returns a torchvision `Compose`. The eval transform resizes and
crops as above, then converts to a tensor and normalizes. The training transform is a random resized crop
(`scale=(0.08, 1.0)`, `ratio=(3/4, 4/3)` by default), horizontal flip (`hflip=0.5`), color jitter
(`color_jitter=0.4`), the optional `auto_augment` policy, tensor conversion and normalization, and random erasing.

- **`train_crop_mode='rkrc'` or `'rkrr'`** replaces the random resized crop with a random resize that keeps the aspect
  ratio, followed by a center (`rkrc`) or random (`rkrr`) crop or pad.
- **`interpolation='random'`** (the `train.py` default for training) picks bilinear or bicubic per image.
- **`no_aug=True`** reduces training to a resize and center crop.

| `auto_augment` | Augmentation |
|---|---|
| `rand-m9-mstd0.5-inc1` | [`RandAugment`](https://arxiv.org/abs/1909.13719) via [`rand_augment_transform`](#timm.data.rand_augment_transform) |
| `augmix-m5-w4-d2` | [AugMix](https://arxiv.org/abs/1912.02781) via [`augment_and_mix_transform`](#timm.data.auto_augment.augment_and_mix_transform) |
| `original`, `originalr`, `v0`, `v0r`, `3a` | [AutoAugment](https://arxiv.org/abs/1805.09501) policy via [`auto_augment_transform`](#timm.data.auto_augment_transform) |

RandAugment option tokens:

- **`m`** sets magnitude (default 10).
- **`n`** sets the number of ops per image (default 2).
- **`p`** sets the probability per op (default 0.5).
- **`mstd`** sets magnitude noise std.
- **`inc1`** selects ops that increase in severity w/ magnitude.
- **`t<name>`** selects the op set (`weights`, `3a`, `3aw`).

AugMix option tokens:

- **`m`** sets severity (default 3).
- **`w`** sets width (default 3).
- **`d`** sets depth (default -1, random 1-3).
- **`b1`** enables blended augmentation.
- **`mstd`** sets magnitude noise std.

AutoAugment accepts `mstd`, e.g. `original-mstd0.5`.

Transform behavior:

- Color jitter is disabled when `auto_augment` is set, unless the string contains `3a`.
- [`RandomErasing`](#timm.data.random_erasing.RandomErasing) (`re_prob`, `re_count`) runs on the normalized tensor.
  `re_mode` fills with zeros (`const`, the default), a random normal color per region (`rand`) or random normal
  pixels (`pixel`, the `train.py` default).
- `use_prefetcher=True` ends the transform at a `uint8` tensor, normalization and random erasing are then done by the
  loader on the device.
- `create_transform` and `create_loader` default to `bilinear` interpolation and ImageNet mean / std, not the model's
  values. Pass the resolved data config to match the model's pretraining.

### Datasets

[`create_dataset`](#timm.data.create_dataset) selects the dataset implementation and reader from a prefix of `name`:

| `name` | Dataset | `root` | Requires |
|---|---|---|---|
| `''` or no prefix | [`ImageDataset`](#timm.data.ImageDataset), images in class sub-folders (searched recursively, the leaf folder name is the class), or a single `.tar` file of the same | Dataset folder or `.tar` file | |
| `torch/<name>` | torchvision `cifar10`, `cifar100`, `mnist`, `kmnist`, `fashion_mnist`, `qmnist`, `inaturalist` (`inat`), `places365`, `imagenet` or `image_folder` (`folder`) | torchvision dataset root | |
| `hfds/<name>` | [`ImageDataset`](#timm.data.ImageDataset) on a Hugging Face Hub dataset (or local path) via `datasets.load_dataset` | Cache dir (optional) | `datasets` |
| `hfids/<name>` | [`IterableImageDataset`](#timm.data.IterableImageDataset), the Hugging Face dataset streamed, or downloaded and prepared first with `download=True` | Cache dir (optional) | `datasets` |
| `tfds/<name>` | [`IterableImageDataset`](#timm.data.IterableImageDataset) on TensorFlow Datasets, `download=True` downloads and builds | TFDS data dir | `tensorflow`, `tensorflow-datasets` |
| `wds/<name>` | [`IterableImageDataset`](#timm.data.IterableImageDataset) on WebDataset tar shards | Shard folder w/ an `_info.json` (or `info.json`) describing the splits | `webdataset` |

Dataset options:

- **`split`** selects the dataset split (default `validation`). For WebDataset it can also be a shard pattern with a
  sample count, `'imagenet-train-{0000..1023}.tar|1281167'`.
- **`search_split=True`** (default) makes folder datasets look for the sub-folder of `root` named `split` or,
  failing that, a synonym of a train (`train`, `training`) or eval (`val`, `valid`, `validation`, `eval`, `evaluation`)
  split.
- **`add_img_extensions`** adds to the default image extensions (`.png`, `.jpg` and `.jpeg`).
- **`class_map`** remaps class names to indices with a dict, a `.txt` (one class per line) or a `.pkl` file.

Iterable datasets (`hfids`, `tfds`, `wds`) also take:

- **`is_training`** enables shuffling.
- **`batch_size`** rounds each worker's share of the samples up to whole batches when training.
- **`num_samples`** sets the sample count, required when the reader can't determine it (`hfids` without split info).
- **`seed`** sets the random seed.
- **`repeats`** multiplies the samples per epoch, the readers pass over the data that many times.

Call `dataset.set_epoch()` each epoch to reshuffle.

Other keyword arguments pass through to the reader. For the Hugging Face, TFDS and WebDataset readers:

- **`input_key`** selects the image field (`image` by default, for `wds` a `jpg`, `png` or `webp` image).
- **`target_key`** selects the label field (`label` by default, for `wds` the `.cls` file or a `.json` sidecar field
  named by `target_key`). A comma separated value reads one binary field per class.
- **`target_format='multihot'`** reads dense multi-hot vectors instead of lists of class indices for multi-label data.
- **`target_transform=MultiLabelTarget(num_classes)`** encodes targets as float multi-hot tensors.
  Pass `dense=True` to [`MultiLabelTarget`](#timm.data.MultiLabelTarget) for multi-hot fields.

The folder and tar readers yield single class indices. See
[multi-label classification](../training_script#multi-label-classification) for details.

### Loaders

[`create_loader`](#timm.data.create_loader) creates the transform with `create_transform` from its own arguments,
assigns it to `dataset.transform` (replacing any existing transform) and returns a `DataLoader`. Training loaders
(`is_training=True`) shuffle and drop the last incomplete batch.

- **Prefetcher.** With `use_prefetcher=True` (default) the dataset yields `uint8` tensors, the batches are moved to
  `device` (default `cuda`, on a side stream for CUDA), converted to `img_dtype`, normalized and randomly erased there.
  Set `device` to the training device, or `use_prefetcher=False` for a plain `DataLoader` of normalized tensors.
- **Distributed.** `distributed=True` uses a `DistributedSampler` for map-style training datasets (or a repeated
  augmentation sampler w/ `num_aug_repeats`) and an ordered sampler for eval that pads each process to the same
  number of samples, which can slightly alter validation results. Iterable datasets shard themselves across
  processes and workers.
- **Collate.** `collate_fn` replaces the default (`fast_collate` w/ the prefetcher, `default_collate` without), e.g.
  with a [`FastCollateMixup`](#timm.data.FastCollateMixup).
- **Augmentation splits.** For AugMix w/ the JSD loss (`--aug-splits`), wrap the dataset in
  [`AugMixDataset`](#timm.data.AugMixDataset)`(dataset, num_splits=n)` and pass `num_aug_splits=n`. Each batch holds
  a clean split and `n - 1` augmented splits, concatenated along the batch dimension by the prefetcher.
- **Scheduled resolution.** `input_size_choices` (`--train-img-sizes`) picks the input size per batch, optionally
  with a batch size per choice (`batch_size_choices`) and a `progressive` schedule from the first to the last choice.

### Mixup and CutMix

[`Mixup`](#timm.data.Mixup) mixes a batch of normalized images on the device, after the loader.
[`FastCollateMixup`](#timm.data.FastCollateMixup) takes the same arguments and mixes `uint8` images while collating
in the loader workers, it requires the prefetcher. Both return dense `(batch_size, num_classes)` targets with
`label_smoothing` (default `0.1`) applied, to be used with a soft target loss such as
`timm.loss.SoftTargetCrossEntropy`, and require an even batch size.

```python
from timm.data import Mixup, FastCollateMixup

# Continuing the example above.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mixup_args = dict(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=1000)

# Prefetcher enabled (default), mixed in the loader's collate.
loader = create_loader(
    dataset,
    **data_config,
    batch_size=4,
    is_training=True,
    collate_fn=FastCollateMixup(**mixup_args),
    device=device,
)
for images, targets in loader:
    ...  # images normalized on device, targets (4, 1000)

# Prefetcher disabled, mixed in the training loop.
loader = create_loader(dataset, **data_config, batch_size=4, is_training=True, use_prefetcher=False)
mixup_fn = Mixup(**mixup_args)
for images, targets in loader:
    images, targets = mixup_fn(images.to(device), targets.to(device))
```

Mixup and CutMix controls:

- **`mixup_alpha > 0`** enables Mixup.
- **`cutmix_alpha > 0`** enables CutMix.
- **`cutmix_minmax`** enables CutMix with a box size range.
- **`switch_prob`** sets the probability of choosing CutMix when both methods are enabled.
- **`prob`** sets the probability of mixing at all.
- **`mode`** draws the parameters once per `batch`, per `pair` of elements or per `elem` (element).
- **`multi_label=True`** mixes dense multi-label targets, smoothing each label towards 0.5.
- **`mixup_enabled = False`** turns mixing off, e.g. for the final epochs (`--mixup-off-epoch`). For `FastCollateMixup`
  it is read in the worker processes, persistent workers keep the value they started with.

### NaFlex variable resolution

[`create_naflex_loader`](#timm.data.create_naflex_loader) loads images as variable length sequences of patches,
w/o resizing to a fixed square size, for models that take NaFlex patch inputs (e.g. the `naflexvit_*` models). Each
batch is a dict of `patches`, `patch_coord`, `patch_valid` and `seq_len`, padded to the batch's sequence length.

- **Training.** The dataset is wrapped in a [`NaFlexMapDatasetWrapper`](#timm.data.NaFlexMapDatasetWrapper) (map-style
  datasets only), which picks one of `train_seq_lens` per batch and random resize crops each image to fit that many
  patches. `batch_size` is the batch size at the longest sequence length, shorter lengths get proportionally larger
  batches (rounded down to a multiple of 8) so the tokens per batch stay roughly constant. `patch_size_choices` (w/
  `patch_size_choice_probs`) also varies the patch size per batch, for models that support it.
- **Eval.** Images are resized to fit `max_seq_len` patches at the given `patch_size` (required for eval) and padded
  to `max_seq_len`.
- **Mixup.** Pass a [`NaFlexMixup`](#timm.data.NaFlexMixup) as `mixup_fn`, it mixes within each batch and does not
  support `mode` or `cutmix_minmax`.

The `train.py` options are:

- **`--naflex-loader`** enables the NaFlex loader.
- **`--naflex-train-seq-lens`** sets the training sequence lengths.
- **`--naflex-max-seq-len`** sets the maximum evaluation sequence length.
- **`--naflex-patch-sizes`** sets the patch sizes for variable patch size training.
- **`--naflex-patch-size-probs`** sets the probabilities for those patch sizes.
- **`--naflex-loss-scale`** selects loss scaling by batch size (`none`, `sqrt` or `linear`).

## API

### Data config[[timm.data.resolve_data_config]]

#### timm.data.resolve_data_config[[timm.data.resolve_data_config]]

```python
timm.data.resolve_data_config(args = None, pretrained_cfg = None, model = None, use_test_size = False, verbose = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/config.py#L8)

#### timm.data.resolve_model_data_config[[timm.data.resolve_model_data_config]]

```python
timm.data.resolve_model_data_config(model, args = None, pretrained_cfg = None, use_test_size = False, verbose = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/config.py#L103)

**Parameters:**

model (nn.Module) : the model instance

args (dict) : command line arguments / configuration in dict form (overrides pretrained_cfg)

pretrained_cfg (dict) : pretrained model config (overrides pretrained_cfg attached to model)

use_test_size (bool) : use the test time input resolution (if one exists) instead of default train resolution

verbose (bool) : enable extra logging of resolved values

**Returns:**

dictionary of config

Resolve Model Data Config
This is equivalent to resolve_data_config() but with arguments re-ordered to put model first.

#### timm.data.resolve_input_data_config[[timm.data.resolve_input_data_config]]

```python
timm.data.resolve_input_data_config(model, args: typing.Optional[typing.Dict[str, typing.Any]] = None, pretrained_cfg: typing.Optional[typing.Dict[str, typing.Any]] = None, use_test_size: bool = False, verbose: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/model_config.py#L17)

Resolve preprocessing against the current model's input constraints.

Fixed dimensions and channels must agree with explicit requests. Flexible
models retain train/test preprocessing recommendations unless overridden.
Normalization uses the prefetch loaders' existing channel adaptation policy.

### Transforms[[timm.data.create_transform]]

#### timm.data.create_transform[[timm.data.create_transform]]

```python
timm.data.create_transform(input_size: typing.Union[int, typing.Tuple[int, int], typing.Tuple[int, int, int]] = 224, is_training: bool = False, no_aug: bool = False, train_crop_mode: typing.Optional[str] = None, scale: typing.Optional[typing.Tuple[float, float]] = None, ratio: typing.Optional[typing.Tuple[float, float]] = None, hflip: float = 0.5, vflip: float = 0.0, color_jitter: typing.Union[float, typing.Tuple[float, ...]] = 0.4, color_jitter_prob: typing.Optional[float] = None, grayscale_prob: float = 0.0, gaussian_blur_prob: float = 0.0, auto_augment: typing.Optional[str] = None, interpolation: str = 'bilinear', mean: typing.Tuple[float, ...] = (0.485, 0.456, 0.406), std: typing.Tuple[float, ...] = (0.229, 0.224, 0.225), re_prob: float = 0.0, re_mode: str = 'const', re_count: int = 1, re_num_splits: int = 0, crop_pct: typing.Optional[float] = None, crop_mode: typing.Optional[str] = None, crop_border_pixels: typing.Optional[int] = None, tf_preprocessing: bool = False, use_prefetcher: bool = False, normalize: bool = True, separate: bool = False, naflex: bool = False, patch_size: typing.Union[int, typing.Tuple[int, int]] = 16, max_seq_len: int = 576, patchify: bool = False, patchify_channels_last: bool = True, patchify_flatten: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/transforms_factory.py#L387)

**Parameters:**

input_size : Target input size (channels, height, width) tuple or size scalar.

is_training : Return training (random) transforms.

no_aug : Disable augmentation for training (useful for debug).

train_crop_mode : Training random crop mode ('rrc', 'rkrc', 'rkrr').

scale : Random resize scale range (crop area, < 1.0 => zoom in).

ratio : Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).

hflip : Horizontal flip probability.

vflip : Vertical flip probability.

color_jitter : Random color jitter component factors (brightness, contrast, saturation, hue). Scalar is applied as (scalar,) * 3 (no hue).

color_jitter_prob : Apply color jitter with this probability if not None (for SimlCLR-like aug).

grayscale_prob : Probability of converting image to grayscale (for SimCLR-like aug).

gaussian_blur_prob : Probability of applying gaussian blur (for SimCLR-like aug).

auto_augment : Auto augment configuration string (see auto_augment.py).

interpolation : Image interpolation mode.

mean : Image normalization mean.

std : Image normalization standard deviation.

re_prob : Random erasing probability.

re_mode : Random erasing fill mode.

re_count : Number of random erasing regions.

re_num_splits : Control split of random erasing across batch size.

crop_pct : Inference crop percentage (output size / resize size).

crop_mode : Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.

crop_border_pixels : Inference crop border of specified # pixels around edge of original image.

tf_preprocessing : Use TF 1.0 inference preprocessing for testing model ports

use_prefetcher : Pre-fetcher enabled. Do not convert image to tensor or normalize.

normalize : Normalization tensor output w/ provided mean/std (if prefetcher not used).

separate : Output transforms in 3-stage tuple.

patchify : Patchify the output instead of relying on prefetcher.

patchify_channels_last : Use channels-last layout within each patch.

patchify_flatten : Flatten eval patches into vectors. Disable when the model needs their spatial dimensions for patch-size interpolation.

**Returns:**

Composed transforms or tuple thereof

#### timm.data.rand_augment_transform[[timm.data.rand_augment_transform]]

```python
timm.data.rand_augment_transform(config_str: str, hparams: typing.Optional[typing.Dict] = None, transforms: typing.Union[str, typing.Dict, typing.List, NoneType] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/auto_augment.py#L762)

**Parameters:**

config_str (str) : String defining configuration of random augmentation. Consists of multiple sections separated by dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining sections, not order specific determine * 'm' - integer magnitude of rand augment * 'n' - integer num layers (number of transform ops selected per image) * 'p' - float probability of applying each layer (default 0.5) * 'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100) * 'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10) * 'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0) * 't' - str name of transform set to use

hparams (dict) : Other hparams (kwargs) for the RandAugmentation scheme

**Returns:**

A PyTorch compatible Transform

Create a RandAugment transform

Examples:
- `'rand-m9-n3-mstd0.5'` is RandAugment with magnitude 9, 3 layers and a magnitude std of 0.5.
- `'rand-mstd1-tweights'` is RandAugment with a magnitude std of 1.0 and weighted transform selection, at the
  default magnitude of 10 and 2 layers.

#### timm.data.auto_augment_transform[[timm.data.auto_augment_transform]]

```python
timm.data.auto_augment_transform(config_str: str, hparams: typing.Optional[typing.Dict] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/auto_augment.py#L586)

**Parameters:**

config_str : String defining configuration of auto augmentation. Consists of multiple sections separated by dashes ('-'). The first section defines the AutoAugment policy (one of 'v0', 'v0r', 'original', 'originalr'). While the remaining sections define other arguments * 'mstd' -  float std deviation of magnitude noise applied

hparams : Other hparams (kwargs) for the AutoAugmentation scheme

**Returns:**

A PyTorch compatible Transform

Create a AutoAugment transform

Examples:
- `'original-mstd0.5'` is AutoAugment with the original policy and a magnitude std of 0.5.

#### timm.data.auto_augment.augment_and_mix_transform[[timm.data.auto_augment.augment_and_mix_transform]]

```python
timm.data.auto_augment.augment_and_mix_transform(config_str: str, hparams: typing.Optional[typing.Dict] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/auto_augment.py#L948)

**Parameters:**

config_str (str) : String defining configuration of random augmentation. Consists of multiple sections separated by dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining sections, not order specific determine 'm' - integer magnitude (severity) of augmentation mix (default: 3) 'w' - integer width of augmentation chain (default: 3) 'd' - integer depth of augmentation chain (-1 is random [1, 3], default: -1) 'b' - integer (bool), blend each branch of chain into end result without a final blend, less CPU (default: 0) 'mstd' -  float std deviation of magnitude noise applied (default: 0) Ex 'augmix-m5-w4-d2' results in AugMix with severity 5, chain width 4, chain depth 2 

hparams : Other hparams (kwargs) for the Augmentation transforms

**Returns:**

A PyTorch compatible Transform

Create AugMix PyTorch transform

#### timm.data.random_erasing.RandomErasing[[timm.data.random_erasing.RandomErasing]]

```python
timm.data.random_erasing.RandomErasing(probability = 0.5, min_area = 0.02, max_area = 0.3333333333333333, min_aspect = 0.3, max_aspect = None, mode = 'const', min_count = 1, max_count = None, num_splits = 0, device = 'cuda')
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/random_erasing.py#L26)

**Parameters:**

probability : Probability that the Random Erasing operation will be performed.

min_area : Minimum percentage of erased area wrt input image area.

max_area : Maximum percentage of erased area wrt input image area.

min_aspect : Minimum aspect ratio of erased area.

mode : pixel color mode, one of 'const', 'rand', or 'pixel' 'const' - erase block is constant color of 0 for all channels 'rand'  - erase block is same per-channel random (normal) color 'pixel' - erase block is per-pixel random (normal) color

Randomly selects a rectangle region in an image and erases its pixels.
'Random Erasing Data Augmentation' by Zhong et al.
See https://arxiv.org/pdf/1708.04896.pdf

This variant of RandomErasing is intended to be applied to either a batch
or single image tensor after it has been normalized by dataset mean and std.

max_count: maximum number of erasing blocks per image, area per box is scaled by count.
per-image count is randomly chosen between 1 and this value.

### Datasets[[timm.data.create_dataset]]

#### timm.data.create_dataset[[timm.data.create_dataset]]

```python
timm.data.create_dataset(name: str, root: typing.Optional[str] = None, split: str = 'validation', search_split: bool = True, class_map: dict = None, load_bytes: bool = False, is_training: bool = False, download: bool = False, batch_size: int = 1, num_samples: typing.Optional[int] = None, seed: int = 42, repeats: int = 0, input_img_mode: str = 'RGB', trust_remote_code: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/dataset_factory.py#L63)

**Parameters:**

name : Dataset name, empty is okay for folder based datasets. The prefix (e.g. `hfds/`) is case-insensitive, the remainder is passed through as-is (case-sensitive) except for `torch/` names.

root : Root folder of dataset (All)

split : Dataset split (All)

search_split : Search for split specific child fold from root so one can specify `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (Folder, Torch)

class_map : Specify class -> index mapping via text file or dict (Folder)

load_bytes : Load data, return images as undecoded bytes (Folder)

download : Download dataset if not present and supported (HFIDS, TFDS, Torch)

is_training : Create dataset in train mode, this is different from the split. For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS, WDS, HFIDS)

batch_size : Batch size hint for iterable datasets (TFDS, WDS, HFIDS)

seed : Seed for iterable datasets (TFDS, WDS, HFIDS)

repeats : Dataset repeats per iteration i.e. epoch (TFDS, WDS, HFIDS)

input_img_mode : Input image color conversion mode e.g. 'RGB', 'L' (folder, TFDS, WDS, HFDS, HFIDS)

trust_remote_code : Trust remote code in Hugging Face Datasets if True (HFDS, HFIDS)

- ****kwargs** : Other args to pass through to underlying Dataset and/or Reader classes

**Returns:**

Dataset object

Dataset factory method

In parentheses after each arg are the type of dataset supported for each arg, one of:
* Folder - default, timm folder (or tar) based ImageDataset
* Torch - torchvision based datasets
* HFDS - Hugging Face Datasets
* HFIDS - Hugging Face Datasets Iterable (streaming mode, with IterableDataset)
* TFDS - Tensorflow-datasets wrapper in IterabeDataset interface via IterableImageDataset
* WDS - Webdataset
* All - any of the above

#### timm.data.ImageDataset[[timm.data.ImageDataset]]

```python
timm.data.ImageDataset(root, reader = None, split = 'train', class_map = None, load_bytes = False, input_img_mode = 'RGB', transform = None, target_transform = None, additional_features = None, **kwargs)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/dataset.py#L23)

#### timm.data.IterableImageDataset[[timm.data.IterableImageDataset]]

```python
timm.data.IterableImageDataset(root, reader = None, split = 'train', class_map = None, is_training = False, batch_size = 1, num_samples = None, seed = 42, repeats = 0, download = False, input_img_mode = 'RGB', input_key = None, target_key = None, transform = None, target_transform = None, max_steps = None, **kwargs)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/dataset.py#L95)

#### timm.data.AugMixDataset[[timm.data.AugMixDataset]]

```python
timm.data.AugMixDataset(dataset, num_splits = 2)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/dataset.py#L175)

Dataset wrapper to perform AugMix or other clean/augmentation mixes

#### timm.data.MultiLabelTarget[[timm.data.MultiLabelTarget]]

```python
timm.data.MultiLabelTarget(num_classes: int, dense: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/target_transforms.py#L5)

Encode a multi-label target as a float32 multi-hot tensor of length num_classes.

By default the target is a sequence of positive class indices: an empty sequence
means no positive labels, repeats are allowed, and negative, out-of-range, or
non-integer values are rejected. With dense=True the target is already a vector
of length num_classes holding bool, int, or float values in [0, 1]; it is validated
and cast to float32, keeping soft values.

### Loaders[[timm.data.create_loader]]

#### timm.data.create_loader[[timm.data.create_loader]]

```python
timm.data.create_loader(dataset: typing.Union[timm.data.dataset.ImageDataset, timm.data.dataset.IterableImageDataset], input_size: typing.Union[int, typing.Tuple[int, int], typing.Tuple[int, int, int]], batch_size: int, is_training: bool = False, no_aug: bool = False, re_prob: float = 0.0, re_mode: str = 'const', re_count: int = 1, re_split: bool = False, train_crop_mode: typing.Optional[str] = None, scale: typing.Optional[typing.Tuple[float, float]] = None, ratio: typing.Optional[typing.Tuple[float, float]] = None, hflip: float = 0.5, vflip: float = 0.0, color_jitter: float = 0.4, color_jitter_prob: typing.Optional[float] = None, grayscale_prob: float = 0.0, gaussian_blur_prob: float = 0.0, auto_augment: typing.Optional[str] = None, num_aug_repeats: int = 0, num_aug_splits: int = 0, interpolation: str = 'bilinear', mean: typing.Tuple[float, ...] = (0.485, 0.456, 0.406), std: typing.Tuple[float, ...] = (0.229, 0.224, 0.225), num_workers: int = 1, distributed: bool = False, crop_pct: typing.Optional[float] = None, crop_mode: typing.Optional[str] = None, crop_border_pixels: typing.Optional[int] = None, collate_fn: typing.Optional[typing.Callable] = None, pin_memory: bool = False, fp16: bool = False, img_dtype: dtype = torch.float32, device: device = torch.device(), use_prefetcher: bool = True, use_multi_epochs_loader: bool = False, persistent_workers: bool = True, worker_seeding: str = 'all', tf_preprocessing: bool = False, input_size_choices: typing.Optional[typing.Sequence[typing.Union[int, typing.Tuple[int, int], typing.Tuple[int, int, int]]]] = None, batch_size_choices: typing.Optional[typing.Sequence[int]] = None, batch_choice_weights: typing.Optional[typing.Sequence[float]] = None, batch_choice_seed: int = 0, batch_choice_schedule: str = 'constant', batch_schedule_epochs: typing.Optional[int] = None, batch_schedule_spread: float = 0.65, batch_schedule_random_mix: float = 0.1, num_batches: typing.Optional[int] = None)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/loader.py#L223)

**Parameters:**

dataset : The image dataset to load.

input_size : Target input size (channels, height, width) tuple or size scalar.

batch_size : Number of samples in a batch.

is_training : Return training (random) transforms.

no_aug : Disable augmentation for training (useful for debug).

re_prob : Random erasing probability.

re_mode : Random erasing fill mode.

re_count : Number of random erasing regions.

re_split : Control split of random erasing across batch size.

scale : Random resize scale range (crop area, &amp;lt; 1.0 => zoom in).

ratio : Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).

hflip : Horizontal flip probability.

vflip : Vertical flip probability.

color_jitter : Random color jitter component factors (brightness, contrast, saturation, hue). Scalar is applied as (scalar,) * 3 (no hue).

color_jitter_prob : Apply color jitter with this probability if not None (for SimlCLR-like aug

grayscale_prob : Probability of converting image to grayscale (for SimCLR-like aug).

gaussian_blur_prob : Probability of applying gaussian blur (for SimCLR-like aug).

auto_augment : Auto augment configuration string (see auto_augment.py).

num_aug_repeats : Enable special sampler to repeat same augmentation across distributed GPUs.

num_aug_splits : Enable mode where augmentations can be split across the batch.

interpolation : Image interpolation mode.

mean : Image normalization mean.

std : Image normalization standard deviation.

num_workers : Num worker processes per DataLoader.

distributed : Enable dataloading for distributed training.

crop_pct : Inference crop percentage (output size / resize size).

crop_mode : Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.

crop_border_pixels : Inference crop border of specified # pixels around edge of original image.

collate_fn : Override default collate_fn.

pin_memory : Pin memory for device transfer.

fp16 : Deprecated argument for half-precision input dtype. Use img_dtype.

img_dtype : Data type for input image.

device : Device to transfer inputs and targets to.

use_prefetcher : Use efficient pre-fetcher to load samples onto device.

use_multi_epochs_loader --

persistent_workers : Enable persistent worker processes when `num_workers` is greater than zero.

worker_seeding : Control worker random seeding at init.

tf_preprocessing : Use TF 1.0 inference preprocessing for testing model ports.

input_size_choices : Per-batch input size choices for scheduled resolution training.

batch_size_choices : Batch size corresponding to each input size choice. Uses `batch_size` for all choices when not specified.

batch_choice_weights : Sampling weights for input size choices. Uniform when not specified.

batch_choice_seed : Random seed used to create and shuffle the batch schedule.

batch_choice_schedule : Choice schedule mode, either `constant` or `progressive`.

batch_schedule_epochs : Number of epochs over which a progressive schedule moves through its choices.

batch_schedule_spread : Standard deviation of the progressive choice window, in choice-index units.

batch_schedule_random_mix : Fraction of uniform random exploration over nonzero-weight choices in a progressive schedule.

num_batches : Fixed number of loader batches per epoch. Inferred from the schedule-average batch size for progressive schedules when not specified.

**Returns:**

DataLoader

### Mixup and CutMix[[timm.data.Mixup]]

#### timm.data.Mixup[[timm.data.Mixup]]

```python
timm.data.Mixup(mixup_alpha = 1.0, cutmix_alpha = 0.0, cutmix_minmax = None, prob = 1.0, switch_prob = 0.5, mode = 'batch', correct_lam = True, label_smoothing = 0.1, num_classes = 1000, multi_label = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/mixup.py#L96)

**Parameters:**

mixup_alpha (float) : mixup alpha value, mixup is active if > 0.

cutmix_alpha (float) : cutmix alpha value, cutmix is active if > 0.

cutmix_minmax (List[float]) : cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.

prob (float) : probability of applying mixup or cutmix per batch or element

switch_prob (float) : probability of switching to cutmix instead of mixup when both are active

mode (str) : how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)

correct_lam (bool) : apply lambda correction when cutmix bbox clipped by image borders

label_smoothing (float) : apply label smoothing to the mixed target tensor

num_classes (int) : number of classes for target

multi_label (bool) : Mix dense multi-label targets, smoothing each label towards 0.5.

Mixup/Cutmix that applies different params to each element or whole batch

#### timm.data.FastCollateMixup[[timm.data.FastCollateMixup]]

```python
timm.data.FastCollateMixup(mixup_alpha = 1.0, cutmix_alpha = 0.0, cutmix_minmax = None, prob = 1.0, switch_prob = 0.5, mode = 'batch', correct_lam = True, label_smoothing = 0.1, num_classes = 1000, multi_label = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/mixup.py#L229)

Fast Collate w/ Mixup/Cutmix that applies different params to each element or whole batch

A Mixup impl that's performed while collating the batches.

### NaFlex[[timm.data.create_naflex_loader]]

#### timm.data.create_naflex_loader[[timm.data.create_naflex_loader]]

```python
timm.data.create_naflex_loader(dataset, patch_size: typing.Union[int, typing.Tuple[int, int], NoneType] = None, patch_size_choices: typing.Optional[typing.List[int]] = None, patch_size_choice_probs: typing.Optional[typing.List[float]] = None, train_seq_lens: typing.Tuple[int, ...] = (128, 256, 576, 784, 1024), max_seq_len: int = 576, batch_size: int = 32, is_training: bool = False, mixup_fn: typing.Optional[typing.Callable] = None, no_aug: bool = False, re_prob: float = 0.0, re_mode: str = 'const', re_count: int = 1, re_split: bool = False, train_crop_mode: typing.Optional[str] = None, scale: typing.Optional[typing.Tuple[float, float]] = None, ratio: typing.Optional[typing.Tuple[float, float]] = None, hflip: float = 0.5, vflip: float = 0.0, color_jitter: float = 0.4, color_jitter_prob: typing.Optional[float] = None, grayscale_prob: float = 0.0, gaussian_blur_prob: float = 0.0, auto_augment: typing.Optional[str] = None, num_aug_repeats: int = 0, num_aug_splits: int = 0, interpolation: str = 'bilinear', mean: typing.Tuple[float, ...] = (0.485, 0.456, 0.406), std: typing.Tuple[float, ...] = (0.229, 0.224, 0.225), channels: int = 3, crop_pct: typing.Optional[float] = None, crop_mode: typing.Optional[str] = None, crop_border_pixels: typing.Optional[int] = None, num_workers: int = 4, distributed: bool = False, rank: int = 0, world_size: int = 1, seed: int = 42, epoch: int = 0, use_prefetcher: bool = True, pin_memory: bool = True, img_dtype: dtype = torch.float32, device: typing.Union[str, torch.device] = torch.device(), persistent_workers: bool = True, worker_seeding: str = 'all', patchify_channels_last: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/naflex_loader.py#L239)

**Parameters:**

dataset : Dataset to load from.

patch_size : Single patch size to use.

patch_size_choices : List of patch sizes for variable patch size training.

patch_size_choice_probs : Probabilities for each patch size choice.

train_seq_lens : Training sequence lengths for dynamic batching.

max_seq_len : Fixed sequence length for validation.

batch_size : Batch size for validation and max training sequence length.

is_training : Whether this is for training (enables dynamic batching).

mixup_fn : Optional mixup function.

no_aug : Disable augmentation.

re_prob : Random erasing probability.

re_mode : Random erasing mode.

re_count : Maximum number of erasing rectangles.

re_split : Random erasing split flag.

train_crop_mode : Training crop mode.

scale : Scale range for random resize crop.

ratio : Aspect ratio range for random resize crop.

hflip : Horizontal flip probability.

vflip : Vertical flip probability.

color_jitter : Color jitter factor.

color_jitter_prob : Color jitter probability.

grayscale_prob : Grayscale conversion probability.

gaussian_blur_prob : Gaussian blur probability.

auto_augment : AutoAugment policy.

num_aug_repeats : Number of augmentation repeats.

num_aug_splits : Number of augmentation splits.

interpolation : Interpolation method.

mean : Normalization mean values.

std : Normalization standard deviation values.

channels : Number of image channels, used for per-channel normalization in the prefetcher.

crop_pct : Crop percentage for validation.

crop_mode : Crop mode.

crop_border_pixels : Crop border pixels.

num_workers : Number of data loading workers.

distributed : Whether using distributed training.

rank : Process rank for distributed training.

world_size : Total number of processes.

seed : Random seed.

epoch : Starting epoch.

use_prefetcher : Whether to use prefetching.

pin_memory : Whether to pin memory.

img_dtype : Image data type.

device : Device to move tensors to.

persistent_workers : Whether to use persistent workers when `num_workers` is greater than zero.

worker_seeding : Worker seeding mode.

patchify_channels_last : Per-patch flat layout. `True` (default): channel index varies fastest (NaFlex default). `False`: C-P-P (channels first — Gemma4 / HF native layout). Forwarded to the upstream `Patchify` and to the prefetcher's normalization layout.

**Returns:**

DataLoader or NaFlexPrefetchLoader instance.

Create a data loader with dynamic sequence length sampling for training.

#### timm.data.NaFlexMapDatasetWrapper[[timm.data.NaFlexMapDatasetWrapper]]

```python
timm.data.NaFlexMapDatasetWrapper(base_dataset: Dataset, patch_size: typing.Union[int, typing.Tuple[int, int], NoneType] = None, patch_size_choices: typing.Optional[typing.List[int]] = None, patch_size_choice_probs: typing.Optional[typing.List[float]] = None, seq_lens: typing.Tuple[int, ...] = (128, 256, 576, 784, 1024), max_tokens_per_batch: int = 16384, transform_factory: typing.Optional[typing.Callable] = None, mixup_fn: typing.Optional[typing.Callable] = None, seed: int = 42, shuffle: bool = True, distributed: bool = False, rank: int = 0, world_size: int = 1, epoch: int = 0, batch_divisor: int = 8, patchify_channels_last: bool = True)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/naflex_dataset.py#L200)

IterableDataset wrapper for a map-style base dataset.

Yields batches with variable sequence lengths. It calculates a canonical
batch schedule (sequence length, batch size pairs) once based on the
total dataset size (padded for distribution). Each epoch, it shuffles
the order of this canonical schedule and the dataset indices.
This ensures a consistent number of batches and samples per epoch
across all ranks. Handles distributed training and multiple workers.

#### set_epoch[[timm.data.NaFlexMapDatasetWrapper.set_epoch]]

```python
set_epoch(epoch: int)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/naflex_dataset.py#L479)

**Parameters:**

epoch : New epoch number.

Set the multiprocessing-safe epoch read by DataLoader workers.

#### timm.data.NaFlexMixup[[timm.data.NaFlexMixup]]

```python
timm.data.NaFlexMixup(num_classes: int, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0, switch_prob: float = 0.5, prob: float = 1.0, local_shuffle: int = 4, label_smoothing: float = 0.0, mixup_off_epoch: int = 0, multi_label: bool = False)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/naflex_mixup.py#L168)

Callable wrapper that combines mixing and target generation.

#### set_epoch[[timm.data.NaFlexMixup.set_epoch]]

```python
set_epoch(epoch: int)
```

[Source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/timm/data/naflex_mixup.py#L209)

Update the mixup state for the given epoch.

### Losses
https://huggingface.co/docs/timm/v1.0.31/reference/loss.md
