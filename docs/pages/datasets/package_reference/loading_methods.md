# Loading methods

Methods for listing and loading datasets:

## Datasets[[datasets.load_dataset]]

#### datasets.load_dataset[[datasets.load_dataset]]

```python
datasets.load_dataset(path: str, name: typing.Optional[str] = None, data_dir: typing.Optional[str] = None, data_files: typing.Union[str, collections.abc.Sequence[str], collections.abc.Mapping[str, typing.Union[str, collections.abc.Sequence[str]]], NoneType] = None, split: typing.Union[str, datasets.splits.Split, list[str], list[datasets.splits.Split], NoneType] = None, cache_dir: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, verification_mode: typing.Union[datasets.utils.info_utils.VerificationMode, str, NoneType] = None, keep_in_memory: typing.Optional[bool] = None, save_infos: bool = False, revision: typing.Union[datasets.utils.version.Version, str, NoneType] = None, token: typing.Union[bool, str, NoneType] = None, streaming: bool = False, num_proc: typing.Optional[int] = None, storage_options: typing.Optional[dict] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/load.py#L1446)

**Parameters:**

path (`str`) : Path or name of the dataset.  - if `path` is a dataset repository on the HF hub (list all available datasets with `huggingface_hub.list_datasets`) -> load the dataset from supported files in the repository (csv, json, parquet, etc.) e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing the data files.  - if `path` is a directory within a Storage Bucket on the HF Hub (list your buckets with `huggingface_hub.list_buckets`) -> load the dataset from supported files in the directory (csv, json, parquet, etc.) e.g. `'buckets/username/bucket_name/my_dataset'`.  - if `path` is a local directory -> load the dataset from supported files in the directory (csv, json, parquet, etc.) e.g. `'./path/to/directory/with/my/csv/data'`.  - if `path` is the name of a dataset builder and `data_files` or `data_dir` is specified (available builders are "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder", "meshfolder") -> load the dataset from the files in `data_files` or `data_dir` e.g. `'parquet'`.  Use a `hf://` path like `'hf://datasets/username/dataset_name'` to allow remote only. Use an absolute path to allow local only. 

name (`str`, *optional*) : Defining the name of the dataset configuration.

data_dir (`str`, *optional*) : Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.

data_files (`str` or `Sequence` or `Mapping`, *optional*) : Path(s) to source data file(s).

split (`Split` or `str`) : Which split of the data to load. If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`). If given, will return a single Dataset. Splits can be combined and specified like in tensorflow-datasets.

cache_dir (`str`, *optional*) : Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.

features (`Features`, *optional*) : Set the features type to use for this dataset.

download_config ([DownloadConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadConfig), *optional*) : Specific download configuration parameters.

download_mode ([DownloadMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, defaults to `REUSE_DATASET_IF_EXISTS`) : Download/generate mode.

verification_mode ([VerificationMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.VerificationMode) or `str`, defaults to `BASIC_CHECKS`) : Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).  

keep_in_memory (`bool`, defaults to `None`) : Whether to copy the dataset in-memory. If `None`, the dataset will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the [improve performance](../cache#improve-performance) section.

revision ([Version](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.Version) or `str`, *optional*) : Version of the dataset to load. As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch. You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.

token (`str` or `bool`, *optional*) : Optional string or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, or not specified, will get token from `"~/.huggingface"`.

streaming (`bool`, defaults to `False`) : If set to `True`, don't download the data files. Instead, it streams the data progressively while iterating on the dataset. An [IterableDataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.IterableDataset) or [IterableDatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.IterableDatasetDict) is returned instead in this case.  Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example. Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats like rar and xz are not yet supported. The tgz format doesn't allow streaming.

num_proc (`int`, *optional*, defaults to `None`) : Number of processes when downloading and generating the dataset locally. Multiprocessing is disabled by default.  

storage_options (`dict`, *optional*, defaults to `None`) : **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.  

- ****config_kwargs** (additional keyword arguments) : Keyword arguments to be passed to the `BuilderConfig` and used in the [DatasetBuilder](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DatasetBuilder).

**Returns:** [Dataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Dataset) or [DatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.DatasetDict)

- if `split` is not `None`: the dataset requested,
- if `split` is `None`, a [DatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.DatasetDict) with each split.

or [IterableDataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.IterableDataset) or [IterableDatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.IterableDatasetDict): if `streaming=True`

- if `split` is not `None`, the dataset is requested
- if `split` is `None`, a `~datasets.streaming.IterableDatasetDict` with each split.

Load a dataset from the Hugging Face Hub, or a local dataset.

You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with `huggingface_hub.list_datasets`.

A dataset is a directory that contains some data files in generic formats (JSON, CSV, Parquet, etc.) and possibly
in a generic structure (Webdataset, ImageFolder, AudioFolder, VideoFolder, MeshFolder, etc.)

This function does the following under the hood:

1. Load a dataset builder:

   * Find the most common data format in the dataset and pick its associated builder (JSON, CSV, Parquet, Webdataset, ImageFolder, AudioFolder, MeshFolder, etc.)
   * Find which file goes into which split (e.g. train/test) based on file and directory names or on the YAML configuration
   * It is also possible to specify `data_files` manually, and which dataset builder to use (e.g. "parquet").

2. Run the dataset builder:

   In the general case:

   * Download the data files from the dataset if they are not already available locally or cached.
   * Process and cache the dataset in typed Arrow tables for caching.

     Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python generic types.
     They can be directly accessed from disk, loaded in RAM or even streamed over the web.

   In the streaming case:

   * Don't download or cache anything. Instead, the dataset is lazily loaded and will be streamed on-the-fly when iterating on it.

3. Return a dataset built from the requested splits in `split` (default: all).

Example:

Load a dataset from the Hugging Face Hub:

```py
>>> from datasets import load_dataset
>>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')

# Load a subset or dataset configuration (here 'sst2')
>>> from datasets import load_dataset
>>> ds = load_dataset('nyu-mll/glue', 'sst2', split='train')

# Manual mapping of data files to splits
>>> data_files = {'train': 'train.csv', 'test': 'test.csv'}
>>> ds = load_dataset('namespace/your_dataset_name', data_files=data_files)

# Manual selection of a directory to load
>>> ds = load_dataset('namespace/your_dataset_name', data_dir='folder_name')
```

Load a dataset from a Storage Bucket on the Hugging Face Hub:

```py
>>> from datasets import load_dataset
>>> ds = load_dataset('buckets/username/bucket_name/rotten_tomatoes', split='train')
```

Load a local dataset:

```py
# Load a CSV file
>>> from datasets import load_dataset
>>> ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

# Load a JSON file
>>> from datasets import load_dataset
>>> ds = load_dataset('json', data_files='path/to/local/my_dataset.json')
```

Load an [IterableDataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.IterableDataset):

```py
>>> from datasets import load_dataset
>>> ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train', streaming=True)
```

Load an image dataset with the `ImageFolder` dataset builder:

```py
>>> from datasets import load_dataset
>>> ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')
```

#### datasets.load_from_disk[[datasets.load_from_disk]]

```python
datasets.load_from_disk(dataset_path: typing.Union[str, bytes, os.PathLike], keep_in_memory: typing.Optional[bool] = None, storage_options: typing.Optional[dict] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/load.py#L1704)

**Parameters:**

dataset_path (`path-like`) : Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3://my-bucket/dataset/train"`) of the [Dataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Dataset) or [DatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.DatasetDict) directory where the dataset/dataset-dict will be loaded from.

keep_in_memory (`bool`, defaults to `None`) : Whether to copy the dataset in-memory. If `None`, the dataset will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the [improve performance](../cache#improve-performance) section. 

storage_options (`dict`, *optional*) : Key/value pairs to be passed on to the file-system backend, if any.  

**Returns:** [Dataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Dataset) or [DatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.DatasetDict)

- If `dataset_path` is a path of a dataset directory: the dataset requested.
- If `dataset_path` is a path of a dataset dict directory, a [DatasetDict](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.DatasetDict) with each split.

Loads a dataset that was previously saved using [save_to_disk()](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Dataset.save_to_disk) from a dataset directory, or
from a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

Example:

```py
>>> from datasets import load_from_disk
>>> ds = load_from_disk('path/to/dataset/directory')
```

#### datasets.load_dataset_builder[[datasets.load_dataset_builder]]

```python
datasets.load_dataset_builder(path: str, name: typing.Optional[str] = None, data_dir: typing.Optional[str] = None, data_files: typing.Union[str, collections.abc.Sequence[str], collections.abc.Mapping[str, typing.Union[str, collections.abc.Sequence[str]]], NoneType] = None, cache_dir: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, revision: typing.Union[datasets.utils.version.Version, str, NoneType] = None, token: typing.Union[bool, str, NoneType] = None, storage_options: typing.Optional[dict] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/load.py#L1187)

**Parameters:**

path (`str`) : Path or name of the dataset.  - if `path` is a dataset repository on the HF hub (list all available datasets with `huggingface_hub.list_datasets`) -> load the dataset builder from supported files in the repository (csv, json, parquet, etc.) e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing the data files.  - if `path` is a directory within a Storage Bucket on the HF Hub (list your buckets with `huggingface_hub.list_buckets`) -> load the dataset from supported files in the directory (csv, json, parquet, etc.) e.g. `'buckets/username/bucket_name/my_dataset'`.  - if `path` is a local directory -> load the dataset builder from supported files in the directory (csv, json, parquet, etc.) e.g. `'./path/to/directory/with/my/csv/data'`.  - if `path` is the name of a dataset builder and `data_files` or `data_dir` is specified (available builders are "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder", "meshfolder") -> load the dataset builder from the files in `data_files` or `data_dir` e.g. `'parquet'`.  Use a `hf://` path like `'hf://datasets/username/dataset_name'` to allow remote only. Use an absolute path to allow local only. 

name (`str`, *optional*) : Defining the name of the dataset configuration.

data_dir (`str`, *optional*) : Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.

data_files (`str` or `Sequence` or `Mapping`, *optional*) : Path(s) to source data file(s).

cache_dir (`str`, *optional*) : Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.

features ([Features](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Features), *optional*) : Set the features type to use for this dataset.

download_config ([DownloadConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadConfig), *optional*) : Specific download configuration parameters.

download_mode ([DownloadMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, defaults to `REUSE_DATASET_IF_EXISTS`) : Download/generate mode.

revision ([Version](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.Version) or `str`, *optional*) : Version of the dataset to load. As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch. You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.

token (`str` or `bool`, *optional*) : Optional string or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, or not specified, will get token from `"~/.huggingface"`.

storage_options (`dict`, *optional*, defaults to `None`) : **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.   

- ****config_kwargs** (additional keyword arguments) : Keyword arguments to be passed to the [BuilderConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.BuilderConfig) and used in the [DatasetBuilder](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DatasetBuilder).

**Returns:**

[DatasetBuilder](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DatasetBuilder)

Load a dataset builder which can be used to:

- Inspect general information that is required to build a dataset (cache directory, config, dataset info, features, data files, etc.)
- Download and prepare the dataset as Arrow files in the cache
- Get a streaming dataset without downloading or caching anything

You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with `huggingface_hub.list_datasets`.

A dataset is a directory that contains some data files in generic formats (JSON, CSV, Parquet, etc.) and possibly
in a generic structure (Webdataset, ImageFolder, AudioFolder, VideoFolder, MeshFolder, etc.)

Example:

```py
>>> from datasets import load_dataset_builder
>>> ds_builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
>>> ds_builder.info.features
{'label': ClassLabel(names=['neg', 'pos']),
 'text': Value('string')}
```

#### datasets.get_dataset_config_names[[datasets.get_dataset_config_names]]

```python
datasets.get_dataset_config_names(path: str, revision: typing.Union[datasets.utils.version.Version, str, NoneType] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, data_files: typing.Union[str, list, dict, NoneType] = None, **download_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/inspect.py#L109)

**Parameters:**

path (`str`) : path to the dataset repository. Can be either:  - a local path to the dataset directory containing the data files, e.g. `'./dataset/squad'` - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with `huggingface_hub.list_datasets`), e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`

revision (`Union[str, datasets.Version]`, *optional*) : If specified, the dataset module will be loaded from the datasets repository at this version. By default: - it is set to the local version of the lib. - it will also try to load it from the main branch if it's not available at the local version of the lib. Specifying a version that is different from your local version of the lib might cause compatibility issues.

download_config ([DownloadConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadConfig), *optional*) : Specific download configuration parameters.

download_mode ([DownloadMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, defaults to `REUSE_DATASET_IF_EXISTS`) : Download/generate mode.

data_files (`Union[Dict, List, str]`, *optional*) : Defining the data_files of the dataset configuration.

- ****download_kwargs** (additional keyword arguments) : Optional attributes for [DownloadConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadConfig) which will override the attributes in `download_config` if supplied, for example `token`.

Get the list of available config names for a particular dataset.

Example:

```py
>>> from datasets import get_dataset_config_names
>>> get_dataset_config_names("nyu-mll/glue")
['cola',
 'sst2',
 'mrpc',
 'qqp',
 'stsb',
 'mnli',
 'mnli_mismatched',
 'mnli_matched',
 'qnli',
 'rte',
 'wnli',
 'ax']
```

#### datasets.get_dataset_infos[[datasets.get_dataset_infos]]

```python
datasets.get_dataset_infos(path: str, data_files: typing.Union[str, list, dict, NoneType] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, revision: typing.Union[datasets.utils.version.Version, str, NoneType] = None, token: typing.Union[bool, str, NoneType] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/inspect.py#L42)

**Parameters:**

path (`str`) : path to the dataset repository. Can be either:  - a local path to the dataset directory containing the data files, e.g. `'./dataset/squad'` - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with `huggingface_hub.list_datasets`), e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`

revision (`Union[str, datasets.Version]`, *optional*) : If specified, the dataset module will be loaded from the datasets repository at this version. By default: - it is set to the local version of the lib. - it will also try to load it from the main branch if it's not available at the local version of the lib. Specifying a version that is different from your local version of the lib might cause compatibility issues.

download_config ([DownloadConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadConfig), *optional*) : Specific download configuration parameters.

download_mode ([DownloadMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, defaults to `REUSE_DATASET_IF_EXISTS`) : Download/generate mode.

data_files (`Union[Dict, List, str]`, *optional*) : Defining the data_files of the dataset configuration.

token (`str` or `bool`, *optional*) : Optional string or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, or not specified, will get token from `"~/.huggingface"`.

- ****config_kwargs** (additional keyword arguments) : Optional attributes for builder class which will override the attributes if supplied.

Get the meta information about a dataset, returned as a dict mapping config name to DatasetInfoDict.

Example:

```py
>>> from datasets import get_dataset_infos
>>> get_dataset_infos('cornell-movie-review-data/rotten_tomatoes')
{'default': DatasetInfo(description="Movie Review Dataset.
 is a dataset of containing 5,331 positive and 5,331 negative processed
ences from Rotten Tomatoes movie reviews...), ...}
```

#### datasets.get_dataset_split_names[[datasets.get_dataset_split_names]]

```python
datasets.get_dataset_split_names(path: str, config_name: typing.Optional[str] = None, data_files: typing.Union[str, collections.abc.Sequence[str], collections.abc.Mapping[str, typing.Union[str, collections.abc.Sequence[str]]], NoneType] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, revision: typing.Union[datasets.utils.version.Version, str, NoneType] = None, token: typing.Union[bool, str, NoneType] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/inspect.py#L295)

**Parameters:**

path (`str`) : path to the dataset repository. Can be either:  - a local path to the dataset directory containing the data files, e.g. `'./dataset/squad'` - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with `huggingface_hub.list_datasets`), e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`

config_name (`str`, *optional*) : Defining the name of the dataset configuration.

data_files (`str` or `Sequence` or `Mapping`, *optional*) : Path(s) to source data file(s).

download_config ([DownloadConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadConfig), *optional*) : Specific download configuration parameters.

download_mode ([DownloadMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, defaults to `REUSE_DATASET_IF_EXISTS`) : Download/generate mode.

revision ([Version](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.Version) or `str`, *optional*) : Version of the dataset to load. As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch. You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.

token (`str` or `bool`, *optional*) : Optional string or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, or not specified, will get token from `"~/.huggingface"`.

- ****config_kwargs** (additional keyword arguments) : Optional attributes for builder class which will override the attributes if supplied.

Get the list of available splits for a particular config and dataset.

Example:

```py
>>> from datasets import get_dataset_split_names
>>> get_dataset_split_names('cornell-movie-review-data/rotten_tomatoes')
['train', 'validation', 'test']
```

## From files

Configurations used to load data files.
They are used when loading local files or a dataset repository:

- local files: `load_dataset("parquet", data_dir="path/to/data/dir")`
- dataset repository: `load_dataset("allenai/c4")`

You can pass arguments to `load_dataset` to configure data loading.
For example you can specify the `sep` parameter to define the [CsvConfig](/docs/datasets/v5.2.0/en/package_reference/loading_methods#datasets.packaged_modules.csv.CsvConfig) that is used to load the data:

```python
load_dataset("csv", data_dir="path/to/data/dir", sep="\t")
```

### Text[[datasets.packaged_modules.text.TextConfig]]

#### datasets.packaged_modules.text.TextConfig[[datasets.packaged_modules.text.TextConfig]]

```python
datasets.packaged_modules.text.TextConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, encoding: str = 'utf-8', encoding_errors: typing.Optional[str] = None, chunksize: int = 10485760, keep_linebreaks: bool = False, sample_by: typing.Literal['line', 'paragraph', 'document'] = 'line')
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/text/text.py#L17)

**Parameters:**

features : (`Features`, *optional*): Cast the data to `features`.

encoding : (`str`, defaults to "utf-8"): Encoding to decode the file.

encoding_errors : (`str`, *optional*): Argument to define what to do in case of encoding error. This is the same as the `error` argument in `open()`.

chunksize : (`Features`, *optional*, defaults to "10MB"): Chunk size to read the data.

keep_linebreaks : (`bool`, defaults to False): Whether to keep line breaks.

sample_by (`Literal["line", "paragraph", "document"]`, defaults to "line") : Whether to load data per line, praragraph or document. By default one row in the dataset = one line.

BuilderConfig for text files.

#### datasets.packaged_modules.text.Text[[datasets.packaged_modules.text.Text]]

```python
datasets.packaged_modules.text.Text(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/text/text.py#L45)

### CSV[[datasets.packaged_modules.csv.CsvConfig]]

#### datasets.packaged_modules.csv.CsvConfig[[datasets.packaged_modules.csv.CsvConfig]]

```python
datasets.packaged_modules.csv.CsvConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, sep: str = ',', delimiter: typing.Optional[str] = None, header: typing.Union[int, list[int], str, NoneType] = 'infer', names: typing.Optional[list[str]] = None, column_names: typing.Optional[list[str]] = None, index_col: typing.Union[int, str, list[int], list[str], NoneType] = None, usecols: typing.Union[list[int], list[str], NoneType] = None, prefix: typing.Optional[str] = None, mangle_dupe_cols: bool = True, engine: typing.Optional[typing.Literal['c', 'python', 'pyarrow']] = None, converters: dict = None, true_values: typing.Optional[list] = None, false_values: typing.Optional[list] = None, skipinitialspace: bool = False, skiprows: typing.Union[int, list[int], NoneType] = None, nrows: typing.Optional[int] = None, na_values: typing.Union[str, list[str], NoneType] = None, keep_default_na: bool = True, na_filter: bool = True, verbose: bool = False, skip_blank_lines: bool = True, thousands: typing.Optional[str] = None, decimal: str = '.', lineterminator: typing.Optional[str] = None, quotechar: str = '"', quoting: int = 0, escapechar: typing.Optional[str] = None, comment: typing.Optional[str] = None, encoding: typing.Optional[str] = None, dialect: typing.Optional[str] = None, error_bad_lines: bool = True, warn_bad_lines: bool = True, skipfooter: int = 0, doublequote: bool = True, memory_map: bool = False, float_precision: typing.Optional[str] = None, chunksize: int = 10000, features: typing.Optional[datasets.features.features.Features] = None, encoding_errors: typing.Optional[str] = 'strict', on_bad_lines: typing.Literal['error', 'warn', 'skip'] = 'error', date_format: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/csv/csv.py#L25)

BuilderConfig for CSV.

#### datasets.packaged_modules.csv.Csv[[datasets.packaged_modules.csv.Csv]]

```python
datasets.packaged_modules.csv.Csv(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/csv/csv.py#L145)

### JSON[[datasets.packaged_modules.json.JsonConfig]]

#### datasets.packaged_modules.json.JsonConfig[[datasets.packaged_modules.json.JsonConfig]]

```python
datasets.packaged_modules.json.JsonConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, encoding: str = 'utf-8', encoding_errors: typing.Optional[str] = None, field: typing.Optional[str] = None, use_threads: bool = True, block_size: typing.Optional[int] = None, chunksize: int = 10485760, newlines_in_values: typing.Optional[bool] = None, on_mixed_types: typing.Optional[typing.Literal['use_json']] = 'use_json', parse_agent_traces: bool = True)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/json/json.py#L46)

BuilderConfig for JSON.

#### datasets.Json[[datasets.Json]]

```python
datasets.Json(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/json/json.py#L64)

### XML[[datasets.packaged_modules.xml.XmlConfig]]

#### datasets.packaged_modules.xml.XmlConfig[[datasets.packaged_modules.xml.XmlConfig]]

```python
datasets.packaged_modules.xml.XmlConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, encoding: str = 'utf-8', encoding_errors: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/xml/xml.py#L15)

BuilderConfig for xml files.

#### datasets.packaged_modules.xml.Xml[[datasets.packaged_modules.xml.Xml]]

```python
datasets.packaged_modules.xml.Xml(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/xml/xml.py#L23)

### Parquet[[datasets.packaged_modules.parquet.ParquetConfig]]

#### datasets.packaged_modules.parquet.ParquetConfig[[datasets.packaged_modules.parquet.ParquetConfig]]

```python
datasets.packaged_modules.parquet.ParquetConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, batch_size: typing.Optional[int] = None, columns: typing.Optional[list[str]] = None, features: typing.Optional[datasets.features.features.Features] = None, filters: typing.Union[pyarrow._compute.Expression, list[tuple], list[list[tuple]], NoneType] = None, fragment_scan_options: typing.Optional[pyarrow._dataset_parquet.ParquetFragmentScanOptions] = None, on_bad_files: typing.Literal['error', 'warn', 'skip'] = 'error')
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/parquet/parquet.py#L20)

**Parameters:**

batch_size (`int`, *optional*) : Size of the RecordBatches to iterate on. The default is the row group size (defined by the first row group).

columns (`list[str]`, *optional*) : List of columns to load, the other ones are ignored. All columns are loaded by default.

features : (`Features`, *optional*): Cast the data to `features`.

filters (`Union[pyarrow.dataset.Expression, list[tuple], list[list[tuple]]]`, *optional*) : Return only the rows matching the filter. If possible the predicate will be pushed down to exploit the partition information or internal metadata found in the data source, e.g. Parquet statistics. Otherwise filters the loaded RecordBatches before yielding them.

fragment_scan_options (`pyarrow.dataset.ParquetFragmentScanOptions`, *optional*) : Scan-specific options for Parquet fragments. This is especially useful to configure buffering and caching.  

on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error") : Specify what to do upon encountering a bad file (a file that can't be read). Allowed values are : * 'error', raise an Exception when a bad file is encountered. * 'warn', raise a warning when a bad file is encountered and skip that file. * 'skip', skip bad files without raising or warning when they are encountered.  

BuilderConfig for Parquet.

Example:

Load a subset of columns:

```python
>>> ds = load_dataset(parquet_dataset_id, columns=["col_0", "col_1"])
```

Stream data and efficiently filter data, possibly skipping entire files or row groups:

```python
>>> filters = [("col_0", "==", 0)]
>>> ds = load_dataset(parquet_dataset_id, streaming=True, filters=filters)
```

Increase the minimum request size when streaming from 32MiB (default) to 128MiB and enable prefetching:

```python
>>> import pyarrow
>>> import pyarrow.dataset
>>> fragment_scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
...     cache_options=pyarrow.CacheOptions(
...         prefetch_limit=1,
...         range_size_limit=128 << 20
...     ),
... )
>>> ds = load_dataset(parquet_dataset_id, streaming=True, fragment_scan_options=fragment_scan_options)
```

#### datasets.packaged_modules.parquet.Parquet[[datasets.packaged_modules.parquet.Parquet]]

```python
datasets.packaged_modules.parquet.Parquet(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/parquet/parquet.py#L93)

### Arrow[[datasets.packaged_modules.arrow.ArrowConfig]]

#### datasets.packaged_modules.arrow.ArrowConfig[[datasets.packaged_modules.arrow.ArrowConfig]]

```python
datasets.packaged_modules.arrow.ArrowConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/arrow/arrow.py#L15)

BuilderConfig for Arrow.

#### datasets.packaged_modules.arrow.Arrow[[datasets.packaged_modules.arrow.Arrow]]

```python
datasets.packaged_modules.arrow.Arrow(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/arrow/arrow.py#L24)

### Vortex[[datasets.packaged_modules.vortex.VortexConfig]]

#### datasets.packaged_modules.vortex.VortexConfig[[datasets.packaged_modules.vortex.VortexConfig]]

```python
datasets.packaged_modules.vortex.VortexConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, batch_size: typing.Optional[int] = None, columns: typing.Optional[list[str]] = None, features: typing.Optional[datasets.features.features.Features] = None, filters: typing.Union[ForwardRef('vortex.expr.Expr'), list[tuple], list[list[tuple]], NoneType] = None, on_bad_files: typing.Literal['error', 'warn', 'skip'] = 'error')
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/vortex/vortex.py#L54)

**Parameters:**

batch_size (`int`, *optional*) : Size of the RecordBatches to iterate on. The default is defined by the Vortex scanner.

columns (`list[str]`, *optional*) : List of columns to load, the other ones are ignored. All columns are loaded by default.

features : (`Features`, *optional*): Cast the data to `features`.

filters (`Union[vortex.expr.Expr, list[tuple], list[list[tuple]]]`, *optional*) : Return only the rows matching the filter. The predicate is pushed down into the Vortex scan so only the matching rows are read. Filters given as a list of tuples (DNF, like the Parquet loader accepts) are converted to a Vortex expression. Nulls follow SQL semantics: a null satisfies no comparison, so a row whose filtered column is null is never returned. Note that `not in` therefore drops nulls, where the Parquet loader keeps them because it builds `~field.isin(values)` and a null is not in the set.

on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error") : Specify what to do upon encountering a bad file (a file that can't be read). Allowed values are : * 'error', raise an Exception when a bad file is encountered. * 'warn', raise a warning when a bad file is encountered and skip that file. * 'skip', skip bad files without raising or warning when they are encountered.

BuilderConfig for Vortex.

Example:

Load a subset of columns:

```python
>>> ds = load_dataset(vortex_dataset_id, columns=["col_0", "col_1"])
```

Stream data and efficiently filter data, skipping entire files or chunks when possible:

```python
>>> filters = [("col_0", "==", 0)]
>>> ds = load_dataset(vortex_dataset_id, streaming=True, filters=filters)
```

#### datasets.packaged_modules.vortex.Vortex[[datasets.packaged_modules.vortex.Vortex]]

```python
datasets.packaged_modules.vortex.Vortex(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/vortex/vortex.py#L226)

### SQL[[datasets.packaged_modules.sql.SqlConfig]]

#### datasets.packaged_modules.sql.SqlConfig[[datasets.packaged_modules.sql.SqlConfig]]

```python
datasets.packaged_modules.sql.SqlConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, sql: typing.Union[str, ForwardRef('sqlalchemy.sql.Selectable')] = None, con: typing.Union[str, ForwardRef('sqlalchemy.engine.Connection'), ForwardRef('sqlalchemy.engine.Engine'), ForwardRef('sqlite3.Connection')] = None, index_col: typing.Union[str, list[str], NoneType] = None, coerce_float: bool = True, params: typing.Union[list, tuple, dict, NoneType] = None, parse_dates: typing.Union[list, dict, NoneType] = None, columns: typing.Optional[list[str]] = None, chunksize: typing.Optional[int] = 10000, features: typing.Optional[datasets.features.features.Features] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/sql/sql.py#L25)

BuilderConfig for SQL.

#### datasets.packaged_modules.sql.Sql[[datasets.packaged_modules.sql.Sql]]

```python
datasets.packaged_modules.sql.Sql(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/sql/sql.py#L92)

### Images[[datasets.packaged_modules.imagefolder.ImageFolderConfig]]

#### datasets.packaged_modules.imagefolder.ImageFolderConfig[[datasets.packaged_modules.imagefolder.ImageFolderConfig]]

```python
datasets.packaged_modules.imagefolder.ImageFolderConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, drop_labels: bool = None, drop_metadata: bool = None, metadata_filenames: list = None, filters: typing.Union[pyarrow._compute.Expression, list[tuple], list[list[tuple]], NoneType] = None, decode_num_threads: int = 32)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/imagefolder/imagefolder.py#L9)

BuilderConfig for ImageFolder.

#### datasets.packaged_modules.imagefolder.ImageFolder[[datasets.packaged_modules.imagefolder.ImageFolder]]

```python
datasets.packaged_modules.imagefolder.ImageFolder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/imagefolder/imagefolder.py#L19)

### Audio[[datasets.packaged_modules.audiofolder.AudioFolderConfig]]

#### datasets.packaged_modules.audiofolder.AudioFolderConfig[[datasets.packaged_modules.audiofolder.AudioFolderConfig]]

```python
datasets.packaged_modules.audiofolder.AudioFolderConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, drop_labels: bool = None, drop_metadata: bool = None, metadata_filenames: list = None, filters: typing.Union[pyarrow._compute.Expression, list[tuple], list[list[tuple]], NoneType] = None, decode_num_threads: int = 32)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/audiofolder/audiofolder.py#L9)

Builder Config for AudioFolder.

#### datasets.packaged_modules.audiofolder.AudioFolder[[datasets.packaged_modules.audiofolder.AudioFolder]]

```python
datasets.packaged_modules.audiofolder.AudioFolder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/audiofolder/audiofolder.py#L19)

### Videos[[datasets.packaged_modules.videofolder.VideoFolderConfig]]

#### datasets.packaged_modules.videofolder.VideoFolderConfig[[datasets.packaged_modules.videofolder.VideoFolderConfig]]

```python
datasets.packaged_modules.videofolder.VideoFolderConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, drop_labels: bool = None, drop_metadata: bool = None, metadata_filenames: list = None, filters: typing.Union[pyarrow._compute.Expression, list[tuple], list[list[tuple]], NoneType] = None, decode_num_threads: int = 32)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/videofolder/videofolder.py#L9)

BuilderConfig for ImageFolder.

#### datasets.packaged_modules.videofolder.VideoFolder[[datasets.packaged_modules.videofolder.VideoFolder]]

```python
datasets.packaged_modules.videofolder.VideoFolder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/videofolder/videofolder.py#L19)

### HDF5[[datasets.packaged_modules.hdf5.HDF5Config]]

#### datasets.packaged_modules.hdf5.HDF5Config[[datasets.packaged_modules.hdf5.HDF5Config]]

```python
datasets.packaged_modules.hdf5.HDF5Config(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, batch_size: typing.Optional[int] = None, features: typing.Optional[datasets.features.features.Features] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/hdf5/hdf5.py#L33)

BuilderConfig for HDF5.

#### datasets.packaged_modules.hdf5.HDF5[[datasets.packaged_modules.hdf5.HDF5]]

```python
datasets.packaged_modules.hdf5.HDF5(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/hdf5/hdf5.py#L40)

ArrowBasedBuilder that converts HDF5 files to Arrow tables using the HF extension types.

### TsFile[[datasets.packaged_modules.tsfile.TsFileConfig]]

#### datasets.packaged_modules.tsfile.TsFileConfig[[datasets.packaged_modules.tsfile.TsFileConfig]]

```python
datasets.packaged_modules.tsfile.TsFileConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, table_name: Optional[str] = None, columns: Optional[list[str]] = None, start_time: Optional[Any] = None, end_time: Optional[Any] = None, input_batch_size: int = 65536, output_batch_size: int = 32, features: Optional[datasets.Features] = None, on_bad_files: Literal['error', 'warn', 'skip'] = 'error', timestamp_unit: Literal['s', 'ms', 'us', 'ns'] = 'ms', timestamp_tz: Optional[str] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/tsfile/tsfile.py#L143)

**Parameters:**

table_name (*str*, *optional*) : Name of the table to read. When unset, the first table found in the first valid file is used. Lookups are case-insensitive.

columns (*list[str]*, *optional*) : Subset of FIELD columns to keep. TAG columns and the TIME column are *always* returned (they identify the device / its timeline and cannot be excluded). Names that refer to TAG or TIME columns, or to fields absent from every file, resolve quietly: TAGs/TIME are emitted as usual, and never-seen fields become all-null list columns. When unset, all FIELDs are returned.

start_time, end_time (*datetime*, *date*, *pa.TimestampScalar*, ISO-8601 *str*, or *int*, *optional*) : Inclusive timestamp range. Either bound may be omitted. `datetime` values are taken in their own tz (UTC if naive); `int` is interpreted as a raw epoch in `timestamp_unit`.

input_batch_size (*int*, *optional*, defaults to 65_536) : Maximum number of rows fetched per Arrow batch from `TsFileReader.query_table`. Controls peak memory while streaming a single device.

output_batch_size (*int*, *optional*, defaults to 32) : Number of devices (output dataset rows) packed into each Arrow record batch yielded to the writer. Also the granularity at which the dataset progress bar advances; smaller values give more responsive feedback on slow per-device reads, larger ones reduce per-batch overhead.

features (*Features*, *optional*) : Final Features schema. When provided, the metadata scan over input files is skipped.

on_bad_files (*Literal["error", "warn", "skip"]*, *optional*, defaults to "error") : What to do if a file cannot be opened or lacks the requested table.

timestamp_unit (*Literal["s", "ms", "us", "ns"]*, *optional*, defaults to "ms") : Time unit for the timestamp column. IoTDB defaults to milliseconds.

timestamp_tz (*str*, *optional*) : Time zone for the timestamp column. `None` means timezone-naive.

BuilderConfig for TsFile (table model) — per-device wide format.

#### datasets.packaged_modules.tsfile.TsFile[[datasets.packaged_modules.tsfile.TsFile]]

```python
datasets.packaged_modules.tsfile.TsFile(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/tsfile/tsfile.py#L233)

Per-device wide-format builder for TsFile (table model).

### Pdf[[datasets.packaged_modules.pdffolder.PdfFolderConfig]]

#### datasets.packaged_modules.pdffolder.PdfFolderConfig[[datasets.packaged_modules.pdffolder.PdfFolderConfig]]

```python
datasets.packaged_modules.pdffolder.PdfFolderConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, drop_labels: bool = None, drop_metadata: bool = None, metadata_filenames: list = None, filters: typing.Union[pyarrow._compute.Expression, list[tuple], list[list[tuple]], NoneType] = None, decode_num_threads: int = 32)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/pdffolder/pdffolder.py#L9)

BuilderConfig for ImageFolder.

#### datasets.packaged_modules.pdffolder.PdfFolder[[datasets.packaged_modules.pdffolder.PdfFolder]]

```python
datasets.packaged_modules.pdffolder.PdfFolder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/pdffolder/pdffolder.py#L19)

### Nifti[[datasets.packaged_modules.niftifolder.NiftiFolderConfig]]

#### datasets.packaged_modules.niftifolder.NiftiFolderConfig[[datasets.packaged_modules.niftifolder.NiftiFolderConfig]]

```python
datasets.packaged_modules.niftifolder.NiftiFolderConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, drop_labels: bool = None, drop_metadata: bool = None, metadata_filenames: list = None, filters: typing.Union[pyarrow._compute.Expression, list[tuple], list[list[tuple]], NoneType] = None, decode_num_threads: int = 32)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/niftifolder/niftifolder.py#L9)

BuilderConfig for NiftiFolder.

#### datasets.packaged_modules.niftifolder.NiftiFolder[[datasets.packaged_modules.niftifolder.NiftiFolder]]

```python
datasets.packaged_modules.niftifolder.NiftiFolder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/niftifolder/niftifolder.py#L20)

### WebDataset[[datasets.packaged_modules.webdataset.WebDataset]]

#### datasets.packaged_modules.webdataset.WebDataset[[datasets.packaged_modules.webdataset.WebDataset]]

```python
datasets.packaged_modules.webdataset.WebDataset(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/webdataset/webdataset.py#L20)

### Harbor[[datasets.packaged_modules.harbor.HarborConfig]]

#### datasets.packaged_modules.harbor.HarborConfig[[datasets.packaged_modules.harbor.HarborConfig]]

```python
datasets.packaged_modules.harbor.HarborConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None, features: typing.Optional[datasets.features.features.Features] = None, strip_canary: bool = True, include_files: bool = True)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/harbor/harbor.py#L95)

**Parameters:**

features (`Features`, *optional*) : Set the features used to cast the generated rows. By default, the features of a Harbor task are used (see `Harbor`).

strip_canary (`bool`, *optional*, defaults to `True`) : Remove the leading "benchmark data canary" comment lines from `instruction.md` and from the instructions of the steps, as Harbor does when it sends them to an agent.

include_files (`bool`, *optional*, defaults to `True`) : List all the files of every task (`environment/`, `tests/`, `solution/`...) in the `files` column. Set it to `False` to skip listing the task directories, which is faster when the dataset is on a remote filesystem.

BuilderConfig for Harbor task repositories.

#### datasets.packaged_modules.harbor.Harbor[[datasets.packaged_modules.harbor.Harbor]]

```python
datasets.packaged_modules.harbor.Harbor(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/packaged_modules/harbor/harbor.py#L116)

Builder for Harbor task repositories: one row per task directory.

It generates the following features:

- **name** (`str`) -- name of the task, as `org/name` in `[task].name` of `task.toml`, or the
  name of the task directory when `task.toml` has no `[task]` section.
- **description** (`str`) -- `[task].description` of `task.toml`.
- **instruction** (`str`) -- contents of `instruction.md`.
- **keywords** (`list` of `str`) -- `[task].keywords` of `task.toml`.
- **schema_version** (`str`) -- `schema_version` declared by `task.toml`.
- **config** (`dict`) -- the whole `task.toml`, as a JSON object.
- **metadata** (`dict`) -- the free-form `[metadata]` section of `task.toml`.
- **files** (`list` of `str`) -- all the files of the task, relative to the task directory.
- **location** (`str`) -- path of the task directory, relative to the root of the dataset.

### Utilities
https://huggingface.co/docs/datasets/v5.2.0/package_reference/utilities.md
