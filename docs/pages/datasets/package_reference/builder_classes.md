# Builder classes

## Builders[[datasets.DatasetBuilder]]

🤗 Datasets relies on two main classes during the dataset building process: [DatasetBuilder](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DatasetBuilder) and [BuilderConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.BuilderConfig).

#### datasets.DatasetBuilder[[datasets.DatasetBuilder]]

```python
datasets.DatasetBuilder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L214)

**Parameters:**

cache_dir (`str`, *optional*) : Directory to cache data. Defaults to `"~/.cache/huggingface/datasets"`.

dataset_name (`str`, *optional*) : Name of the dataset, if different from the builder name. Useful for packaged builders like csv, imagefolder, audiofolder, etc. to reflect the difference between datasets that use the same packaged builder.

config_name (`str`, *optional*) : Name of the dataset configuration. It affects the data generated on disk. Different configurations will have their own subdirectories and versions. If not provided, the default configuration is used (if it exists).    Parameter `name` was renamed to `config_name`.  

hash (`str`, *optional*) : Hash specific to the dataset builder code. Used to update the caching directory when the dataset builder code is updated (to avoid reusing old data). The typical caching directory (defined in `self._relative_data_dir`) is `name/version/hash/`.

base_path (`str`, *optional*) : Base path for relative paths that are used to download files. This can be a remote URL.

features ([Features](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Features), *optional*) : Features types to use with this dataset. It can be used to change the [Features](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Features) types of a dataset, for example.

token (`str` or `bool`, *optional*) : String or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, will get token from `"~/.huggingface"`.

repo_id (`str`, *optional*) : ID of the dataset repository. Used to distinguish builders with the same name but not coming from the same namespace, for example "rajpurkar/squad" and "lhoestq/squad" repo IDs. In the latter, the builder name would be "lhoestq___squad".

data_files (`str` or `Sequence` or `Mapping`, *optional*) : Path(s) to source data file(s). For builders like "csv" or "json" that need the user to specify data files. They can be either local or remote files. For convenience, you can use a `DataFilesDict`.

data_dir (`str`, *optional*) : Path to directory containing source data file(s). Use only if `data_files` is not passed, in which case it is equivalent to passing `os.path.join(data_dir, "**")` as `data_files`. For builders that require manual download, it must be the path to the local directory containing the manually downloaded data.

storage_options (`dict`, *optional*) : Key/value pairs to be passed on to the dataset file-system backend, if any.

writer_batch_size (`int`, *optional*) : Batch size used by the ArrowWriter. It defines the number of samples that are kept in memory before writing them and also the length of the arrow chunks. None means that the ArrowWriter will use its default value.

- ****config_kwargs** (additional keyword arguments) : Keyword arguments to be passed to the corresponding builder configuration class, set on the class attribute [DatasetBuilder.BUILDER_CONFIG_CLASS](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.BuilderConfig). The builder configuration class is [BuilderConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.BuilderConfig) or a subclass of it.

Abstract base class for all datasets.

`DatasetBuilder` has 3 key methods:

- `DatasetBuilder.info`: Documents the dataset, including feature
  names, types, shapes, version, splits, citation, etc.
- [DatasetBuilder.download_and_prepare()](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DatasetBuilder.download_and_prepare): Downloads the source data
  and writes it to disk.
- [DatasetBuilder.as_dataset()](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DatasetBuilder.as_dataset): Generates a [Dataset](/docs/datasets/v5.2.0/en/package_reference/main_classes#datasets.Dataset).

Some `DatasetBuilder`s expose multiple variants of the
dataset by defining a [BuilderConfig](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.BuilderConfig) subclass and accepting a
config object (or name) on construction. Configurable datasets expose a
pre-defined set of configurations in `DatasetBuilder.builder_configs()`.

#### as_dataset[[datasets.DatasetBuilder.as_dataset]]

```python
as_dataset(split: typing.Union[str, datasets.splits.Split, list[str], list[datasets.splits.Split], NoneType] = None, in_memory = False)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L991)

**Parameters:**

split (`datasets.Split`) : Which subset of the data to return.

verification_mode ([VerificationMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.VerificationMode) or `str`, defaults to `BASIC_CHECKS`) : Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).  

in_memory (`bool`, defaults to `False`) : Whether to copy the data in-memory.

**Returns:**

datasets.Dataset

Return a Dataset for the specified split.

Example:

```py
>>> from datasets import load_dataset_builder
>>> builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
>>> builder.download_and_prepare()
>>> ds = builder.as_dataset(split='train')
>>> ds
Dataset({
    features: ['text', 'label'],
    num_rows: 8530
})
```

#### download_and_prepare[[datasets.DatasetBuilder.download_and_prepare]]

```python
download_and_prepare(output_dir: typing.Optional[str] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: typing.Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, verification_mode: typing.Union[datasets.utils.info_utils.VerificationMode, str, NoneType] = None, dl_manager: typing.Optional[datasets.download.download_manager.DownloadManager] = None, base_path: typing.Optional[str] = None, file_format: str = 'arrow', max_shard_size: typing.Union[str, int, NoneType] = None, num_proc: typing.Optional[int] = None, storage_options: typing.Optional[dict] = None, **download_and_prepare_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L702)

**Parameters:**

output_dir (`str`, *optional*) : Output directory for the dataset. Default to this builder's `cache_dir`, which is inside `~/.cache/huggingface/datasets` by default.  

download_config (`DownloadConfig`, *optional*) : Specific download configuration parameters.

download_mode ([DownloadMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.DownloadMode) or `str`, *optional*) : Select the download/generate mode, default to `REUSE_DATASET_IF_EXISTS`.

verification_mode ([VerificationMode](/docs/datasets/v5.2.0/en/package_reference/builder_classes#datasets.VerificationMode) or `str`, defaults to `BASIC_CHECKS`) : Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).  

dl_manager (`DownloadManager`, *optional*) : Specific `DownloadManger` to use.

base_path (`str`, *optional*) : Base path for relative paths that are used to download files. This can be a remote url. If not specified, the value of the `base_path` attribute (`self.base_path`) will be used instead.

file_format (`str`, *optional*) : Format of the data files in which the dataset will be written. Supported formats: "arrow", "parquet". Default to "arrow" format. If the format is "parquet", then image and audio data are embedded into the Parquet files instead of pointing to local files.  

max_shard_size (`Union[str, int]`, *optional*) : Maximum number of bytes written per shard, default is "500MB". The size is based on uncompressed data size, so in practice your shard files may be smaller than `max_shard_size` thanks to Parquet compression for example.  

num_proc (`int`, *optional*, defaults to `None`) : Number of processes when downloading and generating the dataset locally. Multiprocessing is disabled by default.  

storage_options (`dict`, *optional*) : Key/value pairs to be passed on to the caching file-system backend, if any.  

- ****download_and_prepare_kwargs** (additional keyword arguments) : Keyword arguments.

Downloads and prepares dataset for reading.

Example:

Download and prepare the dataset as Arrow files that can be loaded as a Dataset using `builder.as_dataset()`:

```py
>>> from datasets import load_dataset_builder
>>> builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
>>> builder.download_and_prepare()
```

Download and prepare the dataset as sharded Parquet files locally:

```py
>>> from datasets import load_dataset_builder
>>> builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
>>> builder.download_and_prepare("./output_dir", file_format="parquet")
```

Download and prepare the dataset as sharded Parquet files in a cloud storage:

```py
>>> from datasets import load_dataset_builder
>>> storage_options = {"key": aws_access_key_id, "secret": aws_secret_access_key}
>>> builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
>>> builder.download_and_prepare("s3://my-bucket/my_rotten_tomatoes", storage_options=storage_options, file_format="parquet")
```

#### get_imported_module_dir[[datasets.DatasetBuilder.get_imported_module_dir]]

```python
get_imported_module_dir()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L694)

Return the path of the module of this class or subclass.

#### datasets.GeneratorBasedBuilder[[datasets.GeneratorBasedBuilder]]

```python
datasets.GeneratorBasedBuilder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L1292)

Base class for datasets with data generation based on dict generators.

`GeneratorBasedBuilder` is a convenience class that abstracts away much
of the data writing and reading of `DatasetBuilder`. It expects subclasses to
implement generators of feature dictionaries across the dataset splits
(`_split_generators`). See the method docstrings for details.

#### datasets.ArrowBasedBuilder[[datasets.ArrowBasedBuilder]]

```python
datasets.ArrowBasedBuilder(cache_dir: typing.Optional[str] = None, dataset_name: typing.Optional[str] = None, config_name: typing.Optional[str] = None, hash: typing.Optional[str] = None, base_path: typing.Optional[str] = None, info: typing.Optional[datasets.info.DatasetInfo] = None, features: typing.Optional[datasets.features.features.Features] = None, token: typing.Union[bool, str, NoneType] = None, repo_id: typing.Optional[str] = None, data_files: typing.Union[str, list, dict, datasets.data_files.DataFilesDict, NoneType] = None, data_dir: typing.Optional[str] = None, storage_options: typing.Optional[dict] = None, writer_batch_size: typing.Optional[int] = None, config_id: typing.Optional[str] = None, **config_kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L1610)

Base class for datasets with data generation based on Arrow loading functions (CSV/JSON/Parquet).

#### datasets.BuilderConfig[[datasets.BuilderConfig]]

```python
datasets.BuilderConfig(name: str = 'default', version: typing.Union[datasets.utils.version.Version, str, NoneType] = 0.0.0, data_dir: typing.Optional[str] = None, data_files: typing.Union[datasets.data_files.DataFilesDict, datasets.data_files.DataFilesPatternsDict, NoneType] = None, description: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L101)

**Parameters:**

name (`str`, defaults to `default`) : The name of the configuration.

version (`Version` or `str`, defaults to `0.0.0`) : The version of the configuration.

data_dir (`str`, *optional*) : Path to the directory containing the source data.

data_files (`str` or `Sequence` or `Mapping`, *optional*) : Path(s) to source data file(s).

description (`str`, *optional*) : A human description of the configuration.

Base class for `DatasetBuilder` data configuration.

`DatasetBuilder` subclasses with data configuration options should subclass
`BuilderConfig` and add their own properties.

#### create_config_id[[datasets.BuilderConfig.create_config_id]]

```python
create_config_id(config_kwargs: dict, custom_features: typing.Optional[datasets.features.features.Features] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/builder.py#L144)

The config id is used to build the cache directory.
By default it is equal to the config name.
However the name of a config is not sufficient to have a unique identifier for the dataset being generated
since it doesn't take into account:
- the config kwargs that can be used to overwrite attributes
- the custom features used to write the dataset
- the data_files for json/text/csv/pandas datasets

Therefore the config id is just the config name with an optional suffix based on these.

## Download[[datasets.DownloadManager]]

#### datasets.DownloadManager[[datasets.DownloadManager]]

```python
datasets.DownloadManager(dataset_name: typing.Optional[str] = None, data_dir: typing.Optional[str] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, base_path: typing.Optional[str] = None, record_checksums = False)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L71)

#### download[[datasets.DownloadManager.download]]

```python
download(url_or_urls)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L131)

**Parameters:**

url_or_urls (`str` or `list` or `dict`) : URL or `list` or `dict` of URLs to download. Each URL is a `str`.

**Returns:** `str` or `list` or `dict`

The downloaded paths matching the given input `url_or_urls`.

Download given URL(s).

By default, only one process is used for download. Pass customized `download_config.num_proc` to change this behavior.

Example:

```py
>>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
```

#### download_and_extract[[datasets.DownloadManager.download_and_extract]]

```python
download_and_extract(url_or_urls)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L310)

**Parameters:**

url_or_urls (`str` or `list` or `dict`) : URL or `list` or `dict` of URLs to download and extract. Each URL is a `str`.

**Returns:** `extracted_path(s)`

`str`, extracted paths of given URL(s).

Download and extract given `url_or_urls`.

Is roughly equivalent to:

```
extracted_paths = dl_manager.extract(dl_manager.download(url_or_urls))
```

#### extract[[datasets.DownloadManager.extract]]

```python
extract(path_or_paths)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L278)

**Parameters:**

path_or_paths (path or `list` or `dict`) : Path of file to extract. Each path is a `str`.

**Returns:** `extracted_path(s)`

`str`, The extracted paths matching the given input
path_or_paths.

Extract given path(s).

Example:

```py
>>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
>>> extracted_files = dl_manager.extract(downloaded_files)
```

#### iter_archive[[datasets.DownloadManager.iter_archive]]

```python
iter_archive(path_or_buf: typing.Union[str, _io.BufferedReader])
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L234)

**Parameters:**

path_or_buf (`str` or `io.BufferedReader`) : Archive path or archive binary file object.

**Yields:** `tuple[str, io.BufferedReader]`

2-tuple (path_within_archive, file_object).
File object is opened in binary mode.

Iterate over files within an archive.

Example:

```py
>>> archive = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
>>> files = dl_manager.iter_archive(archive)
```

#### iter_files[[datasets.DownloadManager.iter_files]]

```python
iter_files(paths: typing.Union[str, list[str]])
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L259)

**Parameters:**

paths (`str` or `list` of `str`) : Root paths.

**Yields:** `str`

File path.

Iterate over file paths.

Example:

```py
>>> files = dl_manager.download_and_extract('https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip')
>>> files = dl_manager.iter_files(files)
```

#### datasets.StreamingDownloadManager[[datasets.StreamingDownloadManager]]

```python
datasets.StreamingDownloadManager(dataset_name: typing.Optional[str] = None, data_dir: typing.Optional[str] = None, download_config: typing.Optional[datasets.download.download_config.DownloadConfig] = None, base_path: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/streaming_download_manager.py#L47)

Download manager that uses the "::" separator to navigate through (possibly remote) compressed archives.
Contrary to the regular `DownloadManager`, the `download` and `extract` methods don't actually download nor extract
data, but they rather return the path or url that could be opened using the `xopen` function which extends the
built-in `open` function to stream data from remote files.

#### download[[datasets.StreamingDownloadManager.download]]

```python
download(url_or_urls)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/streaming_download_manager.py#L75)

**Parameters:**

url_or_urls (`str` or `list` or `dict`) : URL(s) of files to stream data from. Each url is a `str`.

**Returns:** `url(s)`

(`str` or `list` or `dict`), URL(s) to stream data from matching the given input url_or_urls.

Normalize URL(s) of files to stream data from.
This is the lazy version of `DownloadManager.download` for streaming.

Example:

```py
>>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
```

#### download_and_extract[[datasets.StreamingDownloadManager.download_and_extract]]

```python
download_and_extract(url_or_urls)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/streaming_download_manager.py#L151)

**Parameters:**

url_or_urls (`str` or `list` or `dict`) : URL(s) to stream from data from. Each url is a `str`.

**Returns:** `url(s)`

(`str` or `list` or `dict`), URL(s) to stream data from matching the given input `url_or_urls`.

Prepare given `url_or_urls` for streaming (add extraction protocol).

This is the lazy version of `DownloadManager.download_and_extract` for streaming.

Is equivalent to:

```
urls = dl_manager.extract(dl_manager.download(url_or_urls))
```

#### extract[[datasets.StreamingDownloadManager.extract]]

```python
extract(url_or_urls)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/streaming_download_manager.py#L102)

**Parameters:**

url_or_urls (`str` or `list` or `dict`) : URL(s) of files to stream data from. Each url is a `str`.

**Returns:** `url(s)`

(`str` or `list` or `dict`), URL(s) to stream data from matching the given input `url_or_urls`.

Add extraction protocol for given url(s) for streaming.

This is the lazy version of `DownloadManager.extract` for streaming.

Example:

```py
>>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
>>> extracted_files = dl_manager.extract(downloaded_files)
```

#### iter_archive[[datasets.StreamingDownloadManager.iter_archive]]

```python
iter_archive(urlpath_or_buf: typing.Union[str, _io.BufferedReader])
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/streaming_download_manager.py#L171)

**Parameters:**

urlpath_or_buf (`str` or `io.BufferedReader`) : Archive path or archive binary file object.

**Yields:** `tuple[str, io.BufferedReader]`

2-tuple (path_within_archive, file_object).
File object is opened in binary mode.

Iterate over files within an archive.

Example:

```py
>>> archive = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
>>> files = dl_manager.iter_archive(archive)
```

#### iter_files[[datasets.StreamingDownloadManager.iter_files]]

```python
iter_files(urlpaths: typing.Union[str, list[str]])
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/streaming_download_manager.py#L196)

**Parameters:**

urlpaths (`str` or `list` of `str`) : Root paths.

**Yields:** `str`

File URL path.

Iterate over files.

Example:

```py
>>> files = dl_manager.download_and_extract('https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip')
>>> files = dl_manager.iter_files(files)
```

#### datasets.DownloadConfig[[datasets.DownloadConfig]]

```python
datasets.DownloadConfig(cache_dir: typing.Union[str, pathlib.Path, NoneType] = None, force_download: bool = False, resume_download: bool = False, local_files_only: bool = False, proxies: typing.Optional[dict] = None, user_agent: typing.Optional[str] = None, extract_compressed_file: bool = False, force_extract: bool = False, delete_extracted: bool = False, extract_on_the_fly: bool = False, use_etag: bool = True, num_proc: typing.Optional[int] = None, max_retries: int = 1, token: typing.Union[str, bool, NoneType] = None, storage_options: dict = <factory>, download_desc: typing.Optional[str] = None, disable_tqdm: bool = False)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_config.py#L10)

**Parameters:**

cache_dir (`str` or `Path`, *optional*) : Specify a cache directory to save the file to (overwrite the default cache dir).

force_download (`bool`, defaults to `False`) : If `True`, re-download the file even if it's already cached in the cache dir.

resume_download (`bool`, defaults to `False`) : If `True`, resume the download if an incompletely received file is found.

proxies (`dict`, *optional*) : Deprecated and ignored. Set the `HTTP_PROXY` / `HTTPS_PROXY` environment variables or configure the HTTP client with [huggingface_hub.set_client_factory](https://huggingface.co/docs/huggingface_hub/v2.4.0/en/package_reference/utilities#huggingface_hub.set_client_factory) instead.

user_agent (`str`, *optional*) : Optional string or dict that will be appended to the user-agent on remote requests.

extract_compressed_file (`bool`, defaults to `False`) : If `True` and the path point to a zip or tar file, extract the compressed file in a folder along the archive.

force_extract (`bool`, defaults to `False`) : If `True` when `extract_compressed_file` is `True` and the archive was already extracted, re-extract the archive and override the folder where it was extracted.

delete_extracted (`bool`, defaults to `False`) : Whether to delete (or keep) the extracted files.

extract_on_the_fly (`bool`, defaults to `False`) : If `True`, extract compressed files while they are being read.

use_etag (`bool`, defaults to `True`) : Whether to use the ETag HTTP response header to validate the cached files.

num_proc (`int`, *optional*) : The number of processes to launch to download the files in parallel.

max_retries (`int`, default to `1`) : The number of times to retry an HTTP request if it fails.

token (`str` or `bool`, *optional*) : Optional string or boolean to use as Bearer token for remote files on the Datasets Hub. If `True`, or not specified, will get token from `~/.huggingface`.

storage_options (`dict`, *optional*) : Key/value pairs to be passed on to the dataset file-system backend, if any.

download_desc (`str`, *optional*) : A description to be displayed alongside with the progress bar while downloading the files.

disable_tqdm (`bool`, defaults to `False`) : Whether to disable the individual files download progress bar

Configuration for our cached path manager.

#### datasets.DownloadMode[[datasets.DownloadMode]]

```python
datasets.DownloadMode(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/download/download_manager.py#L50)

`Enum` for how to treat pre-existing downloads and data.

The default mode is `REUSE_DATASET_IF_EXISTS`, which will reuse both
raw downloads and the prepared dataset if they exist.

The generations modes:

|                                     | Downloads | Dataset |
|-------------------------------------|-----------|---------|
| `REUSE_DATASET_IF_EXISTS` (default) | Reuse     | Reuse   |
| `REUSE_CACHE_IF_EXISTS`             | Reuse     | Fresh   |
| `FORCE_REDOWNLOAD`                  | Fresh     | Fresh   |

## Verification[[datasets.VerificationMode]]

#### datasets.VerificationMode[[datasets.VerificationMode]]

```python
datasets.VerificationMode(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/utils/info_utils.py#L22)

`Enum` that specifies which verification checks to run.

The default mode is `BASIC_CHECKS`, which will perform only rudimentary checks to avoid slowdowns
when generating/downloading a dataset for the first time.

The verification modes:

|                           | Verification checks                                                           |
|---------------------------|------------------------------------------------------------------------------ |
| `ALL_CHECKS`              | Split checks and validity (number of files, checksums) of downloaded files    |
| `BASIC_CHECKS` (default)  | Same as `ALL_CHECKS` but without checking downloaded files                    |
| `NO_CHECKS`               | None                                                                          |

## Splits[[datasets.SplitGenerator]]

#### datasets.SplitGenerator[[datasets.SplitGenerator]]

```python
datasets.SplitGenerator(name: str, gen_kwargs: dict = <factory>)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/splits.py#L624)

**Parameters:**

name (`str`) : Name of the `Split` for which the generator will create the examples.

- ****gen_kwargs** (additional keyword arguments) : Keyword arguments to forward to the `DatasetBuilder._generate_examples` method of the builder.

Defines the split information for the generator.

This should be used as returned value of
`GeneratorBasedBuilder._split_generators`.
See `GeneratorBasedBuilder._split_generators` for more info and example
of usage.

Example:

```py
>>> datasets.SplitGenerator(
...     name=datasets.Split.TRAIN,
...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and_extract(url)},
... )
```

#### datasets.Split[[datasets.Split]]

```python
datasets.Split(name)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/splits.py#L417)

`Enum` for dataset splits.

Datasets are typically split into different subsets to be used at various
stages of training and evaluation.

- `TRAIN`: the training data.
- `VALIDATION`: the validation data. If present, this is typically used as
  evaluation data while iterating on a model (e.g. changing hyperparameters,
  model architecture, etc.).
- `TEST`: the testing data. This is the data to report metrics on. Typically
  you do not want to use this during model iteration as you may overfit to it.
- `ALL`: the union of all defined dataset splits.

All splits, including compositions inherit from `datasets.SplitBase`.

See the [guide](../load_hub#splits) on splits for more information.

Example:

```py
>>> datasets.SplitGenerator(
...     name=datasets.Split.TRAIN,
...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and extract(url)},
... ),
... datasets.SplitGenerator(
...     name=datasets.Split.VALIDATION,
...     gen_kwargs={"split_key": "validation", "files": dl_manager.download_and extract(url)},
... ),
... datasets.SplitGenerator(
...     name=datasets.Split.TEST,
...     gen_kwargs={"split_key": "test", "files": dl_manager.download_and extract(url)},
... )
```

#### datasets.NamedSplit[[datasets.NamedSplit]]

```python
datasets.NamedSplit(name: str)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/splits.py#L325)

Descriptor corresponding to a named split (train, test, ...).

Example:

Each descriptor can be composed with other using addition or slice:

```py
split = datasets.Split.TRAIN.subsplit(datasets.percent[0:25]) + datasets.Split.TEST
```

The resulting split will correspond to 25% of the train split merged with
100% of the test split.

A split cannot be added twice, so the following will fail:

```py
split = (
        datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
        datasets.Split.TRAIN.subsplit(datasets.percent[75:])
)  # Error
split = datasets.Split.TEST + datasets.Split.ALL  # Error
```

The slices can be applied only one time. So the following are valid:

```py
split = (
        datasets.Split.TRAIN.subsplit(datasets.percent[:25]) +
        datasets.Split.TEST.subsplit(datasets.percent[:50])
)
split = (datasets.Split.TRAIN + datasets.Split.TEST).subsplit(datasets.percent[:50])
```

But this is not valid:

```py
train = datasets.Split.TRAIN
test = datasets.Split.TEST
split = train.subsplit(datasets.percent[:25]).subsplit(datasets.percent[:25])
split = (train.subsplit(datasets.percent[:25]) + test).subsplit(datasets.percent[:50])
```

#### datasets.NamedSplitAll[[datasets.NamedSplitAll]]

```python
datasets.NamedSplitAll()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/splits.py#L402)

Split corresponding to the union of all defined dataset splits.

#### datasets.ReadInstruction[[datasets.ReadInstruction]]

```python
datasets.ReadInstruction(split_name, rounding = None, from_ = None, to = None, unit = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/arrow_reader.py#L456)

Reading instruction for a dataset.

Examples:

```python
# The following lines are equivalent:
ds = datasets.load_dataset('ylecun/mnist', split='test[:33%]')
ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction.from_spec('test[:33%]'))
ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction('test', to=33, unit='%'))
ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction(
'test', from_=0, to=33, unit='%'))

# The following lines are equivalent:
ds = datasets.load_dataset('ylecun/mnist', split='test[:33%]+train[1:-1]')
ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction.from_spec(
'test[:33%]+train[1:-1]'))
ds = datasets.load_dataset('ylecun/mnist', split=(
datasets.ReadInstruction('test', to=33, unit='%') +
datasets.ReadInstruction('train', from_=1, to=-1, unit='abs')))

# The following lines are equivalent:
ds = datasets.load_dataset('ylecun/mnist', split='test[:33%](pct1_dropremainder)')
ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction.from_spec(
'test[:33%](pct1_dropremainder)'))
ds = datasets.load_dataset('ylecun/mnist', split=datasets.ReadInstruction(
'test', from_=0, to=33, unit='%', rounding="pct1_dropremainder"))

# 10-fold validation:
tests = datasets.load_dataset(
'ylecun/mnist',
[datasets.ReadInstruction('train', from_=k, to=k+10, unit='%')
for k in range(0, 100, 10)])
trains = datasets.load_dataset(
'ylecun/mnist',
[datasets.ReadInstruction('train', to=k, unit='%') + datasets.ReadInstruction('train', from_=k+10, unit='%')
for k in range(0, 100, 10)])
```

#### from_spec[[datasets.ReadInstruction.from_spec]]

```python
from_spec(spec)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/arrow_reader.py#L536)

**Parameters:**

spec (`str`) : Split(s) + optional slice(s) to read + optional rounding if percents are used as the slicing unit. A slice can be specified, using absolute numbers (`int`) or percentages (`int`).

**Returns:**

ReadInstruction instance.

Creates a `ReadInstruction` instance out of a string spec.

Examples:

```
test: test split.
test + validation: test split + validation split.
test[10:]: test split, minus its first 10 records.
test[:10%]: first 10% records of test split.
test[:20%](pct1_dropremainder): first 10% records, rounded with the pct1_dropremainder rounding.
test[:-5%]+train[40%:60%]: first 95% of test + middle 20% of train.
```

#### to_absolute[[datasets.ReadInstruction.to_absolute]]

```python
to_absolute(name2len)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/arrow_reader.py#L608)

**Parameters:**

name2len (`dict`) : Associating split names to number of examples.

**Returns:**

list of _AbsoluteInstruction instances (corresponds to the + in spec).

Translate instruction into a list of absolute instructions.

Those absolute instructions are then to be added together.

## Version[[datasets.Version]]

#### datasets.Version[[datasets.Version]]

```python
datasets.Version(version_str: str, description: typing.Optional[str] = None, major: typing.Union[str, int, NoneType] = None, minor: typing.Union[str, int, NoneType] = None, patch: typing.Union[str, int, NoneType] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/utils/version.py#L30)

**Parameters:**

version_str (`str`) : The dataset version.

description (`str`) : A description of what is new in this version.

major (`str`) --

minor (`str`) --

patch (`str`) --

Dataset version `MAJOR.MINOR.PATCH`.

Example:

```py
>>> VERSION = datasets.Version("1.0.0")
```

### Table Classes
https://huggingface.co/docs/datasets/v5.2.0/package_reference/table_classes.md
