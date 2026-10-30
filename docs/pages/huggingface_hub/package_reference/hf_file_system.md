# Filesystem API

The `HfFileSystem` class provides a pythonic file interface to the Hugging Face Hub based on [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/).

## HfFileSystem[[huggingface_hub.HfFileSystem]]

`HfFileSystem` is based on [fsspec](https://filesystem-spec.readthedocs.io/en/latest/), so it is compatible with most of the APIs that it offers. For more details, check out [our guide](../guides/hf_file_system) and fsspec's [API Reference](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem).

#### huggingface_hub.HfFileSystem[[huggingface_hub.HfFileSystem]]

```python
huggingface_hub.HfFileSystem(*args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L370)

**Parameters:**

endpoint (`str`, *optional*) : Endpoint of the Hub. Defaults to .

token (`bool` or `str`, *optional*) : A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see https://huggingface.co/docs/huggingface_hub/quick-start#authentication). To disable authentication, pass `False`.

block_size (`int`, *optional*) : Block size for reading and writing files.

expand_info (`bool`, *optional*) : Whether to expand the information of the files.

max_workers (`int`, *optional*) : Maximum number of concurrent requests used to list large directories in parallel (see [find()](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem.find)). Defaults to 16. Set it to 1 to disable parallel listings.

live_follow (`bool`, *optional*) : Whether to follow file changes in buckets as they happen on the Hub, so that cached listings stay fresh without manual [HfFileSystem.invalidate_cache()](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem.invalidate_cache) calls. When enabled, listing a bucket opens a background connection to the Hub event feed for that bucket. Has no effect on model, dataset and space repositories (the Hub does not feed changes for those).

- ****storage_options** (`dict`, *optional*) : Additional options for the filesystem. See [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.__init__).

Access a remote Hugging Face Hub repository as if were a local file system.

> [!WARNING]
> [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) provides fsspec compatibility, which is useful for libraries that require it (e.g., reading
>     Hugging Face datasets directly with `pandas`). However, it introduces additional overhead due to this compatibility
>     layer. For better performance and reliability, it's recommended to use `HfApi` methods when possible.

The file system supports paths for the `hf://` protocol, which follows those URL schemes:

* Models, Datasets and Spaces repositories:

```
hf://<repo-id>[@<revision>]/<path/in/repo>
hf://datasets/<repo-id>[@<revision>]/<path/in/repo>
hf://spaces/<repo-id>[@<revision>]/<path/in/repo>
```

* Buckets (generic storage):

```
hf://buckets/<bucket-id>/<path/in/bucket>
```

Note: when using the [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) directly, passing the `hf://` protocol prefix is optional in paths.

Usage:

```python
>>> from huggingface_hub import hffs

>>> # List files
>>> hffs.glob("my-username/my-model/*.bin")
['my-username/my-model/pytorch_model.bin']
>>> hffs.ls("datasets/my-username/my-dataset", detail=False)
['datasets/my-username/my-dataset/.gitattributes', 'datasets/my-username/my-dataset/README.md', 'datasets/my-username/my-dataset/data.json']

>>> # Read/write files
>>> with hffs.open("my-username/my-model/pytorch_model.bin") as f:
...     data = f.read()
>>> with hffs.open("my-username/my-model/pytorch_model.bin", "wb") as f:
...     f.write(data)
```

Specify a token for authentication:
```python
>>> from huggingface_hub import HfFileSystem
>>> hffs = HfFileSystem(token=token)
```

#### cp_file[[huggingface_hub.HfFileSystem.cp_file]]

```python
cp_file(path1: str, path2: str, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1115)

**Parameters:**

path1 (`str`) : Source path to copy from.

path2 (`str`) : Destination path to copy to.

revision (`str`, *optional*) : The git revision to copy from.

Copy a file within or between repositories, or into a bucket.

> [!WARNING]
> Note: When possible, use `HfApi.upload_file()` for better performance.

#### exists[[huggingface_hub.HfFileSystem.exists]]

```python
exists(path, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1336)

**Parameters:**

path (`str`) : Path to check.

**Returns:** `bool`

True if file exists, False otherwise.

Check if a file exists.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.exists).

> [!WARNING]
> Note: When possible, use `HfApi.file_exists()` for better performance.

#### find[[huggingface_hub.HfFileSystem.find]]

```python
find(path: str, maxdepth: int | None = None, withdirs: bool = False, detail: bool = False, refresh: bool = False, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1051)

**Parameters:**

path (`str`) : Root path to list files from.

maxdepth (`int`, *optional*) : Maximum depth to descend into subdirectories.

withdirs (`bool`, *optional*) : Include directory paths in the output. Defaults to False.

detail (`bool`, *optional*) : If True, returns a dict mapping paths to file information. Defaults to False.

refresh (`bool`, *optional*) : If True, bypass the cache and fetch the latest data. Defaults to False.

revision (`str`, *optional*) : The git revision to list from.

**Returns:** `Union[list[str], dict[str, dict[str, Any]]]`

List of paths or dict of file information.

List all files below path.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.find).

> [!TIP]
> Large repositories are listed directory by directory, in parallel, to work around the
> cursor-based pagination of the Hub `tree` endpoint. Tune this with the `max_workers`
> argument of [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem).

#### get_file[[huggingface_hub.HfFileSystem.get_file]]

```python
get_file(rpath, lpath = None, callback = <fsspec.callbacks.NoOpCallback object at 0x7f9673955c90>, outfile = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1444)

**Parameters:**

rpath (`str`) : Remote path to download from.

lpath (`str`, *optional*) : Local path to download to. Can be omitted if `outfile` is provided.

callback (`Callback`, *optional*) : Optional callback to track download progress. Defaults to no callback.

outfile (`IO`, *optional*) : Optional file-like object to write to. If provided, `lpath` is ignored.

Copy single remote file to local.

> [!WARNING]
> Note: When possible, use `HfApi.hf_hub_download()` or `HfApi.download_bucket_files` for better performance.

#### glob[[huggingface_hub.HfFileSystem.glob]]

```python
glob(path: str, maxdepth: int | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1033)

**Parameters:**

path (`str`) : Path pattern to match.

maxdepth (`int`, *optional*) : Maximum depth to descend into directories. By default, no limit.

**Returns:** `list[str]`

List of paths matching the pattern.

Find files by glob-matching.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.glob).

#### info[[huggingface_hub.HfFileSystem.info]]

```python
info(path: str, refresh: bool = False, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1226)

**Parameters:**

path (`str`) : Path to get info for.

refresh (`bool`, *optional*) : If True, bypass the cache and fetch the latest data. Defaults to False.

revision (`str`, *optional*) : The git revision to get info from.

**Returns:** `dict[str, Any]`

Dictionary containing file information (type, size, commit info, etc.).

Get information about a file or directory.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.info).

> [!WARNING]
> Note: When possible, use `HfApi.get_paths_info()` or `HfApi.repo_info()`  for better performance
> (or `HfApi.get_bucket_paths_info()` or `HfApi.bucket_info()` for buckets)

#### invalidate_cache[[huggingface_hub.HfFileSystem.invalidate_cache]]

```python
invalidate_cache(path: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L599)

**Parameters:**

path (`str`, *optional*) : Path to clear from cache. If not provided, clear the entire cache.

Clear the cache for a given path.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.invalidate_cache).

#### isdir[[huggingface_hub.HfFileSystem.isdir]]

```python
isdir(path)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1361)

**Parameters:**

path (`str`) : Path to check.

**Returns:** `bool`

True if path is a directory, False otherwise.

Check if a path is a directory.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.isdir).

#### isfile[[huggingface_hub.HfFileSystem.isfile]]

```python
isfile(path)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1379)

**Parameters:**

path (`str`) : Path to check.

**Returns:** `bool`

True if path is a file, False otherwise.

Check if a path is a file.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.isfile).

#### ls[[huggingface_hub.HfFileSystem.ls]]

```python
ls(path: str, detail: bool = True, refresh: bool = False, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L788)

**Parameters:**

path (`str`) : Path to the directory.

detail (`bool`, *optional*) : If True, returns a list of dictionaries containing file information. If False, returns a list of file paths. Defaults to True.

refresh (`bool`, *optional*) : If True, bypass the cache and fetch the latest data. Defaults to False.

revision (`str`, *optional*) : The git revision to list from.

**Returns:** `list[Union[str, dict[str, Any]]]`

List of file paths (if detail=False) or list of file information
dictionaries (if detail=True).

List the contents of a directory.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.ls).

> [!WARNING]
> Note: When possible, use `HfApi.list_repo_tree()` for better performance.

#### modified[[huggingface_hub.HfFileSystem.modified]]

```python
modified(path: str, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1201)

**Parameters:**

path (`str`) : Path to the file.

**Returns:** `datetime`

Last modified time of the file.

Get the last modified time of a file.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.modified).

#### resolve_path[[huggingface_hub.HfFileSystem.resolve_path]]

```python
resolve_path(path: str, revision: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L520)

**Parameters:**

path (`str`) : Path to resolve.

revision (`str`, *optional*) : The revision of the repo to resolve. Defaults to the revision specified in the path.

**Returns:** `HfFileSystemResolvedPath`

Resolved path information containing `repo_type`, `repo_id`, `revision` and `path_in_repo`.

**Raises:** ``ValueError`` or ``NotImplementedError``

- ``ValueError`` -- 
  If path contains conflicting revision information.
- ``NotImplementedError`` -- 
  If trying to list repositories.

Resolve a Hugging Face file system path into its components.

#### rm[[huggingface_hub.HfFileSystem.rm]]

```python
rm(path: str, recursive: bool = False, maxdepth: int | None = None, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L738)

**Parameters:**

path (`str`) : Path to delete.

recursive (`bool`, *optional*) : If True, delete directory and all its contents. Defaults to False.

maxdepth (`int`, *optional*) : Maximum number of subdirectories to visit when deleting recursively.

revision (`str`, *optional*) : The git revision to delete from.

Delete files from a repository.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm).

> [!WARNING]
> Note: When possible, use `HfApi.delete_file()` for better performance.

#### url[[huggingface_hub.HfFileSystem.url]]

```python
url(path: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1397)

**Parameters:**

path (`str`) : Path to get URL for.

**Returns:** `str`

HTTP URL to access the file or directory on the Hub.

Get the HTTP URL of the given path.

#### walk[[huggingface_hub.HfFileSystem.walk]]

```python
walk(path: str, *args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v2.4.0/src/huggingface_hub/hf_file_system.py#L1017)

**Parameters:**

path (`str`) : Root path to list files from.

**Returns:** `Iterator[tuple[str, list[str], list[str]]]`

An iterator of (path, list of directory names, list of file names) tuples.

Return all files below the given path.

For more details, refer to [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.walk).

### Jobs
https://huggingface.co/docs/huggingface_hub/v2.4.0/package_reference/jobs.md
