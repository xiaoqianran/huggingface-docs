# Datasets

`LeRobotDataset` is the format every LeRobot script reads and writes. It is episode-aware, decodes video
observations on the fly, and round-trips to the Hugging Face Hub.

See [Using LeRobotDataset](../lerobot-dataset-v3) for the format and the common operations,
[Porting Large Datasets](../porting_datasets_v3) for migration, and [Tools](../tools) for the CLI.

## LeRobotDataset[[lerobot.datasets.LeRobotDataset]]

#### lerobot.datasets.LeRobotDataset[[lerobot.datasets.LeRobotDataset]]

```python
lerobot.datasets.LeRobotDataset(repo_id: str, root: str | pathlib.Path | None = None, episodes: list[int] | None = None, episode_filter: collections.abc.Callable[[dict], bool] | None = None, image_transforms: collections.abc.Callable | None = None, delta_timestamps: dict[str, list[float]] | None = None, tolerance_s: float = 0.0001, revision: str | None = None, force_cache_sync: bool = False, download_videos: bool = True, video_backend: str | None = None, return_uint8: bool = False, depth_output_unit: str = 'mm', batch_encoding_size: int = 1, rgb_encoder: lerobot.configs.video.RGBEncoderConfig | None = None, depth_encoder: lerobot.configs.video.DepthEncoderConfig | None = None, encoder_threads: int | None = None, streaming_encoding: bool = False, encoder_queue_maxsize: int = 30, repo_type: str = 'dataset', token: str | bool | None = None, video_decoder_cache_size: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L52)

#### add_frame[[lerobot.datasets.LeRobotDataset.add_frame]]

```python
add_frame(frame: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L500)

**Parameters:**

frame : Dict mapping feature names to their values for this frame. Must include a `'task'` key. Torch tensors are converted to numpy.

**Raises:** `RuntimeError`

- `RuntimeError` -- If the dataset is read-only (no writer).

Add a single frame to the current episode buffer.

Delegates to `DatasetWriter.add_frame`. The dataset must be in
write mode (created via `create` or `resume`).

#### clear_episode_buffer[[lerobot.datasets.LeRobotDataset.clear_episode_buffer]]

```python
clear_episode_buffer(delete_images: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L532)

**Parameters:**

delete_images : If `True`, also remove temporary image files written to disk for the current episode.

**Raises:** `RuntimeError`

- `RuntimeError` -- If the dataset is read-only (no writer).

Discard the current episode buffer without saving.

Delegates to `DatasetWriter.clear_episode_buffer`. Useful for
discarding a failed or interrupted recording episode.

#### clear_image_transforms[[lerobot.datasets.LeRobotDataset.clear_image_transforms]]

```python
clear_image_transforms()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L639)

Remove the transform applied to visual observations.

#### create[[lerobot.datasets.LeRobotDataset.create]]

```python
create(repo_id: str, fps: int, features: dict, root: str | pathlib.Path | None = None, robot_type: str | None = None, use_videos: bool = True, tolerance_s: float = 0.0001, image_writer_processes: int = 0, image_writer_threads: int = 0, video_backend: str | None = None, batch_encoding_size: int = 1, rgb_encoder: lerobot.configs.video.RGBEncoderConfig | None = None, depth_encoder: lerobot.configs.video.DepthEncoderConfig | None = None, metadata_buffer_size: int = 10, streaming_encoding: bool = False, encoder_queue_maxsize: int = 30, encoder_threads: int | None = None, video_files_size_in_mb: int | None = None, data_files_size_in_mb: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L766)

**Parameters:**

repo_id : Repository identifier, typically `'&amp;lcub;hf_user}/&amp;lcub;dataset_name}'`.

fps : Frames per second used during data collection.

features : Feature specification dict mapping feature names to their type/shape metadata.

root : Local directory for dataset storage. Defaults to `$HF_LEROBOT_HOME/&amp;lcub;repo_id}`.

robot_type : Optional robot type string stored in metadata.

use_videos : If `True`, visual modalities are stored as MP4 videos. If `False`, they are stored as images.

tolerance_s : Timestamp synchronization tolerance in seconds.

image_writer_processes : Number of subprocesses for async image writing. `0` means use threads only.

image_writer_threads : Number of threads for async image writing.

video_backend : Video decoding backend (used when reading back).

batch_encoding_size : Number of episodes to accumulate before batch-encoding videos. `1` means encode immediately.

rgb_encoder : Video encoder settings for cameras (codec, quality, etc.). When `None`, `rgb_encoder_defaults()` is used.

depth_encoder : Video encoder settings for depth cameras (codec, quality, etc.). When `None`, `depth_encoder_defaults()` is used.

encoder_threads : Number of encoder threads (global). `None` lets the codec decide.

metadata_buffer_size : Number of episode metadata records to buffer before flushing to parquet.

streaming_encoding : If `True`, encode video frames in real-time during capture instead of writing images first.

encoder_queue_maxsize : Max buffered frames per camera when using streaming encoding.

**Returns:**

A new `LeRobotDataset` in write mode.

Create a new LeRobotDataset from scratch for recording data.

Returns a write-mode dataset with an active `DatasetWriter`. Use
`add_frame` / `save_episode` to populate it, then
`finalize` when done.

#### finalize[[lerobot.datasets.LeRobotDataset.finalize]]

```python
finalize()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L554)

Flush all pending work and close writers.

Must be called after data collection/conversion, otherwise footer metadata
won't be written to the parquet files and the dataset will be invalid.

Idempotent — safe to call multiple times.  DatasetWriter.__del__ acts as a
safety net if this is never called explicitly.

#### get_raw_item[[lerobot.datasets.LeRobotDataset.get_raw_item]]

```python
get_raw_item(idx)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L615)

Get a raw frame without image transforms applied.

Unlike `__getitem__`, this returns the raw HF dataset row at the given
index with no delta-timestamp expansion, video decoding, or image transforms.

#### has_pending_frames[[lerobot.datasets.LeRobotDataset.has_pending_frames]]

```python
has_pending_frames()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L547)

Check if there are unsaved frames in the episode buffer.

#### push_to_hub[[lerobot.datasets.LeRobotDataset.push_to_hub]]

```python
push_to_hub(branch: str | None = None, tags: list | None = None, license: str | None = 'apache-2.0', tag_version: bool = True, push_videos: bool = True, private: bool | None = None, allow_patterns: list[str] | str | None = None, upload_large_folder: bool = False, **card_kwargs)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L647)

**Parameters:**

branch : Optional branch to push to. Created from the current revision if it does not exist.

tags : Optional list of tags for the dataset card.

license : License identifier for the dataset card.

tag_version : If `True`, create a Git tag for the current codebase version.

push_videos : If `False`, skip uploading the `videos/` directory.

private : If `True`, create a private repository. If `None` (default), defer to the org default on the Hub (only affects orgs).

allow_patterns : Glob pattern(s) restricting which files to upload.

upload_large_folder : Kept for backward compatibility and ignored. `upload_folder` handles datasets of all sizes.

- ****card_kwargs** : Additional keyword arguments forwarded to dataset card creation.

Upload the dataset to the Hugging Face Hub.

Creates the repository if it does not exist, uploads all dataset files
(optionally excluding videos), generates a dataset card, and tags the
revision with the current codebase version.

#### resume[[lerobot.datasets.LeRobotDataset.resume]]

```python
resume(repo_id: str, root: str | pathlib.Path | None = None, tolerance_s: float = 0.0001, revision: str | None = None, force_cache_sync: bool = False, video_backend: str | None = None, batch_encoding_size: int = 1, rgb_encoder: lerobot.configs.video.RGBEncoderConfig | None = None, depth_encoder: lerobot.configs.video.DepthEncoderConfig | None = None, encoder_threads: int | None = None, image_writer_processes: int = 0, image_writer_threads: int = 0, streaming_encoding: bool = False, encoder_queue_maxsize: int = 30, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L880)

**Parameters:**

repo_id : Repository identifier of the existing dataset.

root : Local directory of the dataset. When provided, Hub downloads are materialized directly into this directory. When omitted, Hub downloads use a revision-safe snapshot cache under `$HF_LEROBOT_HOME/hub`.

tolerance_s : Timestamp synchronization tolerance in seconds.

revision : Git revision (branch, tag, or commit hash). Defaults to current codebase version tag.

force_cache_sync : If `True`, re-download metadata from the Hub even if a local cache exists.

video_backend : Video decoding backend for reading back data.

batch_encoding_size : Number of episodes to accumulate before batch-encoding videos.

rgb_encoder : Video encoder settings for cameras (codec, quality, etc.). When `None`, `rgb_encoder_defaults()` is used.

depth_encoder : Video encoder settings for depth cameras (codec, quality, etc.). When `None`, `depth_encoder_defaults()` is used.

encoder_threads : Number of encoder threads (global). `None` lets the codec decide.

image_writer_processes : Subprocesses for async image writing.

image_writer_threads : Threads for async image writing.

streaming_encoding : If `True`, encode video in real-time during capture.

encoder_queue_maxsize : Max buffered frames per camera for streaming.

token : Authentication token used if metadata must be downloaded from the Hub. The token is not retained on the dataset instance.

**Returns:**

A `LeRobotDataset` in write mode, ready to append episodes.

Resume recording on an existing dataset.

Loads metadata from an existing dataset (local or Hub) and creates a
`DatasetWriter` for appending new episodes. The underlying HF
dataset is not loaded until `finalize` is called and data is
subsequently read.

#### save_episode[[lerobot.datasets.LeRobotDataset.save_episode]]

```python
save_episode(episode_data: dict | None = None, parallel_encoding: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L515)

**Parameters:**

episode_data : Optional pre-built episode dict. If `None`, uses the internal episode buffer populated by `add_frame`.

parallel_encoding : If `True` and multiple cameras exist, encode videos in parallel using a process pool.

**Raises:** `RuntimeError`

- `RuntimeError` -- If the dataset is read-only (no writer).

Save the current episode buffer to disk.

Delegates to `DatasetWriter.save_episode`. Encodes videos, writes
parquet data, and updates metadata. The episode buffer is reset afterward.

#### select_columns[[lerobot.datasets.LeRobotDataset.select_columns]]

```python
select_columns(column_names: str | list[str])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L607)

Select specific columns from the underlying dataset.

Useful for extracting action sequences during replay without loading all features.
Returns a `datasets.Dataset` containing only the requested columns.

#### set_image_transforms[[lerobot.datasets.LeRobotDataset.set_image_transforms]]

```python
set_image_transforms(image_transforms: collections.abc.Callable | None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L634)

Replace the transform applied to visual observations.

## LeRobotDatasetMetadata[[lerobot.datasets.LeRobotDatasetMetadata]]

#### lerobot.datasets.LeRobotDatasetMetadata[[lerobot.datasets.LeRobotDatasetMetadata]]

```python
lerobot.datasets.LeRobotDatasetMetadata(repo_id: str, root: str | pathlib.Path | None = None, revision: str | None = None, force_cache_sync: bool = False, metadata_buffer_size: int = 10, repo_type: typing.Literal['dataset', 'bucket'] = 'dataset', token: str | bool | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L67)

Metadata container for a LeRobot dataset.

Manages the `info.json`, `stats.json`, `tasks.parquet`, and
`episodes/` parquet files that describe a dataset's structure, content,
and statistics.

#### create[[lerobot.datasets.LeRobotDatasetMetadata.create]]

```python
create(repo_id: str, fps: int, features: dict, robot_type: str | None = None, root: str | pathlib.Path | None = None, use_videos: bool = True, metadata_buffer_size: int = 10, chunks_size: int | None = None, data_files_size_in_mb: int | None = None, video_files_size_in_mb: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L797)

**Parameters:**

repo_id : Repository identifier (e.g. `'user/my_dataset'`).

fps : Frames per second used during data collection.

features : Feature specification dict mapping feature names to their type/shape metadata.

robot_type : Optional robot type string stored in metadata.

root : Local directory for the dataset. Defaults to `$HF_LEROBOT_HOME/&amp;lcub;repo_id}`. Must not already exist.

use_videos : If `True`, visual modalities are encoded as MP4 videos.

metadata_buffer_size : Number of episode metadata records to buffer before flushing to parquet.

chunks_size : Max number of files per chunk directory. `None` uses the default.

data_files_size_in_mb : Max parquet file size in MB. `None` uses the default.

video_files_size_in_mb : Max video file size in MB. `None` uses the default.

**Returns:**

A new `LeRobotDatasetMetadata` instance.

Create metadata for a new LeRobot dataset from scratch.

Initializes the `info.json` file on disk with the provided feature
schema and dataset settings. No episode data is written yet.

#### ensure_readable[[lerobot.datasets.LeRobotDatasetMetadata.ensure_readable]]

```python
ensure_readable()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L225)

Guarantee metadata is fully loaded for read operations.

Idempotent — when metadata is already in memory this is a single
`is None` check.  Call this before transitioning from write to
read mode on the same instance.

#### filter_episodes[[lerobot.datasets.LeRobotDatasetMetadata.filter_episodes]]

```python
filter_episodes(predicate: Callable, candidates: list[int] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L235)

**Parameters:**

predicate : Predicate over per-episode metadata rows used to select episodes.

candidates : Optional list of episode indices to restrict evaluation to.

**Returns:**

List of sorted episode indices that satisfy the predicate.

Filter episodes whose metadata satisfies a given predicate.

#### finalize[[lerobot.datasets.LeRobotDatasetMetadata.finalize]]

```python
finalize()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L202)

Flush metadata buffer and close the parquet writer.

Idempotent — safe to call multiple times.

#### get_chunk_settings[[lerobot.datasets.LeRobotDatasetMetadata.get_chunk_settings]]

```python
get_chunk_settings()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L774)

**Returns:**

Dict containing chunks_size, data_files_size_in_mb, and video_files_size_in_mb.

Get current chunk and file size settings.

#### get_data_file_path[[lerobot.datasets.LeRobotDatasetMetadata.get_data_file_path]]

```python
get_data_file_path(ep_index: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L312)

**Parameters:**

ep_index : Zero-based episode index.

**Returns:**

Path to the parquet file containing this episode's data.

**Raises:** `IndexError`

- `IndexError` -- If `ep_index` is out of range.

Return the relative parquet file path for the given episode index.

#### get_task_index[[lerobot.datasets.LeRobotDatasetMetadata.get_task_index]]

```python
get_task_index(task: str)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L533)

Given a task in natural language, returns its task_index if the task already exists in the dataset,
otherwise return None.

#### get_video_file_path[[lerobot.datasets.LeRobotDatasetMetadata.get_video_file_path]]

```python
get_video_file_path(ep_index: int, vid_key: str)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L336)

**Parameters:**

ep_index : Zero-based episode index.

vid_key : Feature key identifying the video stream (e.g. `'observation.images.laptop'`).

**Returns:**

Path to the video file containing this episode's frames.

**Raises:** `IndexError` or `ValueError`

- `IndexError` -- If `ep_index` is out of range.
- `ValueError` -- If the dataset stores no videos (no `video_path` template).

Return the relative video file path for the given episode and video key.

#### rescale_depth_stats[[lerobot.datasets.LeRobotDatasetMetadata.rescale_depth_stats]]

```python
rescale_depth_stats(output_unit: str)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L416)

Rescale depth feature stats in place from their recorded unit to `output_unit`.

Depth stats are stored in the unit the frames were recorded in
(`features[key]["info"]["depth_unit"]`), while frames are returned in
`output_unit` on read. This converts the unit-bearing stat entries so
stats match the frames consumers see.

#### save_episode[[lerobot.datasets.LeRobotDatasetMetadata.save_episode]]

```python
save_episode(episode_index: int, episode_length: int, episode_tasks: list, episode_stats: dict, episode_metadata: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L643)

**Parameters:**

episode_index : Zero-based index of the episode being saved.

episode_length : Number of frames in this episode.

episode_tasks : List of task descriptions for this episode.

episode_stats : Per-feature statistics for this episode.

episode_metadata : Additional metadata (chunk/file indices, frame ranges, video timestamps, etc.).

Persist episode metadata, update dataset info, and aggregate stats.

Writes the episode's metadata to the buffered parquet writer, increments
the total episode/frame counters in `info.json`, and merges the
episode's statistics into the running dataset statistics.

#### save_episode_tasks[[lerobot.datasets.LeRobotDatasetMetadata.save_episode_tasks]]

```python
save_episode_tasks(tasks: list)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L543)

**Parameters:**

tasks : List of unique task descriptions in natural language.

**Raises:** `ValueError`

- `ValueError` -- If `tasks` contains duplicates.

Register tasks for the current episode and persist to disk.

New tasks that do not already exist in the dataset are assigned
sequential task indices and appended to the tasks parquet file.

#### update_chunk_settings[[lerobot.datasets.LeRobotDatasetMetadata.update_chunk_settings]]

```python
update_chunk_settings(chunks_size: int | None = None, data_files_size_in_mb: int | None = None, video_files_size_in_mb: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L739)

**Parameters:**

chunks_size : Maximum number of files per chunk directory. If None, keeps current value.

data_files_size_in_mb : Maximum size for data parquet files in MB. If None, keeps current value.

video_files_size_in_mb : Maximum size for video files in MB. If None, keeps current value.

Update chunk and file size settings after dataset creation.

This allows users to customize storage organization without modifying the constructor.
These settings control how episodes are chunked and how large files can grow before
creating new ones.

#### update_video_info[[lerobot.datasets.LeRobotDatasetMetadata.update_video_info]]

```python
update_video_info(video_key: str | None = None, video_encoder: lerobot.configs.video.VideoEncoderConfig | None = None, preserve_keys: collections.abc.Iterable[str] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_metadata.py#L685)

**Parameters:**

video_key : If provided, only update this video key. Otherwise update all video keys in the dataset.

video_encoder : Encoder configuration used to produce the videos. When provided, its fields are recorded as `video.&amp;lt;field>` entries alongside the stream-derived `video.*` entries (see `get_video_info`).

preserve_keys : Keys whose existing values are kept instead of being recomputed. `None` (default) recomputes every key.

Populate or refresh per-feature video info in `info.json`.

Warning: this function writes info from first episode videos, implicitly assuming that all videos have
been encoded the same way. Also, this means it assumes the first episode exists.

Always re-probes the videos and overwrites existing info for every recomputed
key. `preserve_keys` lists keys whose existing values must be kept (e.g.
data-intrinsic entries like `is_depth_map` and depth quantization params)
instead of being recomputed.

## MultiLeRobotDataset[[lerobot.datasets.MultiLeRobotDataset]]

#### lerobot.datasets.MultiLeRobotDataset[[lerobot.datasets.MultiLeRobotDataset]]

```python
lerobot.datasets.MultiLeRobotDataset(repo_ids: list, root: str | pathlib.Path | None = None, episodes: dict | None = None, image_transforms: collections.abc.Callable | None = None, delta_timestamps: dict[str, list[float]] | None = None, tolerances_s: dict | None = None, download_videos: bool = True, video_backend: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/multi_dataset.py#L34)

A dataset consisting of multiple underlying `LeRobotDataset`s.

The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
structure of `LeRobotDataset`.

#### clear_image_transforms[[lerobot.datasets.MultiLeRobotDataset.clear_image_transforms]]

```python
clear_image_transforms()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/multi_dataset.py#L111)

Remove the transform from this dataset and its children.

#### set_image_transforms[[lerobot.datasets.MultiLeRobotDataset.set_image_transforms]]

```python
set_image_transforms(image_transforms: collections.abc.Callable | None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/multi_dataset.py#L103)

Replace the transform for this dataset and its children.

## StreamingLeRobotDataset[[lerobot.datasets.StreamingLeRobotDataset]]

#### lerobot.datasets.StreamingLeRobotDataset[[lerobot.datasets.StreamingLeRobotDataset]]

```python
lerobot.datasets.StreamingLeRobotDataset(repo_id: str, root: str | pathlib.Path | None = None, episodes: list[int] | None = None, image_transforms: collections.abc.Callable[[torch.Tensor], torch.Tensor] | None = None, delta_timestamps: dict[str, list[float]] | None = None, tolerance_s: float = 0.0001, revision: str | None = None, force_cache_sync: bool = False, streaming: bool | None = None, buffer_size: int = 1000, max_num_shards: int = 16, seed: int = 42, rng: numpy.random._generator.Generator | None = None, shuffle: bool = True, return_uint8: bool = False, depth_output_unit: str = 'mm', video_backend: str | None = None, data_root: str | pathlib.Path | None = None, episode_pool_size: int | None = None, prefetch_episodes: int = 8, byte_budget_gb: float = 8.0, repeat: bool = False, repo_type: typing.Literal['dataset', 'bucket'] = 'dataset', token: str | bool | None = None, decode_threads: int = 2, decoded_queue_size: int = 8, video_decoder_cache_size: int | None = None, native_http_connections: int | None = None, native_http_subranges: int = 1, sidecar_lock_timeout_s: float = 1800, sampling_strategy: lerobot.streaming.episode_pool.StreamingSamplingStrategy | str = <StreamingSamplingStrategy.REMAINING: 'remaining'>)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L169)

Episode-scoped streaming reader for LeRobot datasets.

Metadata is cached locally, while each rank reads only the Parquet rows and MP4 byte ranges
needed for the complete episodes it owns. With N DataLoader workers, each worker owns a disjoint,
frame-balanced part of the rank's episodes, with its own episode pool and 1/N of the byte budget.
Each process has its own Python GIL, so N workers decode and assemble samples in parallel.
Episode ownership is disjoint and every selected frame is yielded exactly once per iteration.
MP4 sidecars are resolved automatically and built in a revision-keyed local cache when absent.

Example:

Basic usage:
```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Create a streaming dataset with delta timestamps
delta_timestamps = {
    "observation.image": [-1.0, -0.5, 0.0],  # 1 sec ago, 0.5 sec ago, current
    "action": [0.0, 0.1, 0.2],  # current, 0.1 sec future, 0.2 sec future
}

dataset = StreamingLeRobotDataset(
    repo_id="your-dataset-repo-id",
    delta_timestamps=delta_timestamps,
)

# Iterate over the dataset
for i, item in enumerate(dataset):
    print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
    # item will contain stacked frames according to delta_timestamps
    if i >= 10:
        break
```

#### load_state_dict[[lerobot.datasets.StreamingLeRobotDataset.load_state_dict]]

```python
load_state_dict(state: dict)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L975)

**Parameters:**

state (*dict[str, int]*) : Rank-local epoch, sample offset and batch size. Missing keys default to 0, 0 and 1.

**Raises:** `ValueError`

- `ValueError` -- If the epoch or offset is negative, or the batch size is not positive.

Restore the next iterator's anchor position without fetching skipped samples.

Note:
Reproducing anchor order requires the same dataset, seed, sampler, pool settings,
distributed topology and number of DataLoader workers. With several workers, the offset
is the rank's offset in whole batches of `batch_size`; each worker derives its own part,
because the DataLoader takes batches from its workers in turn. Random image transforms
are not restored by this method.

#### num_episodes_for_rank[[lerobot.datasets.StreamingLeRobotDataset.num_episodes_for_rank]]

```python
num_episodes_for_rank(rank: int, world_size: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L775)

Return the number of whole episodes one training rank owns.

#### num_frames_for_rank[[lerobot.datasets.StreamingLeRobotDataset.num_frames_for_rank]]

```python
num_frames_for_rank(rank: int, world_size: int, num_workers: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L762)

Return frames owned by one training rank under balanced whole-episode sharding.

#### set_data_parallel_shard[[lerobot.datasets.StreamingLeRobotDataset.set_data_parallel_shard]]

```python
set_data_parallel_shard(rank: int, world_size: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L671)

Pin the data-parallel rank and world size, ahead of `torch.distributed` and the environment.

A spawned DataLoader worker has no initialized `torch.distributed`; the values reach it by pickling.

#### set_epoch[[lerobot.datasets.StreamingLeRobotDataset.set_epoch]]

```python
set_epoch(epoch: int)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L1005)

**Parameters:**

epoch (`int`) : Coverage epoch used to seed the next sample plan when shuffle is enabled.

**Raises:** ``ValueError``

- ``ValueError`` -- If epoch is negative.

Select a non-negative epoch and reset the rank-local resume offset.

#### state_dict[[lerobot.datasets.StreamingLeRobotDataset.state_dict]]

```python
state_dict()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/streaming_dataset.py#L958)

**Returns:** `dict[str, int]`

Epoch, yielded-sample offset and resume batch size.

Return the iterator's current rank-local position.

Note:
With DataLoader prefetch, yielded samples may not yet have been consumed by training.
The training pipeline restores its position from completed steps instead. With several
DataLoader workers, each worker process holds only its own position.

### Environments
https://huggingface.co/docs/lerobot/v0.7.0/api/envs.md
