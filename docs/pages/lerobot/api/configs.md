# Configuration

LeRobot configuration is plain dataclasses parsed by [draccus](https://github.com/dlwh/draccus), so every
field is settable from the CLI. `TrainPipelineConfig` is the top-level object for `lerobot-train`.

Polymorphic configs (policies, robots, environments) use `draccus.ChoiceRegistry`: a subclass registers
itself with `@register_subclass("name")` and is then selectable by that name on the command line.

## TrainPipelineConfig[[lerobot.configs.train.TrainPipelineConfig]]

#### lerobot.configs.train.TrainPipelineConfig[[lerobot.configs.train.TrainPipelineConfig]]

```python
lerobot.configs.train.TrainPipelineConfig(dataset: DatasetConfig, env: lerobot.envs.configs.EnvConfig | None = None, policy: lerobot.configs.policies.PreTrainedConfig | None = None, reward_model: lerobot.configs.rewards.RewardModelConfig | None = None, output_dir: pathlib.Path | None = None, job_name: str | None = None, resume: bool = False, seed: int | None = 1000, cudnn_deterministic: bool = False, tf32: bool | None = True, num_workers: int = 4, batch_size: int = 8, prefetch_factor: int = 4, persistent_workers: bool = True, dataloader_multiprocessing_context: str | None = 'spawn', steps: int = 100000, env_eval_freq: int = 20000, log_freq: int = 200, eval_steps: int = 0, max_eval_samples: int = 0, tolerance_s: float = 0.0001, save_checkpoint: bool = True, save_freq: int = 20000, checkpoint_format: CheckpointFormat = <CheckpointFormat.SAFETENSORS: 'safetensors'>, use_policy_training_preset: bool = True, optimizer: lerobot.optim.optimizers.OptimizerConfig | None = None, scheduler: lerobot.optim.schedulers.LRSchedulerConfig | None = None, parallelism: ParallelismConfig = <factory>, accelerator: AcceleratorConfig = <factory>, eval: EvalConfig = <factory>, ema: lerobot.optim.ema.EMAConfig | None = None, trackio: TrackioConfig = <factory>, peft: lerobot.configs.default.PeftConfig | None = None, job: JobConfig = <factory>, save_checkpoint_to_hub: bool = False, sample_weighting: lerobot.utils.sample_weighting.SampleWeightingConfig | None = None, rename_map: dict = <factory>)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/train.py#L126)

## PreTrainedConfig[[lerobot.configs.PreTrainedConfig]]

#### lerobot.configs.PreTrainedConfig[[lerobot.configs.PreTrainedConfig]]

```python
lerobot.configs.PreTrainedConfig(n_obs_steps: int = 1, input_features: dict[str, lerobot.configs.types.PolicyFeature] | None = <factory>, output_features: dict[str, lerobot.configs.types.PolicyFeature] | None = <factory>, device: str | None = None, dtype: torch.dtype | None = None, use_amp: bool = False, use_peft: bool = False, push_to_hub: bool = True, repo_id: str | None = None, private: bool | None = None, tags: list[str] | None = None, license: str | None = None, pretrained_path: pathlib.Path | None = None, pretrained_revision: str | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/policies.py#L42)

**Parameters:**

n_obs_steps : Number of environment steps worth of observations to pass to the policy (takes the current step and additional steps going back).

input_features : A dictionary defining the PolicyFeature of the input data for the policy. The key represents the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.

output_features : A dictionary defining the PolicyFeature of the output data for the policy. The key represents the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.

normalization_mapping : A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)

Base configuration class for policy models.

## DatasetConfig[[lerobot.configs.DatasetConfig]]

#### lerobot.configs.DatasetConfig[[lerobot.configs.DatasetConfig]]

```python
lerobot.configs.DatasetConfig(repo_id: str, repo_type: str = 'dataset', root: str | None = None, episodes: list[int] | None = None, exclude_episodes: list[int] | None = None, image_transforms: ImageTransformsConfig = <factory>, revision: str | None = None, use_imagenet_stats: bool = True, video_backend: str = <factory>, return_uint8: bool = False, depth_output_unit: str = 'mm', streaming: bool = False, streaming_episode_pool_size: int | None = None, streaming_sampling_strategy: StreamingSamplingStrategy = <StreamingSamplingStrategy.REMAINING: 'remaining'>, streaming_prefetch_episodes: int = 8, streaming_byte_budget_gb: float = 8.0, streaming_decode_threads: int = 2, streaming_decoded_queue_size: int = 8, streaming_native_http_connections: int | None = None, streaming_native_http_subranges: int = 1, streaming_sidecar_lock_timeout_s: float = 1800, video_decoder_cache_size: int | None = None, eval_split: float = 0.0)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/default.py#L30)

## EvalConfig[[lerobot.configs.EvalConfig]]

#### lerobot.configs.EvalConfig[[lerobot.configs.EvalConfig]]

```python
lerobot.configs.EvalConfig(n_episodes: int = 50, batch_size: int = 0, use_async_envs: bool = True, recording: bool = False, recording_repo_id: str | None = None, recording_private: bool = False, max_episode_steps: int | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/default.py#L169)

## TrackioConfig[[lerobot.configs.TrackioConfig]]

#### lerobot.configs.TrackioConfig[[lerobot.configs.TrackioConfig]]

```python
lerobot.configs.TrackioConfig(enable: bool = False, disable_artifact: bool = True, project: str = 'lerobot', space_id: str | None = None, private: bool | None = None, run_name: str | None = None, resume: str | None = None, capture_console: bool = True)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/default.py#L136)

Training run tracking with Trackio (https://huggingface.co/docs/trackio).

Runs are stored in a local SQLite database under `~/.cache/huggingface/trackio` and viewed with
`trackio show --project <project>`. Set `space_id` to sync the run to a Hugging Face Space instead.

### Motors
https://huggingface.co/docs/lerobot/v0.7.0/api/motors.md
