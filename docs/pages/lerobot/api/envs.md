# Environments

Simulation environments are configured through `EnvConfig` and built by `make_env`. Each subclass
declares its `gym_kwargs` and how to construct the vectorised environments.

See [Environments from the Hub](../envhub) for using published environments and
[Adding a New Benchmark](../adding_benchmarks) for contributing one.

## EnvConfig[[lerobot.envs.EnvConfig]]

#### lerobot.envs.EnvConfig[[lerobot.envs.EnvConfig]]

```python
lerobot.envs.EnvConfig(task: str | None = None, fps: int = 30, features: dict[str, PolicyFeature] = <factory>, features_map: dict[str, str] = <factory>, max_parallel_tasks: int = 1, disable_env_checker: bool = True, server: EnvClientConfig | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/configs.py#L71)

#### create_env[[lerobot.envs.EnvConfig.create_env]]

```python
create_env(task_group: str | None = None, task_id: int | None = None, n_envs: int = 1, use_async_envs: bool = False)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/configs.py#L127)

Create one local or served vector env for one task; no `task_id` means the suite's first task.

#### create_envs[[lerobot.envs.EnvConfig.create_envs]]

```python
create_envs(n_envs: int, use_async_envs: bool = False)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/configs.py#L112)

Create local or served vector envs, as `&amp;lcub;suite: &amp;lcub;task_id: env}}`.

#### get_env_processors[[lerobot.envs.EnvConfig.get_env_processors]]

```python
get_env_processors()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/configs.py#L215)

Return (preprocessor, postprocessor) for this env. Default: identity.

#### sim_config[[lerobot.envs.EnvConfig.sim_config]]

```python
sim_config()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/configs.py#L108)

Return `sim_settings()` with the sim's defaults, as local envs and the env server run it.

#### sim_settings[[lerobot.envs.EnvConfig.sim_settings]]

```python
sim_settings()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/configs.py#L101)

Return the sim settings, as in a server YAML's `sim` section; missing ones use sim defaults.

The default is the Gymnasium env `gym_id` with `gym_kwargs` and delta actions; override otherwise.

## make_env[[lerobot.envs.make_env]]

#### lerobot.envs.make_env[[lerobot.envs.make_env]]

```python
lerobot.envs.make_env(cfg: EnvConfig | str, n_envs: int = 1, use_async_envs: bool = False, hub_cache_dir: str | None = None, trust_remote_code: bool = False)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/factory.py#L65)

**Parameters:**

cfg (`EnvConfig | str`) : The env config, or a Hugging Face Hub repo ID (e.g. `"username/repo"`) whose `env.py` builds the envs.

n_envs (`int`, *optional*, defaults to `1`) : Number of envs in each vector env.

use_async_envs (`bool`, *optional*, defaults to `False`) : Run each env in its own worker process.

hub_cache_dir (`str`, *optional*) : Folder to cache the downloaded `env.py` in. Only used with a repo ID.

trust_remote_code (`bool`, *optional*, defaults to `False`) : Consent to run a Hub repo's code, needed for a repo ID or an EnvHub config.

**Returns:** `dict[str, dict[int, gym.vector.VectorEnv]]`

The vector envs, as `{suite: {task_id: env}}`. A
single-task env has one suite, with task ID 0.

**Raises:** ``ValueError`` or ``RuntimeError`` or ``ModuleNotFoundError``

- ``ValueError`` -- If `n_envs` is below 1.
- ``RuntimeError`` -- If a Hub repo's code would run without `trust_remote_code`.
- ``ModuleNotFoundError`` -- If the simulator, or a package the repo's `env.py` needs, is not installed.

Build the vector envs of a config, or of a Hub repo's `make_env`.

## make_env_config[[lerobot.envs.make_env_config]]

#### lerobot.envs.make_env_config[[lerobot.envs.make_env_config]]

```python
lerobot.envs.make_env_config(env_type: str, **kwargs)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/envs/factory.py#L27)

### Teleoperators
https://huggingface.co/docs/lerobot/v0.7.0/api/teleoperators.md
