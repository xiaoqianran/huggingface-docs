# Policies

Every policy inherits `PreTrainedPolicy`, which combines a `torch.nn.Module` with the Hub mixin, so any
policy can be pushed to and loaded from the Hugging Face Hub with the same two calls.

Each policy has its own guide with training recipes and results — [ACT](../act), [SmolVLA](../smolvla),
[π₀](../pi0), [π₀.₅](../pi05) and the rest are listed under Policies. To add one, see
[Adding a Policy](../bring_your_own_policies).

## PreTrainedPolicy[[lerobot.policies.PreTrainedPolicy]]

#### lerobot.policies.PreTrainedPolicy[[lerobot.policies.PreTrainedPolicy]]

```python
lerobot.policies.PreTrainedPolicy(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L73)

Base class for policy models.

#### chunk_inference_spec[[lerobot.policies.PreTrainedPolicy.chunk_inference_spec]]

```python
chunk_inference_spec()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L308)

Declare the default current-observation chunk-serving author contract.

Inheriting this implementation declares that `predict_action_chunk` does
every preparation needed by direct callers, accepts the complete current
observation, and returns `[B, chunk_size, A]` in model coordinates. Its
canonical processor pair must support whole chunks. `n_action_steps` is
the ordinary playback slice; RTC retains the full prediction horizon.

A policy requiring select_action-only preparation, other action layouts,
or temporal sampling must override this method and reject unsupported
configurations or provide a custom runner. Method presence or a successful
warmup is not evidence for this contract. Policy/processor state is allowed
only within an exclusive session and reset() must clear it for a new run.

#### count_queued_actions[[lerobot.policies.PreTrainedPolicy.count_queued_actions]]

```python
count_queued_actions()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L263)

Number of actions already computed and waiting to be served.

0 for a policy that keeps no action queue (never chunks) or whose queue is
currently empty (about to compute a fresh chunk). Reads the same queues
`drop_queued_actions` clears, named in :attr:*_action_queue_attrs*.

#### drop_queued_actions[[lerobot.policies.PreTrainedPolicy.drop_queued_actions]]

```python
drop_queued_actions()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L245)

Discard actions precomputed by earlier `select_action` calls.

Forces a fresh forward pass on the next `select_action` so a mid-episode conditioning
change (e.g. a new instruction) takes effect at once instead of after the queue drains.
Unlike `reset`, the rest of the episode state is kept.  Call it from the thread
that calls `select_action`.  Clears the queues named in :attr:*_action_queue_attrs*;
a policy that keeps no action queue inherits a no-op.

#### forward[[lerobot.policies.PreTrainedPolicy.forward]]

```python
forward(batch: dict[str, Tensor])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L376)

**Parameters:**

batch (dict[str, Tensor]) : _description_

**Returns:** tuple[Tensor, dict | None]

The loss and potentially other information. Apart from the loss which
is a Tensor, all other items should be logging-friendly, native Python types.

_summary_

#### from_pretrained[[lerobot.policies.PreTrainedPolicy.from_pretrained]]

```python
from_pretrained(pretrained_name_or_path: str | Path, config: PreTrainedConfig | None = None, force_download: bool = False, resume_download: bool | None = None, proxies: dict | None = None, token: str | bool | None = None, cache_dir: str | Path | None = None, local_files_only: bool = False, revision: str | None = None, strict: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L159)

The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
deactivated). To train it, you should first set it back in training mode with `policy.train()`.

#### generate_text[[lerobot.policies.PreTrainedPolicy.generate_text]]

```python
generate_text(batch: dict[str, Any])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L363)

Run the policy's text head on a preprocessed observation batch.

The request rides on `batch` as complementary data (:data:*~lerobot.utils.constants.QUERY_KIND*
/ `QUERY_TEXT`); a `next_subtask` reply is fed straight into `set_task`, so it must be
exactly one subtask, not a plan or a numbered list.  Returns the generated text, and must not
mutate action-producing state (queues, observation history).

#### get_optim_params[[lerobot.policies.PreTrainedPolicy.get_optim_params]]

```python
get_optim_params()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L230)

Returns the policy-specific parameters (parameters or param groups) to be passed on to the optimizer.

#### predict_action_chunk[[lerobot.policies.PreTrainedPolicy.predict_action_chunk]]

```python
predict_action_chunk(batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L389)

Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

Child classes using action chunking should use this method within `select_action` to form the action chunk
cached for selection.

A policy that supports a `noise` keyword starts its sampling from that tensor, shaped as the policy would
draw it. When `noise` is None, the policy draws it as usual.

May return a `PolicyOutput` carrying the chunk and its prediction, as for `select_action`.

#### push_model_to_hub[[lerobot.policies.PreTrainedPolicy.push_model_to_hub]]

```python
push_model_to_hub(cfg: TrainPipelineConfig, peft_model = None, state_dict: dict[str, Tensor] | None = None, dataset_meta: LeRobotDatasetMetadata | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L431)

**Parameters:**

cfg (TrainPipelineConfig) : The training config; saved as *train_config.json* and used to render the model card.

peft_model : The PEFT wrapper when training adapters, whose weights replace the full model weights in the published repo. Defaults to None.

state_dict (dict[str, Tensor] | None) : Ignored; weights are now gathered internally when the policy is sharded. Defaults to None.

dataset_meta (LeRobotDatasetMetadata | None) : Dataset metadata for the model card, if available. Defaults to None.

Publish this policy to the Hub.

Deprecated: use `lerobot.common.train_utils.publish_trained_model()` instead, which
also publishes the pre/post-processors alongside the model.

#### reset[[lerobot.policies.PreTrainedPolicy.reset]]

```python
reset()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L237)

To be called whenever the environment is reset.

Does things like clearing caches.

#### select_action[[lerobot.policies.PreTrainedPolicy.select_action]]

```python
select_action(batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L405)

Return one action to run in the environment (potentially in batch mode).

When the model uses a history of observations, or outputs a sequence of actions, this method deals
with caching.

Returns a bare action tensor, or a *PolicyOutput*: a batch, keyed like the input batch, holding
the action under `"action"` and what the policy expects or believes besides it under
`"prediction"` (a *PolicyPrediction*). The prediction is batched like the action, one value
per environment, and each entry is keyed by what it refers to: predicted frames by the
observation key they predict, language by annotation style ("subtask", "plan", "memory"),
boxes (annotation VQA schema) by the camera they are drawn on. Callers normalize the two
return types with *lerobot.processor.policy_output_to_transition*; *--display_data* shows
the prediction of the first environment.

A prediction holds values for this step: one spanning a chunk (e.g. an imagined clip) is
returned piece by piece, with the action each piece belongs to. Each entry stays displayed
until it is returned again. Predicting is opt-in through a config field of the policy's
choosing, and must not change the actions; text that should steer the policy goes
through *generate_text*.

#### supports_rtc[[lerobot.policies.PreTrainedPolicy.supports_rtc]]

```python
supports_rtc()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L280)

Whether this policy implements Real-Time Chunking inference semantics.

#### supports_text_generation[[lerobot.policies.PreTrainedPolicy.supports_text_generation]]

```python
supports_text_generation()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L359)

Whether this policy implements `generate_text` (override both together).

#### validate_chunk_input_features[[lerobot.policies.PreTrainedPolicy.validate_chunk_input_features]]

```python
validate_chunk_input_features(features: tuple[FeatureSpec, ...])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L284)

Validate canonical inputs before serving, with exact shapes by default.

Policies that explicitly resize or mask missing inputs may override this
contract. Such exceptions belong with the policy, never the transport.
The negotiated wire schema still fixes every supplied feature's shape.

#### wrap_with_peft[[lerobot.policies.PreTrainedPolicy.wrap_with_peft]]

```python
wrap_with_peft(peft_config = None, peft_cli_overrides: dict | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py#L471)

**Parameters:**

peft_config : Optional PEFT adapter configuration (e.g., LoraConfig). If provided, used directly (with CLI overrides applied).

peft_cli_overrides : Optional dict of CLI overrides (method_type, target_modules, r, etc.) These are merged with policy defaults to build the final config.

Wrap this policy with PEFT adapters for parameter-efficient fine-tuning.

This method is the single entry point for PEFT integration. Subclasses should
override `_get_default_peft_targets()` to provide default target modules, and
`_validate_peft_config()` for policy-specific validation.

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

## make_policy[[lerobot.policies.make_policy]]

#### lerobot.policies.make_policy[[lerobot.policies.make_policy]]

```python
lerobot.policies.make_policy(cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata | None = None, env_cfg: EnvConfig | None = None, rename_map: dict[str, str] | None = None, defer_weight_load: bool = False, pretrained_path: str | Path | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/factory.py#L255)

**Parameters:**

cfg (PreTrainedConfig) : The configuration for the policy to be created. If `cfg.pretrained_path` is set, the policy will be loaded with weights from that path.

ds_meta (LeRobotDatasetMetadata | None) : Dataset metadata used to infer feature shapes and types. Also provides statistics for normalization layers.

env_cfg (EnvConfig | None) : Environment configuration used to infer feature shapes and types. One of `ds_meta` or `env_cfg` must be provided.

rename_map (dict[str, str] | None) : Optional mapping of dataset or environment feature keys to match expected policy feature names (e.g., `"left"` → `"camera1"`).

defer_weight_load (bool) : Build the exact policy `from_pretrained` would build — same config resolution, same stats-derived buffers, same device placement and eval mode — but skip the safetensors weight load. Used when resuming from a DCP checkpoint, whose sharded weights stream in after `accelerator.prepare()` (the distributed checkpoint engine overwrites the random init). For a PEFT policy only the adapter weights are skipped, the base weights load as usual.

pretrained_path (str | Path | None) : Load the weights (or PEFT adapter) from here instead of `cfg.pretrained_path`, which keeps naming the model the policy was fine-tuned from once the policy is built. Used when resuming from a checkpoint.

**Returns:** `PreTrainedPolicy`

An instantiated and device-placed policy model.

**Raises:** ``ValueError`` or ``NotImplementedError``

- ``ValueError`` -- If both or neither of `ds_meta` and `env_cfg` are provided.
- ``NotImplementedError`` -- If attempting to use an unsupported policy-backend combination
  (e.g., VQBeT with 'mps').

Instantiate a policy model.

This factory function handles the logic of creating a policy, which requires
determining the input and output feature shapes. These shapes can be derived
either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
can either initialize a new policy from scratch or load a pretrained one.

### Robots
https://huggingface.co/docs/lerobot/v0.7.0/api/robots.md
