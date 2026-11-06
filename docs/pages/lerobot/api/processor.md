# Processors

Processors are the data transformation layer between a robot, a dataset and a policy. A pipeline is a chain
of `ProcessorStep`s; each step declares how it transforms both the data and the feature contract.

See [Introduction to Robot Processors](../introduction_processors) for the concepts,
[Implement your own processor](../implement_your_own_processor) to write a step, and
[Debug your processor pipeline](../debug_processor_pipeline) when a pipeline misbehaves.

## ProcessorStep[[lerobot.processor.ProcessorStep]]

#### lerobot.processor.ProcessorStep[[lerobot.processor.ProcessorStep]]

```python
lerobot.processor.ProcessorStep()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L150)

Abstract base class for a single step in a data processing pipeline.

Each step must implement the `__call__` method to perform its transformation
on a data transition and the `transform_features` method to describe how it
alters the shape or type of data features.

Subclasses can optionally be stateful by implementing `state_dict` and `load_state_dict`.

#### get_config[[lerobot.processor.ProcessorStep.get_config]]

```python
get_config()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L193)

**Returns:**

A JSON-serializable dictionary of configuration parameters.

Returns the configuration of the step for serialization.

#### load_state_dict[[lerobot.processor.ProcessorStep.load_state_dict]]

```python
load_state_dict(state: dict[str, torch.Tensor])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L209)

**Parameters:**

state : A dictionary of state tensors.

Loads the step's state from a state dictionary.

#### reset[[lerobot.processor.ProcessorStep.reset]]

```python
reset()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L221)

Resets the internal state of the processor step, if any.

#### save_artifacts[[lerobot.processor.ProcessorStep.save_artifacts]]

```python
save_artifacts(save_directory: Path)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L217)

Save non-tensor assets and map constructor arguments to relative paths.

#### state_dict[[lerobot.processor.ProcessorStep.state_dict]]

```python
state_dict()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L201)

**Returns:**

A dictionary mapping state names to tensors.

Returns the state of the step (e.g., learned parameters, running means).

#### transform_features[[lerobot.processor.ProcessorStep.transform_features]]

```python
transform_features(features: dict[PipelineFeatureType, dict[str, PolicyFeature]])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L225)

**Parameters:**

features : A dictionary describing the input features for observations, actions, etc.

**Returns:**

A dictionary describing the output features after this step's transformation.

Defines how this step modifies the description of pipeline features.

This method is used to track changes in data shapes, dtypes, or modalities
as data flows through the pipeline, without needing to process actual data.

## DataProcessorPipeline[[lerobot.processor.DataProcessorPipeline]]

#### lerobot.processor.DataProcessorPipeline[[lerobot.processor.DataProcessorPipeline]]

```python
lerobot.processor.DataProcessorPipeline(steps: Sequence[ProcessorStep] = <factory>, name: str = 'DataProcessorPipeline', to_transition: Callable[[TInput], EnvTransition] = <factory>, to_output: Callable[[EnvTransition], TOutput] = <factory>, before_step_hooks: list[Callable[[int, EnvTransition], None]] = <factory>, after_step_hooks: list[Callable[[int, EnvTransition], None]] = <factory>)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L273)

**Parameters:**

steps : A sequence of `ProcessorStep` objects that make up the pipeline.

name : A descriptive name for the pipeline.

to_transition : A function to convert raw input data into the standardized `EnvTransition` format.

to_output : A function to convert the final `EnvTransition` into the desired output format.

before_step_hooks : A list of functions to be called before each step is executed.

after_step_hooks : A list of functions to be called after each step is executed.

A sequential pipeline for processing data, integrated with the Hugging Face Hub.

This class chains together multiple `ProcessorStep` instances to form a complete
data processing workflow. It's generic, allowing for custom input and output types,
which are handled by the `to_transition` and `to_output` converters.

#### from_config[[lerobot.processor.DataProcessorPipeline.from_config]]

```python
from_config(config: dict[str, Any], state_dict: dict[str, dict[str, torch.Tensor]] | None = None, overrides: dict[str, Any] | None = None, to_transition: Callable[[TInput], EnvTransition] | None = None, to_output: Callable[[EnvTransition], TOutput] | None = None)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L792)

**Parameters:**

config : A config dictionary with the same structure as the saved processor JSON.

state_dict : Optional in-memory pipeline state grouped by suffixless state key.

overrides : Optional constructor overrides keyed by registry name or class name.

to_transition : Optional converter from input data to `EnvTransition`.

to_output : Optional converter from `EnvTransition` to output data.

**Returns:**

A processor pipeline built from the config and optional state.

Build a pipeline from an in-memory config and optional state tensors.

#### from_pretrained[[lerobot.processor.DataProcessorPipeline.from_pretrained]]

```python
from_pretrained(pretrained_model_name_or_path: str | Path, config_filename: str, force_download: bool = False, resume_download: bool | None = None, proxies: dict[str, str] | None = None, token: str | bool | None = None, cache_dir: str | Path | None = None, local_files_only: bool = False, revision: str | None = None, overrides: dict[str, Any] | None = None, to_transition: Callable[[TInput], EnvTransition] | None = None, to_output: Callable[[EnvTransition], TOutput] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L647)

**Parameters:**

pretrained_model_name_or_path : The identifier of the repository on the Hugging Face Hub, a path to a local directory, or a path to a single config file.

config_filename : The name of the pipeline's JSON configuration file. Always required to prevent ambiguity when multiple configs exist (e.g., preprocessor vs postprocessor).

force_download : Whether to force (re)downloading the files.

resume_download : Kept for backward compatibility and ignored. Downloads resume automatically.

proxies : Kept for backward compatibility and ignored. Use HTTP_PROXY and HTTPS_PROXY instead.

token : The token to use as HTTP bearer authorization for private Hub repositories.

cache_dir : The path to a specific cache folder to store downloaded files.

local_files_only : If True, avoid downloading files from the Hub.

revision : The specific model version to use (e.g., a branch name, tag name, or commit id).

overrides : A dictionary to override the configuration of specific steps. Keys should match the step's class name or registry name.

to_transition : A custom function to convert input data to `EnvTransition`.

to_output : A custom function to convert the final `EnvTransition` to the output format.

- ****kwargs** : Additional arguments (not used).

**Returns:**

An instance of `DataProcessorPipeline` loaded with the specified configuration and state.

**Raises:** ``FileNotFoundError`` or ``ValueError`` or ``ImportError`` or ``KeyError`` or ``ProcessorMigrationError``

- ``FileNotFoundError`` -- If the config file cannot be found.
- ``ValueError`` -- If configuration is ambiguous or instantiation fails.
- ``ImportError`` -- If a step's class cannot be imported.
- ``KeyError`` -- If an override key doesn't match any step in the pipeline.
- ``ProcessorMigrationError`` -- If the model requires migration to processor format.

Loads a pipeline from a local directory, single file, or Hugging Face Hub repository.

This method implements a simplified loading pipeline with intelligent migration detection:

**Simplified Loading Strategy**:
1. **Config Loading** (_load_config):
   - **Directory**: Load specified config_filename from directory
   - **Single file**: Load file directly (config_filename ignored)
   - **Hub repository**: Download specified config_filename from Hub

2. **Config Validation** (_validate_loaded_config):
   - Format validation: Ensure config is valid processor format
   - Migration detection: Guide users to migrate old LeRobot models
   - Clear errors: Provide actionable error messages

3. **Step Construction** (_build_steps_with_overrides):
   - Class resolution: Registry lookup or dynamic imports
   - Override merging: User parameters override saved config
   - State loading: Load .safetensors files for stateful steps

4. **Override Validation** (_validate_overrides_used):
   - Ensure all user overrides were applied (catch typos)
   - Provide helpful error messages with available keys

**Migration Detection**:
- **Smart detection**: Analyzes JSON files to detect old LeRobot models
- **Precise targeting**: Avoids false positives on other HuggingFace models
- **Clear guidance**: Provides exact migration command to run
- **Error mode**: Always raises ProcessorMigrationError for clear user action

**Loading Examples**:
```python
# Directory loading
pipeline = DataProcessorPipeline.from_pretrained("/models/my_model", config_filename="processor.json")

# Single file loading
pipeline = DataProcessorPipeline.from_pretrained(
    "/models/my_model/processor.json", config_filename="processor.json"
)

# Hub loading
pipeline = DataProcessorPipeline.from_pretrained("user/repo", config_filename="processor.json")

# Multiple configs (preprocessor/postprocessor)
preprocessor = DataProcessorPipeline.from_pretrained(
    "model", config_filename="policy_preprocessor.json"
)
postprocessor = DataProcessorPipeline.from_pretrained(
    "model", config_filename="policy_postprocessor.json"
)
```

**Override System**:
- **Key matching**: Use registry names or class names as override keys
- **Config merging**: User overrides take precedence over saved config
- **Validation**: Ensure all override keys match actual steps (catch typos)
- **Example**: overrides={"NormalizeStep": {"device": "cuda"}}

#### get_config[[lerobot.processor.DataProcessorPipeline.get_config]]

```python
get_config()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L451)

**Returns:**

A dictionary with the same content that `save_pretrained()` writes as JSON.

Return the JSON-serializable pipeline configuration.

#### load_state_dict[[lerobot.processor.DataProcessorPipeline.load_state_dict]]

```python
load_state_dict(state_dict: dict[str, dict[str, torch.Tensor]])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L515)

**Parameters:**

state_dict : A dictionary mapping suffixless state keys to step state dictionaries.

**Raises:** ``KeyError``

- ``KeyError`` -- If loading finds missing expected state or unexpected extra state.

Load pipeline state tensors into the existing steps.

#### process_action[[lerobot.processor.DataProcessorPipeline.process_action]]

```python
process_action(action: PolicyAction | RobotAction | EnvAction)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1786)

**Parameters:**

action : The action data.

**Returns:**

The processed action.

Processes only the action part of a transition through the pipeline.

#### process_complementary_data[[lerobot.processor.DataProcessorPipeline.process_complementary_data]]

```python
process_complementary_data(complementary_data: dict[str, Any])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1853)

**Parameters:**

complementary_data : The complementary data dictionary.

**Returns:**

The processed complementary data dictionary.

Processes only the complementary data part of a transition through the pipeline.

#### process_done[[lerobot.processor.DataProcessorPipeline.process_done]]

```python
process_done(done: bool | torch.Tensor)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1814)

**Parameters:**

done : The done flag.

**Returns:**

The processed done flag.

Processes only the done flag of a transition through the pipeline.

#### process_info[[lerobot.processor.DataProcessorPipeline.process_info]]

```python
process_info(info: dict[str, Any])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1840)

**Parameters:**

info : The info dictionary.

**Returns:**

The processed info dictionary.

Processes only the info dictionary of a transition through the pipeline.

#### process_observation[[lerobot.processor.DataProcessorPipeline.process_observation]]

```python
process_observation(observation: RobotObservation)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1773)

**Parameters:**

observation : The observation dictionary.

**Returns:**

The processed observation dictionary.

Processes only the observation part of a transition through the pipeline.

#### process_reward[[lerobot.processor.DataProcessorPipeline.process_reward]]

```python
process_reward(reward: float | torch.Tensor)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1801)

**Parameters:**

reward : The reward value.

**Returns:**

The processed reward.

Processes only the reward part of a transition through the pipeline.

#### process_truncated[[lerobot.processor.DataProcessorPipeline.process_truncated]]

```python
process_truncated(truncated: bool | torch.Tensor)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1827)

**Parameters:**

truncated : The truncated flag.

**Returns:**

The processed truncated flag.

Processes only the truncated flag of a transition through the pipeline.

#### register_after_step_hook[[lerobot.processor.DataProcessorPipeline.register_after_step_hook]]

```python
register_after_step_hook(fn: Callable[[int, EnvTransition], None])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1697)

**Parameters:**

fn : A callable that accepts the step index and the current transition.

Registers a function to be called after each step.

#### register_before_step_hook[[lerobot.processor.DataProcessorPipeline.register_before_step_hook]]

```python
register_before_step_hook(fn: Callable[[int, EnvTransition], None])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1673)

**Parameters:**

fn : A callable that accepts the step index and the current transition.

Registers a function to be called before each step.

#### reset[[lerobot.processor.DataProcessorPipeline.reset]]

```python
reset()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1721)

Resets the state of all stateful steps in the pipeline.

#### save_pretrained[[lerobot.processor.DataProcessorPipeline.save_pretrained]]

```python
save_pretrained(save_directory: str | Path | None = None, repo_id: str | None = None, push_to_hub: bool = False, card_kwargs: dict[str, Any] | None = None, config_filename: str | None = None, **push_to_hub_kwargs)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L595)

**Parameters:**

save_directory : The directory where the pipeline will be saved. If None, saves to HF_LEROBOT_HOME/processors/{sanitized_pipeline_name}.

repo_id : ID of your repository on the Hub. Used only if `push_to_hub=true`.

push_to_hub : Whether or not to push your object to the Hugging Face Hub after saving it.

card_kwargs : Additional arguments passed to the card template to customize the card.

config_filename : The name of the JSON configuration file. If None, a name is generated from the pipeline's `name` attribute.

- ****push_to_hub_kwargs** : Additional key word arguments passed along to the push_to_hub method.

Saves the pipeline's configuration and state to a directory.

This method creates a JSON configuration file that defines the pipeline's structure
(name and steps). For each stateful step, it also saves a `.safetensors` file
containing its state dictionary.

#### state_dict[[lerobot.processor.DataProcessorPipeline.state_dict]]

```python
state_dict()
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L488)

**Returns:**

A dictionary mapping suffixless state keys to cloned step state dictionaries.

Return pipeline state tensors grouped by state key.

#### step_through[[lerobot.processor.DataProcessorPipeline.step_through]]

```python
step_through(data: TInput)
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L343)

**Parameters:**

data : The input data.

**Yields:**

The `EnvTransition` object, starting with the initial state and then after
each processing step.

Processes data step-by-step, yielding the transition at each stage.

This is a generator method useful for debugging and inspecting the intermediate
state of the data as it passes through the pipeline.

#### transform_features[[lerobot.processor.DataProcessorPipeline.transform_features]]

```python
transform_features(initial_features: dict[PipelineFeatureType, dict[str, PolicyFeature]])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1750)

**Parameters:**

initial_features : A dictionary describing the initial features.

**Returns:**

The final feature description after all transformations.

Applies feature transformations from all steps sequentially.

This method propagates a feature description dictionary through each step's
`transform_features` method, allowing the pipeline to statically determine
the output feature specification without processing any real data.

#### unregister_after_step_hook[[lerobot.processor.DataProcessorPipeline.unregister_after_step_hook]]

```python
unregister_after_step_hook(fn: Callable[[int, EnvTransition], None])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1705)

**Parameters:**

fn : The exact function object that was previously registered.

**Raises:** ``ValueError``

- ``ValueError`` -- If the hook is not found in the list.

Unregisters an 'after_step' hook.

#### unregister_before_step_hook[[lerobot.processor.DataProcessorPipeline.unregister_before_step_hook]]

```python
unregister_before_step_hook(fn: Callable[[int, EnvTransition], None])
```

[Source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1681)

**Parameters:**

fn : The exact function object that was previously registered.

**Raises:** ``ValueError``

- ``ValueError`` -- If the hook is not found in the list.

Unregisters a 'before_step' hook.

## PolicyProcessorPipeline[[lerobot.processor.DataProcessorPipeline]]

#### lerobot.processor.DataProcessorPipeline[[lerobot.processor.DataProcessorPipeline]]

```python
lerobot.processor.DataProcessorPipeline(*args, **kwargs)
```

### Configuration
https://huggingface.co/docs/lerobot/v0.7.0/api/configs.md
