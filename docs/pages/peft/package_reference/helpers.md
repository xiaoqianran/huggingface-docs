# Helper methods

A collection of helper functions for PEFT.

## Checking if a model is a PEFT model[[peft.helpers.check_if_peft_model]]

#### peft.helpers.check_if_peft_model[[peft.helpers.check_if_peft_model]]

```python
peft.helpers.check_if_peft_model(model_name_or_path: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L145)

**Parameters:**

model_name_or_path (`str`) : Model id to check, can be local or on the Hugging Face Hub.

**Returns:** `bool`

True if the model is a PEFT model, False otherwise.

Check if the model is a PEFT model.

## Temporarily Rescaling Adapter Scale in LoraLayer Modules[[peft.helpers.rescale_adapter_scale]]

#### peft.helpers.rescale_adapter_scale[[peft.helpers.rescale_adapter_scale]]

```python
peft.helpers.rescale_adapter_scale(model: Module, multiplier: float | int)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L166)

**Parameters:**

model : The model containing `LoraLayer` modules whose scaling is to be adjusted.

multiplier (float or int) : The multiplier that rescales the `scaling` attribute. Must be of type float or int.

**Raises:** ``ValueError``

- ``ValueError`` -- If the model does not contain any `LoraLayer`
  instances, indicating that the model does not support scaling.

Context manager to temporarily rescale the scaling of the LoRA adapter in a model.

The original scaling values are restored when the context manager exits. This context manager works with the
transformers and diffusers models that have directly loaded LoRA adapters.

For LoRA, applying this context manager with multiplier in [0, 1] is strictly equivalent to applying
[wise-ft](https://huggingface.co/papers/2109.01903) (see [#1940](https://github.com/huggingface/peft/issues/1940)
for details). It can improve the performances of the model if there is a distribution shift between the training
data used for fine-tuning, and the test data used during inference.

Warning: It has been reported that when using Apple's MPS backend for PyTorch, it is necessary to add a short sleep
time after exiting the context before the scales are fully restored.

Example:

```python
>>> model = ModelWithLoraLayer()
>>> multiplier = 0.5
>>> with rescale_adapter_scale(model, multiplier):
...     outputs = model(**inputs)  # Perform operations with the scaled model
>>> outputs = model(**inputs)  # The original scaling values are restored here
```

## Context manager to disable input dtype casting in the `forward` method of LoRA layers[[peft.helpers.disable_input_dtype_casting]]

#### peft.helpers.disable_input_dtype_casting[[peft.helpers.disable_input_dtype_casting]]

```python
peft.helpers.disable_input_dtype_casting(model: Module, active: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L227)

**Parameters:**

model (nn.Module) : The model containing PEFT modules whose input dtype casting is to be adjusted.

active (bool) : Whether the context manager is active (default) or inactive.

Context manager disables input dtype casting to the dtype of the weight.

## Context manager to enable DoRA caching (faster at inference time but requires more memory)[[peft.helpers.DoraCaching]]

#### peft.helpers.DoraCaching[[peft.helpers.DoraCaching]]

```python
peft.helpers.DoraCaching(enabled: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L347)

Context manager to enable DoRA caching, which improves speed of DoRA inference at the expense of memory.

With active caching, the materialized LoRA weight (B @ A) and the weight norm (base weight + LoRA weight) are
cached.

Even within the caching context, if the model is in training mode, caching is disabled. When the model switches to
training mode, the cache will be cleared.

Example:

```py
>>> from peft.helpers import DoraCaching

>>> model.eval()  # put in eval model for caching to work

>>> with DoraCaching():  # use as a context manager
...     output = model(inputs)

>>> dora_caching = DoraCaching()
>>> dora_caching(enabled=True)  # permanently enable caching
>>> output = model(inputs)
>>> dora_caching(enabled=False)  # permanently disable caching
>>> output = model(inputs)
```

## Creating a dummy LoRA configuration for hotswapping[[peft.helpers.create_dummy_lora_config]]

See [starting with a dummy adapter](hotswap#starting-with-a-dummy-adapter) for the workflow and command-line script.

#### peft.helpers.create_dummy_lora_config[[peft.helpers.create_dummy_lora_config]]

```python
peft.helpers.create_dummy_lora_config(configs: Sequence)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L570)

**Parameters:**

configs (`Sequence[LoraConfig]`) : Nonempty sequence of compatible LoRA configurations for the same base model. Each must have explicit, nonempty `target_modules`; strings (including regexes and `"all-linear"`) and inferred targets are not supported. Standard LoRA and rsLoRA are supported, without extra saved modules, biases, or parameters.

**Returns:** `LoraConfig`

A configuration with zero-output initialization and a uniform maximum rank.

**Raises:** ``TypeError`` or ``ValueError``

- ``TypeError`` -- 
  If any configuration is not a LoRA configuration.
- ``ValueError`` -- 
  If no configurations are supplied, or their settings are unsupported or incompatible.

Create a no-op LoRA configuration that reserves capacity for hotswapping.

The purpose of this helper function is to create a dummy LoRA config which can be used to initialze a LoRA model
before hotswapping different LoRA adapters (see
https://huggingface.co/docs/peft/main/en/package_reference/hotswap). One issue with hotswapping is that the first
LoRA adapter that is loaded needs to install LoRA weights on all layers that the later, hotswapped LoRAs will
target. If these LoRAs target disjunct layers, none of them can serve as the first LoRA. This helper function
creates a dummy LoRA that targets the union of the layers of all the individual LoRAs. It also pre-allocates LoRA
weights with a sufficiently large rank.

The generated LoRA config is not intended to be used for further training.

No base model or adapter weights are loaded. Layer filters and exclusions are removed to cover every adapter, which
can over-allocate LoRA weights on additional layers.

Instantiate the result with `get_peft_model`, then call `prepare_model_for_compiled_hotswap` before compiling. The
dummy adapter is initialized such that it is a no-op, i.e. it does not need to be disabled to run inference on the
base model.

Note that some `LoraConfig` arguments are not supported and will result in an error. This is because it would not
be possible to combine those configs into one dummy config.

There is a helper script in the PEFT repository to create the dummy LoRA config from the command line:

`python scripts/create-dummy-lora-for-hotswap.py adapter0 adapter1 [...] --output-dir dummy-lora`

## KappaTune target selection[[peft.helpers.KappaTuneSelector]]

`KappaTuneSelector` and `find_kappa_target_modules` implement a general target selection process from the [KappaTune paper](https://arxiv.org/abs/2506.16289). 

The method identifies modules with higher flexibility (higher output differential entropy) and lower specialization (lower sensitivity to specific input directions).

These properties make the selected modules good candidates for mitigating catastrophic forgetting in any adaptation method that adds trainable parameters, including LoRA, DoRA, LoHa, AdaLoRA, and even direct fine-tuning of the original weights.

#### peft.helpers.KappaTuneSelector[[peft.helpers.KappaTuneSelector]]

```python
peft.helpers.KappaTuneSelector(model: Module, max_dim_size_to_analyze: int = 16384, moe_param_suffixes: tuple[str, ...] | None = None, show_progress: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L391)

Lightweight utility to compute per-module / per-parameter condition numbers and return the best LoRA targets.

Supports:
- Classic nn.Linear modules (target_modules in LoraConfig)
- Modern fused MoE weights stored as 3D nn.Parameter (gate_up_proj / down_proj, gate_proj / up_proj, etc.) used in
  Llama-4, Qwen2_MoE, Qwen3_MoE, Mixtral, OLMoE and similar models. These are returned via target_parameters.

Notes:
- Condition-number computation requires running SVD and can take several minutes on very large models. A progress
bar can be shown/disabled via `show_progress`.

#### peft.find_kappa_target_modules[[peft.find_kappa_target_modules]]

```python
peft.find_kappa_target_modules(model: Module, top_p: float = 0.2, max_dim_size_to_analyze: int = 16384, moe_param_suffixes: tuple[str, ...] | None = None, show_progress: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/helpers.py#L526)

**Parameters:**

model (nn.Module) : Base model whose weights will be analyzed for condition numbers.

top_p (float, optional) : Select the top fraction of candidate modules/parameters with the lowest condition numbers.

max_dim_size_to_analyze (int, optional) : Upper bound on the maximum matrix dimension analyzed via SVD. Defaults to 16384.

moe_param_suffixes (Optional[tuple[str, ...]], optional) : Parameter-name suffixes used to identify fused MoE tensors that should be returned via `target_parameters`. If None, sensible defaults are used.

show_progress (bool, optional) : Whether to display a progress bar while computing condition numbers (SVD-based) across candidate tensors/modules. Disable in CI or other non-interactive environments. Defaults to True.

One-liner convenience function for KappaTune target selection. Returns both target_modules and target_parameters.

### VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks
https://huggingface.co/docs/peft/v0.22.0/package_reference/vblora.md
