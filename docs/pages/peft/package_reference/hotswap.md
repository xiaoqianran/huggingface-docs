# Hotswapping adapters

The idea of hotswapping an adapter is the following: We can already load multiple adapters, e.g. two LoRAs, at the same time. But sometimes, we want to load one LoRA and then replace its weights in-place with the LoRA weights of another adapter. This is now possible the `hotswap_adapter` function.

In general, this should be faster than deleting one adapter and loading the adapter in its place, which would be the how to achieve the same final outcome without hotswapping. Another advantage of hotswapping is that it prevents re-compilation in case the PEFT model is already compiled using `torch.compile`. This can save quite a lot of time.

## Example without `torch.compile`

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter

model_id = ...
inputs = ...
device = ...
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, <path-adapter-0>)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

## Example with `torch.compile`

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap

model_id = ...
inputs = ...
device = ...
max_rank = ...  # maximum rank among all LoRA adapters that will be used
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, <path-adapter-0>)
# Prepare the model to allow hotswapping even if ranks/scalings of 2nd adapter differ.
# You can skip this step if all ranks and scalings are identical.
prepare_model_for_compiled_hotswap(model, target_rank=max_rank)
model = torch.compile(model)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

Note that if you want to hotswap weights that were added through `target_parameters`, i.e. that directly target an `nn.Parameter`, re-compilation and/or graph breaks cannot be prevented. Therefore, it is recommended to avoid using `target_parameters` together with compiled models and hotswapping.

## Starting with a dummy adapter

If none of your adapters targets every layer needed by the others, use [helpers.create_dummy_lora_config()](/docs/peft/v0.22.0/en/package_reference/helpers#peft.helpers.create_dummy_lora_config) to prepare a dummy adapter. Its initial contribution is zero, so you can serve base-model behavior before loading trained adapter weights. The helper only inspects configurations; it does not load the base model or adapter weights.

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftConfig, get_peft_model
from peft.helpers import create_dummy_lora_config
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap

adapter_paths = ["path-to-adapter0", "path-to-adapter1"]
configs = [PeftConfig.from_pretrained(path) for path in adapter_paths]
dummy_config = create_dummy_lora_config(configs)
print("New combined LoraConfig:")
print(dummy_config.to_dict())

base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model = get_peft_model(base_model, dummy_config).eval()
# Ranks are already large enough; preparation also makes scalings safe to change after compilation.
prepare_model_for_compiled_hotswap(model)
model = torch.compile(model)

with torch.inference_mode():
    base_output = model(**inputs)

for path in adapter_paths:
    hotswap_adapter(model, path, adapter_name="default", torch_device=device)
    with torch.inference_mode():
        adapted_output = model(**inputs)
```

The dummy configuration takes the union of `target_modules` and the maximum of all `r` and `rank_pattern` values. It removes `exclude_modules`, `layers_to_transform`, and `layers_pattern` to cover all adapters. This can reserve more adapter memory than an exact union of the targeted layers. Modules absent from an incoming adapter are zeroed during hotswap. Keep the dummy adapter active during initial inference; disabling it changes the forward path and may trigger recompilation when it is enabled again.

The helper requires nonempty, explicit target-module lists or sets. String targets, including regexes and `"all-linear"`, and automatically inferred targets are unsupported. It supports standard LoRA and rsLoRA, with compatible hotswap settings (including matching dropout, rsLoRA, and alpha-pattern settings). Variants, extra saved modules, trained biases/tokens, direct parameter targets, and base-weight-changing initializations are rejected. All adapters must use the same base weights. Configuration checks cannot verify the model's actual layer types; the existing restrictions of compiled hotswapping still apply. For grouped convolutions, the maximum rank may need to be increased to a multiple of all targeted group counts before creating the model.

From a PEFT checkout, you can also save the configuration using:

```bash
python scripts/create-dummy-lora-for-hotswap.py path-to-adapter0 path-to-adapter1 --output-dir dummy-lora
```

Adapter arguments accept local directories or Hugging Face Hub repository IDs. The default output directory is `dummy-lora`. The script writes only `adapter_config.json`, not an adapter checkpoint. Load it with `LoraConfig.from_pretrained("dummy-lora")`, pass it to `get_peft_model`, and prepare the model as above. `PeftModel.from_pretrained` cannot load this directory because it has no adapter weights. If you need the dummy weights as well, call `model.save_pretrained`.

## Caveats

Hotswapping works with transformers models and diffusers models. However, there are some caveats:

- Right now, only LoRA is properly supported.
- It only works for the same PEFT method, so no swapping LoRA and LoHa, for example.
- The adapter that is being swapped in must target the same layers as the previous adapter or a subset of those layers. It cannot target new layers. Therefore, if possible, start with the adapter that targets most layers.
- Creation of a dummy LoRA config requires `target_parameters` to be a list of strings, regexes are not supported. A few other parameters of the `LoraConfig` are also not supported; if they are found, an error is raised. When targeting only some layer indices with `layers_to_transform` and `layers_pattern`, no attempt is made to find the smallest possible overlap, instead the dummy LoRA will be over-provisioned. The same is true for modules excluded via `exclude_modules`.

## API[[peft.utils.hotswap.hotswap_adapter]]

#### peft.utils.hotswap.hotswap_adapter[[peft.utils.hotswap.hotswap_adapter]]

```python
peft.utils.hotswap.hotswap_adapter(model: torch.nn.Module, model_name_or_path: str, adapter_name: str, torch_device: Optional[str] = None, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/utils/hotswap.py#L632)

**Parameters:**

model ([~PeftModel](/docs/peft/v0.22.0/en/package_reference/peft_model#peft.PeftModel)) : The PEFT model with the loaded adapter.

model_name_or_path (`str`) : The name or path of the model to load the new adapter from.

adapter_name (`str`) : The name of the adapter to swap, e.g. `"default"`. The name will stay the same after swapping.

torch_device : (`str`, *optional*, defaults to None): The device to load the new adapter onto.

- ****kwargs** (`optional`) : Additional keyword arguments used for loading the config and weights.

Substitute old adapter data with new adapter data, keeping the rest the same.

As of now, only LoRA is supported.

This function is useful when you want to replace the loaded adapter with a new adapter. The adapter name will
remain the same, but the weights and other parameters will be swapped out.

If the adapters are incomptabile, e.g. targeting different layers or having different alpha values, an error will
be raised.

Example:

```py
>>> import torch
>>> from transformers import AutoModelForCausalLM
>>> from peft import PeftModel
>>> from peft.utils.hotswap import hotswap_adapter

>>> model_id = ...
>>> inputs = ...
>>> device = ...
>>> model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

>>> # load lora 0
>>> model = PeftModel.from_pretrained(model, "path-adapter-0")
>>> model = torch.compile(model)  # optionally compile the model
>>> with torch.inference_mode():
...     output_adapter_0 = model(inputs)

>>> # replace the "default" lora adapter with the new one
>>> hotswap_adapter(model, "path-adapter-1", adapter_name="default", torch_device=device)
>>> with torch.inference_mode():
...     output_adapter_1 = model(inputs).logits
```

#### peft.utils.hotswap.hotswap_adapter_from_state_dict[[peft.utils.hotswap.hotswap_adapter_from_state_dict]]

```python
peft.utils.hotswap.hotswap_adapter_from_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], adapter_name: str, config: LoraConfig, parameter_prefix: str = 'lora_')
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/utils/hotswap.py#L426)

**Parameters:**

model (`nn.Module`) : The model with the loaded adapter.

state_dict (`dict[str, torch.Tensor]`) : The state dict of the new adapter, which needs to be compatible (targeting same modules etc.).

adapter_name (`str`) : The name of the adapter that should be hot-swapped, e.g. `"default"`. The name will remain the same after swapping.

config (`LoraConfig`) : The config of the LoRA adapter. This is used to determine the scaling and rank of the adapter.

parameter_prefix (`str`, *optional*, defaults to `"lora_"`) : The prefix used to identify the adapter's keys in the state dict. For LoRA, this would be `"lora_"` (the default).

**Raises:** ``RuntimeError``

- ``RuntimeError`` -- 
  If the old and the new adapter are not compatible, a RuntimeError is raised.

Swap out the adapter weights from the model with the weights from state_dict.

As of now, only LoRA is supported.

This is a low-level function that assumes that the adapters have been checked for compatibility and that the
state_dict has been correctly mapped to work with PEFT. For a high level function that performs this work for you,
use `hotswap_adapter` instead.

### KaSA
https://huggingface.co/docs/peft/v0.22.0/package_reference/lora_variant_kasa.md

### KaSA

> [!NOTE]
> This is a variant of LoRA and therefore everything that is possible with LoRA is valid for this method except otherwise stated on this page.

[KaSA](https://huggingface.co/papers/2412.06071) (Knowledge-aware Singular-value Adaptation) is a LoRA variant that uses the singular value decomposition of the base weight to filter out task-irrelevant knowledge and parametrizes the update with learnable singular values. It changes vanilla LoRA in two ways:

1. **Knowledge-based SVD truncation of the frozen base weight.** At initialization, the base weight `W` is SVD-factored and its `r` smallest ("noisy"/long-tail) singular components are discarded, leaving the rank-`(k - r)` approximation as the new frozen base (`k = min(in_features, out_features)`). The trainable branch then re-learns in the discarded residual subspace.
2. **Knowledge-aware singular-value adaptation.** The trainable update is parametrized in SVD form with a learnable diagonal of singular values inserted between the LoRA factors: `ΔW = scaling * B @ diag(ΔΣ) @ A`, where `ΔΣ` (`lora_diag`) is a learnable `r`-vector and the only new parameter per layer.

In PEFT, KaSA is configured as a LoRA variant through the `kasa_config` argument on [LoraConfig](/docs/peft/v0.22.0/en/package_reference/lora#peft.LoraConfig):

```py
from peft import KasaConfig, LoraConfig

config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    kasa_config=KasaConfig(beta=1e-4, gamma=1e-3),
)
```

The paper additionally trains with two auxiliary regularizers: an L2 penalty on the learnable singular values (weighted by `beta`) and an orthogonal regularization on the adapter factors (weighted by `gamma`), which softly enforces the semi-orthogonality assumed by the SVD parametrization. These cannot be injected automatically by PEFT, so during training you must add them to the task loss by calling `LoraModel._get_kasa_loss()` on the underlying `LoraModel`:

```py
task_loss = ...  # standard loss returned by your model
kasa_loss = model._get_kasa_loss()  # 0.0 if KaSA is not used
total_loss = task_loss + kasa_loss
```

For detailed usage, see [these instructions](https://github.com/huggingface/peft/tree/main/examples/kasa_finetuning).

#### Caveats

- KaSA is currently supported on standard LoRA linear layers only, and not with `fan_in_fan_out=True` layers (e.g. transformers `Conv1D`).
- KaSA adapters cannot be combined with non-KaSA adapters on the same model, since the base-weight truncation would change the base weights under the other adapters' feet. Multiple KaSA adapters are allowed.
- `convert_to_lora` is not supported: the KaSA update depends on `lora_diag` and on the truncated base weight, neither of which is representable in a vanilla LoRA adapter.
- The SVD truncation of the base weight is **destructive**: adding a KaSA adapter permanently changes the layer's frozen weight. Disabling or unloading the adapter does not restore the original base weight, and `merge` followed by `unmerge` round-trips to the truncated weight, not the original one. This is inherent to the method. Keep the original checkpoint if you need to recover the unmodified base model.
- Loading a trained KaSA adapter with `PeftModel.from_pretrained` re-applies the same truncation to the freshly loaded base weight, so saving and reloading is consistent.

### RandLora: Full-rank parameter-efficient fine-tuning of large models
https://huggingface.co/docs/peft/v0.22.0/package_reference/randlora.md
