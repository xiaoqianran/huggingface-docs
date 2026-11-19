# PaCA: Partial Connection Adaptation

[PaCA](https://huggingface.co/papers/2503.01905) (ICLR 2025) is a PEFT method that targets training speed and activation
memory. It fine-tunes `r` randomly selected input columns ("partial connections") of each targeted pretrained weight.
The only trainable parameters are the updates of these columns; unlike LoRA, PaCA adds no extra low-rank matrices.

Methods such as LoRA run a small low-rank branch next to every pretrained layer. The branch needs few FLOPs, but
it is executed sequentially with the pretrained layer and it keeps its own copy of the input activation for the backward
pass. PaCA avoids both:

- **One GEMM per layer.** The adapted columns are written into the pretrained weight right before its matmul and restored
  right after it, so forward and backward each run a single matmul, exactly like the frozen base layer.
- **Partial activations.** The gradient of the adapted columns only depends on the matching `r` input channels, so only
  those are saved for the backward pass instead of the full input.

In terms of the resulting weights, a PaCA adapter is equivalent to a LoRA adapter of rank `r` whose `A` matrix is a fixed
column selector: `W' = W + delta @ S`, where `S` selects the `r` columns. It can therefore be converted to a LoRA adapter
of rank `r` without loss via [convert_to_lora()](/docs/peft/v0.22.0/en/package_reference/lora_conversion#peft.convert_to_lora), e.g. to serve it with a LoRA inference stack.

When to use PaCA:

- You want LoRA-level accuracy but faster fine-tuning steps and lower activation memory, e.g. to fit longer sequences or
  larger batches on the same GPU.
- You want an adapter that merges exactly into the base weights (the update is a plain column update).

Current constraints:

- Only `nn.Linear` layers (including quantized linear layers) are supported.
- PaCA runs the matmul itself instead of calling the base layer's `forward`, so forward hooks registered on the base layer
  are not called (a warning is emitted). Register hooks on the PaCA layer instead.
- Mixed adapter batches (`adapter_names` argument) are not supported.
- Training with sharded weights (FSDP, DeepSpeed ZeRO-3) has not been tested yet.
- `torch.compile` is supported: under compilation the column swap, matmul and restore run as opaque custom ops (`torch.ops.peft.paca_linear_forward`/`_backward`), so the compiler does not copy the base weights. This requires `torch.library.custom_op` (PyTorch >= 2.4); otherwise the eager implementation is traced.
- `paca_alpha / r` scales the column update and therefore acts as a multiplier of the effective learning rate.
- For inference, merge the adapter (`merge_and_unload()`); the unmerged forward still swaps the columns in and out on every call.

## Usage

```python
from transformers import AutoModelForCausalLM
from peft import PacaConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
config = PacaConfig(r=16, paca_alpha=16, target_modules="all-linear", random_seed=0)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

By default, every targeted layer fine-tunes `r` randomly selected columns; in the paper, this performed on par with
selecting the columns by weight norm or by accumulated gradients. To choose the columns yourself, for example to try
another selection strategy or to use a different number of columns per layer, pass `paca_indices_pattern`. Its keys are
layer names or regular expressions, as in LoRA's `rank_pattern`:

```python
config = PacaConfig(
    r=16,
    paca_alpha=16,
    target_modules=["q_proj", "v_proj"],
    paca_indices_pattern={"layers.0.self_attn.q_proj": [0, 2, 4, 6], "layers.1.self_attn.v_proj": list(range(32))},
)
```

Layers that match a key fine-tune the given columns, all other targeted layers select `r` random columns. The scaling
is `paca_alpha / r` with the `r` of the config in every layer, so `paca_alpha` means the same for all layers.

The abstract from the paper is:

> Prior parameter-efficient fine-tuning (PEFT) algorithms reduce memory usage and computational costs of fine-tuning
> large neural network models by training only a few additional adapter parameters, rather than the entire model.
> However, the reduction in computational costs due to PEFT does not necessarily translate to a reduction in training
> time; although the computational costs of the adapter layers are much smaller than the pretrained layers, it is well
> known that those two types of layers are processed sequentially on GPUs, resulting in significant latency overhead.
> LoRA and its variants avoid this latency overhead by merging the low-rank adapter matrices with the pretrained weights
> during inference. However, those layers cannot be merged during training since the pretrained weights must remain
> frozen while the low-rank adapter matrices are updated continuously over the course of training. Furthermore, LoRA
> and its variants do not reduce activation memory, as the first low-rank adapter matrix still requires the input
> activations to the pretrained weights to compute weight gradients. To mitigate this issue, we propose Partial
> Connection Adaptation (PaCA), which fine-tunes randomly selected partial connections within the pretrained weights
> instead of introducing adapter layers in the model. PaCA not only enhances training speed by eliminating the time
> overhead due to the sequential processing of the adapter and pretrained layers but also reduces activation memory
> since only partial activations, rather than full activations, need to be stored for gradient computation. Compared to
> LoRA, PaCA reduces training time by 22% and total memory usage by 16%, while maintaining comparable accuracy across
> various fine-tuning scenarios, such as fine-tuning on the MMLU dataset and instruction tuning on the Oasst1 dataset.
> PaCA can also be combined with quantization, enabling the fine-tuning of large models such as LLaMA3.1-70B. In
> addition, PaCA enables training with 23% longer sequence and improves throughput by 16% on both NVIDIA A100 GPU and
> INTEL Gaudi2 HPU compared to LoRA.

## Benchmark overview

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PACA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## PacaConfig[[peft.PacaConfig]]

#### peft.PacaConfig[[peft.PacaConfig]]

```python
peft.PacaConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, paca_alpha: int = 8, target_modules: list[str] | str | None = None, exclude_modules: list[str] | str | None = None, layers_to_transform: list[int] | int | None = None, layers_pattern: list[str] | str | None = None, random_seed: int | None = None, paca_indices_pattern: dict | None = <factory>, init_weights: bool = True, modules_to_save: list[str] | None = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/paca/config.py#L23)

**Parameters:**

r (`int`, *optional*, defaults to `8`) : Number of input columns (partial connections) that are fine-tuned in each targeted layer. For a weight of shape `(out_features, in_features)`, PaCA trains `r * out_features` parameters, the same number as the `B` matrix of a LoRA adapter with rank `r`. Must not exceed `in_features`.

paca_alpha (`int`, *optional*, defaults to `8`) : Scaling factor. The update of the selected columns is `paca_alpha / r * delta`. Like `lora_alpha`, this acts as a multiplier of the effective learning rate of the adapter.

target_modules (`Union[list[str], str]`, *optional*) : List of module names or regex expression of the module names to replace with PaCA. For example, `['q_proj', 'v_proj']` or `'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`. If this is specified as `'all-linear'`, all linear modules except the output layer are chosen. Only `nn.Linear` layers are supported.

exclude_modules (`Union[list[str], str]`, *optional*) : The names of the modules to not apply the adapter. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings.

layers_to_transform (`Union[list[int], int]`, *optional*) : The layer indexes to transform. If this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index.

layers_pattern (`Optional[Union[list[str], str]]`, *optional*) : The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.

random_seed (`int`, *optional*, defaults to `None`) : Seed used to select the partial connections. Each layer uses a different selection derived from this seed and the layer name. If `None`, the global torch random number generator is used. The selected indices are always stored in the adapter checkpoint, so the seed is not needed to load a trained adapter.

paca_indices_pattern (`dict`, *optional*) : Optional mapping from layer names or regexp expressions to the input column indices that are fine-tuned in the matching layers instead of `r` randomly selected ones, for example `{'^model.layers.0.self_attn.q_proj': [0, 2, 4]}`. This also allows a different number of columns per layer. The indices of a layer must be unique and smaller than `in_features`. The scaling is `paca_alpha / r` with the `r` of this config in all layers, independent of the number of given indices. Targeted layers that match no key use `r` randomly selected columns.

init_weights (`bool`, *optional*, defaults to `True`) : If `True`, the update of the selected columns is initialized to zero, i.e. the adapted model is identical to the base model at initialization. Set to `False` to initialize with random values; this is only intended for testing.

modules_to_save (`list[str]`, *optional*) : List of modules apart from PaCA layers to be set as trainable and saved in the final checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.

This is the configuration class to store the configuration of a [PacaModel](/docs/peft/v0.22.0/en/package_reference/paca#peft.PacaModel).

PaCA (Partial Connection Adaptation, https://huggingface.co/papers/2503.01905) fine-tunes `r` randomly selected
input columns (partial connections) of each targeted pretrained weight. The only trainable parameters are the
updates of these columns; unlike LoRA, PaCA adds no extra low-rank matrices. Right before the matmul, the adapted
columns are swapped into the pretrained weight and restored right after it, so forward and backward each run a
single GEMM, and only the `r` selected input channels are kept in memory for the backward pass.

## PacaModel[[peft.PacaModel]]

#### peft.PacaModel[[peft.PacaModel]]

```python
peft.PacaModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/paca/model.py#L38)

**Parameters:**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel)) : The model to be adapted.

config ([PacaConfig](/docs/peft/v0.22.0/en/package_reference/paca#peft.PacaConfig)) : The configuration of the PaCA model.

adapter_name (`str`) : The name of the adapter, defaults to `"default"`.

low_cpu_mem_usage (`bool`, `optional`, defaults to `False`) : Create empty adapter weights on meta device. Useful to speed up the loading process.

**Returns:** `torch.nn.Module`

The PaCA model.

Creates a Partial Connection Adaptation (PaCA) model from a pretrained model. The method is described in
https://huggingface.co/papers/2503.01905.

Example:

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import PacaConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = PacaConfig(r=16, paca_alpha=16, target_modules="all-linear")
>>> model = get_peft_model(base_model, config)
```

**Attributes**:
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel)) -- The model to be adapted.
- **peft_config** ([PacaConfig](/docs/peft/v0.22.0/en/package_reference/paca#peft.PacaConfig)): The configuration of the PaCA model.

### Multitask prompt tuning
https://huggingface.co/docs/peft/v0.22.0/package_reference/multitask_prompt_tuning.md
