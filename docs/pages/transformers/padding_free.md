# Padding-free training

Padding-free training (also called packing) concatenates several samples into a single sequence instead of padding each one to a fixed length. The model needs to know where each sample ends so (linear) attention doesn't mix tokens across samples.

There are two ways to provide those boundaries.

- Prepare them ahead of time with a data collator.
- Infer them from `position_ids` at runtime.

The recommended approach is the data collator. This guide explains why and covers the caveats of the `position_ids` path.

> [!WARNING]
> Inferring boundaries from `position_ids` is not the preferred approach, and it only works for standard attention models. Linear-attention models such as Qwen3-Next and Qwen3.5 (Gated DeltaNet) and convolution-based models ignore `position_ids` boundaries and require the data collator. See [Linear attention and convolution models](#linear-attention-and-convolution-models).

## Prepare boundaries with a data collator

Preparing the boundary kwargs up front removes the problems above and behaves identically whether or not you compile.

Use [DataCollatorWithFlattening](/docs/transformers/v5.14.0/en/main_classes/data_collator#transformers.DataCollatorWithFlattening) to flatten each batch and return the boundary information. Set `return_flash_attn_kwargs=True` so the collator precomputes the boundaries instead of leaving them to be inferred from `position_ids` at runtime. Pass it to [Trainer](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer) and don't add an `attention_mask`, since the flattened batch already encodes the boundaries and a mask conflicts with the packed layout.

> [!TIP]
> Padding-free relies on a FlashAttention implementation for standard attention models, since only the FlashAttention kernels expose the variable-length path that a flattened batch needs.
>
> Install the [kernels](./kernels) library, which fetches a prebuilt FlashAttention kernel without requiring a local build. It also works as a fallback when [flash-attn](https://github.com/Dao-AILab/flash-attention) isn't installed locally. Load the model with `attn_implementation="kernels-community/flash-attn2"`.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening, Trainer, TrainingArguments

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.map(
    lambda example: tokenizer(example["text"], truncation=True, max_length=512),
    remove_columns=dataset.column_names,
)

# return_flash_attn_kwargs=True precomputes the sequence boundaries
data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="padding-free-llama"),
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()
```

## Infer boundaries from position_ids

FlashAttention can detect padding-free batches from `position_ids` alone and remains for backward compatibility, because downstream frameworks such as TRL depend on it.

Relying on `position_ids` has two problems.

- Detecting packed sequences from `position_ids` is a dynamic, data-dependent check. It works without compilation, but under `torch.compile` it causes graph breaks. The check is currently restricted to `batch_size == 1` to limit how often it runs, since real batch sizes are usually larger.
- Compiled FlashAttention forces some kwargs to be plain Python `int`s. Inferring them from `position_ids` at runtime forces device-to-host syncs, and on older PyTorch versions an extra graph break from the tensor-to-int conversion.

## Linear attention and convolution models

Gated DeltaNet (GDN), other linear-attention layers, and causal convolutions have no `position_ids`-only path, by design. Preparing the data with the collator is the only supported option for these models.

> [!WARNING]
> Don't rely on `position_ids` alone for GDN, linear-attention, or causal convolution models. Prepare the boundary kwargs, including `seq_idx`, with the data collator.

For these models, set both `return_flash_attn_kwargs=True` and `return_seq_idx=True`.

```python
from transformers import DataCollatorWithFlattening

data_collator = DataCollatorWithFlattening(
    return_flash_attn_kwargs=True,
    return_seq_idx=True,
)
```

The exact kernel packages depend on the model's original implementation. Gated DeltaNet models such as Qwen3-Next and Qwen3.5 use [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and Mamba-based models such as Bamba use [mamba-ssm](https://github.com/state-spaces/mamba). Both rely on [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) for the convolution. Without the right kernels, the model falls back to reference implementations that ignore the boundary kwargs and mix tokens across samples.

> [!TIP]
> Many of these kernels are also available through the [kernels](./kernels) library, which can fetch a compatible build for you. flash-linear-attention typically still needs a direct install.

When the boundary kwargs are missing, the kernels quietly treat the whole batch as one sequence. Nothing raises an error or warning, because a runtime check would add a data-dependent branch that conflicts with `torch.compile`.

## Next steps

- See the [data collators](./data_collators) guide for other collators.
- Browse the [DataCollatorWithFlattening](/docs/transformers/v5.14.0/en/main_classes/data_collator#transformers.DataCollatorWithFlattening) API reference for the full set of arguments.
- Read [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2) for benchmarks and a deeper walkthrough.

### Accelerate
https://huggingface.co/docs/transformers/v5.14.0/accelerate.md

# Accelerate

[Accelerate](https://hf.co/docs/accelerate/index) provides a unified interface for distributed training backends like [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) or [DeepSpeed](https://www.deepspeed.ai/). It detects your environment (number of GPUs, distributed backend, mixed precision, etc.) and automatically configures training, whether you're on 1 GPU with DDP or 8 GPUs with FSDP.

Accelerate wraps the model in the appropriate distributed wrapper, moves it to the correct device, and creates a compatible optimizer. During training, Accelerate uses its own [backward](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.backward) method to handle gradient scaling for mixed precision. [Trainer](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer) calls the appropriate Accelerate APIs and delegates all distributed mechanics to Accelerate.

Configure Accelerate for [Trainer](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer) with either an Accelerate config file or [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments).

## Accelerate config file

Run the [accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) command and answer questions about your hardware and training setup. This creates a `default_config.yaml` file in your cache. The example below is for FSDP.

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_reshard_after_forward: true
  fsdp_cpu_offload: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_activation_checkpointing: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [Trainer](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer)-based script, and Accelerate reads the config file to set up training. The [fsdp_config](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments.fsdp_config) and [deepspeed](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments.deepspeed) args are unnecessary because the Accelerate config file covers the same settings.

```cli
accelerate launch train.py
```

The [accelerator_config](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments.accelerator_config) accepts settings that don't have dedicated top-level arguments. For example, set `non_blocking=True` together with `dataloader_pin_memory()` to overlap data transfer with compute for higher GPU throughput.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    dataloader_pin_memory=True,
    accelerator_config={
        "non_blocking": True,
    },
)
```

## TrainingArguments

Pass a backend-specific config to [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments). The [create_accelerator_and_postprocess()](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer.create_accelerator_and_postprocess) method reads the settings and configures training.

Pass a JSON config file or dict to `~TrainingArguments.fsdp_config`. See [FSDP](./fsdp) for a full guide and config reference.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    fsdp=True,
    fsdp_config="path/to/fsdp.json",
)
```

Pass a JSON config file or dict to `~TrainingArguments.deepspeed`. See [DeepSpeed](./deepspeed) for a full guide and config reference.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    deepspeed="path/to/ds_config.json",
)
```

DDP is configured directly through [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments) fields. See [DDP](./ddp) for details.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
    ddp_timeout=1800,
)
```

## Next steps

- See [DDP](./ddp) for data-parallel training when your model fits on one GPU.
- See [FSDP](./fsdp) for sharding parameters, gradients, and optimizer states across GPUs.
- See [DeepSpeed](./deepspeed) for ZeRO optimization and offloading.

### Continuous batching
https://huggingface.co/docs/transformers/v5.14.0/continuous_batching.md
