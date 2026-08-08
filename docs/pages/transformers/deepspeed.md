# DeepSpeed ZeRO

[DeepSpeed](https://www.deepspeed.ai/) ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy across distributed training by sharding optimizer states, gradients, and parameters across GPUs. ZeRO has three stages, each sharding more state than the last. DeepSpeed also supports offloading to CPU or NVMe memory for further savings. Every additional stage and offload level reduces peak memory, at the cost of more inter-GPU communication.

```text
                  params        grads       opt states
                ┌──────────┐ ┌──────────┐ ┌──────────┐
ZeRO-1          │██████████│ │██████████│ │███░░░░░░░│  GPU 0
                │██████████│ │██████████│ │░░░███░░░░│  GPU 1
                │██████████│ │██████████│ │░░░░░░████│  GPU 2
                └──────────┘ └──────────┘ └──────────┘
                ┌──────────┐ ┌──────────┐ ┌──────────┐
ZeRO-2          │██████████│ │███░░░░░░░│ │███░░░░░░░│  GPU 0
                │██████████│ │░░░███░░░░│ │░░░███░░░░│  GPU 1
                │██████████│ │░░░░░░████│ │░░░░░░████│  GPU 2
                └──────────┘ └──────────┘ └──────────┘
                ┌──────────┐ ┌──────────┐ ┌──────────┐
ZeRO-3          │███░░░░░░░│ │███░░░░░░░│ │███░░░░░░░│  GPU 0
                │░░░███░░░░│ │░░░███░░░░│ │░░░███░░░░│  GPU 1
                │░░░░░░████│ │░░░░░░████│ │░░░░░░████│  GPU 2
                └──────────┘ └──────────┘ └──────────┘
  █ resident    ░ held on another GPU
```

ZeRO-2 shards gradients and optimizer states with lower communication overhead than ZeRO-3. Use ZeRO-3 only when your model doesn't fit across GPUs with ZeRO-2.

## Installation

Install DeepSpeed from PyPI, or install Transformers with the `deepspeed` extra.

```shell
pip install deepspeed
pip install transformers[deepspeed]
```

If you run into CUDA-related install errors, check the [DeepSpeed CUDA](./debugging#deepspeed-cuda) docs. [Installing from source](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source) is the more reliable option because it matches your exact hardware and includes features not yet available in the PyPI release.

## Configure

[Trainer](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer) integrates DeepSpeed through the [deepspeed](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments.deepspeed) argument, which accepts a JSON config file. Alternatively, use an [Accelerate config file](./accelerate#accelerate-config-file) instead of [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments).

Use `"auto"` in your config for values you want DeepSpeed to fill from [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments). If you want to explicitly specify a value, make sure you use the *same* value for both the DeepSpeed argument and [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments).

> [!NOTE]
> See the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference for a complete list of DeepSpeed config options.

```json
"train_micro_batch_size_per_gpu": "auto",  // ← per_device_train_batch_size in TrainingArguments
"gradient_accumulation_steps": "auto",     // ← gradient_accumulation_steps in TrainingArguments
"optimizer.params.lr": "auto",             // ← learning_rate in TrainingArguments
"fp16.enabled": "auto",                    // ← fp16 flag in TrainingArguments
```

Pass the config to the `deepspeed` argument.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    deepspeed="path/to/deepspeed_config.json",
    ...
)
```

```cli
# DeepSpeed launcher
deepspeed --num_gpus 4 train.py

# torchrun
torchrun --nproc_per_node 4 train.py
