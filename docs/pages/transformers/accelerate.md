# Accelerate
accelerate launch --num_processes 4 train.py
```

> [!NOTE]
> Accelerate ignores the `deepspeed` argument in [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments).

Run the [accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) command and answer questions about your hardware and training setup to create a `default_config.yaml` file in your cache.

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: path/to/ds_config.json
machine_rank: 0
num_machines: 1
num_processes: 4
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [Trainer](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.Trainer)-based script.

```shell
accelerate launch --config_file deepspeed_config.yaml train.py
```

## ZeRO stages

Select a ZeRO stage config to use as a starting point.

```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": { "stage": 1 },
    "gradient_clipping": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto"
}
```

```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "allgather_bucket_size": 2e8,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_clipping": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto"
}
```

> [!WARNING]
> ZeRO-3 shards parameters during initialization. You must instantiate [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments) before loading your model — if the model is already on each GPU before DeepSpeed is configured, no memory is saved.

```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": { "device": "cpu", "pin_memory": true },  // optional offloading
        "offload_param":     { "device": "cpu", "pin_memory": true }  // optional offloading
    },
    "gradient_clipping": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto"
}
```

The following fields are important for customizing training.

- `zero_optimization` sets the ZeRO stage.

    ```json
    { "zero_optimization": { "stage": 3 } }
    ```

- Set the batch size and gradient accumulation arguments to `"auto"`. If you manually set these to values that disagree with [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments), training continues silently with the wrong values.

    ```json
    {
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto"
    }
    ```

- `bf16` sets the training precision. Set it to `"auto"` so it mirrors the `bf16` flag in [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments).

    ```json
    { "bf16": { "enabled": "auto" } }
    ```

- `stage3_gather_16bit_weights_on_model_save` performs an all-gather across all GPUs before saving, reconstructing the full tensors from their shards. This is a ZeRO-3 argument.

    ```json
    {
        "zero_optimization": {
            "stage": 3,
            "stage3_gather_16bit_weights_on_model_save": true,
        }
    }
    ```

- Set `overlap_comm` to `true` to hide all-reduce latency behind the backward pass. `allgather_bucket_size` and `reduce_bucket_size` trade communication speed for GPU memory. Lower values use less memory but slow communication.

    ```json
    {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": true,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true
        }
    }
    ```

- `offload_optimizer` offloads the optimizer to CPU memory. To save even more memory, also offload model parameters with `offload_param` (ZeRO-3 only). Set `pin_memory` to `true` to speed up CPU-GPU transfers, but this locks RAM that is unavailable to other processes.

    ```json
    {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": { "device": "cpu", "pin_memory": true },
            "offload_param":     { "device": "cpu", "pin_memory": true }
        }
    }
    ```

- `optimizer` and `scheduler` default to the optimizer and scheduler configured in [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments). Set to `"auto"` so DeepSpeed reads the values from [TrainingArguments](/docs/transformers/v5.14.0/en/main_classes/trainer#transformers.TrainingArguments) unless you need a DeepSpeed-native optimizer like LAMB.

    ```json
    {
        "optimizer": {
            "type": "AdamW",
            "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": { "total_num_steps": "auto", "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" }
        }
    }
    ```

    If you're offloading the optimizer, set `zero_force_ds_cpu_optimizer` to `false` to use DeepSpeed's CPU Adam optimizer.

    ```json
    {
        "zero_force_ds_cpu_optimizer": false
    }
    ```

## Checkpoints

DeepSpeed saves checkpoints in a sharded format that can't be loaded directly with [from_pretrained()](/docs/transformers/v5.14.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). Set `load_best_model_at_end()` to `True` to have Trainer track and reload the best checkpoint at the end of training.

```py
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    deepspeed="ds_config_zero3.json",
    load_best_model_at_end=True,
    ...
)
# after training, save a normal transformers checkpoint
trainer.save_model("./best-model")
```

Setting `save_only_model=True` skips saving the full optimizer state, which means you can't reload the best model at the end of training. Also set `stage3_gather_16bit_weights_on_model_save: true` to reconstruct full weights from their shards. This is required for saving a consolidated 16-bit model artifact or 16-bit state dict with ZeRO-3. Transformers raises an error when `save_only_model=True` is combined with `load_best_model_at_end=True`.

> [!TIP]
> For resuming across different parallelism configurations, see DeepSpeed's [Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing) guide.

## Next steps

- Read the [Zero Redundancy Optimizer](https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero_redundancy_optimizer_(zero)) chapter from The Ultra-Scale Playbook to learn more about how ZeRO works.
- Read the ZeRO papers: [Memory Optimizations Toward Training Trillion Parameter Models](https://hf.co/papers/1910.02054), [Democratizing Billion-Scale Model Training](https://hf.co/papers/2101.06840), and [Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://hf.co/papers/2104.07857).

### Assisted decoding
https://huggingface.co/docs/transformers/v5.14.0/assisted_decoding.md
