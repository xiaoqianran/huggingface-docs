<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 深速

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 实现了[ZeRO paper](https://huggingface.co/papers/1910.02054) 中描述的所有内容。一些显着的优化是：

1.优化器状态划分（ZeRO阶段1）
2.梯度划分（ZeRO阶段2）
3.参数划分（ZeRO阶段3）
4. 定制混合精准训练操控
5.一系列基于 CUDA 扩展的快速优化器
6. 零卸载到 CPU 和磁盘/NVMe
7.模型参数的层次划分（ZeRO++）

ZeRO-Offload 有自己的专用论文：[ZeRO-Offload: Democratizing Billion-Scale Model Training](https://huggingface.co/papers/2101.06840)。论文[ZeRO-Infinity: Breaking the GPU
Memory Wall for Extreme Scale Deep Learning](https://huggingface.co/papers/2104.07857)中描述了 NVMe 支持。

DeepSpeed ZeRO-2 主要仅用于训练，因为其功能对推理没有用处。

DeepSpeed ZeRO-3 也可用于推理，因为它允许在多个 GPU 上加载大型模型，这
在单个 GPU 上不可能实现。

Accelerate 通过 2 个选项集成 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)：1. 通过 `accelerate config` 中的 `deepspeed config file` 规范集成 DeepSpeed 功能。您只需提供自定义配置文件或使用我们的模板。大部分
   本文档重点介绍此功能。这支持 DeepSpeed 的所有核心功能，并为用户提供了很大的灵活性。
   用户可能需要根据配置更改几行代码。
2. 通过 `deepspeed_plugin` 集成。这支持 DeepSpeed 功能的子集，并对其余配置使用默认选项。
   用户无需更改任何代码，对于那些熟悉 DeepSpeed 大部分默认设置的用户来说是有好处的。

## 什么是集成？

培训：

1. Accelerate 集成了 DeepSpeed ZeRO 的所有功能。这包括所有 ZeRO 阶段 1、2 和 3 以及 ZeRO-Offload、ZeRO-Infinity（可以卸载到磁盘/NVMe）和 ZeRO++。
下面是使用 ZeRO - 零冗余优化器的数据并行性的简短描述以及来自此[blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)的图表
![ZeRO Data Parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

（来源：[link](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)）

 a. **第 1 阶段**：跨数据并行工作线程/GPU 的分片优化器状态

 b. **第 2 阶段**：分片优化器状态 + 跨数据并行工作线程/GPU 的梯度c. **第 3 阶段**：跨数据并行工作线程/GPU 的分片优化器状态 + 梯度 + 模型参数

 d. **优化器卸载**：将梯度 + 优化器状态卸载到构建在 ZERO Stage 2 之上的 CPU/磁盘

 e. **参数卸载**：将模型参数卸载到构建在 ZERO Stage 3 之上的 CPU/磁盘

 f. **分层分区**：通过跨节点的数据并行训练和节点内的 ZeRO-3 分片（构建在 ZeRO Stage 3 之上），实现高效的多节点训练。

注意：关于磁盘卸载，磁盘应该是 NVME，以获得不错的速度，但从技术上讲，它可以在任何磁盘上工作

推论：

1. DeepSpeed ZeRO Inference 通过 ZeRO-Infinity 支持 ZeRO 第 3 阶段。它使用与训练相同的 ZeRO 协议，但是
   它不使用优化器和 lr 调度程序，并且只有第 3 阶段相关。欲了解更多详情，请参阅：
   [deepspeed-zero-inference](#deepspeed-zero-inference)。

## 它是如何工作的？

**先决条件**：安装 DeepSpeed 版本 >=0.6.5。请参考[DeepSpeed Installation details](https://github.com/deepspeedai/DeepSpeed#installation)
了解更多信息。

我们将首先看看通过 `accelerate config` 进行的易于使用的集成。
其次是更灵活、功能丰富的`deepspeed config file`集成。

### 加速 DeepSpeed 插件
在你的机器上运行：

```bash
accelerate config
```并回答所提出的问题。它会询问您是否要使用 DeepSpeed 的配置文件，您应该回答“否”。然后回答以下问题以生成基本的 DeepSpeed 配置。
这将生成一个配置文件，该文件将自动用于正确设置
执行时的默认选项

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，以下是如何使用 DeepSpeed 插件运行 NLP 示例 `examples/nlp_example.py` （来自存储库的根目录）：

**ZeRO Stage-2 DeepSpeed 插件示例**
```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
 gradient_accumulation_steps: 1
 gradient_clipping: 1.0
 offload_optimizer_device: none
 offload_param_device: none
 zero3_init_flag: true
 zero_stage: 2
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

**具有 CPU 卸载 DeepSpeed 插件示例的 ZeRO Stage-3**
```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

目前，`Accelerate`支持通过CLI进行以下配置：

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_optimizer_nvme_path`: Decides Nvme Path to offload optimizer states. If unspecified, will default to 'none'.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`offload_param_nvme_path`: Decides Nvme Path to offload parameters. If unspecified, will default to 'none'.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training.
`deepspeed_moe_layer_cls_names`: Comma-separated list of transformer Mixture-of-Experts (MoE) layer class names (case-sensitive) to wrap ,e.g, `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `JetMoEAttention,JetMoEBlock` ...
`deepspeed_hostfile`: DeepSpeed hostfile for configuring multi-node compute resources.
`deepspeed_exclusion_filter`: DeepSpeed exclusion filter string when using mutli-node setup.
`deepspeed_inclusion_filter`: DeepSpeed inclusion filter string when using mutli-node setup.
`deepspeed_multinode_launcher`: DeepSpeed multi-node launcher to use, e.g. `pdsh`, `standard`, `openmpi`, `mvapich`, `mpich`, `slurm`, `nossh` (requires DeepSpeed >= 0.14.5). If unspecified, will default to `pdsh`.
`deepspeed_config_file`: path to the DeepSpeed config file in `json` format. See the next section for more details on this.
```
为了能够调整更多选项，您将需要使用 DeepSpeed 配置文件。

### DeepSpeed 配置文件
在你的机器上运行：

```bash
accelerate config
```

并回答所提出的问题。它会询问您是否要使用 Deepspeed 的配置文件，您回答“是”
并提供 deepspeed 配置文件的路径。
这将生成一个配置文件，该文件将自动用于正确设置
执行时的默认选项

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，以下是如何使用 DeepSpeed 配置文件运行 NLP 示例 `examples/by_feature/deepspeed_with_config_support.py`（从存储库的根目录）：**ZeRO Stage-2 DeepSpeed 配置文件示例**
```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
 deepspeed_config_file: /home/ubuntu/accelerate/examples/deepspeed_config_templates/zero_stage2_config.json
 zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

`zero_stage2_config.json`的内容为：
```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": true,
            "adam_w_mode": true
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

```bash
accelerate launch examples/by_feature/deepspeed_with_config_support.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "./clm/clm_deepspeed_stage2_accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 3 \
--with_tracking \
--report_to "wandb"\
```

**具有 CPU 卸载 DeepSpeed 配置文件示例的 ZeRO Stage-3**
```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
 deepspeed_config_file: /home/ubuntu/accelerate/examples/deepspeed_config_templates/zero_stage3_offload_config.json
 zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```
`zero_stage3_offload_config.json`的内容为：
```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": "auto"
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

```bash
accelerate launch examples/by_feature/deepspeed_with_config_support.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "./clm/clm_deepspeed_stage3_offload_accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--num_train_epochs 3 \
--with_tracking \
--report_to "wandb"\
```

**ZeRO++ 配置示例**
您可以通过使用适当的配置参数来使用 ZeRO++ 的功能。请注意，ZeRO++ 是 ZeRO Stage 3 的扩展。以下是修改配置文件的方法，来自 [DeepSpeed's ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)：

```json
{
    "zero_optimization": {
        "stage": 3,
        "reduce_bucket_size": "auto",

        "zero_quantized_weights": true,
        "zero_hpz_partition_size": 8,
        "zero_quantized_gradients": true,

        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

对于分层分区，分区大小 `zero_hpz_partition_size` 理想情况下应设置为每个节点的 GPU 数量。 （例如，上述配置文件假设每个节点有 8 个 GPU）

**使用 DeepSpeed 配置文件时的重要代码更改**

1. DeepSpeed 优化器和调度器。有关这些的更多信息，
请参阅 [DeepSpeed Optimizers](https://deepspeed.readthedocs.io/en/latest/optimizers.html) 和 [DeepSpeed Schedulers](https://deepspeed.readthedocs.io/en/latest/schedulers.html) 文档。
我们将查看使用这些时代码中需要的更改。一个。 DS Optim + DS Scheduler：当 DeepSpeed 配置文件中同时存在 `optimizer` 和 `scheduler` 键时的情况。
   在这种情况下，将使用这些，用户必须使用 `accelerate.utils.DummyOptim` 和 `accelerate.utils.DummyScheduler` 来替换代码中的 PyTorch/自定义优化器和调度器。
   以下是 `examples/by_feature/deepspeed_with_config_support.py` 的片段，显示了这一点：
   ```python
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )
   ```
   b.自定义 Optim + 自定义调度程序：DeepSpeed 配置文件中缺少 `optimizer` 和 `scheduler` 键的情况。
   在这种情况下，用户无需更改代码，通过 DeepSpeed 插件使用集成时就是这种情况。
   在上面的示例中，我们可以看到，如果 DeepSpeed 配置文件中不存在 `optimizer` 和 `scheduler` 键，则代码保持不变。

   c.自定义 Optim + DS Scheduler：DeepSpeed 配置文件中仅存在 `scheduler` 密钥的情况。
   在这种情况下，用户必须使用 `accelerate.utils.DummyScheduler` 替换代码中的 PyTorch/Custom 调度程序。

   d. DS Optim + 自定义调度程序：DeepSpeed 配置文件中仅存在 `optimizer` 密钥的情况。
   这将导致错误，因为您只能在使用 DS Optim 时使用 DS Scheduler。2. 请注意上述示例 DeepSpeed 配置文件中的 `auto` 值。这些由`prepare`方法自动处理
基于提供给`prepare`方法的模型、数据加载器、虚拟优化器和虚拟调度器。
只有上面示例中指定的 `auto` 字段由 `prepare` 方法处理，其余字段必须由用户显式指定。

`auto` 值计算如下：

- `reduce_bucket_size`：`hidden_size * hidden_size`
- `stage3_prefetch_bucket_size`：`int(0.9 * hidden_size * hidden_size)`
- `stage3_param_persistence_threshold`：`10 * hidden_size`

为了使 `auto` 功能适用于这 3 个配置条目 - Accelerate 将使用 `model.config.hidden_size` 或 `max(model.config.hidden_sizes)` 作为 `hidden_size`。如果这些都不可用，则启动将失败，您必须手动设置这 3 个配置条目。请记住，前 2 个配置条目是通信缓冲区 - 它们越大，通信效率越高，并且它们越大，消耗的 GPU 内存就越多，因此这是一个可调的性能权衡。

**使用 DeepSpeed 配置文件时的注意事项**

下面是在不同场景下使用`deepspeed_config_file`的示例脚本。

代码`test.py`：

```python
from accelerate import Accelerator
from accelerate.state import AcceleratorState

def main():
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")

if __name__ == "__main__":
    main()
```

**场景 1**：手动篡改加速配置文件，其中包含 `deepspeed_config_file` 以及其他条目。

1. `accelerate`配置内容：

```yaml
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: 'cpu'
  offload_param_device: 'cpu'
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
  deepspeed_config_file: 'ds_config.json'
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: null
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
```

2.`ds_config.json`：

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": false,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        }
    },
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": 10,
    "steps_per_print": 2000000
}
```

3、`accelerate launch test.py`的输出：```bash
ValueError: When using `deepspeed_config_file`, the following accelerate config variables will be ignored:
['gradient_accumulation_steps', 'gradient_clipping', 'zero_stage', 'offload_optimizer_device', 'offload_param_device',
'zero3_save_16bit_model', 'mixed_precision'].
Please specify them appropriately in the DeepSpeed config file.
If you are using an accelerate config file, remove other config variables mentioned in the above specified list.
The easiest method is to create a new config following the questionnaire via `accelerate config`.
It will only ask for the necessary config variables when using `deepspeed_config_file`.
```

**场景 2**：使用错误的解决方案创建新的加速配置并检查现在是否抛出不明确的错误。

1.运行`accelerate config`：

```bash
$ accelerate config
-------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes
Please enter the path to the json DeepSpeed config file: ds_config.json
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
How many GPU(s) should be used for distributed training? [1]:4
accelerate configuration saved at ds_config_sample.yaml
```

2. `accelerate`配置内容：

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: ds_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
```

3、`accelerate launch test.py`的输出：

```bash
Distributed environment: DEEPSPEED  Backend: nccl
Num processes: 4
Process index: 0
Local process index: 0
Device: cuda:0
Mixed precision type: bf16
ds_config: {'bf16': {'enabled': True}, 'zero_optimization': {'stage': 3, 'stage3_gather_16bit_weights_on_model_save': False, 'offload_optimizer': {'device': 'none'}, 'offload_param': {'device': 'none'}}, 'gradient_clipping': 1.0, 'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': 10, 'steps_per_print': inf, 'fp16': {'enabled': False}}
```

**场景 3**：在 DeepSpeed 配置文件中将与 DeepSpeed 相关的 `accelerate launch` 命令参数设置为 `"auto"`，并检查是否按预期工作。

1. 新的 `ds_config.json` 和 `"auto"` 用于 `accelerate launch` DeepSpeed 命令参数：

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": "auto",
        "stage3_gather_16bit_weights_on_model_save": "auto",
        "offload_optimizer": {
            "device": "auto"
        },
        "offload_param": {
            "device": "auto"
        }
    },
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "steps_per_print": 2000000
}
```

2、`accelerate launch --mixed_precision="fp16" --zero_stage=3 --gradient_accumulation_steps=5 --gradient_clipping=1.0 --offload_param_device="cpu" --offload_optimizer_device="nvme" --zero3_save_16bit_model="true" test.py`输出：

```bash
Distributed environment: DEEPSPEED  Backend: nccl
Num processes: 4
Process index: 0
Local process index: 0
Device: cuda:0
Mixed precision type: fp16
ds_config: {'bf16': {'enabled': False}, 'zero_optimization': {'stage': 3, 'stage3_gather_16bit_weights_on_model_save': True, 'offload_optimizer': {'device': 'nvme'}, 'offload_param': {'device': 'cpu'}}, 'gradient_clipping': 1.0, 'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': 5, 'steps_per_print': inf, 'fp16': {'enabled': True, 'auto_cast': True}}
```

**注意**：
1. 剩余的 `"auto"` 值在 `accelerator.prepare()` 调用中处理，如第 2 点所述
`Important code changes when using DeepSpeed Config File`。
2. 仅当`gradient_accumulation_steps`为`auto`时，才会使用通过`Accelerator(gradient_accumulation_steps=k)`创建`Accelerator`对象时传递的值。使用 DeepSpeed Plugin 时，将使用其中的值，并将覆盖创建 Accelerator 对象时传递的值。

## 保存和加载

1. ZeRO Stage-1 和 Stage-2 的模型保存和加载没有变化。

2. 在 ZeRO Stage-3 下，`state_dict` 仅包含占位符，因为模型权重分布在多个 GPU 上。
ZeRO Stage-3 有 2 个选项：一个。保存整个 16 位模型权重，以便稍后使用`model.load_state_dict(torch.load(pytorch_model.bin))`直接加载。
   为此，请在 DeepSpeed 配置文件中将 `zero_optimization.stage3_gather_16bit_weights_on_model_save` 设置为 True 或设置
   DeepSpeed 插件中的`zero3_save_16bit_model` 为 True。
   **请注意，此选项需要将权重整合到一个 GPU 上，这可能会很慢并且需要内存，因此仅在需要时使用此功能。**
   下面是 `examples/by_feature/deepspeed_with_config_support.py` 的片段，显示了这一点：
   ```python
   unwrapped_model = accelerator.unwrap_model(model)

   # New Code #
   # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
   # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
   # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
   # For Zero Stages 1 and 2, models are saved as usual in the output directory.
   # The model name saved is `pytorch_model.bin`
   unwrapped_model.save_pretrained(
       args.output_dir,
       is_main_process=accelerator.is_main_process,
       save_function=accelerator.save,
       state_dict=accelerator.get_state_dict(model),
   )
   ```

   b.要获得 32 位权重，首先使用 `model.save_checkpoint()` 保存模型。
   以下是 `examples/by_feature/deepspeed_with_config_support.py` 的片段，显示了这一点：
   ```python
   success = model.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)
   status_msg = f"checkpointing: PATH={PATH}, ckpt_id={ckpt_id}"
   if success:
       logging.info(f"Success {status_msg}")
   else:
       logging.warning(f"Failure {status_msg}")
   ```
   这将在检查点目录中创建 ZeRO 模型和优化器分区以及 `zero_to_fp32.py` 脚本。
   您可以使用此脚本进行离线整合。
   它不需要配置文件或 GPU。下面是它的用法示例：
   ```bash
   $ cd /path/to/checkpoint_dir
   $ ./zero_to_fp32.py . pytorch_model.bin
   Processing zero checkpoint at global_step1
   Detected checkpoint of type zero stage 3, world_size: 2
   Saving fp32 state dict to pytorch_model.bin (total_numel=60506624)
   ```
   要获取 32 位模型用于保存/推理，您可以执行：
   ```python
   from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

   unwrapped_model = accelerator.unwrap_model(model)
   fp32_model = load_state_dict_from_zero_checkpoint(unwrapped_model, checkpoint_dir)
   ```
   如果您只对`state_dict`感兴趣，您可以执行以下操作：
   ```python
   from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

   state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
   ```
   请注意，所有这些函数都需要最终检查点大小的约 2 倍内存（一般 RAM）。## 零推理
DeepSpeed ZeRO Inference 通过 ZeRO-Infinity 支持 ZeRO 第 3 阶段。
它使用与训练相同的 ZeRO 协议，但不使用优化器和 lr 调度器，并且只有第 3 阶段相关。
通过加速集成，您只需准备模型和数据加载器，如下所示：

```python
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
```

## 需要注意的一些注意事项

1.当前集成不支持DeepSpeed的Pipeline Parallelism。
2. 当前集成不支持`mpu`，限制了Megatron-LM支持的张量并行性。
3. 目前集成不支持多模型。

## 多节点 DeepSpeed
DeepSpeed 支持通过各种不同的启动器进行多节点推理和训练。您可以通过在 CLI 或 DeepSpeed 配置文件中设置 `deepspeed_multinode_launcher` 配置来指定不同的启动器。

目前，加速支持传递以下 DeepSpeed 多节点启动器的配置：`pdsh`（默认）、`standard`、`openmpi`、`mvapich`、`mpich`、`slurm`、`nossh`（需要 DeepSpeed >= 0.14.5）。请阅读[DeepSpeed documentation](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)以获取有关不同启动器的更多信息。默认情况下，DeepSpeed 将尝试使用无密码 SSH 从主机节点到其他节点来执行启动器命令。在此配置中，只需在主节点上运行加速启动命令。如果使用 `nossh` 启动器，您将需要使用复制的配置在每个节点上运行加速启动命令。 

## DeepSpeed 资源

与deepspeed相关的内部文档可以找到[here](../package_reference/deepspeed)。

- [Project's github](https://github.com/deepspeedai/DeepSpeed)
- [Usage docs](https://www.deepspeed.ai/getting-started/)
- [API docs](https://deepspeed.readthedocs.io/en/latest/index.html)
- [Blog posts](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

论文：

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://huggingface.co/papers/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://huggingface.co/papers/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://huggingface.co/papers/2104.07857)
- [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://huggingface.co/papers/2306.10209)

最后，请记住`Accelerate`仅集成了DeepSpeed，因此如果您
如果对 DeepSpeed 使用有任何问题或疑问，请通过 [DeepSpeed GitHub](https://github.com/deepspeedai/DeepSpeed/issues) 提出问题。

    对 FSDP 和 DeepSpeed 的异同感兴趣的朋友，请查看[concept guide here](../concept_guides/fsdp_and_deepspeed)！

### 从这里开始！
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/explore.md