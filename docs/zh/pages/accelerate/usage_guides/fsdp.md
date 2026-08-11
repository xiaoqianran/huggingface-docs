<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 完全分片数据并行

为了加速在更大批量上训练大型模型，我们可以使用完全分片的数据并行模型。
这种类型的数据并行范例可以通过对优化器状态、梯度和参数进行分片来拟合更多数据和更大的模型。
要了解有关它及其优点的更多信息，请查看 [Fully Sharded Data Parallel blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)。
我们集成了最新的 PyTorch 的完全分片数据并行 (FSDP) 训练功能。
您所需要做的就是通过配置启用它。

## 它是如何开箱即用的

在你的机器上运行：

```bash
accelerate config
```

并回答所提出的问题。这将生成一个配置文件，该文件将自动用于正确设置
执行时的默认选项

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，以下是在启用 FSDP 的情况下运行 `examples/nlp_example.py` （从存储库的根目录）的方法：

```bash
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py
```

目前，`Accelerate`通过CLI支持以下配置：`fsdp_sharding_strategy`：[1] FULL_SHARD（分片优化器状态、梯度和参数）、[2] SHARD_GRAD_OP（分片优化器状态和梯度）、[3] NO_SHARD (DDP)、[4] HYBRID_SHARD（每个节点内的分片优化器状态、梯度和参数，每个节点都有完整副本）、[5] HYBRID_SHARD_ZERO2 （在每个节点内对优化器状态和梯度进行分片，同时每个节点都有完整副本）。更多信息请参考官方[PyTorch docs](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)。

`fsdp_offload_params` : 决定是否将参数和梯度卸载到CPU

`fsdp_auto_wrap_policy`：[1] TRANSFORMER_BASED_WRAP，[2] SIZE_BASED_WRAP，[3] NO_WRAP`fsdp_transformer_layer_cls_to_wrap`：仅适用于变形金刚。当使用`fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP`时，用户可以提供以逗号分隔的变压器层类名称字符串（区分大小写）进行换行，例如`BertLayer`、`GPTJBlock`、`T5Block`、`BertLayer,BertEmbeddings,BertSelfOutput`。这很重要，因为共享权重的子模块（例如嵌入层）不应最终出现在不同的 FSDP 包装单元中。使用此策略，每个包含多头注意力的块都会发生包装，后跟几个 MLP 层。包括共享嵌入在内的其余层可以方便地包装在相同的最外层 FSDP 单元中。因此，将其用于基于变压器的模型。如果可能，您可以通过将 `yes` 回答为 `Do you want to use the model's `_no_split_modules` to wrap. It will try to use `model._no_split_modules`，将 `model._no_split_modules` 用于 Transformer 模型。

`fsdp_min_num_params`：使用`fsdp_auto_wrap_policy=SIZE_BASED_WRAP`时的最小参数数量。

`fsdp_backward_prefetch_policy`：[1] BACKWARD_PRE，[2] BACKWARD_POST，[3] NO_PREFETCH

`fsdp_forward_prefetch`：如果为 True，则 FSDP 在前向传递中执行时显式预取下一个即将到来的全收集。只能用于静态图模型，因为预取遵循第一次迭代的执行顺序。即，如果子模块的顺序在模型执行期间动态变化，则不要启用此功能。`fsdp_state_dict_type`：[1] FULL_STATE_DICT，[2] LOCAL_STATE_DICT，[3] SHARDED_STATE_DICT

`fsdp_use_orig_params`：如果为True，则在初始化期间允许非均匀`requires_grad`，这意味着支持散布的冻结和可训练参数。此设置在[this post](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019)中讨论的参数高效微调等情况下非常有用。此选项还允许拥有多个优化器参数组。在使用 FSDP 准备/包装模型之前创建优化器时，这应该是`True`。

`fsdp_cpu_ram_efficient_loading`：仅适用于变形金刚型号。如果为 True，则只有第一个进程加载预训练模型检查点，而所有其他进程都具有空权重。如果您在通过 `from_pretrained` 方法加载预训练的 Transformers 模型时遇到错误，则应将其设置为 False。当此设置为 True 时，`fsdp_sync_module_states` 也必须为 True，否则除主进程外的所有进程都会具有随机权重，导致训练期间出现意外行为。为此，请确保在调用 Transformers `from_pretrained` 方法之前初始化分布式进程组。使用 Trainer API 时，分布式进程组会在您创建 `TrainingArguments` 类的实例时初始化。`fsdp_sync_module_states`：如果为真，则每个独立包装的 FSDP 单元将广播 0 级的模块参数。

如需额外且更细致的控制，您可以通过 `FullyShardedDataParallelPlugin` 指定其他 FSDP 参数。
创建 `FullyShardedDataParallelPlugin` 对象时，请向其传递不属于加速配置的参数，或者如果您想覆盖它们。
FSDP 参数将根据加速配置文件或启动命令参数进行选择，您将直接通过 `FullyShardedDataParallelPlugin` 对象传递的其他参数将设置/覆盖它。

下面是一个例子：

```py
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

## 保存和加载

使用 FSDP 模型时建议的新检查点方法是在设置加速配置时使用 `SHARDED_STATE_DICT` 作为 `StateDictType`。
下面是使用加速实用程序`save_state`保存的代码片段。

```py
accelerator.save_state("ckpt")
```

检查检查点文件夹以将模型和优化器视为每个进程的分片：
```
ls ckpt
# optimizer_0  pytorch_model_0  random_states_0.pkl  random_states_1.pkl  scheduler.bin

cd ckpt

ls optimizer_0
# __0_0.distcp  __1_0.distcp

ls pytorch_model_0
# __0_0.distcp  __1_0.distcp
```

要加载它们以恢复训练，请使用加速实用程序`load_state`

```py
accelerator.load_state("ckpt")
```

当使用变压器`save_pretrained`时，通过`state_dict=accelerator.get_state_dict(model)`来保存模型状态字典。
  下面是一个例子：

```diff
  unwrapped_model.save_pretrained(
      args.output_dir,
      is_main_process=accelerator.is_main_process,
      save_function=accelerator.save,
+     state_dict=accelerator.get_state_dict(model),
)
```

### 状态字典`accelerator.get_state_dict` 将使用 `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` 上下文管理器调用底层 `model.state_dict` 实现，以仅获取排名 0 的状态字典，并将其卸载到 CPU。

然后，您可以将 `state` 传递到 `save_pretrained` 方法中。  `StateDictType` 和 `FullStateDictConfig` 有多种模式，您可以使用它们来控制 `state_dict` 的行为。  有关更多信息，请参阅[PyTorch documentation](https://pytorch.org/docs/stable/fsdp.html)。

如果您选择使用`StateDictType.SHARDED_STATE_DICT`，`Accelerator.save_state`期间模型的权重将被拆分为`n`文件，用于模型上的每个子拆分。将它们合并回
训练后稍后加载回模型的单个字典可以使用 `merge_weights` 实用程序：

```py
from accelerate.utils import merge_fsdp_weights

# Our weights are saved usually in a `pytorch_model_fsdp_{model_number}` folder
merge_fsdp_weights("pytorch_model_fsdp_0", "output_path", safe_serialization=True)
```
最终输出将被保存到`model.safetensors`或`pytorch_model.bin`（如果`safe_serialization=False`被传递）。 

也可以使用 CLI 调用：
```bash
accelerate merge-weights pytorch_model_fsdp_0/ output_path
```## FSDP 分片策略与 DeepSpeed ZeRO Stage 之间的映射
* `FULL_SHARD` 映射到 DeepSpeed `ZeRO Stage-3`。分片优化器状态、梯度和参数。
* `SHARD_GRAD_OP` 映射到 DeepSpeed `ZeRO Stage-2`。分片优化器状态和梯度。
* `NO_SHARD` 映射到 `ZeRO Stage-0`。无分片，其中每个 GPU 都有模型、优化器状态和梯度的完整副本。
* `HYBRID_SHARD` 映射到`ZeRO++ Stage-3`，其中`zero_hpz_partition_size=<num_gpus_per_node>`。在这里，这将在每个节点内对优化器状态、梯度和参数进行分片，同时每个节点都有完整的副本。

## 需要注意的一些注意事项

- 如果有多个模型，请按照与相应模型相同的顺序将优化器传递给准备调用，否则 `accelerator.save_state()` 和 `accelerator.load_state()` 将导致错误/意外的行为。
- 此功能与`Transformers`库的`run_translation.py`脚本中的`--predict_with_generate`不兼容。

为了获得更多控制，用户可以利用`FullyShardedDataParallelPlugin`。创建此类的实例后，用户可以将其传递给 Accelerator 类实例化。
有关这些选项的更多信息，请参阅 PyTorch [FullyShardedDataParallel](https://github.com/pytorch/pytorch/blob/0df2e863fbd5993a7b9e652910792bd21a516ff3/torch/distributed/fsdp/fully_sharded_data_parallel.py#L236) 代码。

    对 FSDP 和 DeepSpeed 的异同感兴趣的朋友，请查看[concept guide here](../concept_guides/fsdp_and_deepspeed)！### Mac 上的加速 PyTorch 训练
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/mps.md