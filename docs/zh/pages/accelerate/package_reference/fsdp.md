<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 完全分片数据并行实用程序

##enable_fsdp_ram_efficient_loading[[accelerate.utils.enable_fsdp_ram_efficient_loading]]

####加速.utils.enable_fsdp_ram_efficient_loading[[accelerate.utils.enable_fsdp_ram_efficient_loading]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/fsdp_utils.py#L39)

支持环境中 FSDP 的 Hugging Face 模型的 RAM 高效加载。

##disable_fsdp_ram_efficient_loading[[accelerate.utils.disable_fsdp_ram_efficient_loading]]

####加速.utils.disable_fsdp_ram_efficient_loading[[accelerate.utils.disable_fsdp_ram_efficient_loading]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/fsdp_utils.py#L49)

禁用环境中 FSDP 的 Hugging Face 模型的 RAM 高效加载。

## merge_fsdp_weights[[accelerate.utils.merge_fsdp_weights]]

####加速.utils.merge_fsdp_weights[[accelerate.utils.merge_fsdp_weights]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/fsdp_utils.py#L366)

将分片 FSDP 模型检查点的权重合并为单个组合检查点。应该使用如果
模型使用`SHARDED_STATE_DICT`。权重将被保存到`{output_path}/model.safetensors`如果
`safe_serialization` 否则`pytorch_model.bin`。

注意：这是一个 CPU 密集型进程。

**参数：**

checkpoint_dir (`str`) ：包含 FSDP 检查点的目录（可以是模型或优化器）。

output_path (`str`) ：保存合并检查点的路径。safe_serialization (`bool`，*可选*，默认为`True`)：是否使用安全张量保存合并的权重（推荐）。

remove_checkpoint_dir (`bool`, *可选*, 默认为`False`) : 合并后是否删除检查点目录。

## FulllyShardedDataParallelPlugin[[accelerate.FullyShardedDataParallelPlugin]]

####加速.FullyShardedDataParallelPlugin[[accelerate.FullyShardedDataParallelPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1584)

该插件用于启用完全分片的数据并行性。

set_auto_wrap_policyaccelerate.FullyShardedDataParallelPlugin.set_auto_wrap_policyhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2056[{"name": "model", "val": ""}]

给定`model`，根据传入的策略创建一个`auto_wrap_policy`，如果我们可以使用
`transformer_cls_to_wrap`

**参数：**

fsdp_version（`int`，默认为`1`）：要使用的 FSDP 版本。默认为 1。如果设置为 2，启动程序预计配置将转换为 FSDP2 格式。

sharding_strategy (`Union[str, torch.distributed.fsdp.ShardingStrategy]`, 默认为`'FULL_SHARD'`) : 使用的分片策略。应该是 `str` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy` 的实例。已弃用，取而代之的是`reshard_after_forward`。reshard_after_forward（`Union[str, torch.distributed.fsdp.ShardingStrategy, bool]`，`fsdp_version=1`默认为`'FULL_SHARD'`，`fsdp_version=2`默认为`True`）：要使用的分片策略。如果 `fsdp_version` 设置为 2，则应为布尔值，否则为 `str` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy` 的实例。

back_prefetch (`Union[str, torch.distributed.fsdp.BackwardPrefetch]`，默认为`'NO_PREFETCH'`) ：要使用的向后预取策略。应该是 `str` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch` 的实例。

mix_ precision_policy（`Optional[Union[dict, str, torch.distributed.fsdp.MixedPrecision, torch.distributed.fsdp.MixedPrecisionPolicy]]`，默认为`None`）：使用FullyShardedDataParallel启用混合精度训练的配置。如果传入`dict`，它应该具有以下键：`param_dtype`、`reduce_dtype`和`buffer_dtype`，如果`fsdp_version`设置为2，则可以是`torch.distributed.fsdp.MixedPrecisionPolicy`的实例。如果传入`str`，它应该是以下值之一：fp8， fp16、bf16、fp32，用于设置`param_dtype`、`reduce_dtype`、`buffer_dtype`。

auto_wrap_policy (`Optional(Union[Callable, Literal["transformer_based_wrap", "size_based_wrap", "no_wrap"]]), defaults to `NO_WRAP`) : A callable or string specifying a policy to recursively wrap layers with FSDP. If a string, it must be one of `transformer_based_wrap`, `size_based_wrap`, or `no_wrap`. See `torch.distributed.fsdp.wrap.size_based_wrap_policy`了解其外观的方向。

cpu_offload (`Union[bool, torch.distributed.fsdp.CPUOffload, torch.distributed.fsdp.CPUOffloadPolicy]`, 默认为`False`) : 是否将参数卸载到CPU。如果 `fsdp_version` 设置为 2，则应为 `bool` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffloadPolicy` 的实例。ignored_modules（`Optional[Union[Iterable[torch.nn.Module], str]]`，默认为`None`）：使用 FSDP 包装时要忽略的模块列表。传递字符串时，将使用正则表达式 fullmatch 按名称匹配模块。如果`fsdp_version`设置为2，模块将转换为参数并使用。

state_dict_type（`Union[str, torch.distributed.fsdp.StateDictType]`，默认为`'FULL_STATE_DICT'`）：要使用的状态字典类型。如果是字符串，则它必须是 `full_state_dict`、`local_state_dict` 或 `sharded_state_dict` 之一。

state_dict_config（`Optional[Union[torch.distributed.fsdp.FullStateDictConfig, torch.distributed.fsdp.ShardedStateDictConfig]`，默认为`None`）：要使用的状态字典配置。如果不传入则根据`state_dict_type`确定。

optim_state_dict_config（`Optional[Union[torch.distributed.fsdp.FullOptimStateDictConfig, torch.distributed.fsdp.ShardedOptimStateDictConfig]`，默认为`None`）：要使用的优化状态字典配置。如果不传入则根据`state_dict_type`确定。

limit_all_gathers（`bool`，默认为`True`）：是否让 FSDP 显式同步 CPU 线程以防止太多正在进行的全收集。该布尔值仅影响安排所有收集的分片策略。启用此功能有助于减少 CUDA malloc 重试次数。

use_orig_params (`bool`, 默认为`False`) : 是否使用优化器的原始参数。param_init_fn（`Optional[Callable[[torch.nn.Module], None]`，默认为`None`）：一个`Callable[torch.nn.Module] -> None`，指定当前元设备上的模块应如何初始化到实际设备上。仅当 `sync_module_states` 为 `True` 时适用。默认情况下是一个`lambda`，它在模块上调用`to_empty`。

sync_module_states（`bool`，默认为`False`）：每个单独包装的 FSDP 单元是否应从 0 级广播模块参数，以确保初始化后它们在所有等级中都相同。默认为`False`，除非`cpu_ram_efficient_loading`为`True`，则强制启用。

forward_prefetch（`bool`，默认为`False`）：在前向传递中执行时，是否让 FSDP 显式预取下一个即将到来的全收集。仅与静态图一起使用。

activation_checkpointing（`bool`，默认为`False`）：一种通过清除某些层的激活并在向后传递期间重新计算它们来减少内存使用的技术。实际上，这会用额外的计算时间来换取减少的内存使用量。cpu_ram_efficient_loading（`bool`，默认为`None`）：如果为True，则只有第一个进程加载预训练模型检查点，而所有其他进程都具有空权重。仅适用于变形金刚。使用此功能时，`sync_module_states`需要为`True`。

transformer_cls_names_to_wrap（`Optional[List[str]]`，默认为`None`）：要包装的变压器层类名称列表。仅当`auto_wrap_policy`为`transformer_based_wrap`时适用。

min_num_params（`Optional[int]`，默认为`None`）：模块必须包装的最小参数数量。仅当`auto_wrap_policy`为`size_based_wrap`时适用。
#### set_mixed_ precision[[accelerate.FullyShardedDataParallelPlugin.set_mixed_ precision]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2090)

设置 FSDP 的混合精度策略
#### set_state_dict_type[[accelerate.FullyShardedDataParallelPlugin.set_state_dict_type]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2011)

根据`StateDictType`设置状态字典配置。
#### validate_mixed_ precision_policy[[accelerate.FullyShardedDataParallelPlugin.validate_mixed_ precision_policy]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2142)

验证混合精度策略，将其抽象出来，以便在不需要时不引入导入。

## fsdp2_load_full_state_dict[[accelerate.utils.fsdp2_load_full_state_dict]]####加速.utils.fsdp2_load_full_state_dict[[accelerate.utils.fsdp2_load_full_state_dict]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/fsdp_utils.py#L467)

将完整状态字典（可能仅位于等级 0）加载到分片模型中。这是通过广播来完成的
从等级 0 到所有其他等级的参数。该函数就地修改模型。

**参数：**

Accelerator (`Accelerator`) : 加速器实例

model (`torch.nn.Module`) ：将状态字典加载到的模型，预计位于元设备上，否则可能会出现 VRAM 尖峰

full_sd (`dict`) : 要加载的完整状态字典，只能位于 0 级

cpu_offload (`bool`, 默认为`False`) ：如果为True，分配后将分片参数移至CPU。启用 FSDP CPU 卸载时需要。

## fsdp2_switch_optimizer_parameters[[accelerate.utils.fsdp2_switch_optimizer_parameters]]

####加速.utils.fsdp2_switch_optimizer_parameters[[accelerate.utils.fsdp2_switch_optimizer_parameters]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/fsdp_utils.py#L563)

将优化器的参数切换为新参数（通常情况下为分片参数）。该函数修改了
就地优化器。

**参数：**

Optimizer (`torch.optim.Optimizer`) ：包含原始模型参数的优化器实例映射 (`dict`) ：从原始参数（`data_ptr`指定）到分片参数的映射

## fsdp2_prepare_model[[accelerate.utils.fsdp2_prepare_model]]

####加速.utils.fsdp2_prepare_model[[accelerate.utils.fsdp2_prepare_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/fsdp_utils.py#L645)

就地准备 FSDP2 模型。还返回模型以避免误用原始模型。

**参数：**

Accelerator (`Accelerator`) : 加速器实例

model (`torch.nn.Module`) : 准备的模型

**退货：**

``torch.nn.Module``

准备好的模型

## fsdp2_prepare_auto_wrap_policy

### 实用函数和类
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/utilities.md