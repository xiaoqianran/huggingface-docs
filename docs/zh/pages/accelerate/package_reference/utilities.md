<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 实用函数和类

以下是 🤗 Accelerate 提供的各种实用功能，按用例细分。 

## 常量

🤗 Accelerate 中使用的常量供参考

以下是使用[Accelerator.save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state)时使用的常量

`utils.MODEL_NAME`: `"pytorch_model"`
`utils.OPTIMIZER_NAME`: `"optimizer"`
`utils.RNG_STATE_NAME`: `"random_states"`
`utils.SCALER_NAME`: `"scaler.pt`
`utils.SCHEDULER_NAME`: `"scheduler`

以下是使用[Accelerator.save_model()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_model)时使用的常量

`utils.WEIGHTS_NAME`: `"pytorch_model.bin"`
`utils.SAFE_WEIGHTS_NAME`: `"model.safetensors"`
`utils.WEIGHTS_INDEX_NAME`: `"pytorch_model.bin.index.json"`
`utils.SAFE_WEIGHTS_INDEX_NAME`: `"model.safetensors.index.json"`

## 数据类

这些是整个 🤗 Accelerate 使用的基本数据类，它们可以作为参数传入。

### 独立[[accelerate.utils.ComputeEnvironment]]

这些是用于检查的独立数据类，例如正在使用的分布式系统的类型

####加速.utils.ComputeEnvironment[[accelerate.utils.ComputeEnvironment]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L675)

代表计算环境的类型。

价值观：

- **LOCAL_MACHINE** -- 私有/自定义集群硬件。
- **AMAZON_SAGEMAKER** -- Amazon SageMaker 作为计算环境。

#### 加速.DistributedType[[加速.DistributedType]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L602)

代表一种分布式环境。

价值观：- **NO** -- 不是分布式环境，只是单个进程。
- **MULTI_CPU** -- 分布在多个CPU 节点上。
- **MULTI_GPU** -- 分布在多个 GPU 上。
- **MULTI_MLU** -- 分布在多个 MLU 上。
- **MULTI_SDAA** -- 分布在多个 SDAA 上。
- **MULTI_MUSA** -- 分布在多个 MUSA 上。
- **MULTI_NPU** -- 分布在多个 NPU 上。
- **MULTI_XPU** -- 分布在多个 XPU 上。
- **MULTI_HPU** -- 分布在多个 HPU 上。
- **MULTI_NEURON** -- 分布在多个神经元核心上。
- **DEEPSPEED** -- 使用 DeepSpeed。
- **FSDP** -- 使用完全分片数据并行性 (FSDP)。
- **XLA** -- 使用 TorchXLA。
- **MEGATRON_LM*​​ -- 使用威震天-LM。

####加速.utils.DynamoBackend[[accelerate.utils.DynamoBackend]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L690)

代表发电机后端（参见 https://pytorch.org/docs/stable/torch.compiler.html）。

价值观：- **否** -- 请勿使用割炬发电机。
- **EAGER** -- 使用 PyTorch 运行提取的 GraphModule。这在调试 TorchDynamo 时非常有用
  问题。
- **AOT_EAGER** -- 使用 AotAutograd 而不使用编译器，即仅使用 PyTorch 渴望 AotAutograd
  提取前向和后向图。这对于调试很有用，并且不太可能提供加速。
- **INDUCTOR** -- 通过利用代码生成的 Triton，将 TorchInductor 后端与 AotAutograd 和 cudagraphs 结合使用
  内核。 [Read
  more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- **AOT_TS_NVFUSER** -- 带有 AotAutograd/TorchScript 的 nvFuser。 [Read
  more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
- **NVPRIMS_NVFUSER** -- 带 PrimTorch 的 nvFuser。 [Read
  more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
- **CUDAGRAPHS** -- 使用 AotAutograd 的 cudagraph。 [Read more](https://github.com/pytorch/torchdynamo/pull/757)
- **OFI** -- 使用 Torchscript optimize_for_inference。仅供推论。 [Read
  more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
- **FX2TRT** -- 使用 Nvidia TensorRT 进行推理优化。仅供推论。 [Read
  more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
- **ONNXRT** -- 使用 ONNXRT 在 CPU/GPU 上进行推理。仅供推论。 [Read more](https://onnxruntime.ai/)
- **TENSORRT** -- 使用 ONNXRT 运行 TensorRT 进行推理优化。 [Read
  more](https://github.com/onnx/onnx-tensorrt)
- **AOT_TORCHXLA_TRACE_ONCE** -- 使用 Pytorch/XLA 和 TorchDynamo 优化进行训练。 [Read
  more](https://github.com/pytorch/xla/blob/r2.0/docs/dynamo.md)
- **TORCHXLA_TRACE_ONCE** -- 使用 Pytorch/XLA 和 TorchDynamo 优化进行推理。 [Read
  more](https://github.com/pytorch/xla/blob/r2.0/docs/dynamo.md)- **TVM** -- 使用 Apache TVM 进行推理优化。 [Read more](https://tvm.apache.org/)
- **HPU_BACKEND** -- 使用 HPU 后端进行推理优化。

####加速.utils.LoggerType[[accelerate.utils.LoggerType]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L743)

代表一种受支持的实验跟踪器

价值观：

- **ALL** -- 支持的环境中所有可用的跟踪器
- **TENSORBOARD** -- TensorBoard 作为实验跟踪器
- **WANDB** -- wandb 作为实验跟踪器
- **TRACKIO** -- trackio 作为实验跟踪器
- **COMETML** -- comet_ml 作为实验跟踪器
- **MLFLOW** -- mlflow 作为实验跟踪器
- **CLEARML** --clearml 作为实验跟踪器
- **DVCLIVE** -- dvclive 作为实验跟踪器
- **SWANLAB** -- swanlab 作为实验跟踪器

####加速.utils.PrecisionType[[accelerate.utils.PrecisionType]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L771)

表示用于浮点值的精度类型

价值观：

- **否** -- 使用全精度 (FP32)
- **FP16** -- 使用半精度
- **BF16** -- 使用大脑浮点精度

####加速.utils.RNGType[[accelerate.utils.RNGType]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L787)

一个枚举。

####加速.utils.SageMakerDistributedType[[accelerate.utils.SageMakerDistributedType]][Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L641)

代表一种分布式环境。

价值观：

- **NO** -- 不是分布式环境，只是单个进程。
- **DATA_PARALLEL** -- 使用 sagemaker 分布式数据并行性。
- **MODEL_PARALLEL** -- 使用 sagemaker 分布式模型并行性。

### Kwargs[[accelerate.AutocastKwargs]]

这些是 Accelerate 在底层处理的整个 PyTorch 生态系统中特定交互的可配置参数。

#### 加速.AutocastKwargs[[加速.AutocastKwargs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L115)

在 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义 `torch.autocast` 的行为方式。请参阅
[context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) 的文档以了解更多信息
每个论点的信息。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import AutocastKwargs

kwargs = AutocastKwargs(cache_enabled=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

#### Accelerate.DistributedDataParallelKwargs[[accelerate.DistributedDataParallelKwargs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L157)

在您的 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义模型如何包装在
`torch.nn.parallel.DistributedDataParallel`。请参考这个文档
[wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 了解更多
每个论点的信息。

`gradient_as_bucket_view` 仅在 PyTorch 1.7.0 及更高版本中可用。

`static_graph` 仅在 PyTorch 1.11.0 及更高版本中可用。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

####加速.utils.FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L457)已弃用。请使用正确的 FP8 配方 kwargs 类之一，例如 `TERecipeKwargs` 或 `MSAMPRecipeKwargs`
相反。

####加速.GradScalerKwargs[[加速.GradScalerKwargs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L243)

在您的 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义混合精度的行为，特别是如何
创建使用的`torch.amp.GradScaler`或`torch.cuda.amp.GradScaler`。请参考这个文档
[scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) 了解有关每个参数的更多信息。

`torch.cuda.amp.GradScaler` 仅在 PyTorch 1.5.0 及更高版本中可用，`torch.amp.GradScaler` 是
仅在 PyTorch 2.4.0 及更高版本中可用。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

kwargs = GradScalerKwargs(backoff_factor=0.25)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

####加速.InitProcessGroupKwargs[[加速.InitProcessGroupKwargs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L275)

在您的[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)中使用此对象来自定义分布式进程的初始化。请参考
到这个文档
[method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 了解更多
每个论点的信息。

注意：如果`timeout`设置为`None`，则默认值将基于`backend`的设置方式。

```python
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

####加速.utils.KwargsHandler[[accelerate.utils.KwargsHandler]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L70)

为数据类实现 `to_kwargs()` 方法的内部 mixin。

to_kwargsaccelerate.utils.KwargsHandler.to_kwargshttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L78[]返回一个字典，其中包含与此类的默认值不同的属性。

## 插件[[accelerate.DeepSpeedPlugin]]

这些是可以传递给 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 对象的插件。虽然它们在文档的其他地方定义， 
为了方便起见，所有这些都可以在这里查看：

####加速.DeepSpeedPlugin[[加速.DeepSpeedPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1120)

该插件用于集成 DeepSpeed。

deepspeed_config_processaccelerate.DeepSpeedPlugin.deepspeed_config_processhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1390[{"name": "prefix", "val": " = ''"}, {"name": "mismatches", "val": " = None"}, {“name”：“config”，“val”：“= None”}，{“name”：“must_match”，“val”：“= True”}，{“name”：“**kwargs”，“val”：“”}]
使用 kwargs 中的值处理 DeepSpeed 配置。

**参数：**

hf_ds_config（`Any`，默认为`None`）：DeepSpeed 配置文件或字典或类`accelerate.utils.deepspeed.HfDeepSpeedConfig` 的对象的路径。

gradient_accumulation_steps（`int`，默认为`None`）：更新优化器状态之前累积梯度的步骤数。如果未设置，将直接使用`Accelerator`中的值。gradient_clipping（`float`，默认为`None`）：启用带值的渐变裁剪。

Zero_stage（`int`，默认为`None`）：可能的选项为0、1、2、3。默认值将从环境变量中获取。

is_train_batch_min（`bool`，默认为`True`）：如果同时指定了train和eval数据加载器，这将决定`train_batch_size`。

offload_optimizer_device（`str`，默认为`None`）：可能的选项为none|cpu|nvme。仅适用于 ZeRO 第 2 阶段和第 3 阶段。

offload_param_device（`str`，默认为`None`）：可能的选项为none|cpu|nvme。仅适用于 ZeRO Stage 3。

offload_optimizer_nvme_path（`str`，默认为`None`）：可能的选项是/nvme|/local_nvme。仅适用于 ZeRO Stage 3。

offload_param_nvme_path（`str`，默认为`None`）：可能的选项为 /nvme|/local_nvme。仅适用于 ZeRO Stage 3。

Zero3_init_flag（`bool`，默认为`None`）：指示是否保存16位模型的标志。仅适用于 ZeRO Stage-3。

Zero3_save_16bit_model（`bool`，默认为`None`）：指示是否保存16位模型的标志。仅适用于 ZeRO Stage-3。Transformer_moe_cls_names（`str`，默认为`None`）：以逗号分隔的 Transformers MoE 层类名称列表（区分大小写）。例如`MixtralSparseMoeBlock`、`Qwen2MoeSparseMoeBlock`、`JetMoEAttention`、`JetMoEBlock`等。

enable_msamp（`bool`，默认为`None`）：指示是否启用 MS-AMP 后端进行 FP8 训练的标志。

msasmp_opt_level（`Optional[Literal["O1", "O2"]]`，默认为`None`）：MS-AMP 的优化级别（默认为“O1”）。仅当 `enable_msamp` 为 True 时才适用。应为 ['O1' 或 'O2'] 之一。
#### 选择[[accelerate.DeepSpeedPlugin.select]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1552)

设置 HfDeepSpeedWeakref 以使用当前的 deepspeed 插件配置

####加速.FullyShardedDataParallelPlugin[[accelerate.FullyShardedDataParallelPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1584)

该插件用于启用完全分片的数据并行性。

set_auto_wrap_policyaccelerate.FullyShardedDataParallelPlugin.set_auto_wrap_policyhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2056[{"name": "model", "val": ""}]

给定 `model`，根据传入的策略创建一个 `auto_wrap_policy`，如果我们可以使用
`transformer_cls_to_wrap`

**参数：**fsdp_version（`int`，默认为`1`）：要使用的 FSDP 版本。默认为 1。如果设置为 2，启动程序预计配置将转换为 FSDP2 格式。

sharding_strategy (`Union[str, torch.distributed.fsdp.ShardingStrategy]`, 默认为`'FULL_SHARD'`) : 使用的分片策略。应该是 `str` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy` 的实例。已弃用，取而代之的是`reshard_after_forward`。

reshard_after_forward（`Union[str, torch.distributed.fsdp.ShardingStrategy, bool]`，`fsdp_version=1`默认为`'FULL_SHARD'`，`fsdp_version=2`默认为`True`）：要使用的分片策略。如果 `fsdp_version` 设置为 2，则应为布尔值，否则为 `str` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy` 的实例。

back_prefetch (`Union[str, torch.distributed.fsdp.BackwardPrefetch]`，默认为`'NO_PREFETCH'`) ：要使用的向后预取策略。应该是 `str` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch` 的实例。

mix_ precision_policy（`Optional[Union[dict, str, torch.distributed.fsdp.MixedPrecision, torch.distributed.fsdp.MixedPrecisionPolicy]]`，默认为`None`）：使用FullyShardedDataParallel启用混合精度训练的配置。如果传入`dict`，它应该具有以下键：`param_dtype`、`reduce_dtype`和`buffer_dtype`，如果`fsdp_version`设置为2，则可以是`torch.distributed.fsdp.MixedPrecisionPolicy`的实例。如果传入`str`，它应该是以下之一值：fp8、fp16、bf16、fp32，用于设置`param_dtype`、`reduce_dtype`、`buffer_dtype`。auto_wrap_policy (`Optional(Union[Callable, Literal["transformer_based_wrap", "size_based_wrap", "no_wrap"]]), defaults to `NO_WRAP`) : A callable or string specifying a policy to recursively wrap layers with FSDP. If a string, it must be one of `transformer_based_wrap`, `size_based_wrap`, or `no_wrap`. See `torch.distributed.fsdp.wrap.size_based_wrap_policy`了解其外观的方向。

cpu_offload (`Union[bool, torch.distributed.fsdp.CPUOffload, torch.distributed.fsdp.CPUOffloadPolicy]`, 默认为`False`) : 是否将参数卸载到CPU。如果 `fsdp_version` 设置为 2，则应为 `bool` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload` 或 `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffloadPolicy` 的实例。

忽略的模块（`Optional[Union[Iterable[torch.nn.Module], str]]`，默认为`None`）：使用 FSDP 包装时要忽略的模块列表。传递字符串时，将使用正则表达式 fullmatch 按名称匹配模块。如果`fsdp_version`设置为2，模块将转换为参数并使用。

state_dict_type（`Union[str, torch.distributed.fsdp.StateDictType]`，默认为`'FULL_STATE_DICT'`）：要使用的状态字典类型。如果是字符串，则它必须是 `full_state_dict`、`local_state_dict` 或 `sharded_state_dict` 之一。

state_dict_config（`Optional[Union[torch.distributed.fsdp.FullStateDictConfig, torch.distributed.fsdp.ShardedStateDictConfig]`，默认为`None`）：要使用的状态字典配置。如果不传入则根据`state_dict_type`确定。

optim_state_dict_config（`Optional[Union[torch.distributed.fsdp.FullOptimStateDictConfig, torch.distributed.fsdp.ShardedOptimStateDictConfig]`，默认为`None`）：要使用的优化状态字典配置。如果不传入则根据`state_dict_type`确定。limit_all_gathers（`bool`，默认为`True`）：是否让 FSDP 显式同步 CPU 线程以防止太多正在进行的全收集。该布尔值仅影响安排所有收集的分片策略。启用此功能有助于减少 CUDA malloc 重试次数。

use_orig_params (`bool`, 默认为`False`) : 是否使用优化器的原始参数。

param_init_fn（`Optional[Callable[[torch.nn.Module], None]`，默认为`None`）：一个`Callable[torch.nn.Module] -> None`，指定当前元设备上的模块应如何初始化到实际设备上。仅当 `sync_module_states` 为 `True` 时适用。默认情况下是一个`lambda`，它在模块上调用`to_empty`。

sync_module_states（`bool`，默认为`False`）：每个单独包装的 FSDP 单元是否应从 0 级广播模块参数，以确保初始化后它们在所有等级中都相同。默认为`False`，除非`cpu_ram_efficient_loading`为`True`，则强制启用。

forward_prefetch（`bool`，默认为`False`）：在前向传递中执行时，是否让 FSDP 显式预取下一个即将到来的全收集。仅与静态图一起使用。activation_checkpointing（`bool`，默认为`False`）：一种通过清除某些层的激活并在向后传递期间重新计算它们来减少内存使用的技术。实际上，这会用额外的计算时间来换取减少的内存使用量。

cpu_ram_efficient_loading（`bool`，默认为`None`）：如果为True，则只有第一个进程加载预训练模型检查点，而所有其他进程都具有空权重。仅适用于变形金刚。使用此功能时，`sync_module_states`需要为`True`。

transformer_cls_names_to_wrap（`Optional[List[str]]`，默认为`None`）：要包装的变压器层类名称列表。仅当 `auto_wrap_policy` 为 `transformer_based_wrap` 时适用。

min_num_params（`Optional[int]`，默认为`None`）：模块必须包装的最小参数数量。仅当 `auto_wrap_policy` 为 `size_based_wrap` 时适用。
#### set_mixed_ precision[[accelerate.FullyShardedDataParallelPlugin.set_mixed_ precision]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2090)

设置 FSDP 的混合精度策略
#### set_state_dict_type[[accelerate.FullyShardedDataParallelPlugin.set_state_dict_type]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2011)根据`StateDictType`设置状态字典配置。
#### validate_mixed_ precision_policy[[accelerate.FullyShardedDataParallelPlugin.validate_mixed_ precision_policy]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2142)

验证混合精度策略，将其抽象出来，以便在不需要时不引入导入。

####加速.utils.GradientAccumulationPlugin[[accelerate.utils.GradientAccumulationPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L979)

用于配置梯度累积行为的插件。您只能通过`gradient_accumulation_plugin`或
`gradient_accumulation_steps` 至 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)。两者都通过会引发错误。

示例：

```python
from accelerate.utils import GradientAccumulationPlugin

gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=2)
accelerator = Accelerator(gradient_accumulation_plugin=gradient_accumulation_plugin)
```

**参数：**

num_steps (`int`) ：累积梯度的步数。

adjust_scheduler (`bool`，*可选*，默认为`True`)：是否调整调度程序步骤以考虑累积的步骤数。如果所使用的调度器没有针对梯度累积进行调整，则应为`True`。

sync_with_dataloader (`bool`，*可选*，默认为`True`)：是否在数据加载器结束时同步设置渐变。sync_each_batch (`bool`, *可选*) : 是否在每个数据批次同步设置梯度。设置为 `True` 可能会降低在分布式训练中使用梯度累积时的内存需求，但会降低速度。

####加速.utils.MegatronLMPlugin[[accelerate.utils.MegatronLMPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L2316)

Megatron-LM 插件，用于启用张量、管道、序列和数据并行性。还可以启用选择性
激活重新计算和优化的融合内核。

**参数：**

tp_ Degree (`int`, 默认为`None`) : 张量并行度。

pp_ Degree (`int`, 默认为`None`) : 管道并行度。

num_micro_batches（`int`，默认为`None`）：微批次数。

梯度裁剪（`float`，默认为`None`）：基于全局L2范数的梯度裁剪值（0表示禁用）。

equence_parallelism（`bool`，默认为`None`）：启用序列并行。

recompute_activations（`bool`，默认为`None`）：启用选择性激活重新计算。

use_distributed_optimizr（`bool`，默认为`None`）：启用分布式优化器。pipeline_model_parallel_split_rank（`int`，默认为`None`）：编码器和解码器应拆分的排名。

num_layers_per_virtual_pipeline_stage（`int`，默认为`None`）：每个虚拟管道阶段的层数。

is_train_batch_min（`str`，默认为`True`）：如果同时指定了tran和eval数据加载器，这将决定`micro_batch_size`。

train_iters（`int`，默认为`None`）：所有训练运行中要训练的样本总数。请注意，使用 `MegatronLMDummyScheduler` 时应提供 train-iters 或 train-samples。

train_samples（`int`，默认为`None`）：所有训练运行中要训练的样本总数。请注意，使用 `MegatronLMDummyScheduler` 时应提供 train-iters 或 train-samples。

weight_decay_incr_style (`str`, 默认为`'constant'`) : 权重衰减增量函数。选择=[“常数”，“线性”，“余弦”]。

start_weight_decay (`float`，默认为`None`)：L2正则化的初始权重衰减系数。

end_weight_decay（`float`，默认为`None`）：L2正则化的运行结束权重衰减系数。

lr_decay_style (`str`，默认为`'linear'`)：学习率衰减函数。选择=['常数'，'线性'，'余弦']。lr_decay_iters（`int`，默认为`None`）：学习率衰减的迭代次数。如果 None 默认为`train_iters`。

lr_decay_samples（`int`，默认为`None`）：学习率衰减的样本数。如果 None 默认为`train_samples`。

lr_warmup_iters（`int`，默认为`None`）：线性预热学习率的迭代次数。

lr_warmup_samples（`int`，默认为`None`）：线性预热学习率的样本数。

lr_warmup_fraction (`float`，默认为`None`)：lr-warmup-(iters/samples) 线性预热学习率的分数。

min_lr (`float`，默认为`0`)：学习率的最小值。调度程序剪辑值低于此阈值。

Consumer_samples（`List`，默认为`None`）：以与`accelerator.prepare`调用的数据加载器相同的顺序消耗的样本数。

no_wd_decay_cond（`Optional`，默认为`None`）：禁用权重衰减的条件。

scale_lr_cond（`Optional`，默认为`None`）：缩放学习​​率的条件。

lr_mult (`float`，默认为`1.0`)：学习率乘数。megatron_dataset_flag (`bool`, 默认为`False`) : 数据集的格式是否遵循 Megatron-LM Indexed/Cached/MemoryMapped 格式。

seq_length（`int`，默认为`None`）：要处理的最大序列长度。

encoder_seq_length（`int`，默认为`None`）：编码器要处理的最大序列长度。

解码器_seq_length（`int`，默认为`None`）：解码器要处理的最大序列长度。

tensorboard_dir（`str`，默认为`None`）：保存tensorboard日志的路径。

set_all_logging_options (`bool`, 默认为`False`) ：是否设置所有日志记录选项。

eval_iters（`int`，默认为`100`）：为评估验证/测试运行的迭代次数。

eval_interval（`int`，默认为`1000`）：在验证集上运行评估之间的间隔。

return_logits (`bool`, 默认为`False`) ：是否从模型返回logits。

custom_train_step_class（`Optional`，默认为`None`）：自定义训练步骤类。

custom_train_step_kwargs（`Optional`，默认为`None`）：自定义训练步骤kwargs。

custom_model_provider_function（`Optional`，默认为`None`）：自定义模型提供程序函数。custom_prepare_model_function (`Optional`, 默认为`None`) : 自定义准备模型函数。

custom_megatron_datasets_provider_function（`Optional`，默认为`None`）：自定义megatron train_valid_test数据集提供程序函数。

custom_get_batch_function (`Optional`, 默认为`None`) : 自定义获取批处理函数。

custom_loss_function (`Optional`, 默认为`None`) : 自定义损失函数。

other_megatron_args（`Optional`，默认为`None`）：其他 Megatron-LM 参数。请参阅威震天-LM。

####加速.utils.TorchDynamoPlugin[[accelerate.utils.TorchDynamoPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1031)

该插件用于使用 PyTorch 2.0 编译模型

**参数：**

后端（`DynamoBackend`，默认为`None`）：有效的 Dynamo 后端。有关更多详细信息，请参阅 https://pytorch.org/docs/stable/torch.compiler.html。

模式（`str`，默认为`None`）：可能的选项有“default”、“reduce-overhead”或“max-autotune”。

fullgraph (`bool`, 默认为`None`) : 是否可以将模型分成多个子图。

动态（`bool`，默认为`None`）：是否使用动态形状进行跟踪。

options (`Any`，默认为`None`)：传递到后端的选项字典。禁用（`bool`，默认为`False`）：将torch.compile（）变成无操作以进行测试

use_regional_compilation（`bool`，默认为`None`）：通过针对同一类的重复块并按顺序编译它们以命中编译器的缓存，使用它来减少 torch.compile() 的冷启动编译时间。例如，在`GPT2LMHeadModel`中，重复的块/类是`GPT2Block`，并且可以作为`model.transformer.h[0]`进行访问。模型的其余部分（例如 model.lm_head）是单独编译的。

## 配置[[accelerate.utils.BnbQuantizationConfig]]

这些是可以配置并传递到适当集成的类

####加速.utils.BnbQuantizationConfig[[accelerate.utils.BnbQuantizationConfig]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L3055)

用于启用 BitsAndBytes 4 位和 8 位量化的插件

**参数：**

load_in_8bit（`bool`，默认为`False`）：启用8位量化。

llm_int8_threshold（`float`，默认为`6.0`）：轮廓阈值。仅当`load_in_8bit=True`时相关。

load_in_4bit（`bool`，默认为`False`）：启用4bit量化。

bnb_4bit_quant_type（`str`，默认为`fp4`）：设置`bnb.nn.Linear4Bit`层中的量化数据类型。选项为 {'fp4','np4'}。bnb_4bit_use_double_quant（`bool`，默认为`False`）：启用嵌套量化，其中第一次量化的量化常数被再次量化。

bnb_4bit_compute_dtype（`bool`，默认为`fp16`）：这设置可能与输入时间不同的计算类型。例如，输入可能是 fp32，但计算可以设置为 bf16 以提高速度。选项为 {'fp32','fp16','bf16'}。

torch_dtype（`torch.dtype`，默认为`None`）：设置剩余非量化层的dtype。 `bitsandbytes` 库建议将 8 位模型的值设置为 `torch.float16`，并使用与 4 位模型的计算数据类型相同的数据类型。

Skip_modules（`List[str]`，默认为`None`）：我们不量化的模块的显式列表。这些模块的数据类型将为`torch_dtype`。

keep_in_fp32_modules（`List`，默认为`None`）：我们不量化的模块的显式列表。我们把它们保存在`torch.float32`。

####加速.DataLoaderConfiguration[[加速.DataLoaderConfiguration]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L821)

调用`accelerator.prepare`时dataloader相关项的配置。

**参数：**split_batches（`bool`，默认为`False`）：加速器是否应该跨设备拆分数据加载器生成的批次。如果`True`，实际使用的批量大小在任何类型的分布式进程上都是相同的，但它必须是您正在使用的`num_processes`的整数倍。如果`False`，实际使用的批量大小将是脚本中设置的批量大小乘以进程数。

dispatch_batches (`bool`，默认为`None`)：如果设置为`True`，则加速器准备的数据加载器仅在主进程上迭代，然后批次被拆分并广播到每个进程。对于底层数据集为 `IterableDataset` 的 `DataLoader`，将默认为 `True`，否则为 `False`。

Even_batches（`bool`，默认为`True`）：如果设置为`True`，在所有进程的总批次大小不能完全划分数据集的情况下，数据集开头的样本将被复制，以便批次可以在所有工作人员之间平均分配。use_seedable_sampler（`bool`，默认为`False`）：是否使用完全可播种的随机采样器（`data_loader.SeedableRandomSampler`）。确保使用不同的采样技术可以完全重现训练结果。虽然种子与种子之间的结果可能有所不同，但平均而言，当使用多种不同的种子进行比较时，差异可以忽略不计。还应该与 [set_seed()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.set_seed) 一起运行以获得最佳结果。

data_seed (`int`，默认为`None`)：使用`use_seedable_sampler`时用于底层生成器的种子。如果`None`，生成器将使用来自torch的当前默认种子。

non_blocking（`bool`，默认为`False`）：如果设置为`True`，加速器准备的数据加载器将利用非阻塞主机到设备传输，从而允许数据加载器通信和计算之间更好的重叠。建议准备好的数据加载器将`pin_memory`设置为`True`才能正常工作。

use_stateful_dataloader（`bool`，默认为`False`）：如果设置为`True`，则加速器准备的数据加载器将由[torchdata.StatefulDataLoader](https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader)支持。这需要安装支持StatefulDataLoader的`torchdata`版本0.8.0或更高版本。####加速.utils.ProjectConfiguration[[accelerate.utils.ProjectConfiguration]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L916)

根据项目内部需求配置加速器对象。

set_directoriesaccelerate.utils.ProjectConfiguration.set_directorieshttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L968[{"name": "project_dir", "val": ": Typing.Optional[str] = None"}]
将 `self.project_dir` 和 `self.logging_dir` 设置为适当的值。

**参数：**

project_dir (`str`，默认为`None`)：用于存储数据的目录路径。

logging_dir（`str`，默认为`None`）：用于存储本地兼容记录器日志的目录路径。如果没有，则默认为 `project_dir`。

automatic_checkpoint_naming（`bool`，默认为`False`）：是否应自动迭代命名保存的状态。

Total_limit（`int`，默认为`None`）：要保留的总保存状态的最大数量。

iteration (`int`, 默认为 `0`) : 当前保存迭代。

save_on_each_node (`bool`，默认为`False`) ：进行多节点分布式训练时，是在每个节点上保存模型和检查点，还是仅在主节点上保存。

## 环境变量这些是可以针对不同用例启用的环境变量

* `ACCELERATE_DEBUG_MODE` (`str`): 是否在调试模式下运行加速。更多信息请参见[here](../usage_guides/debug)。

## 数据操作和操作[[accelerate.utils.broadcast]]

其中包括模仿相同的`torch`操作但可以在分布式进程上使用的数据操作。

#### 加速.utils.broadcast[[accelerate.utils.broadcast]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L600)

将张量嵌套列表/元组/字典中的张量递归广播到所有设备。

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：要收集的数据。

from_process (`int`, *可选*, 默认为 0) : 发送数据的进程

**退货：**

与`tensor`相同的数据结构，所有张量都广播到正确的设备。

####加速.utils.broadcast_object_list[[accelerate.utils.broadcast_object_list]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L675)

将可腌制对象的列表从一个进程广播到其他进程。

**参数：**

object_list（可挑选对象列表）：要广播的对象列表。该列表将被就地修改。

from_process (`int`，*可选*，默认为 0) ：发送数据的进程。

**退货：**包含进程 0 中的对象的同一列表。

####加速.utils.concatenate[[accelerate.utils.concatenate]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L719)

递归连接具有相同形状的张量列表的嵌套列表/元组/字典中的张量。
如果只有一批数据，则按原样返回。

**参数：**

data（张量列表的嵌套列表/元组/字典`torch.Tensor`）：要连接的数据。

暗淡（`int`，*可选*，默认为 0）：要连接的维度。

**退货：**

与`data`相同的数据结构，所有张量都连接在一起。

####加速.utils.convert_outputs_to_fp32[[accelerate.utils.convert_outputs_to_fp32]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L939)

####加速.utils.convert_to_fp32[[accelerate.utils.convert_to_fp32]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L889)

将 FP16/BF16 精度的张量嵌套列表/元组/字典的元素递归转换为 FP32。

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：从 FP16/BF16 转换为 FP32 的数据。

**退货：**

与 `tensor` 相同的数据结构，所有 FP16/BF16 精度的张量都转换为 FP32。

####加速.utils.gather[[accelerate.utils.gather]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L424)从所有设备递归地将张量收集到张量的嵌套列表/元组/字典中。

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：要收集的数据。

**退货：**

与`tensor`相同的数据结构，所有张量都发送到正确的设备。

####加速.utils.gather_object[[accelerate.utils.gather_object]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L505)

从所有设备递归地将对象收集到对象的嵌套列表/元组/字典中。

**参数：**

object（可挑选对象的嵌套列表/元组/字典）：要收集的数据。

**退货：**

与`object`相同的数据结构，所有对象都发送到每个设备。

####加速.utils.get_grad_scaler[[accelerate.utils.get_grad_scaler]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L2110)

一个通用助手，它将根据环境初始化正确的`GradScaler`实现并返回
它。

**参数：**

Distribution_type (`DistributedType`，*可选*，默认为 None) ：分布式环境的类型。

kwargs ：所使用的 `GradScaler` 构造函数的附加参数。

####加速.utils.get_mixed_ precision_context_manager[[accelerate.utils.get_mixed_ precision_context_manager]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L2066)

返回用于自动转换混合精度的上下文管理器**参数：**

native_amp (`bool`，*可选*，默认为 False) ：是否实际启用混合精度。

cache_enabled (`bool`, *可选*, 默认为 True) : 是否启用自动投射中的权重缓存。

####加速.utils.listify[[accelerate.utils.listify]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L284)

递归地在嵌套列表/元组/字典中查找张量并将其转换为数字列表。

**参数：**

data（`torch.Tensor`的嵌套列表/元组/字典）：要转换为常规数字的数据。

**退货：**

与`data`相同的数据结构，用数字列表代替`torch.Tensor`。

####加速.utils.pad_across_processes[[accelerate.utils.pad_across_processes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L749)

将所有设备中张量的嵌套列表/元组/字典中的张量递归填充到相同的大小，以便它们
可以安全地收集。

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：要收集的数据。

暗淡（`int`，*可选*，默认为 0）：要填充的尺寸。

pad_index (`int`，*可选*，默认为 0) ：要填充的值。

pad_first (`bool`, *可选*, 默认为`False`) : 是否在开头或结尾填充。####加速.utils.recursively_apply[[accelerate.utils.recursively_apply]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L85)

在作为给定基类型的嵌套列表/元组/字典的数据结构上递归地应用函数。

**参数：**

func (`callable`) ：递归应用的函数。

data（`main_type`的嵌套列表/元组/字典）：要应用`func`的数据

- ***args** ：应用于解包数据时将传递给`func`的位置参数。

main_type (`type`，*可选*，默认为`torch.Tensor`)：应用`func`的对象的基本类型。

error_on_other_type (`bool`，*可选*，默认为`False`)：如果解压`data`后，我们得到一个不是`main_type`类型的对象，是否返回错误。如果`False`，该函数将保持与`main_type`类型不同的对象不变。

- ****kwargs** （附加关键字参数，*可选*）：应用于解包数据时将传递给 `func`​​ 的关键字参数。

**退货：**

与`data`相同的数据结构和`func`应用于`main_type`类型的每个对象。

####加速.utils.reduce[[accelerate.utils.reduce]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L845)递归地减少所有进程中张量列表的嵌套列表/元组/字典中的张量
给定操作的平均值。

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：要减少的数据。

归约（`str`，*可选*，默认为`"mean"`）：一种归约方法。可以是“平均值”、“总和”、“最大值”或“无”

scale (`float`, *可选*) : 在reduce之后应用的默认缩放值，仅在XLA上有效。

**退货：**

与`data`相同的数据结构，但所有张量都减少了。

####加速.utils.send_to_device[[accelerate.utils.send_to_device]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L136)

将张量的嵌套列表/元组/字典中的元素递归发送到给定设备。

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：发送到给定设备的数据。

device (`torch.device`) ：要将数据发送到的设备。

non_blocking（`bool`，*可选*，默认为`False`）：如果`True`，则异步执行到设备的传输，这可以将数据移动与计算重叠。仅当设备支持时有效（例如 CUDA）。Skip_keys（`str`或`List[str]`，*可选*）：字典`tensor`中的键或键列表，其值不应发送到给定的`device`。使用这些键的条目将保留在其原始设备上。

**退货：**

与`tensor`相同的数据结构，所有张量都发送到正确的设备。

####加速.utils.slice_tensors[[accelerate.utils.slice_tensors]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L699)

递归地在张量的嵌套列表/元组/字典中获取切片。

**参数：**

data（`torch.Tensor`的嵌套列表/元组/字典）：要切片的数据。

tensor_slice (`slice`) ：要拍摄的切片。

**退货：**

与 `data` 相同的数据结构，具有所有张量切片。

## 环境检查[[accelerate.utils.is_bf16_available]]

这些功能检查当前工作环境的状态，包括有关操作系统本身的信息、它可以支持的内容以及是否安装了特定的依赖项。 

####加速.utils.is_bf16_available[[accelerate.utils.is_bf16_available]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/imports.py#L164)

检查是否支持 bf16，可以选择忽略 TPU

####加速.utils.is_mps_available[[accelerate.utils.is_mps_available]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/imports.py#L334)检查 MPS 设备是否可用。所需的最低版本是 1.12。

####加速.utils.is_npu_available[[accelerate.utils.is_npu_available]]

检查是否安装了 `torch_npu` 以及环境中是否存在 NPU

####加速.utils.is_torch_version[[accelerate.utils.is_torch_version]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/versions.py#L46)

通过操作将当前 PyTorch 版本与给定引用进行比较。

**参数：**

操作 (`str`) ：运算符的字符串表示形式，例如 `">"` 或 `"<="`

version (`str`) : PyTorch 的字符串版本

####加速.utils.is_torch_xla_available[[accelerate.utils.is_torch_xla_available]]

检查`torch_xla`是否可用。要在安装了 torch xla 的环境中训练本机 pytorch 作业，请设置
将 USE_TORCH_XLA 设置为 false。

####加速.utils.is_xpu_available[[accelerate.utils.is_xpu_available]]

检查 XPU 加速是否可通过 Stock PyTorch (>=2.7) 使用
如果环境中存在 XPU，则有可能

## 环境操作[[accelerate.utils.patch_environment]]

####加速.utils.patch_environment[[accelerate.utils.patch_environment]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/environment.py#L416)一个上下文管理器，它将添加传递给 `os.environ` 的每个关键字参数，并在退出时删除它们。

将`kwargs`中的值转换为字符串并将所有键大写。

示例：

```python
>>> import os
>>> from accelerate.utils import patch_environment

>>> with patch_environment(FOO="bar"):
...     print(os.environ["FOO"])  # prints "bar"
>>> print(os.environ["FOO"])  # raises KeyError
```

####加速.utils.clear_environment[[accelerate.utils.clear_environment]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/environment.py#L381)

将暂时清除环境变量的上下文管理器。

当这个上下文退出时，之前的环境变量将会回来。

示例：

```python
>>> import os
>>> from accelerate.utils import clear_environment

>>> os.environ["FOO"] = "bar"
>>> with clear_environment():
...     print(os.environ)
...     os.environ["FOO"] = "new_bar"
...     print(os.environ["FOO"])
{}
new_bar

>>> print(os.environ["FOO"])
bar
```

####加速.commands.config.default.write_basic_config[[accelerate.commands.config.default.write_basic_config]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/commands/config/default.py#L37)

创建并保存要在可能具有多个 GPU 的本地计算机上使用的基本集群配置。还会
如果是纯 CPU 机器，请设置 CPU。

**参数：**

混合精度（`str`，*可选*，默认为“否”）：使用混合精度。应为“no”、“fp16”或“bf16”之一

save_location（`str`，*可选*，默认为`default_json_config_file`）：可选的自定义保存位置。使用`accelerate launch`时应传递给`--config_file`。默认位置位于 Huggingface 缓存文件夹 (`~/.cache/huggingface`) 内，但可以通过设置 `HF_HOME` 环境变量（后跟 `accelerate/default_config.yaml`）来覆盖。第一次设置🤗加速时，可以使用[~utils.write_basic_config]作为快速配置的替代方案，而不是运行`accelerate config`。

####加速.utils.set_numa_affinity[[accelerate.utils.set_numa_affinity]]

将当前进程分配给特定的 NUMA 节点。理想情况下，每个节点至少有 2 个 cpu 时效率最高。

该结果在调用之间被缓存。如果您想覆盖它，请使用
`accelerate.utils.environment.override_numa_afifnity`。

**参数：**

local_process_index (int) ：当前服务器上当前进程的索引。

verbose (bool, *可选*) ：是否打印每个进程的新 cpu 核心分配。如果启用`ACCELERATE_DEBUG_MODE`，则默认为 True。

####加速.utils.environment.override_numa_affinity[[accelerate.utils.environment.override_numa_affinity]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/environment.py#L298)

覆盖为当前进程设置的任何 NUMA 关联性。这是非常费力的并且需要重新计算
设置亲和力，理想情况下您应该使用 `utils.environment.set_numa_affinity` 代替。

**参数：**

local_process_index (int) ：当前服务器上当前进程的索引。

verbose (bool, *可选*) : 是否注销每个CPU的分配。如果启用`ACCELERATE_DEBUG_MODE`，则默认为 True。####加速.utils.purge_accelerate_environment[[accelerate.utils.purge_accelerate_environment]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/environment.py#L453)

装饰器用于清理加速被装饰类或函数设置的环境变量。

在某些情况下，调用某些类或函数可能会导致加速环境变量被设置而不是
之后正在清理。例如，调用时：

训练参数(fp16=True, ...)

将设置以下环境变量：

ACCELERATE_MIXED_PRECISION=fp16

这可能会影响后续代码，因为环境变量优先于 TrainingArguments(fp16=False)。这是
尤其与单元测试相关，我们希望避免单个测试对一个测试产生副作用
另一个。使用此装饰器装饰单元测试函数或整个类，以确保每次测试后，env
变量被清理。这适用于unittest.TestCase和普通类（pytest）；它也适用于当
装饰父类。

## 内存[[accelerate.find_executable_batch_size]]

####加速.find_executable_batch_size[[accelerate.find_executable_batch_size]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/memory.py#L119)一个基本的装饰器将尝试执行`function`。如果由于内存不足或相关异常而失败
CUDNN，批量大小乘以0.9，传递给`function`

`function` 必须接受 `batch_size` 参数作为其第一个参数。

示例：

```python
>>> from accelerate.utils import find_executable_batch_size

>>> @find_executable_batch_size(starting_batch_size=128)
... def train(batch_size, model, optimizer):
...     ...

>>> train(model, optimizer)
```

**参数：**

function (`callable`, *可选*) ：要包装的函数

starting_batch_size (`int`, *可选*) : 尝试适应内存的批量大小

reduce_batch_size_fn (`callable`, *可选*) ：在内存不足错误后确定新批次大小的函数。如果未提供，则每次失败时批量大小都会乘以 0.9。该函数不带参数，并且应以 `int` 的形式返回新的（减小的）批量大小。

## 建模[[accelerate.utils.calculate_maximum_sizes]]

这些实用程序与与 PyTorch 模型交互相关

####加速.utils.calculate_maximum_sizes[[accelerate.utils.calculate_maximum_sizes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1067)

计算模型及其最大层的总大小

####加速.utils.compute_module_sizes[[accelerate.utils.compute_module_sizes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L664)

计算给定模型的每个子模块的大小。####加速.utils.extract_model_from_parallel[[accelerate.utils.extract_model_from_parallel]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/other.py#L248)

从分布式容器中提取模型。

**参数：**

model (`torch.nn.Module`) ：要提取的模型。

keep_fp32_wrapper (`bool`, *可选*) : 是否从模型中删除混合精度钩子。

keep_torch_compile (`bool`, *可选*) : 是否解开编译后的模型。

recursive (`bool`，*可选*，默认为`False`) : 是否从`model`递归地提取`module.module`的所有情况，并递归地解包子子层，而不仅仅是顶级分布式容器。

**退货：**

``torch.nn.Module``

提取的模型。

####加速.utils.get_balanced_memory[[accelerate.utils.get_balanced_memory]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L931)

为 [infer_auto_device_map()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.infer_auto_device_map) 计算一个 `max_memory` 字典，这将平衡每个可用 GPU 的使用。

所有计算都是通过分析模型参数的大小和数据类型来完成的。因此，该模型可以在
元设备（就像在`init_empty_weights`上下文管理器中初始化一样）。

**参数：**

model (`torch.nn.Module`) ：要分析的模型。max_memory (`Dict`, *可选*) ：最大内存的字典设备标识符。如果未设置，将默认为最大可用内存。示例：`max_memory={0: "1GB"}`。

no_split_module_classes (`List[str]`, *可选*) ：永远不应跨设备拆分的层类名称列表（例如具有剩余连接的任何层）。

dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。

Special_dtypes (`Dict[str, Union[str, torch.device]]`, *可选*) ：如果提供，则考虑某些特定权重的特殊 dtypes（将覆盖用作所有权重默认值的 dtype）。

low_zero (`bool`, *可选*) ：最小化 GPU 0 上的权重数量，这在用于其他操作（如 Transformers 生成函数）时很方便。

####加速.utils.get_max_layer_size[[accelerate.utils.get_max_layer_size]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L718)

实用程序函数将扫描命名模块列表并返回一个完整层使用的最大大小。的
层的定义是：
- 没有直接子级的模块（只有参数和缓冲区）
- 类名在列表`no_split_module_classes`中的模块

**参数：**模块 (`List[Tuple[str, torch.nn.Module]]`) ：我们要确定最大层大小的命名模块列表。

module_sizes (`Dict[str, int]`) ：将每个层名称映射到其大小的字典（由`compute_module_sizes`生成）。

no_split_module_classes (`List[str]`) ：我们不想拆分的层的类名称列表。

**退货：**

``Tuple[int, List[str]]``

图层的最大大小以及实现该最大大小的图层名称列表。

####加速.infer_auto_device_map[[accelerate.infer_auto_device_map]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1295)

计算给定模型的设备映射，优先考虑 GPU，然后卸载到 CPU，最后卸载到磁盘，
这样：
- 我们不会超出任何 GPU 的可用内存。
- 如果需要卸载到 CPU，则 GPU 0 上始终留有空间来放回 CPU 上卸载的层
  具有最大的尺寸。
- 如果需要卸载到 CPU，我们不会超出 CPU 上可用的 RAM。
- 如果需要卸载到磁盘，CPU 上总是留有空间来放回磁盘上卸载的层
  其尺寸最大。所有计算都是通过分析模型参数的大小和数据类型来完成的。因此，该模型可以在
元设备（就像在`init_empty_weights`上下文管理器中初始化一样）。

**参数：**

model (`torch.nn.Module`) ：要分析的模型。

max_memory (`Dict`, *可选*) ：最大内存的字典设备标识符。如果未设置，将默认为最大可用内存。示例：`max_memory={0: "1GB"}`。

no_split_module_classes (`List[str]`, *可选*) ：永远不应跨设备拆分的层类名称列表（例如具有剩余连接的任何层）。

dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。

Special_dtypes (`Dict[str, Union[str, torch.device]]`, *可选*) ：如果提供，则考虑某些特定权重的特殊 dtypes（将覆盖用作所有权重默认值的 dtype）。

verbose (`bool`，*可选*，默认为`False`)：是否在函数构建 device_map 时提供调试语句。

clean_result (`bool`，*可选*，默认为`True`) ：通过将同一设备上的所有子模块分组在一起来清理生成的 device_map。offload_buffers (`bool`，*可选*，默认为`False`) ：在CPU或硬盘驱动器上卸载的层中，是否卸载缓冲区以及参数。

Fallback_allocation (`bool`，*可选*，默认为`False`)：当常规分配失败时，尝试使用 BFS 分配适合大小限制的模块。

####加速.load_checkpoint_in_model[[accelerate.load_checkpoint_in_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1805)

在模型内加载（可能是分片的）检查点，可能将权重按原样发送到给定设备
已加载。

一旦跨设备加载，您仍然需要在模型上调用[dispatch_model()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.dispatch_model)以使其能够运行。至
将检查点加载和调度分组在一个调用中，使用[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)。

**参数：**

model (`torch.nn.Module`) ：我们要在其中加载检查点的模型。

检查点（`str` 或 `os.PathLike`）：要加载的文件夹检查点。它可以是： - 包含整个模型状态字典的文件的路径 - 包含分片检查点索引的 `.json` 文件的路径 - 包含唯一 `.index.json` 文件和检查点分片的文件夹的路径。 - 包含唯一 pytorch_model.bin 或 model.safetensors 文件的文件夹的路径。device_map (`Dict[str, Union[int, str, torch.device]]`, *可选*) ：指定每个子模块应该去哪里的映射。它不需要细化到每个参数/缓冲区名称，一旦给定的模块名称在里面，它的每个子模块都会被发送到同一个设备。

offload_folder（`str`或`os.PathLike`，*可选*）：如果`device_map`包含任何值`"disk"`，我们将在其中卸载权重的文件夹。

dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。

offload_state_dict（`bool`，*可选*，默认为`False`）：如果是`True`，将暂时卸载硬盘上的CPU状态字典，以避免在CPU状态字典+最大分片的权重不适合时耗尽CPU RAM。

offload_buffers (`bool`，*可选*，默认为`False`)：是否将缓冲区包含在卸载到磁盘的权重中。

keep_in_fp32_modules(`List[str]`, *可选*) ：我们保留在 `torch.float32` dtype 中的模块列表。

offload_8bit_bnb (`bool`, *可选*) : 是否启用 cpu/磁盘上的 8 位模块卸载。strict (`bool`，*可选*，默认为`False`)：是否严格强制检查点state_dict中的键与模型state_dict的键匹配。

full_state_dict（`bool`，*可选*，默认为`True`）：如果设置为`True`，则将收集加载的state_dict中的所有张量。加载的state_dict中不会有ShardedTensor和DTensor。

Broadcast_from_rank0（`False`，*可选*，默认为`False`）：当选项为`True`时，必须初始化分布式`ProcessGroup`。 rank0应该接收一个完整的state_dict，并将state_dict中的张量一一广播到其他rank。其他级别将根据模型中的本地分片接收张量和分片（如果适用）。

####加速.utils.load_offloaded_weights[[accelerate.utils.load_offloaded_weights]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L890)

将权重从卸载文件夹加载到模型中。

**参数：**

model (`torch.nn.Module`) ：加载权重的模型。

索引 (`dict`) ：包含从模型卸载的每个参数的参数名称及其元数据的字典。

offload_folder (`str`) ：存储卸载权重的文件夹。####加速.utils.load_state_dict[[accelerate.utils.load_state_dict]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1637)

从给定文件加载检查点。如果检查点是安全张量格式并且传递了设备映射，则
权重可以直接快速加载到 GPU 上。

**参数：**

checkpoint_file (`str`) ：要加载的检查点的路径。

device_map (`Dict[str, Union[int, str, torch.device]]`, *可选*) ：指定每个子模块应该去哪里的映射。它不需要细化到每个参数/缓冲区名称，一旦给定的模块名称在里面，它的每个子模块都会被发送到同一个设备。

####加速.utils.offload_state_dict[[accelerate.utils.offload_state_dict]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/offload.py#L85)

卸载给定文件夹中的状态字典。

**参数：**

save_dir（`str`或`os.PathLike`）：卸载状态字典的目录。

state_dict (`Dict[str, torch.Tensor]`) ：要卸载的张量字典。

####加速.utils.retie_parameters[[accelerate.utils.retie_parameters]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L622)

如果链接被破坏（例如添加钩子时），则重新绑定给定模型中的参数。

**参数：**

model (`torch.nn.Module`) : 重新绑定参数的模型。tited_pa​​rams (`List[List[str]]`) ：参数名称到由`find_tied_parameters`获得的绑定参数名称的映射。

####加速.utils.set_module_tensor_to_device[[accelerate.utils.set_module_tensor_to_device]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L227)

用于在特定设备上设置模块的给定张量（缓冲区参数）的辅助函数（请注意，执行
`param.to(device)` 创建一个不链接到参数的新张量，这就是我们需要这个函数的原因）。

**参数：**

module (`torch.nn.Module`) ：我们要移动的张量所在的模块。

tensor_name (`str`) ：参数/缓冲区的全名。

device (`int`, `str` or `torch.device`) : 设置张量的设备。

value (`torch.Tensor`, *可选*) ：张量的值（从元设备到任何其他设备时很有用）。

dtype (`torch.dtype`, *可选*) ：如果传递参数的值将被转换为此`dtype`。否则，`value`将被转换为模型中现有参数的数据类型。

fp16_statistics (`torch.HalfTensor`, *可选*) : 在模块上设置的 fp16 统计信息列表，用于 8 位模型序列化。tie_params_map (Dict[int, Dict[torch.device, torch.Tensor]], *可选*, 默认为 `None`) ：指向已调度绑定权重的设备字典的当前数据指针的映射。对于给定的执行设备，此参数可用于为所有其他设备重用设备上共享权重的第一个可用指针，而不是复制内存。

non_blocking（`bool`，*可选*，默认为`False`）：如果`True`，如果可能的话，设备传输将相对于主机异步。

clear_cache (`bool`, *可选*, 默认为`True`) : 在设备上设置张量后是否清除设备缓存。

####加速.utils.get_module_children_bottom_up[[accelerate.utils.get_module_children_bottom_up]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/other.py#L566)

按自下而上的顺序遍历模型并按该顺序返回子模块。

**参数：**

model (`torch.nn.Module`) : 获取其子代的模型

**退货：**

``list[torch.nn.Module]``

`model` 的子模块列表，按自下而上的顺序排列。最后一个元素是
`model` 本身。

## 并行[[accelerate.utils.extract_model_from_parallel]]

其中包括并行工作时应使用的通用实用程序。####加速.utils.extract_model_from_parallel[[accelerate.utils.extract_model_from_parallel]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/other.py#L248)

从分布式容器中提取模型。

**参数：**

model (`torch.nn.Module`) ：要提取的模型。

keep_fp32_wrapper (`bool`, *可选*) : 是否从模型中删除混合精度钩子。

keep_torch_compile (`bool`, *可选*) : 是否解开编译后的模型。

recursive (`bool`，*可选*，默认为`False`) : 是否从`model`递归地提取`module.module`的所有情况，并递归地解包子子层，而不仅仅是顶级分布式容器。

**退货：**

``torch.nn.Module``

提取的模型。

####加速.utils.save[[accelerate.utils.save]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/other.py#L384)

将数据保存到磁盘。代替`torch.save()`使用。

**参数：**

obj：要保存的数据

f ：用于保存数据的文件（或类文件对象）

save_on_each_node (`bool`, *可选*, 默认为`False`) : 是否只保存在全局主进程上

safe_serialization（`bool`，*可选*，默认为`False`）：是否使用`safetensors`或传统的PyTorch方式（使用`pickle`）保存`obj`。

####加速.utils.load[[accelerate.utils.load]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/other.py#L434)`torch.load()` 的兼容直接替换，如果 `torch` 版本为，则可以使用 `weights_only`
2.4.0 或更高版本。否则会忽略 kwarg。

还将添加（然后删除）numpy 数组的例外

**参数：**

f ：用于加载数据的文件（或类文件对象）

map_location ：指定如何重新映射存储位置的函数、`torch.device`、字符串或字典

- ****kwargs** ：传递给 `torch.load()` 的附加关键字参数。

####加速.utils.wait_for_everyone[[accelerate.utils.wait_for_everyone]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/other.py#L336)

在脚本中引入一个阻塞点，确保所有进程在继续之前都已到达此点。

确保所有进程都会到达此指令，否则您的进程之一将永远挂起。

## 随机[[accelerate.utils.set_seed]]

这些实用程序涉及所有随机状态的设置和同步。

####加速.utils.set_seed[[accelerate.utils.set_seed]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/random.py#L40)

用于可重现行为的辅助函数，用于在 `random`、`numpy`、`torch` 中设置种子。

**参数：**

种子 (`int`) ：要设置的种子。device_specific (`bool`，*可选*，默认为`False`)：是否在每个设备上使用`self.process_index`稍微不同的种子。

确定性（`bool`，*可选*，默认为`False`）：是否在可用的情况下使用确定性算法。可以减慢训练速度。

####加速.utils.synchronize_rng_state[[accelerate.utils.synchronize_rng_state]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/random.py#L81)

####加速.synchronize_rng_states[[accelerate.synchronize_rng_states]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/random.py#L163)

## PyTorch XLA[[accelerate.utils.install_xla]]

其中包括在将 PyTorch 与 XLA 结合使用时有用的实用程序。

####加速.utils.install_xla[[accelerate.utils.install_xla]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/torch_xla.py#L20)

帮助函数根据 Google Colaboratory 中的`torch` 版本安装合适的 xla 轮子。

示例：

```python
>>> from accelerate.utils import install_xla

>>> install_xla(upgrade=True)
```

**参数：**

升级（`bool`，*可选*，默认为`False`）：是否升级`torch`并安装最新的`torch_xla`车轮。

## 加载模型权重[[accelerate.load_checkpoint_in_model]]

其中包括可用于加载检查点的实用程序。

####加速.load_checkpoint_in_model[[accelerate.load_checkpoint_in_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1805)在模型内加载（可能是分片的）检查点，可能将权重按原样发送到给定设备
已加载。

一旦跨设备加载，您仍然需要在模型上调用[dispatch_model()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.dispatch_model)以使其能够运行。至
将检查点加载和调度分组到一个调用中，使用[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)。

**参数：**

model (`torch.nn.Module`) ：我们要加载检查点的模型。

checkpoint (`str` 或 `os.PathLike`) ：要加载的文件夹检查点。它可以是： - 包含整个模型状态字典的文件的路径 - 包含分片检查点索引的 `.json` 文件的路径 - 包含唯一 `.index.json` 文件和检查点分片的文件夹的路径。 - 包含唯一 pytorch_model.bin 或 model.safetensors 文件的文件夹的路径。

device_map (`Dict[str, Union[int, str, torch.device]]`, *可选*) ：指定每个子模块应该去哪里的映射。它不需要细化到每个参数/缓冲区名称，一旦给定的模块名称在里面，它的每个子模块都会被发送到同一个设备。

offload_folder（`str`或`os.PathLike`，*可选*）：如果`device_map`包含任何值`"disk"`，我们将在其中卸载权重的文件夹。dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。

offload_state_dict（`bool`，*可选*，默认为`False`）：如果是`True`，将暂时卸载硬盘上的CPU状态字典，以避免在CPU状态字典+最大分片的权重不适合时耗尽CPU RAM。

offload_buffers（`bool`，*可选*，默认为`False`）：是否将缓冲区包含在卸载到磁盘的权重中。

keep_in_fp32_modules(`List[str]`, *可选*) ：我们保留在 `torch.float32` dtype 中的模块列表。

offload_8bit_bnb (`bool`, *可选*) : 是否启用 cpu/磁盘上的 8 位模块卸载。

strict (`bool`，*可选*，默认为`False`)：是否严格强制检查点state_dict中的键与模型state_dict的键匹配。

full_state_dict（`bool`，*可选*，默认为`True`）：如果设置为`True`，则将收集加载的state_dict中的所有张量。加载的state_dict中不会有ShardedTensor和DTensor。Broadcast_from_rank0（`False`，*可选*，默认为`False`）：当选项为`True`时，必须初始化分布式`ProcessGroup`。 rank0应该接收一个完整的state_dict，并将state_dict中的张量一一广播到其他rank。其他级别将根据模型中的本地分片接收张量和分片（如果适用）。

## 量化[[accelerate.utils.load_and_quantize_model]]

其中包括对量化模型有用的实用程序。

####加速.utils.load_and_quantize_model[[accelerate.utils.load_and_quantize_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/bnb.py#L44)

该函数将使用`bnb_quantization_config`中传递的相关配置来量化输入模型。如果
模型在元设备中，我们将根据传递的`device_map`加载和调度权重。如果
模型已经加载，我们将量化模型并将模型放在GPU上，

**参数：**

型号 (`torch.nn.Module`) ：输入型号。该模型可以已加载或在元设备上

bnb_quantization_config (`BnbQuantizationConfig`) : 位和字节量化参数weights_location（`str`或`os.PathLike`）：要加载的文件夹weights_location。它可以是： - 包含整个模型状态字典的文件的路径 - 包含分片检查点索引的 `.json` 文件的路径 - 包含唯一 `.index.json` 文件和检查点分片的文件夹的路径。 - 包含唯一 pytorch_model.bin 文件的文件夹的路径。

device_map (`Dict[str, Union[int, str, torch.device]]`, *可选*) ：指定每个子模块应该去哪里的映射。它不需要细化到每个参数/缓冲区名称，一旦给定的模块名称在里面，它的每个子模块都会被发送到同一个设备。

no_split_module_classes (`List[str]`, *可选*) ：永远不应跨设备拆分的层类名称列表（例如具有剩余连接的任何层）。

max_memory (`Dict`, *可选*) ：最大内存的字典设备标识符。如果未设置，将默认为最大可用内存。

offload_folder（`str`或`os.PathLike`，*可选*）：如果`device_map`包含任何值`"disk"`，我们将在其中卸载权重的文件夹。offload_state_dict (`bool`，*可选*，默认为`False`)：如果是`True`，将暂时卸载硬盘上的CPU状态字典，以避免在CPU状态字典+最大分片的权重不适合时耗尽CPU RAM。

**退货：**

``torch.nn.Module``

量化模型

### FSDP1 与 FSDP2
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/fsdp1_vs_fsdp2.md