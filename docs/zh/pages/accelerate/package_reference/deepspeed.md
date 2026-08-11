<!-- huggingface-docs: machine-translated zh-CN from English source -->

# DeepSpeed 实用程序

## DeepSpeed 插件

## get_active_deepspeed_plugin[[accelerate.utils.get_active_deepspeed_plugin]]

####加速.utils.get_active_deepspeed_plugin[[accelerate.utils.get_active_deepspeed_plugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L100)

返回当前活动的 DeepSpeedPlugin。

####加速.DeepSpeedPlugin[[加速.DeepSpeedPlugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1120)

该插件用于集成 DeepSpeed。

deepspeed_config_processaccelerate.DeepSpeedPlugin.deepspeed_config_processhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1390[{"name": "prefix", "val": " = ''"}, {"name": "mismatches", "val": " = None"}, {“name”：“config”，“val”：“= None”}，{“name”：“must_match”，“val”：“= True”}，{“name”：“**kwargs”，“val”：“”}]
使用 kwargs 中的值处理 DeepSpeed 配置。

**参数：**

hf_ds_config（`Any`，默认为`None`）：DeepSpeed 配置文件或字典或`accelerate.utils.deepspeed.HfDeepSpeedConfig`类对象的路径。

gradient_accumulation_steps（`int`，默认为`None`）：更新优化器状态之前累积梯度的步骤数。如果未设置，将直接使用`Accelerator`中的值。gradient_clipping (`float`, 默认为`None`) : 启用带值的渐变裁剪。

Zero_stage（`int`，默认为`None`）：可能的选项为0、1、2、3。默认值将从环境变量中获取。

is_train_batch_min（`bool`，默认为`True`）：如果同时指定了train和eval数据加载器，这将决定`train_batch_size`。

offload_optimizer_device（`str`，默认为`None`）：可能的选项为none|cpu|nvme。仅适用于 ZeRO 第 2 阶段和第 3 阶段。

offload_param_device（`str`，默认为`None`）：可能的选项为none|cpu|nvme。仅适用于 ZeRO Stage 3。

offload_optimizer_nvme_path（`str`，默认为`None`）：可能的选项为 /nvme|/local_nvme。仅适用于 ZeRO Stage 3。

offload_param_nvme_path（`str`，默认为`None`）：可能的选项为 /nvme|/local_nvme。仅适用于 ZeRO Stage 3。

Zero3_init_flag（`bool`，默认为`None`）：指示是否保存16位模型的标志。仅适用于 ZeRO Stage-3。

Zero3_save_16bit_model（`bool`，默认为`None`）：指示是否保存16位模型的标志。仅适用于 ZeRO Stage-3。Transformer_moe_cls_names（`str`，默认为`None`）：以逗号分隔的 Transformers MoE 层类名称列表（区分大小写）。例如`MixtralSparseMoeBlock`、`Qwen2MoeSparseMoeBlock`、`JetMoEAttention`、`JetMoEBlock`等。

enable_msamp（`bool`，默认为`None`）：指示是否启用 MS-AMP 后端进行 FP8 训练的标志。

msasmp_opt_level（`Optional[Literal["O1", "O2"]]`，默认为`None`）：MS-AMP 的优化级别（默认为“O1”）。仅当 `enable_msamp` 为 True 时才适用。应为 ['O1' 或 'O2'] 之一。
#### 选择[[accelerate.DeepSpeedPlugin.select]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L1552)

设置 HfDeepSpeedWeakref 以使用当前的 deepspeed 插件配置

####加速.utils.DummyScheduler[[accelerate.utils.DummyScheduler]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L362)

虚拟调度程序呈现模型参数或参数组，这主要用于遵循常规训练
当在 deepspeed 配置文件中指定调度程序配置时循环。

**参数：**

优化器 (`torch.optim.optimizer.Optimizer`) ：要包装的优化器。

Total_num_steps (int, *可选*) ：总步数。

Warmup_num_steps (int, *可选*) ：预热的步骤数。

lr_scheduler_callable（可调用，*可选*）：创建 LR 调度程序的可调用函数。它只接受一个参数`optimizer`。- **kwargs**（附加关键字参数，*可选*）：其他参数。

## DeepSpeedEnginerWrapper[[accelerate.utils.DeepSpeedEngineWrapper]]

####加速.utils.DeepSpeedEngineWrapper[[accelerate.utils.DeepSpeedEngineWrapper]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L253)

deepspeed.runtime.engine.DeepSpeedEngine 的内部包装器。这用于遵循传统的训练循环。

get_global_grad_normaccelerate.utils.DeepSpeedEngineWrapper.get_global_grad_normhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L286[]
从 DeepSpeed 引擎获取全局梯度范数。

**参数：**

引擎 (deepspeed.runtime.engine.DeepSpeedEngine) : 要包装的 Deepspeed 引擎

## DeepSpeedOptimizerWrapper[[accelerate.utils.DeepSpeedOptimizerWrapper]]

####加速.utils.DeepSpeedOptimizerWrapper[[accelerate.utils.DeepSpeedOptimizerWrapper]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L295)

deepspeed 优化器的内部包装器。

**参数：**

优化器 (`torch.optim.optimizer.Optimizer`) ：要包装的优化器。

## DeepSpeedSchedulerWrapper[[accelerate.utils.DeepSpeedSchedulerWrapper]]

####加速.utils.DeepSpeedSchedulerWrapper[[accelerate.utils.DeepSpeedSchedulerWrapper]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L322)

deepspeed 调度程序的内部包装器。**参数：**

调度程序 (`torch.optim.lr_scheduler.LambdaLR`) ：要包装的调度程序。

优化器（一个或一系列`torch.optim.Optimizer`）--

## DummyOptim[[accelerate.utils.DummyOptim]]

####加速.utils.DummyOptim[[accelerate.utils.DummyOptim]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py#L339)

虚拟优化器呈现模型参数或参数组，这主要用于遵循常规训练
当在 deepspeed 配置文件中指定优化器配置时循环。

**参数：**

lr（浮点数）：学习率。

params (iterable) ：可迭代的参数以优化或定义参数组

Weight_decay (float) ：权重衰减。

- **kwargs**（附加关键字参数，*可选*）：其他参数。

## 虚拟调度程序

### 命令行
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/cli.md