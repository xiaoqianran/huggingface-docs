<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Megatron-LM 实用程序

## MegatronLMPlugin[[accelerate.utils.MegatronLMPlugin]]

####加速.utils.MegatronLMPlugin[[accelerate.utils.MegatronLMPlugin]]

```python
accelerate.utils.MegatronLMPlugin(tp_degree: int = None, pp_degree: int = None, use_custom_fsdp: bool = None, overlap_cpu_optimizer_d2h_h2d: bool = None, no_load_optim: bool = None, eod_mask_loss: bool = None, no_save_optim: bool = None, optimizer_cpu_offload: bool = None, use_precision_aware_optimizer: bool = None, decoder_last_pipeline_num_layers: int = None, recompute_granularity: str = None, recompute_method: str = None, recompute_num_layers: int = None, attention_backend: bool = None, expert_model_parallel_size: int = None, context_parallel_size: int = None, attention_dropout: float = None, hidden_dropout: float = None, attention_softmax_in_fp32: bool = None, expert_tensor_parallel_size: int = None, calculate_per_token_loss: bool = None, use_rotary_position_embeddings: bool = None, num_micro_batches: int = None, gradient_clipping: float = None, sequence_parallelism: bool = None, recompute_activations: bool = None, use_distributed_optimizer: bool = None, pipeline_model_parallel_split_rank: int = None, num_layers_per_virtual_pipeline_stage: int = None, is_train_batch_min: str = True, train_iters: int = None, train_samples: int = None, weight_decay_incr_style: str = 'constant', start_weight_decay: float = None, end_weight_decay: float = None, lr_decay_style: str = 'linear', lr_decay_iters: int = None, lr_decay_samples: int = None, lr_warmup_iters: int = None, lr_warmup_samples: int = None, lr_warmup_fraction: float = None, min_lr: float = 0, consumed_samples: list = None, no_wd_decay_cond: typing.Optional[typing.Callable] = None, scale_lr_cond: typing.Optional[typing.Callable] = None, lr_mult: float = 1.0, megatron_dataset_flag: bool = False, seq_length: int = None, encoder_seq_length: int = None, decoder_seq_length: int = None, tensorboard_dir: str = None, set_all_logging_options: bool = False, eval_iters: int = 100, eval_interval: int = 1000, return_logits: bool = False, custom_train_step_class: typing.Optional[typing.Any] = None, custom_train_step_kwargs: typing.Optional[dict[str, typing.Any]] = None, custom_model_provider_function: typing.Optional[typing.Callable] = None, custom_prepare_model_function: typing.Optional[typing.Callable] = None, custom_megatron_datasets_provider_function: typing.Optional[typing.Callable] = None, custom_get_batch_function: typing.Optional[typing.Callable] = None, custom_loss_function: typing.Optional[typing.Callable] = None, other_megatron_args: typing.Optional[dict[str, typing.Any]] = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L2335)

**参数：**

tp_ Degree (`int`, 默认为`None`) : 张量并行度。

pp_ Degree (`int`, 默认为`None`) : 管道并行度。

num_micro_batches（`int`，默认为`None`）：微批次数。

梯度裁剪（`float`，默认为`None`）：基于全局L2范数的梯度裁剪值（0表示禁用）。

equence_parallelism（`bool`，默认为`None`）：启用序列并行。

recompute_activations（`bool`，默认为`None`）：启用选择性激活重新计算。

use_distributed_optimizer（`bool`，默认为`None`）：启用分布式优化器。

pipeline_model_parallel_split_rank（`int`，默认为`None`）：编码器和解码器应拆分的排名。

num_layers_per_virtual_pipeline_stage（`int`，默认为`None`）：每个虚拟管道阶段的层数。

is_train_batch_min（`str`，默认为`True`）：如果同时指定了tran和eval数据加载器，这将决定`micro_batch_size`。train_iters（`int`，默认为`None`）：所有训练运行中要训练的样本总数。请注意，使用 `MegatronLMDummyScheduler` 时应提供 train-iters 或 train-samples。

train_samples（`int`，默认为`None`）：所有训练运行中要训练的样本总数。请注意，使用 `MegatronLMDummyScheduler` 时应提供 train-iters 或 train-samples。

Weight_decay_incr_style (`str`，默认为`'constant'`)：权重衰减增量函数。选择=[“常数”，“线性”，“余弦”]。

start_weight_decay（`float`，默认为`None`）：L2正则化的初始权重衰减系数。

end_weight_decay（`float`，默认为`None`）：L2 正则化的运行结束权重衰减系数。

lr_decay_style (`str`，默认为`'linear'`)：学习率衰减函数。选择=['常数'，'线性'，'余弦']。

lr_decay_iters（`int`，默认为`None`）：学习率衰减的迭代次数。如果 None 默认为`train_iters`。

lr_decay_samples（`int`，默认为`None`）：学习率衰减的样本数。如果 None 默认为`train_samples`。

lr_warmup_iters（`int`，默认为`None`）：线性预热学习率的迭代次数。lr_warmup_samples（`int`，默认为`None`）：线性预热学习率的样本数。

lr_warmup_fraction (`float`，默认为`None`)：lr-warmup-(iters/samples) 线性预热学习率的分数。

min_lr (`float`，默认为`0`)：学习率的最小值。调度程序剪辑值低于此阈值。

Consumer_samples（`List`，默认为`None`）：以与`accelerator.prepare`调用的数据加载器相同的顺序消耗的样本数。

no_wd_decay_cond（`Optional`，默认为`None`）：禁用权重衰减的条件。

scale_lr_cond（`Optional`，默认为`None`）：缩放学习​​率的条件。

lr_mult (`float`，默认为`1.0`)：学习率乘数。

megatron_dataset_flag (`bool`, 默认为`False`) : 数据集的格式是否遵循 Megatron-LM Indexed/Cached/MemoryMapped 格式。

seq_length（`int`，默认为`None`）：要处理的最大序列长度。

encoder_seq_length（`int`，默认为`None`）：编码器要处理的最大序列长度。

解码器_seq_length（`int`，默认为`None`）：解码器要处理的最大序列长度。

tensorboard_dir（`str`，默认为`None`）：保存tensorboard日志的路径。set_all_logging_options (`bool`, 默认为`False`) : 是否设置所有日志选项。

eval_iters（`int`，默认为`100`）：为评估验证/测试运行的迭代次数。

eval_interval（`int`，默认为`1000`）：在验证集上运行评估之间的时间间隔。

return_logits (`bool`, 默认为`False`) ：是否从模型返回logits。

custom_train_step_class（`Optional`，默认为`None`）：自定义训练步骤类。

custom_train_step_kwargs（`Optional`，默认为`None`）：自定义训练步骤kwargs。

custom_model_provider_function（`Optional`，默认为`None`）：自定义模型提供程序函数。

custom_prepare_model_function (`Optional`, 默认为`None`) : 自定义准备模型函数。

custom_megatron_datasets_provider_function（`Optional`，默认为`None`）：自定义megatron train_valid_test数据集提供程序函数。

custom_get_batch_function (`Optional`, 默认为`None`) : 自定义获取批处理函数。

custom_loss_function (`Optional`, 默认为`None`) : 自定义损失函数。

other_megatron_args（`Optional`，默认为`None`）：其他 Megatron-LM 参数。请参阅威震天-LM。Megatron-LM 插件，用于启用张量、管道、序列和数据并行性。还可以启用选择性
激活重新计算和优化的融合内核。

## MegatronLMDummyScheduler[[accelerate.utils.MegatronLMDummyScheduler]]

####加速.utils.MegatronLMDummyScheduler[[accelerate.utils.MegatronLMDummyScheduler]]

```python
accelerate.utils.MegatronLMDummyScheduler(optimizer, total_num_steps = None, warmup_num_steps = 0, **kwargs)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L378)

**参数：**

优化器 (`torch.optim.optimizer.Optimizer`) ：要包装的优化器。

Total_num_steps (int) ：总步数。

Warmup_num_steps (int) ：热身的步骤数。

- **kwargs**（附加关键字参数，*可选*）：其他参数。

虚拟调度程序呈现模型参数或参数组，这主要用于遵循常规训练
当在 deepspeed 配置文件中指定调度程序配置时循环。

## MegatronLMDummyDataLoader[[accelerate.utils.MegatronLMDummyDataLoader]]

####加速.utils.MegatronLMDummyDataLoader[[accelerate.utils.MegatronLMDummyDataLoader]]

```python
accelerate.utils.MegatronLMDummyDataLoader(**dataset_kwargs)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L162)

**参数：**

- ****dataset_kwargs** ：威震天数据参数。

虚拟数据加载器呈现模型参数或参数组，这主要用于遵循常规训练

## AbstractTrainStep[[accelerate.utils.AbstractTrainStep]]####加速.utils.AbstractTrainStep[[accelerate.utils.AbstractTrainStep]]

```python
accelerate.utils.AbstractTrainStep(name)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L415)

用于批处理、前向传递和损失处理程序的抽象类。

## GPTTrainStep[[accelerate.utils.GPTTrainStep]]

####加速.utils.GPTTrainStep[[accelerate.utils.GPTTrainStep]]

```python
accelerate.utils.GPTTrainStep(accelerator, args)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L574)

**参数：**

args (`argparse.Namespace`) ：威震天-LM 参数。

GPT 训练步骤类。

## BertTrainStep[[accelerate.utils.BertTrainStep]]

####加速.utils.BertTrainStep[[accelerate.utils.BertTrainStep]]

```python
accelerate.utils.BertTrainStep(accelerator, args)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L432)

**参数：**

args (`argparse.Namespace`) ：威震天-LM 参数。

Bert训练步课。

## T5TrainStep[[accelerate.utils.T5TrainStep]]

####加速.utils.T5TrainStep[[accelerate.utils.T5TrainStep]]

```python
accelerate.utils.T5TrainStep(accelerator, args)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L718)

**参数：**

args (`argparse.Namespace`) ：威震天-LM 参数。

T5列车阶梯班。

## avg_losses_across_data_parallel_group[[accelerate.utils.avg_losses_across_data_parallel_group]]

####加速.utils.avg_losses_across_data_parallel_group[[accelerate.utils.avg_losses_across_data_parallel_group]]

```python
accelerate.utils.avg_losses_across_data_parallel_group(losses)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/megatron_lm.py#L1217)

**参数：**

loss (List[Tensor]) ：跨数据并行组的平均损失列表。

数据并行组的平均损失。### 加速器
https://huggingface.co/docs/accelerate/v1.15.0/package_reference/accelerator.md