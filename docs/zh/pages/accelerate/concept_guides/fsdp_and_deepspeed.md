<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FSDP 与 DeepSpeed

Accelerate 通过集成两个极其强大的分布式训练工具，即[Pytorch FSDP](../usage_guides/fsdp)和[Microsoft DeepSpeed](../usage_guides/deepspeed)，提供了灵活的训练框架。本教程的目的是找出相似之处，并概述潜在的差异，以使用户能够在这两个框架之间无缝切换。

  要在框架之间切换，我们建议启动代码 `accelerate launch` 使用 `--config_file` 传递正确的配置文件，或者直接为 [FSDP and DeepSpeed](../package_reference/cli#accelerate-launch) 传递相应的参数。

  示例 Accelerate 配置可以在此处找到 [DeepSpeed](../usage_guides/deepspeed#accelerate-deepspeed-plugin) 和 [FSDP](../usage_guides/fsdp#how-it-works-out-of-the-box)，或者在 [example zoo under "Launch Configurations"](../usage_guides/explore)
 

本教程仅适用于单节点、多 GPU 场景。

## 配置功能

模型张量被分成不同的 GPU，以尝试扩大模型大小；这在 FSDP 中称为“分片”，在 DeepSpeed 中称为“分区”。 FSDP 分片和 DeepSpeed ZeRO（分区）阶段分别由 `--fsdp_sharding_strategy` 和 `--zero_stage` 配置。  特别是，FSDP `FULL_SHARD` 映射到 DeepSpeed ZeRO 阶段 `3`；看到这个[comprehensive mapping between FSDP sharding and DeepSpeed ZeRO settings](../usage_guides/fsdp#mapping-between-fsdp-sharding-strategies-and-deepspeed-zero-stages)。下表总结并分组了类似的设置：集团|框架|配置|示例|限制（如果有）
--|--|--|--|--
分片/分区| FSDP 深度速度 | `--fsdp_sharding_strategy``--zero_stage` | `1` (`FULL_SHARD`) `3` | 
卸载| FSDP 深度速度 | `--fsdp_offload_params``--offload_param_device``--offload_optimizer_device` | `true``cpu``cpu` |全有或全无  
模型加载| FSDP 深度速度 | `--fsdp_cpu_ram_efficient_loading``--zero3_init_flag` | `true``true` |只有零3
高效的检查点 | FSDP 深度速度 | `--fsdp_state_dict_type``--zero3_save_16bit_model` |  `SHARDED_STATE_DICT``true` |  只有零3
权重预取 | FSDP 深度速度 | `--fsdp_forward_prefetch``--fsdp_backward_prefetch`无 | `true``BACKWARD_PRE` | 
型号| FSDP 深度速度 |  `--fsdp_auto_wrap_policy``--fsdp_transformer_layer_cls_to_wrap`无 | `TRANSFORMER_BASED_WRAP` |通常不需要对用户透明。
参数调用| FSDP 深度速度 | `--fsdp_use_orig_params`无 | `true` | `torch.compile`所需，对用户透明
参数同步 | FSDP 深度速度 | `--fsdp_sync_module_states`无 | `true` | 
培训| FSDP 深度速度 |无`--gradient_accumulation_steps``--gradient_clipping` | `auto``auto` |对用户透明

以上详细说明请参考[⟦T41⟧ launch documentation](../package_reference/cli#accelerate-launch)。

    要访问其他 DeepSpeed 配置，例如混合精度设置， 
    你需要传入一个`--deepspeed_config_file`，参见[documentation](../usage_guides/deepspeed#deepspeed-config-file)。  

    DeepSpeed也可以通过[DeepSpeedPlugin](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin)进行配置，例如`DeepSpeedPlugin.zero_stage`相当于`--zero_stage`，`DeepSpeedPlugin.hf_ds_config`可用于传递`--deepeed_config_file.`

    FSDP 也可以通过 [FullyShardedDataParallelPlugin](/docs/accelerate/v1.14.0/en/package_reference/fsdp#accelerate.FullyShardedDataParallelPlugin) 配置，例如 `FullyShardedDataParallelPlugin.sharding_strategy` 相当于 `--fsdp_sharding_strategy`。
    

### 检查点请注意，虽然 FSDP 可以通过 `--fsdp_state_dict_type` 配置来保存完整/分片检查点。

    对于 DeepSpeed Zero3，可以通过`--zero3_save_16bit_model true`，这可以方便地将模型合并到单个等级并保存；这是 FSDP 相当于 `fsdp_state_dict_type: FULL_STATE_DICT`。 

    对于大型模型，将模型合并到单个等级可能会非常慢。

    为了更快地设置检查点，对于 FSDP 使用`fsdp_state_dict_type: SHARDED_STATE_DICT`，对于 DeepSpeed Zero3 使用[use the ⟦T53⟧ script to post-convert sharded checkpoints](https://www.deepspeed.ai/tutorials/zero/#extracting-weights)。

### 卸载

FSDP 仅允许“全有或全无”卸载（即卸载参数、梯度和优化器，或将它们全部保留在 GPU 中），但 DeepSpeed 可以以不同的方式卸载参数和优化器。此外，DeepSpeed 还支持[offloading to NVME](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)。

### 预取

FSDP 允许两种预取配置`--fsdp_forward_prefetch` 和`--fsdp_backward_prefetch`，以增加通信/计算的重叠，但需要额外的内存，请参阅[FSDP documentation](https://pytorch.org/docs/stable/fsdp.html)。 
对于 DeepSpeed，预取将在需要时打开，并且它的打开取决于某些超参数，如`stage3_param_persistence_threshold`、`stage3_max_reuse_distance` 等、[that can be configured for Zero3](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)；如果您没有在 deepspeed 配置文件中明确设置这些超参数，`accelerate` 可能会自动设置这些超参数。

    对于 FSDP，如果内存允许，请设置`fsdp_backward_prefetch: BACKWARD_PRE`以提高吞吐量。

### 模型加载虽然 FSDP 需要显式的 `--fsdp_cpu_ram_efficient_loading true` 来激活高效的模型加载，但每当使用 DeepSpeed Zero3 时，`transformers` 都会激活类似的功能。

    对于 FSDP，每当设置 `--fsdp_cpu_ram_efficient_loading true` 时，`accelerate` 都会自动将 `sync_module_states` 设置为 true。 
    为了RAM有效加载，权重将仅在单个等级中加载，因此需要`sync_module_states`将权重广播到其他等级。

### 型号

FSDP 需要一个显式的 `--fsdp_auto_wrap_policy` 算法来决定如何调度全收集和减少分散操作。但对于 DeepSpeed 来说，这对用户来说是透明的。

    对于 FSDP，只需设置`fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`。通过最新的`transformers`版本，我们尽力找出适合高频变压器型号的`fsdp_transformer_layer_cls_to_wrap`。但是，如果您收到相关错误，请指定这一点。

### 参数调用

如果使用 `torch.compile`，FSDP 需要显式 `--fsdp_use_orig_params` 标志，请参阅[the pytorch documentation](https://pytorch.org/docs/stable/fsdp.html#module-torch.distributed.fsdp)。对于 DeepSpeed，这对用户来说是透明的。

    对于FSDP，使用`torch.compile`时请设置`fsdp_use_orig_params: True`。

## 培训

Deepspeed 需要显式的 `--gradient_accumulation_steps` 和 `--gradient_clipping` 标志。对于 FSDP，这对用户来说是透明的。

    使用 DeepSpeed 时，设置 `gradient_accumulation_steps: "auto"` 和 `gradient_clipping: "auto"` 自动获取 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 或 `TrainingArguments` 中设置的值（如果使用 `transformers`）。## 关于数据精度处理的差异

为了讨论 FSDP 和 Deepspeed 中如何处理数据精度，首先概述一下这些框架中如何处理模型参数是有启发性的。在将模型/优化器参数分布到 GPU 上之前，需要进行参数准备，首先将它们“展平”为一维[⟦T80⟧](https://pytorch.org/docs/stable/tensors.html#torch-tensor)。 FSDP / DeepSpeed 的实现在存储这些“扁平”参数的`dtype`方面有所不同，并且对于[⟦T82⟧](https://pytorch.org/docs/stable/optim.html#module-torch.optim)如何分配其`dtype`也存在影响。下表概述了两个框架的流程； “本地”列表示在每个 GPU 级别发生的进程，因此向上转换产生的任何内存开销应理解为按所使用的 GPU 数量进行摊销。

    根据经验，为了具有自动混合精度的稳定训练，所有可训练参数都必须位于 `torch.float32` 中。流程|本地|框架|详情
--|--|--|--
正在加载，即`AutoModel.from_pretrained(..., torch_dtype=torch_dtype)`|  
准备工作，即创建“平面参数” | ✅ | FSDP 深度速度 |创建于`torch_dtype`。忽略在 `float32` 创建的 `torch_dtype`。
优化器初始化| ✅ | FSDP 深度速度 |在 `torch_dtype` 中创建参数 在 `float32` 中创建参数
训练步骤，即向前、向后、减少 | | FSDP 深度速度 |遵循 [⟦T91⟧](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision) 遵循 `deepspeed_config_file` 混合精度设置。
优化器（前置步骤）| ✅ | FSDP 深度速度 |向上转换（如果有）到`torch_dtype`向上转换到`float32`
优化器（实际步骤）| ✅ | FSDP 深度速度 |发生在`torch_dtype` 发生在`float32`。

    因此，当使用少量 GPU 时，请注意由于准备期间的向上转换而可能产生大量内存开销。

    使用 FSDP，在没有混合精度的情况下，可以以低精度 `torch_dtype` 操作[⟦T97⟧](https://pytorch.org/docs/stable/optim.html#module-torch.optim)，这在使用少量 GPU 时可能会有所帮助。 

    对于混合精度，FSDP 和 DeepSpeed 将在模型准备步骤中向上转换（参见上表）。但请注意，FSDP 将以升级后的精度保存检查点；如果指定了 `--zero3_save_16bit_model`，Deepspeed 仍可能保存低精度检查点。为了阐明上表，请考虑下面的具体示例；为了简洁起见，优化器预步骤和实际步骤结合在一起。使用 FSDP，可以在如下所示的两种模式下运行，但 DeepSpeed 只能在其中一种模式下运行。

框架|模型加载（`torch_dtype`）|混合精度 |准备工作（本地）|培训|优化器（本地）
--|--|--|--|--|--
FSDP | BF16 |默认（无）| BF16 | BF16 | BF16
FSDP | BF16 | BF16 | FP32 | BF16 | FP32
深速| BF16 | BF16 | FP32 | BF16 | FP32

### 梯度同步
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/gradient_synchronization.md