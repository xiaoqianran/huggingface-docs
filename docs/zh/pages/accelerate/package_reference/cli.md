<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 命令行 

以下是所有可用命令的列表🤗 使用其参数加速

## 加速配置

**命令**：

`accelerate config` 或 `accelerate-config`

启动一系列提示，为您的训练系统创建并保存 `default_config.yml` 配置文件。应该 
始终首先在您的机器上运行。

**用法**： 

```bash
accelerate config [arguments]
```

**可选参数**：
* `--config_file CONFIG_FILE` (`str`) -- 用于存储配置文件的路径。会默认在缓存位置有一个名为default_config.yaml的文件，这是内容
                        环境`HF_HOME`后缀为“accelerate”，或者如果你没有这样的环境变量，则为你的缓存目录
                        （`~/.cache`或`XDG_CACHE_HOME`的内容）后缀为`huggingface`。
* `-h`, `--help` (`bool`) -- 显示帮助信息并退出

## 加速配置默认值

**命令**：

`accelerate config default` 或 `accelerate-config default`

为 Accelerate 创建一个默认配置文件，仅设置几个标志。

**用法**： 

```bash
accelerate config default [arguments]
```**可选参数**：
* `--config_file CONFIG_FILE` (`str`) -- 用于存储配置文件的路径。会默认在缓存位置有一个名为default_config.yaml的文件，这是内容
                        环境`HF_HOME`后缀为“accelerate”，或者如果你没有这样的环境变量，则为你的缓存目录
                        （`~/.cache`或`XDG_CACHE_HOME`的内容）后缀为`huggingface`。

* `-h`, `--help` (`bool`) -- 显示帮助消息并退出
* `--mixed_precision {no,fp16,bf16}` (`str`) -- 是否使用混合精度训练。在 FP16 和 BF16 (bfloat16) 训练之间进行选择。 BF16 训练仅在 Nvidia Ampere GPU 和 PyTorch 1.10 或更高版本上受支持。

## 加速配置更新

**命令**：

`accelerate config update` 或 `accelerate-config update`

使用最新的默认值更新现有配置文件，同时保留旧配置。

**用法**： 

```bash
accelerate config update [arguments]
```**可选参数**：
* `--config_file CONFIG_FILE` (`str`) -- 要更新的配置文件的路径。会默认在缓存位置有一个名为default_config.yaml的文件，这是内容
                        环境`HF_HOME`后缀为“accelerate”，或者如果你没有这样的环境变量，则为你的缓存目录
                        （`~/.cache`或`XDG_CACHE_HOME`的内容）后缀为`huggingface`。

* `-h`, `--help` (`bool`) -- 显示帮助消息并退出

## 加速环境

**命令**：

`accelerate env` 或 `accelerate-env` 或 `python -m accelerate.commands.env`

列出传递的 🤗 加速配置文件的内容。在 [GitHub repository](https://github.com/huggingface/accelerate) 上打开问题时应始终使用。

**用法**：

```bash
accelerate env [arguments]
```

**可选参数**：
* `--config_file CONFIG_FILE` (`str`) -- 用于存储配置文件的路径。会默认在缓存位置有一个名为default_config.yaml的文件，这是内容
                        环境`HF_HOME`后缀为“accelerate”，或者如果你没有这样的环境变量，则为你的缓存目录
                        （`~/.cache`或`XDG_CACHE_HOME`的内容）后缀为`huggingface`。
* `-h`, `--help` (`bool`) -- 显示帮助消息并退出

## 加速启动

**命令**：

`accelerate launch` 或 `accelerate-launch` 或 `python -m accelerate.commands.launch`使用正确的参数在分布式系统上启动指定的脚本。

**用法**： 

```bash
accelerate launch [arguments] {training_script} --{training_script-argument-1} --{training_script-argument-2} ...
```

**立场论点**：

- `{training_script}` -- 并行启动脚本的完整路径
- `--{training_script-argument-1}` -- 训练脚本的参数

**可选参数**：

* `-h`, `--help` (`bool`) -- 显示帮助消息并退出
* `--config_file CONFIG_FILE` (`str`)-- 用于启动脚本中默认值的配置文件。
* `-m`、`--module` (`bool`) -- 更改每个进程以将启动脚本解释为 Python 模块，并以与“python -m”相同的行为执行。
* `--no_python` (`bool`) -- 跳过在训练脚本前添加“python” - 直接执行即可。当脚本不是 Python 脚本时很有用。
* `--debug` (`bool`) -- 当出现故障时是否打印 torch.distributed 堆栈跟踪。
* `-q`、`--quiet` (`bool`) -- 沉默启动堆栈跟踪中的子进程错误，仅显示相关的回溯。 （仅适用于 DeepSpeed 和单进程配置）。

其余参数通过 `accelerate config` 配置，并从指定的 `--config_file`（或默认配置）中读取 
价值观。它们也可以手动传入。

**硬件选择参数**：* `--cpu` (`bool`) -- 是否强制在CPU上进行训练。
* `--multi_gpu` (`bool`) -- 是否应该启动分布式 GPU 训练。
* `--tpu` (`bool`) -- 是否应该启动TPU训练。

**资源选择参数**：

以下参数对于微调可用硬件的使用方式很有用

* `--mixed_precision {no,fp16,bf16,fp8}` (`str`) -- 是否使用混合精度训练。在 FP16 和 BF16 (bfloat16) 训练之间进行选择。 BF16 训练仅在 Nvidia Ampere GPU 和 PyTorch 1.10 或更高版本上受支持。
* `--num_processes NUM_PROCESSES` (`int`) -- 并行启动的进程总数。
* `--num_machines NUM_MACHINES` (`int`) -- 本次培训使用的机器总数。
* `--num_cpu_threads_per_process NUM_CPU_THREADS_PER_PROCESS` (`int`) -- 每个进程的CPU线程数。可以调整以获得最佳性能。
* `--enable_cpu_affinity` (`bool`) -- 是否应启用CPU亲和性和平衡。目前仅在 NVIDIA 硬件上受支持。

**训练范式参数**：

以下参数对于选择要使用的训练范例很有用。* `--use_deepspeed` (`bool`) -- 是否使用 DeepSpeed 进行训练。
* `--use_fsdp` (`bool`) -- 是否使用FullyShardedDataParallel进行训练。
* `--use_megatron_lm` (`bool`) -- 是否使用Megatron-LM进行训练。

**分布式 GPU 参数**：

以下参数仅在传递`multi_gpu`或通过`accelerate config`配置多GPU训练时有用： 

* `--gpu_ids` (`str`) -- 应该使用哪些 GPU（按 id）在这台机器上进行训练，以逗号分隔列表形式
* `--same_network` (`bool`) -- 用于多节点训练的所有机器是否存在于同一个本地网络上。
* `--machine_rank` (`int`) -- 启动此脚本的机器的等级。
* `--main_process_ip` (`str`) -- 等级0的机器的IP地址。
* `--main_process_port` (`int`) -- 用于与 0 级机器通信的端口。
* `-t`、`--tee` (`str`) -- Tee std 流到日志文件以及控制台。
* `--log_dir` (`str`) -- 使用 torchrun/torch.distributed.run 作为启动器时用于日志文件的基本目录。与 --tee 一起使用来重定向 std 流信息日志文件。
* `--role` (`str`) -- 用户定义的工人角色。
* `--rdzv_backend` (`str`) -- 使用的集合方法，例如 'static' （默认）或 'c10d'* `--rdzv_conf` (`str`) -- 附加集合点配置 (=,=,...)。
* `--max_restarts` (`int`) -- 失败前工作组重新启动的最大次数。
* `--monitor_interval` (`int`) -- 监控worker状态的时间间隔，以秒为单位。

**TPU 参数**：

以下参数仅在传递`tpu`或通过`accelerate config`配置TPU训练时才有用： 

* `--tpu_cluster` (`bool`) -- 是否使用 GCP TPU pod 进行训练。
* `--tpu_use_sudo` (`bool`) -- 在每个 Pod 中运行 TPU 训练脚本时是否使用`sudo`。
* `--vm` (`str`) -- 单个计算虚拟机实例名称列表。如果未提供，我们假设使用实例组。适用于 TPU 荚。
* `--env` (`str`) -- 要在计算虚拟机实例上设置的环境变量列表。适用于 TPU 荚。
* `--main_training_function` (`str`) -- 脚本中要执行的主函数的名称（仅适用于 TPU 训练）。
* `--downcast_bf16` (`bool`) -- 在 TPU 上使用 bf16 精度时，是否将 float 和 double 张量都转换为 bfloat16，或者 double 张量仍保留为 float32。

**DeepSpeed 参数**：

以下参数仅在传递 `use_deepspeed` 或通过 `accelerate config` 配置 `deepspeed` 时有用：* `--deepspeed_config_file` (`str`) -- DeepSpeed 配置文件。
* `--zero_stage` (`int`) -- DeepSpeed 的 ZeRO 优化阶段。
* `--offload_optimizer_device` (`str`) -- 决定在何处 (none|cpu|nvme) 卸载优化器状态。
* `--offload_param_device` (`str`) -- 决定在何处 (none|cpu|nvme) 卸载参数。
* `--offload_optimizer_nvme_path` (`str`) -- 决定卸载优化器状态的 Nvme 路径。
* `--gradient_accumulation_steps` (`int`) -- 训练脚本中没有使用gradient_accumulation_steps。
* `--gradient_clipping` (`float`) -- 训练脚本中使用的梯度裁剪值。
* `--zero3_init_flag` (`str`) -- 决定是否(true|false)启用`deepspeed.zero.Init`来构建海量模型。仅适用于 DeepSpeed ZeRO Stage-3。
* `--zero3_save_16bit_model` (`str`) -- 决定在使用 ZeRO Stage-3 时是否（true|false）保存 16 位模型权重。仅适用于 DeepSpeed ZeRO Stage-3。
* `--deepspeed_hostfile` (`str`) -- 用于配置多节点计算资源的 DeepSpeed 主机文件。
* `--deepspeed_exclusion_filter` (`str`) -- 使用多节点设置时的 DeepSpeed 排除过滤器字符串。
* `--deepspeed_inclusion_filter` (`str`) -- 使用多节点设置时的 DeepSpeed 包含过滤器字符串。
* `--deepspeed_multinode_launcher` (`str`) -- 要使用的 DeepSpeed 多节点启动器。* `--deepspeed_moe_layer_cls_names` (`str`) -- 要换行的变压器 MoE 层类名称的逗号分隔列表（区分大小写），例如，`MixtralSparseMoeBlock` `Qwen2MoeSparseMoeBlock`、`JetMoEAttention,JetMoEBlock`

**完全分片数据并行性参数**：

以下参数仅在传递 `use_fsdp` 或通过 `accelerate config` 配置完全分片数据并行性时有用：

* `--fsdp_offload_params` (`str`) -- 决定是否（true|false）将参数和梯度卸载到CPU。
* `--fsdp_min_num_params` (`int`) -- FSDP 默认自动换行的最小参数数量。
* `--fsdp_sharding_strategy` (`int`) -- FSDP 的分片策略。
* `--fsdp_auto_wrap_policy` (`str`) -- FSDP 的自动换行策略。
* `--fsdp_transformer_layer_cls_to_wrap` (`str`) -- 要换行的 Transformer 层类名称（区分大小写），例如 `BertLayer`、`GPTJBlock`、`T5Block` ...
* `--fsdp_backward_prefetch_policy` (`str`) -- FSDP 的向后预取策略。
* `--fsdp_state_dict_type` (`str`) -- FSDP 的状态字典类型。
* `--fsdp_forward_prefetch` (`str`) -- FSDP 前向预取。
* `--fsdp_use_orig_params` (`str`) -- 如果为 True，则允许在 FSDP 单元中混合非均匀 `requires_grad`。
* `--fsdp_cpu_ram_efficient_loading` (`str`) -- 如果为 true，则只有第一个进程加载预训练模型检查点，而所有其他进程都具有空权重。使用此功能时，`--fsdp_sync_module_states` 需要为 True。
* `--fsdp_sync_module_states` (`str`) -- 如果为 true，则每个独立包装的 FSDP 单元将从 0 级开始广播模块参数。* `--fsdp_activation_checkpointing` (`bool`) -- 决定前向传递过程中是否释放中间激活，并留下一个检查点作为占位符

**威震天-LM 参数**：

以下参数仅在传递 `use_megatron_lm` 或通过 `accelerate config` 配置 Megatron-LM 时有用：

* `--megatron_lm_tp_degree` (``) -- Megatron-LM 的张量并行度 (TP)。
* `--megatron_lm_pp_degree` (``) -- Megatron-LM 的管道并行度 (PP)。
* `--megatron_lm_num_micro_batches` (``) -- PP 度 > 1 时威震天-LM 的微批次数量。
* `--megatron_lm_sequence_parallelism` (``) -- 决定当 TP 度 > 1 时是否启用序列并行性 (true|false)。
* `--megatron_lm_recompute_activations` (``) -- 决定是否（true|false）启用选择性激活重新计算。
* `--megatron_lm_use_distributed_optimizer` (``) -- 决定是否（true|false）使用分布式优化器，该优化器在数据并行 (DP) 级别上分片优化器状态和梯度。
* `--megatron_lm_gradient_clipping` (``) -- Megatron-LM 基于全局 L2 Norm 的梯度裁剪值（0 表示禁用）。

**FP8 参数**：* `--fp8_backend` (`str`) -- 选择一个后端来使用 FP8 进行训练（`te` 或 `msamp`）
* `--fp8_use_autocast_during_eval` (`bool`) -- 在eval模式下是否使用FP8自动转换（仅当`--fp8_backend=te`通过时有用）。通常，如果未通过此标准，则会找到更好的指标。
* `--fp8_margin` (`int`) -- 用于梯度缩放的边距（仅在传递 `--fp8_backend=te` 时有用）。
* `--fp8_interval` (`int`) -- 重新计算缩放因子的频率的间隔（仅在传递 `--fp8_backend=te` 时有用）。
* `--fp8_format` (`str`) -- 用于 FP8 配方的格式（仅在传递 `--fp8_backend=te` 时有用）。
* `--fp8_amax_history_len` (`int`) -- 用于缩放因子计算的历史记录长度（仅在传递`--fp8_backend=te`时有用）。
* `--fp8_amax_compute_algo` (`str`) -- 用于计算缩放因子的算法。 （仅当`--fp8_backend=te`通过时才有用）。
* `--fp8_override_linear_precision` (`Tuple[bool, bool, bool]`) -- 是否以更高精度执行`fprop`、`dgrad`、`wgrad` GEMMS。
* `--fp8_opt_level` (`str`) -- MS-AMP 应该使用什么级别的 8 位集体通信（只有当`--fp8_backend=msamp` 通过时才有用）

**AWS SageMaker 参数**：

以下参数仅在 SageMaker 中训练时有用* `--aws_access_key_id AWS_ACCESS_KEY_ID` (`str`) -- 用于启动 Amazon SageMaker 训练作业的 AWS_ACCESS_KEY_ID
* `--aws_secret_access_key AWS_SECRET_ACCESS_KEY` (`str`) -- 用于启动 Amazon SageMaker 训练作业的 AWS_SECRET_ACCESS_KEY

## 加速估计内存

**命令**：

`accelerate estimate-memory` 或 `accelerate-estimate-memory` 或 `python -m accelerate.commands.estimate`

估计集线器上托管的特定模型需要加载的总 vRAM 以及训练估计值。需要安装`huggingface_hub`。 

    执行推理时，通常在结果中添加 ≤20% 作为总体分配[as referenced here](https://blog.eleuther.ai/transformer-math/)。将来我们将有更广泛的估计，这些估计将自动包含在计算中。

**用法**： 

```bash
accelerate estimate-memory {MODEL_NAME} --library_name {LIBRARY_NAME} --dtypes {dtype_1} {dtype_2} ...
```

**必需的参数**：

* `MODEL_NAME` (`str`)-- Hugging Face Hub 上的型号名称

**可选参数**：* `--library_name {timm,transformers}` (`str`) -- 与模型集成的库，例如 `transformers`，仅当此信息未存储在 Hub 上时才需要
* `--dtypes {float32,float16,int8,int4}` (`[{float32,float16,int8,int4} ...]`) -- 用于模型的数据类型，必须是 `float32`、`float16`、`int8` 和 `int4` 中的一种（或多种）
* `--trust_remote_code` (`bool`) -- 是否允许在 Hub 上在其自己的建模文件中定义自定义模型。仅应为您信任且已在其中读取代码的存储库传递此选项，因为它将执行本地计算机上的集线器上存在的代码。

## 加速 tpu-config

`accelerate tpu-config`

**用法**：

```bash
accelerate tpu-config [arguments]
```

**可选参数**：
* `-h`, `--help` (`bool`) -- 显示帮助消息并退出

**配置参数**：

可以通过`accelerate config`配置的参数。

* `--config_file` (`str`) -- 用于加速的配置文件的路径。
* `--tpu_name` (`str`) -- 要使用的 TPU 的名称。如果未指定，将使用配置文件中指定的 TPU。
* `--tpu_zone` (`str`) -- 要使用的 TPU 区域。如果未指定，将使用配置文件中指定的区域。

**TPU 参数**：

选项参数在 TPU 内部运行。* `--command_file` (`str`) -- 包含启动时在 Pod 上运行的命令的文件的路径。
* `--command` (`str`) -- 在 pod 上运行的命令。可以多次通过。
* `--install_accelerate` (`bool`) -- 是否在 Pod 上安装加速。默认为 False。
* `--accelerate_version` (`str`) -- 安装在 Pod 上的加速版本。如果未指定，将使用最新的 pypi 版本。指定“dev”以从 GitHub 安装。
* `--debug` (`bool`) -- 如果设置，将打印将要运行的命令而不是运行它。

## 加速测试

`accelerate test` 或 `accelerate-test`

运行 `accelerate/test_utils/test_script.py` 来验证 🤗 Accelerate 是否已在您的系统上正确配置并运行。 

**用法**： 

```bash
accelerate test [arguments]
```

**可选参数**：
* `--config_file CONFIG_FILE` (`str`) -- 用于存储配置文件的路径。会默认在缓存位置有一个名为default_config.yaml的文件，这是内容
                        环境`HF_HOME`后缀为“accelerate”，或者如果你没有这样的环境变量，则为你的缓存目录
                        （`~/.cache`或`XDG_CACHE_HOME`的内容）后缀为`huggingface`。
* `-h`, `--help` (`bool`) -- 显示帮助信息并退出### 管道并行性
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/inference.md