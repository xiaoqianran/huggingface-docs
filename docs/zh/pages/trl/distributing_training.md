<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 分发训练

> [!警告]
> 部分正在建设中。欢迎贡献！

## 使用 TRL 进行多 GPU 训练

TRL 中的训练器使用 [🤗 Accelerate](https://github.com/huggingface/accelerate) 来实现跨多个 GPU 或节点的分布式训练。为此，首先通过运行创建一个 [🤗 Accelerate](https://github.com/huggingface/accelerate) 配置文件

```bash
accelerate config
```

并根据您的多 GPU/多节点设置回答问题。然后，您可以通过运行以下命令来启动分布式训练：

```bash
accelerate launch train.py
```

我们还在[examples folder](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs)中提供了可用作模板的配置文件。要使用这些模板，只需在启动作业时传递配置文件的路径，例如：

```shell
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml train.py <SCRIPT_ARGS>
```

这会自动将工作负载分配到所有可用的 GPU 上。

在底层，[🤗 Accelerate](https://github.com/huggingface/accelerate)为每个 GPU 创建一个模型。各工序：

- 处理自己的一批数据
- 计算该批次的损失和梯度
- 在所有 GPU 之间共享梯度更新

![multi gpu](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/multi_gpu.png)

有效批量大小计算如下：

$$
\text{批量大小} = \text{每个设备\_train\_batch\_size} \times \text{num\_devices} \times \text{梯度\_累积\_步数}
$$

为了在扩展到多个 GPU 时保持一致的批量大小，请确保相应地更新 `per_device_train_batch_size` 和 `gradient_accumulation_steps`。例如，这些配置是等效的，并且应该产生相同的结果：

| GPU 数量 |每台设备的批量大小 |梯度累积步骤|评论 |
| --- | --- | --- | --- |
| 1 | 32 | 32 1 |内存使用率可能较高，但训练速度更快 |
| 1 | 4 | 8 |内存使用率较低，训练速度较慢 |
| 8 | 4 | 1 |多 GPU 实现两全其美 |

> [!提示]
> 每个 GPU 拥有一个模型可能会导致内存使用量较高，这对于大型模型或低内存 GPU 来说可能不可行。在这种情况下，您可以利用[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)，它提供模型分片、零冗余优化器、混合精度训练以及卸载到 CPU 或 NVMe 等优化。请查看我们的 [DeepSpeed Integration](deepspeed_integration) 指南了解更多详细信息。

## 长上下文训练的序列并行性

序列并行（也称为上下文并行）是一种并行化技术，通过将序列维度拆分到多个 GPU 上，可以使用更长的序列进行训练。每个 GPU 处理序列的一部分，允许您使用比单个 GPU 内存所能容纳的序列更长的序列进行训练。> [!注意]
> **术语澄清：** 本节描述用于分割序列以实现更长的上下文训练的并行技术：
> - **上下文并行性 (CP)**：跨 GPU 分割序列（通过 FSDP2 实现为环注意）
> - **序列并行性 (SP)**：序列分割的另一种形式（通过 DeepSpeed 实现为 ALST/Ulysses）
>
> CP和SP都不同于传统的序列并行与张量并行（TP+SP）一起使用以减少激活记忆。使用这里的技术，并行度倍增：`TP=2`和`CP=2`将需要4个GPU（2×2），而传统的`TP+SP=2`只需要2个GPU，因为它们共享相同的等级。
>
> 在 Accelerate 的 `ParallelismConfig` 中：
> - 使用 `cp_size` 与 `cp_backend="torch"` 进行环注意 (FSDP2)
> - 将 `sp_size` 与 `sp_backend="deepspeed"` 结合使用用于 ALST/Ulysses (DeepSpeed)

序列并行在以下情况下特别有用：

- 你想用很长的序列（>32k tokens）进行训练
- 单 GPU 内存不足以满足您所需的序列长度
- 您需要在整个上下文中保持序列一致性

### 可用的实现

TRL 支持两种序列并行实现，每种实现都有不同的特征：1. **Ring Attention (FSDP2)** - 使用基于环的通信来实现极长序列的内存高效处理
2. **ALST/Ulysses (DeepSpeed)** - 使用注意力头并行性通过高带宽互连实现更快的训练

> [!重要]
> **序列长度术语：** 使用上下文并行时，序列会跨 GPU 分割，引入两个概念：
> - **全局序列长度**：跨 GPU 分割之前的完整序列长度
> - **微序列长度**：分割后每个GPU的序列长度
>
> 在TRL中，`max_seq_length`（或`max_length`）指的是**全局序列长度**。该框架自动处理分割成微序列：
> - **Ring Attention (FSDP2)**：使用`cp_size`来分割序列。使用`max_seq_length=8192`和`cp_size=4`，每个GPU处理2048个令牌。
> - **ALST/Ulysses (DeepSpeed)**：使用 `sp_size`（与 `sp_backend="deepspeed"`）分割序列。使用`max_seq_length=8192`和`sp_size=2`，每个GPU处理4096个令牌。
>
> 在计算批量大小和训练指标时，训练器会自动考虑上下文并行性。

### 在 Ring Attention 和 Ulysses 之间进行选择

下面的比较表突出显示了两种方法之间的主要区别：|特色 |环注意（FSDP2）| ALST/尤利西斯 (DeepSpeed) |
|---------|----------|------------------------|
| **方法** |环自注意力 |注意头部并行度|
| **后端** | PyTorch FSDP2 | DeepSpeed ZeRO |
| **注意** |仅 SDPA | Flash Attention 2 或 SDPA |
| **最小加速度** | 1.11.0+ | 1.12.0+ |
| **最低 DeepSpeed** |不适用 | 0.18.1+ |
| **序列可分性** | `cp_size * 2` | `sp_size` |
| **零阶段** |不适用 | ZeRO 阶段 1/2/3 |

**环注意在以下情况下效果更好：**
- 您需要处理极长的序列（1M+ 令牌）
- 该模型的注意力头数有限（环注意力不受头数限制）
- 您需要灵活地缩放到任何序列长度
- 网络拓扑有限（Ring Attention 适用于简单的 P2P 环通信）

**尤利西斯在以下情况下会更好：**
- 您拥有高带宽、低延迟互连（NVLink、InfiniBand）
- 该模型有许多注意力头，可以跨 GPU 分割
- 您想要较低的通讯量
- 对于中等长度的序列（最多约 500k 个标记），您希望获得更快的训练速度**关键权衡：**
- **通信量：** Ulysses 的通信量较低，通过良好的互连使其效率更高。 Ring Attention具有更高的通信量，但对于不同的网络拓扑更加灵活。
- **注意力头约束：** Ulysses 受到注意力头数量的限制（需要`num_heads >= sp_size`）。无论模型架构如何，环注意力都会随着序列长度而变化。
- **网络敏感性：** Ulysses 全方位通信对网络延迟很敏感。 Ring Attention 使用 P2P 环通信，更能容忍不同的网络条件。

详细对比请参见[Ulysses and Ring Attention blog post](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)。

### 环注意实现（FSDP2）

Ring Attention 使用类似环形的通信模式，其中每个 GPU 处理序列的一部分并将信息传递到环中的下一个 GPU。

#### 要求和限制1. **Accelerate 1.11.0 或更高版本** 需要环注意力/上下文并行支持
2. **需要 FSDP2 (PyTorch FSDP v2)** 作为分布式训练后端
3. **SDPA Attention** - 目前不支持 Flash Attention
4. **序列长度整除性** - 序列必须能被`cp_size * 2`整除。这是使用数据整理器中的 `pad_to_multiple_of` 参数自动处理的。

#### 配置

##### 加速配置

使用提供的加速配置文件之一（例如 [⟦T38⟧](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/context_parallel_2gpu.yaml) 对于 2 个 GPU）：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: true  # Enable activation checkpointing for memory efficiency
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2  # Number of GPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 1
  parallelism_config_tp_size: 1
  parallelism_config_cp_size: 2  # Context parallel size
```

##### 训练配置

```python
from trl import SFTConfig

training_args = SFTConfig(
    # required
    pad_to_multiple_of=4,           # ensures divisibility by cp_size * 2
    # to get the most out of CP
    max_length=16384,               # long sequence length
    packing=True,                   # use packing to reduce padding
    use_liger_kernel=True,          # compatible with CP
    gradient_checkpointing=False,   # The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously
    per_device_train_batch_size=1,
    ...
)
```

然后，使用适当的加速配置文件启动训练脚本：

```bash
accelerate launch --config_file context_parallel_2gpu.yaml train.py
```

#### 最佳实践

1. **使用 `pad_to_multiple_of` 参数** - 这是现在确保序列长度整除性的推荐方法：
   - 对于`cp_size=2`：使用`pad_to_multiple_of=4`（自`cp_size * 2 = 4`）
   - 对于`cp_size=4`：使用`pad_to_multiple_of=8`（自`cp_size * 2 = 8`）
   - 数据整理器自动将序列填充到所需的倍数，确保与 CP 的兼容性

2. **使用带填充的打包** - 默认的 BFD（最佳拟合递减）策略完美运行：
   - 保留序列边界并保持训练质量
   - 与 `padding_free=True` 和标准填充模式无缝配合3. **与其他内存优化相结合**，例如 Liger 内核、bfloat16 和梯度检查点

4. **从较小的上下文并行大小开始**（2-4 个 GPU），然后再进行扩展

5. **监控所有 GPU 的内存使用情况**，以确保平衡工作负载

#### 基准测试环注意力

我们对 Ring Attention 进行了基准测试，以强调其在训练效率方面的潜在改进。  
我们的实验是使用 **1、2、4 和 8 个 H100 GPU** 进行的，但结果可以扩展到具有更多节点和 GPU 的更大集群。

对于设置，我们使用提供的加速配置微调 **8B 模型** ([Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B))  
（[⟦T47⟧](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/context_parallel_2gpu.yaml)）。  
我们根据每次运行的 GPU 数量调整了 `num_processes` 和 `parallelism_config_cp_size`。  
使用[sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)示例脚本并结合上述参数进行训练。

下面的结果总结了不同数量 GPU 的**最大可训练序列长度**和**每秒迭代次数**。标记为 `OOM` 的值表示配置内存不足且无法训练。这些结果表明，**上下文并行性 (CP) 可通过更多 GPU 有效扩展**，从而能够对更长的序列进行训练。借助 **8 个 GPU**，超过 **300k token** 的上下文长度变得可行，解锁极长上下文的训练，同时保持合理的吞吐量。  

  
  

> [!提示]
> Accelerate 还支持**N 维并行（ND-parallelism）**，使您能够组合不同的并行化策略，在多个 GPU 上高效地分配模型训练。  
>
> 您可以在[Accelerate ND-parallelism guide](https://github.com/huggingface/accelerate/blob/main/examples/torch_native_parallelism/README.md#nd-parallelism)中了解更多信息并探索配置示例。

### ALST/Ulysses 实施 (DeepSpeed)

ALST（北极长序列训练）/Ulysses 使用注意力头并行性在 GPU 上分割长序列，并与 DeepSpeed 的 ZeRO 优化器配合使用。

> [!注意]
> **有关并行性配置的技术说明：**
> - **DeepSpeed ALST/Ulysses** 在 YAML 和 Python API 中使用 `sp_size` 和 `sp_backend="deepspeed"`
> - **Ring Attention (FSDP2)** 使用 `cp_size` 和 `cp_backend="torch"`
>
> 在计算有效批量大小和训练指标时，训练器会自动考虑 CP 和 SP。

#### 要求和限制1. **需要 DeepSpeed 0.18.1 或更高版本**
2. **Accelerate 1.12.0 或更高版本** 是 ALST/Ulysses 序列并行支持所必需的
3. **Attention 实现** - 推荐 Flash Attention 2（干净输出），SDPA 作为后备
4. **序列长度整除性** - 序列必须能被`sp_size`整除。在训练配置中使用`pad_to_multiple_of`。
5. **并行度配置** - 必须保证`dp_replicate_size × dp_shard_size × sp_size = num_processes`

#### 配置

##### 加速配置

使用提供的加速配置文件（[⟦T58⟧](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/alst_ulysses_4gpu.yaml)）：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  zero_stage: 3
  seq_parallel_communication_data_type: bf16
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: 1
num_processes: 4  # Number of GPUs
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 2  # Enables 2D parallelism with SP
  parallelism_config_tp_size: 1
  parallelism_config_sp_size: 2  # Sequence parallel size
  parallelism_config_sp_backend: deepspeed
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: flash_attention_2
```

##### 训练配置

```python
from trl import SFTConfig

training_args = SFTConfig(
    # required
    pad_to_multiple_of=2,    # Must equal sp_size
    # to get the most out of SP
    max_length=4096,
    packing=True,
    attn_implementation="flash_attention_2",
    per_device_train_batch_size=1,
    ...
)
```

然后，使用适当的加速配置文件启动训练脚本：

```bash
accelerate launch --config_file examples/accelerate_configs/alst_ulysses_4gpu.yaml train.py
```

#### 2D 并行性

上面的 4 个 GPU 配置通过将数据并行性 (DP) 与序列并行性 (SP) 相结合，自动启用 2D 并行性。对于`sp_size=2`和`dp_shard_size=2`，4个GPU的组织方式如下：
- 2个序列并行组（跨序列处理相同的数据）
- 2个数据并行组（处理不同数据）

要调整不同 GPU 数量的并行度，请修改 YAML 配置：| GPU | sp_size | dp_shard_size | dp_shard_size |使用案例| YAML 更改 |
|------|---------|----------------|---------|----------------|
| 4 | 2 | 2 |平衡-更长的序列+更多的数据| `num_processes: 4`、`sp_size: 2`、`dp_shard_size: 2` |
| 4 | 4 | 1 |最大序列长度的纯 SP | `num_processes: 4`、`sp_size: 4`、`dp_shard_size: 1` |
| 8 | 2 | 4 |大型培训 | `num_processes: 8`、`sp_size: 2`、`dp_shard_size: 4` |

#### 最佳实践

1. **使用`pad_to_multiple_of`**确保序列可被`sp_size`整除
2. **使用 Flash Attention 2** 进行干净的输出（SDPA 有效，但显示打包警告）
3. **从`sp_size=2`**开始，然后缩放到更大的值
4. **对大型模型使用 DeepSpeed ZeRO Stage 3**
5. **结合内存优化**，如 Liger 内核和梯度检查点
6. **验证并行配置**：确保`dp_replicate_size × dp_shard_size × sp_size = num_processes`

#### 完整示例

以下是如何使用内置 [⟦T74⟧](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) 脚本和 4 个 GPU 运行 ALST/Ulysses 训练：

```bash
accelerate launch --config_file examples/accelerate_configs/alst_ulysses_4gpu.yaml \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2e-4 \
    --max_steps 100 \
    --max_seq_length 4096 \
    --packing \
    --packing_strategy wrapped \
    --dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --output_dir output-alst-4gpu \
    --logging_steps 10 \
    --report_to trackio
```

该命令自动执行：
- 跨 4 个 GPU 配置 2D 并行性（SP=2，DP=2）
- 使用 Flash Attention 2 进行清洁训练
- 启用自动填充打包以确保序列可分性
- 利用 DeepSpeed ZeRO Stage 3 提高内存效率

### 进一步阅读#### 一般资源
- [Hugging Face Blog: Understanding Ulysses and Ring Attention](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention) - Ring Attention 与 Ulysses 方法的详细比较
- [Accelerate: Context Parallelism Guide](https://huggingface.co/docs/accelerate/concept_guides/context_parallelism)
- [Hugging Face Blog: Enabling Long-Context Training with Sequence Parallelism in Axolotl](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)

#### 环注意 (FSDP2)
- [Ultrascale Playbook - Context Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism)
- [Accelerate Example: 128k Sequence Length](https://github.com/huggingface/accelerate/blob/main/examples/torch_native_parallelism/README.md#context-parallelism-128k-sequence-length)
- [Accelerate ND-parallelism Guide](https://github.com/huggingface/accelerate/blob/main/examples/torch_native_parallelism/README.md#nd-parallelism)

#### ALST/尤利西斯（DeepSpeed）
- [DeepSpeed Sequence Parallelism Documentation](https://www.deepspeed.ai/tutorials/ds-sequence/)
- [Snowflake Engineering Blog: Arctic Long Sequence Training (ALST)](https://www.snowflake.com/en/engineering-blog/arctic-long-sequence-training-multi-million-token-ai/)

## 多节点训练

当单台机器没有足够的 GPU 时，TRL 可以使用 [🤗 Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#multi-node-training) 在多台机器（节点）上扩展训练。

### 加速配置
创建用于多节点训练的`accelerate`配置文件（例如`multi_node.yaml`）。关键领域：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 2
machine_rank: 0  # 0 for main node, 1 for second node
main_process_ip: 10.0.0.1  # IP of rank 0 node
main_process_port: 29500
num_processes: 16  # total processes across nodes
mixed_precision: bf16
use_cpu: false
same_network: true
```

调整 `num_processes` 以匹配所有节点上的 GPU 总数。

> [!注意]
> 将 `10.0.0.1` 替换为 0 级（主）节点的实际 IP 地址。

### 启动

#### 选项 1：手动启动（非 HPC）

在每个节点上手动运行以下命令：
```bash
# Node 0 (main node)
accelerate launch --config_file multi_node.yaml --machine_rank 0 train.py

# Node 1
accelerate launch --config_file multi_node.yaml --machine_rank 1 train.py
```
#### 选项 2：SLURM 启动（HPC 集群）

对于使用 SLURM 作业调度程序的集群，创建作业脚本（例如 `slurm_job.sh`）：
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --job-name=trl_multi

srun accelerate launch --config_file multi_node.yaml train.py
```

然后提交作业：
```bash
sbatch slurm_job.sh
```

SLURM 自动将训练分布在所有请求的节点和 GPU 上，并且`srun` 配置多节点通信所需的环境变量。

**关键 SLURM 指令：**
- `--nodes=2`：请求2个计算节点
- `--gpus-per-node=8`：为每个节点分配 8 个 GPU（总共 16 个）
- `--job-name`：用于在作业队列中跟踪的标签您可以通过设置`distributed_type: DEEPSPEED`并添加`deepspeed_config`块将多节点与DeepSpeed结合起来。请参阅[DeepSpeed integration guide](https://huggingface.co/docs/trl/en/deepspeed_integration)。

### 进一步阅读

- [Accelerate: Launching Scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [Accelerate: Example Zoo](https://huggingface.co/docs/accelerate/usage_guides/training_zoo)
- [SLURM Workload Manager Documentation](https://slurm.schedmd.com/) - 用于集群作业调度

### TRL - 变形金刚强化学习
https://huggingface.co/docs/trl/v1.9.2/index.md