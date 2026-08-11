<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 威震天-LM

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 能够大规模训练大型 Transformer 语言模型。
它为基于变压器的预训练提供了高效的张量、管道和基于序列的模型并行性
语言模型，例如[GPT](https://huggingface.co/papers/2005.14165)（仅解码器）、[BERT](https://huggingface.co/papers/1810.04805)（仅编码器）和[T5](https://huggingface.co/papers/1910.10683)（编码器-解码器）。
有关详细信息以及幕后工作原理，请参阅 github [repo](https://github.com/NVIDIA/Megatron-LM)。

## 什么是集成？

Accelerate 集成了 Megatron-LM 的以下功能，以实现大规模预训练/微调
BERT（编码器）、GPT（解码器）或 T5 模型（编码器和解码器）：

a. **张量并行性（TP）**：减少内存占用，而无需在节点内等级上进行太多额外的通信。
每个张量被分成多个块，每个分片驻留在单独的 GPU 上。在每个步骤中，都会处理相同的小批量数据
每个分片独立并行，然后在所有 GPU 之间同步（`all-reduce` 操作）。 
在简单的转换器层中，这会导致前向路径中有 2 个`all-reduces`，后向路径中有 2 个`all-reduces`。
欲了解更多详情，请参阅研究论文[Megatron-LM: Training Multi-Billion Parameter Language Models Using
Model Parallelism](https://huggingface.co/papers/1909.08053)和 
博文的这一部分[The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed#tensor-parallelism)。b. **管道并行性 (PP)**：减少内存占用并通过节点间并行化实现大规模训练。 
通过 PipeDream-Flush 计划/1F1B 计划和 Interleaved 1F1B 计划减少幼稚 PP 的泡沫。 
各层均匀分布在 PP 阶段。例如，如果模型有 `24` 层，而我们有 `4` GPU
管道并行性，每个 GPU 将有 `6` 层 (24/4)。有关减少 PP 闲置时间的时间表的更多详细信息，
请参阅研究论文[Efficient Large-Scale Language Model Training on GPU Clusters
Using Megatron-LM](https://huggingface.co/papers/2104.04473)和 
博文的这一部分[The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed#pipeline-parallelism)。c. **序列并行性 (SP)**：减少内存占用，无需任何额外的通信。仅在使用 TP 时适用。
它减少了所需的激活内存，因为它可以防止相同的副本出现在张量并行列上 
将 `all-reduce` 替换为 `reduce-scatter` 后，`no-op` 操作将被 `all-gather` 替换。 
作为`all-reduce = reduce-scatter + all-gather`，这可以节省大量的激活内存，而无需增加通信成本。 
简而言之，它沿着序列维度对每个 Transformer 层的输出进行分片，例如， 
如果序列长度为`1024`并且TP大小为`4`，则每个GPU将为每个样本拥有`256`令牌（1024/4）。 
这增加了可支持训练的批量大小。欲了解更多详情，请参阅研究论文
[Reducing Activation Recomputation in Large Transformer Models](https://huggingface.co/papers/2205.05198)。d. **通过分布式优化器实现数据并行性 (DP)**：通过跨 DP 等级分片优化器状态和梯度来减少内存占用
（与跨数据并行列复制优化器状态的传统方法相比）。 
例如，当使用Adam优化器进行混合精度训练时，每个参数占用12字节内存。
这在 GPU 之间平均分配，即，如果我们有 4 个 GPU，每个参数将占 3 个字节 (12/4)。
欲了解更多详细信息，请参阅研究论文[ZeRO: Memory Optimizations Toward Training Trillion
Parameter Models](https://huggingface.co/papers/1910.02054)和博客的以下部分 
[The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed#zero-data-parallelism)。e. **专家并行性 (EP)** Megatron-LM 中的专家并行性用于专家混合 (MoE) 层，其中存在许多“专家”（小型前馈网络），但每个令牌只激活少数专家。威震天不是将所有专家都放在每个 GPU 上，而是将不同的专家分布在不同的 GPU 上——这就是专家并行。在训练过程中，令牌被路由到托管所选专家的 GPU，在那里进行计算，然后发回，从而降低内存成本。它通常与大型模型的张量/管道并行性相结合。
f. **完整激活重新计算**：通过智能激活检查点显着减少激活的内存占用。
它不存储占用大量内存的激活，同时可以快速重新计算，从而在内存和重新计算之间实现了很好的权衡。
例如，对于 GPT-3，这会导致激活所需的内存减少 70%，但代价是
重新计算激活的 FLOP 开销仅为 2.7%。欲了解更多详情，请参阅研究论文 
[Reducing Activation Recomputation in Large Transformer Models](https://huggingface.co/papers/2205.05198)。g。 **融合内核**：融合Softmax、混合精度融合层范数和融合梯度累积来计算线性层的权重梯度。
PyTorch JIT编译了Fused GeLU和Fused Bias+Dropout+Residualaddition。

h. **支持索引数据集**：用于大规模训练的高效数据集二进制格式。支持`mmap`、`cached`索引文件和`lazy`加载器格式。

我。 **检查点重塑和互操作性**：用于重塑变量 Megatron-LM 检查点的实用程序 
张量和管道并行大小与深受喜爱的 Transformers 分片检查点相同，因为它拥有大量工具的大力支持
例如加速大模型推理、威震天-DeepSpeed 推理等。 
还支持将 Transformers 分片检查点转换为可变张量和管道并行大小的 Megatron-LM 检查点
用于大规模训练。  

## 先决条件 

您需要安装最新的 pytorch、cuda、nccl 和 NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) 版本以及 nltk 库。
更多详情请参见[documentation](https://github.com/NVIDIA/Megatron-LM#setup)。 
设置环境的另一种方法是从 NGC 获取包含所有必需安装的 NVIDIA PyTorch 容器。以下是设置 conda 环境的分步方法：

1.创建虚拟环境
```
conda create --name ml
```

2.假设机器安装了CUDA 11.3，安装对应的PyTorch GPU Version
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3.安装Nvidia APEX
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

4. 安装威震天-LM

```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 9a1c0d05c992c8a241da384ab27dce2021bb56dd
you need to manually move gpt_builders.py to megatron/training and update
include = [
    "megatron.core", 
    "megatron.core.*",
    "megatron.training",
    "megatron.training.*",
    "megatron.legacy",
    "megatron.legacy.*",
]
in pyproject.toml file to unblock yourself from using Megatron
pip install --no-use-pep517 -e .
```

## 准备 Megaton-LM 检查点
如果您想微调模型，请确保准备好 torch dist 格式检查点。如果您只能访问 Huggingface 模型，请考虑将其转换为威震天可接受的 torch dist 格式检查点。一个例子可以使用 slime 的脚本，以 GLM 模型为例：
```
source /your/path/to/slime/scripts/models/glm4.5-355B-A32B.sh
srun torchrun --nproc-per-node 8 \
   /your/path/to/slime/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /your/path/to/huggingface/models/GLM4.5-355B-A32B \
    --save /your/path/to/megatron/models/GLM4.5-355B-A32B_torch_dist

```
转换完成后，请确保： 1. `/your/path/to/megatron/models/GLM4.5-355B-A32B_torch_dist`下：将`latest_checkpointed_iteration.txt`的内容从`release`更改为`0`，并将目录`release`重命名为`iter_0000000`； 2：在配置中，确保 `megatron_lm_no_load_optim` 为 true，这样就不需要优化器状态。

## 加速 Megatron-LM 插件

重要功能通过`accelerate config`命令直接支持。 
使用 Megatron-LM 功能的相应问题的示例如下所示：

```bash
:~$ accelerate config --config_file "megatron_gpt_config.yaml"
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
Do you want to use DeepSpeed? [yes/NO]: 
Do you want to use FullyShardedDataParallel? [yes/NO]: 
Do you want to use Megatron-LM ? [yes/NO]: yes
What is the Tensor Parallelism degree/size? [1]:2
Do you want to enable Sequence Parallelism? [YES/no]: 
What is the Pipeline Parallelism degree/size? [1]:2
What is the number of micro-batches? [1]:2
Do you want to enable selective activation recomputation? [YES/no]: 
Do you want to use distributed optimizer which shards optimizer state and gradients across data parallel ranks? [YES/no]: 
What is the gradient clipping value based on global L2 Norm (0 to disable)? [1.0]: 
How many GPU(s) should be used for distributed training? [1]:4
Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: bf16
```

结果配置如下所示：

```
~$ cat megatron_gpt_config.yaml 
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MEGATRON_LM
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config:
  megatron_lm_gradient_clipping: 1.0
  megatron_lm_num_micro_batches: 2
  megatron_lm_pp_degree: 2
  megatron_lm_recompute_activations: true
  megatron_lm_sequence_parallelism: true
  megatron_lm_tp_degree: 2
  megatron_lm_use_distributed_optimizer: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
```我们以GPT预训练为例。官方`run_clm_no_trainer.py`所需的最小改动 
使用Megatron-LM的方法如下：

1. 由于Megatron-LM使用自己的Optimizer实现，因此需要使用与其兼容的相应调度器。
因此，仅支持 Megatron-LM 的调度程序。用户需要创建`accelerate.utils.MegatronLMDummyScheduler`。
示例如下：

```python
from accelerate.utils import MegatronLMDummyScheduler

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    lr_scheduler = MegatronLMDummyScheduler(
        optimizer=optimizer,
        total_num_steps=args.max_train_steps,
        warmup_num_steps=args.num_warmup_steps,
    )
else:
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
```

2. 现在获取总批量大小的详细信息需要了解张量和管道并行大小。
获取有效总批量大小的示例如下所示：

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    total_batch_size = accelerator.state.megatron_lm_plugin.global_batch_size
else:
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
```

3. 使用 Megatron-LM 时，损失已在数据并行组中进行平均

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    losses.append(loss)
else:
    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    losses = torch.tensor(losses)
else:
    losses = torch.cat(losses)
```

4.对于Megatron-LM，我们需要使用`accelerator.save_state`保存模型

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
    accelerator.save_state(args.output_dir)
else:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
```

就是这样！我们很高兴出发🚀。请在路径`accelerate/examples/by_feature/megatron_lm_gpt_pretraining.py`的示例文件夹中找到示例脚本。
让我们使用 4 个 A100-80GB GPU 来运行 `gpt-large` 模型架构。

```bash
accelerate launch --config_file megatron_gpt_config.yaml \
examples/by_feature/megatron_lm_gpt_pretraining.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--block_size 1024 \
--learning_rate 5e-5 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 5 \
--with_tracking \
--report_to "wandb" \
--output_dir "awesome_model"
```

以下是输出日志中的一些重要摘录：

```bash
Loading extension module fused_dense_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 3.569 seconds
 > padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
Building gpt model in the pre-training mode.
The Megatron LM model weights are initialized at random in `accelerator.prepare`. Please use `accelerator.load_checkpoint` to load a pre-trained checkpoint matching the distributed setup.
Preparing dataloader
Preparing dataloader
Preparing model
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 210753280
 > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 209445120
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 210753280
 > number of parameters on (tensor, pipeline) model parallel rank (0, 1): 209445120
Preparing optimizer
Preparing scheduler
> learning rate decay style: linear
10/10/2022 22:57:22 - INFO - __main__ - ***** Running training *****
10/10/2022 22:57:22 - INFO - __main__ -   Num examples = 2318
10/10/2022 22:57:22 - INFO - __main__ -   Num Epochs = 5
10/10/2022 22:57:22 - INFO - __main__ -   Instantaneous batch size per device = 24
10/10/2022 22:57:22 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 48
10/10/2022 22:57:22 - INFO - __main__ -   Gradient Accumulation steps = 1
10/10/2022 22:57:22 - INFO - __main__ -   Total optimization steps = 245
 20%|████████████▍                                                 | 49/245 [01:04<04:09,  1.27s/it]
 10/10/2022 22:58:29 - INFO - __main__ - epoch 0: perplexity: 1222.1594275215962 eval_loss: 7.10837459564209
 40%|████████████████████████▊                                     | 98/245 [02:10<03:07,  1.28s/it]
 10/10/2022 22:59:35 - INFO - __main__ - epoch 1: perplexity: 894.5236583794557 eval_loss: 6.796291351318359
 60%|████████████████████████████████████▌                        | 147/245 [03:16<02:05,  1.28s/it]
 10/10/2022 23:00:40 - INFO - __main__ - epoch 2: perplexity: 702.8458788508042 eval_loss: 6.555137634277344
 80%|████████████████████████████████████████████████▊            | 196/245 [04:22<01:02,  1.28s/it]
 10/10/2022 23:01:46 - INFO - __main__ - epoch 3: perplexity: 600.3220028695281 eval_loss: 6.39746618270874
100%|█████████████████████████████████████████████████████████████| 245/245 [05:27<00:00,  1.28s/it]
```

还有大量其他选项/功能可以使用 `accelerate.utils.MegatronLMPlugin` 设置。

## 利用编写自定义训练步骤和 Megatron-LM 索引数据集的高级功能要利用更多功能，请仔细阅读以下详细信息。

1. 以下是使用 Megatron-LM 时自定义训练步骤所需的更改示例。 
您将实现`accelerate.utils.AbstractTrainStep`或从其相应的子级继承 
`accelerate.utils.GPTTrainStep`、`accelerate.utils.BertTrainStep` 或 `accelerate.utils.T5TrainStep`。

```python
from accelerate.utils import MegatronLMDummyScheduler, GPTTrainStep, avg_losses_across_data_parallel_group

# Custom loss function for the Megatron model
class GPTTrainStepWithCustomLoss(GPTTrainStep):
    def __init__(self, megatron_args, **kwargs):
        super().__init__(megatron_args)
        self.kwargs = kwargs

    def get_loss_func(self):
        def loss_func(inputs, loss_mask, output_tensor):
            batch_size, seq_length = output_tensor.shape
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = losses.view(-1) * loss_mask

            # Resize and average loss per sample
            loss_per_sample = loss.view(batch_size, seq_length).sum(axis=1)
            loss_mask_per_sample = loss_mask.view(batch_size, seq_length).sum(axis=1)
            loss_per_sample = loss_per_sample / loss_mask_per_sample

            # Calculate and scale weighting
            weights = torch.stack([(inputs == kt).float() for kt in self.kwargs["keytoken_ids"]]).sum(axis=[0, 2])
            weights = 1.0 + self.kwargs["alpha"] * weights
            # Calculate weighted average
            weighted_loss = (loss_per_sample * weights).mean()

            # Reduce loss across data parallel groups
            averaged_loss = avg_losses_across_data_parallel_group([weighted_loss])

            return weighted_loss, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.loss_func, tokens, loss_mask)

        return forward_step

def main():
    # Custom loss function for the Megatron model
    keytoken_ids = []
    keywords = ["plt", "pd", "sk", "fit", "predict", " plt", " pd", " sk", " fit", " predict"]
    for keyword in keywords:
        ids = tokenizer([keyword]).input_ids[0]
        if len(ids) == 1:
            keytoken_ids.append(ids[0])
    accelerator.print(f"Keytoken ids: {keytoken_ids}")
    accelerator.state.megatron_lm_plugin.custom_train_step_class = GPTTrainStepWithCustomLoss
    accelerator.state.megatron_lm_plugin.custom_train_step_kwargs = {
        "keytoken_ids": keytoken_ids,
        "alpha": 0.25,
    }
```

2. 为了使用 Megatron-LM 数据集，还需要进行一些更改。这些数据集的数据加载器
仅在每个张量并行组的等级 0 上可用。因此，有些级别的数据加载器不会
可用，这需要对训练循环进行调整。能够做到这一切表明了如何
Accelerate 是灵活且可扩展的。所需的更改如下。

一个。对于 Megatron-LM 索引数据集，我们需要使用 `MegatronLMDummyDataLoader` 
并将所需的数据集参数传递给它，例如 `data_path`、`seq_length` 等。 
有关可用参数的列表，请参阅[here](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py#L804)。 
    
```python
from accelerate.utils import MegatronLMDummyDataLoader

megatron_dataloader_config = {
    "data_path": args.data_path,
    "splits_string": args.splits_string,
    "seq_length": args.block_size,
    "micro_batch_size": args.per_device_train_batch_size,
}
megatron_dataloader = MegatronLMDummyDataLoader(**megatron_dataloader_config)
accelerator.state.megatron_lm_plugin.megatron_dataset_flag = True
```

b. `megatron_dataloader` 重复 3 次以获得训练、验证和测试数据加载器
按照`args.splits_string`比例
    
```python
model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, _ = accelerator.prepare(
    model, optimizer, lr_scheduler, megatron_dataloader, megatron_dataloader, megatron_dataloader
)
```c.训练和评估循环的更改，因为数据加载器仅适用于张量并行等级 0
因此，只有当数据加载器不是 `None` 时我们才需要迭代，否则提供空字典
因此，我们使用 `while` 循环进行循环，并在 `completed_steps` 等于 `args.max_train_steps` 时中断
这类似于 Megatron-LM 设置，其中用户在使用 Megaton-LM 索引数据集时必须提供`max_train_steps`。
这显示了 Accelerate 的灵活性和可扩展性。

```python
while completed_steps < args.max_train_steps:
    model.train()
    batch = next(train_dataloader) if train_dataloader is not None else {}
    outputs = model(**batch)
    loss = outputs.loss
    ...

    if completed_steps % eval_interval == 0:
        eval_completed_steps = 0
        losses = []
        while eval_completed_steps < eval_iters:
            model.eval()
            with torch.no_grad():
                batch = next(eval_dataloader) if eval_dataloader is not None else {}
                outputs = model(**batch)
```

    
## 检查点重塑和互操作性实用程序

1. 这些脚本存在于相应模型下的 Transformers 库中。 
目前支持GPT型号[checkpoint_reshaping_and_interoperability.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py)

2. 下面是检查点从 Megatron-LM 转换为通用 Transformers 分片检查点的示例。
```bash
python checkpoint_reshaping_and_interoperability.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "gpt/iter_0005000" \
--save_path "gpt/trfs_checkpoint" \
--max_shard_size "200MB" \
--tokenizer_name "gpt2" \
--print-checkpoint-structure
```

3. 使用`tp_size=2`、`pp_size=2`和`dp_size=2`将检查点从变形金刚转换为威震天。
```bash
python checkpoint_utils/megatgron_gpt2/checkpoint_reshaping_and_interoperability.py \
--load_path "gpt/trfs_checkpoint" \
--save_path "gpt/megatron_lm_checkpoint" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 2 \
--target_data_parallel_size 2 \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 128 \
--use_distributed_optimizer \
--print-checkpoint-structure
```

## Megatron-LM GPT 模型支持返回 logits 和用于文本生成的`megatron_generate` 函数

1. 返回logits需要在MegatronLMPlugin中设置`require_logits=True`，如下所示。 
这些将在管道的最后阶段提供。
```python
megatron_lm_plugin = MegatronLMPlugin(return_logits=True)
```2. Megatron-LM GPT模型的`megatron_generate`方法：这将使用Tensor和Pipeline Parallelism来完成 
当使用带/不带top_k/top_p采样的贪婪时生成一批输入，以及当使用波束搜索解码时生成单独的提示输入。 
仅支持 Transformer 生成的功能的子集。这将有助于通过张量和管道并行性使用大型模型 
用于生成（已经进行键值缓存并默认使用融合内核）。
这要求数据并行大小为 1，序列并行性和激活检查点被禁用。
它还需要指定分词器的词汇文件和合并文件的路径。 
以下示例展示了如何为 Megatron-LM GPT 模型配置和使用`megatron_generate`方法。
```python
# specifying tokenizer's vocab and merges file
vocab_file = os.path.join(args.resume_from_checkpoint, "vocab.json")
merge_file = os.path.join(args.resume_from_checkpoint, "merges.txt")
other_megatron_args = {"vocab_file": vocab_file, "merge_file": merge_file}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)

# inference using `megatron_generate` functionality
tokenizer.pad_token = tokenizer.eos_token
max_new_tokens = 64
batch_texts = [
    "Are you human?",
    "The purpose of life is",
    "The arsenal was constructed at the request of",
    "How are you doing these days?",
]
batch_encodings = tokenizer(batch_texts, return_tensors="pt", padding=True)

# top-p sampling
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"],
    batch_encodings["attention_mask"],
    max_new_tokens=max_new_tokens,
    top_p=0.8,
    top_p_decay=0.5,
    temperature=0.9,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# top-k sampling
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"],
    batch_encodings["attention_mask"],
    max_new_tokens=max_new_tokens,
    top_k=50,
    temperature=0.9,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# adding `bos` token at the start
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"], batch_encodings["attention_mask"], max_new_tokens=max_new_tokens, add_BOS=True
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# beam search => only takes single prompt
batch_texts = ["The purpose of life is"]
batch_encodings = tokenizer(batch_texts, return_tensors="pt", padding=True)
generated_tokens = model.megatron_generate(
    batch_encodings["input_ids"],
    batch_encodings["attention_mask"],
    max_new_tokens=max_new_tokens,
    num_beams=20,
    length_penalty=1.5,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)
```

3. 使用 `megatron_generate` 方法用于 Megatron-LM GPT 模型的端到端示例位于
[megatron_gpt2_generation.py](https://github.com/pacman100/accelerate-megatron-test/blob/main/src/inference/megatron_gpt2_generation.py) 与 
配置文件[megatron_lm_gpt_generate_config.yaml](https://github.com/pacman100/accelerate-megatron-test/blob/main/src/Configs/megatron_lm_gpt_generate_config.yaml).
带有加速启动命令的 bash 脚本可在 [megatron_lm_gpt_generate.sh](https://github.com/pacman100/accelerate-megatron-test/blob/main/megatron_lm_gpt_generate.sh) 获得。
脚本的输出日志可在[megatron_lm_gpt_generate.log](https://github.com/pacman100/accelerate-megatron-test/blob/main/output_logs/megatron_lm_gpt_generate.log)获得。

## 支持 ROPE 和 ALiBi 位置嵌入和多查询注意力

1. 对于 ROPE/ALiBi 注意力，将 `position_embedding_type` 和 `("absolute" | "rotary" | "alibi")` 传递到 `MegatronLMPlugin`，如下所示。
```python
other_megatron_args = {"position_embedding_type": "alibi"}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)
```2. 对于多查询注意力机制，将`attention_head_type`和`("multihead" | "multiquery")`传递给`MegatronLMPlugin`，如下所示。
```python
other_megatron_args = {"attention_head_type": "multiquery"}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)
```

## 注意事项

1. 支持变形金刚 GPT2、Megatron-BERT 和 T5 型号。
这涵盖仅解码器、仅编码和编码器-解码器模型类。

2. 模型前向传递仅返回损失，如下所示 
管道、张量和数据并行性在幕后有相当复杂的相互作用。
`model(**batch_data)` 调用返回损耗在数据并行队列中平均。
这对于大多数情况来说都很好，其中预训练作业使用 Megatron-LM 功能运行，并且
您可以使用损失轻松计算`perplexity`。 
对于 GPT 模型，除了损失之外还支持返回 logits。 
这些 logits 不是跨数据并行等级收集的。使用`accelerator.utils.gather_across_data_parallel_groups`
跨数据并行等级收集逻辑。这些逻辑和标签可用于计算各种 
绩效指标。 

3. 主要过程是最后一个等级，因为损失/logits 在管道的最后阶段可用。
使用时，`accelerator.is_main_process`和`accelerator.is_local_main_process`返回最后排名`True` 
威震天-LM 集成。4. 在`accelerator.prepare`调用中，创建与给定 Transformers 模型对应的 Megatron-LM 模型
具有随机权重。请使用`accelerator.load_state`加载具有匹配TP、PP和DP分区的Megatron-LM检查点。

5. 目前，检查点重塑和互操作性支持仅适用于 GPT。 
很快就会扩展到BERT和T5。

6. `gradient_accumulation_steps` 需要为 1. 使用 Megatron-LM 时，管道并行中的微批次 
设置与梯度累积同义。 

7. 使用Megatron-LM时，使用`accelerator.save_state`和`accelerator.load_state`保存和加载检查点。

8. 下面是从 Megatron-LM 模型架构到等效 Transformer 模型架构的映射。
仅支持这些 Transformer 模型架构。

一个。威震天-LM [BertModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/bert_model.py) : 
配置模型类型中带有`megatron-bert`的变压器模型，例如， 
[MegatronBERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert)
    
b.威震天-LM [GPTModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py) : 
配置模型类型中带有`gpt2`的变压器模型，例如， 
[OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
   
c.威震天-LM [T5Model](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/t5_model.py) : 
配置模型类型中带有`t5`的变压器模型，例如， 
[T5](https://huggingface.co/docs/transformers/model_doc/t5) 和 
[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)

### 英特尔高迪
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/gaudi.md