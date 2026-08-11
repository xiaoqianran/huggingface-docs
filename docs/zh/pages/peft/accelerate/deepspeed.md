<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 深速

[DeepSpeed](https://www.deepspeed.ai/) 是一个专为具有数十亿参数的大型模型的分布式训练的速度和规模而设计的库。其核心是零冗余优化器 (ZeRO)，它将优化器状态 (ZeRO-1)、梯度 (ZeRO-2) 和参数 (ZeRO-3) 跨数据并行进程进行分片。这大大减少了内存使用量，使您能够将训练扩展到十亿个参数模型。为了释放更高的内存效率，ZeRO-Offload 在优化过程中利用 CPU 资源来减少 GPU 计算和内存。

🤗 Accelerate 支持这两个功能，您可以将它们与 🤗 PEFT 一起使用。 

## 兼容`bitsandbytes`量化+LoRA

下表总结了 PEFT 的 LoRA、[⟦T20⟧](https://github.com/TimDettmers/bitsandbytes) 库和 DeepSpeed Zero 阶段在微调方面的兼容性。 DeepSpeed Zero-1 和 2 对推理没有影响，因为阶段 1 对优化器状态进行分片，而阶段 2 对优化器状态和梯度进行分片：

| DeepSpeed 阶段 |兼容吗？ |
|---|---|
|零-1 |  🟢 |
|零2 |  🟢 |
|零-3 |  🟢 |

对于 DeepSpeed Stage 3 + QLoRA，请参阅下面的[Use PEFT QLoRA and DeepSpeed with ZeRO3 for finetuning large models on multiple GPUs](#use-peft-qlora-and-deepspeed-with-zero3-for-finetuning-large-models-on-multiple-gpus)部分。为了确认这些观察结果，我们使用 QLoRA + PEFT 和可用的加速配置 [here](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs) 运行 [Transformers Reinforcement Learning (TRL) library](https://github.com/huggingface/trl) 的 SFT（监督微调）[official example scripts](https://github.com/huggingface/trl/tree/main/examples)。我们在 2x NVIDIA T4 GPU 上运行这些实验。

# 将 PEFT 和 DeepSpeed 与 ZeRO3 结合使用，在多个设备和多个节点上微调大型模型

本部分指南将帮助您了解如何使用我们的 DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/sft/train.py) 执行 SFT。您将配置脚本以在单台计算机上的 8xH100 80GB GPU 上使用 LoRA 和 ZeRO-3 对 Llama-70B 模型执行 SFT（监督微调）。您可以通过更改加速配置将其配置为扩展到多台计算机。

## 配置

首先使用 🤗 Accelerate 对 [create a DeepSpeed configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) 运行以下命令。 `--config_file` 标志允许您将配置文件保存到特定位置，否则它将作为 `default_config.yaml` 文件保存在🤗 Accelerate 缓存中。

配置文件用于在启动训练脚本时设置默认选项。

```bash
accelerate config --config_file deepspeed_config.yaml
```

系统会询问您一些有关您的设置的问题，并配置以下参数。在此示例中，您将使用 ZeRO-3，因此请确保选择这些选项。

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them. Pass the same value as you would pass via cmd argument else you will encounter mismatch error.
`gradient_clipping`: Enable gradient clipping with value. Don't set this as you will be passing it via cmd arguments.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2. Set this as `none` as don't want to enable offloading.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3. Set this as `none` as don't want to enable offloading.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3. Set this to `True`.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3. Set this to `True`.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. Set this to `True`.
```完成后，相应的配置应如下所示，您可以在 [deepspeed_config.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/deepspeed_config.yaml) 的 config 文件夹中找到它：

```yml
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 4
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 启动命令

启动命令位于[run_peft_deepspeed.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_deepspeed.sh)，如下所示：
```bash
accelerate launch --config_file "configs/deepspeed_config.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_steps 0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-lora-deepspeed" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False
```

请注意，我们使用的 LoRA 等级为 8，alpha=16，并针对所有线性层。我们正在传递 deepspeed 配置文件并在 ultrachat 数据集的子集上微调 70B Llama 模型。

## 重要部分

让我们更深入地研究一下脚本，以便您可以了解发生了什么，并了解它是如何工作的。

首先要知道的是，该脚本使用 DeepSpeed 进行分布式训练，因为 DeepSpeed 配置已通过。 [SFTTrainer](https://huggingface.co/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 类处理使用传递的 peft 配置创建 PEFT 模型的所有繁重工作。之后，当您调用 `trainer.train()` 时，[SFTTrainer](https://huggingface.co/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 内部使用 🤗 Accelerate 准备模型、优化器和训练器，使用 DeepSpeed 配置创建 DeepSpeed 引擎，然后对其进行训练。主要代码片段如下：

```python
# trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)
trainer.accelerator.print(f"{trainer.model}")

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
trainer.save_model()
```

## 内存使用情况

在上面的示例中，每个 GPU 消耗的内存为 64 GB (80%)，如下面的屏幕截图所示：

    

训练运行的 GPU 内存使用情况## 更多资源
您还可以参考这篇博客文章[Falcon 180B Finetuning using 🤗 PEFT and DeepSpeed](https://medium.com/@sourabmangrulkar/falcon-180b-finetuning-using-peft-and-deepspeed-b92643091d99)，了解如何在 2 台机器上的 16 个 A100 GPU 上微调 180B Falcon 模型。

# 使用 PEFT QLoRA 和 DeepSpeed 与 ZeRO3 在多个 GPU 上微调大型模型

在本节中，我们将了解如何使用 QLoRA 和 DeepSpeed Stage-3 在 2X40GB GPU 上微调 70B llama 模型。
为此，我们首先需要`bitsandbytes>=0.43.3`、`accelerate>=1.0.1`、`transformers>4.44.2`、`trl>0.11.4`和`peft>0.13.0`。使用 Accelerate 配置时，我们需要将 `zero3_init_flag` 设置为 true。以下是可以在[deepspeed_config_z3_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/deepspeed_config_z3_qlora.yaml)找到的配置：

```yml
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
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

下面给出了启动命令，可在 [run_peft_qlora_deepspeed_stage3.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_qlora_deepspeed_stage3.sh) 获得：
```
accelerate launch --config_file "configs/deepspeed_config_z3_qlora.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_steps 0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-qlora-dsz3" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16"
```

请注意，传递的新参数 `bnb_4bit_quant_storage_dtype` 表示用于打包 4 位参数的数据类型。例如，当它设置为 `bfloat16` 时，**32/4 = 8** 4 位参数在量化后打包在一起。

在训练代码方面，重要的代码变化是： 

```diff
...

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit_quantization,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
+   bnb_4bit_quant_storage=quant_storage_dtype,
)

...

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
+   dtype=quant_storage_dtype or torch.float32,
)
```

请注意，`AutoModelForCausalLM` 的`dtype` 与`bnb_4bit_quant_storage` 数据类型相同。就是这样。其他一切都由 Trainer 和 TRL 处理。

## 内存使用情况在上面的示例中，每个 GPU 消耗的内存为 **36.6 GB**。因此，需要 8X80GB GPU（采用 DeepSpeed Stage 3+LoRA）和几个 80GB GPU（采用 DDP+QLoRA），现在需要 2X40GB GPU。这使得大型模型的微调变得更加容易。

# 将 PEFT 和 DeepSpeed 与 ZeRO3 和 CPU 卸载结合使用，在单个 GPU 上微调大型模型
本部分指南将帮助您学习如何使用我们的 DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py)。您将配置脚本来训练大型模型，以使用 ZeRO-3 和 CPU Offload 进行条件生成。

> [!提示]
> 💡 为了帮助您入门，请查看我们的 [causal language modeling](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py) 和 [conditional generation](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py) 示例训练脚本。您可以根据自己的应用程序调整这些脚本，如果您的任务与脚本中的任务类似，甚至可以直接使用它们。

## 配置

首先使用 🤗 Accelerate 对 [create a DeepSpeed configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) 运行以下命令。 `--config_file` 标志允许您将配置文件保存到特定位置，否则它将作为 `default_config.yaml` 文件保存在🤗 Accelerate 缓存中。

配置文件用于在启动训练脚本时设置默认选项。

```bash
accelerate config --config_file ds_zero3_cpu.yaml
```系统会询问您一些有关您的设置的问题，并配置以下参数。在此示例中，您将使用 ZeRO-3 和 CPU-Offload，因此请确保选择这些选项。

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. 
```

示例 [configuration file](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/accelerate_ds_zero3_cpu_offload_config.yaml) 可能如下所示。最需要注意的是，`zero_stage`设置为`3`，`offload_optimizer_device`和`offload_param_device`设置为`cpu`。

```yml
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
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

## 重要部分

让我们更深入地研究一下脚本，以便您可以了解发生了什么，并了解它是如何工作的。

在 [⟦T42⟧](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py#L103) 函数中，脚本创建一个 [Accelerator](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 类来初始化分布式训练的所有必要要求。

> [!提示]
> 💡 随意更改 `main` 函数内的模型和数据集。如果您的数据集格式与脚本中的格式不同，您可能还需要编写自己的预处理函数。该脚本还为您正在使用的 🤗 PEFT 方法创建一个配置，在本例中为 LoRA。 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)指定了任务类型和重要参数，例如低秩矩阵的维度、矩阵缩放因子和LoRA层的丢失概率。如果您想使用不同的 🤗 PEFT 方法，请确保将 `LoraConfig` 替换为适当的 [class](../package_reference/tuners)。

```diff
 def main():
+    accelerator = Accelerator()
     model_name_or_path = "facebook/bart-large"
     dataset_name = "twitter_complaints"
+    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
```

在整个脚本中，您将看到 [main_process_first](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.main_process_first) 和 [wait_for_everyone](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.wait_for_everyone) 函数，它们有助于控制和同步进程的执行。

[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 函数采用一个基本模型和您之前准备创建 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 的 `peft_config`：

```diff
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
```

将所有相关的训练对象传递给 🤗 Accelerate 的 [prepare](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)，以确保一切都准备好进行训练：

```py
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
)
```

接下来的代码检查`Accelerator`中是否使用了DeepSpeed插件，如果该插件存在，那么我们检查是否使用ZeRO-3。当模型参数分片时，在推理期间调用 `generate` 函数调用以同步 GPU 时，将使用此条件标志：

```py
is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
```

在训练循环中，通常的 `loss.backward()` 被🤗 Accelerate 的 [backward](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.backward) 取代，后者根据您的配置使用正确的 `backward()` 方法：

```diff
  for epoch in range(num_epochs):
      with TorchTracemalloc() as tracemalloc:
          model.train()
          total_loss = 0
          for step, batch in enumerate(tqdm(train_dataloader)):
              outputs = model(**batch)
              loss = outputs.loss
              total_loss += loss.detach().float()
+             accelerator.backward(loss)
              optimizer.step()
              lr_scheduler.step()
              optimizer.zero_grad()
```仅此而已！脚本的其余部分处理训练循环、评估，甚至将其推送到中心。

## 火车

运行以下命令来启动训练脚本。之前，您将配置文件保存到 `ds_zero3_cpu.yaml`，因此您需要使用 `--config_file` 参数将路径传递给启动器，如下所示：

```bash
accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
```

您将看到一些跟踪训练期间内存使用情况的输出日志，一旦完成，脚本就会返回准确性并将预测与标签进行比较：

```bash
GPU Memory before entering the train : 1916
GPU Memory consumed at the end of the train (end-begin): 66
GPU Peak Memory consumed during the train (max-begin): 7488
GPU Total Peak Memory consumed during the train (max): 9404
CPU Memory before entering the train : 19411
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 19411
epoch=4: train_ppl=tensor(1.0705, device='cuda:0') train_epoch_loss=tensor(0.0681, device='cuda:0')
100%|████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:27<00:00,  3.92s/it]
GPU Memory before entering the eval : 1982
GPU Memory consumed at the end of the eval (end-begin): -66
GPU Peak Memory consumed during the eval (max-begin): 672
GPU Total Peak Memory consumed during the eval (max): 2654
CPU Memory before entering the eval : 19411
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 19411
accuracy=100.0
eval_preds[:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
```

# 注意事项
1. 目前不支持使用 PEFT 和 DeepSpeed 时进行合并，并且会引发错误。
2. 使用 CPU 卸载时，使用 PEFT 缩小优化器状态和适配器权重梯度的主要收益将在 CPU RAM 上实现，并且不会节省 GPU 内存。
3. 与禁用 CPU 卸载相比，DeepSpeed Stage 3 和 qlora 与 CPU 卸载一起使用时会导致更多 GPU 内存使用。 

> [!提示]
> 💡 当您有需要合并（和取消合并）权重的代码时，请尝试事先使用 DeepSpeed Zero-3 手动收集参数：
>
> ```python
> import deepspeed
>
> is_ds_zero_3 = ... # check if Zero-3
>
> with deepspeed.zero.GatheredParameters(list(model.parameters()), enabled= is_ds_zero_3):
>     model.merge_adapter()
>     # do whatever is needed, then unmerge in the same context if unmerging is required
>     ...
>     model.unmerge_adapter()
> ```### 完全分片数据并行
https://huggingface.co/docs/peft/v0.20.0/accelerate/fsdp.md