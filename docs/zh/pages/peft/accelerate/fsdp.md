<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 完全分片数据并行

[Fully sharded data parallel](https://pytorch.org/docs/stable/fsdp.html) (FSDP) 是为高达 1T 参数的大型预训练模型的分布式训练而开发的。 FSDP 通过跨数据并行进程对模型参数、梯度和优化器状态进行分片来实现这一点，并且它还可以将分片的模型参数卸载到 CPU。 FSDP 提供的内存效率允许您将训练扩展到更大的批次或模型大小。

PEFT 支持 FSDP1 和 [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。您可以通过 Transformers [Trainer](https://huggingface.co/docs/transformers/v5.23.0/en/main_classes/trainer#transformers.Trainer) 配置 FSDP 或直接在 PyTorch 训练循环中使用它。

## FSDP2 与变形金刚

使用 PyTorch 2.6 或更新版本和当前版本的 Transformers、Accelerate、TRL 和 PEFT。 [SFT example](https://github.com/huggingface/peft/tree/main/examples/sft) 使用[SFTTrainer](https://huggingface.co/docs/trl/v1.18.1/en/sft_trainer#trl.SFTTrainer)。以下是典型的 FSDP2 配置：

```json
{
  "fsdp": true,
  "fsdp_config": {
    "version": 2,
    "reshard_after_forward": true,
    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "cpu_ram_efficient_loading": true,
    "state_dict_type": "SHARDED_STATE_DICT",
    "activation_checkpointing": false
  }
}
```

`TRANSFORMER_BASED_WRAP` 使用模型的转换器层定义。 `reshard_after_forward` 在每次前向传递后释放收集的参数。 FSDP2 对各个参数进行分片，因此它不使用 FSDP1 的 `use_orig_params` 设置。更多配置选项请参见[Transformers FSDP guide](https://huggingface.co/docs/transformers/fsdp)。

编辑示例的 [fsdp2_config.json](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp2_config.json) 中的模型、数据集和训练选项。从 `examples/sft` 开始，为每个 GPU 启动一个进程：

```bash
torchrun --standalone --nproc_per_node=2 train.py configs/fsdp2_config.json
```

在不支持 BF16 的 GPU 上，在 JSON 文件中设置 `"bf16": false` 和 `"fp16": true`。对于 v0.22 之前的 PEFT 版本，请将 `activation_checkpointing` 设置为 `false` 以避免空适配器导出。 PEFT v0.22 包括 [adapter export fix](https://github.com/huggingface/peft/pull/4108)。

### 保存并恢复

要恢复训练，请保留相同的 FSDP 配置并在 JSON 文件中设置`"resume_from_checkpoint": "path/to/checkpoint-N"`。训练检查点包含分片模型和优化器状态。

训练结束时，该示例在导出适配器之前收集完整的状态字典：

```python
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
```

所有队伍必须参加集合；只有主进程写入适配器文件。收集完整状态字典会暂时增加内存使用量。

为了进行推理，加载导出的适配器：

```python
import torch

from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("qwen-sft-lora-fsdp2", dtype=torch.float32)
```

如果 SFT 运行添加了增加基本模型嵌入的标记，请在加载适配器之前匹配训练时嵌入大小。 SFT 示例将增长的嵌入填充为 8 的倍数：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel

adapter_path = "qwen-sft-lora-fsdp2"
config = PeftConfig.from_pretrained(adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, dtype=torch.float32)
# Match the SFT example: grow the embeddings if needed, but never shrink them.
if len(tokenizer) > base_model.get_input_embeddings().num_embeddings:
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(base_model, adapter_path)
```

## FSDP1 加速

本节使用 SFT [training script](https://github.com/huggingface/peft/blob/main/examples/sft/train.py) 在单台机器上的 8xH100 80GB GPU 上对 Llama-70B 与 LoRA 和 FSDP1 进行微调。您可以通过更改 Accelerate 配置来扩展到多台计算机。

＃＃＃ 配置首先使用 🤗 Accelerate 对 [create a FSDP configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) 运行以下命令。 `--config_file` 标志允许您将配置文件保存到特定位置，否则它将作为 `default_config.yaml` 文件保存在🤗 Accelerate 缓存中。

配置文件用于在启动训练脚本时设置默认选项。

```bash
accelerate config --config_file fsdp_config.yaml
```

系统会询问您一些有关您的设置的问题，并配置以下参数。在此示例中，您将回答如下图所示的调查问卷。

    

创建 Accelerate 的配置以使用 FSDP

完成后，相应的配置应如下所示，您可以在 [fsdp_config.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config.yaml) 的 config 文件夹中找到它：

```yml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_version: 1
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
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

### 启动命令

启动命令位于[run_peft_fsdp.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_fsdp.sh)，如下所示：
```bash
accelerate launch --config_file "configs/fsdp_config.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_length 2048 \
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
--output_dir "llama-sft-lora-fsdp" \
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

请注意，我们使用的 LoRA 等级为 8，alpha=16，并针对所有线性层。我们正在传递 FSDP 配置文件并在 [ultrachat dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) 的子集上微调 70B Llama 模型。

### 重要部分

让我们更深入地研究一下脚本，以便您可以了解发生了什么，并了解它是如何工作的。首先要知道的是，该脚本使用 FSDP 进行分布式训练，因为 FSDP 配置已通过。 [SFTTrainer](https://huggingface.co/docs/trl/v1.18.1/en/sft_trainer#trl.SFTTrainer) 类处理使用传递的 peft 配置创建 PEFT 模型的所有繁重工作。之后，当您调用 `trainer.train()` 时，Trainer 在内部使用 🤗 Accelerate 准备模型，优化器和训练器使用 FSDP 配置创建 FSDP 包装模型，然后进行训练。主要代码片段如下：

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
if hasattr(trainer.model, "print_healthcheck"):
    trainer.model.print_healthcheck()

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
```

使用 FSDP1，`use_orig_params=False` 通过分别包装可训练参数和冻结参数来减少内存使用。 Current Transformers Trainer 为此案例安装了 PEFT 的 `fsdp_auto_wrap_policy`；该示例不需要手动设置。自定义 Accelerate FSDP1 训练循环必须在准备模型之前设置该策略。此平坦参数包装要求不适用于 FSDP2。

### 内存使用情况

在上面的示例中，每个 GPU 消耗的内存为 72-80 GB (90-98%)，如下面的屏幕截图所示。最后GPU内存略有增加是在使用`FULL_STATE_DICT`状态字典类型而不是`SHARDED_STATE_DICT`保存模型时，以便模型具有可以在推理过程中使用`from_pretrained`方法正常加载的适配器权重：

    

训练运行的 GPU 内存使用情况## FSDP1 与 QLoRA

在本节中，我们将了解如何使用 QLoRA 和 FSDP 在 2X24GB GPU 上微调 70B llama 模型。 [Answer.AI](https://www.answer.ai/) 与bitsandbytes 和 Hugging Face 合作🤗 开源代码，支持使用 FSDP+QLoRA，并在其富有洞察力的博文[You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) 中解释了整个过程。现在它已集成到 Hugging Face 生态系统中。 

使用当前的 PEFT、Transformers、TRL 和 Accelerate 以及 `bitsandbytes>=0.43.3`。对于此 FSDP1 配置，设置 `fsdp_cpu_ram_efficient_loading=true`、`fsdp_use_orig_params=false` 和 `fsdp_offload_params=true`（CPU 卸载）。该配置可在[fsdp_config_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config_qlora.yaml)获得：

```yml
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false                                                                                                                                                                 
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_version: 1
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

下面给出了启动命令，可在[run_peft_qlora_fsdp.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_qlora_fsdp.sh)获得：
```
accelerate launch --config_file "configs/fsdp_config_qlora.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_length 2048 \
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
--output_dir "llama-sft-qlora-fsdp" \
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

请注意传递的新参数`bnb_4bit_quant_storage_dtype`，它表示用于打包 4 位参数的数据类型。例如，当它设置为 `bfloat16` 时，**16/4 = 4** 4 位参数在量化后打包在一起。当使用`bfloat16`进行混合精度训练时，`bnb_4bit_quant_storage_dtype`可以是`bfloat16`（用于纯`bfloat16`微调），也可以是`float32`用于自动混合精度（这会消耗更多GPU内存）。当使用`float16`进行混合精度训练时，`bnb_4bit_quant_storage_dtype`应设置为`float32`，以实现稳定的自动混合精度训练。

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
```请注意，`AutoModelForCausalLM` 的`dtype` 与`bnb_4bit_quant_storage` 数据类型相同。就是这样。其他一切都由 Trainer 和 TRL 处理。

### 内存使用情况

在上面的示例中，每个 GPU 消耗的内存为 **19.6 GB**，而 CPU RAM 使用量约为 **107 GB**。禁用 CPU 卸载时，GPU 内存使用率为 **35.6 GB/GPU**。因此，之前需要 16X80GB GPU 进行全面微调，8X80GB GPU 采用 FSDP+LoRA，以及几个 80GB GPU 采用 DDP+QLoRA，现在需要 2X24GB GPU。这使得大型模型的微调变得更加容易。

## 更多资源
您还可以参考 [llama-recipes](https://github.com/facebookresearch/llama-recipes/?tab=readme-ov-file#fine-tuning) 存储库和 [Getting started with Llama](https://llama.meta.com/get-started/#fine-tuning) 指南，了解如何使用 FSDP 和 PEFT 进行微调。## 注意事项
1. 对于 FSDP1（上面使用的 Accelerate FSDP 集成），不支持合并并会引发错误。使用 FSDP2 (`fully_shard`)、`merge_adapter()` 和 `unmerge_adapter()` 达到分片参数：在 `Linear`、`Embedding` 上合并普通 LoRA（包括 `safe_merge`）和卷积基础层可以正常工作，就像`Linear` 上的 DoRA 和 (IA)^3 一样。这些方法未涵盖其他适配器或层组合，它们要么引发错误，要么可能默默地无法合并，因此在依赖结果之前请验证结果。一个已知的例子是 DoRA 层，其 `base_layer` 被其自己的 FSDP2 模块包装，从而引发。
2. 传递`modules_to_save`配置参数目前尚未测试。
3. 使用 CPU Offloading 时的 GPU 内存节省目前未经测试。
4. 使用 FSDP+QLoRA 时，`paged_adamw_8bit` 目前会导致保存检查点时出错。
5. 使用 FSDP 进行 DoRA 训练应该有效（尽管速度低于 LoRA）。如果与位和字节 (QDoRA) 结合使用，4 位量化也应该可以工作，但 8 位量化存在已知问题，不建议使用。
6. 使用 FSDP2，`set_adapter()` 和 `set_requires_grad()` 对 [PeftModel](/docs/peft/v0.22.0/en/package_reference/peft_model#peft.PeftModel) 内的 FSDP2 模块进行重新分片，包括 `PeftModel` 本身，以便 `requires_grad` 更改到达分片参数。7. 仅当 FSDP2 替换了其中已注册的参数对象（就像转发对分片参数所做的那样），或者重新启用从入口时更改的 `requires_grad` 时，退出 `disable_adapter()` 才会退出。
8. `merge_adapter()`和`unmerge_adapter()`也对FSDP2模块进行了重新分片，使得merge和unmerge达到分片参数；如果没有重新分片，合并将最终出现在前向左注册的未分片参数中，并且下一个重新分片将删除它。合并本身被写入分片参数中。
9. 对合并适配器层的参数进行分片的 FSDP2 模块不会被 `set_adapter()` 和 `set_requires_grad()` 重新分片，因此在调用这些方法之前取消合并。 FSDP2下输入`disable_adapter()`会自动取消合并合并层，使取消合并达到分片参数；因此，合并不会在上下文中保留下来。
10. 使用`reshard_after_forward=False`，或者对于具有默认`None`的根模块，下一个前向再次聚集。
11. 包装`PeftModel`的FSDP2模块不会重新分片；在这些方法之前调用`reshard()`。
12. 提示学习方法和适应提示尚未涵盖。13. 对于 FSDP2/HSDP，在每个参与等级上调用 `get_peft_model_state_dict()` 或 `save_pretrained()`：收集分布式适配器张量是一个集合。对于`save_pretrained()`，仅在写作等级上通过`is_main_process=True`。仅收集选定的适配器条目（包括明确请求的嵌入或`modules_to_save`）；未选择的基本模型权重仍保持分片状态。

### 深速
https://huggingface.co/docs/peft/v0.22.0/accelerate/deepspeed.md