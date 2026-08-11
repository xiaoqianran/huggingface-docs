<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 完全分片数据并行

[Fully sharded data parallel](https://pytorch.org/docs/stable/fsdp.html) (FSDP) 是为高达 1T 参数的大型预训练模型的分布式训练而开发的。 FSDP 通过跨数据并行进程对模型参数、梯度和优化器状态进行分片来实现这一点，并且它还可以将分片的模型参数卸载到 CPU。 FSDP 提供的内存效率允许您将训练扩展到更大的批次或模型大小。

🤗 Accelerate 支持这两个功能，您可以将它们与 🤗 PEFT 一起使用。 

# 使用 PEFT 和 FSDP
本部分指南将帮助您了解如何使用我们的 DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/sft/train.py) 执行 SFT。您将配置脚本以在单台计算机上的 8xH100 80GB GPU 上使用 LoRA 和 FSDP 对 Llama-70B 模型执行 SFT（监督微调）。您可以通过更改加速配置将其配置为扩展到多台计算机。

## 配置

首先使用 🤗 Accelerate 对 [create a FSDP configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) 运行以下命令。 `--config_file` 标志允许您将配置文件保存到特定位置，否则它将作为 `default_config.yaml` 文件保存在🤗 Accelerate 缓存中。

配置文件用于在启动训练脚本时设置默认选项。

```bash
accelerate config --config_file fsdp_config.yaml
```系统会询问您一些有关您的设置的问题，并配置以下参数。在此示例中，您将回答如下图所示的调查问卷。

    

创建 Accelerate 的配置以使用 FSDP

完成后，相应的配置应如下所示，您可以在 [fsdp_config.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config.yaml) 的 config 文件夹中找到它：

```yml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
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

## 启动命令

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

## 重要部分

让我们更深入地研究一下脚本，以便您可以了解发生了什么，并了解它是如何工作的。

首先要知道的是，该脚本使用 FSDP 进行分布式训练，因为 FSDP 配置已通过。 [SFTTrainer](https://huggingface.co/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 类处理使用传递的 peft 配置创建 PEFT 模型的所有繁重工作。之后，当您调用 `trainer.train()` 时，Trainer 在内部使用 🤗 Accelerate 准备模型，优化器和训练器使用 FSDP 配置创建 FSDP 包装模型，然后进行训练。主要代码片段如下：

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
if model_args.use_peft_lora:
    # handle PEFT+FSDP case
    trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
```这里，当前在将 FSDP 与 PEFT 结合使用时需要注意的一件主要事情是，`use_orig_params` 需要是 `False` 才能实现 GPU 内存节省。由于`use_orig_params=False`，FSDP 的自动换行策略需要更改，以便可训练参数和不可训练参数分开换行。这是通过下面的代码片段完成的，该代码片段使用 PEFT 中的 util 函数`fsdp_auto_wrap_policy`：

```
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
```

## 内存使用情况

在上面的示例中，每个 GPU 消耗的内存为 72-80 GB (90-98%)，如下面的屏幕截图所示。最后GPU内存略有增加是在使用`FULL_STATE_DICT`状态字典类型而不是`SHARDED_STATE_DICT`保存模型时，以便模型具有可以在推理过程中使用`from_pretrained`方法正常加载的适配器权重：

    

训练运行的 GPU 内存使用情况

# 使用 PEFT QLoRA 和 FSDP 在多个 GPU 上微调大型模型

在本节中，我们将了解如何使用 QLoRA 和 FSDP 在 2X24GB GPU 上微调 70B llama 模型。 [Answer.AI](https://www.answer.ai/) 与bitsandbytes 和 Hugging Face 合作🤗 开源代码，支持使用 FSDP+QLoRA，并在其富有洞察力的博文[You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) 中解释了整个过程。现在它已集成到 Hugging Face 生态系统中。为此，我们首先需要`bitsandbytes>=0.43.3`、`accelerate>=1.0.1`、`transformers>4.44.2`、`trl>0.11.4`和`peft>0.13.0`。使用 Accelerate 配置时，我们需要设置 `fsdp_cpu_ram_efficient_loading=true`、`fsdp_use_orig_params=false` 和 `fsdp_offload_params=true`（CPU 卸载）。当不使用加速启动器时，您可以交替设置环境变量`export FSDP_CPU_RAM_EFFICIENT_LOADING=true`。  在这里，我们将使用加速配置，下面是可以在[fsdp_config_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config_qlora.yaml)找到的配置：

```yml
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false                                                                                                                                                                 
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
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

请注意传递的新参数`bnb_4bit_quant_storage_dtype`，它表示用于打包 4 位参数的数据类型。例如，当它设置为 `bfloat16` 时，**16/4 = 4** 4 位参数在量化后打包在一起。当使用`bfloat16`进行混合精度训练时，`bnb_4bit_quant_storage_dtype`可以是`bfloat16`（用于纯`bfloat16`微调），也可以是`float32`（用于自动混合精度）（这会消耗更多GPU内存）。当使用`float16`进行混合精度训练时，`bnb_4bit_quant_storage_dtype`应设置为`float32`，以实现稳定的自动混合精度训练。

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

## 内存使用情况在上面的示例中，每个 GPU 消耗的内存为 **19.6 GB**，而 CPU RAM 使用量约为 **107 GB**。禁用 CPU 卸载时，GPU 内存使用率为 **35.6 GB/GPU**。因此，之前需要 16X80GB GPU 进行全面微调，8X80GB GPU 采用 FSDP+LoRA，以及几个 80GB GPU 采用 DDP+QLoRA，现在需要 2X24GB GPU。这使得大型模型的微调变得更加容易。

## 更多资源
您还可以参考 [llama-recipes](https://github.com/facebookresearch/llama-recipes/?tab=readme-ov-file#fine-tuning) 存储库和 [Getting started with Llama](https://llama.meta.com/get-started/#fine-tuning) 指南，了解如何使用 FSDP 和 PEFT 进行微调。

## 注意事项
1. 目前不支持使用 PEFT 和 FSDP 时进行合并，并且会引发错误。
2. 传递`modules_to_save`配置参数目前尚未测试。
3. 使用 CPU Offloading 时的 GPU 内存节省目前未经测试。
4. 使用 FSDP+QLoRA 时，`paged_adamw_8bit` 目前会导致保存检查点时出错。
5. 使用 FSDP 进行 DoRA 训练应该有效（尽管速度低于 LoRA）。如果与位和字节 (QDoRA) 结合使用，4 位量化也应该可以工作，但 8 位量化存在已知问题，不建议使用。

### 调音器
https://huggingface.co/docs/peft/v0.20.0/package_reference/tuners.md