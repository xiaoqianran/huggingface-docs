<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通用在线 Logit 蒸馏 (GOLD) 培训师

[⟦T225⟧](https://huggingface.co/models?other=sft,gold)

## 概述

通用在线 Logit 蒸馏 (GOLD) 是通用 Logit 蒸馏 (ULD) 的扩展，支持
学生/教师与不同的标记器配对。它对齐两个分词器生成的文本范围并合并
关联的 logits，因此不会删除完成标记。这使得跨标记器知识蒸馏成为可能，包括
混合模式家庭（例如，LLaMA 学生与 Qwen 老师）。

关键能力：1. **跨标记器对齐** – GOLD 增量解码学生和教师标记，对具有相同可见文本的段落进行分组，并合并每个组内的概率。这保证了即使令牌边界不同，损失项也会在完全完成时计算。
2. **混合 ULD 损失** – 当启用 `uld_use_hybrid_loss` 时，GOLD 直接比较精确的词汇匹配，并针对不匹配的标记回退到原始排序概率 ULD 损失。这提高了词汇量仅与老师部分重叠的学生的稳定性。
3. **与 GKD 无缝集成** – GOLD 从 [experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer) 继承了在策略与离策略调度，因此您可以在一次训练运行中结合序列级 KD、广义 JSD 和交叉分词器蒸馏。

> [!注意]
> GOLD 目前是 `trl.experimental` 命名空间的一部分。当功能迭代时，API 可能会发生更改，恕不另行通知。

## 使用技巧

`GOLDTrainer` 子类 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 并接受与其他 TRL 训练器相同的数据集（ChatML 样式列表）
消息）。 `GOLDConfig` 上的重要配置标志包括：* `use_uld_loss` – 切换通用 Logit 蒸馏。将其设置为 `True` 以进行交叉标记器设置。
* `teacher_tokenizer_name_or_path` – `use_uld_loss=True` 时需要； GOLD 使用教师标记器来对齐标记。
* `uld_use_hybrid_loss`、`uld_hybrid_matched_weight`、`uld_hybrid_unmatched_weight` – 启用并加权混合
  匹配/不匹配损失。
* `beta`、`lmbda`、`seq_kd` – 继承自[experimental.gkd.GKDConfig](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDConfig)，控制广义 JSD 插值和同策略
  采样率。
* `num_generations`、`generation_batch_size` – 控制跨梯度累积窗口的缓冲推出生成。
  `generation_batch_size` 是每个优化器步骤中每个工作线程的唯一提示数。

一个最小的端到端示例：

```python
from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer

train_dataset = load_dataset(
    "HuggingFaceTB/OpenR1-Math-220k-default-verified",
    "all",
    split="train[:1024]",
)

trainer = GOLDTrainer(
    model="meta-llama/Llama-3.2-1B-Instruct",
    teacher_model="Qwen/Qwen2.5-0.5B-Instruct",
    args=GOLDConfig(output_dir="gold-model", use_uld_loss=True, teacher_tokenizer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    train_dataset=train_dataset,
)
trainer.train()
```

对于快速启动工作流程，您可以依赖如上所示的字符串标识符 - 训练器将为您加载模型和标记器。仅对于需要对初始化进行细粒度控制的高级用例，建议显式实例化 `AutoModelForCausalLM`、`AutoTokenizer` 或填充 `GOLDConfig`。

当您需要自定义模型加载、分词器设置或训练参数时，更明确的设置可能如下所示：

```python
from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "meta-llama/Llama-3.2-1B-Instruct"
teacher_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(student_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)

train_dataset = load_dataset(
    "HuggingFaceTB/Countdown-Task-GOLD",
    "verified_Qwen2.5-0.5B-Instruct",
    split="train",
)

training_args = GOLDConfig(
    output_dir="gold-model",
    per_device_train_batch_size=1,
    teacher_model_name_or_path=teacher_name,
    teacher_tokenizer_name_or_path=teacher_name,
    use_uld_loss=True,
    uld_use_hybrid_loss=True,
)

trainer = GOLDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```> [!注意]
> GOLD 缓冲一个完整的优化器窗口生成批次 (`per_device_train_batch_size * gradient_accumulation_steps`)
> 并在累积步骤中重复使用它。如果最终批次尺寸过小，GOLD 会发出警告并丢弃最后一批
> (`Dropping last batch due to unexpected batch size`)。设置 `dataloader_drop_last=True` 以避免此警告。

### 预期的数据集类型

GOLD 需要 [conversational](dataset_formats#conversational) [language modeling](dataset_formats#language-modeling) 数据集，例如：

```python
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}
```

`GOLDTrainer` 保留原始消息，以便 ChatML 整理器可以使用正确的内容构建提示和补全
边界。

## 代币合并如何运作

当学生和老师使用不同的标记器时，相同的文本可能会被不同地分割：

- **学生**：`"Hugging Face"` → 1 个代币
- **老师**：`"Hugging"`、`" Face"` → 2 个代币

GOLD 对齐这些序列，并将教师的多标记概率合并为单个分布，可以与学生的单标记分布进行比较。

### 概率合并

对于映射到单个学生标记的教师标记序列`[token₀, token₁, ..., tokenₖ]`，GOLD 计算：

```
P_merged(y) = P(y | context) × P(token₁ | token₀, context) × ... × P(tokenₖ | ..., context)
```

其中：
- `P(y | context)` 是第一个位置上所有词汇标记的边际概率分布
- `P(tokenᵢ | ..., context)` 是生成的实际令牌的**标量**条件概率**关键见解**：仅将**实际延续标记**的条件概率提取为标量。然后通过乘以这些标量概率来缩放第一个位置的完整边际分布。

这可以确保：
1. **实际生成序列的正确联合概率**（通过链式法则）
2. **反事实标记的合理近似**（按相同的连续可能性缩放）
3. **非标准化分布**，为 ULD 损失计算保留正确的相对概率

### 示例

鉴于：
```
P(x₀):         ["HF": 0.6,  "is": 0.3,  "cool": 0.1]
P(x₁ | "HF"):  ["HF": 0.05, "is": 0.9,  "cool": 0.05]
```

如果将标记 0 和 1 合并，实际序列为`["HF", "is"]`：
```
P_merged("HF")   = 0.6 × 0.9 = 0.54  ✓ (correct joint probability)
P_merged("is")   = 0.3 × 0.9 = 0.27
P_merged("cool") = 0.1 × 0.9 = 0.09
```

合并的分布是非标准化的（总和为 0.81），但这对于使用排序和 L1 距离的 ULD 损失计算来说是有意且正确的。

## 示例脚本

使用 [⟦T41⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/gold.py) 从命令行启动 GOLD 训练。该脚本通过标准 `ModelConfig` 标志支持完整的训练和 LoRA。

```bash
python examples/scripts/gold.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-model \
    --num_train_epochs 1 \
    --push_to_hub
```

## 训练视觉语言模型`GOLDTrainer` 支持VLM 到VLM 蒸馏。学生和老师都必须是视觉语言模型。要训​​练 VLM，请提供具有 `image` 列（每个样本单个图像）或 `images` 列（每个样本图像列表）的数据集。有关预期数据集结构的更多信息，请参阅 [Dataset Format — Vision datasets](dataset_formats#vision-datasets) 部分。

当学生和教师共享相同的架构和分词器（例如 Qwen3-VL-8B 到 Qwen3-VL-2B）时，标准广义 JSD 损失直接适用。当它们具有不同的`model_type`（例如Qwen3-VL到LFM2.5-VL）时，设置`use_uld_loss=True`以通过Universal Logit Distillation启用跨分词器对齐。图像通过每个模型的处理器单独处理。

```python
from datasets import load_dataset
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl.experimental.gold import GOLDConfig, GOLDTrainer

student_name = "Qwen/Qwen3-VL-2B-Instruct"
teacher_name = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(student_name, padding_side="left")
student_model = AutoModelForImageTextToText.from_pretrained(student_name, dtype=torch.bfloat16)
teacher_model = AutoModelForImageTextToText.from_pretrained(teacher_name, dtype=torch.bfloat16)

train_dataset = load_dataset("trl-lib/llava-instruct-mix", split="train")

trainer = GOLDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=GOLDConfig(
        output_dir="gold-vlm-model",
        max_length=None,
        teacher_model_name_or_path=teacher_name,
        use_uld_loss=False,
    ),
    train_dataset=train_dataset,
    processing_class=processor,
)
trainer.train()
```

对于跨系列蒸馏，将 `use_uld_loss=True` 和 `teacher_tokenizer_name_or_path` 设置为教师模型名称。

使用 [⟦T50⟧](https://github.com/huggingface/trl/blob/main/trl/experimental/gold/gold_vlm.py) 从命令行启动 GOLD VLM 训练：

```bash
# Same-family distillation (JSD loss, vLLM enabled)
accelerate launch trl/experimental/gold/gold_vlm.py \
    --student_model_name Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct

# Cross-family distillation (ULD loss, local generation)
accelerate launch trl/experimental/gold/gold_vlm.py \
    --student_model_name LiquidAI/LFM2.5-VL-1.6B \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct \
    --use_uld_loss \
    --no-use_vllm
```> [!提示]
> 对于 VLM，不支持 `truncation_mode='keep_end'`，因为图像标记驻留在序列的提示部分，并且可能会被静默删除。使用`truncation_mode='keep_start'`（默认）或在`GOLDConfig`中设置`max_length=None`。这使得模型能够处理完整的序列长度，而无需截断图像标记。
>
> ```python
> GOLDConfig(max_length=None, ...)
> ```
>
> 仅当您已验证截断不会删除整个数据集的图像标记时，才使用`max_length`。

> [!注意]
> 跨架构 VLM 蒸馏需要 `use_uld_loss=True`。如果您尝试跨架构蒸馏而不损失 ULD，训练器将引发错误。

## GOLDTrainer[[trl.experimental.gold.GOLDTrainer]]- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于`True`，加载*args.output_dir* 中由前一个实例保存的最后一个检查点
  `Trainer`。如果存在，训练将从此处加载的模型/优化器/调度器状态恢复。
- **试用**（`optuna.Trial`或`dict[str, Any]`，*可选*）--
  用于超参数搜索的试运行或超参数字典。
- **ignore_keys_for_eval** (`list[str]`，*可选*) --
  模型输出中的键列表（如果它是字典），在以下情况下应忽略这些键：
  收集训练期间评估的预测。`~trainer_utils.TrainOutput`包含全局步数、训练损失和指标的对象。

主要培训切入点。

将保存模型，以便您可以使用`from_pretrained()`重新加载它。

只会从主进程中保存。- **commit_message** (`str`，*可选*，默认为`"End of training"`) --
  推送时要提交的消息。
- **阻塞**（`bool`，*可选*，默认为`True`）--
  函数是否仅在 `git push` 完成时返回。
- **令牌**（`str`，*可选*，默认为`None`）--
  具有写入权限的令牌，可以覆盖 Trainer 的原始参数。
- **修订**（`str`，*可选*）--
  要提交的 git 修订版本。默认为“主”分支的头部。
- **kwargs**（`dict[str, Any]`，*可选*）--
  传递到 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## GOLDConfig[[trl.experimental.gold.GOLDConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "chat_template_path", "val": ": str | None = None"}, {"name": "dataset_text_field", "val": ": str = 'text'"}, {"name": "dataset_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "dataset_text_field", "val": ": str = 'text'"}无 = 无"}, {"name": "dataset_num_proc", "val": ":整数 |无 = 无"}, {"name": "eos_token", "val": ": str |无 = 无"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "truncation_mode", "val": ": str = 'keep_start'"}, {"name": "shuffle_dataset", "val": ": bool = False"}, {"name": "packing", "val": ": bool = False"}, {"name": "packing_strategy", "val": ": str = 'bfd'"}, {"name": "padding_free", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": int |无 = 无"}, {"name": "eval_packing", "val": ": bool |无 = 无"}, {"name": "completion_only_loss", "val": ": bool |无 = 无"}, {"name": "assistant_only_loss", "val": ": bool = False"}, {"name": "loss_type", "val": ": str |无 = 无"}, {"name": "activation_offloading", "val": ": bool = False"}, {"name": "pad_token", "val": ": str |无 = 无"}, {"name": "温度", "val": ": float = 0.9"}, {"name": "top_p", "val": ": float = 0.95"}, {"name": "top_k", "val": ": int = 0"}, {"name": "lmbda", "val": ": float = 0.5"}, {"name": "beta", "val": ": float = 0.5"}, {"name": "max_completion_length", "val": ": int = 128"}, {"name": "teacher_model_name_or_path", "val": ": str |无 = 无"}, {"name": "teacher_model_revision", "val": ": str |没有任何= None"}, {"name": "teacher_model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "teacher_tokenizer_name_or_path", "val": ": str |无 = None"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "seq_kd", "val": ": bool = False"}, {"name": "num_ Generations", "val": ": int = 1"}, {"name": " Generation_batch_size", "val": ": int |无 = 无"}, {"name": "use_uld_loss", "val": ": bool = False"}, {"name": "uld_token_merge_strategy", "val": ": str = 'observed'"}, {"name": "use_extended_uld", "val": ": bool = True"}, {"name": "uld_use_hybrid_loss", "val": ": bool = False"}, {"name": "uld_hybrid_matched_weight", "val": ": float |无 = 无"}, {"name": "uld_hybrid_unmatched_weight", "val": ": float |无 = 无"}, {"name": "uld_crossentropy_weight", "val": ": float = 0.0"}, {"name": "uld_distillation_weight", "val": ": float = 1.0"}, {"name": "uld_student_Temperature", "val": ": float = 1.0"}, {"name": "uld_teacher_Temperature", "val": ": float = 1.0"}, {"name": "uld_skip_student_eos", "val": ": bool = True"}, {"name": "uld_skip_teacher_eos", "val": ": bool = True"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": “vllm_mode”，“val”：": str = 'colocate'"}, {"name": "vllm_server_base_url", "val": ": str | None = None"}, {"name": "vllm_server_host", "val": ": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8001"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_gpu_memory_utilization", "val": ": float = 0.9"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "vllm_max_model_length", "val": ": int | None = None"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_structed_outputs_regex", "val": ": str | None = None"}, {"name": "vllm_sync_Frequency", "val": ": int = 1"}, {"name": "vllm_enable_sleep_mode", "val": ": bool = False"}, {"name": "log_completions", "val": ": bool = False"}, {"name": "log_completions_steps", "val": ": int = 100"}, {"name": "num_completions_to_print", "val": ": int | None = None"}, {"name": "wandb_log_unique_prompts", "val": ": bool = True"}, {"name": "callbacks", "val": ": list = "}]}>
控制生成和训练循环的参数- **温度**（`float`，*可选*，默认为`0.9`）--
  取样温度。温度越高，完成的随机性越大。
- **top_p** (`float`，*可选*，默认为`0.95`) --
  如果设置为浮动 < 1, only the smallest set of most probable tokens with probabilities that add up to
  ⟦T89⟧ or higher are kept for generation.
- **top_k** (⟦T90⟧, *optional*, defaults to ⟦T91⟧) --
  Number of highest probability vocabulary tokens to keep for top-k-filtering. If ⟦T92⟧, top-k-filtering is
  disabled and all tokens are considered.
- **lmbda** (⟦T93⟧, *optional*, defaults to ⟦T94⟧) --
  Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
  student-generated outputs).
- **beta** (⟦T95⟧, *optional*, defaults to ⟦T96⟧) --
  Interpolation coefficient between ⟦T97⟧ and ⟦T98⟧ of the Generalized Jensen-Shannon Divergence loss. When
  beta is ⟦T99⟧, the loss is the KL divergence. When beta is ⟦T100⟧, the loss is the Inverse KL Divergence.
- **max_completion_length** (⟦T101⟧, *optional*, defaults to ⟦T102⟧) --
  Maximum number of tokens to generate per completion.
- **teacher_model_name_or_path** (⟦T103⟧, *optional*) --
  Model name or path of the teacher model. If ⟦T104⟧, the teacher model will be the same as the model being
  trained.
- **teacher_model_revision** (⟦T105⟧ or ⟦T106⟧, *optional*, defaults to ⟦T107⟧) --
  Model revision of the teacher model (e.g., branch name, tag, or commit hash). If ⟦T108⟧, the default
  revision is used.
- **teacher_model_init_kwargs** (⟦T109⟧, *optional*) --
  Keyword arguments to pass to ⟦T110⟧ when instantiating the teacher model
  from a string.
- **teacher_tokenizer_name_or_path** (⟦T111⟧, *optional*) --
  Tokenizer name or path for the teacher model. If None when using ULD loss, will use the same tokenizer as
  the student model (not recommended for cross-tokenizer distillation).
- **disable_dropout** (⟦T112⟧, *optional*, defaults to ⟦T113⟧) --
  Whether to disable dropout in the model.
- **seq_kd** (⟦T114⟧, *optional*, defaults to ⟦T115⟧) --
  Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
  teacher-generated output).
- **num_generations** (⟦T116⟧, *optional*, defaults to ⟦T117⟧) --
  Number of generations per prompt. Each prompt is repeated this many times in the generation batch.
- **generation_batch_size** (⟦T118⟧ or ⟦T119⟧, *optional*, defaults to ⟦T120⟧) --
  Number of unique prompts per worker per optimizer step. If ⟦T121⟧, it is computed from
  ⟦T122⟧.
Parameters that control the ULD loss

- **use_uld_loss** (⟦T123⟧, *optional*, defaults to ⟦T124⟧) --
  Whether to use Universal Logit Distillation (ULD) loss instead of Generalized Jensen-Shannon Divergence
  loss.
- **use_extended_uld** (⟦T125⟧, *optional*, defaults to ⟦T126⟧) --
  Whether to enable extended ULD alignment that uses tokenizers to align and merge token probabilities
  across student and teacher tokenizations. When ⟦T127⟧, the trainer will compute token mappings and merge
  probabilities for split tokens; when ⟦T128⟧, ULD will use simple positional truncation like in the
  original ULD paper.
- **uld_token_merge_strategy** (⟦T129⟧, *optional*, defaults to ⟦T130⟧) --
  Strategy used to align answer logits and merge token probabilities in the ULD loss. With ⟦T131⟧, the
  answer logits are sliced at the answer positions and split tokens (when ⟦T132⟧) are merged
  by multiplying the marginal distribution at the first position by the scalar conditional probabilities of
  the actual later tokens. With ⟦T133⟧, the answer-logit slice is shifted one position earlier so that
  ⟦T134⟧ predicts ⟦T135⟧ (chain rule), and split tokens are merged using the last position's full
  distribution, conditioned on the actual prefix tokens, multiplied by the scalar probabilities of the
  earlier tokens. The logit-alignment shift applies whether or not ⟦T136⟧ is enabled.
- **uld_use_hybrid_loss** (⟦T137⟧, *optional*, defaults to ⟦T138⟧) --
  Whether to use a hybrid loss that combines ULD loss and JSD loss. When ⟦T139⟧, the final loss is a
  combination of JSD for known token mappings and ULD for unknown token mappings.
- **uld_hybrid_matched_weight** (⟦T140⟧ or ⟦T141⟧, *optional*) --
  Weight for the matched token loss component when using hybrid ULD + JSD loss. This weight scales the JSD
  loss computed over tokens that have a direct mapping between student and teacher tokenizations. If ⟦T142⟧,
  uses adaptive weighting based on vocabulary overlap. Must be set together with
  ⟦T143⟧ (both ⟦T144⟧ or both ⟦T145⟧).
- **uld_hybrid_unmatched_weight** (⟦T146⟧ or ⟦T147⟧, *optional*) --
  Weight for the unmatched token loss component when using hybrid ULD + JSD loss. This weight scales the ULD
  loss computed over tokens that do not have a direct mapping between student and teacher tokenizations. If
  ⟦T148⟧, uses adaptive weighting based on vocabulary overlap. Must be set together with
  ⟦T149⟧ (both ⟦T150⟧ or both ⟦T151⟧).
- **uld_crossentropy_weight** (⟦T152⟧, *optional*, defaults to ⟦T153⟧) --
  Weight for the cross-entropy loss component in ULD loss. If 0, only ULD distillation loss is used.
- **uld_distillation_weight** (⟦T154⟧, *optional*, defaults to ⟦T155⟧) --
  Weight for the distillation loss component in ULD loss.
- **uld_student_temperature** (⟦T156⟧, *optional*, defaults to ⟦T157⟧) --
  Temperature for student logits in ULD loss computation.
- **uld_teacher_temperature** (⟦T158⟧, *optional*, defaults to ⟦T159⟧) --
  Temperature for teacher logits in ULD loss computation.
- **uld_skip_student_eos** (⟦T160⟧, *optional*, defaults to ⟦T161⟧) --
  Whether to skip EOS token for student in ULD loss computation.
- **uld_skip_teacher_eos** (⟦T162⟧, *optional*, defaults to ⟦T163⟧) --
  Whether to skip EOS token for teacher in ULD loss computation.
Parameters that control vLLM integration

- **use_vllm** (⟦T164⟧, *optional*, defaults to ⟦T165⟧) --
  Whether to use vLLM for generating completions from the student model. Requires ⟦T166⟧ to be installed.
- **vllm_mode** (⟦T167⟧, *optional*, defaults to ⟦T168⟧) --
  Mode for student vLLM integration. Either ⟦T169⟧ (connect to a running TRL vLLM server) or ⟦T170⟧
  (run vLLM in the same process).
- **vllm_server_host** (⟦T171⟧, *optional*, defaults to ⟦T172⟧) --
  Host of the vLLM server for the student model (if ⟦T173⟧).
- **vllm_server_port** (⟦T174⟧, *optional*, defaults to ⟦T175⟧) --
  Port of the vLLM server for the student model (if ⟦T176⟧).
- **vllm_server_timeout** (⟦T177⟧, *optional*, defaults to ⟦T178⟧) --
  Timeout for connecting to the student vLLM server (if ⟦T179⟧).
- **vllm_gpu_memory_utilization** (⟦T180⟧, *optional*, defaults to ⟦T181⟧) --
  GPU memory utilization for the colocated student vLLM engine (if ⟦T182⟧). It is recommended
  to set this to a low value if the student and teacher models share the same GPU.
- **vllm_tensor_parallel_size** (⟦T183⟧, *optional*, defaults to ⟦T184⟧) --
  Tensor parallel size for the colocated student vLLM engine (if ⟦T185⟧).
- **vllm_structured_outputs_regex** (⟦T186⟧, *optional*) --
  Regex for vLLM structured outputs for the student model.
- **vllm_server_base_url** (⟦T187⟧, *optional*) --
  Base URL for the vLLM server (e.g., ⟦T188⟧). If provided, ⟦T189⟧ and
  ⟦T190⟧ are ignored.
- **vllm_group_port** (⟦T191⟧, *optional*, defaults to ⟦T192⟧) --
  Port for the vLLM weight-update group (NCCL communicator). Unless the port is occupied, there is no need to
  change it.
- **vllm_max_model_length** (⟦T193⟧, *optional*) --
  Maximum model sequence length for the colocated vLLM engine when ⟦T194⟧. Defaults to the
  model's maximum context length.
- **vllm_model_impl** (⟦T195⟧, *optional*, defaults to ⟦T196⟧) --
  Model implementation backend to use in vLLM. Use ⟦T197⟧ (default) or ⟦T198⟧.
- **vllm_sync_frequency** (⟦T199⟧, *optional*, defaults to ⟦T200⟧) --
  Frequency (in training steps) to synchronize student model weights to vLLM engine. Set to 1 to sync after
  every step.
- **vllm_enable_sleep_mode** (⟦T201⟧, *optional*, defaults to ⟦T202⟧) --
  Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU memory usage
  low, but waking the engine adds host–device transfer latency.

Parameters that control logging

- **log_completions** (⟦T203⟧, *optional*, defaults to ⟦T204⟧) --
  Whether to log a sample of (prompt, completion) pairs every ⟦T205⟧ steps. If ⟦T206⟧ is installed,
  it prints the sample. If ⟦T207⟧ logging is enabled, it logs it to ⟦T208⟧.
- **log_completions_steps** (⟦T209⟧, *optional*, defaults to ⟦T210⟧) --
  Number of steps between logging (prompt, completion) pairs. Only used if ⟦T211⟧ is set to
  ⟦T212⟧.
- **num_completions_to_print** (⟦T213⟧ or ⟦T214⟧, *optional*) --
  Number of completions to print with ⟦T215⟧. If ⟦T216⟧, all completions are logged.
- **wandb_log_unique_prompts** (⟦T217⟧, *optional*, defaults to ⟦T218⟧) --
  Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.
- **callbacks** (⟦T219⟧, *optional*, defaults to ⟦T220⟧) --
  The callbacks to run during training.

Configuration class for ⟦T221⟧.

This class includes only the parameters that are specific to GOLD training. For a full list of training arguments,
please refer to the ⟦T235⟧ and ⟦T236⟧ documentation.

> [!NOTE]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `learning_rate`：默认为`1e-7`，而不是`5e-5`。

### 示例
https://huggingface.co/docs/trl/v1.9.2/example_overview.md