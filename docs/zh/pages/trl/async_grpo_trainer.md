<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 异步GRPO

> [!重要]
> 该训练器需要 `vllm>=0.22.0` 和 `transformers>=5.2.0`。对于分布式训练，仅支持 FSDP2（不支持 DeepSpeed ZeRO）。
>
> 目前，`vllm` 和 `transformers` 具有冲突的依赖约束。要解决此问题，请先安装 vLLM，然后强制安装变压器：
>
> ```bash
> pip install 'vllm>=0.22.0'
> pip install 'transformers>=5.2.0' --no-deps
> ```

## 概述

`AsyncGRPOTrainer` 实现相同的 [GRPO](grpo_trainer) 算法，但将推出生成与训练分离。后台工作人员不断从 vLLM 服务器传输完成数据，同时训练循环消耗它们，因此生成和梯度更新重叠而不是交替。 API 镜像 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) — 有关 GRPO 方法本身的完整详细信息（优势计算、KL 估计、损失公式、奖励函数等），请参阅 [GRPO Trainer](grpo_trainer) 文档。并非[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)的所有功能都可用；支持的参数请参见`AsyncGRPOConfig`。

该训练器由[Quentin Gallouédec](https://huggingface.co/qgallouedec)和[Amine Dirhoussi](https://huggingface.co/aminediroHF)贡献。

## 它与[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)有何不同

在标准[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)中，生成和训练是连续的：生成批次，计算损失，更新权重，重复。即使在 [vLLM colocate mode](grpo_trainer#speed-up-training-with-vllm-powered-generation) 中，生成在相同的 GPU 上运行，一个阶段也必须在另一个阶段开始之前完成。`AsyncGRPOTrainer` 区分了这两个问题：

- **推出工作人员**（后台进程）— 向 vLLM 服务器发送提示，使用奖励函数对完成情况进行评分，计算优势，并将准备好训练的样本推送到队列中。
- **训练循环**（主流程）——从队列中提取样本，计算截短的代理损失，并更新模型权重。

Rollout Worker 在由训练器生成的单独进程中运行，因此奖励计算永远不会与 GIL 的训练循环竞争。这对于您可以传递为 `reward_funcs`、`tools` 和 `environment_factory` 的内容有两个后果（对于后者，请参阅 [OpenEnv guide](openenv)，其中涵盖了合约和可用的集成）：> [!警告]
> 因为我们在单独的进程中运行 rollout Worker，所以传递给它的所有内容都是 **pickled**。因此，每个奖励函数、工具和`environment_factory`（以及它们关闭的任何内容）都必须是可挑选的：使用模块级函数、[⟦T16⟧](https://docs.python.org/3/library/functools.html#functools.partial)或**可调用类实例**。 Lambda 和闭包将在 `trainer.train()` 提高 `TypeError`。这与[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)不同，在[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)中，奖励函数被称为进程内，闭包起作用。
>
> 推出过程也使用 `CUDA_VISIBLE_DEVICES=""` 运行，因此无法使用 GPU。 **GPU 支持的奖励模型**（例如 `AutoModelForSequenceClassification` 记分器）仍然可以正确加载，但会默默地回退到 **CPU**（请注意，在 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 中，这样的奖励模型共享训练器的 GPU）。保持奖励函数在 CPU 端且轻量级（验证器如`accuracy_reward`、格式/长度检查）。
>
> 如果您确实需要 GPU 奖励模型，建议的方法是在单独的 GPU 上**在其自己的推理引擎（vLLM、TGI 等）后面提供服务，并通过 HTTP 调用轻量级、可挑选的奖励函数。这使得奖励模型保留在自己的设备上，而部署过程仅依赖 CPU，并且它的扩展独立于训练器。在每个`weight_sync_steps`训练步骤之后，更新的权重都会通过NCCL传输到vLLM服务器，以便后续世代反映最新的策略。

由于生成和训练同时运行，因此训练样本可能是由稍旧版本的模型生成的。 `max_staleness` 参数控制样本在被丢弃之前可以落后多少次权重更新。

发送到 vLLM 服务器的并发请求数由`max_inflight_tasks` 控制。默认情况下，它自动设置为`max_staleness × per_device_train_batch_size × gradient_accumulation_steps × num_processes`——训练器在过时之前可以消耗的最大样本数。生成超过此数量是浪费的，因为多余的样本将被丢弃。

## 快速开始

```python
# train_async_grpo.py
from datasets import load_dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3-4B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

vLLM 服务器和训练器必须在 **单独的 GPU** 上运行。使用 `CUDA_VISIBLE_DEVICES` 对 GPU 进行分区。例如，使用 2 个 GPU，您可以在 GPU 0 上运行 vLLM 服务器，在 GPU 1 上运行训练器，如下所示：

```bash
# Terminal 1: vLLM server on GPU 0 (dev mode + NCCL weight transfer are required)
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B \
    --max-model-len 4096 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'
```

> [!提示]
> 将 `--max-model-len` 设置为您期望的最大总序列长度（提示 + 完成）。较低的值会减少服务器上的 GPU 内存使用量，从而为 KV 缓存释放更多内存并提高吞吐量。一个好的起点是提示长度加上配置中的`max_completion_length`。```bash
# Terminal 2: training on GPU 1
CUDA_VISIBLE_DEVICES=1 accelerate launch train_async_grpo.py
```

## 设计理念

该训练器有意保持最小化，并不意味着发展成为通用解决方案。如果您需要不受支持的功能，我们建议克隆存储库并直接根据您的需求调整培训器。只有当社区需求很大时才会考虑新功能。

## AsyncGRPOConfig[[trl.experimental.async_grpo.AsyncGRPOConfig]]

"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ":bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool | None = None"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_determinism", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"名称": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_parameters", "val": ": bool | None = None"}, {"name": "ddp_bucket_cap_mb", "val": ": int | None = None"}, {"name": "ddp_broadcast_buffers", "val": ": bool | None = None"}, {"name": "ddp_static_graph", "val": ": bool | None = None"}, {"name": "ddp_backend", "val": ": str | None = None"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str | None = None"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | str | None = None"}, {"name": "deepspeed", "val": ": dict | str | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str | None = None"}, {"name": "warmup_ratio", "val": ": float | None = None"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, 打字。Any] | STR | None = None"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "num_ Generations", "val": ": int = 8"}, {"name": "max_completion_length", "val": ": int = 2048"}, {"name": "温度", "val": ": float = 1.0"}, {"name": "chat_template_kwargs", "val": ": dict |无 = 无"}, {"name": "max_tool_calling_iterations", "val": ": int |无 = 无"}, {"name": "fork_threshold_tokens", "val": ": int = 1024"}, {"name": "vllm_server_base_url", "val": ": str = 'http://localhost:8000'"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "request_timeout", "val": ": int = 600"}, {"name": "epsilon", "val": ": float = 0.2"}, {"name": "epsilon_high", "val": ": float |无 = 无"}, {"name": "token_budget", "val": ": int |无 = 无"}, {"name": "max_inflight_tasks", "val": ": int = -1"}, {"name": "max_staleness", "val": ": int = 4"}, {"name": "queue_maxsize", "val": ": int = 1024"}, {"name": "weight_sync_steps", "val": ": int = 1"}, {"name": "heartbeat_stale_after_s", "val": ": float = 300.0"}, {"name": "log_completions", "val": ": bool = False"}, {"name": "num_completions_to_print", "val": ": int |无 = 无"}]}>
控制模型的参数

- **model_init_kwargs** （`dict[str, Any]` 或 `str`，*可选*）--
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 的关键字参数，在实例化时使用
  来自路径的模型。
- **trust_remote_code**（`bool`，*可选*，默认为`False`）--
  是否允许加载从 Hub 发送自定义 Python 代码的模型和标记器。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 和 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)。
- **router_aux_loss_coef** (`float`，*可选*，默认为`0.001`) --
  负载平衡辅助损耗系数。仅在训练混合专家时有效
  （教育部）模型；对于其他模型，它不执行任何操作。辅助损失被添加到训练损失中
  重量。设置为`0.0`以禁用它。

控制生成的参数- **num_ Generations** （`int`，*可选*，默认为`8`）--
  每次提示采样的代数。
- **max_completion_length**（`int`，*可选*，默认为`2048`）--
  生成的补全的最大长度。
- **温度**（`float`，*可选*，默认为`1.0`）--
  取样温度。温度越高，完成的随机性越大。
- **chat_template_kwargs** (`dict[str, Any]`，*可选*) --
  生成补全时传递给 `apply_chat_template` 函数的附加关键字参数。
- **max_tool_calling_iterations** (`int`，*可选*) --
  训练代理时工具调用的最大轮次。如果`None`，则没有限制和生成
  当模型在没有工具调用的情况下生成响应转弯或总响应长度达到
  `max_completion_length`。
- **fork_threshold_tokens** (`int`，*可选*，默认为`1024`) --
  通过每轮重新标记整个对话，将多轮对话转变为训练行
  并将结果与迄今为止持有的令牌进行协调：干净的附加保留一行，重写（删除
  推理，总结历史）分叉一个新行。当回合的重新标记提示漂移到最后一个回合时生成的答案，决定是根据 **drift size** 做出的 - 有多少之前训练过的 token
  重新调整会掩盖上下文。小于这么多代币的漂移被视为重新代币化
  摆动（根据上下文重新调整）；更大的漂移——例如模板删除的长推理块 -
  分叉一个新行，以便那些经过训练的令牌保留其训练信号，而不是被默默地屏蔽。

控制 vLLM 服务器的参数

- **vllm_server_base_url**（`str`，*可选*，默认为`"http --//localhost:8000"`）：
  用于生成的 vLLM 服务器的基本 URL（例如，`"http://localhost:8000"`）。
- **vllm_server_timeout**（`float`，*可选*，默认为`240.0`）--
  等待 vLLM 服务器准备就绪的总超时持续时间（以秒为单位）。
- **request_timeout**（`int`，*可选*，默认为`600`）--
  对 vLLM 服务器的单个 HTTP 请求的超时（以秒为单位）。

控制训练的参数- **epsilon**（`float`，*可选*，默认为`0.2`）--
  剪裁的 Epsilon 值。
- **epsilon_high**（`float`，*可选*）--
  剪裁的上限 epsilon 值。如果未指定，则默认与下限相同的值
  在参数`epsilon`中指定。论文[DAPO](https://huggingface.co/papers/2503.14476)推荐`0.28`。
- **token_budget**（`int`，*可选*）--
  动态打包到单行（一个 DP 等级向前）的真实令牌的最大数量
  令牌预算微批处理。当`> 0`时，`TokenBudgetBatcher`形成Σ Lᵢ²平衡微批次
  每个行都保持在此预算内，限制峰值内存，与样本计数无关（
  每行的样本数变为动态）。如果`None`（默认），则设置为 vLLM 服务器的
  `max_model_len`（在列车开始时查询）——提示上限 + 完成长度——所以没有推出样本
  可能会超出预算。长度超过 `token_budget` 的样本不适合任何行，并用
  警告。设置 `<= 0` 禁用令牌预算，而是打包固定的 `per_device_train_batch_size ×
  每个微批次的 num_processes 个样本，在各行之间保持 Σ Lᵢ² 平衡。

控制异步转出管道的参数- **max_inflight_tasks**（`int`，*可选*，默认为`-1`）--
  发送到 vLLM 服务器的并发生成任务的最大数量。默认为 `-1`（自动），即
  将其设置为`max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`。
  如果使用工具使用环境，您可能需要根据并行环境的数量手动设置
  你可以跑。
- **max_staleness**（`int`，*可选*，默认为`4`）--
  rollout 样本之前可以落后于当前模型版本的最大权重更新步骤数
  被丢弃。
- **queue_maxsize** (`int`，*可选*，默认为`1024`) --
  转出队列中缓冲的转出样本的最大数量。
- **weight_sync_steps**（`int`，*可选*，默认为`1`）--
  权重同步到 vLLM 服务器之间的训练步骤数。
- **heartbeat_stale_after_s** (`float`，*可选*，默认为`300.0`) --
  自推出工作人员最后一次心跳以来的秒数，之后培训师将其视为
  挂起并中止。

控制日志记录的参数- **log_completions** (`bool`，*可选*，默认为`False`) --
  是否每 `logging_steps` 步骤记录（提示、完成）对的样本。
- **num_completions_to_print** (`int`，*可选*) --
  使用 `rich` 打印的完成数。如果`None`，则记录所有完成情况。

`AsyncGRPOTrainer`的配置类。

此类仅包含特定于异步 GRPO 训练的参数。完整列表
训练参数请参考[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。注意默认值
该类别中的内容可能与 [TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments) 中的内容不同。

> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`1`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`1e-6`，而不是`5e-5`。
> - `lr_scheduler_type`：默认为 `constant` 而不是 `linear`（见下文）。> [!注意]
> 消息模式协调下的训练时长和学习率：
> 多轮对话可以分叉成可变数量的训练行（对话的重写
> 开始一个新行），因此每个时期的样本数以及优化器步骤数不是
> 预先知道。结果是：
> - `num_train_epochs` 通过对 *prompt* 数据集的完整传递来限制训练，计算为
> 实际训练过的不同提示。这与叉子产生的行数无关，因此要求
> N epoch 总是对数据进行 N 次训练。当`max_steps`未设置时，这是停止条件
> 而`max_steps`只是一个安全上限。
> - `max_steps`，如果显式设置 (`> 0`)，则接管作为停止条件（由优化器步骤限制，而不是
> 比按纪元）并禁用基于纪元的停止。
> - `lr_scheduler_type` 默认为 `constant`，因为衰减范围是在优化器步骤中测量的，这
> 当步数取决于分叉速率时，无法预先获知。对于衰减的学习率，设置
> 腐烂的时间表和明确的`max_steps`。

## AsyncGRPOTrainer[[trl.experimental.async_grpo.AsyncGRPOTrainer]]- **型号** (`str`) --
  待训练的模型。必须是一个字符串，是模型内托管的预训练模型的*模型 ID*
  Huggingface.co 上的存储库，或包含使用保存的模型权重的*目录*的路径
  [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
  使用[from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained)。型号名称也用于识别
  用于生成的 vLLM 服务器上的模型。
- **reward_funcs** (`RewardFunc | list[RewardFunc]`，*可选*) --
  用于计算奖励的奖励函数。为了计算奖励，我们将所有奖励称为
  具有提示和完成功能并总结奖励。提供奖励时可以省略
  通过 `environment_factory` 受到环境影响（见下文）。可以是：- 单一奖励功能：该功能提供提示和生成的完成，加上
    数据集中的任何其他列。它应该返回奖励列表。奖励函数可以是
    同步或异步，当奖励不适用时也可以返回`None`
    样品。这对于多任务训练非常有用，其中不同的奖励函数适用于不同的类型
    样品。当奖励函数为样本返回 `None` 时，该奖励函数将被排除在外
    该样本的奖励计算。欲了解更多详情，请参阅[Using a custom reward
    function](#using-a-custom-reward-function)。
  - 奖励函数列表，其中每一项都是如上所述的奖励函数。来自各方的奖励
    函数被求和。与 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 不同，奖励是在生成的子进程中计算的，因此每个奖励函数（以及
  与 `tools` 和 `environment_factory`）必须是可挑选的：使用模块级函数，
  `functools.partial`，或可调用类实例 — lambda 和闭包将在启动时失败。孩子
  进程也使用 `CUDA_VISIBLE_DEVICES=""` 运行，因此 GPU 支持的奖励模型在 CPU 上运行（速度较慢），而不是在 CPU 上运行
  训练器的 GPU。
- **参数**（`AsyncGRPOConfig`，*可选*）--
  该训练器的配置。如果`None`，则使用默认配置。
- **train_dataset** （`Dataset` 或 `IterableDataset`，*可选*）--
  用于训练的数据集。它必须包含列 `"prompt"`。数据集中的任何其他列都是
  被忽略。样本的格式可以是：

  - [Standard](dataset_formats#standard)：每个样本都包含纯文本。
  - [Conversational](dataset_formats#conversational)：每个样本都包含结构化消息（例如，角色
    和内容）。仅当提供了 `environment_factory` 并且环境拥有（或程序上
  生成）数据，从其 `reset()` 方法返回提示。在这种情况下，必须设置`max_steps`
  定义训练长度。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，*可选*）--
  处理类用于处理数据。填充边必须设置为“左”。如果`None`，则
  处理类从带有[from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)的模型名称加载。一个
  必须设置填充令牌`tokenizer.pad_token`。如果处理类没有设置填充标记，
  `tokenizer.eos_token` 将用作默认值。
- **回调**（[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)列表，*可选*）--
  用于自定义训练循环的回调列表。将它们添加到详细的默认回调列表中
  在[here](https://huggingface.co/docs/transformers/main_classes/callback)。如果您想删除使用的默认回调之一，请使用 [remove_callback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer.remove_callback)
  方法。
- **优化器**（`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`，*可选*，默认为`(None, None)`）--
  包含要使用的优化器和调度器的元组。将默认为您的 `AdamW` 实例
  模型和由[get_linear_schedule_with_warmup](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup)给出的调度器，由`args`控制。
- **工具**（`Callable`列表，*可选*）--
  模型在生成过程中可以调用的可调用工具函数（同步或异步）列表。每个工具
  应该是一个标准的 Python 函数，具有正确类型提示的参数和返回值，以及
  Google 风格的文档字符串描述其目的、参数和返回值。有关更多详细信息，请参阅：
  https://huggingface.co/docs/transformers/en/chat_extras#passing-tools。该模型使用函数的名称，
  类型提示和文档字符串来确定如何调用它。确保模特的聊天模板支持工具
  使用并且它已经针对工具调用进行了微调。
- **环境工厂**（`EnvironmentFactory`或`dict[str, EnvironmentFactory]`，*可选*）--
  创建并返回环境实例的可调用对象，或将环境名称映射到的字典这样的可调用对象。环境类应该定义可以在生成过程中作为工具调用的方法。
  每种方法应符合上述`tools`相同的要求。环境必须
  还实现了一个可调用的 `reset` 方法，可用于重置各代之间的状态。 `reset`
  方法应该返回 `None` 或字符串：当它返回字符串时，该字符串会附加到
  生成之前的最后一条用户消息。环境还可以定义一个`get_reward`方法，不采取任何措施
  参数并返回 `float`：当存在时，环境拥有奖励，并且 `get_reward` 被调用
  每次完成部署一次，根据环境的内部状态对其进行评分。它作为一个额外的
  奖励源（权重为 1，记录在环境的类名下）与 `reward_funcs` 一起，其中
  然后变成可选的。对于单个可调用对象，每个示例都使用相同的环境，每个部署都有一个实例，因此它们的
  交互保持隔离。对于字典，每个示例都必须带有一个 `environment` 字段来选择其
  按名称指定环境，并且只有该环境的工具会在其提示中公开 - 让单个运行混合
  任务（例如编码环境和游戏）。此功能是实验性的，可能会更改或删除
  任何时间，恕不另行通知。
- **rollout_worker**（`RolloutWorkerProtocol`，*可选*）--
  自定义部署工作人员实施`RolloutWorkerProtocol`。如果`None`，则默认`AsyncRolloutWorker`
  创建后，它会生成一个无 CUDA 的子进程，并根据训练者的完成情况进行评分
  `reward_funcs`。传递一个自定义工作人员以插入不同的部署/评分后端 - 例如，
  一种在自己的 GPU 上运行奖励模型的模型。
- **权重转移**（`WeightTransferProtocol`，*可选*）--
  自定义权重同步后端实现`WeightTransferProtocol`。如果`None`，则默认
  创建`WeightTransferClient`，将训练师的权重流式传输到配置的 vLLM 服务器中
  国家癌症中心。这与 `rollout_worker` 无关：自定义 rollout Worker 仍然获得权重同步。通过无操作
  实施以禁用训练器端重量同步。组相对策略优化 (GRPO) 方法的培训师。该算法最初是在
纸[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
Models](https://huggingface.co/papers/2402.03300)。该训练器是 GRPO 的异步版本，其中
生成被卸载到外部 vLLM 服务器，该服务器与训练、解耦部署一起异步运行
来自梯度更新循环。

示例：

```python
>>> from trl.experimental.async_grpo import AsyncGRPOTrainer
>>> from trl.rewards import accuracy_reward
>>> from datasets import load_dataset

>>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

>>> trainer = AsyncGRPOTrainer(
...     model="Qwen/Qwen2.5-0.5B-Instruct",
...     reward_funcs=accuracy_reward,
...     train_dataset=dataset,
... )
>>> trainer.train()
```

## RolloutWorkerProtocol[[trl.experimental.async_grpo.async_grpo_trainer.RolloutWorkerProtocol]]

- **rollout_buffer** (`queue.Queue`) --
  排队训练器排水；工人将得分的`RolloutSample`推到上面。
Rollout Worker 必须实现的接口才能作为 `rollout_worker` 传递到 `AsyncGRPOTrainer`。

默认的 `AsyncRolloutWorker` 生成一个无 CUDA 的子进程，并根据训练器的完成情况进行评分
`reward_funcs`。实现此协议以插入自定义推出/评分后端 - 例如，
在自己的 GPU 上运行奖励模型。

如果工人在 `stale_after_s` 秒内崩溃或停止生产，则引发。

开始生产推广。在初始权重同步后，在火车开始时调用一次。

停止工作线程并释放其资源。在火车末端打电话。

告诉工作人员哪个策略版本现已生效，以便它可以标记或丢弃过时的样本。### 减少内存使用
https://huggingface.co/docs/trl/v1.9.2/reducing_memory_usage.md