<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 命令行界面 (CLI)

TRL 提供了强大的命令行界面 (CLI)，可以使用监督微调 (SFT)、直接偏好优化 (DPO) 等方法来微调大型语言模型 (LLM)。 CLI 抽象了大部分样板文件，让您可以快速且可重复地启动训练作业。

## 命令

目前支持的命令有：

### 训练命令

- `trl dpo`：通过 DPO 微调 LLM
- `trl grpo`：使用 GRPO 微调法学硕士
- `trl kto`：通过 KTO 微调 LLM
- `trl reward`：训练奖励模型
- `trl rloo`：用 RLOO 微调法学硕士
- `trl sft`：用 SFT 微调 LLM

### 其他命令

- `trl env`：获取系统信息
- `trl vllm-serve`：使用 vLLM 提供模型

## 使用 TRL CLI 进行微调

### 基本用法

您可以通过指定所需参数（例如模型和数据集）直接从 CLI 启动训练：

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name stanfordnlp/imdb
```

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name anthropic/hh-rlhf
```

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized
```

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward
```

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward
```

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto-mix-14k
```

### 使用配置文件

为了保持 CLI 命令干净且可重复，您可以在 YAML 配置文件中定义所有训练参数：

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
```

启动：

```bash
trl sft --config sft_config.yaml
```

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: anthropic/hh-rlhf
```

启动：

```bash
trl dpo --config dpo_config.yaml
```

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/ultrafeedback_binarized
```

启动：

```bash
trl reward --config reward_config.yaml
```

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
```

启动：

```bash
trl grpo --config grpo_config.yaml
```

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
```

启动：

```bash
trl rloo --config rloo_config.yaml
```

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/kto-mix-14k
```

启动：```bash
trl kto --config kto_config.yaml
```

### 通过加速扩大规模

TRL CLI 本身支持 [🤗 Accelerate](https://huggingface.co/docs/accelerate)，可以轻松地跨多个 GPU、机器扩展训练，或使用 DeepSpeed 等高级设置 — 所有这些都来自同一个 CLI。

您可以将任何 `accelerate launch` 参数直接传递给 `trl`，例如 `--num_processes`。欲了解更多信息，请参阅[Using accelerate launch](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch)。

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name stanfordnlp/imdb \
  --num_processes 4
```

或者，使用配置文件：

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
num_processes: 4
```

启动：

```bash
trl sft --config sft_config.yaml
```

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name anthropic/hh-rlhf \
  --num_processes 4
```

或者，使用配置文件：

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: anthropic/hh-rlhf
num_processes: 4
```

启动：

```bash
trl dpo --config dpo_config.yaml
```

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --num_processes 4
```

或者，使用配置文件：

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/ultrafeedback_binarized
num_processes: 4
```

启动：

```bash
trl reward --config reward_config.yaml
```

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --num_processes 4
```

或者，使用配置文件：

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
num_processes: 4
```

启动：

```bash
trl grpo --config grpo_config.yaml
```

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --num_processes 4
```

或者，使用配置文件：

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
num_processes: 4
```

启动：

```bash
trl rloo --config rloo_config.yaml
```

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto-mix-14k \
  --num_processes 4
```

或者，使用配置文件：

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/kto-mix-14k
num_processes: 4
```

启动：

```bash
trl kto --config kto_config.yaml
```

### 使用`--accelerate_config`加速配置

`--accelerate_config` 标志可让您轻松使用 [🤗 Accelerate](https://github.com/huggingface/accelerate) 配置分布式训练。该标志接受：

- 预定义配置文件的名称（内置于 TRL 中），或者
- 自定义 Accelerate YAML 配置文件的路径。

#### 预定义的配置文件

TRL 提供了多种即用型 Accelerate 配置来简化常见的训练设置：|名称 |描述 |
| --- | --- |
| `fsdp1` |完全分片数据并行阶段 1 |
| `fsdp2` |完全分片数据并行阶段 2 |
| `zero1` | DeepSpeed ZeRO 第 1 阶段 |
| `zero2` | DeepSpeed ZeRO 第 2 阶段 |
| `zero3` | DeepSpeed ZeRO 第 3 阶段 |
| `multi_gpu` |多 GPU 训练 |
| `single_gpu` |单 GPU 训练 |

要使用其中之一，只需将名称传递给`--accelerate_config`。 TRL会自动从`trl/accelerate_config/`加载相应的配置文件。

#### 用法示例

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name stanfordnlp/imdb \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

或者，使用配置文件：

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

启动：

```bash
trl sft --config sft_config.yaml
```

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name anthropic/hh-rlhf \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

或者，使用配置文件：

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: anthropic/hh-rlhf
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

启动：

```bash
trl dpo --config dpo_config.yaml
```

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

或者，使用配置文件：

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/ultrafeedback_binarized
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

启动：

```bash
trl reward --config reward_config.yaml
```

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

或者，使用配置文件：

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

启动：

```bash
trl grpo --config grpo_config.yaml
```

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

或者，使用配置文件：

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

启动：

```bash
trl rloo --config rloo_config.yaml
```

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto-mix-14k \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

或者，使用配置文件：

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/kto-mix-14k
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

启动：

```bash
trl kto --config kto_config.yaml
```

### 使用数据集混合物

您可以使用数据集混合将多个数据集组合成单个训练数据集。这对于对不同数据源进行训练或想要混合不同类型的数据时非常有用。

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: stanfordnlp/imdb
  - path: roneneldan/TinyStories
```

启动：

```bash
trl sft --config sft_config.yaml
```

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: BAAI/Infinity-Preference
  - path: argilla/Capybara-Preferences
```

启动：

```bash
trl dpo --config dpo_config.yaml
```

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: trl-lib/tldr-preference
  - path: trl-lib/lm-human-preferences-sentiment
```

启动：

```bash
trl reward --config reward_config.yaml
```

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: HuggingFaceH4/Polaris-Dataset-53K
  - path: trl-lib/DeepMath-103K
reward_funcs:
  - accuracy_reward
```

启动：

```bash
trl grpo --config grpo_config.yaml
```

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: HuggingFaceH4/Polaris-Dataset-53K
  - path: trl-lib/DeepMath-103K
reward_funcs:
  - accuracy_reward
```

启动：

```bash
trl rloo --config rloo_config.yaml
```

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: trl-lib/kto-mix-14k
  - path: argilla/ultrafeedback-binarized-preferences-cleaned
```

启动：

```bash
trl kto --config kto_config.yaml
```要查看用于定义数据集混合的所有可用关键字，请参阅 [scripts.utils.DatasetConfig](/docs/trl/v1.9.2/en/script_utils#trl.scripts.utils.DatasetConfig) 和 [DatasetMixtureConfig](/docs/trl/v1.9.2/en/script_utils#trl.DatasetMixtureConfig) 类。

## 获取系统信息

您可以通过运行以下命令获取系统信息：

```bash
trl env
```

这将打印出系统信息，包括 GPU 信息、CUDA 版本、PyTorch 版本、变压器版本、TRL 版本以及已安装的任何可选依赖项。

```txt
Copy-paste the following information when reporting an issue:

- Platform: Linux-5.15.0-1048-aws-x86_64-with-glibc2.31
- Python version: 3.11.9
- PyTorch version: 2.4.1
- accelerator(s): NVIDIA H100 80GB HBM3
- Transformers version: 4.45.0.dev0
- Accelerate version: 0.34.2
- Accelerate config: 
  - compute_environment: LOCAL_MACHINE
  - distributed_type: DEEPSPEED
  - mixed_precision: no
  - use_cpu: False
  - debug: False
  - num_processes: 4
  - machine_rank: 0
  - num_machines: 1
  - rdzv_backend: static
  - same_network: True
  - main_training_function: main
  - enable_cpu_affinity: False
  - deepspeed_config: {'gradient_accumulation_steps': 4, 'offload_optimizer_device': 'none', 'offload_param_device': 'none', 'zero3_init_flag': False, 'zero_stage': 2}
  - downcast_bf16: no
  - tpu_use_cluster: False
  - tpu_use_sudo: False
  - tpu_env: []
- Datasets version: 3.0.0
- HF Hub version: 0.24.7
- TRL version: 0.12.0.dev0+acb4d70
- bitsandbytes version: 0.41.1
- DeepSpeed version: 0.15.1
- Diffusers version: 0.30.3
- Liger-Kernel version: 0.3.0
- LLM-Blender version: 0.0.2
- OpenAI version: 1.46.0
- PEFT version: 0.12.0
- vLLM version: not installed
```

报告问题时需要此信息。

### 纳什-MD 训练师
https://huggingface.co/docs/trl/v1.9.2/nash_md_trainer.md