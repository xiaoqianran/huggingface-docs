<!-- huggingface-docs: machine-translated zh-CN from English source -->

#GRPO 培训师

[⟦T625⟧](https://huggingface.co/models?other=grpo,trl)

## 概述

TRL 支持 GRPO Trainer 来训练语言模型，如 [Zhihong Shao](https://huggingface.co/syhia)、[Peiyi Wang](https://huggingface.co/peiyiwang89)、[Qihao Zhu](https://huggingface.co/zqh11)、Runxin Xu、[Junxiao Song](https://huggingface.co/haha-point)、Mingchuan Zhu、Y.K. Li、Y. Wu、[Daya Guo](https://huggingface.co/guoday) 的论文 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) 中所述。

论文摘要如下：

> 由于其复杂性和结构化性质，数学推理对语言模型提出了重大挑战。在本文中，我们介绍了 DeepSeekMath 7B，它继续使用来自 Common Crawl 的 120B 数学相关标记以及自然语言和代码数据来预训练 DeepSeek-Coder-Base-v1.5 7B。 DeepSeekMath 7B 在不依赖外部工具包和投票技术的情况下，在竞赛级 MATH 基准测试中取得了 51.7% 的骄人成绩，接近 Gemini-Ultra 和 GPT-4 的性能水平。 DeepSeekMath 7B 的 64 个样本的自一致性在 MATH 上达到 60.9%。 DeepSeekMath 的数学推理能力归因于两个关键因素：首先，我们通过精心设计的数据选择管道来利用公开可用的网络数据的巨大潜力。其次，我们介绍组相对策略优化（GRPO），它是近端策略的一种变体优化（PPO），增强数学推理能力，同时优化 PPO 的内存使用。

这个训练后的方法是[Quentin Gallouédec](https://huggingface.co/qgallouedec)贡献的。

## 快速开始

此示例演示如何使用 GRPO 方法训练模型。我们根据 [DeepMath-103K dataset](https://huggingface.co/datasets/trl-lib/DeepMath-103K) 的提示训练[Qwen2.5 0.5B Instruct model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。您可以在此处查看数据集中的数据：

<iframe
  src="https://huggingface.co/datasets/trl-lib/DeepMath-103K/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

下面是训练模型的脚本。

```python
# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_grpo.py
```

分布在 8 个 GPU 上，训练大约需要 1 天。

![GRPO curves](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_curves.png)

> **注意：** 上面的奖励曲线是用`Qwen/Qwen2-0.5B-Instruct`生成的。 `Qwen/Qwen2.5-0.5B-Instruct` 的结果在质量上是相似的。

## 深入研究 GRPO 方法

GRPO 是一种在线学习算法，这意味着它通过使用训练模型本身在训练过程中生成的数据来迭代改进。 GRPO 目标背后的直觉是最大化生成的完井优势，同时确保模型保持接近参考政策。要了解 GRPO 的工作原理，可以将其分为四个主要步骤：**生成补全**、**计算优势**、**估计 KL 散度**和**计算损失**。![GRPO visual](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_visual.png)

### 生成补全

在每个训练步骤中，我们对一批提示进行采样，并为每个提示生成一组 \\( G \\) 补全（表示为 \\( o_i \\)）。

### 计算优势

对于每个 \\( G \\) 序列，我们使用奖励模型或奖励函数计算奖励。为了与奖励模型的比较性质保持一致（通常在同一问题的输出之间的比较数据集上进行训练），计算优势以反映这些相对比较。其标准化如下：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

这种方法的名称为：**组相对策略优化 (GRPO)**。

> [!提示]
> 论文[Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783)表明，按 \\( \text{std}(\mathbf{r}) \\) 进行缩放可能会导致问题级别的难度偏差。您可以通过在 [GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig) 中设置 `scale_rewards=False` 来禁用此缩放。
> 请注意，关闭基于标准的缩放也会消除方差归一化，因此更新幅度直接取决于原始奖励规模和批次组成。> [!提示]
> 如[Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)](https://huggingface.co/papers/2508.08221)所示，计算本地（组）级别的平均值和全局（批次）级别的标准差可以实现更稳健的奖励塑造。您可以通过在[GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig)中设置`scale_rewards="batch"`来使用此缩放策略。

### 估计 KL 散度

KL 散度是使用[Schulman et al. (2020)](http://joschu.net/blog/kl-approx.html)引入的近似器来估计的。近似器定义如下：

$$\mathbb{D}_{\text{KL}}\left[\pi_\theta \|\pi_{\text{ref}}\right] = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1,
$$

### Computing the loss

The objective is to maximize the advantage while ensuring that the model remains close to the reference policy. Consequently, the loss is defined as follows:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\left[\pi_\theta(o_{i,t} \mid q, o_{i,< t})\right]_{\text{no grad}}} \hat{A}_{i,t} - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right],
$$

where the first term represents the scaled advantage and the second term penalizes deviations from the reference policy through KL divergence.

> [!提示]
> 请注意，与 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) 中的原始公式相比，我们不按 \\( \frac{1}{|o_i|} \\) 进行缩放，因为论文 [Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783) 中表明这会引入响应级别长度偏差。更多详情参见[loss types](#loss-types)。> [!提示]
> 请注意，与 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) 中的原始公式相比，我们默认使用 \\( \beta = 0.0 \\)，这意味着不使用 KL 散度项。这一选择是由最近的几项研究（例如[Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://huggingface.co/papers/2503.24290)）推动的，这些研究表明 KL 散度项对于 GRPO 训练来说并不是必需的。因此，排除它已成为常见做法（例如[Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783)、[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://huggingface.co/papers/2503.14476)）。如果您希望包含 KL 散度项，可以将 [GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig) 中的 `beta` 设置为非零值。

在原始论文中，该公式被推广为通过利用**裁剪代理目标**来解释每次生成后的多次更新（表示为 \\( \mu \\)，可以使用 [GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig) 中的 `num_iterations` 设置）：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \left[ \min \left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})} \hat{A}_{i,t}, \, \text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right],
$$

where  \\(\text{clip}(\cdot, 1 - \epsilon, 1 + \epsilon) \\) ensures that updates do not deviate excessively from the reference policy by bounding the policy ratio between  \\( 1 - \epsilon \\) and  \\( 1 + \epsilon \\).
When  \\( \mu = 1 \\) (default in TRL), the clipped surrogate objective simplifies to the original objective.

#### Loss Types

Several formulations of the objective have been proposed in the literature. Initially, the objective of GRPO was defined as follows:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} l_{i,t},
$$

where

$$
l_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\left[\pi_\theta(o_{i,t} \mid q, o_{i,< t})\right]_{\text{no grad}}} \hat{A}_{i,t} - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right].
$$

The ⟦T652⟧ highlights the limitations of the GRPO algorithm’s sample-level loss in long-CoT scenarios, where longer responses are under-penalized, leading to poorer quality outputs. The proposed solution is a token-level normalization, which better handles longer sequences by assigning more balanced rewards to individual tokens, regardless of response length:

$$
\mathcal{L}_{\text{DAPO}}(\theta) = - \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} l_{i,t},
$$

To use this formulation, set ⟦T39⟧ in ⟦T653⟧.

Furthermore, it was demonstrated in the paper ⟦T654⟧ that the initial GRPO formulation introduces a response length bias. They show that while the DAPO formulation reduces this bias, it does not eliminate it completely. To fully remove this bias, they propose dividing by a constant instead of the sequence length, resulting in the following formulation:

$$
\mathcal{L}_{\text{Dr. GRPO}}(\theta) = - \frac{1}{LG} \sum_{i=1}^G \sum_{t=1}^{|o_i|} l_{i,t},
$$

This constant is recommended to be the maximum completion length. To use this formulation, set ⟦T40⟧ in the ⟦T655⟧.

Alternatively, in the ⟦T656⟧, the Qwen team proposes replacing the "hard" clipping mechanism of GRPO with a smooth, temperature-controlled soft gating mechanism. While GRPO zeroes out gradients when the policy deviates too far from the reference, SAPO uses a soft trust region that smoothly decays the gradient weight. This allows the model to retain useful learning signals from "near-on-policy" tokens while suppressing noise from extreme deviations.

The loss function is defined as:

$$
\mathcal{L}_{\text{SAPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} f_{i,t} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \right) \hat{A}_{i,t}
$$

The soft-gating function  \\( f_{i,t} \\) is defined using the sigmoid function  \\( \sigma \\) as:

$$
f_{i,t}(x) = \sigma \left( \tau_{i,t} (x - 1) \right) \cdot \frac{4}{\tau_{i,t}}
$$

The temperature  \\( \tau_{i,t} \\) is chosen based on the sign of the advantage  \\( \hat{A}_{i,t} \\):

$$
\tau_{i,t} = \begin{cases} 
\tau_{\text{pos}}, & \text{if } \hat{A}_{i,t} > 0 \\
\tau_{\text{否定}}, & \text{否则}
\结束{案例}
$$他们建议使用不对称温度 \\( \tau_{\text{neg}} > \tau_{\text{pos}} \\) （默认值为 \\( \tau_{\text{pos}}=1.0, \tau_{\text{neg}}=1.05 \\) ）。这确保了模型对“坏”行为受到更严格的惩罚，以防止不稳定，同时对“好”行为更加宽容。

要使用此公式，请在 [GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig) 中设置 `loss_type="sapo"`。

## 记录的指标

在培训和评估时，我们记录以下指标：

- `num_tokens`：迄今为止处理的令牌总数，包括提示和完成。使用工具时，仅计算非工具令牌。
- `step_time`：每个训练步骤（包括生成）所花费的平均时间（以秒为单位）。
- `completions/mean_length`：生成的补全的平均长度。使用工具时，仅计算非工具令牌。
- `completions/min_length`：生成的补全的最小长度。使用工具时，仅计算非工具令牌。
- `completions/max_length`：生成的补全的最大长度。使用工具时，仅计算非工具令牌。
- `completions/mean_terminated_length`：以 EOS 终止的生成完成的平均长度。使用工具时，仅计算非工具令牌。- `completions/min_terminated_length`：以 EOS 终止的生成完成的最小长度。使用工具时，仅计算非工具令牌。
- `completions/max_terminated_length`：以 EOS 终止的生成完成的最大长度。使用工具时，仅计算非工具令牌。
- `completions/clipped_ratio`：截断（剪辑）完成的比率。
- `rewards/{reward_func_name}/mean`：特定奖励函数的平均奖励。当环境通过 `get_reward` 拥有奖励时，`{reward_func_name}` 是环境的类名称。
- `rewards/{reward_func_name}/std`：奖励与特定奖励函数的标准差。
- `reward`：各功能奖励相加后的总体平均奖励（按`reward_weights`加权）。
- `reward_std`：跨函数总奖励的标准差（由`reward_weights`加权），在整个批次上计算。
- `frac_reward_zero_std`：生成批次中奖励标准为零的样本比例，这意味着该提示几乎没有多样性（所有答案都正确或不正确）。
- `policy_loss`：策略梯度损失值（在任何熵奖励之前）。当`entropy_coef`非零或`use_adaptive_entropy=True`时记录。
- `entropy`：生成的完成中标记预测的平均熵。 （如果`mask_truncated_completions=True`，则排除屏蔽序列标记。）- `entropy_coef`：当前熵正则化系数。当`entropy_coef`非零或`use_adaptive_entropy=True`时记录。当 `use_adaptive_entropy=True` 时，每个优化器步骤更新一次。
- `kl`：模型和参考模型之间的平均 KL 散度，根据生成的补全计算得出。仅当 `beta` 非零时才记录。
- `clip_ratio/region_mean`：GRPO 目标被剪切以保持在信任区域内的标记（或序列，如果`importance_sampling_level="sequence"`）概率的比率： \\( \text{clip}\left( r_{i,t}(\theta), 1 - \epsilon_\mathrm{low}, 1 + \epsilon_\mathrm{high} \right)\,, \quad r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})} \\). A higher value means more tokens are clipped, which constrains how much the policy $\pi_\theta$ can change.
- ⟦T73⟧: The average ratio of token (or sequence, if ⟦T74⟧) probabilities that were clipped on the lower bound of the trust region:  \\(r_{i,t}(\theta) < 1 - \epsilon_\mathrm{low}\\).
- ⟦T75⟧: The smallest per-completion fraction of tokens (or the sequence itself, if ⟦T76⟧) clipped on the lower bound of the trust region:  \\(r_{i,t}(\theta) < 1 - \epsilon_\mathrm{low}\\).
- ⟦T77⟧: The average ratio of token (or sequence, if ⟦T78⟧) probabilities that were clipped on the upper bound of the trust region:  \\(r_{i,t}(\theta) > 1 + \epsilon_\mathrm{high}\\)。
- `clip_ratio/high_max`：在信任区域的上限上剪切的标记的最大每次完成分数（或序列本身，如果`importance_sampling_level="sequence"`）：\\(r_{i,t}(\theta) > 1 + \epsilon_\mathrm{high}\\)。

## 定制

### 利用 vLLM 驱动的生成加速训练

使用在线方法进行训练时，生成通常是主要瓶颈。为了加速生成，您可以使用[vLLM](https://github.com/vllm-project/vllm)，这是一种适用于法学硕士的高吞吐量、低延迟推理引擎。要启用它，首先使用以下命令安装软件包

```shell
pip install trl[vllm]
```我们支持在训练期间使用 vLLM 的两种方式：**服务器模式**和**共置模式**。

> [!提示]
> 默认情况下，vLLM 生成会激活截断重要性采样，以解决使用不同框架时出现的生成训练不匹配问题。可以通过设置`vllm_importance_sampling_correction=False`关闭此功能。欲了解更多信息，请参阅[Truncated Importance Sampling](paper_index#truncated-importance-sampling)

#### 选项 1：共置模式

在此模式下，vLLM 在训练器进程内运行并与训练模型共享 GPU 内存。这避免了启动单独的服务器，并且可以提高 GPU 利用率，但可能会导致训练 GPU 上的内存争用。这是默认模式。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

#### 选项 2：服务器模式

在此模式下，vLLM 在单独的进程中运行（并使用单独的 GPU），并通过 HTTP 与训练器通信。如果您有用于推理的专用 GPU，那么这是理想的选择。

1. **启动 vLLM 服务器**：

   ```bash
   trl vllm-serve --model <model_name>
   ```

2. **在训练脚本中启用服务器模式**：

   ```python
   from trl import GRPOConfig

   training_args = GRPOConfig(
       ...,
       use_vllm=True,
       vllm_mode="server",
   )
   ```

> [!警告]
> 确保服务器使用与训练器不同的 GPU，否则可能会遇到 NCCL 错误。您可以通过 `CUDA_VISIBLE_DEVICES` 环境变量指定要使用的 GPU。> [!提示]
> 根据模型大小和训练的总体 GPU 内存要求，您可能需要调整 [GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig) 中的 `vllm_gpu_memory_utilization` 参数，以避免利用率不足或内存不足错误。
>
> 我们提供了[HF Space](https://huggingface.co/spaces/trl-lib/recommend-vllm-memory)来帮助根据您的模型配置和实验设置估计建议的GPU内存利用率。只需如下使用即可获得`vllm_gpu_memory_utilization`推荐：
>
> 
>
> 如果推荐值在您的环境中不起作用，我们建议在推荐值的基础上添加一个小的缓冲区（例如+0.05或+0.1）以确保稳定性。
>
> 如果您仍然发现出现内存不足错误，请将 `vllm_enable_sleep_mode` 设置为 True，并且 vllm 参数和缓存将在优化步骤期间卸载。欲了解更多信息，请参阅[Reducing Memory Usage with vLLM Sleep Mode](reducing_memory_usage#vllm-sleep-mode)。

> [!提示]
> 默认情况下，GRPO 对 vLLM 使用 `MASTER_ADDR=localhost` 和 `MASTER_PORT=12345`，但您可以通过相应设置环境变量来覆盖这些值。

欲了解更多信息，请参阅[Speeding up training with vLLM](speeding_up_training#vllm-for-fast-generation-in-online-methods)。#### 处理训练与推理不匹配的问题
虽然 vLLM 极大地加速了推理，但它也将推理引擎与训练引擎解耦。理论上，这些引擎在数学上是相同的，但实际上，由于精度效应和硬件特定的优化，它们可以产生不同的输出。这种差异反映了两个系统的不同优化目标。推理引擎旨在最大化采样吞吐量（通常以每秒令牌数来衡量），同时保持可接受的采样保真度。相反，训练框架关注梯度计算的数值稳定性和精度，通常使用更高精度的格式（例如 FP32）作为主权重和优化器状态。这些不同的优先级和限制导致训练和推理之间不可避免地出现微妙的不匹配。这种不匹配会导致梯度更新出现偏差，据观察，这种更新会破坏训练的稳定性([[1]](https://fengyao.notion.site/off-policy-rl)[[2]](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15e dda）[[3]]（https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl）[[4]]（https://huggingface.co/papers/2510.26788）[[5]]（https://huggingface.co/papers/2510.18855））。为简单起见，考虑 REINFORCE 策略梯度：

$$
\nabla_\theta \mathcal{J}(x,\theta)
= \mathbb{E}_{y \sim \pi^\text{train}(\cdot \mid x,\theta)}
\left[ \nabla_\theta \log \pi^\text{train}(y \mid x,\theta) \cdot R(x,y) \right]
$$

这里 \\( x \\) 表示从某些数据分布中采样的提示， \\( \pi^\text{train} \\) 是训练引擎实现的策略。通过循环中的 vLLM，我们获得单独的推理策略 \\( \pi^\text{inference} \\)，因此有效策略梯度变为$$
\nabla_\theta \mathcal{J}_{\text{有偏差}}(x,\theta)
= \mathbb{E}_{y \sim \pi^\text{推理}(\cdot \mid x,\theta)}
\left[ \nabla_\theta \log \pi^\text{train}(y \mid x,\theta) \cdot R(x,y) \right]。
$$

这将原本基于策略的强化学习问题变成了基于策略的强化学习问题。

纠正这种分布变化的标准方法是**重要性抽样（IS）**。我们提供两种 IS 变体：[Truncated Importance Sampling (TIS)](paper_index#truncated-importance-sampling) 和 [Masked Importance Sampling (MIS)](paper_index#masked-importance-sampling)。两种变体都可以应用于令牌级别或序列级别。让 \\( \rho \\) 表示重要性权重，例如每个标记 \\( \rho_t \\) 或每个序列 \\( \rho_{\text{seq}} \\) 。在 TIS 下，超出范围 `[vllm_importance_sampling_clip_min, vllm_importance_sampling_clip_max]` 的比率将被剪裁，

$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})。
$$

最初的[TIS paper](https://huggingface.co/papers/1606.02647)提出了单一上限裁剪机制，即\\(\min(\rho, C_{\max})\\)。 TRL 中的实现通过引入下界来概括这一点，产生上面所示的双边公式，其灵感来自于 [IcePop](paper_index#masked-importance-sampling) 方法。  
请注意，在 IcePop 中，边界标记为 \\( \alpha \\) 和 \\( \beta \\)，而在 TRL 中我们使用 \\( C_{\min} \\) 和 \\( C_{\max} \\)。在 MIS 下，超出此范围的比率将设置为零，因此这些样本对梯度没有贡献。换句话说，在 TIS 下，离群样本的权重会降低，在 MIS 下会被丢弃。配置标志`vllm_importance_sampling_mode`选择IS变体（屏蔽或截断）和粒度（令牌级别或序列级别）。

重要性采样是对训练-推理不匹配的原则性算法响应。然而，还有更直接的方法试图减少两个发动机本身之间的不匹配。其中大部分是工程解决方案。例如，推理机中的[MiniMax M1 uses an FP32 language model head](https://huggingface.co/papers/2506.13585)。 Thinking Machines 已经探索了[deterministic inference kernels](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，尽管这会带来显着的效率成本。 vLLM 通过在 Thinking Machines 的批量不变确定性内核上构建，已经展示了[bitwise consistent policies](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html)，但截至 2025 年 11 月，相对于标准 vLLM 推理，仍然存在相当大的吞吐量损失。

### 使用 Transformer 连续批处理加速训练作为 vLLM 的替代方案，您可以使用 Transformer 的内置连续批处理引擎来加快生成速度。连续批处理会立即从批次中删除已完成的序列，而不是等待最慢的序列完成。对于具有可变完成长度的任务（例如，数学推理），在大批量大小（N≥32）下，与默认`generate()`相比，这会产生更快的生成速度和更低的 VRAM 使用率。

> [!提示]
> 连续批处理是一种直接升级，无需服务器设置或重量同步。它在进程内运行，非常适合单 GPU 训练或内存受限的环境。为了获得大规模的最大生成吞吐量，请改用 vLLM。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_transformers_continuous_batching=True,
    transformers_continuous_batching_config={
        "use_cuda_graph": False,
        "max_memory_percent": 0.4,  # lower values leave more VRAM for the training backward pass
    },
)
```

> [!提示]
> TRL 默认 `max_memory_percent` 为 `0.5`（而不是 Transformer 的 `0.9`），以便为训练后向传递留下足够的 VRAM。对于大型生成批次 (N≥32) 或者如果您看到内存不足错误，请将其调低至 `0.3`–`0.4`。

有关完整的训练示例，请参阅[⟦T96⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_continuous_batching.py)。

### 大规模 GRPO：在多个节点上训练 70B+ 模型

在训练像 **Qwen2.5-72B** 这样的大型模型时，您需要进行多项关键优化，以使训练高效且可跨多个 GPU 和节点进行扩展。这些包括：- **DeepSpeed ZeRO 第 3 阶段**：ZeRO 利用数据并行性在多个 GPU 和 CPU 之间分配模型状态（权重、梯度、优化器状态），从而减少每个设备上的内存和计算要求。由于大型模型无法适应单个 GPU，因此需要使用 ZeRO Stage 3 来训练此类模型。欲了解更多详情，请参阅[DeepSpeed Integration](deepspeed_integration)。
- **Accelerate**：Accelerate 是一个简化跨多个 GPU 和节点的分布式训练的库。它提供了一个简单的API来启动分布式训练，并处理分布式训练的复杂性，例如数据并行、梯度累积和分布式数据加载。欲了解更多详情，请参阅[Distributing Training](distributing_training)。
- **vLLM**：请参阅上一节，了解如何使用 vLLM 来加速生成。

下面是一个示例 SLURM 脚本，用于在多个节点上使用 GRPO 训练 70B 模型。该脚本在 4 个节点上训练模型，并使用第 5 个节点进行 vLLM 驱动的生成。

```sh
#!/bin/bash
#SBATCH --nodes=5
#SBATCH --gres=gpu:8

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 4 nodes for training and the 5th node for vLLM
TRAIN_NODES="${NODELIST[@]:0:4}"  # Nodes 0, 1, 2, 3 for training
VLLM_NODE="${NODELIST[4]}"  # Node 4 for vLLM

# Run training on the first 4 nodes (Group 1)
srun --nodes=4 --ntasks=4 --nodelist="${NODELIST[@]:0:4}" accelerate launch \
     --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
     --num_processes 32 \
     --num_machines 4 \
     --main_process_ip ${NODELIST[0]} \
     --machine_rank $SLURM_PROCID \
     --rdzv_backend c10d \
     train_grpo.py \
     --server_ip $VLLM_NODE &

# Run vLLM server on the 5th node (Group 2)
srun --nodes=1 --ntasks=1 --nodelist="${NODELIST[4]}" trl vllm-serve --model Qwen/Qwen2.5-72B --tensor_parallel_size 8 &

wait
```

```python
import argparse

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = GRPOConfig(
        per_device_train_batch_size=4,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),  # from ip-X-X-X-X to X.X.X.X
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-72B",
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=dataset
    )
    trainer.train()

if __name__=="__main__":
    main()
```

### 使用自定义奖励函数

[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 支持使用自定义奖励函数而不是密集奖励模型。为了确保兼容性，您的奖励函数必须满足以下要求：奖励函数可以是同步 Python 可调用函数，也可以是异步 `async def` 协程。当您提供多个异步奖励函数时，它们会同时等待（通过`asyncio.gather`并行运行），因此它们的延迟会重叠。

1. **输入参数**：
   - 该函数必须接受以下作为关键字参数：
     - `prompts`（包含提示），
     - `completions`（包含生成的补全），
     - `completion_ids`（包含标记化完成），
     - `trainer_state` ([TrainerState](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerState))：训练器的当前状态。这可用于实现动态奖励功能，例如课程学习，其中奖励根据训练进度进行调整。
     - `log_extra`：可调用的`log_extra(column: str, values: list)`，用于向完成表添加额外的列。请参阅示例 6。在分布式训练中，所有进程都记录同一组密钥非常重要。
     - `log_metric`：可调用的`log_metric(name: str, value: float)`，用于将标量指标与`kl`、`entropy`等一起记录为绘图。请参阅示例 6。在分布式训练中，所有进程都记录相同的键集非常重要。- `environments`：环境实例列表，每次完成一个。仅在提供 `environment_factory` 时出现。使用它来读取剧集期间累积的状态（例如，`env.counter`）。对于有状态的环境，更喜欢通过 `get_reward` 方法让环境拥有奖励——请参阅 [Rewards](#rewards)。
     - 数据集可能具有的所有列名称（但`prompt`）。例如，如果数据集包含名为 `ground_truth` 的列，则将使用 `ground_truth` 作为关键字参数调用该函数。

     满足此要求的最简单方法是在函数签名中使用`**kwargs`。
   - 根据数据集格式，输入会有所不同：
     - 对于 [standard format](dataset_formats#standard)、`prompts` 和 `completions` 将是字符串列表。
     - 对于[conversational format](dataset_formats#conversational)，`prompts`和`completions`将是消息字典列表。

2. **返回值**：函数必须返回浮点数列表。每个浮点数代表一次完成对应的奖励。

#### 示例 1：奖励更长的完成时间

以下是奖励较长完成时间的标准格式奖励函数的示例：

```python
def reward_func(completion_ids, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of token count)."""
    return [float(len(ids)) for ids in completion_ids]
```

您可以按如下方式测试：

```python
>>> prompts = ["The sky is", "The sun is"]  # not used in the reward function, but the trainer will pass it
>>> completions = [" blue.", " in the sky."]  # not used in the reward function, but the trainer will pass it
>>> completion_ids = [[6303, 13], [304, 279, 12884, 13]]
>>> reward_func(prompts=prompts, completions=completions, completion_ids=completion_ids)
[2.0, 4.0]
```

#### 示例 1.1：奖励更长的完成时间（基于字符数）与前面的示例相同，但这次奖励函数基于字符数而不是令牌。

```python
def reward_func(completions, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of character count)."""
    return [float(len(completion)) for completion in completions]
```

您可以按如下方式测试：

```python
>>> prompts = ["The sky is", "The sun is"]
>>> completions = [" blue.", " in the sky."]
>>> completion_ids = [[6303, 13], [304, 279, 12884, 13]]  # not used in the reward function, but the trainer will pass it
>>> reward_func(prompts=prompts, completions=completions, completion_ids=completion_ids)
[6.0, 12.0]
```

#### 示例 2：以特定格式奖励完成情况

下面是一个奖励函数的示例，用于检查完成是否具有特定格式。这个例子的灵感来自于论文[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948)中使用的_formatreward_函数。
它专为对话格式而设计，其中提示和完成由结构化消息组成。

```python
import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

您可以按如下方式测试该功能：

```python
>>> prompts = [
...     [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
...     [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
... ]
>>> completions = [
...     [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
...     [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
... ]
>>> format_reward_func(prompts=prompts, completions=completions)
[1.0, 0.0]
```

#### 示例 3：根据参考奖励完成情况

下面是一个奖励函数的示例，用于检查完成是否正确。这个例子的灵感来自于论文[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948)中使用的_准确度奖励_函数。
此示例是针对 [standard format](dataset_formats#standard) 设计的，其中数据集包含名为 `ground_truth` 的列。

```python
import re

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

您可以按如下方式测试该功能：

```python
>>> prompts = ["Problem: Solve the equation $2x + 3 = 7$. Solution:", "Problem: Solve the equation $3x - 5 = 10$."]
>>> completions = [r" The solution is \boxed{2}.", r" The solution is \boxed{6}."]
>>> ground_truth = ["2", "5"]
>>> reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)
[1.0, 0.0]
```

#### 示例 4：多任务奖励函数下面是在[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)中使用多个奖励函数的示例。在这个例子中，我们定义了两个特定于任务的奖励函数：`math_reward_func`和`coding_reward_func`。 `math_reward_func` 根据数学问题的正确性来奖励，而 `coding_reward_func` 根据解决方案是否有效来奖励编码问题。

```python
from datasets import Dataset
from trl import GRPOTrainer

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(
    [
        {"prompt": "What is 2+2?", "task": "math"},
        {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
        {"prompt": "What is 3*4?", "task": "math"},
        {"prompt": "Write a function that returns the product of two numbers.", "task": "code"},
    ]
)

# Math-specific reward function
def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            # Calculate math-specific reward
            correct = check_math_solution(prompt, completion)
            reward = 1.0 if correct else -1.0
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            rewards.append(None)
    return rewards

# Coding-specific reward function
def coding_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "coding":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards

# Use both task-specific reward functions
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)

trainer.train()
```

在此示例中，`math_reward_func`和`coding_reward_func`设计用于处理包含数学和编码问题的混合数据集。数据集中的`task`列用于确定对每个问题应用哪个奖励函数。如果数据集中的样本没有相关的奖励函数，则奖励函数将返回`None`，[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)将继续执行有效的函数和任务。这使得[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)能够处理具有不同适用性的多种奖励功能。

请注意，[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)将忽略奖励函数返回的`None`奖励，仅考虑相关函数返回的奖励。这确保了模型针对相关任务进行训练，并忽略没有相关奖励函数的任务。

#### 示例 5：异步奖励函数自定义奖励函数也可以定义为 `async def` 协程。如果您的奖励取决于缓慢的 I/O（例如，调用远程服务），这非常有用。当您传递多个异步奖励函数时，[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)会同时执行它们，因此它们的延迟会重叠。

下面是模拟 I/O 绑定操作的异步奖励函数的最小示例：

```python
import asyncio

async def async_reward_func(prompts, completions, **kwargs):
    # Simulate an I/O-bound call (e.g., HTTP request, database lookup)
    await asyncio.sleep(0.01)
    # Simple toy reward: 1.0 if the completion is non-empty, else 0.0
    return [1.0 if completion else 0.0 for completion in completions]
```

#### 示例 6：记录额外的列和指标

下面是一个奖励函数的示例，它将额外的列记录到完成表中，并将标量指标记录为绘图。

```python
import re

def reward_func(completions, ground_truth, log_extra=None, log_metric=None, **kwargs):
    extracted = [re.search(r"\\boxed\{(.*?)\}", c) for c in completions]
    extracted = [m.group(1) if m else None for m in extracted]
    rewards = [1.0 if e == gt else 0.0 for e, gt in zip(extracted, ground_truth)]

    if log_extra:
        log_extra("golden_answer", list(ground_truth))
        log_extra("extracted_answer", [e or "[none]" for e in extracted])

    if log_metric:
        log_metric("accuracy", sum(rewards) / len(rewards))

    return rewards
```

#### 将奖励函数传递给训练器

要使用您的自定义奖励函数，请将其传递给[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)，如下所示：

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    reward_funcs=reward_func,
    ...,
)
```

您可以将多个奖励函数作为列表传递；该列表可能包括同步和异步函数：

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    reward_funcs=[reward_func, async_reward_func1, async_reward_func2],
    ...,
)
```

奖励将计算为每个函数奖励的总和，或者如果配置中提供了`reward_weights`，则计算为加权总和。

请注意，[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)支持多种不同类型的奖励功能。有关更多详细信息，请参阅参数文档。

### 熵正则化为了鼓励探索并防止策略崩溃为接近确定性的输出，您可以向训练目标添加熵奖励。熵正则化增强了 GRPO 损失，如下所示：

$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta) - \alpha \cdot \mathcal{H}(\pi_\theta),
$$

其中 \\(\mathcal{H}(\pi_\theta)\\) 是策略的平均每个令牌熵， \\(\alpha\\) 是熵系数。无论`loss_type`如何，奖励始终是每个令牌的平均熵；它不会重新调整以匹配损失类型的策略标准化（例如GRPO博士的`batch_size * max_completion_length`分母），因此`entropy_coef`对于每种损失类型具有相同的含义。

**静态熵** — 整个训练过程中的固定系数：

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(entropy_coef=0.05, ...)
```

**自适应熵** — 根据目标熵在每个优化器步骤中更新系数，如 [Skywork-OR1](https://huggingface.co/papers/2505.22312) 中所述。当当前熵等于或低于`entropy_target`时，系数增加`entropy_coef_delta`；否则它会递减。仅当熵等于或低于目标时才应用系数（即非零）：

```python
training_args = GRPOConfig(
    entropy_coef=0.01,          # initial coefficient
    use_adaptive_entropy=True,
    entropy_target=5.0,         # target mean per-token entropy (nats); tune for your model
    entropy_coef_delta=0.005,   # step size per optimizer step
    entropy_coef_min=0.0,
    entropy_coef_max=1.0,
    ...
)
```典型的语言模型的每个标记的熵为 2-10 nat，因此默认的`entropy_target=0.2`几乎从不触发正则化——只有当熵等于或低于目标时，即接近完全崩溃时，奖励才会起作用。将其设置为对您的模型有意义的值，例如接近您在训练早期观察到的熵（记录为 `entropy` 指标）。当使用`top_entropy_quantile < 1.0`时，`entropy_target`适用于高熵令牌子集——该子集的熵将高于记录的完整令牌`entropy`，因此请相应地进行校准。

当`use_adaptive_entropy=True`时，当前熵系数`entropy_coef`与每个检查点一起保存并在恢复时恢复，因此训练是完全可恢复的。

### GRPO 快速实验

RapidFire AI 是一个开源实验引擎，位于 TRL 之上，让您可以一次启动多个 GRPO 配置，甚至在单个 GPU 上也是如此。 RapidFire 无需按顺序尝试配置，而是让您**更早地查看所有学习曲线，停止表现不佳的运行，并在飞行中使用新设置克隆有前途的运行**，而无需重新启动。欲了解更多信息，请参阅[RapidFire AI Integration](rapidfire_integration)。

## 代理培训GRPO 支持**代理训练**：模型在生成过程中调用工具并从结果中学习。

- **工具**是暴露给模型的普通 Python 函数（同步或异步）。使用 `tools` 进行无状态调用（计算器、网络搜索）。
- **环境**是更通用的形式：每次部署时新鲜构建的有状态对象，其公共方法作为工具公开，加上一个`reset`生命周期挂钩和一个可选的`get_reward`，让它拥有奖励。当您需要每次部署状态、重置挂钩或环境拥有的奖励时，请使用`environment_factory`。

它们组成 — 您可以将独立的 `tools` 与 `environment_factory` 一起传递。

### 工具

`tools` 参数需要一个 Python 函数列表（同步或异步），用于定义代理可用的工具：

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    tools=[tool1, tool2],
    ...,
)
```

每个工具必须是一个标准的 Python 函数，具有 **类型提示的参数和返回类型**，以及描述其用途、参数和返回值的 **Google 风格的文档字符串**。
欲了解更多详情，请参阅[Passing tools guide](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools)。> [!提示]
> GRPO 工具调用循环要求聊天模板“保留前缀”（附加工具消息不得更改较早消息的呈现方式）。对于已知的模型系列（例如 Qwen3、DeepSeek-V3），启用工具后，TRL 会自动交换已修补的训练模板。完整列表请参见[Chat Templates](chat_templates#training-templates)。

示例：

```python
from trl import GRPOTrainer

def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the two integers.
    """
    return a * b

async def async_add(a: int, b: int) -> int:
    """
    Asynchronously adds two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b

trainer = GRPOTrainer(
    tools=[multiply, async_add],
    ...,
)
```

### 环境

您还可以通过`environment_factory`提供工具。在此模式下，[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 每次部署都会创建一个环境实例，并将环境的公共方法公开为工具。

> [!重要]
> `environment_factory` 需要 `transformers>=5.2.0`。

当您使用`environment_factory`时，环境拥有数据：没有外部`train_dataset`。每次推出时，`reset()`都会生成任务——从环境所拥有的语料库中进行自我采样，或按程序生成状态——并返回提示。 `max_steps` 设置训练长度。以下是对目标进行自采样并公开 `increment` 方法（作为工具公开）的环境的最小示例。

```python
import random

from trl import GRPOConfig, GRPOTrainer

class IncrementEnv:
    # Reserved methods
    def reset(self, **kwargs) -> str | None:  # required; called at the start of each rollout
        self.counter = 0
        self.target = random.randint(1, 6)  # self-sample the task
        return f"Increment the counter by {self.target}."  # returned string becomes the prompt

    def get_reward(self) -> float:  # optional: the environment scores itself from its own state
        return float(self.counter == self.target)

    # Public methods (exposed as tools)
    def increment(self, step: int) -> int:
        """
        Increment the internal counter.

        Args:
            step: Value to add to the counter.

        Returns:
            The updated counter value.
        """
        self.counter += step
        return self.counter

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=GRPOConfig(max_steps=1000, chat_template_kwargs={"enable_thinking": False}),
    environment_factory=IncrementEnv,
)
trainer.train()
```

环境类有两个保留方法：`reset`和`get_reward`。与任何其他公共方法不同，这些方法不会作为工具暴露给模型。- `reset`（必需）在推出开始时调用。它可以返回 `None` 或字符串。在 GRPO 中，当它返回一个字符串时，该字符串就是用户提示符。
- `get_reward`（可选，同步或异步）不接受任何参数并返回`float`：环境根据其自身的内部状态对刚刚运行的剧集进行评分（游戏是否以胜利结束？这个词被猜到了吗？）。每次完成的部署都会调用一次，并充当奖励来源。

> [!注意]
> 由于环境在每个 `reset()` 上进行自采样，GRPO 组的 `G` 成员可能不会共享相同的初始状态，从而使组基线稍微嘈杂。对于组内的相同状态，让`reset()`从共享密钥确定性地导出状态。

#### 提供外部数据集（可选）

您可以提供外部`train_dataset`，而不是让环境自行采样。例如，当您的任务已经存在于数据集中，或者在 [multiple environments](#multiple-environments) 之间路由时。每次推出时，训练器都会对一行进行采样，使用其 `"prompt"` 列作为提示，并将该行的其他列作为关键字参数传递给 `reset()`（以便环境从数据集中读取任务）。

```python
from datasets import Dataset
from trl import GRPOTrainer

# Each row carries the prompt and the task data (here, the counter target to reach).
dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": f"Increment the counter to {i}."}] for i in range(1, 7)],
        "target": list(range(1, 7)),
    }
)

class IncrementEnv:
    def reset(self, target, **kwargs) -> None:  # the row's `target` column arrives as a keyword argument
        self.counter = 0
        self.target = target

    def get_reward(self) -> float:
        return float(self.counter == self.target)

    def increment(self, step: int) -> int:  # exposed as a tool (see the full docstring above)
        self.counter += step
        return self.counter

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,  # the dataset owns the task; the "target" column reaches reset(target=...)
    environment_factory=IncrementEnv,
)
trainer.train()
```

### 奖励环境拥有的奖励和训练师拥有的奖励并不排斥。通过`reward_funcs`获得从完成中自然计算的奖励（例如对文本的格式检查）并让环境通过`get_reward`返回其与状态相关的奖励。所有来源均已汇总； `reward_weights` 仅适用于`reward_funcs`（环境拥有自己的规模）。

```python
def format_reward(completions, **kwargs):  # trainer-owned: scores the completion text
    ...

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=GRPOConfig(max_steps=1000),
    reward_funcs=format_reward,          # trainer-owned: scores the completion
    environment_factory=IncrementEnv,    # env-owned: scores its internal state via `get_reward`
)
```

`reward_func`还可以直接读取环境状态：当设置`environment_factory`时，训练器传递一个`environments` kwarg（每次完成一个实例）。这有时很有用，但对于依赖于状态的奖励，更喜欢`get_reward`。

这对于 [⟦T180⟧](async_grpo_trainer) 来说是相同的。

### 多个环境

要在一次运行中训练混合任务（例如编码任务和游戏），请将映射环境名称的字典传递给工厂，而不是单个可调用的。每个部署都会通过数据集中的 `environment` 字段选择其环境，并且**仅公开该环境的工具**。这可以避免泄漏不相关的工具（游戏的`move`工具到编码示例，反之亦然）。每个环境仍然拥有自己的数据并从`reset()`返回提示，因此数据集只需要`environment`列在它们之间路由。

```python
import random

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

class CodingEnv:
    def reset(self, **kwargs) -> str:
        n = random.randint(1, 20)
        return f"Compute the {n}th Fibonacci number."

    def run_code(self, code: str) -> str:  # exposed as a tool only for "coding" examples
        """Run the given Python code and return its stdout.

        Args:
            code: The Python source to execute.

        Returns:
            The captured standard output.
        """
        ...

class GameEnv:
    def reset(self, **kwargs) -> str:
        return "Reach the goal tile."

    def move(self, action: str) -> str:  # exposed as a tool only for "game" examples
        """Apply an action to the game and return the new observation.

        Args:
            action: One of "up", "down", "left", "right".

        Returns:
            The observation after the move.
        """
        ...

# The dataset only routes: each row picks the environment for that rollout
dataset = Dataset.from_dict({"environment": ["coding", "game"] * 500})

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=GRPOConfig(chat_template_kwargs={"enable_thinking": False}),
    train_dataset=dataset,
    reward_funcs=reward_func,
    environment_factory={"coding": CodingEnv, "game": GameEnv},
)
trainer.train()
```奖励函数仍然通过`reward_kwargs["environments"]`接收每个示例的环境，因此它可以在环境类型上分支（例如`isinstance(env, CodingEnv)`）。环境实例可以跨步骤重用（在情节之间重置），因此昂贵的`__init__`只需支付一次，而不是每个步骤。 `environment` 字段由训练器消耗，并且**不**转发到 `reset`。

### 多模式工具响应

工具可以通过返回内容块列表来返回图像和文本。这对于 VLM 代理训练非常有用，其中该工具提供视觉反馈（例如屏幕截图、绘图、相机捕获）。

```python
from PIL import Image

def take_screenshot() -> list:
    """
    Takes a screenshot of the current screen.

    Returns:
        The screenshot image with a description.
    """
    img = Image.open("screenshot.png")
    return [{"type": "image", "image": img}, {"type": "text", "text": "Here is the screenshot."}]
```

返回的图像会自动注入到对话中，并传递到 VLM 以供后续生成。

### 支持的型号

测试用：

- [**Gemma4**](https://huggingface.co/collections/google/gemma-4) — 例如，`google/gemma-4-E2B-it`
- **GLM-4-MoE**（[4.5](https://huggingface.co/collections/zai-org/glm-45)、[4.6](https://huggingface.co/collections/zai-org/glm-46) 或 [4.7](https://huggingface.co/collections/zai-org/glm-47)） — 例如，`zai-org/GLM-4.7`
- [**GPT-OSS**](https://huggingface.co/collections/openai/gpt-oss) — 例如，`openai/gpt-oss-20b`
- [**Llama 3.1**](https://huggingface.co/collections/meta-llama/llama-31) — 例如，`meta-llama/Llama-3.1-8B-Instruct`
- [**Llama 3.2**](https://huggingface.co/collections/meta-llama/llama-32) — 例如，`meta-llama/Llama-3.2-3B-Instruct`
- [**Qwen2.5**](https://huggingface.co/collections/Qwen/qwen25) — 例如，`Qwen/Qwen2.5-0.5B-Instruct`
- [**Qwen3**](https://huggingface.co/collections/Qwen/qwen3) — 例如，`Qwen/Qwen3-0.6B`
- [**Qwen3-VL**](https://huggingface.co/collections/Qwen/qwen3-vl) — 例如，`Qwen/Qwen3-VL-2B-Instruct`
- [**Qwen3.5**](https://huggingface.co/collections/Qwen/qwen35) — 例如，`Qwen/Qwen3.5-2B`
- [**Qwen3.6**](https://huggingface.co/collections/Qwen/qwen36) — 例如，`Qwen/Qwen3.6-35B-A3B`> [!提示]
> 不保证与所有法学硕士的兼容性。如果您认为应该支持某个模型，请随时在 GitHub 上提出问题 - 或者更好的是，提交包含所需更改的拉取请求。

### 环境集成

所有环境都插入同一个`environment_factory`插槽，因此它们可以在 TRL 级别互换 - 选择其生态系统适合您的任务的环境：

|整合|它是什么 |当 | 时使用它
|---|---|---|
| [OpenEnv](openenv) |开放环境标准（Gymnasium 风格的 API，通过 WebSocket 或容器化执行提供服务），由 Hugging Face 和社区支持。 |您正在使用 Hub 中现成的 OpenEnv 环境，或者根据开放标准定义您自己的环境（例如 Wordle、Sudoku、Catch）。 |
| [OpenReward](openreward) |与 ORS 环境集成（[openreward.ai](https://openreward.ai) 目录或您自己的 ORS 服务器）；任务**和**奖励通过 HTTP 提供。 |您想要针对 ORS 环境进行训练：目录（例如 `Eigent/SETA`）、您在自己的基础设施上自行托管的环境或您正在开发的本地服务器。 || [Harbor](harbor) |与Harbor任务套件的集成：每个任务都是一条指令，一个真实的沙箱图像（`docker`，`e2b`，...）和一个沙箱内验证器。 |您想要针对 Harbor 任务套件进行训练：一棵任务树，每个任务都有一个独立的沙箱和验证器（例如，一个数据分析代理，它探索沙箱中的文件并编写评分者检查的答案）。 |

## 视觉语言模型 (VLM) 训练

GRPO 支持在包含文本和图像的多模态数据集上训练视觉语言模型 (VLM)。

### 支持的型号

测试用：

- **Gemma3** — 例如，`google/gemma-3-4b-it`
- **LLaVA-NeXT** — 例如，`llava-hf/llava-v1.6-mistral-7b-hf`
- **Qwen2-VL** — 例如，`Qwen/Qwen2-VL-2B-Instruct`
- **Qwen2.5-VL** — 例如，`Qwen/Qwen2.5-VL-3B-Instruct`
- **SmolVLM2** — 例如，`HuggingFaceTB/SmolVLM2-2.2B-Instruct`
  
> [!提示]
> 不保证与所有 VLM 的兼容性。如果您认为应该支持某个模型，请随时在 GitHub 上提出问题 - 或者更好的是，提交包含所需更改的拉取请求。

### 快速入门

使用 [grpo\_vlm.py](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py) 微调 VLM。 [⟦T209⟧](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 训练命令示例：

```bash
accelerate launch \
  --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
  examples/scripts/grpo_vlm.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir grpo-Qwen2.5-VL-3B-Instruct \
  --learning_rate 1e-5 \
  --dtype bfloat16 \
  --max_completion_length 1024 \
  --use_vllm \
  --vllm_mode colocate \
  --use_peft \
  --lora_target_modules "q_proj", "v_proj" \
  --log_completions
```

### 配置提示- 在视觉语言投影层上使用 LoRA
- 启用 4 位量化以减少内存使用
- VLM 是内存密集型的 — 从较小的批量大小开始
- 大多数型号与 vLLM 兼容（`server` 和 `colocate` 模式）

### 数据集格式

每个训练样本应包括：

- `prompt`：通过处理器的聊天模板格式化的文本
- `image`/`images`：PIL 图像或 PIL 图像列表

训练器通过模型的图像处理器自动处理图像到张量的转换。

## GRPOTrainer[[trl.GRPOTrainer]]

- **型号**（`str`或[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)或`PeftModel`）--
  待训练的模型。可以是：- 一个字符串，是在 Huggingface.co 上的模型存储库中托管的预训练模型的 *模型 id*，或者
    包含使用保存的模型权重的*目录*的路径
    [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
    使用`<ModelArchitecture>.from_pretrained`（其中`<ModelArchitecture>`源自模型
    配置）与`args.model_init_kwargs`中的关键字参数。如果 `dtype` 未指定
    `args.model_init_kwargs`，默认为`float32`。这不同于
    [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)，其中（自 Transformers v5 起）推断 dtype
    从模型配置。
  - 一个[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)对象。仅支持因果语言模型。
  - 一个`PeftModel`对象。仅支持因果语言模型。
- **reward_funcs**（`RewardFunc | list[RewardFunc]`，*可选*）--
  用于计算奖励的奖励函数。为了计算奖励，我们将所有奖励称为
  具有提示和完成功能并总结奖励。提供奖励时可以省略
  通过 `environment_factory` 受到环境影响（见下文）。可以是：- 单一奖励函数，例如：
    - 字符串：huggingface.co 上模型存储库内托管的预训练模型的 *模型 ID*，或
    包含使用保存的模型权重的*目录*的路径
    [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
    使用 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification.from_pretrained) 与 `num_labels=1` 以及
    `args.model_init_kwargs` 中的关键字参数。
    - [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 对象：仅支持序列分类模型。
    - 自定义奖励功能：该功能提供提示和生成的完成，
      加上数据集中的任何其他列。它应该返回奖励列表。定制奖励
      函数可以是同步的也可以是异步的，并且当奖励是时也可以返回`None`
      不适用于这些样品。这对于多任务训练非常有用，其中不同的奖励
      函数适用于不同类型的样本。当奖励函数返回样本的`None`时，
      该奖励函数被排除在该样本的奖励计算之外。有关更多详细信息，请参阅
      [Using a custom reward
      function](#using-a-custom-reward-function)。训练者的状态也会传递给奖励函数。训练器的状态是一个实例
      [TrainerState](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerState) 可以通过访问 `trainer_state` 参数来访问
      奖励函数的签名。
  - 奖励函数列表，其中每个项目可以独立地是上述任何类型。混合不同
  列表中的类型（例如，字符串模型 ID 和自定义奖励函数）是允许的。
- **参数**（[GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig)，*可选*）--
  该训练器的配置。如果`None`，则使用默认配置。
- **train_dataset** （`Dataset` 或 `IterableDataset`，*可选*）--
  用于训练的数据集。它必须包含列 `"prompt"`。数据集中的任何其他列都是
  被忽略。样本的格式可以是：

  - [Standard](dataset_formats#standard)：每个样本都包含纯文本。
  - [Conversational](dataset_formats#conversational)：每个样本都包含结构化消息（例如，角色
    和内容）。

  仅当提供了 `environment_factory` 并且环境拥有（或程序上
  生成）数据，从其 `reset()` 方法返回提示。在这种情况下，必须设置`max_steps`
  定义训练长度。当`train_dataset`是`IterableDataset`（例如流数据集）时，`max_steps`必须是
  在训练参数中设置，因为无法推断其长度和训练步骤的总数
  需要限制训练循环并配置学习率调度程序。
- **eval_dataset**（`Dataset`、`IterableDataset`、`DatasetDict`、`IterableDatasetDict` 或 `dict[str, Dataset | IterableDataset]`）--
  用于评估的数据集。必须满足与`train_dataset`相同的要求。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理类用于处理数据。填充边必须设置为“左”。如果`None`，则
  处理类从带有[from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)的模型名称加载。一个
  必须设置填充令牌`tokenizer.pad_token`。如果处理类没有设置填充标记，
  `tokenizer.eos_token` 将用作默认值。
- **奖励处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)或`list[PreTrainedTokenizerBase]`，*可选*）--
  与`reward_funcs`中指定的奖励函数对应的处理类。可以是：- 单一处理类：当`reward_funcs`仅包含一个奖励函数时使用。
  - 处理类列表：必须与`reward_funcs`中奖励函数的顺序和长度相匹配。
  如果设置为`None`，或者如果对应于[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)的列表元素是
  `None`，模型的分词器自动加载使用
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)。对于`reward_funcs`中属于自定义奖励的元素
  函数（不是[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)），`reward_processing_classes`中的相应条目
  被忽略。
- **回调**（[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)列表，*可选*）--
  用于自定义训练循环的回调列表。将它们添加到详细的默认回调列表中
  在[here](https://huggingface.co/docs/transformers/main_classes/callback)。如果您想删除使用的默认回调之一，请使用 [remove_callback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer.remove_callback)
  方法。
- **优化器**（`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`，*可选*，默认为`(None, None)`）--
  包含要使用的优化器和调度器的元组。将默认为您的 `AdamW` 实例
  模型和由[get_linear_schedule_with_warmup](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup)给出的调度器，由`args`控制。
- **量化配置**（[BitsAndBytesConfig](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/quantization#transformers.BitsAndBytesConfig)，*可选*）--
  从模型标识符加载模型时使用的量化配置。与`peft_config`结合
  用于 QLoRA 培训。如果模型已实例化，则忽略。
- **peft_config**（`PeftConfig`，*可选*）--
  PEFT 配置用于包裹模型。如果`None`，则模型未包装。
- **工具**（`Callable`列表，*可选*）--
  模型在生成过程中可以调用的可调用工具函数（同步或异步）列表。每个工具
  应该是一个标准的 Python 函数，具有正确类型提示的参数和返回值，以及
  Google 风格的文档字符串描述其目的、参数和返回值。有关更多详细信息，请参阅：
  https://huggingface.co/docs/transformers/en/chat_extras#passing-tools。该模型使用函数的名称，类型提示和文档字符串来确定如何调用它。确保模特的聊天模板支持工具
  使用并且它已经针对工具调用进行了微调。
- **rollout_func** (`RolloutFunc`，*可选*) --
  用于生成补全的函数。它接收分配给当前的提示列表
  流程和训练器实例。它必须返回一个带有 `"prompt_ids"`、`"completion_ids"` 的字典，并且
  `"logprobs"`字段，并且可以选择返回`"logprob_token_ids"`（与`"logprobs"`形状相同）。任意
  其他字段被转发到奖励函数。该函数接收原始的每进程提示符片段
  没有重复；它负责返回每个提示的正确完成次数（请参阅
  `num_generations` / `num_generations_eval` 在训练器上）。此功能是实验性的，可能会更改或
  随时删除，恕不另行通知。
- **环境工厂**（`EnvironmentFactory`或`dict[str, EnvironmentFactory]`，*可选*）--
  创建并返回环境实例的可调用对象，或将环境名称映射到的字典
  这样的可调用对象。环境类应该定义可以在生成过程中作为工具调用的方法。
  每种方法应符合上述`tools`相同的要求。环境必须还实现了一个可调用的 `reset` 方法，可用于重置各代之间的状态。 `reset`
  方法应该返回 `None` 或字符串：当它返回字符串时，该字符串将附加到
  生成之前的最后一条用户消息。环境还可以定义一个`get_reward`方法，不需要
  参数并返回 `float`：当存在时，环境拥有奖励，并且 `get_reward` 被调用
  每次完成部署一次，根据环境的内部状态对其进行评分。它作为一个额外的
  奖励源（权重为 1，记录在环境的类名下）与 `reward_funcs` 一起，其中
  然后变成可选的。

  对于单个可调用对象，每个示例都使用相同的环境，每个部署都有一个实例，因此它们的
  交互保持隔离。对于字典，每个示例都必须带有一个 `environment` 字段来选择其
  按名称指定环境，并且只有该环境的工具会在其提示中公开 - 让单个运行混合
  任务（例如编码环境和游戏）。此功能是实验性的，可能会更改或删除
  任何时间，恕不另行通知。组相对策略优化 (GRPO) 方法的培训师。该算法最初是在
纸[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
Models](https://huggingface.co/papers/2402.03300)。

示例：

```python
>>> from trl import GRPOTrainer
>>> from trl.rewards import accuracy_reward
>>> from datasets import load_dataset

>>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

>>> trainer = GRPOTrainer(
...     model="Qwen/Qwen2.5-0.5B-Instruct",
...     reward_funcs=accuracy_reward,
...     train_dataset=dataset,
... )
>>> trainer.train()
```

- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于 `True`，加载 *args.output_dir* 中由前一个实例保存的最后一个检查点
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
  传递给 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的 🤗 模型中心。

## GRPOConfig[[trl.GRPOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int |无 = 无"}, {"name": "remove_unused_columns", "val": ": bool |无 = False"}, {"name": "label_names", "val": ": list[str] |无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": 字符串 |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "disable_dropout", "val": ": bool = False"}, {"name": "cast_lm_head_to_fp32", "val": ": bool = False"}, {"name": "num_ Generations", "val": ": int |无 = 8"}, {"name": "num_ Generations_eval", "val": ": int |无 = N一"}, {"name": "max_completion_length", "val": ": int |无 = 256"}，{"name"："ds3_gather_for_ Generation"，"val"："：bool = True"}，{"name"："shuffle_dataset"，"val"："：bool |无 = True"}, {"name": "pad_to_multiple_of", "val": ": int |无 = 无"}, {"name": " Generation_batch_size", "val": ": int |无 = 无"}, {"name": "steps_per_ Generation", "val": ": int |无 = 无"}, {"name": "温度", "val": ": float = 1.0"}, {"name": "top_p", "val": ": float = 1.0"}, {"name": "top_k", "val": ": int = 0"}, {"name": "min_p", "val": ": float |无 = 无"}, {"name": " Generation_kwargs", "val": ": dict |无 = 无"}, {"name": "chat_template_kwargs", "val": ": dict | None = None"}, {"name": "repetition_penalty", "val": ": float = 1.0"}, {"name": "cache_implementation", "val": ": str |无 = 无"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": "vllm_mode", "val": ": str = 'colocate'"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_enable_sleep_mode", “val”：“：bool = False”}，{“name”：“vllm_structed_outputs_regex”，“val”：“：str |无 = 无"}, {"name": "vllm_server_base_url", "val": ": str |无 = 无"}, {"name": "vllm_server_host", "val":": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8000"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_gpu_memory_utilization", "val": ": float = 0.3"}, {"name": "vllm_max_model_length", "val": ": int | None = None"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "beta", "val": ": float = 0.0"}, {"name": "num_iterations", "val": ": int = 1"}, {"name": "epsilon", "val": ": float = 0.2"}, {"name": "delta", "val": ": float | None = None"}, {"name": "epsilon_high", "val": ": None = None"}, {"name": "sapo_temp_neg", "val": ": float = 1.05"}, {"name": "sapo_Temperature_pos", "val": ": float = 1.0"}, {"name": "vespo_k_pos", "val": ": float = 2.0"}, {"name": "vespo_lambda_pos", "val": ": float = 3.0"}, {"name": "vespo_k_neg", "val": ": float = 3.0"}, {"name": "vespo_lambda_neg", "val": ": float = 2.0"}, {"name": "importance_sampling_level", "val": ": str = 'token'"}, {"name": "reward_weights", "val": ": list[float] | None = None"}, {"name": "multi_objective_aggregation", "val": ": str = 'sum_then_normalize'"}, {"name": "scale_rewards", "val": ": str = 'group'"}, {"name": "loss_type", "val": ": str = 'dapo'"}, {"name": "mask_truncated_completions", "val": ": bool = False"}, {"name": "sync_ref_model", "val": ": bool = False"}, {"name": "ref_model_mixup_alpha", "val": ": float = 0.6"}, {"name": "ref_model_sync_steps", "val": ": int = 512"}, {"name": "top_entropy_quantile", "val": ": float = 1.0"}, {"name": "entropy_coef", "val": ": float = 0.0"}, {"name": "use_adaptive_entropy", "val": ": bool = False"}, {"name": "entropy_coef_min", "val": ": float = 0.0"}, {"name": "entropy_coef_max", "val": ": float = 1.0"}, {"name": "entropy_coef_delta", "val": ": float = 0.005"}, {"name": "entropy_target", "val": ": float = 0.2"}, {"name": "max_tool_calling_iterations", "val": ": int |无 = 无"}, {"name": "vllm_importance_sampling_ Correction", "val": ": bool = True"}, {"name": "vllm_importance_sampling_mode", "val": ": str = 'sequence_mask'"}, {"name": "vllm_importance_sampling_clip_max", "val": ": float |无 = 3.0"}, {"name": "vllm_importance_sampling_clip_min", "val": ": float |无 = 无"}, {"name": "off_policy_mask_threshold", "val": ": float |无 = 无"}, {"name": "use_bias_ Correction_kl", "val": ": bool = False"}, {"name": "log_complitions", "val": ": bool = False"}, {"name": "log_multimodal", "val": ": bool = True"}, {"name": "num_completions_to_print", "val": ": int |无 = 无"}, {"name": "log_unique_prompts", "val": ": bool = False"}, {"name": "log_completions_hub_repo", "val": ": str |无 = 无"}, {"name": "use_transformers_continuous_batching", "val": ": bool = False"}, {"name": "transformers_continuous_batching_config", "val": ": dict |无 = 无"}, {"name": "use_transformers_paged", "val": ": bool = False"}, {"name": "vllm_importance_sampling_cap", "val": ": float |无 = 无"}]}>
控制模型和参考模型的参数- **model_init_kwargs** (`str`, `dict[str, Any]`, *可选*) --
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 的关键字参数，在 `model` 时使用
  [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 的参数以字符串形式提供。
- **trust_remote_code** (`bool`，*可选*，默认为`False`) --
  是否允许加载从 Hub 发送自定义 Python 代码的模型和标记器。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 和
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)。也适用于奖励模型和奖励标记器负载。
- **router_aux_loss_coef** (`float`，*可选*，默认为`0.001`) --
  负载平衡辅助损耗系数。仅在训练混合专家时有效
  （教育部）模型；对于其他模型，它不执行任何操作。辅助损失被添加到训练损失中
  重量。设置为 `0.0` 以禁用它。
- **disable_dropout**（`bool`，*可选*，默认为`False`）--
  是否在模型中禁用 dropout。这对于使用参考模型进行训练非常有用，因为它可以防止
  该模型针对同一输入生成不同的对数概率。
- **cast_lm_head_to_fp32**（`bool`，*可选*，默认为`False`）--
  是否将策略和参考模型的语言建模头转换为 float32。根据推荐[ScaleRL](https://huggingface.co/papers/2510.13786)食谱。仅当模型支持此标志
  具有解开的词嵌入和语言建模头层，即模型配置中的`tie_word_embeddings`
  是假的。

控制数据预处理的参数

- **remove_unused_columns** (`bool`，*可选*，默认为`False`) --
  是否仅保留数据集中`"prompt"`列。如果您使用自定义奖励函数
  需要除`"prompts"`和`"completions"`之外的任何列，您应该将其保留为`False`。
- **num_ Generations**（`int`，*可选*，默认为`8`）--
  每次提示采样的代数。有效批量大小 (num_processes * per_device_batch_size
  *gradient_accumulation_steps) 必须能被这个值整除。
- **num_ Generations_eval** （`int` 或 `None`，*可选*）--
  评估期间采样的代数。这允许在评估期间使用更少的代数
  节省计算。如果`None`，则使用`num_generations`的值。
- **max_completion_length**（`int`或`None`，*可选*，默认为`256`）--
  生成的补全的最大长度。
- **ds3_gather_for_ Generation**（`bool`，*可选*，默认为`True`）--此设置适用于 DeepSpeed ZeRO-3。如果启用，则会收集策略模型权重以进行生成，
  提高生成速度。但是，禁用此选项允许训练超出 VRAM 的模型
  单个 GPU 的容量，尽管代价是生成速度较慢。禁用此选项不兼容
  与 vLLM 一代。
- **shuffle_dataset**（`bool`，*可选*，默认为`True`）--
  是否打乱训练数据集。
- **pad_to_multiple_of** (`int`，*可选*) --
  如果设置，提示 id 和完成 id 将被填充为此值的倍数。

控制生成的参数- ** Generation_batch_size ** （`int`，*可选*）--
  用于生成的批量大小。如果`None`，则默认为有效训练批量大小：
  `per_device_train_batch_size * num_processes * steps_per_generation`。换句话说，有一个
  每个优化步骤都会生成批处理。与`steps_per_generation`互斥。
- **每代步数**（`int`，*可选*）--
  每代的步数。如果`None`，则默认为`gradient_accumulation_steps`。互斥
  与`generation_batch_size`。
- **温度**（`float`，默认为`1.0`）--
  取样温度。温度越高，完成的随机性越大。
- **top_p** (`float`，*可选*，默认为`1.0`) --
  控制要考虑的顶级令牌的累积概率的浮点数。必须在 (0, 1] 中。设置为
  `1.0` 考虑所有代币。
- **top_k** (`int`，*可选*，默认为`0`) --
  要保留用于 top-k 过滤的最高概率词汇标记的数量。如果`0`，top-k-filtering是
  禁用并考虑所有令牌。
- **min_p** (`float`, *可选*) --
  最小令牌概率，将按最可能令牌的概率进行缩放。它必须是一个
  值介于 `0.0` 和 `1.0` 之间。典型值在`0.01-0.2`范围内。- ** Generation_kwargs ** （`dict[str, Any]`，*可选*）--
  要传递给 [GenerationConfig](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/text_generation#transformers.GenerationConfig) 的其他关键字参数（如果使用变压器）或
  采样完成时的`SamplingParams`（如果使用 vLLM）。这可用于进一步定制
  生成行为，如设置`suppress_tokens`、`num_beams`等。如果包含冲突的key
  使用其他生成参数（如`min_p`、`top_p`等），它们将覆盖它们。
- **chat_template_kwargs** (`dict[str, Any]`，*可选*) --
  生成补全时传递给 `apply_chat_template` 函数的附加关键字参数。
- **重复惩罚**（`float`，*可选*，默认为`1.0`）--
  根据新标记是否出现在提示中以及到目前为止生成的文本中对新标记进行惩罚的浮动。
  值 > `1.0` 鼓励模型使用新代币，而值 < ⟦T379⟧ encourage the model to repeat
  tokens.
- **cache_implementation** (⟦T380⟧, *optional*) --
  Implementation of the cache method for faster generation when ⟦T381⟧ is set to ⟦T382⟧.

Parameters that control generation acceleration powered by vLLM

- **use_vllm** (⟦T383⟧, *optional*, defaults to ⟦T384⟧) --
  Whether to use vLLM for generating completions. If set to ⟦T385⟧, the trainer will use vLLM for generation
  instead of the default model.generate(). Requires ⟦T386⟧ to be installed.
- **vllm_mode** (⟦T387⟧, *optional*, defaults to ⟦T388⟧) --
  Mode to use for vLLM integration when ⟦T389⟧ is set to ⟦T390⟧. Must be one of ⟦T391⟧ or
  ⟦T392⟧.

  - ⟦T393⟧: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
    server is running (start with ⟦T394⟧).
  - ⟦T395⟧: vLLM will run in the same process and share the training GPUs. This avoids the need for a
    separate server but may cause resource contention with training.
- **vllm_model_impl** (⟦T396⟧, *optional*, defaults to ⟦T397⟧) --
  Model implementation to use for vLLM. Must be one of ⟦T398⟧ or ⟦T399⟧. ⟦T400⟧: Use
  the ⟦T401⟧ backend for model implementation. ⟦T402⟧: Use the ⟦T403⟧ library for model
  implementation.
- **vllm_structured_outputs_regex** (⟦T404⟧, *optional*) --
  Regex for vLLM structured outputs. If ⟦T405⟧ (default), structured outputs is disabled.

Parameters that control the vLLM server (only used when ⟦T406⟧ is ⟦T407⟧)

- **vllm_server_base_url** (⟦T408⟧, *optional*) --
  Base URL for the vLLM server (e.g., ⟦T409⟧). If provided, ⟦T410⟧ and
  ⟦T411⟧ are ignored.
- **vllm_server_host** (⟦T412⟧, *optional*, defaults to ⟦T413⟧) --
  Host of the vLLM server to connect to. Ignored if ⟦T414⟧ is provided.
- **vllm_server_port** (⟦T415⟧, *optional*, defaults to ⟦T416⟧) --
  Port of the vLLM server to connect to. Ignored if ⟦T417⟧ is provided.
- **vllm_server_timeout** (⟦T418⟧, *optional*, defaults to ⟦T419⟧) --
  Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
  timeout, a ⟦T420⟧ is raised.
- **vllm_group_port** (⟦T421⟧, *optional*, defaults to ⟦T422⟧) --
  Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
  is occupied, there is no need to change it.

Parameters that control colocated vLLM execution (only used when ⟦T423⟧ is ⟦T424⟧)

- **vllm_gpu_memory_utilization** (⟦T425⟧, *optional*, defaults to ⟦T426⟧) --
  Control the GPU memory utilization for vLLM. This setting only applies when ⟦T427⟧ is set to
  ⟦T428⟧. If you are using ⟦T429⟧, this parameter must be passed separately when
  launching the vLLM server via the ⟦T430⟧ flag.
- **vllm_max_model_length** (⟦T431⟧, *optional*) --
  Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus
  ⟦T432⟧; if omitted, it is inferred from the model config.
- **vllm_tensor_parallel_size** (⟦T433⟧, *optional*, defaults to ⟦T434⟧) --
  Control the tensor parallel size for vLLM. This setting only applies when ⟦T435⟧ is set to
  ⟦T436⟧. If you are using ⟦T437⟧, this parameter must be passed separately when
  launching the vLLM server via the ⟦T438⟧ flag.
- **vllm_enable_sleep_mode** (⟦T439⟧, *optional*, defaults to ⟦T440⟧) --
  Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
  waking the engine adds host–device transfer latency.

Parameters that control generation acceleration powered by transformers continuous batching

- **use_transformers_continuous_batching** (⟦T441⟧, *optional*, defaults to ⟦T442⟧) --
  Whether to use transformers' continuous batching engine for generating completions. Requires
  ⟦T443⟧.
- **transformers_continuous_batching_config** (⟦T444⟧, *optional*) --
  Keyword arguments for ⟦T445⟧.

Parameters that control the training

- **beta** (⟦T446⟧, *optional*, defaults to ⟦T447⟧) --
  KL coefficient. If ⟦T448⟧ (default), the reference model is not loaded, reducing memory usage and improving
  training speed. ⟦T745⟧ use a value of ⟦T449⟧.
- **num_iterations** (⟦T450⟧, *optional*, defaults to ⟦T451⟧) --
  Number of iterations per batch (denoted as μ in the algorithm).
- **epsilon** (⟦T452⟧, *optional*, defaults to ⟦T453⟧) --
  Epsilon value for clipping.
- **delta** (⟦T454⟧, *optional*) --
  Enables the upper clipping bound in two-sided GRPO loss when set to a float. If ⟦T455⟧ (default), standard
  GRPO clipping is used. Recommended to be greater than ⟦T456⟧ when enabled. This method is introduced in
  the ⟦T746⟧.
- **epsilon_high** (⟦T457⟧, *optional*) --
  Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
  specified in argument ⟦T458⟧. Paper ⟦T747⟧ recommends ⟦T459⟧.
  When used with ⟦T460⟧, this corresponds to the ε_max param specified in the ⟦T748⟧ and the recommended value is ⟦T461⟧.
- **sapo_temperature_neg** (⟦T462⟧, *optional*, defaults to ⟦T463⟧) --
  Temperature for tokens with non-positive advantage scores used in the ⟦T464⟧ loss function. This parameter
  is introduced in the ⟦T749⟧.
- **sapo_temperature_pos** (⟦T465⟧, *optional*, defaults to ⟦T466⟧) --
  Temperature for tokens with positive advantage scores used in the ⟦T467⟧ loss function. This parameter is
  introduced in the ⟦T750⟧.
- **vespo_k_pos** (⟦T468⟧, *optional*, defaults to ⟦T469⟧) --
  k parameter for positive advantages, it is the power exponent in the VESPO loss. Controls how aggressively
  we down-weight samples with low importance weights (when the importance sampling ratio < 1).
- **vespo_lambda_pos** (⟦T470⟧, *optional*, defaults to ⟦T471⟧) --
  lambda parameter for positive advantages, it is the decay factor in the VESPO loss. Controls how
  aggressively we down-weight samples with high importance weights (when the importance sampling ratio > 1)。
- **vespo_k_neg**（`float`，*可选*，默认为`3.0`）--
  k参数表示负优势，它是VESPO损失的幂指数。控制攻击强度
  我们降低重要性权重较低的样本的权重（当重要性采样率< 1).
- **vespo_lambda_neg** (⟦T474⟧, *optional*, defaults to ⟦T475⟧) --
  lambda parameter for negative advantages, it is the exponential decay factor in the VESPO loss. Controls
  how aggressively we down-weight samples with high importance weights (when the importance sampling ratio >
  1）。
- **重要性采样级别**（`str`，*可选*，默认为`"token"`）--控制是否在`"token"`或`"sequence"`级别计算重要性采样率。 `"token"`
  保留原始的每个令牌对数概率比（每个令牌一个权重）。 `"sequence"` 平均
  对有效标记之间的概率比进行对数，以生成每个序列的单个比率。 [GSPO
  paper](https://huggingface.co/papers/2507.18071) 表明序列级采样通常会产生更多结果
  稳定的训练并与序列级奖励更好地保持一致。
- **奖励权重**（`list[float]`，*可选*）--
  每个奖励函数的权重。必须与奖励函数的数量相匹配。如果`None`，所有奖励都是
  与重量 `1.0` 同等加权。
- **多目标聚合**（`str`，*可选*，默认为`"sum_then_normalize"`）--
  聚合多个奖励函数的方法。支持的值为：- `"sum_then_normalize"`（默认）：首先将每个奖励函数的加权奖励相加，然后应用
    `scale_rewards`指定的奖励缩放/标准化（有关详细信息，请参阅`scale_rewards`）。
  - `"normalize_then_sum"`：首先标准化/缩放跨代的每个奖励函数（在每个
    组），然后使用指定的权重对标准化奖励求和。则总奖励为
    形成优势时在批次级别标准化。这是论文中建议的方法
    [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL
    Optimization](https://huggingface.co/papers/2601.05242)。
- **scale_rewards**（`str`或`bool`，*可选*，默认为`"group"`）--
  指定奖励的扩展策略。支持的值为：

  - `True` 或 `"group"`（默认）：奖励按每组内的标准差缩放，确保
    组内的单位方差。
  - `"batch"`：奖励按照整个批次的标准差进行缩放，如
    [PPO Lite paper](https://huggingface.co/papers/2508.08221)。
  - `False` 或 `"none"`：不应用缩放。 [Dr. GRPO
    paper](https://huggingface.co/papers/2503.20783) 建议不要扩展奖励，因为扩展的依据是
    标准差引入了问题级别的难度偏差。
- **loss_type** (`str`，*可选*，默认为`"dapo"`) --
  指定要使用的损失公式。支持的值为：- `"grpo"`：通过对序列长度进行标准化来聚合令牌级别的损失。不推荐，因为
    长度偏差——这种方法倾向于更喜欢具有积极优势的较短完成和较长的完成
    具有负面优势。
  - `"dr_grpo"`：通过使用全局常量标准化来聚合代币级别的损失。这个方法是
    在[Dr. GRPO paper](https://huggingface.co/papers/2503.20783)中引入以消除长度偏差。
    常数的值对应于`max_completion_length`。
  - `"dapo"`（默认）：通过标准化中的活跃代币数量来聚合代币级别的损失
    全球累计批次。该方法在[DAPO
    paper](https://huggingface.co/papers/2503.14476)中引入，以消除长度偏差。
  - `"bnpo"`：通过本地活跃代币数量标准化来聚合代币级别的损失
    批次。请注意，标准化仅在本地批次上执行，因此结果可能略有不同
    尽管有效批量大小恒定，但仍取决于本地批量大小。使用时
    `per_device_train_batch_size==1`，损失相当于GRPO损失。
  - `"cispo"`：剪辑重要性采样权重，而不是优势缩放重要性权重。这然后将剪裁后的权重乘以优势和策略模型的对数概率。个人代币
    通过对全局累计批次中的活跃代币数量进行标准化来汇总损失。
    该方法是在[MiniMax-M1 paper](https://huggingface.co/papers/2506.13585)中引入的。
  - `"sapo"`：软自适应策略优化损失，如[Soft Adaptive Policy Optimization
    paper](https://huggingface.co/papers/2511.20347)中介绍。用平滑的、
    温度控制门，自适应地减弱离策略更新，同时保留有用的
    学习信号。
  - `"luspo"`：长度无偏序列策略优化损失。缩放每个序列的序列级损失
    序列的长度损失。这是 GSPO 的修改，需要
    `importance_sampling_level="sequence"`。在[LUSPO
    paper](https://huggingface.co/papers/2602.05261)中引入。
  - `"vespo"`：变分序列级软策略优化。用平滑的、
    非对称伽玛加权函数直接应用于序列级重要性权重。引入于
    [VESPO paper](https://huggingface.co/papers/2602.10693)。
- **mask_truncated_completions**（`bool`，*可选*，默认为`False`）--
  启用后，截断的完成将被排除在损失计算之外，从而防止它们被错误地惩罚并在训练期间引入噪音。根据
  [DAPO](https://huggingface.co/papers/2503.14476)论文，这是训练稳定性的一个很好的实践。
- **sync_ref_model**（`bool`，*可选*，默认为`False`）--
  是否每`ref_model_sync_steps`步同步参考模型与活动模型，使用
  `ref_model_mixup_alpha` 参数。这种同步源于
  [TR-DPO](https://huggingface.co/papers/2404.09656)纸。
- **ref_model_mixup_alpha** (`float`，*可选*，默认为`0.6`) --
  [TR-DPO](https://huggingface.co/papers/2404.09656) 论文中的 α 参数，用于控制混合
  更新期间当前策略和之前的参考策略之间的差异。参考政策是
  根据等式更新：`π_ref = α * π_θ + (1 - α) * π_ref_prev`。要使用此参数，您
  必须设置`sync_ref_model=True`。
- **ref_model_sync_steps**（`int`，*可选*，默认为`512`）--
  [TR-DPO](https://huggingface.co/papers/2404.09656) 论文中的 τ 参数，它决定了如何
  当前策略经常与参考策略同步。要使用此参数，您必须
  设置`sync_ref_model=True`。
- **top_entropy_quantile** (`float`，*可选*，默认为`1.0`) --
  来自 [Beyond the 80/20 Rule](https://huggingface.co/papers/2506.01939) 的 ρ 参数。保留在政策中
  损失项仅是每个序列的概率分布熵的标记的 top-ρ 分位数定位，提高业绩。范围：`[0.0-1.0]`。 `0.0` 的值会掩盖除最高熵标记之外的所有标记；
  `1.0` 保留所有令牌。论文推荐值为`0.2`。如果与使用
  `mask_truncated_completions=True`，仅考虑来自非截断完成的标记。
- **entropy_coef** (`float`，*可选*，默认为`0.0`) --
  损失中熵正则化项的系数。正值会增加熵奖励
  通过防止政策崩溃到接近确定性的输出来鼓励探索。奖金是
  无论`loss_type`如何，始终是每个令牌的平均熵；它不会重新调整以匹配损失类型
  政策正常化，因此`entropy_coef`对于每种损失类型都有相同的含义。当
  `use_adaptive_entropy=True`，它作为初始系数，并在每个优化器步骤中更新。
  设置为 `0.0`（默认）时无效。
- **use_adaptive_entropy**（`bool`，*可选*，默认为`False`）--
  是否使用自适应熵控制，介绍于
  [Skywork-OR1](https://huggingface.co/papers/2505.22312)。启用后，熵系数
  `entropy_coef` 会在每个优化器步骤中更新：当当前
  熵低于`entropy_target`，否则递减。该系数仅适用于熵等于或低于`entropy_target`。
- **entropy_coef_min** (`float`，*可选*，默认为`0.0`) --
  使用自适应熵控制时熵系数的下界。
- **entropy_coef_max** (`float`，*可选*，默认为`1.0`) --
  使用自适应熵控制时熵系数的上限。
- **entropy_coef_delta** (`float`，*可选*，默认为`0.005`) --
  在自适应熵控制期间调整每个优化器步骤的熵系数的步长。
- **entropy_target**（`float`，*可选*，默认为`0.2`）--
  自适应熵控制使用的目标平均每个令牌熵（以 nat 为单位）。系数仅为
  当当前熵降至或低于该值时应用。通过相同的令牌集进行测量
  策略损失：默认情况下所有完成令牌，或仅高熵子集
  `top_entropy_quantile < 1.0`。典型的语言模型的每个标记的熵在 2-10 范围内
  nats，因此`0.2`的默认值几乎不会触发正则化（仅在接近完全熵的情况下）
  崩溃）；将其设置为接近您在训练早期观察到的熵（记录为 `entropy`指标），因此奖金在政策崩溃之前参与（并在以下情况下考虑代币子集）
  使用`top_entropy_quantile`）。
- **max_tool_calling_iterations**（`int`，*可选*）--
  训练代理时工具调用的最大轮次。如果`None`，则没有限制和代数
  当模型在没有工具调用的情况下生成响应转弯或总响应长度达到
  `max_model_length`。
- **vllm_importance_sampling_ Correction**（`bool`，*可选*，默认为`True`）--
  是否应用重要性采样 (IS) 来纠正 vLLM 完成日志概率和 vLLM 完成日志概率之间的不匹配
  重新计算训练日志概率。如果设置为`False`，则不应用任何 IS，无论
  `vllm_importance_sampling_mode`。当`True`时，所选模式决定如何计算 IS 比率
  并受到约束。
- **vllm_importance_sampling_mode**（`str`，*可选*，默认为`"sequence_mask"`）--
  指定当`vllm_importance_sampling_correction=True`时如何执行重要性采样。可能
  值为：- `"token_truncate"`：令牌级截断 IS（默认）。每个代币的比率被剪裁为
  [C_最小值，C_最大值]。
  - `"token_mask"`：令牌级屏蔽IS。 [C_min, C_max] 之外的每个代币比率设置为零。
  - `"sequence_truncate"`：序列级截断 IS。单个序列比率被剪辑为
  [C_min, C_max] 并应用于序列中的所有标记。
  - `"sequence_mask"`：序列级屏蔽IS。比率超出 [C_min, C_max] 的序列被屏蔽
  出来。
- **vllm_importance_sampling_clip_max**（`float`，*可选*，默认为`3.0`）--
  `vllm_importance_sampling_mode` 使用的重要性采样上限 C_max。对于 `*_truncate` 模式，
  重要性比率在 C_max 处从上方剪裁。对于`*_mask`模式，大于C_max的比率设置为
  零。
- **vllm_importance_sampling_clip_min**（`float`，*可选*）--
  `vllm_importance_sampling_mode` 使用的重要性采样下限 C_min。对于 `*_truncate` 模式，
  比率在 C_min 处从下方剪裁。对于 `*_mask` 模式，低于 C_min 的比率设置为零。要严格
  掩模比率低于 C_min，无上限，设置 `vllm_importance_sampling_clip_max=None`。
- **off_policy_mask_threshold**（`float`，*可选*）--
  离策略序列屏蔽的阈值。如果`None`，则禁用离策略序列屏蔽。设置后，具有负面优势和高 KL 散度的序列被屏蔽以稳定训练。这个
  参数对应于[DeepSeek-V3.2
  paper](https://huggingface.co/papers/2512.02556)公式9中的`delta`阈值。它期望一个正值（例如 0.5）。
- **use_bias_ Correction_kl** （`bool`，*可选*，默认为`False`）--
  是否使用具有重要性采样校正的无偏 KL 散度估计器。这纠正了
  KL 散度通过乘以重要性采样率来估计。这在
  [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556)。

控制日志记录的参数- **log_completions**（`bool`，*可选*，默认为`False`）--
  是否每 `logging_steps` 步骤记录（提示、完成）对的样本。如果安装了`rich`，
  它打印样本。如果启用了 `wandb` 和/或 `trackio` 日志记录，则会将其记录到 `wandb` 和/或
  `trackio`。
- **log_multimodal**（`bool`，*可选*，默认为`True`）--
  是否将多模式内容（图像、视频等）与完成情况一起记录。禁用此功能可以减少
  使用高分辨率多模式数据时的日志大小。
- **num_completions_to_print** (`int`，*可选*) --
  使用 `rich` 打印的完成数。如果`None`，则记录所有完成情况。
- **log_unique_prompts**（`bool`，*可选*，默认为`False`）--
  是否记录独特的提示。如果`True`，则仅记录唯一的提示。如果`False`，所有提示都是
  已记录。
- **log_completions_hub_repo**（`str`，*可选*）--
  Hugging Face Hub 存储库以保存完成情况。应该是一个完整的存储库名称，例如
  `'username/reponame'` 或 `'orgname/reponame'`，或者只是 `'reponame'` 在这种情况下，存储库将是
  在当前登录的 Hugging Face 用户的命名空间中创建。请注意，该存储库将是公开的除非您设置 `hub_private_repo=True` 或您的组织默认创建私有存储库。”

已弃用的参数

- **use_transformers_paged** --

  

  参数 `use_transformers_paged` 已弃用，并将在 v2.0.0 版本中删除。使用
  `use_transformers_continuous_batching` 代替。

  

- **vllm_importance_sampling_cap** --

  

  参数 `vllm_importance_sampling_cap` 已弃用，并将在 v2.0.0 中删除。使用
  `vllm_importance_sampling_clip_max` 代替。

  

[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 的配置类。

此类仅包含特定于 GRPO 训练的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以把这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在
命令行。

> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`1e-6`，而不是`5e-5`。

### 异步GRPO
https://huggingface.co/docs/trl/v1.9.2/async_grpo_trainer.md