<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 论文索引

> [!提示]
> 下面的每篇论文都链接到其 Hugging Face 论文页面。您还可以使用 `hf` CLI 从终端读取其中任何内容，例如`hf papers read 2402.03300` — 对于编码代理来说特别方便。

## 群组相关策略优化

与[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer)相关的论文。

### DeepSeekMath：突破开放语言模型中数学推理的极限

**📜论文**：https://huggingface.co/papers/2402.03300

引入组相对策略优化 (GRPO)，并通过以数学为中心的预训练加上组相对 PPO 式优化显示出强大的数学推理收益。通过 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 在 TRL 中使用。

```python
from trl import GRPOConfig, GRPOTrainer

# The paper doesn't specify its hyperparameters, so here we provide hyperparameters from "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning" instead.
training_args = GRPOConfig(
    loss_type="grpo",
    beta=0.001,  # "the KL coefficient to 0.001"
    epsilon=10.0, # "the GRPO clip ratio ϵ to 10"
    num_generations=16,  # "For each question, we sample 16 outputs..."
    max_completion_length=32_768,  # "...with a maximum length of 32,768"
    steps_per_generation=16,  # "To accelerate training, each rollout generates 8,192 outputs, which are randomly split into 16 minibatches"
    # "resulting in a training batch size of 512". One way to achieve this setting with 1 device is per_device_train_batch_size=4, gradient_accumulation_steps=128
    per_device_train_batch_size=4,
    gradient_accumulation_steps=128,  
)
trainer = GRPOTrainer(
    ...,
    args=training_args,
)
```

### DeepSeek-R1：通过强化学习激励法学硕士的推理能力

**📜论文**：https://huggingface.co/papers/2501.12948DeepSeek-R1 通过多级管道实现了与 OpenAI-o1 相当的推理性能，该管道从纯粹的强化学习 (RL) 过渡到精致的、符合人类需求的模型。与它的前身 DeepSeek-R1-Zero 在基本模型上使用纯强化学习不同，R1 遵循结构化的四阶段演变：
1. 冷启动：基础模型根据一小组高质量、长思想链 (CoT) 数据进行微调，以提供稳定的起点。
2. 面向推理的强化学习：大规模强化学习用于提高数学、编码和逻辑的性能，使用基于规则的奖励和语言一致性奖励来减少语言混合。
3.拒绝采样和SFT：RL检查点通过拒绝采样生成60万个推理样本，与20万个非推理（一般）样本相结合，为第二轮监督微调创建新的数据集。
4. 适用于所有场景的强化学习：最后的强化学习阶段使模型与所有领域的人类偏好（有益和无害）保持一致，同时保持推理强度。

蒸馏：赋予小模型力量该论文的一个关键贡献是证明推理模式可以从大型模型（DeepSeek-R1）中提炼成更小的密集模型（例如 Qwen 和 Llama 系列）。人们发现，对于小型模型来说，蒸馏比从头开始使用纯强化学习进行训练更有效。

您可以使用 GRPOTrainer 来复制此管道的推理密集阶段。 
```python
from trl import GRPOConfig, GRPOTrainer

# Example configuration for a reasoning-oriented GRPO stage
# Based on the Open-R1 recipe for Qwen-7B
training_args = GRPOConfig(
    learning_rate=4.0e-5,
    max_completion_length=32768, # Support for long Chain-of-Thought
    num_generations=16,          # Sample 16 outputs per prompt for group relative advantage
    beta=0.001,                  # KL coefficient
    use_vllm=True,               # Use vLLM backend for accelerated rollout generation
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[accuracy_reward, format_reward], # R1-Zero used rule-based rewards
)

trainer.train()
```

### 组序列策略优化

**📜论文**：https://huggingface.co/papers/2507.18071

GSPO 是 GRPO 的变体，它在序列级别而不是每个令牌计算重要性采样权重。要重现纸张的设置，请使用以下配置：

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    beta=0.0,  # GSPO set KL regularization to zero: https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306 
    epsilon=3e-4,  # GSPO paper (v2), section 5.1
    epsilon_high=4e-4,  # GSPO paper (v2), section 5.1
    gradient_accumulation_steps=1,
    steps_per_generation=4,  # partition rollout batch into 4 mini-batches. GSPO paper (v2), section 5.1. Must be 4 times gradient_accumulation_steps
)
```

请注意，此方法仅在训练稍微偏离策略时才有效，例如，当`steps_per_generation > gradient_accumulation_steps`或`num_iterations > 1`时。否则，实际上相当于没有修改。

TRL 还提供了 GSPO 令牌的实验性实现，请参阅[Experimental - GSPO-Token](gspo_token)。

#### 政策比率：GRPO 与 GSPO

在 GSPO 中，策略比率是在序列级别定义的。换句话说，它是当前策略生成序列的概率与旧策略生成相同序列的概率之间的比率。

序列似然定义为：$$
算法 1 中的 \pi_\theta (o_i | q) = \prod_{t=1}^{|o_i|} \pi_\theta (o_{i,t} | q, o_{i, < t} ),
$$

where  \\( \pi_\theta \\) is the policy  \\( \pi \\) with parameters  \\(\theta\\),  \\( o_i \\) is the  \\( i \\)-th output sequence  \\( o \\) and  \\(o_{i,t}\\) is the  \\( t \\)-th token in this sequence,  \\( q \\) is the input query. The sequence likelihood ratio  \\( s_i (\theta) \\) is defined as:

$$
s_i (\theta) = \left(\frac{\pi_\theta (o_i | q)}{\pi_{\theta_{old}} (o_i | q)} \right)^{\frac{1}{|o_i|}}
$$

The exponent  \\( \frac{1}{|o_i|} \\) represents a sequence-length normalization, minimizing the influence of sequence length in sequence likelihood. In other terms, it computes the geometric mean of token probabilities, ensuring a fair comparison across sequences of varying lengths.

While GSPO defines the policy ratio at the sequence level, GRPO operates at the token level. Specifically, GRPO computes an importance ratio for each token in the sequence:

$$
w_{i,t}(\theta) = \frac{\pi_\theta (o_{i,t} | q, o_{i,< t})}{\pi_{\theta_{\text{old}}} (o_{i,t} | q, o_{i,< t})}
$$

This token-level ratio is then combined with a shared advantage  \\( \hat{A}_i \\), and the GRPO objective clips and optimizes each token independently across the sequence.

### Geometric-Mean Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2507.20673

Geometric-Mean Policy Optimization (GMPO) is a GRPO variant that maximizes the *geometric* mean of the token-level importance ratios instead of the arithmetic mean. The geometric mean is far less sensitive to outlier ratios, so the policy update is more stable and tolerates a much wider clipping range. Clipping is applied per token, in log space, and is one-sided per the advantage sign (the standard PPO trust region) — crucially, *before* the geometric mean is taken. This is what distinguishes GMPO from GSPO (⟦T81⟧), which clips the sequence-level ratio *after* averaging.

The GMPO objective replaces GRPO's per-token arithmetic mean with the geometric mean of the (clipped) token ratios:

$$
\mathcal{J}_{\text{GMPO}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \prod_{t=1}^{|o_i|} \min\left[ w_{i,t}(\theta)^{\operatorname{sgn}(\hat{A}_i)},\ \operatorname{clip}\left(w_{i,t}(\theta)^{\operatorname{sgn}(\hat{A}_i)}, \epsilon_1, \epsilon_2\right) \right]^{\operatorname{sgn}(\hat{A}_i)} \right)^{\frac{1}{|o_i|}} \hat{A}_i \right]
$$

where  \\( w_{i,t}(\theta) \\) is the per-token importance ratio. In practice the product and clipping are computed in log space for numerical stability, and the clip range  \\( (\epsilon_1, \epsilon_2) = (e^{-0.4}, e^{0.4}) \\) is markedly wider than GRPO/DAPO to encourage exploration.

TRL provides an experimental implementation, see ⟦T146⟧:

⟦T3⟧

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**📜 Paper**: https://huggingface.co/papers/2503.14476

The DAPO algorithm includes 5 key components:

- Overlong Filtering
- Clip-Higher
- Soft Overlong Punishment
- Token-level Loss
- Dynamic Sampling (⚠️ Not supported in TRL)

To reproduce the paper's setting, use this configuration:

⟦T4⟧

### Demystifying Long Chain-of-Thought Reasoning in LLMs

**📜 Paper**: https://huggingface.co/papers/2502.03373

This paper studies long chain-of-thought RL and introduces two complementary rule-based rewards:

- A **cosine length-scaled reward** (⟦T147⟧, Appendix C.1) that scales the correctness reward by completion length: a correct completion is rewarded more when it is shorter, while a wrong completion is penalized less when it is longer (preserving exploration).
- An **n-gram repetition penalty** (⟦T148⟧, Appendix C.2) that discourages the degenerate, repetitive completions that emerge as a reward-hacking strategy under length shaping.

The two are designed to be used together:

⟦T5⟧

### INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning

**📜 Paper**: https://huggingface.co/papers/2505.07291

INTELLECT-2 is the first globally distributed reinforcement learning training run of a 32 billion parameter language model using fully asynchronous RL across a dynamic, heterogeneous swarm of permissionless compute contributors. The authors propose modifications to the standard GRPO training recipe, including two-sided GRPO clipping for increased training stability. To reproduce the paper's setting, use this configuration:

⟦T6⟧

### Skywork-OR1: Open Reasoning Models

**📜 Paper**: https://huggingface.co/papers/2505.22312

Skywork-OR1 is a family of open reasoning models trained with GRPO. The paper introduces **adaptive entropy control**: an entropy regularization term ⟦T82⟧ is added to the GRPO objective, and the coefficient ⟦T83⟧ is automatically adjusted each optimizer step. When the model's mean per-token entropy falls at or below a target, ⟦T84⟧ is incremented to encourage more exploration; otherwise it is decremented. The bonus is only applied while entropy is at or below the target. To replicate this adaptive entropy control, use the following configuration:

⟦T7⟧

### Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

**📜 Paper**: https://huggingface.co/papers/2506.01939

A minority of tokens with high entropy act as reasoning "forks" in the CoT path, driving exploration and performance gains for RLVR, while low-entropy majority tokens contribute little or even impede learning. RLVR mainly adjusts high-entropy tokens, largely preserving the base model’s overall entropy patterns. Thus landing on the 80/20 rule, training on only 20% of the tokens with the highest entropy is comparable or supasses full-gradient updates for Qwen3 models.

The paper's main results use vanilla DAPO (⚠️ Dynamic Sampling is not supported in TRL). To replicate the main results, use the following configuration:

⟦T8⟧

### Dr. GRPO: Understanding R1-Zero-Like Training: A Critical Perspective

**📜 Paper**: https://huggingface.co/papers/2503.20783

A study of R1-Zero training identifies pretraining effects on RL performance and proffers Dr. GRPO to enhance token efficiency, achieving superior accuracy on AIME 2024. To reproduce the paper's setting, use this configuration:

⟦T9⟧

### Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)

**📜 Paper**: https://huggingface.co/papers/2508.08221

The authors of this paper find that the combination of:

1. scaling rewards by the standard deviation computed over the entire batch and
2. aggregating loss over the total number of tokens

can unlock the learning capability of critic-free policies using vanilla PPO loss. Their results demonstrate that this simple combination consistently improves performance, surpassing strategies like GRPO and ⟦T149⟧.

TRL supports using these learnings to train a GRPO model by:

⟦T10⟧

Note that when using gradient accumulation, the loss is aggregated over the total number of tokens in the batch, but not over the accumulated batch. For more details, see the ⟦T150⟧.

### Truncated Importance Sampling

**📰 Blog**: https://fengyao.notion.site/off-policy-rl

**📜 Paper**: https://huggingface.co/papers/1606.02647

Online policy learning methods commonly use an optimized inference framework for rollout generation (e.g vLLM) that is separate from the training backend. This introduces a rollout-training mismatch, exemplified in the following PPO objective:

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{inference}}}(\theta_{\mathrm{old}})}
\Bigl[
\min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}
$$

Despite  \\( \textcolor{red}{\pi_{\text{inference}}} \\) and  \\( \textcolor{blue}{\pi_{\text{training}}} \\) sharing the same model parameters  \\( \theta \\), they can produce significantly different token probabilities. This unexpected behavior implicitly breaks the on-policy assumption, and silently turns training off-policy.

Truncated Importance Sampling (TIS) addresses this issue by adapting the model update via importance-sampling correction. The gradient computation of the aforementioned PPO objective becomes

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{inference}}}(\theta_{\mathrm{old}})}
\Bigl[
\underbrace{\text{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\text{inference}}}(a, \theta_{\mathrm{old}})}, C_{\min}, C_{\max}\bigr)}_{\text{truncated importance ratio}} \cdot
\nabla_\theta
\min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}
$$

where  \\( C_{\min} \\) and  \\( C_{\max} \\) are hyper-parameters. TIS is implemented in GRPO, and is enabled by selecting a ⟦T85⟧ variant that includes the term ⟦T86⟧, such as ⟦T87⟧ or ⟦T88⟧.

⟦T11⟧

### Masked Importance Sampling

**📰 Blog**: https://ringtech.notion.site/icepop

**📜 Paper**: https://huggingface.co/papers/2510.18855

**📰 Blog**: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

Masked Importance Sampling (MIS) addresses the same issue as ⟦T151⟧ but replaces clipping with masking. MIS takes a more decisive stance by discarding updates whose discrepancy falls outside the range  \\( [C_{\min}, C_{\max}] \\).

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{inference}}}(\theta_{\mathrm{old}})}
\Bigl[
\underbrace{\mathbf{1}\left[
C_{\min} \le \frac{\pi_{\text{training}}(a, \theta_{\mathrm{old}})}
{\pi_{\text{inference}}(a, \theta_{\mathrm{old}})}
\le C_{\max}
\right]
\cdot
\frac{\pi_{\text{training}}(a, \theta_{\mathrm{old}})}
{\pi_{\text{inference}}(a, \theta_{\mathrm{old}})}}_{\text{masked importance ratio}} \cdot
\nabla_\theta
\min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}
$$

MIS is implemented for GRPO, and is enabled by selecting a ⟦T89⟧ variant that includes the term ⟦T90⟧, such as ⟦T91⟧ or ⟦T92⟧.

⟦T12⟧

### Sequence-level Importance Sampling

**📰 Blog**: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

The theoretically principled way to correct for the training-inference distribution shift is importance sampling, as introduced in the two papers above ⟦T152⟧ and ⟦T153⟧. However, the choice of formulation is crucial for keeping the gradient unbiased and ensuring stable training.

This work shows that sequence-level importance sampling is the sound approach for addressing the training–inference mismatch. Although token-level importance sampling achieves lower variance than a sequence-level ratio, it introduces bias and is therefore argued to be unsuitable for autoregressive models. The token-level gradient estimator is

$$
\mathbb{E}_{x\sim\mathcal{D},\, y\sim \pi^{\text{inference}}_\theta(\cdot|x)}
\Bigg[
  R(x,y)\,\cdot\,
  \sum_{t=0}^{|y|-1}
    \frac{\pi^{\text{training}}_\theta(y_t\,|\,x, y_{<t})}
         {\pi^{\text{inference}}_\theta(y_t\,|\,x, y_{<t})}
    \,\nabla_\theta \log \pi^{\text{training}}_\theta(y_t\,|\,x, y_{<t})
\Bigg]
$$
The correct, unbiased policy gradient estimator applies a single importance ratio over the entire generated sequence (trajectory)  \\( y \\), The Sequence-Level IS estimator looks like:

$$
\mathbb{E}_{x\sim\mathcal{D},\, y\sim \pi^{\text{inference}}_\theta(\cdot|x)}
\Bigg[
  \frac{\pi^{\text{training}}_\theta(y|x)}
       {\pi^{\text{inference}}_\theta(y|x)}
  \, R(x,y)\,
  \nabla_\theta \log \pi^{\text{training}}_\theta(y|x)
\Bigg]
$$

TRL exposes the Importance Sampling granularity level through the ⟦T93⟧ configuration parameter where ⟦T94⟧ modes implement a sequence-level importance sampling ratio and ⟦T95⟧ a per-token ratio.

### The Art of Scaling Reinforcement Learning

**📜 Paper**: https://huggingface.co/papers/2510.13786

A systematic study that defines a framework for analyzing and predicting reinforcement learning scaling in large language models, identifies key design choices that affect compute efficiency and propose a best-practice recipe called ScaleRL.

You can partially reproduce the ScaleRL recipe using the ⟦T154⟧ with the following configs:

⟦T13⟧

### It Takes Two: Your GRPO Is Secretly DPO

**📜 Paper**: https://huggingface.co/papers/2510.00977

Shows that GRPO's effectiveness stems from an implicit contrastive objective rather than accurate advantage estimation via large group sizes. This establishes a formal connection between GRPO and DPO, where group size only affects Monte Carlo estimators of the contrastive objective. The authors introduce 2-GRPO — using just two rollouts — which matches the performance of 16-GRPO at significantly lower training cost. Used in TRL via ⟦T155⟧ with ⟦T96⟧. To reproduce the paper's setting, use this configuration:

⟦T14⟧

### Soft Adaptive Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2511.20347

Soft Adaptive Policy Optimization (SAPO), replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO.

To reproduce the paper's setting, use this configuration:

⟦T15⟧

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

**📜 Paper**: https://huggingface.co/papers/2512.02556

DeepSeek-V3.2 technical report introduces several techniques to enhance the performance of GRPO. In TRL we implement:

- The **Unbiased KL Estimate**, which corrects the K3 estimator (as used in the original GRPO implementation) to obtain an unbiased KL estimate using the importance-sampling
ratio between the current policy  \\( \pi_\theta \\) and the behavior policy  \\( \pi_{\text{old}} \\).

$$
\mathrm{D}_{\mathrm{KL}}\!\left(\pi_\theta(o_{i,t}) \,\|\, \pi_{\text{ref}}(o_{i,t})\right) =
\textcolor{red}{\frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}\mid q, o_{i,<t})}}
\left(
  \frac{\pi_{\text{ref}}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
  -
  \log \frac{\pi_{\text{ref}}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
  - 1
\right).
$$

To enable this feature, set the ⟦T97⟧ parameter to ⟦T98⟧ in the ⟦T156⟧, and ⟦T99⟧:

⟦T16⟧

- The **Off-Policy Masking**, which stabilizes training by ignoring sequences where the policy performs poorly (negative advantage) **and** has drifted significantly from the old policy (high KL divergence).

The off-policy binary mask  \\(\textcolor{red}{M_{i,t}}\\) is defined as:

$$
\textcolor{red}{M_{i,t}} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i, \textcolor{blue}{\delta} \\
1 & \text{otherwise}
\end{cases}
$$

This mask is then applied to the GRPO loss as follows:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min \left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})} \hat{A}_{i,t}, \, \text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) \textcolor{red}{M_{i,t}} - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]
$$

To enable this feature, use the ⟦T100⟧ (corresponding to  \\( \textcolor{blue}{\delta} \\)) in the ⟦T157⟧:

⟦T17⟧

While the paper doesn't specify a  \\( \textcolor{blue}{\delta} \\) value used, a good starting point could be  \\( \textcolor{blue}{\delta} = 0.5 \\). If training seems too conservative or too many sequences are masked, you can increase the value.
For reference,  \\( \textcolor{blue}{\delta} = 1.0 \\) corresponds to an average log-ratio divergence of 1 nat per token, i.e. on sequences where this threshold is exceeded, the old policy was on average  \\( e^1 \approx 2.7 \\) times more likely to generate these tokens than the current policy.

### GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

**📜 Paper**: https://huggingface.co/papers/2601.05242

GDPO is a reinforcement learning optimization method designed for multi-reward training. While existing approaches commonly apply Group Relative Policy Optimization (GRPO) in multi-reward settings, the authors show that this leads to reward advantages collapse, reducing training signal resolution and causing unstable or failed convergence. GDPO resolves this issue by decoupling reward normalization across individual rewards, preserving their relative differences and enabling more faithful preference optimization. To enable GDPO for multi-reward RL training, simply set:

For a group of  \\( N \\) rewards and  \\( G \\) samples per group, GDPO normalizes each reward independently:

$$
A_n^{(i,j)} = \frac{r_n^{(i,j)} - \text{mean}\{r_n^{(i,1)}, \ldots, r_n^{(i,G)}\}}{\text{std}\{r_n^{(i,1)}, \ldots, r_n^{(i,G)}\} + \epsilon}
$$

The normalized group advantage is then aggregated across rewards:

$$
A^{(i,j)} = \sum_{n=1}^{N} w_n A_n^{(i,j)}
$$

The final per-batch normalization produces:

$$
\hat{A}^{(i,j)} = \frac{A^{(i,j)} - \text{mean}_{i',j'}\{A^{(i',j')}\}}{\text{std}_{i',j'}\{A^{(i',j')}\} + \epsilon}
$$

Here,  \\( \text{mean}_{i',j'}\{A^{(i',j')}\} \\) and  \\( \text{std}_{i',j'}\{A^{(i',j')}\} \\) denote statistics over all groups in the batch.

⟦T18⟧

Note that this method only has an effect when training involve more than one reward function.

The authors provide a easy-to-use, slurm-free training example that enable the community to quickly validate GDPO’s effectiveness over GRPO, see ⟦T158⟧.

### Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR

**📜 Paper**: https://huggingface.co/papers/2602.05261

Length-Unbiased Sequence Policy Optimization (LUSPO) modifies GSPO by scaling each sequence's loss by its length. This corrects GSPO's gradient bias that penalizes longer responses. To reproduce the paper's setting, use this configuration:

⟦T19⟧

### VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training

**📜 Paper**: https://huggingface.co/papers/2602.10693

VESPO addresses training instability in off-policy RL caused by policy staleness, asynchronous updates, and train-inference mismatches. Rather than relying on heuristic token-level clipping (GRPO) or sequence-length normalization (GSPO), VESPO derives a principled reshaping kernel from a variational framework. In practice, this yields a smooth, asymmetric Gamma weighting function that gracefully suppresses extreme sequence-level importance weights without introducing length bias.

$$
\mathcal{L}_{\text{VESPO}}(\theta) = - \mathbb{E}_{\tau \sim \mu} \left[ \underbrace{W(\tau)^{k} \cdot \exp\left(\lambda
(1 - W(\tau))\right)}_{\phi(W) \text{ detached }} \cdot \mathcal{A}(\tau) \cdot \log \pi_\theta(\tau) \right]
$$

with  \\( W(\tau) = \frac{\pi_\theta(\tau)}{\mu(\tau)} \\) the sequence level importance ratio, and  \\( \phi(W) \\) is detached from the computation graph to serve as a gradient scaling coefficient.

⟦T20⟧

## Optimal Advantage Regression

Papers relating to the ⟦T159⟧.

### Accelerating RL for LLM Reasoning with Optimal Advantage Regression

**📜 Paper**: https://huggingface.co/papers/2505.20686

A\*-PO (Optimal Advantage Regression) is a two-stage RL method for LLM reasoning that avoids both an online critic and multi-sample group rollouts. Stage 1 estimates the optimal value ⟦T101⟧ offline from reference-policy samples; Stage 2 performs on-policy updates with a single generation per prompt using a squared-error regression loss ⟦T102⟧. This yields faster training and lower peak memory than PPO/GRPO/REBEL. The method assumes a binary verifiable reward and cannot exceed the reference policy's Pass@K. See ⟦T160⟧. The official code can be found in ⟦T161⟧.

⟦T21⟧

## Direct Policy Optimization

Papers relating to the ⟦T162⟧

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**📜 Paper**: https://huggingface.co/papers/2305.18290

Direct Preference Optimization (DPO) fine-tunes language models more efficiently and with better performance compared to reinforcement learning from human feedback (RLHF), by directly optimizing policy training based on human preferences. To reproduce the paper's setting, use this configuration:

⟦T22⟧

### SLiC-HF: Sequence Likelihood Calibration with Human Feedback

**📜 Paper**: https://huggingface.co/papers/2305.10425

Sequence Likelihood Calibration (SLiC) is shown to be an effective and simpler alternative to Reinforcement Learning from Human Feedback (RLHF) for learning from human preferences in language models. To reproduce the paper's setting, use this configuration:

⟦T23⟧

These parameters only appear in the ⟦T163⟧

### Statistical Rejection Sampling Improves Preference Optimization

**📜 Paper**: https://huggingface.co/papers/2309.06657

Proposes **RSO**, selecting stronger preference pairs via statistical rejection sampling to boost offline preference optimization; complements DPO/SLiC. They also introduce a new loss defined as:

$$
\mathcal{L}_{\text{hinge-norm}}(\pi_\theta)
= \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
\left[
\max\left(0,\; 1 - \left[\gamma \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \gamma \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right]\right)
\right]
$$

To train with RSO-filtered data and the hinge-norm loss, you can use the following code:

⟦T24⟧

### Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints

**📜 Paper**: https://huggingface.co/papers/2309.16240

Proposes  \(( f \\)-DPO, extending DPO by replacing the usual reverse-KL regularizer with a general \(( f \\)-divergence, letting you trade off mode-seeking vs mass-covering behavior (e.g. forward KL, JS,  \(( \alpha \\)-divergences). The only change is replacing the DPO log-ratio margin with an **f′ score**:

$$
\mathcal{L}_{f\text{-DPO}}(\pi_\theta)
= \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
\left[
-\log \sigma\left(
\beta \textcolor{red}{f'}\textcolor{red}{\Big(}\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}\textcolor{red}{\Big)}
-
\beta \textcolor{red}{f'}\textcolor{red}{\Big(}\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\textcolor{red}{\Big)}
\right)
\right]
$$

Where  \\( f' \\) is the derivative of the convex function defining the chosen  \(( f \\)-divergence.

To reproduce:

⟦T25⟧

### A General Theoretical Paradigm to Understand Learning from Human Preferences

**📜 Paper**: https://huggingface.co/papers/2310.12036

Learning from human preferences can be written as a single KL-regularized objective over pairwise preference probabilities,

$$
\max_\pi ;\mathbb{E}\big[\Psi\left(p^*(y \succ y' \mid x)\right)\big] - \tau\mathrm{KL}(\pi||\pi_{\text{ref}}),
$$

which reveals RLHF and DPO as special cases corresponding to the logit choice of  \\( \Psi \\).
The paper shows that this logit transform amplifies near-deterministic preferences and effectively weakens KL regularization, explaining overfitting.
Using the **Identity transform (IPO)** avoids this pathology by optimizing preferences directly, without assuming a Bradley–Terry reward model.
To reproduce the paper's setting, use this configuration:

⟦T26⟧

These parameters only appear in the ⟦T164⟧

### Towards Efficient and Exact Optimization of Language Model Alignment

**📜 Paper**: https://huggingface.co/papers/2402.00856

The paper shows that direct preference methods like DPO optimize the wrong KL direction, leading to blurred preference capture, and proposes EXO as an efficient way to exactly optimize the human‑preference alignment objective by leveraging reverse KL probability matching rather than forward KL approximations. To reproduce the paper's setting, use this configuration:

⟦T27⟧

### Noise Contrastive Alignment of Language Models with Explicit Rewards

**📜 Paper**: https://huggingface.co/papers/2402.05369

The paper reframes language-model alignment as a *noise-contrastive classification* problem, proposing InfoNCA to learn a policy from explicit rewards (or preferences) by matching a reward-induced target distribution over responses, and showing DPO is a special binary case. It then introduces NCA, which adds an absolute likelihood term to prevent the likelihood collapse seen in purely relative (contrastive) objectives.

With pairwise preferences, treat the chosen/rejected \\( K=2 \\), define scores \\( r=\beta(\log\pi_\theta-\log\pi_{\text{ref}}) \\), and apply the NCA preference loss \\( -\log\sigma(r_w)-\tfrac12\log\sigma(-r_w)-\tfrac12\log\sigma(-r_l) \\).

To reproduce the paper's setting, use this configuration:

⟦T28⟧

### Provably Robust DPO: Aligning Language Models with Noisy Feedback

**📜 Paper**: https://huggingface.co/papers/2403.00409

DPO breaks under noisy human preferences because label flips bias the objective. Robust DPO fixes this by analytically debiasing the DPO loss under a simple noise model, with provable guarantees.

$$
\mathcal{L}_{\text{robust}}(\pi_\theta) = \frac{(1-\varepsilon)\mathcal{L}_{\text{DPO}}(y_w, y_l) - \varepsilon\mathcal{L}_{\text{DPO}}(y_l, y_w)}
{1-2\varepsilon}
$$

Where  \\( \mathcal{L}_{\text{DPO}} \\) is the DPO loss defined in ⟦T165⟧ and  \\( \varepsilon \\) is the probability of a label flip.

This single correction turns noisy preference data into an unbiased estimator of the clean DPO objective.

⟦T29⟧

### Binary Classifier Optimization for Large Language Model Alignment

**📜 Paper**: https://huggingface.co/papers/2404.04656

Theoretical analysis and a new algorithm, Binary Classifier Optimization, explain and enhance the alignment of large language models using binary feedback signals. To reproduce the paper's setting, use this configuration:

BCO reframes language-model alignment as behavioral cloning from an optimal reward-weighted distribution, yielding simple supervised objectives that avoid RL while remaining theoretically grounded.
It supports both unpaired reward data and pairwise preference data, with a reward-shift–invariant formulation that reduces to a DPO-style loss in the preference setting.

For the pairwise preference setting, the BCO loss is defined as:

$$
\mathcal{L}_{\text{bco\_pair}}(\pi_\theta) =
\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
\left[
-\log \sigma\Big(
\beta[(\log\pi_\theta-\log\pi_{\text{ref}})(y_w)
-
(\log\pi_\theta-\log\pi_{\text{ref}})(y_l)]
\Big)
\right]
$$

To reproduce the paper in this setting, use this configuration:

⟦T30⟧

For the unpaired version, the user should utilize ⟦T166⟧ and ⟦T167⟧.

### Learn Your Reference Model for Real Good Alignment

**📜 Paper**: https://huggingface.co/papers/2404.09656

Trust Region DPO (TR-DPO) updates the reference policy during training, demonstrating effectiveness against DPO on the Anthropic HH and TLDR datasets, outperforming DPO by up to 19% measured by automatic evaluation with GPT-4, improving coherence, correctness, level of detail, helpfulness, and harmlessness. To reproduce the paper's setting, use this configuration:

⟦T31⟧

### Iterative Reasoning Preference Optimization

**📜 Paper**: https://huggingface.co/papers/2404.19733

Iterative RPO improves reasoning by repeatedly generating chain-of-thought candidates, building preference pairs from correct vs. incorrect answers, and training with a DPO + NLL objective. The extra NLL term is key for learning to actually generate winning traces.

TRL can express the DPO + NLL objective by mixing ⟦T103⟧ (DPO) with ⟦T104⟧ (NLL):

⟦T32⟧

Note that the paper uses an iterative loop: each iteration regenerates CoT candidates with the current model, then retrains on fresh preference pairs. TRL does not automate that loop for you.

### Self-Play Preference Optimization for Language Model Alignment

**📜 Paper**: https://huggingface.co/papers/2405.00675

A self-play method called SPPO for language model alignment achieves state-of-the-art performance by approximating Nash equilibrium policy in a constant-sum game setting, outperforming other approaches with limited data. To reproduce the paper's setting, use this configuration:

⟦T33⟧

### Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer

**📜 Paper**: https://huggingface.co/papers/2405.16436

Regularized Preference Optimization (RPO) mitigates overoptimization in RLHF by fusing the DPO loss with the SFT loss, provably preventing the policy from choosing actions with spurious high proxy rewards. To reproduce the paper's setting, use this configuration:

⟦T34⟧

### Distributional Preference Alignment of LLMs via Optimal Transport

**📜 Paper**: https://huggingface.co/papers/2406.05882

Alignment via Optimal Transport (AOT) aligns large language models distributionally by penalizing violations of stochastic dominance between positive and negative sample distributions, achieving state-of-the-art performance on alignment benchmarks. To reproduce the paper's setting, use this configuration:

⟦T35⟧

or, for the unpaired version:

⟦T36⟧

There is no additional hyperparameter in the paper.

### Discovering Preference Optimization Algorithms with and for Large Language Models

**📜 Paper**: https://huggingface.co/papers/2406.08414

An LLM-driven method automatically discovers performant preference optimization algorithms, leading to a new algorithm called DiscoPOP that blends logistic and exponential losses. To reproduce the paper's setting, use this configuration:

⟦T37⟧

### WPO: Enhancing RLHF with Weighted Preference Optimization

**📜 Paper**: https://huggingface.co/papers/2406.11827

WPO reweights preference pairs by their policy probabilities to reduce the off-policy gap in DPO-style training. The loss is:

$$
\mathcal{L}_{\text{WPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \textcolor{red}{w(x, y_w) w(x, y_l)} \log p(y_w \succ y_l \mid x) \right]
$$

where the weight  \\( w(x, y) \\) is defined as:

$$
w(x, y) = \exp\left(\frac{1}{|y|}\sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\sum_{v \in \mathcal{V}} \pi_\theta(v \mid x, y_{<t})^2}\right)
$$

To reproduce the paper's setting, use this configuration:

⟦T38⟧

### Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment

**📜 Paper**: https://huggingface.co/papers/2408.06266

CLAIR and APO enhance LLM alignment through more contrastive preference pairs and controlled alignment objectives, improving model performance close to GPT4-turbo. To reproduce the paper's setting, use this configuration:

⟦T39⟧

⟦T40⟧

These parameters only appear in the ⟦T168⟧

### Length Desensitization in Direct Preference Optimization

**📜 Paper**: https://huggingface.co/papers/2409.06411

Shows that standard DPO is inherently length-sensitive, which often pushes models toward overly long or verbose generations. The paper proposes LD-DPO, which modifies the sequence log-prob aggregation by splitting the longer response into a shared prefix (up to the shorter response length) and an excess tail, then downweighting the tail with a factor  \\( \alpha \in [0,1] \\):

$$
\log \pi_\theta(y_{\text{long}}|x) = \log \pi_\theta(y_{1:l_p}|x) + \alpha \cdot \log \pi_\theta(y_{l_p+1:l}|x, y_{1:l_p}),
\quad
l_p=\min(|y_w|,|y_l|).
$$

Setting  \\( \alpha=1 \\) recovers standard  \\( \alpha \\) reduces verbosity while preserving preference quality.
The optimal  \\( \alpha \\) depends on the model family and whether you’re training a base vs. instruct model, but the paper suggests  \\( \alpha=0.5 \\) as a strong default starting point.

⟦T41⟧

### Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization

**📜 Paper**: https://huggingface.co/papers/2411.10442

Introduces Mixed Preference Optimization (MPO) to improve multimodal reasoning in MLLMs, addressing distribution shift and weak Chain-of-Thought (CoT) after standard pre-training and SFT. The paper contributes (1) MMPR, an automated pipeline for high-quality multimodal preference data, and (2) MPO, a combined preference objective (pairwise + BCO-style + SFT) that boosts CoT. InternVL2-8B-MPO reaches 67.0 on MathVista (+8.7 over InternVL2-8B), comparable to the 10× larger InternVL2-76B. Used in TRL via ⟦T169⟧ with composite loss. To reproduce the paper's setting, use this configuration:

⟦T42⟧

### TÜLU 3: Pushing Frontiers in Open Language Model Post-Training

**📜 Paper**: https://huggingface.co/papers/2411.15124

The length-normalized sigmoid loss addresses length bias in DPO by dividing chosen and rejected log-ratio scores by their respective completion lengths before computing the Bradley-Terry loss. This per-token normalization was introduced in ⟦T170⟧ (Appendix I "DPO w/ LN" ablation) as an average log-probability reward for a reference-free setting, and was later adopted for standard reference-model-based DPO in post-training recipes such as ⟦T171⟧ (Section 4.3). The loss is:

$$
\mathcal{L}_{\text{sigmoid\_norm}} = -\log\sigma\!\left(\beta \left({\color{red}\frac{1}{|y_w|}}\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - {\color{red}\frac{1}{|y_l|}}\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right),
$$
which can be set with:

⟦T43⟧

## Kahneman–Tversky Optimization

Papers relating to the ⟦T172⟧

### KTO: Model Alignment as Prospect Theoretic Optimization

**📜 Paper**: https://huggingface.co/papers/2402.01306

KTO derives an alignment objective from prospect theory and learns directly from **binary** human feedback (liked/disliked), matching or surpassing DPO-style methods while handling imbalanced/noisy signals well.
To reproduce the paper's setting, you can use the default configuration of ⟦T173⟧:

⟦T44⟧

## Supervised Fine-Tuning

Papers relating to the ⟦T174⟧

### EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes

**📜 Paper**: https://huggingface.co/papers/2508.00180

Bias-Corrected Exponential Moving Average (BEMA) improves the stability and efficiency of language model fine-tuning by reducing stochasticity and eliminating bias. To use BEMA with SFT as described in the paper, you can use the ⟦T175⟧:

⟦T45⟧

### On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**📜 Paper**: https://huggingface.co/papers/2508.05629

Dynamic Fine-Tuning (DFT) improves the generalization of Large Language Models (LLMs) by dynamically rescaling gradients, outperforming standard Supervised Fine-Tuning (SFT) and showing competitive results in offline reinforcement learning.

$$
\mathcal{L}_{\text{DFT}}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ - \sum_{t=1}^{|y|} \textcolor{red}{\text{sg}\big(\pi_\theta(y_t \mid y_{<t}, x)\big)} \; \log \pi_\theta(y_t \mid y_{<t}, x) \right]
$$

where  \\( \text{sg}(\cdot) \\) is the stop-gradient operator. To use DFT with SFT as described in the paper, you can use the ⟦T105⟧ argument:

⟦T46⟧

To closely match the paper’s setup, you can use the following configuration (see Sec. 4.1). Authors also mention that the hyperparameters are not very sensitive (Sec. 4.3):

⟦T47⟧

### Fewer Truncations Improve Language Modeling

**📜 Paper**: https://huggingface.co/papers/2404.10830

The paper shows that the standard concatenate-then-split preprocessing (⟦T106⟧) used for LLM training causes many documents to be arbitrarily truncated, which harms learning. It proposes packing document chunks into context windows using a Best-Fit Decreasing bin-packing algorithm, greatly reducing truncation while keeping high token utilization and improving model performance. TRL implements this as the ⟦T107⟧ packing strategy in ⟦T176⟧. For more details on packing, see the ⟦T177⟧.

⟦T48⟧

### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

**📜 Paper**: https://huggingface.co/papers/1910.10683

The T5 paper proposes a unified text-to-text framework for transfer learning and introduces **sequence packing** (Section 3.5.2): grouping multiple short sequences into fixed-length blocks to reduce padding and improve training efficiency. Packing is supported in TRL via ⟦T178⟧ with the ⟦T179⟧. To enable packing with TRL, use this configuration:

⟦T49⟧

## Parameter-Efficient Fine-Tuning (PEFT)

For general details on using PEFT with TRL, please refer to the ⟦T180⟧ guide.

### LoRA: Low-Rank Adaptation of Large Language Models

**📜 Paper**: https://huggingface.co/papers/2106.09685

Low-Rank Adaptation (LoRA) reduces the number of trainable parameters and GPU memory usage in large-scale pre-trained models while maintaining or improving performance on downstream tasks. TRL integrates LoRA via the ⟦T181⟧ and can be easily enabled in any TRL trainer by passing a ⟦T108⟧ to the ⟦T109⟧ argument. Here is an example of using LoRA with the ⟦T182⟧:

⟦T50⟧

### DoRA: Weight-Decomposed Low-Rank Adaptation

**📜 Paper**: https://huggingface.co/papers/2402.09353

Weight-Decomposed Low-Rank Adaptation (DoRA) can improve the performance of LoRA, especially at low ranks. DoRA decomposes pre-trained weight into two component: magnitude and direction. Direction is handled by normal LoRA, and magnitude is learnable parameters. TRL integrate DoRA via the ⟦T183⟧ and can be easily enable through setting ⟦T110⟧ to the ⟦T111⟧.

⟦T51⟧

## Reinforce Leave-One-Out

Papers relating to the ⟦T184⟧

### Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs

**📜 Paper**: https://huggingface.co/papers/2402.14740

RLOO is a variant of REINFORCE that reduces variance by using leave-one-out baselines. It computes rewards by comparing each sample against the average of all other samples in the batch, providing more stable gradients than standard REINFORCE. To reproduce the paper's setting, use this configuration:

⟦T52⟧

### REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models

**📜 Paper**: https://huggingface.co/papers/2501.03262

REINFORCE++ is an enhanced variant of the classical REINFORCE algorithm that incorporates key optimization techniques from PPO while eliminating the need for a critic network. It achieves simplicity, enhanced training stability, and reduced computational overhead through global advantage normalization across the entire batch. To approximate the paper's setting with the ⟦T185⟧, use this configuration:

⟦T53⟧

## Odds Ratio Preference Optimization

Papers relating to the ⟦T186⟧

### ORPO: Monolithic Preference Optimization without Reference Model

**📜 Paper**: https://huggingface.co/papers/2403.07691

The introduction of a reference model-free monolithic odds ratio preference optimization algorithm (ORPO) enhances preference alignment during supervised fine-tuning, surpassing larger models in key evaluations. To reproduce the paper's setting, use this configuration:

⟦T54⟧

## Contrastive Preference Optimization

Papers relating to the ⟦T187⟧

### Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation

**📜 Paper**: https://huggingface.co/papers/2401.08417

Introduces Contrastive Preference Optimization (CPO), a preference-based method for machine translation that trains models to avoid adequate-but-imperfect translations instead of mimicking references as in SFT. The paper analyzes limitations of SFT on MT (including reference quality issues) and shows that applying CPO to ALMA with only 22K parallel sentences yields ALMA-R, which matches or exceeds WMT competition winners and GPT-4 on WMT'21–WMT'23. Used in TRL via ⟦T188⟧. To reproduce the paper's setting, use this configuration:

⟦T55⟧

### SimPO: Simple Preference Optimization with a Reference-Free Reward

**📜 Paper**: https://huggingface.co/papers/2405.14734

SimPO is a simpler yet more effective preference optimization approach that uses the average log probability of a sequence as the implicit reward, eliminating the need for a reference model. It introduces a target reward margin to the Bradley-Terry objective to encourage a larger margin between winning and losing responses. To reproduce the paper's setting, use this configuration:

⟦T56⟧

### AlphaPO -- Reward shape matters for LLM alignment

**📜 Paper**: https://huggingface.co/papers/2501.03884

AlphaPO is a new Direct Alignment Algorithms (DAAs) method that leverages an alpha-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. To reproduce the paper's setting, use this configuration:

⟦T57⟧

## Triple Preference Optimization

Papers relating to the ⟦T189⟧

### Triple Preference Optimization: Achieving Better Alignment using a Single Step Optimization

**📜 Paper**: https://huggingface.co/papers/2405.16681

Introduces Triple Preference Optimization (TPO), a preference learning method that aligns an LLM with three responses per prompt — a gold (⟦T112⟧) completion, a preferred (⟦T113⟧) completion and a dispreferred (⟦T114⟧) completion — in a single optimization step. TPO combines a contrastive objective on the (chosen, rejected) pair with a supervised NLL term on the gold response, removing the need for a separate SFT stage and the reference model used in DPO. Used in TRL via ⟦T190⟧. To reproduce the paper's setting (Llama-3-Base, 5K), use this configuration:

⟦T58⟧

To use the TPO-L variant (length-normalized log-probabilities with a target reward margin γ), set ⟦T115⟧ and ⟦T116⟧:

⟦T59⟧

## Nash Learning from Human Feedback

Papers relating to the ⟦T191⟧

### Nash Learning from Human Feedback

**📜 Paper**: https://huggingface.co/papers/2312.00886

Introduces Nash-MD, an alternative to standard RLHF that learns a preference model conditioned on two inputs and finds a policy at the Nash equilibrium. Instead of optimizing against a reward model, Nash-MD produces policies that consistently generate responses preferred over those of any competing policy. The algorithm is based on mirror descent principles. Used in TRL via ⟦T192⟧.

⟦T60⟧

## Reward Modeling

Papers relating to the ⟦T193⟧ and ⟦T194⟧

### Solving math word problems with process- and outcome-based feedback

**📜 Paper**: https://huggingface.co/papers/2211.14275

Compares process-based supervision (per-step reasoning feedback) and outcome-based supervision (final-answer only) for math reasoning on GSM8K. Outcome-based training yields similar final-answer error with less labeling, but process-based supervision or learned process reward models (PRMs) are needed to reduce reasoning-step errors. The paper improves prior best from 16.8% to 12.7% final-answer error and 14.0% to 3.4% reasoning error among correct-answer solutions. Used in TRL via ⟦T195⟧. To train a PRM using TRL, use this configuration:

⟦T61⟧

The paper does not specify training hyperparameters; it focuses on comparing process-based vs outcome-based supervision strategies.

### Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking

**📜 Paper**: https://huggingface.co/papers/2312.09244

This paper proposed an auxiliary loss function designed to directly learn a centered reward model. This auxiliary loss minimizes the squared sum of the rewards, encouraging the model to naturally produce mean-zero outputs and thereby resolving the issue of underdetermination.

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x,y^+,y^-) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y^+) - r_\theta(x, y^-)) \textcolor{red}{- \eta \cdot (r_\theta(x, y^+) + r_\theta(x, y^-))^2} \right].
$$

To use this auxiliary loss with ⟦T196⟧, you can use the ⟦T117⟧ argument in ⟦T197⟧ as follows:

⟦T62⟧

### Llama 2: Open Foundation and Fine-Tuned Chat Models

**📜 Paper**: https://huggingface.co/papers/2307.09288

In this paper, the authors propose to leverage their preference ratings being decomposed as a scale of four points (e.g., _significantly better_) to provide more informative feedback to the reward model. This is done by adding a margin to the loss function, which encourages the reward model to assign larger gaps in scores for pairs with higher preference ratings.

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x,y^+,y^-,\textcolor{red}{m}) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y^+) - r_\theta(x, y^-) \textcolor{red}{- m}) \right].
$$

You can add a margin to the loss by adding a ⟦T118⟧ column to the dataset. The following example shows how to set up a the "Margin Small" setting of the paper.

⟦T63⟧

## Online Direct Preference Optimization

Papers relating to the ⟦T119⟧

### Direct Language Model Alignment from Online AI Feedback

**📜 Paper**: https://huggingface.co/papers/2402.04792

Online DPO improves direct alignment from preferences methods by providing real-time feedback from a model, outperforming both DPO and PPO methods.

To use Online DPO, you can use the ⟦T120⟧.

### Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF

**📜 Paper**: https://huggingface.co/papers/2405.21046

XPO augments the DPO objective with a novel and principled exploration bonus, empowering the algorithm to explore outside the support of the initial model and human feedback data. It is a one-line change to online DPO that is provably sample-efficient and converges to a near-optimal language model policy. The paper defines α > 0 (乐观系数) 和 β > 0 (KL 正则化)，但未指定数值。以下配置使用 TRL 默认值：

```python
from trl.experimental.xpo import XPOConfig

training_args = XPOConfig(
    alpha=1e-5,  # α exploration bonus weight, α ≥ 0 where α=0 reduces to online DPO (TRL default)
    beta=0.1,  # β KL regularization coefficient (TRL default)
)
```

## 蒸馏

有关在教师模型的帮助下训练学生模型的论文。

### 语言模型的策略蒸馏：从自身产生的错误中学习

**📜论文**：https://huggingface.co/papers/2306.13649

引入广义知识蒸馏 (GKD)，它通过根据教师反馈训练学生自己生成的输出（而不是一组固定的序列）来解决自回归模型的 KD 中的分布不匹配问题。 GKD 支持灵活的损失函数（例如，当学生无法匹配老师时超出 KL）并与 RL 微调 (RLHF) 集成。该论文报告了摘要、翻译、算术推理和指令调整的结果。通过 [experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer) 在 TRL 中使用，它暴露了论文的 on/off-policy 混合 (`lmbda`)。 [experimental.distillation.DistillationTrainer](/docs/trl/v1.9.2/en/distillation_trainer#trl.experimental.distillation.DistillationTrainer) 对于始终在线策略的情况实现相同的广义 JSD 目标。要重现纸张的设置，请使用以下配置：

```python
from trl.experimental.gkd import GKDConfig

# XSum summarization task (Table A.1 of the paper)
training_args = GKDConfig(
    lmbda=0.5,  # λ student data fraction (Section 3 of the paper)
    beta=0.5,  # β Generalized JSD interpolation, 0=KL, 1=reverse KL (Section 3 of the paper)
    temperature=1.0,  # student training temperature (Appendix A of the paper)
    max_steps=40000,  # training steps (Table A.1 of the paper)
    learning_rate=3e-4,  # learning rate (Table A.1 of the paper)
    per_device_train_batch_size=32,  # batch size (Table A.1 of the paper)
    warmup_steps=2000,  # warm-up steps (Table A.1 of the paper)
    max_new_tokens=64,  # max output tokens (Table A.1 of the paper)
)
```### 关于在政策蒸馏的立场偏差

**📜论文**：https://huggingface.co/papers/2606.22600

引入重要性加权的策略蒸馏 (IW-OPD)，它通过根据累积的师生前缀差异重新加权采样令牌蒸馏更新来解决 OPD 中的位置偏差。早期的令牌保持较大的权重，而高漂移后的后期令牌则被降低权重。通过 `experimental.iw_opd.IWOPDTrainer` 和 `distillation_objective="iw_opd"` 在 TRL 中使用。

该论文通过修剪策略梯度设置 (verl) 和 vLLM 推出来优化 IW-OPD。 `IWOPDTrainer`公开了下面匹配的蒸馏和推出设置；论文中的策略优化设置，例如裁剪范围`0.2`、双裁剪常数`3.0`、内部 PPO epoch、熵系数、KL 奖励惩罚、辅助 KL 和 rollout 重要性校正不是 `IWOPDConfig` 参数。

```python
from trl.experimental.iw_opd import IWOPDConfig

# Table 6 and Algorithm 1 of the paper, mapped to IWOPDConfig where available.
training_args = IWOPDConfig(
    distillation_objective="iw_opd",
    iw_opd_gamma=0.5,  # γ amplification, Algorithm 1 and Appendix C.3
    lmbda=1.0,  # fully on-policy rollouts
    learning_rate=1e-5,  # Table 6
    per_device_train_batch_size=1,  # Table 6 uses PPO micro-batch size 1 per GPU
    gradient_accumulation_steps=32,  # with 32 GPUs, this gives the paper's 1024-prompt batch
    num_generations=1,  # Table 6 rollout samples per prompt
    temperature=1.0,  # Table 6 training decoding temperature
    top_p=1.0,  # Table 6 training decoding top-p
    max_prompt_length=2048,  # Table 6
    max_completion_length=16384,  # Table 6
    warmup_ratio=0.0,  # Table 6
    use_vllm=True,  # Table 6 uses vLLM rollouts
    vllm_sync_frequency=1,  # refresh rollout policy after each update
    save_steps=10,  # Table 6 checkpoint frequency
    eval_steps=10,  # Table 6 validation frequency
)
```

### 合规蒸馏

**📰博客**：https://thinkingmachines.ai/blog/on-policy-distillation/按策略蒸馏涉及一个学生模型，为每批训练数据生成部署。随后，我们从学生和教师模型中获得了每个推出令牌的概率分布。然后对学生模型进行优化，以最小化其自身令牌分布与教师模型令牌分布之间的负 Kullback-Leibler (KL) 差异。

|方法|取样|奖励信号|
|------------------------|------------|----------------|
|监督微调 |政策外|密集|
|强化学习 |在保政策 |稀疏|
|策略蒸馏 |在保政策 |密集|

按策略蒸馏已被证明优于 SFT、GRPO，并且可用于恢复 SFT 期间丢失的泛化能力。

此外，同策略蒸馏的计算效率更高，并且在使用有限数据进行训练时不太容易出现过度拟合。

要使用 TRL 训练基于策略蒸馏的模型，您可以使用以下配置以及 [experimental.distillation.DistillationTrainer](/docs/trl/v1.9.2/en/distillation_trainer#trl.experimental.distillation.DistillationTrainer) 和 [experimental.distillation.DistillationConfig](/docs/trl/v1.9.2/en/distillation_trainer#trl.experimental.distillation.DistillationConfig)：

```python
from trl.experimental.distillation import DistillationConfig

training_args = DistillationConfig(
    lmbda=1.0,  # student produces rollouts for all batches
    beta=1.0,  # to ensure reverse-kl as the loss function
    teacher_model_name_or_path="teacher-model",  # specify the teacher model
)
```

或者，您可以使用 [experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer) 和 [experimental.gkd.GKDConfig](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDConfig)：

```python
from trl.experimental.gkd import GKDConfig

training_args = GKDConfig(
    lmbda=1.0,  # student produces rollouts for all batches
    beta=1.0,  # to ensure reverse-kl as the loss function
    teacher_model_name_or_path="teacher-model",  # specify the teacher model
)
```您还可以使用 `GOLDTrainer` 和 `GOLDConfig` 通过类似的配置执行策略蒸馏：

```python
from trl.experimental.gold import GOLDConfig

config = GOLDConfig(
    lmbda=1.0, # student produces rollouts for all batches
    beta=1.0, # to ensure reverse-kl as the loss function
    teacher_model_name_or_path="teacher-model", # specify the teacher model

)
```

### 大语言模型的知识蒸馏

**📜论文**：https://huggingface.co/papers/2306.08543

MiniLLM是第一个on-policy知识蒸馏方法，它最小化教师和学生模型之间的序列级反向KLD，并通过强化学习进行优化。

它是[Think Machine Lab's On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)的通用版本，可以选择添加分布级单步蒸馏信号（如`beta=1`时的GKD）和长上下文反向KLD信号。

或者，您可以使用 [experimental.minillm.MiniLLMTrainer](/docs/trl/v1.9.2/en/minillm_trainer#trl.experimental.minillm.MiniLLMTrainer) 和 [experimental.minillm.MiniLLMConfig](/docs/trl/v1.9.2/en/minillm_trainer#trl.experimental.minillm.MiniLLMConfig) 执行 MiniLLM 蒸馏，如下所示：

```python
from datasets import load_dataset
from trl.experimental.minillm import MiniLLMTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

trainer = MiniLLMTrainer(
    model="Qwen/Qwen3-0.6B",
    teacher_model="Qwen/Qwen3-1.7B",
    train_dataset=dataset,
)
trainer.train()
```

欲了解更多详情，请参阅[MiniLLM Trainer documentation](minillm_trainer)。

### 通过自蒸馏强化学习

**📜论文**：https://huggingface.co/papers/2601.20802自蒸馏策略优化（SDPO）通过将丰富的文本反馈（例如，运行时错误、判断评估）转换为密集的学习信号，无需任何外部教师或明确的奖励模型，从而通过可验证的奖励来增强强化学习。 SDPO 将当前以反馈为条件的模型视为自学模型，并将其基于反馈的下一个代币预测提炼回策略中。值得注意的是，SDPO 的性能还优于标准 RLVR 环境中的基线，后者仅通过使用成功的推出作为失败尝试的隐式反馈来返回标量反馈。

```python
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

training_args = SDPOConfig(
    distillation_alpha=0.5,                # Jensen-Shannon divergence (recommended)
    distillation_mode="topk_logits",       # Explicitly select top-K logit distillation
    distillation_topk=100,                 # Required for top-K logit distillation
    distillation_is_clip=2.0,              # Importance sampling clipping
    distillation_weight=1.0,               # Convex weight: (1-w)*policy + w*distillation; 1.0 = pure distillation
    use_successful_as_teacher=True,        # Use successful rollouts as teacher
    teacher_model_kind="ema",              # Supported: "base", "live", "ema"
    teacher_update_rate=0.05,              # EMA update rate
    include_environment_feedback=True,     # required to use the dataset's privileged_context (defaults to False)
)

trainer = SDPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=...,
    args=training_args,
    train_dataset=...,
)
trainer.train()
```

预期的数据集列：

- `prompt`
- `privileged_context` 用于可选环境反馈

欲了解更多详情，请参阅[SDPO Trainer documentation](sdpo_trainer)。

###自我蒸馏实现持续学习

**📜论文**：https://huggingface.co/papers/2601.19897自蒸馏微调（SDFT）通过在训练期间生成补全来执行策略自蒸馏，然后将这些相同补全的明确的教师条件视图提炼回给学生。教师由`teacher_model_kind`选择：`"base"`（初始学生）、`"live"`（当前学生）或`"ema"`（指数平均教师）。教师提示由学生`prompt`加上数据集`privileged_context`内部组成。

```python
from datasets import Dataset

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["Example answer: 4."],
    }
)

training_args = SDFTConfig(
    distillation_alpha=0.5,
    distillation_mode="topk_logits",
    distillation_topk=5,
    max_completion_length=64,
)

trainer = SDFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

预期的数据集列：

- `prompt`
- `privileged_context` 仅包含额外的教师专用信息

欲了解更多详情，请参阅[SDFT Trainer documentation](sdft_trainer)。

### 极其简单的自蒸馏改进了代码生成

**📜论文**：https://huggingface.co/papers/2604.01193简单自蒸馏 (SSD) 通过在训练时温度和截断配置下对模型的完成进行采样，然后使用标准交叉熵损失对这些原始的、未经验证的样本进行微调，从而改进代码生成。不需要奖励模型、验证器、教师模型或强化学习。 SSD 以上下文相关的方式重塑令牌分布：在“锁定”位置（语法几乎不存在歧义）抑制干扰项尾部，同时保留“分叉”位置（存在多个有效延续）的多样性。

```python
from trl.experimental.ssd import SSDConfig, SSDTrainer

training_args = SSDConfig(
    temperature=0.6,                       # Training-time sampling temperature (T_train)
    top_k=20,                              # Training-time top-k truncation
    top_p=0.95,                            # Training-time top-p truncation
    max_completion_length=65536,
    learning_rate=5e-6,
)

trainer = SSDTrainer(
    model="Qwen/Qwen3-4B-Instruct",
    args=training_args,
    train_dataset=...,
)
trainer.train()
```

预期的数据集列：

- `prompt`

欲了解更多详情，请参阅[SSD Trainer documentation](ssd_trainer)。

## 分布式训练

### ZeRO：训练万亿参数模型的内存优化

**📜论文**：https://huggingface.co/papers/1910.02054

ZeRO（零冗余优化器）通过跨设备划分优化器状态、梯度和参数，同时保持低通信量和高计算粒度，消除了数据和模型并行训练中的内存冗余。这样可以有效地训练大型模型，否则 GPU 内存无法容纳这些模型。TRL 通过 [DeepSpeed integration](deepspeed_integration) 支持 ZeRO。要使用它，请提供包含您所需设置的 DeepSpeed 配置文件，

```yaml
# config.yaml
distributed_type: DEEPSPEED
num_processes: 2
deepspeed_config:
  zero_stage: 3
```

并使用`accelerate launch --config_file config_file`启动训练脚本。

```sh
accelerate launch --config_file config.yaml train.py
```

## 近端策略优化

与[experimental.ppo.PPOTrainer](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOTrainer)相关的论文

### 近端策略优化算法

**📜论文**：https://huggingface.co/papers/1707.06347

引入近端策略优化 (PPO)：策略梯度方法，在多个小批量时期内交替收集部署和优化修剪的代理目标。 PPO 保留了信赖域方法（例如 TRPO）的优点，具有更简单的实现和强大的经验样本效率，并在机器人和 Atari 基准测试中得到了验证。通过 [experimental.ppo.PPOTrainer](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOTrainer) 在 TRL 中使用。要将 PPO 与 TRL 结合使用，请使用以下配置：

```python
from trl.experimental.ppo import PPOConfig

training_args = PPOConfig(
    cliprange=0.2,  # ε clipping range (Section 3 and Table 3 of the paper, Mujoco setting)
    num_ppo_epochs=4,  # K epochs of minibatch updates (TRL default; paper uses K=10 Mujoco, K=3 Atari)
    gamma=1.0,  # γ discount factor (TRL default for LLM tasks; paper uses γ=0.99)
    lam=0.95,  # λ GAE parameter (Table 3 of the paper, Mujoco setting)
    kl_coef=0.05,  # KL penalty coefficient (Section 4 of the paper discusses adaptive KL)
    vf_coef=0.1,  # c₁ value function loss weight (Equation 9 of the paper)
)
```

### OpenEnv 集成，用于培训法学硕士与环境
https://huggingface.co/docs/trl/v1.9.2/openenv.md