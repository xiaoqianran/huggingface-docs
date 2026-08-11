<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 例子

该目录包含一系列示例，演示如何将 TRL 库用于各种应用程序。我们提供用于高级用例的**脚本**和用于轻松启动和交互式实验的**笔记本**。

这些笔记本是独立的，可以在 **免费 Colab** 上运行，而脚本可以在 **单 GPU、多 GPU 或 DeepSpeed** 设置上运行。

**开始使用**

安装 TRL 和其他依赖项，如下所示：

```bash
pip install --upgrade trl[quantization]
```

检查其他可选依赖项[here](https://github.com/huggingface/trl/blob/main/pyproject.toml)。

对于脚本，您还需要一个 🤗 Accelerate 配置（推荐用于多 GPU 设置）：

```bash
accelerate config # will prompt you to define the training configuration
```

这允许您在单 GPU 或多 GPU 设置中使用 `accelerate launch` 运行脚本。

## 笔记本

这些笔记本更易于运行，专为快速进行 TRL 实验而设计。笔记本列表可以在[⟦T5⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/)目录中找到。|笔记本|描述 |在 Colab 中打开 |
|----------|-------------|----------------|
| [⟦T6⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_trl_lora_qlora.ipynb) |在免费 Colab 上使用 QLoRA 的 GRPO | [⟦T79⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_trl_lora_qlora.ipynb) |
| [⟦T7⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_agent.ipynb) | GRPO 代理培训 |由于 Colab GPU 出现 OOM 而无法使用 |
| [⟦T8⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_rnj_1_instruct.ipynb) | GRPO rnj-1-instruct with QLoRA 在 Colab 上使用 TRL 添加推理功能 | [⟦T80⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_rnj_1_instruct.ipynb) |
| [⟦T9⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_ministral3_vl.ipynb) |在免费 Colab 上使用 TRL 与 QLoRA 进行监督微调 (SFT) Ministral 3 | [⟦T81⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_ministral3_vl.ipynb) |
| [⟦T10⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_ministral3_vl.ipynb) | GRPO Ministral 3 与 QLoRA 在免费 Colab 上使用 TRL | [⟦T82⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_ministral3_vl.ipynb) |
| [⟦T11⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_nemotron_3.ipynb) | NVIDIA Nemotron 3 型号上采用 LoRA 的 SFT | [⟦T83⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_nemotron_3.ipynb) |
| [⟦T12⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_trl_lora_qlora.ipynb) |在免费 Colab 上使用 QLoRA 进行监督微调 (SFT) | [⟦T84⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb) |
| [⟦T13⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_qwen_vl.ipynb) |在免费 Colab 上使用 TRL 与 QLoRA 进行监督微调 (SFT) Qwen3-VL | [⟦T85⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_qwen_vl.ipynb) |
| [⟦T14⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_tool_calling.ipynb) |在没有本机工具调用支持的情况下使用 SFT 和 QLoRA 来教学工具调用模型 | [⟦T86⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_tool_calling.ipynb) |
| [⟦T15⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_qwen3_vl.ipynb) | GRPO Qwen3-VL 与 QLoRA 在免费 Colab 上使用 TRL | [⟦T87⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_qwen3_vl.ipynb) |

### OpenEnv 笔记本

这些笔记本演示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 的 `environment_factory` 在 [OpenEnv](openenv) 环境中训练模型。有关更多详细信息，请参阅[OpenEnv Integration](openenv)指南。|笔记本|描述 |在 Colab 中打开 |
|----------|-------------|----------------|
| [⟦T17⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/openenv_wordle_grpo.ipynb) | GRPO 在 OpenEnv 环境中玩 Wordle | [⟦T88⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) |
| [⟦T18⟧](https://github.com/huggingface/trl/tree/main/examples/notebooks/openenv_sudoku_grpo.ipynb) | GRPO 在 OpenEnv 环境中玩数独 | [⟦T89⟧](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_grpo.ipynb) |

## 脚本

脚本保存在 [⟦T19⟧](https://github.com/huggingface/trl/blob/main/trl/scripts) 和 [⟦T20⟧](https://github.com/huggingface/trl/blob/main/examples/scripts) 目录中。他们展示了如何使用不同的训练器，例如 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer)、`PPOTrainer`、[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer)、[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 等。

|文件 |描述 |
| --- | --- |
| [⟦T22⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/bco.py) |此脚本展示了如何使用 [KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 和 BCO 损失来微调模型，以使用 [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) 数据集提高指令遵循性、真实性、诚实性和帮助性。 |
| [⟦T23⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py) |此脚本展示了如何使用 [experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer) 来微调模型，以提高使用 [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) 数据集的有用性和无害性。 |
| [⟦T24⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/distillation.py) |该脚本展示了如何使用 [experimental.distillation.DistillationTrainer](/docs/trl/v1.9.2/en/distillation_trainer#trl.experimental.distillation.DistillationTrainer) 将教师模型提炼为学生模型，支持完整训练、混合开/关策略和 LoRA。 |
| [⟦T25⟧](https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py) |该脚本展示了如何使用 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 来微调模型。 |
| [⟦T26⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_vlm.py) |此脚本展示了如何使用 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 微调视觉语言模型，以使用 [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) 数据集减少幻觉。 || [⟦T27⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/gkd.py) |此脚本展示了如何使用 [experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer) 微调模型。 |
| [⟦T28⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/gold.py) |此脚本展示了如何使用 [experimental.gold.GOLDTrainer](/docs/trl/v1.9.2/en/gold_trainer#trl.experimental.gold.GOLDTrainer) 通过教师模型的在线蒸馏来微调模型。 |
| [⟦T29⟧](https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py) |此脚本展示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 微调模型。 |
| [⟦T30⟧](https://github.com/huggingface/trl/blob/main/trl/scripts/grpo_agent.py) |此脚本展示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 微调模型以启用代理使用。 |
| [⟦T31⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py) |此脚本展示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 微调多模态模型，以使用 [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 数据集进行推理。 |
| [⟦T32⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_continuous_batching.py) |该脚本展示了如何将 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 与 Transformer 的连续批处理引擎结合使用，以更快地生成具有可变完成长度的大批次。 |
| [⟦T33⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/gspo.py) |此脚本展示了如何通过 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 使用 GSPO 来微调模型，以便使用 [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) 数据集进行推理。 |
| [⟦T34⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/gspo_vlm.py) |此脚本展示了如何通过 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 使用 GSPO 来微调多模态模型，以使用 [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 数据集进行推理。 |
| [⟦T35⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py) |此脚本展示了如何使用 [KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 微调模型。 |
| [⟦T36⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/mpo_vlm.py) |此脚本展示了如何通过 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 使用 MPO，使用 [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) 数据集和一组带有权重的损失权重根据偏好来对齐模型。 || [⟦T37⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/nash_md.py) |此脚本展示了如何使用 [experimental.nash_md.NashMDTrainer](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDTrainer) 微调模型。 |
| [⟦T38⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/online_dpo.py) |该脚本展示了如何使用 [experimental.online_dpo.OnlineDPOTrainer](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOTrainer) 来微调模型。 |
| [⟦T39⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/online_dpo_vlm.py) |该脚本展示了如何使用 [experimental.online_dpo.OnlineDPOTrainer](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOTrainer) 微调视觉语言模型。 |
| [⟦T40⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) |此脚本展示了如何使用 [experimental.orpo.ORPOTrainer](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOTrainer) 来微调模型，以提高使用 [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) 数据集的有用性和无害性。 |
| [⟦T41⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/harbor/data_agent.py) |此脚本展示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 针对 Harbor 任务套件训练模型，并使用可插入的基本代理（`bash` / `jupyter` / `terminal_notes` 线束位于 `examples/scripts/harbor/harnesses/` 下）。请参阅 [Harbor Integration](harbor) 设置和使用指南。 |
| [⟦T46⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openreward/seta.py) |此脚本展示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 针对 openreward.ai 目录上的 SETA ORS 环境训练模型。有关设置和使用的信息，请参阅[OpenReward Integration](openreward)指南。 |
| [⟦T47⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py) |该脚本展示了如何使用 [experimental.ppo.PPOTrainer](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOTrainer) 微调模型，以提高其继续使用积极情绪或物理描述性语言的文本的能力。 |
| [⟦T48⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py) |此脚本展示了如何使用 [experimental.ppo.PPOTrainer](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOTrainer) 微调模型以提高其生成 TL;DR 摘要的能力。 |
| [⟦T49⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/prm.py) |此脚本展示了如何使用 [experimental.prm.PRMTrainer](/docs/trl/v1.9.2/en/prm_trainer#trl.experimental.prm.PRMTrainer) 微调流程监督奖励模型 (PRM)。 || [⟦T50⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py) |此脚本展示了如何使用 [RewardTrainer](/docs/trl/v1.9.2/en/reward_trainer#trl.RewardTrainer) 在您自己的数据集上训练结果奖励模型 (ORM)。 |
| [⟦T51⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/rloo.py) |该脚本展示了如何使用[RLOOTrainer](/docs/trl/v1.9.2/en/rloo_trainer#trl.RLOOTrainer)微调模型以提高其解决数学问题的能力。 |
| [⟦T52⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sdft.py) |该脚本展示了如何使用 [experimental.sdft.SDFTTrainer](/docs/trl/v1.9.2/en/sdft_trainer#trl.experimental.sdft.SDFTTrainer) 进行自蒸馏微调 (SDFT)。 |
| [⟦T53⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sdpo.py) |此脚本展示了如何使用 [experimental.sdpo.SDPOTrainer](/docs/trl/v1.9.2/en/sdpo_trainer#trl.experimental.sdpo.SDPOTrainer) 使用 [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) 数据集通过可验证的数学奖励和可选环境反馈来微调模型。 |
| [⟦T54⟧](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) |此脚本展示了如何使用 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 微调模型。 |
| [⟦T55⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_gemma3.py) |此脚本展示了如何使用 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 微调 Gemma 3 模型。 |
| [⟦T56⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_nemotron_3.py) |此脚本展示了如何使用 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 微调 NVIDIA Nemotron 3 模型。 |
| [⟦T57⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_tiny_aya_tool_calling.py) |此脚本展示了如何使用 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 来教导模型调用工具，而无需使用 [bebechien/SimpleToolCalling](https://huggingface.co/datasets/bebechien/SimpleToolCalling) 数据集的本机工具调用支持。 |
| [⟦T58⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py) |该脚本展示了如何使用[SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer)在聊天设置中微调视觉语言模型。该脚本仅在 [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf)、[LLaVA 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) 和 [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) 模型上进行了测试，因此用户可能会在其他模型架构中看到意外行为。 || [⟦T59⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py) |此脚本展示了如何使用 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 微调视觉到文本任务的 Gemma 3 模型。 |
| [⟦T60⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_smol_vlm.py) |该脚本展示了如何使用 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 微调 SmolVLM 模型。 |
| [⟦T61⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/ssd.py) |此脚本展示了如何在代码生成中使用 [experimental.ssd.SSDTrainer](/docs/trl/v1.9.2/en/ssd_trainer#trl.experimental.ssd.SSDTrainer) 进行简单自蒸馏 (SSD)。 |
| [⟦T62⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/ssd_eval.py) |该脚本使用 vLLM 和官方 `codegen_metrics` (pass@k) 评估 LiveCodeBench 上经过 SSD 训练的检查点。 |
| [⟦T64⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/tpo.py) |此脚本演示如何使用 [experimental.tpo.TPOTrainer](/docs/trl/v1.9.2/en/tpo_trainer#trl.experimental.tpo.TPOTrainer) 使用 [tpo-alignment/triple-preference-ultrafeedback-40K](https://huggingface.co/datasets/tpo-alignment/triple-preference-ultrafeedback-40K) 数据集进行三重偏好优化 (TPO)。 |
| [⟦T65⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/xpo.py) |此脚本展示了如何使用 [experimental.xpo.XPOTrainer](/docs/trl/v1.9.2/en/xpo_trainer#trl.experimental.xpo.XPOTrainer) 微调模型。 |

### OpenEnv 脚本

这些脚本演示了如何使用 [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 的 `environment_factory` 在 [OpenEnv](openenv) 环境中训练模型。有关更多详细信息，请参阅 [OpenEnv Integration](openenv) 指南。|文件 |描述 |
| --- | --- |
| [⟦T67⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) |使用 Echo 环境进行 GRPO 训练（最小示例）。 |
| [⟦T68⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/wordle.py) |使用 Wordle (TextArena) 环境进行 GRPO 训练。 |
| [⟦T69⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/catch.py) |使用 Catch (OpenSpiel) 环境进行 GRPO 训练。 |
| [⟦T70⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/sudoku.py) |使用数独环境进行 GRPO 训练。 |
| [⟦T71⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/multi_env.py) |多环境 GRPO 训练：Wordle + Catch 在同一训练中运行。 |
| [⟦T72⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/browsergym.py) |使用适用于 VLM 的 BrowserGym 环境进行 GRPO 培训。 |
| [⟦T73⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/browsergym_llm.py) |使用 BrowserGym 环境为法学硕士进行 GRPO 培训。 |
| [⟦T74⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/carla.py) |使用 CARLA 环境进行自动驾驶 GRPO 培训。 |
| [⟦T75⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/carla_vlm.py) |使用 CARLA 对具有多模式工具响应（相机图像）的 VLM 进行 GRPO 训练。 |
| [⟦T76⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/carla_vlm_gemma.py) |使用 CARLA 对 Gemma 4 进行 GRPO 训练，具有多模式工具响应（相机图像）。 |

## 分布式训练（针对脚本）

您可以使用 🤗 Accelerate 在多个 GPU 上运行脚本：

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

对于 DeepSpeed ZeRO-{1,2,3}：

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

根据需要调整 `NUM_GPUS` 和 `--all_arguments_of_the_script`。

### CPO 培训师
https://huggingface.co/docs/trl/v1.9.2/cpo_trainer.md