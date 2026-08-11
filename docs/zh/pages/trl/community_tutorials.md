<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 社区教程

社区教程由 Hugging Face 社区的活跃成员制作，他们希望与他人分享他们的知识和专业知识。它们是了解库及其功能以及开始使用核心类和模式的好方法。

## 语言模型

### 教程

|任务|班级 |描述 |作者 |教程 |科拉布 |
| --- | --- | --- | --- | --- | --- |
|强化学习 | [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) | TRL 中的 GRPO 和 vLLM 高效在线培训 | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/grpo_vllm_online_training) | [⟦T5⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/grpo_vllm_online_training.ipynb) |
|强化学习 | [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) |法学硕士在 TRL 中使用 GRPO 进行推理培训后 | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl) | [⟦T6⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_llm_grpo_trl.ipynb) |
|强化学习 | [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) | Mini-R1：重现 Deepseek R1 的“顿悟时刻”强化学习教程 | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/mini-deepseek-r1) | [⟦T7⟧](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb) |
|强化学习 | [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) | LLaMA 3.1-8B 上的 RL 与 GRPO 和 Unsloth 优化 | [Andrea Manzoni](https://huggingface.co/AManzoni) | [Link](https://colab.research.google.com/github/amanzoni1/fine_tuning/blob/main/RL_LLama3_1_8B_GRPO.ipynb) | [⟦T8⟧](https://colab.research.google.com/github/amanzoni1/fine_tuning/blob/main/RL_LLama3_1_8B_GRPO.ipynb) | 
|指令调优| [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |使用 ChatML 格式和 QLoRA 微调 Google Gemma LLM | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/fine-tune-google-gemma) | [⟦T9⟧](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/gemma-lora-example.ipynb) |
|结构化生成| [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |使用 QLoRA 和 PEFT 微调 ​​Llama-2-7B 以生成 JSON 格式的波斯语产品目录 | [Mohammadreza Esmaeilian](https://huggingface.co/Mohammadreza) | [Link](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format) | [⟦T10⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format.ipynb) ||偏好优化 | [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) |使用直接偏好优化来对齐 Mistral-7b 以实现人类偏好对齐 | [Maxime Labonne](https://huggingface.co/mlabonne) | [Link](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html) | [⟦T11⟧](https://colab.research.google.com/github/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb) |
|偏好优化 | [experimental.orpo.ORPOTrainer](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOTrainer) |使用 ORPO 结合指令调整和偏好对齐对 Llama 3 进行微调 | [Maxime Labonne](https://huggingface.co/mlabonne) | [Link](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html) | [⟦T12⟧](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi) |
|指令调优| [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |如何通过 Hugging Face 调整 2025 年开放的法学硕士 | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/fine-tune-llms-in-2025) | [⟦T13⟧](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2025.ipynb) |
|阶梯推理| [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) |使用 vLLM 进行逐步推理的监督强化学习 (SRL) | [Deepak Swaminathan](https://huggingface.co/s23deepak) | [Link](https://github.com/s23deepak/Supervised-Reinforcement-Learning) | [⟦T14⟧](https://colab.research.google.com/github/s23deepak/Supervised-Reinforcement-Learning/blob/main/notebooks/srl_grpo_tutorial.ipynb) |

### 视频

|任务|标题 |作者 |视频 |
| --- | --- | --- | --- |
|指令调优|使用 Hugging Face TRL 微调开放 AI 模型 | [Wietse Venema](https://huggingface.co/wietsevenema) | [](https://youtu.be/cnGyyM0vOes) |
|指令调优|如何使用 Hugging Face、TRL 和 smoltalk 数据集微调 smol-LM | [Mayurji](https://huggingface.co/iammayur) | [](https://youtu.be/jKdXv3BiLu0) |

⚠️“如何使用 Hugging Face、TRL 和 smoltalk 数据集微调 smol-LM”的已弃用功能通知（点击展开）

> [!警告]
> 本教程使用了两个已弃用的功能：
>
> - `SFTTrainer(..., tokenizer=tokenizer)`：使用`SFTTrainer(..., processing_class=tokenizer)`代替，或者干脆省略它（将从模型中推断）。
> - `setup_chat_format(model, tokenizer)`：使用`SFTConfig(..., chat_template_path="Qwen/Qwen3-0.6B")`，其中`chat_template_path`指定要复制其聊天模板的模型。## 视觉语言模型

### 教程

|任务|班级 |描述 |作者 |教程 |科拉布 |
| --- | --- | --- | --- | --- | --- |
|视觉质量保证 | [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |微调 Qwen2-VL-7B 以在 ChartQA 数据集上进行视觉问答 | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) | [⟦T15⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_trl.ipynb) |
|视觉质量保证 | [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |在消费级 GPU 上使用 TRL 微调 SmolVLM | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl) | [⟦T16⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_smol_vlm_sft_trl.ipynb) |
|搜索引擎优化描述| [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |微调 Qwen2-VL-7B 以从图像生成 SEO 友好的描述 | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl) | [⟦T17⟧](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-multimodal-llms-with-trl.ipynb) |
|视觉质量保证 | [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) | PaliGemma 🤝 直接偏好优化 | [Merve Noyan](https://huggingface.co/merve) | [Link](https://github.com/merveenoyan/smol-vision/blob/main/PaliGemma_DPO.ipynb) | [⟦T18⟧](https://colab.research.google.com/github/merveenoyan/smol-vision/blob/main/PaliGemma_DPO.ipynb) |
|视觉质量保证 | [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) |在消费级 GPU 上使用直接偏好优化 (DPO) 和 TRL 来微调 SmolVLM | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct) | [⟦T19⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_dpo_smolvlm_instruct.ipynb) |
|物体检测接地| [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) |使用 TRL 微调 VLM 以实现物体检测接地 | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_object_detection_grounding) | [⟦T20⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_object_detection_grounding.ipynb) |
|视觉质量保证 | [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) |使用 MPO 通过 TRL 微调视觉语言模型 | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_mpo) | [⟦T21⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_mpo.ipynb) |
|强化学习 | [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) |使用 TRL 训练 VLM 以使用 GRPO 进行推理 | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl) | [⟦T22⟧](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_grpo_trl.ipynb) |

## 语音语言模型

### 教程|任务|班级 |描述 |作者 |教程 |
| --- | --- | --- | --- | --- |
|文字转语音 | [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) |使用 TRL 通过 GRPO 对语音语言模型进行后期训练 | [Steven Zheng](https://huggingface.co/Steveeeeeeen) | [Link](https://huggingface.co/blog/Steveeeeeeen/llasa-grpo) |

## 贡献

如果您想将教程添加到此列表中，请打开 PR 来添加它。如果它与社区相关，我们将对其进行审核并合并。

### 广义知识蒸馏训练器
https://huggingface.co/docs/trl/v1.9.2/gkd_trainer.md