<!-- huggingface-docs: machine-translated zh-CN from English source -->

# TRL - 变形金刚强化学习

TRL 是一个全栈库，我们提供了一组工具来训练 Transformer 语言模型，方法包括监督微调 (SFT)、组相对策略优化 (GRPO)、直接偏好优化 (DPO)、奖励建模等。
该库与 🤗 [transformers](https://github.com/huggingface/transformers) 集成。

## 🎉 新消息

**🌍 多环境代理强化学习：** [⟦T0⟧](grpo_trainer) 现在支持每个示例环境选择和环境拥有的奖励 - 在一次运行中混合多个沙盒任务套件，并让每个环境使用 [Harbor](harbor) 和 [OpenEnv](openenv) 定义自己的评分。

**🎯 KTO 现已稳定：** 在与 [⟦T2⟧](dpo_trainer) 进行完全对齐后，[⟦T1⟧](kto_trainer) 升级为稳定 API。

## 分类法

以下是当前的 TRL 培训师列表，按方法类型组织（⚡️ = vLLM 支持；🧪 = 实验性）。

### 网上方法

- [⟦T3⟧](grpo_trainer)⚡️
- [⟦T4⟧](rloo_trainer)⚡️
- [⟦T5⟧](online_dpo_trainer)🧪⚡️
- [⟦T6⟧](nash_md_trainer)🧪⚡️
- [⟦T7⟧](ppo_trainer)🧪
- [⟦T8⟧](xpo_trainer)🧪⚡️

### 奖励建模

- [⟦T9⟧](reward_trainer)
- [⟦T10⟧](prm_trainer)🧪

### 离线方法

- [⟦T11⟧](sft_trainer)
- [⟦T12⟧](dpo_trainer)
- [⟦T13⟧](kto_trainer)
- [⟦T14⟧](bco_trainer)🧪
- [⟦T15⟧](cpo_trainer)🧪
- [⟦T16⟧](orpo_trainer)🧪

###知识蒸馏

- [⟦T17⟧](gkd_trainer)🧪
- [⟦T18⟧](minillm_trainer)🧪

您还可以在[TRL Hugging Face organization](https://huggingface.co/trl-lib)中探索TRL相关的模型、数据集和演示。

＃＃ 学习在 🤗 [smol course](https://github.com/huggingface/smol-course) 中使用 TRL 和其他库学习后期培训。

## 内容

该文档分为以下部分：

- **入门**：安装和快速入门指南。
- **概念指南**：数据集格式、培训常见问题解答和理解日志。
- **操作指南**：减少内存使用、加快训练速度、分配训练等。
- **集成**：DeepSpeed、Liger Kernel、PEFT 等。
- **示例**：示例概述、社区教程等。
- **API**：训练器、实用程序等。

## 博客文章发布于 2026 年 3 月 27 日
      TRL v1：当领域自身假设无效时保留的训练后库
    
    
      
      发布于 2025 年 10 月 23 日
      共同构建开放代理生态系统：OpenEnv 简介
    
    
      
      发布于 2025 年 8 月 7 日
      TRL 中的视觉语言模型对齐⚡️
    
    
      
      发布于 2025 年 6 月 3 日
      没有 GPU 落后：通过 TRL 中的同地 vLLM 释放效率
    
    
      
      发布于 2025 年 5 月 25 日
      🐯 Liger GRPO 遇见 TRL
    
    
      
      发布于 2025 年 1 月 28 日
      Open-R1：DeepSeek-R1的完全开放复制品
    
    
      
      发布于 2024 年 7 月 10 日
      使用 TRL 进行视觉语言模型的偏好优化
    
    
      
      发布于 2024 年 6 月 12 日
      将 RL 放回 RLHF
    
    
      
      发布于 2023 年 9 月 29 日
      通过 TRL 使用 DDPO 微调稳定扩散模型
    
    
      
      发布于 2023 年 8 月 8 日
      使用 DPO 微调 Llama 2
    
    
      
      发布于 2023 年 4 月 5 日
      StackLLaMA：使用 RLHF 训练 LLaMA 的实践指南发布于 2023 年 3 月 9 日
      在 24GB 消费级 GPU 上使用 RLHF 微调 20B LLM
    
    
      
      发布于 2022 年 12 月 9 日
      从人类反馈中说明强化学习
    
  

## 会谈

  
    
      
      2025 年 10 月 30 日发表的演讲
      使用 TRL 进行微调

### PPO 培训师
https://huggingface.co/docs/trl/v1.9.2/ppo_trainer.md