<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PEFT 类型

[PeftType](/docs/peft/v0.20.0/en/package_reference/peft_types#peft.PeftType) 包括 PEFT 支持的适配器，[TaskType](/docs/peft/v0.20.0/en/package_reference/peft_types#peft.TaskType) 包括 PEFT 支持的任务。

## PeftType[[peft.PeftType]]

#### peft.PeftType[[peft.PeftType]]

```python
peft.PeftType(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/peft_types.py#L19)

PEFT 中不同类型适配器的枚举类。

支持的 PEFT 类型：
- 提示_调整
- 多任务提示调整
- P_调整
- 前缀_调整
- 洛拉
- 阿达洛拉
——英国广播电视公司
- ADAPTION_PROMPT
-IA3
- 贝夫特
- 乐活
- 洛克尔
- 奥夫特
- XLORA
- 保利
- LN_调整
- 维拉
——弗罗德
- 傅里叶夫
- 人力资源管理局
- 骨头
- 错过
- 兰德洛拉
- 希拉
- C3A
- 道路
- 波夫
- OSF
- 德洛拉
- 格拉罗拉
- ADAMSS
- 德夫特

## 任务类型[[peft.TaskType]]

#### peft.TaskType[[peft.TaskType]]

```python
peft.TaskType(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/peft_types.py#L103)

PEFT 支持的不同类型任务的枚举类。

支持的任务类型概述：
- SEQ_CLS：文本分类。
- SEQ_2_SEQ_LM：序列到序列语言建模。
- CAUSAL_LM：因果语言建模。
- TOKEN_CLS：令牌分类。
- QUESTION_ANS：回答问题。
- FEATURE_EXTRACTION：特征提取。提供可用作嵌入或特征的隐藏状态
  用于下游任务。

### AutoPeft模型
https://huggingface.co/docs/peft/v0.20.0/package_reference/auto_class.md