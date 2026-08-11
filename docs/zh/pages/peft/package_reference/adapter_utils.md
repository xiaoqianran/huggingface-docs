<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 石蒜

[LyCORIS](https://hf.co/papers/2309.14859)（Lora beYond Conventionalmethods，Other Rank adjustment Implements for Stablediffusion）是类似 LoRA 的矩阵分解适配器，可以修改 UNet 的交叉注意力层。 [LoHa](loha)和[LoKr](lokr)方法继承自此处的`Lycoris`类。

## LycorisConfig[[peft.tuners.lycoris_utils.LycorisConfig]]

#### peft.tuners.lycoris_utils.LycorisConfig[[peft.tuners.lycoris_utils.LycorisConfig]]

```python
peft.tuners.lycoris_utils.LycorisConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, rank_pattern: Optional[dict] = <factory>, alpha_pattern: Optional[dict] = <factory>)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lycoris_utils.py#L35)

LyCORIS 类似适配器的基本配置

## LycorisLayer[[peft.tuners.lycoris_utils.LycorisLayer]]

#### peft.tuners.lycoris_utils.LycorisLayer[[peft.tuners.lycoris_utils.LycorisLayer]]

```python
peft.tuners.lycoris_utils.LycorisLayer(base_layer: nn.Module)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lycoris_utils.py#L60)

LyCORIS 类似适配器的基础层

#### 合并[[peft.tuners.lycoris_utils.LycorisLayer.merge]]

```python
merge(safe_merge: bool = False, adapter_names: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lycoris_utils.py#L114)

**参数：**

safe_merge (`bool`, *可选*) ：如果`True`，合并操作将在原始权重的副本中执行，并在合并权重之前检查 NaN。如果您想检查合并操作是否会产生 NaN，这很有用。默认为`False`。

adapter_names (`List[str]`, *可选*) ：应合并的适配器名称列表。如果`None`，所有活动适配器将被合并。默认为`None`。将活动适配器权重合并到基本权重中

#### 取消合并[[peft.tuners.lycoris_utils.LycorisLayer.unmerge]]

```python
unmerge()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lycoris_utils.py#L168)

此方法从基本权重中取消合并所有合并的适配器层。

## LycorisTuner[[peft.tuners.lycoris_utils.LycorisTuner]]

#### peft.tuners.lycoris_utils.LycorisTuner[[peft.tuners.lycoris_utils.LycorisTuner]]

```python
peft.tuners.lycoris_utils.LycorisTuner(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lycoris_utils.py#L194)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)) ：Lora 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

LyCORIS 类似适配器的基础调谐器

### 多任务提示调优
https://huggingface.co/docs/peft/v0.20.0/package_reference/multitask_prompt_tuning.md