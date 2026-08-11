<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 权重分解低阶适应（DoRA）

> [!注意]
> 这是 LoRA 的一个变体，因此，除非本页另有说明，LoRA 的所有功能都适用于此方法。

该技术将权重的更新分解为两个部分：大小和方向。方向由普通 LoRA 处理，而幅度由单独的可学习参数处理。这可以提高 LoRA 的性能，尤其是在低级别时。有关 DoRA 的更多信息，请参阅 https://huggingface.co/papers/2402.09353。

```py
from peft import LoraConfig

config = LoraConfig(use_dora=True, ...)
```

如果部分模型或 DoRA 适配器被卸载到 CPU，您可以通过在 `config.runtime_config` 中使用`ephemeral_gpu_offload=True`，以一些临时（短暂）VRAM 开销为代价获得显着的加速。

```py
from peft import LoraConfig, LoraRuntimeConfig

config = LoraConfig(use_dora=True, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True), ...)
```

带有 DoRA 适配器的 `PeftModel` 也可以使用 `from_pretrained` 方法以及 `load_adapter` 方法加载 `ephemeral_gpu_offload=True` 标志。

```py
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, peft_model_id, ephemeral_gpu_offload=True)
```

## 优化

DoRA 针对评估模式下或 dropout 设置为 0 时的模型进行了优化（计算速度更快且占用内存更少）。我们重用
当时的基本结果以获得加速。
跑步[dora finetuning](https://github.com/huggingface/peft/blob/main/examples/dora_finetuning/dora_finetuning.py)
在 4090 上使用 `CUDA_VISIBLE_DEVICES=0 ZE_AFFINITY_MASK=0 time python examples/dora_finetuning/dora_finetuning.py --quantize --lora_dropout 0 --batch_size 16 --eval_step 2 --use_dora`，梯度累积设置为 2，最大步长设置为 20，得到以下观察结果：| |没有优化|通过优化 |
| :--: | :--: | :--: |
|火车运行时间（秒）| 359.7298 | **279.2676** |
|每秒训练样本 | 1.779 | 1.779 **2.292** |
|火车每秒步数| 0.056 | 0.056 **0.072** |

此外，通过使用`DoraCaching`辅助上下文可以进一步提高DoRA的运行时性能。这要求模型处于 `eval` 模式：

```py
from peft.helpers import DoraCaching

model.eval()
with DoraCaching():
    output = model(inputs)
```

对于 [⟦T13⟧](https://huggingface.co/meta-llama/Llama-3.1-8B)，[DoRA caching benchmark script](https://github.com/huggingface/peft/blob/main/examples/dora_finetuning/dora-caching.py) 表明，与 LoRA 相比：

- 没有缓存的 DoRA 需要 139% 的时间
- 不带缓存的 DoRA 需要多 4% 的内存
- 带缓存的 DoRA 需要 17% 的时间
- 带缓存的 DoRA 需要多 41% 的内存

因此，缓存可以使 DoRA 的推理速度显着加快，但它也需要更多的内存。理想情况下，如果用例允许，只需合并 DoRA 适配器即可避免内存和运行时开销。

## 注意事项

- DoRA 目前仅支持嵌入、线性和 Conv2d 层。
- DoRA 比纯 LoRA 引入了更大的开销，因此建议合并权重进行推理，参见[LoraModel.merge_and_unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.merge_and_unload)。
- DoRA 应使用以位和字节量化的权重（“QDoRA”）。然而，在将 QDoRA 与 DeepSpeed Zero2 结合使用时，已报告出现问题。### OFT
https://huggingface.co/docs/peft/v0.20.0/package_reference/oft.md