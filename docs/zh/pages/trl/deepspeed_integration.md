<!-- huggingface-docs: machine-translated zh-CN from English source -->

# DeepSpeed 集成

> [!警告]
> 部分正在建设中。欢迎贡献！

TRL 支持使用 DeepSpeed 进行训练，DeepSpeed 是一个实现高级训练优化技术的库。其中包括优化器状态分区、卸载、梯度分区等。

DeepSpeed 集成了[Zero Redundancy Optimizer (ZeRO)](https://huggingface.co/papers/1910.02054)，可以根据设备数量按比例缩放模型大小，并保持持续的高效率。

![ZeRO Stages](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/zero_stages.png)

## 安装

要将 DeepSpeed 与 TRL 结合使用，请使用以下命令进行安装：

```bash
pip install deepspeed
```

## 使用 DeepSpeed 运行训练脚本

无需修改您的训练脚本。只需使用 DeepSpeed 配置文件运行即可：

```bash
accelerate launch --config_file <ACCELERATE_WITH_DEEPSPEED_CONFIG_FILE.yaml> train.py
```

我们在 [⟦T3⟧](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs) 目录中提供了即用型 DeepSpeed 配置文件。例如，要使用 ZeRO Stage 2 运行训练，请使用以下命令：

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml train.py
```

## 其他资源

有关 DeepSpeed 插件的更多信息，请参阅 🤗 Accelerate [documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)。

### 奖励建模
https://huggingface.co/docs/trl/v1.9.2/reward_trainer.md