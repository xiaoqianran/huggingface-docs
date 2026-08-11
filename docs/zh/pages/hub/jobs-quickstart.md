<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 快速入门

在本指南中，您将在几分钟内运行一个作业来微调 Hugging Face 基础设施上的开源模型。
确保您已登录 Hugging Face，并且您的帐户或组织上有 [pre-paid credits](https://huggingface.co/settings/billing)。然后，您可以访问您的[Jobs page](https://huggingface.co/settings/jobs)来创建和管理作业。

## 开始使用

首先安装 Hugging Face CLI：

### 1. 安装 CLI

推荐方法：

```bash
>>> curl -LsSf https://hf.co/cli/install.sh | bash
```

或者使用自制程序：

```bash
>>> brew install hf
```

或者使用紫外线：

```bash
>>> uv tool install hf
```

### 2. 登录您的 Hugging Face 帐户

登录

```bash
>>> hf auth login
```

### 3. 使用 `hf jobs` 命令创建您的第一个作业

运行 UV 命令或脚本

```bash
>>> hf jobs uv run python -c 'print("Hello from the cloud!")'
Job started with ID: 693aef401a39f67af5a41c0e
View at: https://huggingface.co/jobs/lhoestq/693aef401a39f67af5a41c0e
Hello from the cloud!
```

```bash
>>> echo "print('Hello from uv script!')" > script.py
>>> hf jobs uv run script.py
Job started with ID: 695f6cd8d2f3efac77e8cf7f
View at: https://huggingface.co/jobs/lhoestq/695f6cd8d2f3efac77e8cf7f
Hello from uv script!
```

运行 Docker 命令

```bash
>>> hf jobs run ubuntu echo 'Hello from the cloud!'
Job started with ID: 693aee76c67c9f186cfe233e
View at: https://huggingface.co/jobs/lhoestq/693aee76c67c9f186cfe233e
Hello from the cloud!
```

### 4.检查你的第一份工作

作业日志显示在您的终端中，但您也可以在作业页面中看到它们。打开作业页面可以查看作业信息、状态和日志：

## 训练脚本

这是一个简单的训练脚本，用于使用监督微调 (SFT) 将基本模型微调为会话模型。它使用 [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 模型、[trl-lib/Capybara](https://huggingface.co/datasets/trl-lib/Capybara) 数据集和 [TRL](https://huggingface.co/docs/trl/en/index) 库，并将生成的模型以 `"Qwen2.5-0.5B-SFT"` 的名称保存到您的 Hugging Face 帐户中：

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()
trainer.push_to_hub("Qwen2.5-0.5B-SFT")
```

将此脚本保存为 `train.py`，我们现在可以在 Hugging Face Jobs 上使用 UV 运行它。## 运行训练作业

`hf jobs`采用多个参数：使用`--flavor`选择硬件，使用`--timeout`选择最大持续时间，并使用`--env`和`--secrets`传递环境变量。在这里，我们使用带有 `--flavor a100-large` 的 A100 大型 GPU 风格，并使用 `--secrets HF_TOKEN` 将您的 Hugging Face 令牌作为秘密传递，以便能够将生成的模型推送到您的帐户。请参阅 [Persist your results](./jobs-manage#persist-your-results) 了解如何确保作业的输出在完成后仍然存在。

此外，UV 接受 `--with` 参数来定义 python 依赖项，因此我们使用 `--with trl` 来提供 `trl` 库。

您现在可以运行最终命令，如下所示：

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 6h \
    --with trl \
    --secrets HF_TOKEN \
    train.py
```

日志出现在您的终端中，您可以安全地按 Ctrl+C 停止流式传输日志，作业将继续运行。

```
...
Downloaded nvidia-cudnn-cu12 
Downloaded torch
Installed 66 packages in 233ms
Generating train split: 100%|██████████| 15806/15806 [00:00<00:00, 76686.50 examples/s]
Generating test split: 100%|██████████| 200/200 [00:00<00:00, 43880.36 examples/s]
Tokenizing train dataset: 100%|██████████| 15806/15806 [00:41<00:00, 384.97 examples/s]
Truncating train dataset: 100%|██████████| 15806/15806 [00:00<00:00, 212272.92 examples/s]
The model is already on multiple devices. Skipping the move to device specified in `args`.
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': 1.7357, 'grad_norm': 4.8733229637146, 'learning_rate': 1.9969635627530365e-05, 'entropy': 1.7238958358764649, 'num_tokens': 59528.0, 'mean_token_accuracy': 0.6124177813529968, 'epoch': 0.01}
{'loss': 1.6239, 'grad_norm': 6.200186729431152, 'learning_rate': 1.9935897435897437e-05, 'entropy': 1.644005584716797, 'num_tokens': 115219.0, 'mean_token_accuracy': 0.6259662985801697, 'epoch': 0.01}
{'loss': 1.4449, 'grad_norm': 6.167325496673584, 'learning_rate': 1.990215924426451e-05, 'entropy': 1.5156117916107177, 'num_tokens': 171787.0, 'mean_token_accuracy': 0.6586395859718323, 'epoch': 0.02}
{'loss': 1.6023, 'grad_norm': 5.133708953857422, 'learning_rate': 1.986842105263158e-05, 'entropy': 1.6885507702827454, 'num_tokens': 226067.0, 'mean_token_accuracy': 0.6271904468536377, 'epoch': 0.02}
```

关注 Hugging Face 职位页面上的职位进展：

在 CLI 中监控 GPU 使用情况和其他指标或使用 [MacOS menu bar](./jobs-manage#macos-menu-bar)。通过 CLI，您将获得：

```bash
>>> hf jobs stats
JOB ID                   CPU % NUM CPU MEM % MEM USAGE        NET I/O         GPU UTIL % GPU MEM % GPU MEM USAGE   
------------------------ ----- ------- ----- ---------------- --------------- ---------- --------- --------------- 
695e83c5d2f3efac77e8cf18 8%    12.0    7.18% 10.9GB / 152.5GB 0.0bps / 0.0bps 100%       31.92%    25.9GB / 81.2GB
```

工作完成后，在您的帐户中找到您的模型：

恭喜！您只需运行第一个作业来微调开源模型 🔥请随意在本地尝试您的模型并使用例如评估它单击“使用此模型”即可创建[transformers](https://huggingface.co/docs/transformers)，或者使用“部署”按钮将其一键部署到[Inference Endpoints](https://huggingface.co/docs/inference-endpoints)。

### 数据设计师
https://huggingface.co/docs/hub/datasets-data-designer.md