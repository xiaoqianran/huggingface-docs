<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 PaddleNLP

[⟦T3⟧](https://github.com/PaddlePaddle/PaddleNLP)利用[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)框架，是一个易于使用且功能强大的NLP库，具有出色的预训练模型库，支持从研究到工业应用的广泛NLP任务。

## 在 Hub 中探索 PaddleNLP

您可以通过[models page](https://huggingface.co/models?library=paddlenlp&sort=downloads)左侧筛选找到`PaddleNLP`型号。

Hub 上的所有型号均具有以下功能：

1. 自动生成的模型卡，其中包含有助于发现的简短描述和元数据标签。
2. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。

3. 推理提供程序小部件，允许发出推理请求。

4. 在 Spaces 上轻松将模型部署为 Gradio 应用程序。

## 安装

首先，您可以按照 [PaddlePaddle Quick Start](https://www.paddlepaddle.org.cn/en/install) 安装 PaddlePaddle 框架以及您最喜欢的操作系统、包管理器和计算平台。

`paddlenlp` 通过 pip 提供快速的单行安装：

```
pip install -U paddlenlp
```

## 使用现有模型与 `transformer` 模型类似，`paddlenlp` 库提供了一个简单的单行代码，通过设置 `from_hf_hub=True` 从 Hugging Face Hub 加载模型！根据您想要使用它们的方式，您可以使用 `Taskflow` 函数来使用高级 API，也可以使用 `AutoModel` 和 `AutoTokenizer` 进行更多控制。

```py
# Taskflow provides a simple end-to-end capability and a more optimized experience for inference
from paddlenlp.transformers import Taskflow
taskflow = Taskflow("fill-mask", task_path="PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)

# If you want more control, you will need to define the tokenizer and model.
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)
model = AutoModelForMaskedLM.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)
```

如果您想了解如何加载特定模型，可以单击`Use in paddlenlp`，您将获得一个可以加载它的工作片段！

## 分享你的模型

您可以使用所有 `Model` 和 `Tokenizer` 类下的 `save_to_hf_hub` 方法来共享您的 `PaddleNLP` 模型。

```py
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)
model = AutoModelForMaskedLM.from_pretrained("PaddlePaddle/ernie-1.0-base-zh", from_hf_hub=True)

tokenizer.save_to_hf_hub(repo_id="<my_org_name>/<my_repo_name>")
model.save_to_hf_hub(repo_id="<my_org_name>/<my_repo_name>")
```

## 其他资源

- PaddlePaddle 安装[guide](https://www.paddlepaddle.org.cn/en/install)。
- PaddleNLP [GitHub Repo](https://github.com/PaddlePaddle/PaddleNLP)。
- [PaddlePaddle on the Hugging Face Hub](https://huggingface.co/PaddlePaddle)

### 音频数据集
https://huggingface.co/docs/hub/datasets-audio.md