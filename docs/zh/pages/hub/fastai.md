<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在 Hugging Face 中使用 fastai

`fastai` 是一个开源深度学习库，它利用 PyTorch 和 Python 提供高级组件来训练快速、准确的神经网络，并在文本、视觉和表格数据上提供最先进的输出。 

## 在 Hub 中探索 fastai

您可以通过[models page](https://huggingface.co/models?library=fastai&sort=downloads)左侧筛选找到`fastai`型号。

Hub 上的所有型号均具有以下功能：
1. 自动生成的模型卡，其中包含有助于发现的简短描述和元数据标签。
2. 一个交互式小部件，您可以使用它直接在浏览器中使用模型（用于图像分类）
3. 推理提供程序小部件，允许发出推理请求（用于图像分类）。

## 使用现有模型

`huggingface_hub` 库是一个轻量级 Python 客户端，具有从 Hub 下载模型的实用功能。

```bash
pip install huggingface_hub["fastai"]
```

安装库后，您只需使用`from_pretrained_fastai`方法即可。此方法不仅加载模型，而且在保存模型时验证`fastai`版本，这对于再现性很重要。

```py
from huggingface_hub import from_pretrained_fastai

learner = from_pretrained_fastai("espejelomar/identify-my-cat")

_,_,probs = learner.predict(img)
print(f"Probability it's a cat: {100*probs[1].item():.2f}%")

# Probability it's a cat: 100.00%
```如果您想了解如何加载特定模型，可以单击`Use in fastai`，您将获得一个可以加载它的工作片段！ 

## 分享你的模型

您可以使用 `push_to_hub_fastai` 方法分享您的 `fastai` 模型。

```py
from huggingface_hub import push_to_hub_fastai

push_to_hub_fastai(learner=learn, repo_id="espejelomar/identify-my-cat")
```

## 其他资源

* 快泰[course](https://course.fast.ai/)。
* 快泰[website](https://www.fast.ai/)。
* 与集线器[docs](https://docs.fast.ai/huggingface.html)集成。
* 与集线器[announcement](https://huggingface.co/blog/fastai)集成。

### 使用 Docker 运行
https://huggingface.co/docs/hub/spaces-run-with-docker.md