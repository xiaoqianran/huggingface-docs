<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在 Hugging Face 中使用 timm

`timm`，也称为 [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)，是最先进的 PyTorch 图像模型、预训练权重以及用于训练、推理和验证的实用脚本的开源集合。

本文档重点介绍 Hugging Face Hub 中的 `timm` 功能，而不是 `timm` 库本身。有关`timm`库的详细信息，请访问[its documentation](https://huggingface.co/docs/timm)。

您可以使用 [models page](https://huggingface.co/models?library=timm&sort=downloads) 左侧的过滤器在 Hub 上找到多个 `timm` 型号。

Hub 上的所有型号都具有几个有用的功能：
1. 自动生成的模型卡，模型作者可以通过[information about their model](./model-cards)完成。
2. 元数据标签帮助用户发现相关的`timm`型号。
3. [interactive widget](./models-widgets)，您可以使用它直接在浏览器中玩模型。
4. 允许用户提出推理请求的[Inference Providers](./models-inference)。

## 使用 Hub 中的现有模型

只要安装了 `timm`，Hugging Face Hub 中的任何 `timm` 模型都可以通过一行代码加载！从 Hub 选择模型后，请将前缀为 `hf-hub:` 的模型 ID 传递给 `timm` 的 `create_model` 方法来下载并实例化模型。

```py
import timm

# Loading https://huggingface.co/timm/eca_nfnet_l0
model = timm.create_model("hf-hub:timm/eca_nfnet_l0", pretrained=True)
```如果您想了解如何加载特定模型，可以单击**在 timm 中使用**，您将获得一个工作片段来加载它！

### 推论

下面的代码片段展示了如何对从 Hub 加载的 `timm` 模型执行推理：

```py
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Load from Hub 🔥
model = timm.create_model(
    'hf-hub:nateraw/resnet50-oxford-iiit-pet',
    pretrained=True
)

# Set model to eval mode for inference
model.eval()

# Create Transform
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

# Get the labels from the model config
labels = model.pretrained_cfg['label_names']
top_k = min(len(labels), 5)

# Use your own image file here...
image = Image.open('boxer.jpg').convert('RGB')

# Process PIL image with transforms and add a batch dimension
x = transform(image).unsqueeze(0)

# Pass inputs to model forward function to get outputs
out = model(x)

# Apply softmax to get predicted probabilities for each class
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Grab the values and indices of top 5 predicted classes
values, indices = torch.topk(probabilities, top_k)

# Prepare a nice dict of top k predictions
predictions = [
    {"label": labels[i], "score": v.item()}
    for i, v in zip(indices, values)
]
print(predictions)
```

这应该会给您留下一个预测列表，如下所示：

```py
[
    {'label': 'american_pit_bull_terrier', 'score': 0.9999998807907104},
    {'label': 'staffordshire_bull_terrier', 'score': 1.0000000149011612e-07},
    {'label': 'miniature_pinscher', 'score': 1.0000000149011612e-07},
    {'label': 'chihuahua', 'score': 1.0000000149011612e-07},
    {'label': 'beagle', 'score': 1.0000000149011612e-07}
]
```

## 分享你的模型

您可以将您的 `timm` 模型直接分享到 Hugging Face Hub。这会将模型的新版本发布到 Hugging Face Hub，并为您创建模型存储库（如果尚不存在）。

在推送模型之前，请确保您已登录 Hugging Face：

```sh
python -m pip install huggingface_hub
hf auth login
```

或者，如果您更喜欢使用 Jupyter 或 Colaboratory 笔记本工作，安装 `huggingface_hub` 后，您可以使用以下方式登录：

```py
from huggingface_hub import notebook_login
notebook_login()
```

然后，使用 `push_to_hf_hub` 方法推送模型：

```py
import timm

# Build or load a model, e.g. timm's pretrained resnet18
model = timm.create_model('resnet18', pretrained=True, num_classes=4)

###########################
# [Fine tune your model...]
###########################

# Push it to the 🤗 Hub
timm.models.hub.push_to_hf_hub(
    model,
    'resnet18-random-classifier',
    model_config={'labels': ['a', 'b', 'c', 'd']}
)

# Load your model from the Hub
model_reloaded = timm.create_model(
    'hf-hub:<your-username>/resnet18-random-classifier',
    pretrained=True
)
```

## 其他资源

* timm（pytorch-image-models）[GitHub Repo](https://github.com/rwightman/pytorch-image-models)。
*蒂姆[documentation](https://huggingface.co/docs/timm)。
* [timmdocs](https://timm.fast.ai) [Aman Arora](https://github.com/amaarora) 提供的附加文档。
* [Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055) by [Chris Hughes](https://github.com/Chris-hughes10)。

### 数据设计师
https://huggingface.co/docs/hub/datasets-data-designer.md