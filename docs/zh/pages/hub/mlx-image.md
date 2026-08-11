<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 mlx-image

[⟦T4⟧](https://github.com/riccardomusmeci/mlx-image)是[Riccardo Musmeci](https://github.com/riccardomusmeci)在Apple[MLX](https://github.com/ml-explore/mlx)上开发的图像模型库。它试图复制伟大的[timm](https://github.com/huggingface/pytorch-image-models)，但适用于 MLX 型号。

## 在 Hub 上探索 mlx-image

您可以通过使用 `mlx-image` 库名称进行过滤来查找 `mlx-image` 模型，如 [this query](https://huggingface.co/models?library=mlx-image&sort=trending) 中所示。
还有一个开放的 [mlx-vision](https://huggingface.co/mlx-vision) 社区，供贡献者转换和发布 MLX 格式的权重。

## 安装

```bash
pip install mlx-image
```

## 型号

模型权重可在 HuggingFace 的 [⟦T7⟧](https://huggingface.co/mlx-vision) 社区上找到。

要加载具有预训练权重的模型：
```python
from mlxim.model import create_model

# loading weights from HuggingFace (https://huggingface.co/mlx-vision/resnet18-mlxim)
model = create_model("resnet18") # pretrained weights loaded from HF

# loading weights from local file
model = create_model("resnet18", weights="path/to/resnet18/model.safetensors")
```

列出所有可用型号：

```python
from mlxim.model import list_models
list_models()
```

## ImageNet-1K 结果

前往[results-imagenet-1k.csv](https://github.com/riccardomusmeci/mlx-image/blob/main/results/results-imagenet-1k.csv)查看每个转换为`mlx-image`的模型及其在不同设置的ImageNet-1K上的表现。

> **TL;DR** 性能与 PyTorch 实现的原始模型相当。

## 与 PyTorch 和其他熟悉的工具的相似之处

`mlx-image` 尝试尽可能接近 PyTorch：
- `DataLoader` -> 您可以定义自己的`collate_fn`，也可以使用`num_workers`来加快数据加载速度
- `Dataset` -> `mlx-image` 已经支持 `LabelFolderDataset` （又好又旧的 PyTorch `ImageFolder`）和 `FolderDataset` （其中包含图像的通用文件夹）- `ModelCheckpoint` -> 跟踪最佳模型并将其保存到磁盘（类似于 PyTorchLightning）。它还建议尽早停止

## 培训

训练与 PyTorch 类似。以下是如何训练模型的示例：

```python
import mlx.nn as nn
import mlx.optimizers as optim
from mlxim.model import create_model
from mlxim.data import LabelFolderDataset, DataLoader

train_dataset = LabelFolderDataset(
    root_dir="path/to/train",
    class_map={0: "class_0", 1: "class_1", 2: ["class_2", "class_3"]}
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
model = create_model("resnet18") # pretrained weights loaded from HF
optimizer = optim.Adam(learning_rate=1e-3)

def train_step(model, inputs, targets):
    logits = model(inputs)
    loss = mx.mean(nn.losses.cross_entropy(logits, target))
    return loss

model.train()
for epoch in range(10):
    for batch in train_loader:
        x, target = batch
        train_step_fn = nn.value_and_grad(model, train_step)
        loss, grads = train_step_fn(x, target)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
```

## 其他资源

* [mlx-image repository](https://github.com/riccardomusmeci/mlx-image)
* [mlx-vision community](https://huggingface.co/mlx-vision)

## 联系方式

如果您有任何疑问，请发送电子邮件至`riccardomusmeci92@gmail.com`。

### 图像数据集
https://huggingface.co/docs/hub/datasets-image.md