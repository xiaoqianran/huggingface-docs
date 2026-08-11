<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 从 Jupyter Notebooks 启动分布式训练

本教程教您如何使用分布式系统上的 Jupyter Notebook 中的 🤗 Accelerate 微调计算机视觉模型。
您还将学习如何设置一些所需的要求，以确保正确配置您的环境、正确准备数据，以及最后如何启动培训。

    本教程还可以作为 Jupyter Notebook [here](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_cv_example.ipynb) 提供

## 配置环境

在执行任何训练之前，系统中必须存在 Accelerate 配置文件。通常这可以通过在终端中运行以下命令并回答提示来完成：

```bash
accelerate config
```

但是，如果一般默认设置没问题，并且您*不*在 TPU 上运行，则 Accelerate 有一个实用程序可以通过 [utils.write_basic_config()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.commands.config.default.write_basic_config) 将您的设备配置快速写入配置文件。

以下代码将在写入配置后重新启动 Jupyter，因为调用 CUDA 运行时或 XPU 运行时来执行此操作。CUDA 和 XPU 在多设备系统上只能初始化一次。可以在笔记本中进行调试并调用 CUDA/XPU，但为了最终训练，需要执行完整的清理和重新启动。
    

```python
import os
from accelerate.utils import write_basic_config

write_basic_config()  # Write a config file
os._exit(00)  # Restart the notebook
```

## 准备数据集和模型

接下来您应该准备数据集。如前所述，在准备 `DataLoaders` 和模型时应格外小心，以确保**没有**放置在*任何* GPU 上。 

如果这样做，建议将该特定代码放入一个函数中，并从笔记本启动器界面中调用该函数，稍后将显示。 

确保根据指示下载数据集[here](https://github.com/huggingface/accelerate/tree/main/examples#simple-vision-example)

```python
import os, re, torch, PIL
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from accelerate import Accelerator
from accelerate.utils import set_seed
from timm import create_model
```

首先，您需要创建一个函数来根据文件名提取类名：

```python
import os

data_dir = "../../images"
fnames = os.listdir(data_dir)
fname = fnames[0]
print(fname)
```

```python out
beagle_32.jpg
```

在本例中，标签是`beagle`。使用正则表达式，您可以从文件名中提取标签：

```python
import re

def extract_label(fname):
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]
```

```python
extract_label(fname)
```

您可以看到它正确返回了我们文件的正确名称：

```python out
"beagle"
```

接下来应该创建一个 `Dataset` 类来处理抓取图像和标签：

```python
class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}
```现在构建数据集。在训练函数之外，您可以找到并声明所有文件名和标签，并将它们用作训练函数内的引用 
推出功能：

```python
fnames = [os.path.join("../../images", fname) for fname in fnames if fname.endswith(".jpg")]
```

接下来收集所有标签：

```python
all_labels = [extract_label(fname) for fname in fnames]
id_to_label = list(set(all_labels))
id_to_label.sort()
label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}
```

接下来，您应该创建一个 `get_dataloaders` 函数，它将返回您构建的数据加载器。如前所述，如果数据是自动的 
在构建 `DataLoaders` 时发送到 GPU 或 TPU 设备，必须使用此方法构建它们。 

```python
def get_dataloaders(batch_size: int = 64):
    "Builds a set of dataloaders with a batch_size"
    random_perm = np.random.permutation(len(fnames))
    cut = int(0.8 * len(fnames))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # For training a simple RandomResizedCrop will be used
    train_tfm = Compose([RandomResizedCrop((224, 224), scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset([fnames[i] for i in train_split], image_transform=train_tfm, label_to_id=label_to_id)

    # For evaluation a deterministic Resize will be used
    eval_tfm = Compose([Resize((224, 224)), ToTensor()])
    eval_dataset = PetsDataset([fnames[i] for i in eval_split], image_transform=eval_tfm, label_to_id=label_to_id)

    # Instantiate dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=4)
    return train_dataloader, eval_dataloader
```

最后，您应该导入稍后使用的调度程序：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```

## 编写训练函数

现在您可以构建训练循环。 [notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher) 的工作原理是传入一个函数来调用，该函数将在分布式系统上运行。

这是动物分类问题的基本训练循环：

    代码已被拆分，以便对每个部分进行解释。可以复制和粘贴的完整版本将在最后提供

```python
def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    accelerator = Accelerator(mixed_precision=mixed_precision)
```

首先，您应该在训练循环中尽早设置种子并创建一个 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 对象。如果在 TPU 上进行训练，您的训练循环应将模型作为参数，并且应将其实例化 
    在训练循环函数之外。参见[TPU best practices](../concept_guides/training_tpu) 
    了解原因

接下来，您应该构建数据加载器并创建模型：

```python
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))
```

    您在这里构建模型，以便种子也控制新的权重初始化

当您在此示例中执行迁移学习时，模型的编码器开始冻结，因此模型的头部可以 
仅最初训练：

```python
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
```

标准化图像批次将使训练速度更快一些：

```python
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]
```

要使这些常量在活动设备上可用，您应该将其设置为加速器的设备：

```python
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)
```

接下来实例化用于训练的其余 PyTorch 类：

```python
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(train_dataloader))
```

在将所有内容传递给[prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)之前。

    没有需要记住的特定顺序，您只需按照提供给准备方法的顺序解压对象即可。

```python
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
```

现在训练模型：

```python
    for epoch in range(5):
        model.train()
        for batch in train_dataloader:
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
```

与训练循环相比，评估循环看起来略有不同。通过的元素数量以及总体 
每个批次的总准确度将添加到两个常数中：

```python
        model.eval()
        accurate = 0
        num_elems = 0
```接下来是标准 PyTorch 循环的其余部分：

```python
        for batch in eval_dataloader:
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
```

最后一个主要区别之前。 

进行分布式评估时，需要传递预测和标签 
[gather()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.gather)，以便所有数据在当前设备上可用，并且可以实现正确计算的指标：

```python
            accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()
```

现在你只需要计算这个问题的实际指标，你可以使用 [print()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.print) 在主进程上打印它：

```python
        eval_metric = accurate.item() / num_elems
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
```

下面提供了此训练循环的完整版本：

```python
def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Build dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)

    # Instantiate the model (you build the model here so that the seed also controls new weight initializations)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # You can normalize the batches of images to be a bit faster
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]

    # To make these constants available on the active device, set it to the accelerator device
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)

    # Instantiate the learning rate scheduler
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(train_dataloader))

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the objects in the same order you gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Now you train the model
    for epoch in range(5):
        model.train()
        for batch in train_dataloader:
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        accurate = 0
        num_elems = 0
        for batch in eval_dataloader:
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / num_elems
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
```

## 使用笔记本启动器

剩下的就是使用[notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher)。

您传入函数、参数（作为元组）以及要训练的进程数。 （更多信息请参见[documentation](../package_reference/launchers)）

```python
from accelerate import notebook_launcher
```

```python
args = ("fp16", 42, 64)
notebook_launcher(training_loop, args, num_processes=2)
```

如果在多个节点上运行，则需要在每个节点上设置 Jupyter 会话并同时运行启动单元。

对于包含 2 个节点（计算机）、每个节点有 8 个 GPU 且主计算机的 IP 地址为“172.31.43.8”的环境，它看起来像这样：

```python
notebook_launcher(training_loop, args, master_addr="172.31.43.8", node_rank=0, num_nodes=2, num_processes=8)
```

在另一台机器上的第二个 Jupyter 会话中：

    注意 `node_rank` 的变化

```python
notebook_launcher(training_loop, args, master_addr="172.31.43.8", node_rank=1, num_nodes=2, num_processes=8)
```

在 TPU 上运行的情况下，它看起来像这样：

```python
model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

args = (model, "fp16", 42, 64)
notebook_launcher(training_loop, args, num_processes=8)
```要启动具有弹性的训练过程并实现容错，您可以使用 PyTorch 提供的 `elastic_launch` 功能。这需要设置额外的参数，例如`rdzv_backend`和`max_restarts`。以下是如何使用具有弹性功能的`notebook_launcher`的示例：

```python
notebook_launcher(
    training_loop,
    args,
    num_processes=2,
    max_restarts=3
)
```

当它运行时，它将打印进度并说明您运行的设备数量。本教程使用两个 GPU 运行：

```python out
Launching training on 2 GPUs.
epoch 0: 88.12
epoch 1: 91.73
epoch 2: 92.58
epoch 3: 93.90
epoch 4: 94.71
```

就是这样！

请注意，[notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher)忽略加速配置文件，根据配置使用启动：

```bash
accelerate launch
```

## 调试 

运行 `notebook_launcher` 时的一个常见问题是收到 CUDA/XPU 已初始化问题。这通常源于
来自笔记本中调用 PyTorch `torch.cuda` 或 `torch.xpu` 子库的导入或先前代码。为了帮助缩小问题范围，
您可以在您的环境中使用 `ACCELERATE_DEBUG_MODE=yes` 启动 `notebook_launcher` 并进行额外检查
生成时将确保可以创建常规进程并毫无问题地使用 CUDA/XPU。 （您的 CUDA/XPU 代码之后仍然可以运行）。

## 结论

该笔记本展示了如何从 Jupyter Notebook 内部执行分布式训练。需要记住的一些要点：- 确保保存传递给 [notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher) 的函数使用 CUDA/XPU（或 CUDA/XPU 导入）的所有代码
- 将`num_processes`设置为用于训练的设备数量（例如GPU、XPU、CPU、TPU等的数量）
- 如果使用 TPU，请在训练循环函数之外声明您的模型

### 将 Accelerate 添加到您的代码中
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/migration.md