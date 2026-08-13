<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 瞄准空间

**Aim** 是一款易于使用且功能强大的开源实验跟踪器。 Aim 记录您的训练运行，并启用漂亮的 UI 来比较它们，并提供 API 来以编程方式查询它们。
ML 工程师和研究人员使用 Aim 浏览器，只需点击几下即可比较 1000 次训练运行。

查看 [Aim docs](https://aimstack.readthedocs.io/en/latest/) 了解有关 Aim 的更多信息。
如果您对新功能有想法或发现错误，请随时[open a feature request or report a bug](https://github.com/aimhubio/aim/issues/new/choose)。

在以下部分中，您将了解如何在 Hugging Face Hub 空间上部署 Aim 并直接从 Hub 探索您的训练运行。

## 在空间上部署瞄准

您只需单击一下即可在 Spaces 上部署 Aim！

    

创建空间后，您将看到`Building`状态，一旦变为`Running,`，您的空间就可以使用了！

现在，当您导航到 Space 的 **App** 部分时，您可以访问 Aim UI。

## 将您的实验与 Aim on Spaces 进行比较

让我们使用在 MNIST 上训练的 PyTorch CNN 的快速示例来演示端到端 Aim on Spaces 部署。
完整的示例位于 [Aim repo examples folder](https://github.com/aimhubio/aim/blob/main/examples/pytorch_track.py) 中。

```python
from aim import Run
from aim.pytorch import track_gradients_dists, track_params_dists

# Initialize a new Run
aim_run = Run()
...
items = {'accuracy': acc, 'loss': loss}
aim_run.track(items, epoch=epoch, context={'subset': 'train'})

# Track weights and gradients distributions
track_params_dists(model, aim_run)
track_gradients_dists(model, aim_run)
```Aim 跟踪的实验存储在`.aim` 文件夹中。 **要在空间中使用 Aim UI 显示日志，您需要将 `.aim` 文件夹压缩为 `tar.gz` 文件，并使用 `git` 或空间的文件和版本部分将其上传到您的空间。**

这是一个 bash 命令：

```bash
tar -czvf aim_repo.tar.gz .aim
```

就是这样！现在打开空间的应用程序部分，Aim UI 即可使用您的日志。
以下是我们所期待的：

![Aim UI on HF Hub Spaces](https://user-images.githubusercontent.com/23078323/232034340-0ba3ebbf-0374-4b14-ba80-1d36162fc994.png)

使用 Aim 的 Pythonic 搜索来过滤您的跑步。您可以编写 pythonic [queries against](https://aimstack.readthedocs.io/en/latest/using/search.html) 您跟踪的所有内容 - 指标、超参数等。查看 HF Hub Spaces 上的一些 [examples](https://huggingface.co/aimstack)。

> [!提示]
> 请注意，如果您的日志采用 TensorBoard 格式，您可以使用一个命令轻松将其转换为 Aim，并使用许多可用的高级和高性能训练运行比较功能。

## 有关 HF 空间的更多信息

- [HF Docker spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [HF Docker space examples](https://huggingface.co/docs/hub/spaces-sdks-docker-examples)

## 反馈和支持

如果您有改进建议或需要支持，请在[Aim GitHub repo](https://github.com/aimhubio/aim)上提出问题。

[Aim community Discord](https://github.com/aimhubio/aim#-community) 也可用于社区讨论。

### 音频数据集
https://huggingface.co/docs/hub/datasets-audio.md