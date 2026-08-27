<!-- huggingface-docs: machine-translated zh-CN from English source -->

# TensorBoard 记录器

TensorBoard 是一个用于机器学习实验的可视化工具包。 TensorBoard 允许跟踪和可视化
损失和准确性等指标、可视化模型图、查看直方图、显示图像等等。
TensorBoard 与 Hugging Face Hub 完美集成。 Hub 自动检测 TensorBoard 痕迹（例如
`tfevents`）当推送到集线器时，集线器启动一个实例来可视化它们。获取有关 TensorBoard 的更多信息
Hub 上的集成，请查看 [this guide](https://huggingface.co/docs/hub/tensorboard)。

为了从这种集成中受益，`huggingface_hub`提供了一个自定义记录器来将日志推送到集线器。它的工作原理是
[SummaryWriter](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html) 的直接替代品，无需额外
需要代码。跟踪仍然保存在本地，并且后台作业定期将它们推送到集线器。

## HFSummaryWriter[[huggingface_hub.HFSummaryWriter]]

#### Huggingface_hub.HFSummaryWriter[[huggingface_hub.HFSummaryWriter]]

```python
huggingface_hub.HFSummaryWriter(*args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_tensorboard_logger.py#L45)

**参数：**

repo_id (`str`) ：日志将推送到的存储库的 id。

logdir (`str`, *可选*) ：将写入日志的目录。如果未指定，则底层 `SummaryWriter` 对象将创建本地目录。commit_every（`int`或`float`，*可选*）：将日志推送到集线器的频率（以分钟为单位）。默认为 5 分钟。

squash_history (`bool`, *可选*) ：是否在每次提交后压缩存储库的历史记录。默认为`False`。压缩提交对于避免存储库变得太大时性能下降很有用。

repo_type (`str`, *可选*) ：日志将推送到的存储库的类型。默认为“模型”。

repo_revision (`str`, *可选*) ：日志将推送到的存储库的修订版本。默认为“主”。

repo_private (`bool`, *可选*) ：是否将存储库设为私有。如果`None`（默认），则存储库将是公开的，除非组织默认为私有。如果存储库已存在，则忽略此值。

path_in_repo (`str`, *可选*) ：存储库中将推送日志的文件夹的路径。默认为“tensorboard/”。

repo_allow_patterns （`list[str]` 或 `str`，*可选*）：要包含在上传中的模式列表。默认为`"*.tfevents.*"`。查看[upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder)了解更多详情。repo_ignore_patterns（`list[str]`或`str`，*可选*）：要在上传中排除的模式列表。查看[upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder)了解更多详情。

令牌（`str`，*可选*）：身份验证令牌。将默认为存储的令牌。请参阅 https://huggingface.co/settings/token 了解更多详细信息

kwargs ：传递给 `SummaryWriter` 的附加关键字参数。

围绕张量板的 `SummaryWriter` 进行包装，将训练日志推送到 Hub。

数据在本地记录，然后异步推送到集线器。将数据推送到集线器是在单独的过程中完成的
线程以避免阻塞训练脚本。特别是，如果由于任何原因上传失败（例如连接失败）
问题），主脚本不会被中断。数据每隔`commit_every`自动推送到Hub
分钟（默认为每 5 分钟一次）。

> [!警告]
> `HFSummaryWriter` 是实验性的。其 API 将来可能会发生更改，恕不另行通知。

示例：
```diff
# Taken from https://pytorch.org/docs/stable/tensorboard.html
- from torch.utils.tensorboard import SummaryWriter
+ from huggingface_hub import HFSummaryWriter

import numpy as np

- writer = SummaryWriter()
+ writer = HFSummaryWriter(repo_id="username/my-trained-model")

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

```py
>>> from huggingface_hub import HFSummaryWriter

# Logs are automatically pushed every 15 minutes (5 by default) + when exiting the context manager
>>> with HFSummaryWriter(repo_id="test_hf_logger", commit_every=15) as logger:
...     logger.add_scalar("a", 1)
...     logger.add_scalar("b", 2)
```

### 管理集合
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/collections.md