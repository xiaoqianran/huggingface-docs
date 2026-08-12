<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 网络数据集

[WebDataset](https://github.com/webdataset/webdataset)是一个为大型数据集编写I/O管道的库。
其顺序 I/O 和分片功能使其对于将大规模数据集流式传输到 DataLoader 特别有用。

## WebDataset 格式

WebDataset 文件是包含一系列数据文件的 TAR 存档。
具有相同前缀的所有连续数据文件都被视为同一示例的一部分（例如，图像/音频文件及其标签或元数据）：

标签和元数据可以位于 `.json` 文件中、`.txt`（用于标题、描述）或`.cls`（用于类索引）中。

大型 WebDataset 由许多称为分片的文件组成，其中每个分片都是一个 TAR 存档。
每个分片通常约为 1GB，但完整数据集可能有多个 TB！

## 多模式支持

WebDataset 专为多模式数据集而设计，即图像、音频和/或视频数据集。

事实上，由于媒体文件往往很大，WebDataset 的顺序 I/O 可以实现大量读取和缓冲，从而实现最佳的数据加载速度。

以下是支持的数据格式的非详尽列表：

- 图片：jpeg、png、tiff
- 音频：mp3、m4a、wav、flac
- 视频：mp4、mov、avi
- 其他：npy、npz完整列表会随着时间的推移而变化，并取决于实施情况。例如，您可以在源代码[here](https://github.com/webdataset/webdataset/blob/main/src/webdataset/autodecode.py)中找到`webdataset`包支持哪些格式。

## 流媒体

流式 TAR 存档速度很快，因为它读取连续的数据块。
它比逐个读取单独的数据文件要快几个数量级。

WebDataset 流在从磁盘和云存储读取时都提供高速性能，这使其成为馈送到 DataLoader 的理想格式：

例如，以下是如何直接从 Hugging Face 流式传输 [timm/imagenet-12k-wds](https://huggingface.co/datasets/timm/imagenet-12k-wds) 数据集：

首先你需要[Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)，例如使用：

```
hf auth login
```

然后您可以使用 WebDataset 流式传输数据集：

```python
>>> import webdataset as wds
>>> from huggingface_hub import get_token
>>> from torch.utils.data import DataLoader

>>> hf_token = get_token()
>>> url = "https://huggingface.co/datasets/timm/imagenet-12k-wds/resolve/main/imagenet12k-train-{{0000..1023}}.tar"
>>> url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
>>> dataset = wds.WebDataset(url).decode()
>>> dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
```

## 随机播放

一般来说，WebDataset 格式的数据集已经被打乱并准备好提供给 DataLoader。
但您仍然可以使用 WebDataset 的近似洗牌来重新洗牌数据。

除了打乱分片列表之外，WebDataset 还使用缓冲区来打乱数据集，而无需任何速度成本：

要打乱分片文件列表并从打乱缓冲区中随机采样：

```python
>>> buffer_size = 1000
>>> dataset = (
...     wds.WebDataset(url, shardshuffle=True)
...     .shuffle(buffer_size)
...     .decode()
... )
```

### 审核日志
https://huggingface.co/docs/hub/audit-logs.md