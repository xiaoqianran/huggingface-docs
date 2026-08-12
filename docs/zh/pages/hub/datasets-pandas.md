<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 熊猫

[Pandas](https://github.com/pandas-dev/pandas)是一个广泛使用的Python数据分析工具包。
由于它使用[fsspec](https://filesystem-spec.readthedocs.io)来读写远程数据，因此您可以使用Hugging Face路径（[⟦T20⟧](/docs/huggingface_hub/guides/hf_file_system#integrations)）在Hub上读写数据。

## 加载数据框

您可以从本地文件或远程存储（例如拥抱脸部数据集）加载数据。 Pandas 支持多种格式，包括 CSV、JSON 和 Parquet：

```python
>>> import pandas as pd
>>> df = pd.read_csv("path/to/data.csv")
```

要从 Hugging Face 加载文件，路径需要以 `hf://` 开头。例如，[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)数据集存储库的路径是`hf://datasets/stanfordnlp/imdb`。 Hugging Face 上的数据集包含多个 Parquet 文件。 Parquet 文件格式旨在提高数据帧的读写效率，并使跨数据分析语言共享数据变得容易。以下是加载文件`plain_text/train-00000-of-00001.parquet`的方法：

```python
>>> import pandas as pd
>>> df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
>>> df
                                                    text  label
0      I rented I AM CURIOUS-YELLOW from my video sto...      0
1      "I Am Curious: Yellow" is a risible and preten...      0
2      If only to avoid making this type of film in t...      0
3      This film was probably inspired by Godard's Ma...      0
4      Oh, brother...after hearing about this ridicul...      0
...                                                  ...    ...
24995  A hit at the time but now better categorised a...      1
24996  I love this movie like no other. Another time ...      1
24997  This film and it's sequel Barry Mckenzie holds...      1
24998  'The Adventures Of Barry McKenzie' started lif...      1
24999  The story centers around Barry McKenzie who mu...      1
```

有关 Hugging Face 路径及其实现方式的更多信息，请参阅[the client library's documentation on the HfFileSystem](/docs/huggingface_hub/guides/hf_file_system)。

> [!提示]
> 相同的 `hf://` 路径也适用于 [Storage Buckets](./storage-buckets)：
> ```python
> >>> df = pd.read_parquet("hf://buckets/username/my-bucket/data.parquet")
> >>> df.to_parquet("hf://buckets/username/my-bucket/output.parquet")
> ```

## 保存数据框

您可以使用 `to_csv/to_json/to_parquet` 将 pandas DataFrame 保存到本地文件或直接保存到 Hugging Face。

要保存 Hugging Face 上的 DataFrame，首先需要[Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)，例如使用：

```
hf auth login
```

然后你可以[Create a dataset repository](/docs/huggingface_hub/quick-start#create-a-repository)，例如使用：

```python
from huggingface_hub import HfApi

HfApi().create_repo(repo_id="username/my_dataset", repo_type="dataset")
```最后，您可以在 Pandas 中使用 [Hugging Face paths](/docs/huggingface_hub/guides/hf_file_system#integrations)：

```python
import pandas as pd

df.to_parquet("hf://datasets/username/my_dataset/imdb.parquet")

# or write in separate files if the dataset has train/validation/test splits
df_train.to_parquet("hf://datasets/username/my_dataset/train.parquet")
df_valid.to_parquet("hf://datasets/username/my_dataset/validation.parquet")
df_test .to_parquet("hf://datasets/username/my_dataset/test.parquet")
```

请注意，Hugging Face 上的 Parquet 文件经过优化，可提高存储效率、加速下载和上传，并实现高效的数据集流式传输和编辑：

* [Parquet Content Defined Chunking](https://huggingface.co/blog/parquet-cdc) 针对 Hugging Face 的存储后端[Xet](https://huggingface.co/docs/hub/en/xet/index) 优化了 Parquet。由于基于块的重复数据删除，它可以加速上传和下载，并允许高效的文件编辑
* 页面索引可在流式传输时加速过滤器并实现高效的随机访问，例如在[Dataset Viewer](https://huggingface.co/docs/dataset-viewer)

Pandas 需要额外的参数来编写优化的 Parquet 文件：

```python
import pandas as pd

df.to_parquet(
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)
```

* `use_content_defined_chunking=True` 为 [deduplication](https://huggingface.co/blog/parquet-cdc) 和 [editing](./datasets-editing) 启用 Parquet 内容定义分块（需要 `pyarrow>=21.0`）
* `write_page_index=True` 在 Parquet 元数据中包含页面索引，对于 [streaming and random access](./datasets-streaming)

> [!提示]
> 内容定义分块 (CDC) 使 Parquet 编写器以相同方式对重复数据进行分块和压缩的方式对数据页进行分块。
> 如果没有 CDC，页面会被任意分块，因此由于压缩而无法检测到重复数据。
> 感谢 CDC，Hugging Face 的 Parquet 上传和下载速度更快，因为重复数据仅上传或下载一次。

了解有关 Xet [here](https://huggingface.co/join/xet) 的更多信息。## 利用 Xet 重复数据删除技术进行 Parquet

优化的 Parquet 文件是使用内容定义分块写入的，可实现重复数据删除。
这可以加速上传，因为 Hugging Face 上已经存在的数据块不需要再次上传，从而节省了大量的 I/O。

例如，此代码上传`df`的内容，然后对于`edited_df`，上传速度更快，因为它只上传更改的块：

```python
import pandas as pd

df.to_parquet(
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)

edited_df = ...  # e.g. with added/modified/removed rows or columns

edited_df.to_parquet(
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)
```

块约为 64kB，Parquet 每列保存数据列，因此实际上，编辑优化的 Parquet 文件时会发生以下情况：

* 添加一个新列 -> 仅上传新列的块
* 添加/编辑/删除一行 -> 每列上传一个块

除此之外，包含元数据的 Parquet 页脚块也会被上传。

## 使用图像

您可以加载包含元数据文件的文件夹，其中包含图像名称或路径的字段，结构如下：

```
Example 1:            Example 2:
folder/               folder/
├── metadata.csv      ├── metadata.csv
├── img000.png        └── images
├── img001.png            ├── img000.png
...                       ...
└── imgNNN.png            └── imgNNN.png
```

您可以像这样迭代图像路径：

```python
import pandas as pd

folder_path = "path/to/folder/"
df = pd.read_csv(folder_path + "metadata.csv")
for image_path in (folder_path + df["file_name"]):
    ...
```

由于数据集位于 [supported structure](https://huggingface.co/docs/hub/en/datasets-image#additional-columns)（带有 `file_name` 字段的 `metadata.csv` 或 `.jsonl` 文件）中，因此您可以将此数据集保存到 Hugging Face，数据集查看器会显示 Hugging Face 上的元数据和图像。

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=folder_path,
    repo_id="username/my_image_dataset",
    repo_type="dataset",
)
```### 图像方法和 Parquet

使用 [pandas-image-methods](https://github.com/lhoestq/pandas-image-methods) 可以在图像列上启用 `PIL.Image` 方法。它还可以将数据集保存为包含图像和元数据的单个 Parquet 文件：

```python
import pandas as pd
from pandas_image_methods import PILMethods

pd.api.extensions.register_series_accessor("pil")(PILMethods)

df["image"] = (folder_path + df["file_name"]).pil.open()
df.to_parquet("data.parquet")
```

所有`PIL.Image`方法都可用，例如

```python
df["image"] = df["image"].pil.rotate(90)
```

## 使用音频

您可以加载包含元数据文件的文件夹，其中包含音频名称或路径的字段，结构如下：

```
Example 1:            Example 2:
folder/               folder/
├── metadata.csv      ├── metadata.csv
├── rec000.wav        └── audios
├── rec001.wav            ├── rec000.wav
...                       ...
└── recNNN.wav            └── recNNN.wav
```

您可以像这样迭代音频路径：

```python
import pandas as pd

folder_path = "path/to/folder/"
df = pd.read_csv(folder_path + "metadata.csv")
for audio_path in (folder_path + df["file_name"]):
    ...
```

由于数据集位于 [supported structure](https://huggingface.co/docs/hub/en/datasets-audio#additional-columns)（带有 `file_name` 字段的 `metadata.csv` 或 `.jsonl` 文件）中，因此您可以将其保存到 Hugging Face，并且 Hub 数据集查看器会同时显示元数据和音频。 

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=folder_path,
    repo_id="username/my_audio_dataset",
    repo_type="dataset",
)
```

### 音频方法和 Parquet

使用 [pandas-audio-methods](https://github.com/lhoestq/pandas-audio-methods) 您可以在音频列上启用 `soundfile` 方法。它还可以将数据集保存为包含音频和元数据的单个 Parquet 文件：

```python
import pandas as pd
from pandas_audio_methods import SFMethods

pd.api.extensions.register_series_accessor("sf")(SFMethods)

df["audio"] = (folder_path + df["file_name"]).sf.open()
df.to_parquet("data.parquet")
```

这使得与 `librosa` 一起使用变得很容易，例如用于重采样：

```python
df["audio"] = [librosa.load(audio, sr=16_000) for audio in df["audio"]]
df["audio"] = df["audio"].sf.write()
```

## 使用变形金刚

您可以在 pandas DataFrames 上使用 `transformers` 管道来分类、生成文本、图像等。
本节展示了一些使用 `tqdm` 作为进度条的示例。

> [!提示]
> 管道不接受 `tqdm` 对象作为输入，但您可以使用 python 生成器，格式为 `x for x in tqdm(...)`### 文本分类

```python
from transformers import pipeline
from tqdm import tqdm

pipe = pipeline("text-classification", model="clapAI/modernBERT-base-multilingual-sentiment")

# Compute labels
df["label"] = [y["label"] for y in pipe(x for x in tqdm(df["text"]))]
# Compute labels and scores
df[["label", "score"]] = [(y["label"], y["score"]) for y in pipe(x for x in tqdm(df["text"]))]
```

### 文本生成

```python
from transformers import pipeline
from tqdm import tqdm

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

# Generate chat response
prompt = "What is the main topic of this sentence ? REPLY IN LESS THAN 3 WORDS. Sentence: '{}'"
df["output"] = [y["generated_text"][1]["content"] for y in pipe([{"role": "user", "content": prompt.format(x)}] for x in tqdm(df["text"]))]
```

### 工作
https://huggingface.co/docs/hub/jobs.md