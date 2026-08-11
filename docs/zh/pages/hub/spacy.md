<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 spaCy

`spaCy` 是一个流行的高级自然语言处理库，广泛应用于整个行业。 `spaCy` 可以轻松使用和训练用于命名实体识别、文本分类、词性标记等任务的管道，并允许您构建强大的应用程序来处理和分析大量文本。

## 在 Hub 中探索 spaCy 模型

`spaCy` 3.3 的官方型号在`spaCy` [Organization Page](https://huggingface.co/spacy)。社区中的任何人都可以分享他们的`spaCy`模型，您可以通过[models page](https://huggingface.co/models?library=spacy)左侧的过滤找到这些模型。

集线器上的所有型号都具有有用的功能
1. 自动生成的模型卡，其中包含标签方案、指标、组件等。
2. 右上角的评估部分，您可以在其中查看指标。
3. 元数据标签有助于发现并包含许可证和语言等信息。
4. 一个交互式小部件，您可以使用它直接在浏览器中玩模型
5. 一个推理提供程序小部件，允许发出推理请求。

## 使用现有模型

Hub 中的所有 `spaCy` 型号都可以使用 pip install 直接安装。

```bash
pip install "en_core_web_sm @ https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl"
```要查找感兴趣的链接，您可以转到具有 `spaCy` 模型的存储库。当您打开存储库时，您可以单击`Use in spaCy`，您将获得一个工作片段，您可以使用它来安装和加载模型！

安装后，您可以将模型加载为任何 spaCy 管道。

```python
# Using spacy.load().
import spacy
nlp = spacy.load("en_core_web_sm")

# Importing as module.
import en_core_web_sm
nlp = en_core_web_sm.load()
```

## 分享你的模型

### 使用 spaCy CLI（推荐）

`spacy-huggingface-hub` 库扩展了 `spaCy` 本机 CLI，因此人们可以轻松地将打包的模型推送到 Hub。

您可以从 pip 安装 spacy-huggingface-hub：

```bash
pip install spacy-huggingface-hub
```

然后可以检查命令是否注册成功

```bash
python -m spacy huggingface-hub --help
```

要使用 CLI 进行推送，您可以使用 `huggingface-hub push` 命令，如下所示。

```bash
python -m spacy huggingface-hub push [whl_path] [--org] [--msg] [--local-repo] [--verbose]
```|论证|类型 |描述 |
| -------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `whl_path` | str / `Path` | [⟦T21⟧](https://spacy.io/api/cli#package)打包的`.whl`文件的路径。                                |
| `--org`、`-o` | STR |管道应上传到的组织的可选名称。                                                       |
| `--msg`、`-m` | STR |提交用于更新的消息。默认为`"Update spaCy pipeline"`。                                                      |
| `--local-repo`、`-l` | str / `Path` |模型存储库的本地路径（如果不存在则将创建）。默认为当前工作目录中的`hub`。 |
| `--verbose`、`-V` |布尔 |输出用于调试的附加信息，例如完整生成的集线器元数据。                                                   |然后，您可以上传使用[⟦T33⟧](https://spacy.io/api/cli#package)打包的任何管道。确保设置 `--build wheel` 以输出二进制 .whl 文件。上传器将从管道包中读取所有元数据，包括自动生成的漂亮`README.md`和`meta.json`中提供的模型详细信息。

```bash
hf auth login
python -m spacy package ./en_ner_fashion ./output --build wheel
cd ./output/en_ner_fashion-0.0.0/dist
python -m spacy huggingface-hub push en_ner_fashion-0.0.0-py3-none-any.whl
```

只需一分钟，您就可以在 Hub 中获取打包的模型，直接在浏览器中试用，并与社区其他成员共享。所有必需的元数据都会为您上传，您甚至可以获得一张很酷的模型卡。

该命令将输出两件事：

* 在 Hub 中哪里可以找到您的存储库！例如，https://huggingface.co/spacy/en_core_web_sm
* 以及如何直接从 Hub 安装管道！

### 来自 Python 脚本

您可以使用 Python 中的 `push` 函数。它返回一个字典，其中包含已发布模型的`"url"`和“`whl_url`”以及wheel文件，您稍后可以使用`pip install`进行安装。

```py
from spacy_huggingface_hub import push

result = push("./en_ner_fashion-0.0.0-py3-none-any.whl")
print(result["url"])
```

## 其他资源

* spacy-huggingface-hub [library](https://github.com/explosion/spacy-huggingface-hub)。
* 推出[blog post](https://huggingface.co/blog/spacy)
* spaCy v 3.1 [Announcement](https://explosion.ai/blog/spacy-v3-1#huggingface-hub)
* 斯帕西[documentation](https://spacy.io/universe/project/spacy-huggingface-hub/)

### 存储库入门
https://huggingface.co/docs/hub/repositories-getting-started.md