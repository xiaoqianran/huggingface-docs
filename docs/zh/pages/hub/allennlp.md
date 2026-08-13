<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 AllenNLP

`allennlp` 是一个 NLP 库，用于针对不同语言任务开发最先进的模型。它为现代 NLP 中的常见组件和模型提供高级抽象和 API。它还提供了一个可扩展的框架，可以轻松运行和管理 NLP 实验。

## 在 Hub 中探索 allennlp

您可以通过[models page](https://huggingface.co/models?library=allennlp)左侧的筛选在Hub上找到`allennlp`型号。

集线器上的所有型号都具有有用的功能
1. 带有自动托管 TensorBoard 跟踪的训练指标选项卡。
2. 有助于发现的元数据标签。
3. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
4. 推理提供程序小部件，允许发出推理请求。

## 使用现有模型

您可以使用 `Predictor` 类加载 Hub 上的现有模型。要实现此目的，请使用 `from_path` 方法并使用 `"hf://"` 前缀和存储库 ID。这是一个端到端的示例。

```py
import allennlp_models
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("hf://allenai/bidaf-elmo")
predictor_input = {
    "passage": "My name is Wolfgang and I live in Berlin", 
    "question": "Where do I live?"
}
predictions = predictor.predict_json(predictor_input)
```

要获取这样的片段，您可以单击右上角的`Use in AllenNLP`，

## 分享你的模型第一步是将模型保存在本地。例如，您可以使用[⟦T9⟧](https://docs.allennlp.org/main/api/models/archival/#archive_model)方法将模型保存为`model.tar.gz`文件。然后，您可以将压缩的模型推送到集线器。当您使用 `allennlp` 训练模型时，该模型会自动序列化，因此您可以将其用作首选选项。

### 使用 AllenNLP CLI

要使用 CLI 进行推送，您可以使用 `allennlp push_to_hf` 命令，如下所示。

```bash
allennlp push_to_hf --repo_name test_allennlp --archive_path model 
```

|论证|类型 |描述 |
|---------------------------------------- |-------------- |-------------------------------------------------------------------------------------------------------------------------------------------- |
| `--repo_name`、`-n` | str / `Path` | Hub 上存储库的名称。                                                                                            	|
| `--organization`、`-o` | STR |管道应上传到的组织的可选名称。                                                       	|| `--serialization-dir`、`-s` | str / `Path` |包含序列化模型的目录的路径。                                                                                  	|
| `--archive-path`、`-a` | str / `Path` |如果您使用压缩模型（例如 model/model.tar.gz）而不是序列化路径，则可以使用此标志。              	|
| `--local-repo-path`、`-l` | str / `Path` |模型存储库的本地路径（如果不存在则将创建）。默认为当前工作目录中的`hub`。 	|
| `--commit-message`、`-c` | STR |提交用于更新的消息。默认为 `"update repository"`。                                                          	|

### 来自 Python 脚本

`push_to_hf`函数与bash脚本具有相同的参数。

```py
from allennlp.common.push_to_hf import push_to_hf

serialization_dir = "path/to/serialization/directory"
push_to_hf(
    repo_name="my_repo_name",
    serialization_dir=serialization_dir,
    local_repo_path=self.local_repo_path
)
```

只需一分钟，您就可以在 Hub 中获取您的模型，直接在浏览器中试用，并与社区其他成员共享。所有必需的元数据将为您上传！

## 其他资源

* AllenNLP [website](https://allenai.org/allennlp)。
* AllenNLP [repository](https://github.com/allenai/allennlp)。

### 数据集
https://huggingface.co/docs/hub/datasets.md