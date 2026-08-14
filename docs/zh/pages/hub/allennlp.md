<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 AllenNLP

`allennlp` is a NLP library for developing state-of-the-art models on different linguistic tasks. It provides high-level abstractions and APIs for common components and models in modern NLP. It also provides an extensible framework that makes it easy to run and manage NLP experiments.

## 在 Hub 中探索 allennlp

You can find `allennlp` models on the Hub by filtering at the left of the [models page](https://huggingface.co/models?library=allennlp).

All models on the Hub come up with useful features
1. A training metrics tab with automatically hosted TensorBoard traces.
2. 有助于发现的元数据标签。
3. An interactive widget you can use to play out with the model directly in the browser.
4. An Inference Providers widget that allows to make inference requests.

## 使用现有模型

You can use the `Predictor` class to load existing models on the Hub. To achieve this, use the `from_path` method and use the `"hf://"` prefix with the repository id.这是一个端到端的示例。

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

To get a snippet such as this, you can click `Use in AllenNLP` at the top right,

## 分享你的模型第一步是将模型保存在本地。 For example, you can use the [⟦T9⟧](https://docs.allennlp.org/main/api/models/archival/#archive_model) method to save the model as a `model.tar.gz` file.然后，您可以将压缩的模型推送到集线器。 When you train a model with `allennlp`, the model is automatically serialized so you can use that as a preferred option.

### 使用 AllenNLP CLI

To push with the CLI, you can use the `allennlp push_to_hf` command as seen below.

```bash
allennlp push_to_hf --repo_name test_allennlp --archive_path model 
```

|论证|类型 | Description                                                                                                                   	|
|---------------------------------------- |-------------- |-------------------------------------------------------------------------------------------------------------------------------------------- |
| `--repo_name`、`-n` | str / `Path` | Hub 上存储库的名称。                                                                                            	|
| `--organization`、`-o` | STR | Optional name of organization to which the pipeline should be uploaded.                                                       	|| `--serialization-dir`、`-s` | str / `Path` |包含序列化模型的目录的路径。                                                                                  	|
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

### 您的第一个 Docker 空间：使用 T5 生成文本
https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo.md