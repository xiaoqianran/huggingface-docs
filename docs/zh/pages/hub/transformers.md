<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在抱脸处使用🤗 `transformers`

🤗 `transformers` 是由 Hugging Face 和社区维护的库，用于 Pytorch、TensorFlow 和 JAX 的最先进的机器学习。它提供了数千个预训练模型来执行不同模式（例如文本、视觉和音频）的任务。我们有点偏见，但我们真的很喜欢🤗`transformers`！

## 探索 Hub 中的 🤗 变形金刚

Hub 中有超过 630,000 个`transformers` 模型，您可以通过[the models page](https://huggingface.co/models?library=transformers&sort=downloads) 左侧的过滤找到它们。 

您可以找到许多不同任务的模型：

* 从上下文中提取答案（[question-answering](https://huggingface.co/models?library=transformers&pipeline_tag=question-answering&sort=downloads)）。
* 从大文本中创建摘要 ([summarization](https://huggingface.co/models?library=transformers&pipeline_tag=summarization&sort=downloads))。
* 对文本进行分类（例如垃圾邮件或非垃圾邮件，[text-classification](https://huggingface.co/models?library=transformers&pipeline_tag=text-classification&sort=downloads)）。
* 使用 GPT ([text-generation](https://huggingface.co/models?library=transformers&pipeline_tag=text-generation&sort=downloads)) 等模型生成新文本。
* 识别句子中的词性（动词、主语等）或实体（国家、组织等）（[token-classification](https://huggingface.co/models?library=transformers&pipeline_tag=token-classification&sort=downloads)）。
* 将音频文件转录为文本（[automatic-speech-recognition](https://huggingface.co/models?library=transformers&pipeline_tag=automatic-speech-recognition&sort=downloads)）。
* 对音频文件中的说话者或语言进行分类 ([audio-classification](https://huggingface.co/models?library=transformers&pipeline_tag=audio-classification&sort=downloads))。
* 检测图像中的对象 ([object-detection](https://huggingface.co/models?library=transformers&pipeline_tag=object-detection&sort=downloads))。
* 分割图像（[image-segmentation](https://huggingface.co/models?library=transformers&pipeline_tag=image-segmentation&sort=downloads)）。
* 进行强化学习（[reinforcement-learning](https://huggingface.co/models?library=transformers&pipeline_tag=reinforcement-learning&sort=downloads)）！

如果您想测试模型而无需下载模型，则可以直接在浏览器中试用模型，这要归功于浏览器内的小部件！## Transformers 存储库文件

[Transformers](https://hf.co/docs/transformers/index)模型存储库通常包含模型文件和预处理器文件。

  

### 型号

- **`config.json`** 文件存储有关模型架构的详细信息，例如隐藏层数量、词汇大小、注意力头数量、每个头的尺寸等。该元数据是模型蓝图。
- **`model.safetensors`** 文件存储模型预训练层和权重。对于大型模型，safetensors 文件被分片以限制加载它所需的内存量。浏览 **`model.safetensors.index.json`** 文件以查看从哪个安全张量文件加载模型权重。

  ```json
  {
  "metadata": {
    "total_size": 16060522496
  },
  "weight_map": {
    "lm_head.weight": "model-00004-of-00004.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    ...
    }
  }
  ```

  您还可以通过单击模型卡上的 ↗ 按钮来可视化此映射。

  
    
  

  与 [pickle](./security-pickle#use-your-own-serialization-format) 相比，[Safetensors](https://hf.co/docs/safetensors/index) 是一种更安全、更快速的序列化格式，用于存储模型权重。您可能会遇到以 **`bin`**、**`pth`** 或 **`ckpt`** 等格式腌制的权重，但 **`safetensors`** 在模型生态系统中越来越多地采用作为更好的替代方案。- 模型还可能有一个 **`generation_config.json`** 文件，其中存储有关如何生成文本的详细信息，例如是否采样、采样的顶部标记、温度以及用于启动和停止生成的特殊标记。

### 预处理器

- **`tokenizer_config.json`** 文件存储模型添加的特殊标记。这些特殊标记向模型发出许多信号，例如句子的开头、聊天模板的特定格式或指示图像。该文件还显示模型可以接受的最大输入序列长度、预处理器类及其返回的输出。
- **`tokenizer.json`** 文件存储模型学习的词汇。
- **`special_tokens_map.json`** 是特殊令牌的映射。例如，在[Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/special_tokens_map.json)中，字符串标记的开头是`"<|begin_of_text|>"`。

> [!提示]
> 对于其他模式，`tokenizer_config.json` 文件由 `preprocessor_config.json` 替换。

## 使用现有模型

所有`transformer`型号距离使用仅一步之遥！根据您想要使用它们的方式，您可以使用 `pipeline` 函数来使用高级 API，也可以使用 `AutoModel` 进行更多控制。

```py
# With pipeline, just specify the task and the model id from the Hub.
from transformers import pipeline
pipe = pipeline("text-generation", model="distilbert/distilgpt2")

# If you want more control, you will need to define the tokenizer and model.
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

您还可以从特定版本（基于提交哈希、标签名称或分支）加载模型，如下所示：

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="v2.0.1"  # tag name, or branch name, or commit hash
)
```如果您想了解如何加载特定模型，可以单击`Use in Transformers`，您将获得一个可以加载它的工作片段！如果您需要有关模型架构的更多信息，还可以单击代码片段底部的“阅读模型文档”。

## 分享你的模型

要阅读有关使用 `transformers` 共享模型的所有信息，请参阅官方文档中的 [Share a model](https://huggingface.co/docs/transformers/model_sharing) 指南。

`transformers` 中的许多类，例如模型和分词器，都有一个 `push_to_hub` 方法，可以轻松地将文件上传到存储库。

```py
# Pushing model to your own account
model.push_to_hub("my-awesome-model")

# Pushing your tokenizer
tokenizer.push_to_hub("my-awesome-model")

# Pushing all things after training
trainer.push_to_hub()
```

您还可以做更多事情，因此我们建议您查看 [Share a model](https://huggingface.co/docs/transformers/model_sharing) 指南。

## 其他资源

* 变形金刚[library](https://github.com/huggingface/transformers)。
* 变形金刚[docs](https://huggingface.co/docs/transformers/index)。
* 分享型号[guide](https://huggingface.co/docs/transformers/model_sharing)。

### 门控组集合
https://huggingface.co/docs/hub/enterprise-gating-group-collections.md