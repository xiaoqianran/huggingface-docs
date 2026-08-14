<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 小部件

## 什么是小部件？

许多模型存储库都有一个小部件，允许任何人直接在浏览器中运行推理。这些小部件由 [Inference Providers](https://huggingface.co/docs/inference-providers) 提供支持，在我们的无服务器推理合作伙伴的支持下，它为开发人员提供了对数百种机器学习模型的简化、统一的访问。

以下是当前流行型号的一些示例：

- [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) - 最先进的开放权重对话模型
- [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) - 用于图像编辑的开放权重变压器模型
- [Falconsai's NSFW Detection](https://huggingface.co/Falconsai/nsfw_image_detection) - 图像内容审核
- [ResembleAI's Chatterbox](https://huggingface.co/ResembleAI/chatterbox) - 生产级开源文本转语音模型。

您可以在 [models page](https://huggingface.co/models?inference_provider=all&sort=trending) 上探索更多模型及其小部件，或在 [Inference Playground](https://huggingface.co/playground) 中交互式尝试它们。

## 启用小部件

当模型由至少一个推理提供程序托管时，将显示小部件，以确保模型推理的最佳性能和可靠性。提供商自主选择和控制他们部署的模型。显示的小部件类型（文本生成、文本到图像等）是从模型的 `pipeline_tag` 推断出来的，这是 Hub 尝试为所有模型自动计算的特殊标签。唯一的例外是`conversational`小部件，它显示在`pipeline_tag`为`text-generation`或`image-text-to-text`的模型上，只要它们也被标记为`conversational`。为了简单起见，我们选择每个模型**仅公开一个**小部件。

对于某些库，例如`transformers`，可以根据配置文件（`config.json`）自动推断模型类型。架构可以确定类型：例如，`AutoModelForTokenClassification`对应于`token-classification`。如果您对此感兴趣，可以在[this gist](https://gist.github.com/julien-c/857ba86a6c6a895ecd90e7f7cab48046)中查看伪代码。

对于大多数其他用例，我们使用模型标签来确定模型任务类型。例如，如果[model card metadata](./model-cards)中有`tag: text-classification`，则推断出的`pipeline_tag`将为`text-classification`。

**您始终可以在 [model card metadata](./model-cards#model-card-metadata) 中使用 `pipeline_tag: xxx` 手动覆盖管道类型。**（您也可以使用元数据 GUI 编辑器来执行此操作）。

### 如何控制模型的小部件示例输入？

您可以在模型卡元数据部分指定小部件输入：

```yaml
widget:
  - text: "This new restaurant has amazing food and great service!"
    example_title: "Positive Review"
  - text: "I'm really disappointed with this product. Poor quality and overpriced."
    example_title: "Negative Review"
  - text: "The weather is nice today."
    example_title: "Neutral Statement"
```您可以提供多个示例输入。在小部件的示例下拉菜单中，它们将显示为 `Example 1`、`Example 2` 等。您也可以选择提供 `example_title`。

```yaml
widget:
  - text: "Is this review positive or negative? Review: Best cast iron skillet you will ever buy."
    example_title: "Sentiment analysis"
  - text: "Barack Obama nominated Hilary Clinton as his secretary of state on Monday. He chose her because she had ..."
    example_title: "Coreference resolution"
  - text: "On a shelf, there are five books: a gray book, a red book, a purple book, a blue book, and a black book ..."
    example_title: "Logic puzzles"
  - text: "The two men running to become New York City's next mayor will face off in their first debate Wednesday night ..."
    example_title: "Reading comprehension"
```

此外，您可以在模型卡元数据中指定非文本示例输入。有关所有小部件类型的示例输入格式的完整列表，请参阅[here](./models-widgets-examples)。对于视觉和音频小部件类型，请提供带有 `src` 而不是 `text` 的示例输入。

例如，允许用户通过以下方式从两个示例音频文件中进行选择以执行自动语音识别任务：

```yaml
widget:
  - src: https://example.org/somewhere/speech_samples/sample1.flac
    example_title: Speech sample 1
  - src: https://example.org/somewhere/speech_samples/sample2.flac
    example_title: Speech sample 2
```

请注意，您还可以在模型存储库中包含示例文件并使用
他们是：

```yaml
widget:
  - src: https://huggingface.co/username/model_repo/resolve/main/sample1.flac
    example_title: Custom Speech Sample 1
```

但更方便的是，如果文件位于相应的模型存储库中，您可以仅使用存储库中的文件名或文件路径：

```yaml
widget:
  - src: sample1.flac
    example_title: Custom Speech Sample 1
```

或者如果它嵌套在存储库中：

```yaml
widget:
  - src: nested/directory/sample1.flac
```

我们在 [default-widget-inputs.ts file](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/default-widget-inputs.ts) 中提供了某些语言和大多数小部件类型的示例输入。如果缺少一些示例，我们欢迎社区 PR 补充！

## 输出示例

作为示例输入的扩展，对于每个小部件示例，您还可以选择直接在 `output` 属性中描述相应的模型输出。当推理提供程序尚不支持模型时，这非常有用，以便模型页面仍然可以展示模型的工作原理及其给出的结果。

例如，对于 [automatic-speech-recognition](./models-widgets-examples#automatic-speech-recognition) 型号：

```yaml
widget:
  - src: sample1.flac
    output:
      text: "Hello my name is Julien"
```

`output` 属性应该是一个 YAML 字典，表示推理提供程序的输出格式。

对于输出文本的模型，请参见上面的示例。

对于输出标签的模型（例如 [text-classification](./models-widgets-examples#text-classification) 模型），输出应如下所示：

```yaml
widget:
  - text: "I liked this movie"
    output:
      - label: POSITIVE
        score: 0.8
      - label: NEGATIVE
        score: 0.2
```

最后，对于输出图像、音频或任何其他类型资产的模型，输出应包含链接到存储库内的文件名或路径或远程 URL 的 `url` 属性。例如，对于文本到图像模型：

```yaml
widget:
  - text: "picture of a futuristic tiger, artstation"
    output:
      url: images/tiger.jpg
```

我们还可以在 Hugging Face UI 中显示示例输出，例如，用于文本到图像模型来显示酷图像生成的图库。

## 小部件可用性和提供商支持

并非所有型号都有可用的小部件。小部件的可用性取决于：1. **任务支持**：模型的任务必须得到推理提供商网络中至少一个提供商的支持
2. **提供商可用性**：至少有一个提供商必须为特定模型提供服务
3. **模型配置**：模型必须具有正确的元数据和配置文件

要查看支持的任务的完整列表，请查看[our dedicated documentation page](https://huggingface.co/docs/inference-providers/tasks/index)。

所有提供者及其支持的任务的列表可在 [this documentation page](https://huggingface.co/docs/inference-providers/index#partners) 中找到。

对于没有提供商支持的模型，您仍然可以在模型卡中使用 [example outputs](#example-outputs) 展示功能。

如果社区有足够的兴趣，您还可以直接在模型页面上单击_请求提供商支持_，以鼓励提供商为模型提供服务。

## 使用推理游乐场探索模型

在将模型集成到您的应用程序之前，您可以使用[Inference Playground](https://huggingface.co/playground)以交互方式测试它们。游乐场允许您：

- 使用自定义提示测试不同的[chat completion models](https://huggingface.co/models?inference_provider=all&sort=trending&other=conversational)
- 比较不同模型的响应
- 试验温度、最大令牌等推理参数
- 找到适合您特定用例的完美型号Playground 使用为小部件提供支持的相同推理提供程序基础设施，因此当您将模型集成到自己的应用程序中时，您可以期待类似的性能和功能。

### 在 Data Studio 中编辑数据集
https://huggingface.co/docs/hub/datasets-cell-editing.md