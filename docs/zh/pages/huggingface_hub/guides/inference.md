<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在服务器上运行推理

推理是使用经过训练的模型对新数据进行预测的过程。由于此过程可能是计算密集型的，因此在专用或外部服务上运行可能是一个有趣的选择。
`huggingface_hub` 库提供了一个统一的接口，可以为 Hugging Face Hub 上托管的模型跨多个服务运行推理：

1. [Inference Providers](https://huggingface.co/docs/inference-providers/index)：对数百种机器学习模型的简化、统一访问，由我们的无服务器推理合作伙伴提供支持。这种新方法建立在我们之前的无服务器推理 API 的基础上，在世界一流的提供商的帮助下，提供了更多的模型、改进的性能和更高的可靠性。请参阅 [documentation](https://huggingface.co/docs/inference-providers/index#partners) 了解受支持的提供商列表。
2. [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)：一款轻松将模型部署到生产环境的产品。推理由 Hugging Face 在您选择的云提供商的专用、完全托管的基础设施中运行。
3. 本地端点：您还可以通过将客户端连接到本地端点来使用[llama.cpp](https://github.com/ggerganov/llama.cpp)、[Ollama](https://ollama.com/)、[vLLM](https://github.com/vllm-project/vllm)、[LiteLLM](https://docs.litellm.ai/docs/simple_proxy)或[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)等本地推理服务器运行推理。> [!提示]
> [InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 是一个 Python 客户端，对我们的 API 进行 HTTP 调用。如果你想直接使用 HTTP 调用
> 您首选的工具（curl、postman...），请参阅[Inference Providers](https://huggingface.co/docs/inference-providers/index)文档
> 或访问 [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) 文档页面。
>
> 对于 Web 开发，[JS client](https://huggingface.co/docs/huggingface.js/inference/README) 已发布。
> 如果您对游戏开发感兴趣，您可以看看我们的[C# project](https://github.com/huggingface/unity-api)。

## 开始使用

让我们开始执行文本到图像的任务：

```python
>>> from huggingface_hub import InferenceClient

# Example with an external provider (e.g. replicate)
>>> replicate_client = InferenceClient(
    provider="replicate",
    api_key="REDACTED",
)
>>> replicate_image = replicate_client.text_to_image(
    "A flying car crossing a futuristic cityscape.",
    model="black-forest-labs/FLUX.1-schnell",
)
>>> replicate_image.save("flying_car.png")

```

在上面的示例中，我们使用第三方提供商[Replicate](https://replicate.com/)初始化了[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)。使用提供程序时，您必须指定要使用的模型。模型id必须是Hugging Face Hub上的模型id，而不是第三方提供商的模型id。
在我们的示例中，我们根据文本提示生成了图像。返回值是一个 `PIL.Image` 对象，可以保存到文件中。有关更多详细信息，请查看 [text_to_image()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_to_image) 文档。

现在让我们看一个使用 [chat_completion()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion) API 的示例。此任务使用 LLM 从消息列表生成响应：

```python
>>> from huggingface_hub import InferenceClient
>>> messages = [
    {
        "role": "user",
        "content": "What is the capital of France?",
    }
]
>>> client = InferenceClient(
    provider="together",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="REDACTED",
)
>>> client.chat_completion(messages, max_tokens=100)
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason="eos_token",
            index=0,
            message=ChatCompletionOutputMessage(
                role="assistant", content="The capital of France is Paris.", name=None, tool_calls=None
            ),
            logprobs=None,
        )
    ],
    created=1719907176,
    id="",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    object="text_completion",
    system_fingerprint="2.0.4-sha-f426a33",
    usage=ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=17, total_tokens=25),
)
```在上面的示例中，我们使用了第三方提供商 ([Together AI](https://www.together.ai/)) 并指定了我们要使用的模型 (`"meta-llama/Meta-Llama-3-8B-Instruct"`)。然后，我们给出了要完成的消息列表（此处是一个问题），并向 API 传递了一个附加参数 (`max_token=100`)。输出是遵循 OpenAI 规范的 `ChatCompletionOutput` 对象。生成的内容可以通过`output.choices[0].message.content`访问。有关更多详细信息，请查看 [chat_completion()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion) 文档。

> [!警告]
> API 设计得很简单。并非所有参数和选项都可供最终用户使用或描述。退房
> [this page](https://huggingface.co/docs/api-inference/detailed_parameters) 如果您有兴趣了解更多
> 每个任务可用的所有参数。

### 使用特定的提供商

如果要使用特定的提供者，可以在初始化客户端时指定。默认值为“auto”，它将选择模型可用的第一个提供程序，按用户在 https://hf.co/settings/inference-providers 中的顺序排序。请参阅 [Supported providers and tasks](#supported-providers-and-tasks) 部分以获取受支持的提供商列表。

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="replicate", api_key="REDACTED")
```

### 使用特定模型

如果您想使用特定模型怎么办？您可以将其指定为参数或直接在实例级别指定：

```python
>>> from huggingface_hub import InferenceClient
# Initialize client for a specific model
>>> client = InferenceClient(provider="together", model="meta-llama/Llama-3.1-8B-Instruct")
>>> client.text_to_image(...)
# Or use a generic client but pass your model as an argument
>>> client = InferenceClient(provider="together")
>>> client.text_to_image(..., model="meta-llama/Llama-3.1-8B-Instruct")
```> [!提示]
> 使用“hf-inference”提供程序时，每个任务都会附带 Hub 上 100 万多个可用模型中的推荐模型。
> 然而，这一建议可能会随着时间的推移而改变，因此最好在决定使用哪个模型后明确设置一个模型。
> 对于第三方提供商，您必须始终指定与该提供商兼容的型号。
>
> 访问中心上的 [Models](https://huggingface.co/models?inference=warm) 页面，探索通过推理提供程序可用的模型。

### 使用推理端点

我们上面看到的示例使用推理提供程序。虽然这些被证明对于原型设计非常有用
并快速测试。一旦准备好将模型部署到生产中，您将需要使用专用基础设施。
这就是[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)发挥作用的地方。它允许您部署
任何模型并将其公开为私有 API。部署后，您将获得一个可以使用完全相同的 URL 进行连接的 URL
代码与以前一样，仅更改 `model` 参数：

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
# or
>>> client = InferenceClient()
>>> client.text_to_image(..., model="https://uu149rez6gw9ehej.eu-west-1.aws.endpoints.huggingface.cloud/deepfloyd-if")
```

请注意，您不能同时指定 URL 和提供程序 - 它们是互斥的。 URL 用于直接连接到已部署的端点。

### 使用本地端点您可以使用 [InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 与在您自己的计算机上运行的本地推理服务器（llama.cpp、vllm、litellm 服务器、TGI、mlx 等）运行聊天完成。 API 应与 OpenAI API 兼容。

```python
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(model="http://localhost:8080")

>>> response = client.chat.completions.create(
...     messages=[
...         {"role": "user", "content": "What is the capital of France?"}
...     ],
...     max_tokens=100
... )
>>> print(response.choices[0].message.content)
```

> [!提示]
> 与 OpenAI Python 客户端类似，[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 可用于通过任何 OpenAI REST API 兼容端点运行聊天完成推理。

### 身份验证

身份验证可以通过两种方式完成：

**通过 Hugging Face 路由** ：使用 Hugging Face 作为代理访问第三方提供商。呼叫将使用我们的提供商密钥通过 Hugging Face 的基础设施进行路由，并且使用量将直接记入您的 Hugging Face 帐户。

您可以使用[User Access Token](https://huggingface.co/docs/hub/security-tokens)进行身份验证。您可以使用 `api_key` 参数直接提供您的 Hugging Face 令牌：

```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="hf_****"  # Your HF token
)
```

如果您*不*传递 `api_key`，客户端将尝试查找并使用本地存储在您计算机上的令牌。如果您之前已登录，通常会发生这种情况。有关登录的详细信息，请参阅[Authentication Guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)。

```python
>>> client = InferenceClient(
    provider="replicate",
    token="hf_****"  # Your HF token
)
```

**直接访问提供商**：使用您自己的 API 密钥直接与提供商的服务交互：
```python
>>> client = InferenceClient(
    provider="replicate",
    api_key="r8_****"  # Your Replicate API key
)
```

欲了解更多详情，请参阅[Inference Providers pricing documentation](https://huggingface.co/docs/inference-providers/pricing#routed-requests-vs-direct-calls)。

## 支持的提供者和任务[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 的目标是提供最简单的接口来在任何提供商的 Hugging Face 模型上运行推理。它有一个简单的 API，支持最常见的任务。下表显示了哪些提供程序支持哪些任务：

|任务|巴斯坦|大脑 |连贯|深基础设施 |法艾 |无羽人工智能 |烟花人工智能|格罗克 |高频推理 |诺维塔人工智能 |恩斯卡尔 | OVHcloud AI 端点 |公共人工智能|复制|斯卡威|一起|波速|在|
| --------------------------------------------------- | -------- | -------- | ------ | --------- | ------ | -------------- | ------------ | ---- | ------------ | --------- | ------ | -------------------- | --------- | --------- | -------- | -------- | --------- | --- |
| [audio_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.audio_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ || [audio_to_audio()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.audio_to_audio) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [automatic_speech_recognition()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.automatic_speech_recognition) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [chat_completion()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| [document_question_answering()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.document_question_answering) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [feature_extraction()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.feature_extraction) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ || [fill_mask()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.fill_mask) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [image_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.image_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [image_segmentation()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.image_segmentation) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [image_to_image()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.image_to_image) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [image_to_video()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.image_to_video) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ || [image_to_text()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.image_to_text) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [object_detection()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.object_detection) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [question_answering()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.question_answering) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [sentence_similarity()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.sentence_similarity) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [summarization()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.summarization) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ || [table_question_answering()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.table_question_answering) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [text_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [text_generation()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation) | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| [text_to_image()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_to_image) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [text_to_speech()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_to_speech) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ || [text_to_video()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_to_video) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| [tabular_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.tabular_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [tabular_regression()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.tabular_regression) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [token_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.token_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [translation()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.translation) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ || [visual_question_answering()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.visual_question_answering) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [zero_shot_image_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.zero_shot_image_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| [zero_shot_classification()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.zero_shot_classification) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

> [!提示]
> 查看 [Tasks](https://huggingface.co/tasks) 页面以了解有关每个任务的更多信息。

## OpenAI 兼容性

`chat_completion` 任务遵循 [OpenAI's Python client](https://github.com/openai/openai-python) 语法。这对你来说意味着什么？这意味着，如果您习惯使用`OpenAI`的API，您只需更新2行代码就可以切换到`huggingface_hub.InferenceClient`来使用开源模型！

```diff
- from openai import OpenAI
+ from huggingface_hub import InferenceClient

- client = OpenAI(
+ client = InferenceClient(
    base_url=...,
    api_key=...,
)

output = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```就是这样！唯一需要的更改是将 `from openai import OpenAI` 替换为 `from huggingface_hub import InferenceClient`，将 `client = OpenAI(...)` 替换为 `client = InferenceClient(...)`。您可以通过将模型 ID 作为 `model` 参数传递，从 Hugging Face Hub 中选择任何 LLM 模型。 [Here is a list](https://huggingface.co/models?pipeline_tag=text-generation&other=conversational,text-generation-inference&sort=trending) 支持的型号。对于身份验证，您应该传递有效的 [User Access Token](https://huggingface.co/settings/tokens) 作为 `api_key` 或使用 `huggingface_hub` 进行身份验证（请参阅 [authentication guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)）。

所有输入参数和输出格式都严格相同。特别是，您可以传递 `stream=True` 来接收生成的令牌。您还可以使用 [AsyncInferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient) 使用 `asyncio` 运行推理：

```diff
import asyncio
- from openai import AsyncOpenAI
+ from huggingface_hub import AsyncInferenceClient

- client = AsyncOpenAI()
+ client = AsyncInferenceClient()

async def main():
    stream = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

asyncio.run(main())
```你可能想知道为什么使用[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)而不是OpenAI的客户端？原因有以下几个：
1. [InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)配置为Hugging Face服务。您不需要提供 `base_url` 即可使用推理提供程序运行模型。如果您的计算机已正确登录，您也不需要提供 `token` 或 `api_key`。
2. [InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 专为文本生成推理 (TGI) 和 `transformers` 框架量身定制，这意味着您可以放心，它将始终与最新更新保持一致。
3. [InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 与我们的推理端点服务集成，可以更轻松地启动推理端点、检查其状态并在其上运行推理。查看 [Inference Endpoints](./inference_endpoints) 指南了解更多详细信息。

> [!提示]
> `InferenceClient.chat.completions.create` 只是 `InferenceClient.chat_completion` 的别名。查看[chat_completion()](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion)的封装参考了解更多详情。实例化客户端时的`base_url`和`api_key`参数也是`model`和`token`的别名。定义这些别名是为了减少从 `OpenAI` 切换到 `InferenceClient` 时的摩擦。

## 函数调用函数调用允许 LLM 与外部工具交互，例如定义的函数或 API。这使用户能够轻松构建针对特定用例和实际任务定制的应用程序。
`InferenceClient` 实现与 OpenAI Chat Completions API 相同的工具调用接口。这是使用 [Novita](https://novita.ai/) 作为推理提供者的工具调用的简单示例：

```python
from huggingface_hub import InferenceClient

tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Paris, France"
                        }
                    },
                    "required": ["location"],
                },
            }
        }
]

client = InferenceClient(provider="novita")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
    {
        "role": "user",
        "content": "What's the weather like the next 3 days in London, UK?"
    }
    ],
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.tool_calls[0].function.arguments)

```

> [!提示]
> 请参阅提供商的文档以验证其支持哪些模型进行函数/工具调用。

## 结构化输出和 JSON 模式

InferenceClient 支持 JSON 模式以实现语法上有效的 JSON 响应，并支持结构化输出以实现架构强制响应。 JSON 模式提供没有严格结构的机器可读数据，而结构化输出保证有效的 JSON 并遵守预定义的模式，以实现可靠的下游处理。

我们遵循 JSON 模式和结构化输出的 OpenAI API 规范。您可以通过 `response_format` 参数启用它们。以下是使用 [Cerebras](https://www.cerebras.ai/) 作为推理提供者的结构化输出示例：```python
from huggingface_hub import InferenceClient

json_schema = {
    "name": "book",
    "schema": {
        "properties": {
            "name": {
                "title": "Name",
                "type": "string",
            },
            "authors": {
                "items": {"type": "string"},
                "title": "Authors",
                "type": "array",
            },
        },
        "required": ["name", "authors"],
        "title": "Book",
        "type": "object",
    },
    "strict": True,
}

client = InferenceClient(provider="cerebras")

completion = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[
        {"role": "system", "content": "Extract the books information."},
        {"role": "user", "content": "I recently read 'The Great Gatsby' by F. Scott Fitzgerald."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": json_schema,
    },
)

print(completion.choices[0].message)
```
> [!提示]
> 请参阅提供商的文档以验证他们支持哪些模型的结构化输出和 JSON 模式。

## 异步客户端

还提供了基于`asyncio`和`httpx`的异步版本客户端。所有异步 API 端点均可通过 [AsyncInferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient) 获得。其初始化和 API 与仅同步版本完全相同。

```py
# Code must be run in an asyncio concurrent context.
# $ python -m asyncio
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> async for token in await client.text_generation("The Huggingface Hub is", stream=True):
...     print(token, end="")
 a platform for sharing and discussing ML-related content.
```

有关`asyncio`模块的更多信息，请参阅[official documentation](https://docs.python.org/3/library/asyncio.html)。

## MCP 客户端

`huggingface_hub` 库现在包含一个实验性的 [MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient)，旨在使大型语言模型 (LLM) 能够通过 [Model Context Protocol](https://modelcontextprotocol.io) (MCP) 与外部工具交互。该客户端扩展了[AsyncInferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)以无缝集成工具使用。

[MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient) 连接到公开工具的 MCP 服务器（本地 `stdio` 脚本或远程 `http`/`sse` 服务）。它将这些工具提供给法学硕士（通过[AsyncInferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)）。如果 LLM 决定使用工具，[MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient) 管理对 MCP 服务器的执行请求，并将工具的输出转发回 LLM，通常实时传输结果。在以下示例中，我们通过 [Novita](https://novita.ai/) 推理提供程序使用 [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) 模型。然后，我们添加一个远程 MCP 服务器，在本例中是一个 SSE 服务器，它使 Flux 图像生成工具可供 LLM 使用。

```python
import os

from huggingface_hub import ChatCompletionInputMessage, ChatCompletionStreamOutput, MCPClient

async def main():
    async with MCPClient(
        provider="novita",
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=os.environ["HF_TOKEN"],
    ) as client:
        await client.add_mcp_server(type="sse", url="https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse")

        messages = [
            {
                "role": "user",
                "content": "Generate a picture of a cat on the moon",
            }
        ]

        async for chunk in client.process_single_turn_with_tools(messages):
            # Log messages
            if isinstance(chunk, ChatCompletionStreamOutput):
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="")

            # Or tool calls
            elif isinstance(chunk, ChatCompletionInputMessage):
                print(
                    f"\nCalled tool '{chunk.name}'. Result: '{chunk.content if len(chunk.content) < 1000 else chunk.content[:1000] + '...'}'"
                )

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

为了更简单的开发，我们提供了更高级别的[Agent](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.Agent)类。这个“小代理”通过管理聊天循环和状态来简化会话代理的创建，本质上充当[MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient)的包装器。它被设计为一个简单的 while 循环，构建在 [MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient) 之上。您可以直接从命令行运行这些代理：

```bash
# install latest version of huggingface_hub with the mcp extra
pip install -U huggingface_hub[mcp]
# Run an agent that uses the Flux image generation tool
tiny-agents run julien-c/flux-schnell-generator

```

启动后，代理将加载并列出从其连接的 MCP 服务器中发现的工具，然后就可以等待您的提示了！

## 高级提示

在上一节中，我们看到了[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)的主要方面。让我们深入了解一些更高级的技巧。

### 计费作为 HF 用户，您每月可以获得积分，可以通过 Hub 上的各种提供商进行推理。您获得的积分金额取决于您的帐户类型（免费、PRO 或 Enterprise Hub）。您需要为每个推理请求付费，具体取决于提供商的定价表。默认情况下，请求将记入您的个人帐户。但是，可以设置计费，以便通过简单地将 `bill_to="<your_org_name>"` 传递到 `InferenceClient` 来向您所属的组织收取请求费用。为此，您的组织必须订阅 Enterprise Hub。有关计费的更多详细信息，请查看[this guide](https://huggingface.co/docs/api-inference/pricing#features-using-inference-providers)。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(provider="fal-ai", bill_to="openai")
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```

请注意，无法向您不属于的其他用户或组织收费。如果您想授予其他人一些学分，您必须与他们创建一个联合组织。

### 超时

推理调用可能会花费大量时间。默认情况下，[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)将“无限期”等待，直到推理完成。如果您想在工作流程中获得更多控制，可以将 `timeout` 参数设置为特定值（以秒为单位）。如果超时延迟到期，则会引发 [InferenceTimeoutError](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError)，您可以在代码中捕获它：

```python
>>> from huggingface_hub import InferenceClient, InferenceTimeoutError
>>> client = InferenceClient(timeout=30)
>>> try:
...     client.text_to_image(...)
... except InferenceTimeoutError:
...     print("Inference timed out after 30s.")
```

### 二进制输入某些任务需要二进制输入，例如在处理图像或音频文件时。在这种情况下，[InferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)
尝试尽可能宽容并接受不同的类型：
- 原始`bytes`
- 类似文件的对象，以二进制形式打开（`with open("audio.flac", "rb") as f: ...`）
- 指向本地文件的路径（`str`或`Path`）
- 指向远程文件的 URL (`str`)（例如 `https://...`）。在这种情况下，文件将先下载到本地
被发送到 API。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
```

### 运行和管理作业
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/jobs.md