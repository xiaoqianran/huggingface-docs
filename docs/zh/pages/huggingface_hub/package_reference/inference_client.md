<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 推论

推理是使用经过训练的模型对新数据进行预测的过程。由于此过程可能是计算密集型的，因此在专用或外部服务上运行可能是一个有趣的选择。
`huggingface_hub` 库提供了一个统一的接口，可以为 Hugging Face Hub 上托管的模型跨多个服务运行推理：

1. [Inference Providers](https://huggingface.co/docs/inference-providers/index)：对数百种机器学习模型的简化、统一访问，由我们的无服务器推理合作伙伴提供支持。这种新方法建立在我们之前的无服务器推理 API 的基础上，在世界一流的提供商的帮助下，提供了更多的模型、改进的性能和更高的可靠性。请参阅 [documentation](https://huggingface.co/docs/inference-providers/index#partners) 了解受支持的提供商列表。
2. [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)：一款轻松将模型部署到生产环境的产品。推理由 Hugging Face 在您选择的云提供商的专用、完全托管的基础设施中运行。
3. 本地端点：您还可以通过将客户端连接到这些本地端点，使用[llama.cpp](https://github.com/ggerganov/llama.cpp)、[Ollama](https://ollama.com/)、[vLLM](https://github.com/vllm-project/vllm)、[LiteLLM](https://docs.litellm.ai/docs/simple_proxy)或[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)等本地推理服务器运行推理。

这些服务可以通过[InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)对象来调用。请参考[this guide](../guides/inference)
有关如何使用它的更多信息。## 推理客户端[[huggingface_hub.InferenceClient]]

#### Huggingface_hub.InferenceClient[[huggingface_hub.InferenceClient]]

```python
huggingface_hub.InferenceClient(model: str | None = None, provider: typing.Union[typing.Literal['baseten', 'cerebras', 'cohere', 'deepinfra', 'fal-ai', 'featherless-ai', 'fireworks-ai', 'groq', 'hf-inference', 'novita', 'nscale', 'openai', 'ovhcloud', 'publicai', 'replicate', 'scaleway', 'together', 'wavespeed', 'zai-org'], typing.Literal['auto'], NoneType] = None, token: str | None = None, timeout: float | None = None, headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None, bill_to: str | None = None, base_url: str | None = None, api_key: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L123)

**参数：**

model (`str`, `optional`) ：运行推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，例如`meta-llama/Meta-Llama-3-8B-Instruct` 或已部署推理端点的 URL。默认为“无”，在这种情况下，系统会自动为任务选择推荐模型。注：为了更好地兼容OpenAI客户端，`model`已别名为`base_url`。这两个论点是互斥的。如果将 URL 作为 `model` 或 `base_url` 传递以完成聊天，则 `(/v1)/chat/completions` 后缀路径将附加到 URL。

提供者（`str`，*可选*）：用于推理的提供者的名称。可以是 `"baseten"`、`"cerebras"`、`"cohere"`、`"deepinfra"`、`"fal-ai"`、`"featherless-ai"`、`"fireworks-ai"`、`"groq"`、`"hf-inference"`、 `"novita"`、`"nscale"`、`"openai"`、`"ovhcloud"`、`"publicai"`、`"replicate"`、`"scaleway"`、`"together"`、`"wavespeed"` 或`"zai-org"`。默认为“auto”：自动路由，默认为“最快”的提供商；您可以在 https://hf.co/settings/inference-providers 切换到“最便宜”或“首选”提供商订单。如果 model 是 URL 或传递了 `base_url`，则不使用 `provider`。令牌（`str`，*可选*）：拥抱脸部令牌。如果未提供，将默认为本地保存的令牌。注：为了更好地兼容OpenAI客户端，`token`已别名为`api_key`。这两个参数是互斥的并且具有完全相同的行为。

timeout (`float`, `optional`) ：等待服务器响应的最大秒数。默认为 None，这意味着它将循环直到服务器可用。

headers (`dict[str, str]`, `optional`) ：发送到服务器的附加标头。默认情况下，仅发送授权和用户代理标头。该字典中的值将覆盖默认值。

bill_to (`str`, `optional`) ：用于请求的计费帐户。默认情况下，请求在用户帐户上计费。请求只能向用户所属且已订阅 Enterprise Hub 的组织计费。

cookies (`dict[str, str]`, `optional`) ：发送到服务器的附加cookie。

base_url (`str`, `optional`) ：运行推理的基本 URL。这是来自 `model` 的重复参数，以使 [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 遵循与 `openai.OpenAI` 客户端相同的模式。如果设置了`model`则无法使用。默认为无。api_key (`str`, `optional`) ：用于身份验证的令牌。这是来自 `token` 的重复参数，以使 [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 遵循与 `openai.OpenAI` 客户端相同的模式。如果设置了`token`则无法使用。默认为无。

初始化一个新的推理客户端。

[InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 旨在提供统一的推理体验。客户端可以使用
与（免费）推理 API、自托管推理端点或第三方推理提供商无缝连接。

#### audio_classification[[huggingface_hub.InferenceClient.audio_classification]]

```python
audio_classification(audio: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, top_k: int | None = None, function_to_apply: typing.Optional[ForwardRef('AudioClassificationOutputTransform')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L299)

**参数：**

audio (Union[str, Path, bytes, BinaryIO]) ：要分类的音频内容。它可以是原始音频字节、本地音频文件或指向音频文件的 URL。

model（`str`，*可选*）：用于音频分类的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的音频分类模型。

top_k (`int`, *可选*) ：指定后，将输出限制为前 K 个最可能的类。

function_to_apply (`"AudioClassificationOutputTransform"`, *可选*) ：应用于模型输出以检索分数的函数。**返回：** `list[AudioClassificationOutputElement]`

包含预测标签及其置信度的[AudioClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.AudioClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对提供的音频内容执行音频分类。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.audio_classification("audio.flac")
[
    AudioClassificationOutputElement(score=0.4976358711719513, label='hap'),
    AudioClassificationOutputElement(score=0.3677836060523987, label='neu'),
    ...
]
```

#### audio_to_audio[[huggingface_hub.InferenceClient.audio_to_audio]]

```python
audio_to_audio(audio: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L356)

**参数：**

audio (Union[str, Path, bytes, BinaryIO]) ：模型的音频内容。它可以是原始音频字节、本地音频文件或指向音频文件的 URL。

model (`str`, *可选*) ：模型可以是任何接受一个音频文件并返回另一个音频文件的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的audio_to_audio模型。

**返回：** `list[AudioToAudioOutputElement]`

[AudioToAudioOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.AudioToAudioOutputElement) 项目列表，其中包含 blob 中的音频标签、内容类型和音频内容。

**提高：** ``InferenceTimeoutError`` or `HfHubHTTPError`- ``InferenceTimeoutError`` -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据模型执行与音频到音频相关的多项任务（例如：语音增强、源分离）。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> audio_output = client.audio_to_audio("audio.flac")
>>> for i, item in enumerate(audio_output):
>>>     with open(f"output_{i}.flac", "wb") as f:
            f.write(item.blob)
```

#### 自动语音识别[[huggingface_hub.InferenceClient.自动语音识别]]

```python
automatic_speech_recognition(audio: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, extra_body: dict | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L408)

**参数：**

audio (Union[str, Path, bytes, BinaryIO]) ：要转录的内容。它可以是原始音频字节、本地音频文件或音频文件的 URL。

model（`str`，*可选*）：用于 ASR 的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用 ASR 的默认推荐模型。

extra_body (`dict`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**退货：** [AutomaticSpeechRecognitionOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.AutomaticSpeechRecognitionOutput)

包含转录文本和可选的时间戳块的项目。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对给定的音频内容执行自动语音识别（ASR 或音频转文本）。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.automatic_speech_recognition("hello_world.flac").text
"hello world"
```

#### chat_completion[[huggingface_hub.InferenceClient.chat_completion]]

```python
chat_completion(messages: list, model: str | None = None, stream: bool = False, frequency_penalty: float | None = None, logit_bias: list[float] | None = None, logprobs: bool | None = None, max_tokens: int | None = None, n: int | None = None, presence_penalty: float | None = None, response_format: typing.Union[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputResponseFormatText, huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputResponseFormatJSONSchema, huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputResponseFormatJSONObject, NoneType] = None, seed: int | None = None, stop: list[str] | None = None, stream_options: huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputStreamOptions | None = None, temperature: float | None = None, tool_choice: typing.Union[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputToolChoiceClass, ForwardRef('ChatCompletionInputToolChoiceEnum'), NoneType] = None, tool_prompt: str | None = None, tools: list[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputTool] | None = None, top_logprobs: int | None = None, top_p: float | None = None, extra_body: dict | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L535)

**参数：**

消息（[ChatCompletionInputMessage](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputMessage)列表）：由角色和内容对组成的对话历史记录。

model (`str`, *可选*) ：用于聊天完成的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用基于聊天的文本生成的默认推荐模型。有关更多详细信息，请参阅 https://huggingface.co/tasks/text- Generation 。如果`model`是模型ID，它将作为`model`参数传递到服务器。如果您想在请求负载中设置`model`时定义自定义URL，则必须在初始化[InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)时设置`base_url`。

Frequency_penalty (`float`, *可选*) ：根据迄今为止文本中的现有频率对新标记进行惩罚。范围：[-2.0，2.0]。默认为 0.0。logit_bias (`list[float]`, *可选*) ：调整特定标记出现在生成的输出中的可能性。

logprobs (`bool`, *可选*) ：是否返回输出标记的对数概率。如果为 true，则返回消息内容中返回的每个输出标记的对数概率。

max_tokens (`int`, *可选*) ：响应中允许的最大令牌数。默认为 100。

n (`int`, *可选*) ：为每个提示生成的完成数。

Presence_penalty (`float`, *可选*) ：-2.0 到 2.0 之间的数字。正值根据新标记目前是否出现在文本中来对其进行惩罚，从而增加模型谈论新主题的可能性。

response_format (`ChatCompletionInputGrammarType()`, *可选*) ：语法约束。可以是 JSONSchema 或正则表达式。

种子（可选`int`，*可选*）：可重复控制流的种子。默认为无。

stop (`list[str]`, *可选*) ：最多四个字符串，触发响应结束。默认为无。

流（`bool`，*可选*）：启用实时响应流。默认为 False。

Stream_options ([ChatCompletionInputStreamOptions](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputStreamOptions), *可选*) ：流式完成的选项。温度（`float`，*可选*）：控制生成的随机性。较低的值可确保较少的随机完成。范围：[0, 2]。默认为 1.0。

top_logprobs (`int`, *可选*) ：0 到 5 之间的整数，指定每个标记位置最有可能返回的标记数量，每个标记都有一个关联的对数概率。如果使用此参数，则 logprobs 必须设置为 true。

top_p (`float`, *可选*) ：最有可能进行采样的下一个单词的分数。必须介于 0 和 1 之间。默认为 1.0。

tool_choice（[ChatCompletionInputToolChoiceClass](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputToolChoiceClass)或`ChatCompletionInputToolChoiceEnum()`，*可选*）：用于完成的工具。默认为“自动”。

tool_prompt (`str`, *可选*) ：附加在工具之前的提示。

工具（[ChatCompletionInputTool](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputTool)列表，*可选*）：模型可能调用的工具列表。目前，仅支持函数作为工具。使用它来提供模型可以为其生成 JSON 输入的函数列表。

extra_body (`dict`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**返回：** [ChatCompletionOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionOutput) 或 [ChatCompletionStreamOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionStreamOutput) 的可迭代从服务器返回的生成文本：
- 如果`stream=False`，生成的文本将作为[ChatCompletionOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionOutput)返回（默认）。
- 如果`stream=True`，则生成的文本将作为[ChatCompletionStreamOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionStreamOutput) 的序列逐个令牌返回。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

一种使用指定语言模型完成对话的方法。

> [!提示]
> 为了与 OpenAI 客户端兼容，`client.chat_completion` 方法别名为 `client.chat.completions.create`。
> 输入和输出完全相同，使用任一语法都会产生相同的结果。
> 查看[Inference guide](https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility)
> 有关 OpenAI 兼容性的更多详细信息。

> [!提示]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：

```py
>>> from huggingface_hub import InferenceClient
>>> messages = [{"role": "user", "content": "What is the capital of France?"}]
>>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
>>> client.chat_completion(messages, max_tokens=100)
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason='eos_token',
            index=0,
            message=ChatCompletionOutputMessage(
                role='assistant',
                content='The capital of France is Paris.',
                name=None,
                tool_calls=None
            ),
            logprobs=None
        )
    ],
    created=1719907176,
    id='',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    object='text_completion',
    system_fingerprint='2.0.4-sha-f426a33',
    usage=ChatCompletionOutputUsage(
        completion_tokens=8,
        prompt_tokens=17,
        total_tokens=25
    )
)
```

使用流式传输的示例：
```py
>>> from huggingface_hub import InferenceClient
>>> messages = [{"role": "user", "content": "What is the capital of France?"}]
>>> client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
>>> for token in client.chat_completion(messages, max_tokens=10, stream=True):
...     print(token)
ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content='The', role='assistant'), index=0, finish_reason=None)], created=1710498504)
ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' capital', role='assistant'), index=0, finish_reason=None)], created=1710498504)
(...)
ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' may', role='assistant'), index=0, finish_reason=None)], created=1710498504)
```

使用 OpenAI 语法的示例：
```py
# instead of `from openai import OpenAI`
from huggingface_hub import InferenceClient

# instead of `client = OpenAI(...)`
client = InferenceClient(
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
```

直接使用带有额外（特定于提供商的）参数的第三方提供商的示例。使用量将在您的 Together AI 帐户上计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="together",  # Use Together AI provider
...     api_key="REDACTED",  # Pass your Together API key directly
... )
>>> client.chat_completion(
...     model="meta-llama/Meta-Llama-3-8B-Instruct",
...     messages=[{"role": "user", "content": "What is the capital of France?"}],
...     extra_body={"safety_model": "Meta-Llama/Llama-Guard-7b"},
... )
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="novita",  # Use Novita provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> client.chat_completion(
...     model="meta-llama/Meta-Llama-3-8B-Instruct",
...     messages=[{"role": "user", "content": "What is the capital of France?"}],
... )
```使用图像 + 文本作为输入的示例：
```py
>>> from huggingface_hub import InferenceClient

# provide a remote URL
>>> image_url ="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
# or a base64-encoded image
>>> image_path = "/path/to/image.jpeg"
>>> with open(image_path, "rb") as f:
...     base64_image = base64.b64encode(f.read()).decode("utf-8")
>>> image_url = f"data:image/jpeg;base64,{base64_image}"

>>> client = InferenceClient("meta-llama/Llama-3.2-11B-Vision-Instruct")
>>> output = client.chat.completions.create(
...     messages=[
...         {
...             "role": "user",
...             "content": [
...                 {
...                     "type": "image_url",
...                     "image_url": {"url": image_url},
...                 },
...                 {
...                     "type": "text",
...                     "text": "Describe this image in one sentence.",
...                 },
...             ],
...         },
...     ],
... )
>>> output
The image depicts the iconic Statue of Liberty situated in New York Harbor, New York, on a clear day.
```

使用工具示例：
```py
>>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
>>> messages = [
...     {
...         "role": "system",
...         "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
...     },
...     {
...         "role": "user",
...         "content": "What's the weather like the next 3 days in San Francisco, CA?",
...     },
... ]
>>> tools = [
...     {
...         "type": "function",
...         "function": {
...             "name": "get_current_weather",
...             "description": "Get the current weather",
...             "parameters": {
...                 "type": "object",
...                 "properties": {
...                     "location": {
...                         "type": "string",
...                         "description": "The city and state, e.g. San Francisco, CA",
...                     },
...                     "format": {
...                         "type": "string",
...                         "enum": ["celsius", "fahrenheit"],
...                         "description": "The temperature unit to use. Infer this from the users location.",
...                     },
...                 },
...                 "required": ["location", "format"],
...             },
...         },
...     },
...     {
...         "type": "function",
...         "function": {
...             "name": "get_n_day_weather_forecast",
...             "description": "Get an N-day weather forecast",
...             "parameters": {
...                 "type": "object",
...                 "properties": {
...                     "location": {
...                         "type": "string",
...                         "description": "The city and state, e.g. San Francisco, CA",
...                     },
...                     "format": {
...                         "type": "string",
...                         "enum": ["celsius", "fahrenheit"],
...                         "description": "The temperature unit to use. Infer this from the users location.",
...                     },
...                     "num_days": {
...                         "type": "integer",
...                         "description": "The number of days to forecast",
...                     },
...                 },
...                 "required": ["location", "format", "num_days"],
...             },
...         },
...     },
... ]

>>> response = client.chat_completion(
...     model="meta-llama/Meta-Llama-3-70B-Instruct",
...     messages=messages,
...     tools=tools,
...     tool_choice="auto",
...     max_tokens=500,
... )
>>> response.choices[0].message.tool_calls[0].function
ChatCompletionOutputFunctionDefinition(
    arguments={
        'location': 'San Francisco, CA',
        'format': 'fahrenheit',
        'num_days': 3
    },
    name='get_n_day_weather_forecast',
    description=None
)
```

使用response_format的示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
>>> messages = [
...     {
...         "role": "user",
...         "content": "I saw a puppy a cat and a raccoon during my bike ride in the park. What did I see and when?",
...     },
... ]
>>> response_format = {
...     "type": "json",
...     "value": {
...         "properties": {
...             "location": {"type": "string"},
...             "activity": {"type": "string"},
...             "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
...             "animals": {"type": "array", "items": {"type": "string"}},
...         },
...         "required": ["location", "activity", "animals_seen", "animals"],
...     },
... }
>>> response = client.chat_completion(
...     messages=messages,
...     response_format=response_format,
...     max_tokens=500,
... )
>>> response.choices[0].message.content
'{

y": "bike ride",
": ["puppy", "cat", "raccoon"],
_seen": 3,
n": "park"}'
```

#### document_question_answering[[huggingface_hub.InferenceClient.document_question_answering]]

```python
document_question_answering(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], question: str, model: str | None = None, doc_stride: int | None = None, handle_impossible_answer: bool | None = None, lang: str | None = None, max_answer_len: int | None = None, max_question_len: int | None = None, max_seq_len: int | None = None, top_k: int | None = None, word_boxes: list[list[float] | str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L937)

**参数：**

image (`Union[str, Path, bytes, BinaryIO]`) ：上下文的输入图像。它可以是原始字节、图像文件或在线图像的 URL。

问题（`str`）：要回答的问题。

model (`str`, *可选*) ：用于文档问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果不提供，将使用默认推荐的文档问答模型。默认为无。

doc_stride (`int`, *可选*) ：如果文档中的单词太长而无法满足模型的问题，它将被分成几个有重叠的块。该参数控制重叠的大小。

handle_impossible_answer (`bool`, *可选*) : 是否接受不可能作为答案

lang (`str`, *可选*) ：运行 OCR 时使用的语言。默认为英文。max_answer_len (`int`, *可选*) ：预测答案的最大长度（例如，仅考虑长度较短的答案）。

max_question_len (`int`, *可选*) ：标记化后问题的最大长度。如果需要，它将被截断。

max_seq_len (`int`, *可选*) ：传递给模型的每个块的标记中总句子（上下文+问题）的最大长度。如果需要，上下文将被分割成几个块（使用 doc_stride 作为重叠）。

top_k (`int`, *可选*) ：要返回的答案数量（将按可能性顺序选择）。如果上下文中没有足够的可用选项，则可以返回少于 top_k 的答案。

word_boxes (`list[Union[list[float], str`, *可选*) ：单词和边界框的列表（标准化 0->1000）。如果提供，推理将跳过 OCR 步骤并使用提供的边界框。

**返回：** `list[DocumentQuestionAnsweringOutputElement]`

包含预测标签、相关概率、单词 ID 和页码的[DocumentQuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.DocumentQuestionAnsweringOutputElement) 项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。回答有关文档图像的问题。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.document_question_answering(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")
[DocumentQuestionAnsweringOutputElement(answer='us-001', end=16, score=0.9999666213989258, start=16)]
```

#### feature_extraction[[huggingface_hub.InferenceClient.feature_extraction]]

```python
feature_extraction(text: str | list[str], normalize: bool | None = None, prompt_name: str | None = None, truncate: bool | None = None, truncation_direction: typing.Optional[typing.Literal['left', 'right']] = None, dimensions: int | None = None, encoding_format: typing.Optional[typing.Literal['float', 'base64']] = None, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1024)

**参数：**

text (*str* 或 *list[str]*) ：要嵌入的文本或文本列表。

model (*str*, *可选*) ：用于特征提取任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的特征提取模型。默认为无。

规范化（*bool*，*可选*）：是否规范化嵌入。仅在由文本嵌入推理支持的服务器上可用。

Prompt_name (*str*, *可选*) ：用于编码的提示的名称。如果未设置，则不会应用任何提示。必须是 *Sentence Transformers* 配置 *prompts* 字典中的键。例如，如果 `prompt_name` 是“query”并且 `prompts` 是 &lcub;"query": "query: ",...}，则句子“法国的首都是什么？”将被编码为“查询：法国的首都是哪里？”因为提示文本将被添加到任何要编码的文本之前。truncate (*bool*, *可选*) ：是否截断嵌入。仅在由文本嵌入推理支持的服务器上可用。

truncation_direction (*Literal["left", "right"]*, *可选*) ：当传递 *truncate=True* 时，应截断输入的哪一侧。

维度（*int*，*可选*）：生成的输出嵌入应具有的维度数。仅适用于兼容 OpenAI 的嵌入端点。

coding_format (*Literal["float", "base64"]*, *可选*) ：输出嵌入的格式。 “float”或“base64”。仅适用于兼容 OpenAI 的嵌入端点。

**返回：** `*np.ndarray*`

将输入文本表示为 float32 numpy 数组的嵌入。

**引发：** [*InferenceTimeoutError*] 或 [*HfHubHTTPError*]

- [*推理超时错误*] -- 
  如果模型不可用或请求超时。
- [*HfHubHTTPError*] -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

为给定文本或一批文本生成嵌入。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.feature_extraction("Hi, who are you?")
array([[ 2.424802  ,  2.93384   ,  1.1750331 , ...,  1.240499, -0.13776633, -0.7889173 ],
[-0.42943227, -0.6364878 , -1.693462  , ...,  0.41978157, -2.4336355 ,  0.6162071 ],
...,
[ 0.28552425, -0.928395  , -1.2077185 , ...,  0.76810825, -2.1069427 ,  0.6236161 ]], dtype=float32)
```

#### fill_mask[[huggingface_hub.InferenceClient.fill_mask]]

```python
fill_mask(text: str, model: str | None = None, targets: list[str] | None = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1107)

**参数：**text (`str`) ：要填充的字符串，必须包含 [MASK] 标记（检查模型卡以获取掩码的确切名称）。

model (`str`, *可选*) ：用于填充遮罩任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的填充掩模模型。

目标（`list[str`，*可选*）：通过时，模型会将分数限制为通过的目标，而不是在整个词汇表中查找。如果提供的目标不在模型词汇中，它们将被标记化，并且将使用第一个生成的标记（带有警告，并且可能会更慢）。

top_k (`int`, *可选*) ：传递时，覆盖要返回的预测数。

**返回：** `list[FillMaskOutputElement]`

包含预测标签的[FillMaskOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.FillMaskOutputElement)项目列表，关联
概率、标记参考和完整的文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

用缺失的单词填补一个洞（准确地说是标记）。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.fill_mask("The goal of life is <mask>.")
[
    FillMaskOutputElement(score=0.06897063553333282, token=11098, token_str=' happiness', sequence='The goal of life is happiness.'),
    FillMaskOutputElement(score=0.06554922461509705, token=45075, token_str=' immortality', sequence='The goal of life is immortality.')
]
```#### get_endpoint_info[[huggingface_hub.InferenceClient.get_endpoint_info]]

```python
get_endpoint_info(model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L3277)

**参数：**

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

**返回：** `dict[str, Any]`

有关端点的信息。

获取有关已部署端点的信息。

此端点仅在由文本生成推理 (TGI) 或文本嵌入推理 (TEI) 支持的端点上可用。
由`transformers`供电的端点返回空负载。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
>>> client.get_endpoint_info()
{
    'model_id': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'model_sha': None,
    'model_dtype': 'torch.float16',
    'model_device_type': 'cuda',
    'model_pipeline_tag': None,
    'max_concurrent_requests': 128,
    'max_best_of': 2,
    'max_stop_sequences': 4,
    'max_input_length': 8191,
    'max_total_tokens': 8192,
    'waiting_served_ratio': 0.3,
    'max_batch_total_tokens': 1259392,
    'max_waiting_tokens': 20,
    'max_batch_size': None,
    'validation_workers': 32,
    'max_client_batch_size': 4,
    'version': '2.0.2',
    'sha': 'dccab72549635c7eb5ddb17f43f0b7cdff07c214',
    'docker_label': 'sha-dccab72'
}
```

#### health_check[[huggingface_hub.InferenceClient.health_check]]

```python
health_check(model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L3335)

**参数：**

model（`str`，*可选*）：推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

**返回：** `bool`

如果一切正常，则属实。

检查已部署端点的运行状况。

运行状况检查仅适用于由文本生成推理 (TGI) 或文本嵌入推理 (TEI) 提供支持的推理端点。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient("https://jzgu0buei5.us-east-1.aws.endpoints.huggingface.cloud")
>>> client.health_check()
True
```#### image_classification[[huggingface_hub.InferenceClient.image_classification]]

```python
image_classification(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, function_to_apply: typing.Optional[ForwardRef('ImageClassificationOutputTransform')] = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1163)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：要分类的图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model (`str`，*可选*)：用于图像分类的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的图像分类模型。

function_to_apply (`"ImageClassificationOutputTransform"`, *可选*) ：应用于模型输出以检索分数的函数。

top_k (`int`, *可选*) ：指定后，将输出限制为前 K 个最可能的类。

**返回：** `list[ImageClassificationOutputElement]`

包含预测标签和相关概率的[ImageClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ImageClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定的模型对给定图像执行图像分类。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[ImageClassificationOutputElement(label='Blenheim spaniel', score=0.9779096841812134), ...]
```

#### image_segmentation[[huggingface_hub.InferenceClient.image_segmentation]]

```python
image_segmentation(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, mask_threshold: float | None = None, overlap_mask_area_threshold: float | None = None, subtask: typing.Optional[ForwardRef('ImageSegmentationSubtask')] = None, threshold: float | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1213)

**参数：**image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：要分割的图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model（`str`，*可选*）：用于图像分割的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的图像分割模型。

mask_threshold (`float`, *可选*) ：将预测掩码转换为二进制值时使用的阈值。

overlap_mask_area_threshold (`float`, *可选*) ：掩码重叠阈值以消除小的、断开的段。

子任务（`"ImageSegmentationSubtask"`，*可选*）：要执行的分割任务，具体取决于模型功能。

阈值（`float`，*可选*）：过滤掉预测掩模的概率阈值。

**退货：** `list[ImageSegmentationOutputElement]`

包含分段掩码和关联属性的[ImageSegmentationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ImageSegmentationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定的模型对给定图像执行图像分割。> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_segmentation("cat.jpg")
[ImageSegmentationOutputElement(score=0.989008, label='LABEL_184', mask=<PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>), ...]
```

#### image_to_image[[huggingface_hub.InferenceClient.image_to_image]]

```python
image_to_image(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], prompt: str | None = None, negative_prompt: str | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, model: str | None = None, target_size: huggingface_hub.inference._generated.types.image_to_image.ImageToImageTargetSize | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1281)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：用于翻译的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

提示（`str`，*可选*）：指导图像生成的文本提示。

negative_prompt (`str`, *可选*) ：一个提示，用于指导图像生成中不包含哪些内容。

num_inference_steps (`int`, *可选*) ：用于扩散模型。去噪步数。更多的去噪步骤通常会带来更高质量的图像，但代价是推理速度变慢。

Guiding_scale (`float`, *可选*) ：用于扩散模型。较高的引导比例值会鼓励模型生成与文本提示紧密相关的图像，但代价是图像质量较低。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。target_size (`ImageToImageTargetSize`, *可选*) ：输出图像的大小（以像素为单位）。仅某些提供商和特定型号支持此参数。当不支持时它将被忽略。

**返回：** `Image`

翻译后的图像。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定模型执行图像到图像的转换。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
>>> image.save("tiger.jpg")
```

#### image_to_text[[huggingface_hub.InferenceClient.image_to_text]]

```python
image_to_text(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1436)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：标题的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

**退货：** [ImageToTextOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ImageToTextOutput)

生成的文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

获取输入图像并返回文本。

根据您的用例（图像字幕、光学字符识别），模型可以有非常不同的输出
（OCR）、Pix2Struct 等）。请查看型号卡以了解有关型号特性的更多信息。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.image_to_text("cat.jpg")
'a cat standing in a grassy field '
>>> client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
'a dog laying on the grass next to a flower pot '
```

#### image_to_video[[huggingface_hub.InferenceClient.image_to_video]]

```python
image_to_video(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, prompt: str | None = None, negative_prompt: str | None = None, num_frames: float | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, seed: int | None = None, target_size: huggingface_hub.inference._generated.types.image_to_video.ImageToVideoTargetSize | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1357)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：用于生成视频的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

提示（`str`，*可选*）：指导视频生成的文本提示。

negative_prompt（`str`，*可选*）：一个提示，用于指导视频生成中不包含哪些内容。num_frames (`float`, *可选*) ：num_frames 参数决定生成多少视频帧。

num_inference_steps (`int`, *可选*) ：用于扩散模型。去噪步数。更多的去噪步骤通常会带来更高质量的图像，但代价是推理速度变慢。

Guiding_scale (`float`, *可选*) ：用于扩散模型。较高的指导比例值会鼓励模型生成与文本提示紧密相关的视频，但会降低图像质量。

种子（`int`，*可选*）：用于视频生成的种子。

target_size (`ImageToVideoTargetSize`, *可选*) ：输出视频帧的像素大小。

num_inference_steps (`int`, *可选*) ：去噪步骤的数量。更多的去噪步骤通常会带来更高质量的视频，但代价是推理速度变慢。

种子（`int`，*可选*）：随机数生成器的种子。

**返回：** `bytes`

生成的视频。

从输入图像生成视频。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> video = client.image_to_video("cat.jpg", model="Wan-AI/Wan2.2-I2V-A14B", prompt="turn the cat into a tiger")
>>> with open("tiger.mp4", "wb") as f:
...     f.write(video)
```

#### object_detection[[huggingface_hub.InferenceClient.object_detection]]

```python
object_detection(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, threshold: float | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1482)

**参数：**image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：用于检测对象的图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model (`str`, *可选*) ：用于对象检测的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的对象检测模型 (DETR)。

阈值（`float`，*可选*）：进行预测所需的概率。

**返回：** `list[ObjectDetectionOutputElement]`

包含边界框和关联属性的 [ObjectDetectionOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ObjectDetectionOutputElement) 项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError` 或 ``ValueError``

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。
- ``ValueError`` -- 
  如果请求输出不是List。

使用指定的模型对给定图像执行对象检测。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.object_detection("people.jpg")
[ObjectDetectionOutputElement(score=0.9486683011054993, label='person', box=ObjectDetectionBoundingBox(xmin=59, ymin=39, xmax=420, ymax=510)), ...]
```

#### Question_answering[[huggingface_hub.InferenceClient.question_answering]]

```python
question_answering(question: str, context: str, model: str | None = None, align_to_words: bool | None = None, doc_stride: int | None = None, handle_impossible_answer: bool | None = None, max_answer_len: int | None = None, max_question_len: int | None = None, max_seq_len: int | None = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1530)

**参数：**

问题（`str`）：需要回答的问题。

context (`str`) ：问题的上下文。model (`str`) ：用于问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。

align_to_words (`bool`, *可选*) ：尝试将答案与真实单词对齐。提高空间分隔语言的质量。可能会对非空格分隔的语言（例如日语或中文）造成伤害

doc_stride (`int`, *可选*) ：如果上下文太长而无法满足模型的问题，它将被分成几个有重叠的块。该参数控制重叠的大小。

handle_impossible_answer (`bool`, *可选*) ：是否接受不可能作为答案。

max_answer_len (`int`, *可选*) ：预测答案的最大长度（例如，仅考虑长度较短的答案）。

max_question_len (`int`, *可选*) ：标记化后问题的最大长度。如果需要，它将被截断。

max_seq_len (`int`, *可选*) ：传递给模型的每个块的标记中总句子（上下文+问题）的最大长度。如果需要，上下文将被分割成几个块（使用 docStride 作为重叠）。top_k (`int`, *可选*) ：要返回的答案数量（将按可能性顺序选择）。请注意，如果上下文中没有足够的选项，我们将返回少于 topk 的答案。

**退货：** 联合[⟦T353⟧, list[QuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.QuestionAnsweringOutputElement)]

当top_k为1或未提供时，它返回单个`QuestionAnsweringOutputElement`。
当top_k大于1时，返回`QuestionAnsweringOutputElement`的列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

从给定文本中检索问题的答案。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.question_answering(question="What's my name?", context="My name is Clara and I live in Berkeley.")
QuestionAnsweringOutputElement(answer='Clara', end=16, score=0.9326565265655518, start=11)
```

####句子相似度[[huggingface_hub.InferenceClient.sentence_similarity]]

```python
sentence_similarity(sentence: str, other_sentences: list, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1614)

**参数：**

句子 (`str`) ：与其他句子进行比较的主要句子。

other_sentences (`list[str]`) ：要比较的句子列表。

model (`str`, *可选*) ：用于句子相似度任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果不提供，将使用默认推荐的句子相似度模型。默认为无。

**返回：** `list[float]`

表示输入文本的嵌入。**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

通过比较句子和其他句子列表的嵌入来计算它们之间的语义相似度。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.sentence_similarity(
...     "Machine learning is so easy.",
...     other_sentences=[
...         "Deep learning is so straightforward.",
...         "This is so difficult, like rocket science.",
...         "I can't believe how much I struggled with this.",
...     ],
... )
[0.7785726189613342, 0.45876261591911316, 0.2906220555305481]
```

#### 总结[[huggingface_hub.InferenceClient.summarization]]

```python
summarization(text: str, model: str | None = None, clean_up_tokenization_spaces: bool | None = None, generate_parameters: dict[str, typing.Any] | None = None, truncation: typing.Optional[ForwardRef('SummarizationTruncationStrategy')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1667)

**参数：**

text (`str`) ：要摘要的输入文本。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐模型进行汇总。

clean_up_tokenization_spaces (`bool`, *可选*) ：是否清理文本输出中潜在的额外空格。

generate_parameters（`dict[str, Any]`，*可选*）：文本生成算法的附加参数化。

截断（`"SummarizationTruncationStrategy"`，*可选*）：要使用的截断策略。

**返回：** [SummarizationOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.SummarizationOutput)

生成的摘要文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定模型生成给定文本的摘要。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.summarization("The Eiffel tower...")
SummarizationOutput(generated_text="The Eiffel tower is one of the most famous landmarks in the world....")
```

#### table_question_answering[[huggingface_hub.InferenceClient.table_question_answering]]

```python
table_question_answering(table: dict, query: str, model: str | None = None, padding: typing.Optional[ForwardRef('Padding')] = None, sequential: bool | None = None, truncation: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1725)

**参数：**

table (`str`) ：表示为列表字典的数据表，其中条目是标题，列表是所有值，所有列表必须具有相同的大小。

query (`str`) ：要向表询问的纯文本查询。

model (`str`) ：用于表格问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。

填充（`"Padding"`，*可选*）：激活和控制填充。

顺序（`bool`，*可选*）：是否按顺序或批量进行推理。批处理速度更快，但考虑到序列的对话性质，像 SQA 这样的模型需要按顺序进行推理，以提取序列内的关系。

截断（`bool`，*可选*）：激活并控制截断。

**返回：** [TableQuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TableQuestionAnsweringOutputElement)包含答案、坐标、单元格和使用的聚合器的表格问答输出。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

从表格中给出的信息中检索问题的答案。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> query = "How many stars does the transformers repository have?"
>>> table = {"Repository": ["Transformers", "Datasets", "Tokenizers"], "Stars": ["36542", "4512", "3934"]}
>>> client.table_question_answering(table, query, model="google/tapas-base-finetuned-wtq")
TableQuestionAnsweringOutputElement(answer='36542', coordinates=[[0, 1]], cells=['36542'], aggregator='AVERAGE')
```

#### tabular_classification[[huggingface_hub.InferenceClient.tabular_classification]]

```python
tabular_classification(table: dict, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1787)

**参数：**

表 (`dict[str, Any]`) ：要分类的属性集。

model (`str`，*可选*)：用于表格分类任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的表格分类模型。默认为无。

**返回：** `List`

标签列表，初始表中每行一个。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据一组属性对目标类别（一组）进行分类。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> table = {
...     "fixed_acidity": ["7.4", "7.8", "10.3"],
...     "volatile_acidity": ["0.7", "0.88", "0.32"],
...     "citric_acid": ["0", "0", "0.45"],
...     "residual_sugar": ["1.9", "2.6", "6.4"],
...     "chlorides": ["0.076", "0.098", "0.073"],
...     "free_sulfur_dioxide": ["11", "25", "5"],
...     "total_sulfur_dioxide": ["34", "67", "13"],
...     "density": ["0.9978", "0.9968", "0.9976"],
...     "pH": ["3.51", "3.2", "3.23"],
...     "sulphates": ["0.56", "0.68", "0.82"],
...     "alcohol": ["9.4", "9.8", "12.6"],
... }
>>> client.tabular_classification(table=table, model="julien-c/wine-quality")
["5", "5", "5"]
```#### tabular_regression[[huggingface_hub.InferenceClient.tabular_regression]]

```python
tabular_regression(table: dict, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1842)

**参数：**

table (`dict[str, Any]`) ：存储在表中的属性集。用于预测目标的属性可以是数值属性和分类属性。

model (`str`, *可选*) ：用于表格回归任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的表格回归模型。默认为无。

**返回：** `List`

预测数字目标值的列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

给定表中的一组属性/特征，预测数字目标值。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> table = {
...     "Height": ["11.52", "12.48", "12.3778"],
...     "Length1": ["23.2", "24", "23.9"],
...     "Length2": ["25.4", "26.3", "26.5"],
...     "Length3": ["30", "31.2", "31.1"],
...     "Species": ["Bream", "Bream", "Bream"],
...     "Width": ["4.02", "4.3056", "4.6961"],
... }
>>> client.tabular_regression(table, model="scikit-learn/Fish-Weight")
[110, 120, 130]
```

#### 文本分类[[huggingface_hub.InferenceClient.text_classification]]

```python
text_classification(text: str, model: str | None = None, top_k: int | None = None, function_to_apply: typing.Optional[ForwardRef('TextClassificationOutputTransform')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L1892)

**参数：**

text (`str`) ：要分类的字符串。model（`str`，*可选*）：用于文本分类任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本分类模型。默认为无。

top_k (`int`, *可选*) ：指定后，将输出限制为前 K 个最可能的类。

function_to_apply (`"TextClassificationOutputTransform"`, *可选*) ：应用于模型输出以检索分数的函数。

**返回：** `list[TextClassificationOutputElement]`

包含预测标签和相关概率的[TextClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对给定文本执行文本分类（例如情感分析）。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.text_classification("I like you")
[
    TextClassificationOutputElement(label='POSITIVE', score=0.9998695850372314),
    TextClassificationOutputElement(label='NEGATIVE', score=0.0001304351753788069),
]
```

#### 文本生成[[huggingface_hub.InferenceClient.text_ Generation]]

```python
text_generation(prompt: str, details: bool | None = None, stream: bool | None = None, model: str | None = None, adapter_id: str | None = None, best_of: int | None = None, decoder_input_details: bool | None = None, do_sample: bool | None = None, frequency_penalty: float | None = None, grammar: huggingface_hub.inference._generated.types.text_generation.TextGenerationInputGrammarType | None = None, max_new_tokens: int | None = None, repetition_penalty: float | None = None, return_full_text: bool | None = None, seed: int | None = None, stop: list[str] | None = None, stop_sequences: list[str] | None = None, temperature: float | None = None, top_k: int | None = None, top_n_tokens: int | None = None, top_p: float | None = None, truncate: int | None = None, typical_p: float | None = None, watermark: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L2100)

**参数：**

提示符(`str`)：输入文字。详细信息（`bool`，*可选*）：默认情况下，text_ Generation 返回一个字符串。如果您想要详细的输出（标记、概率、种子、完成原因等），请通过`details=True`。仅适用于运行 `text-generation-inference` 后端的型号。

流（`bool`​​，*可选*）：默认情况下，text_ Generation 返回完整的生成文本。如果您想要返回令牌流，请传递`stream=True`。仅适用于运行 `text-generation-inference` 后端的型号。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

adapter_id（`str`，*可选*）：Lora 适配器 ID。

best_of (`int`, *可选*) ：生成 best_of 序列并返回最高 token logprobs 的序列。

Decoder_input_details (`bool`, *可选*) ：返回解码器输入令牌 logprobs 和 ids。您还必须设置 `details=True` 才能将其考虑在内。默认为`False`。

do_sample (`bool`, *可选*) : 激活 logits 采样Frequency_penalty（`float`，*可选*）：-2.0 到 2.0 之间的数字。正值根据迄今为止文本中的现有频率对新标记进行惩罚，从而降低模型逐字重复同一行的可能性。

语法（[TextGenerationInputGrammarType](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextGenerationInputGrammarType)，*可选*）：语法约束。可以是 JSONSchema 或正则表达式。

max_new_tokens (`int`, *可选*) : 生成的令牌的最大数量。默认为 100。

repetition_penalty (`float`, *可选*) ：重复惩罚的参数。 1.0 表示没有处罚。更多详情请参见[this paper](https://arxiv.org/pdf/1909.05858.pdf)。

return_full_text (`bool`, *可选*) : 是否将提示添加到生成的文本中

种子（`int`，*可选*）：随机采样种子

stop (`list[str]`, *可选*) ：如果生成了 `stop` 的成员，则停止生成令牌。

stop_sequences (`list[str]`, *可选*) ：已弃用的参数。请使用 `stop` 代替。

温度（`float`，*可选*）：用于对 logits 分布进行建模的值。

top_n_tokens (`int`, *可选*) ：在每个生成步骤返回有关 `top_n_tokens` 最有可能的标记的信息，而不仅仅是采样的标记。top_k (`int`, *可选`) ：为 top-k 过滤保留的最高概率词汇标记的数量。

top_p（`float`，*可选`) : If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p`或更高版本保留用于生成。

truncate (`int`, *可选`) ：将输入标记截断为给定大小。

典型_p（`float`，*可选`）：典型解码质量请参阅[Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666)了解更多信息

水印（`bool`，*可选*）：使用[A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)添加水印

**返回：** `Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]`

从服务器返回的生成文本：
- 如果`stream=False`和`details=False`，生成的文本将作为`str`返回（默认）
- 如果`stream=True`和`details=False`，生成的文本将逐个标记作为`Iterable[str]`返回
- 如果`stream=False`和`details=True`，则返回生成的文本，其中包含更多详细信息作为[TextGenerationOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextGenerationOutput)
- 如果`details=True`和`stream=True`，生成的文本将作为[TextGenerationStreamOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextGenerationStreamOutput)的可迭代对象逐个返回

**提高：** ``ValidationError`` or [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) or `HfHubHTTPError`

- ``ValidationError`` -- 
  如果输入值无效。不会对服务器进行 HTTP 调用。
- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据提示，生成以下文本。> [!提示]
> 如果您想从聊天消息生成响应，您应该使用 [InferenceClient.chat_completion()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion) 方法。
> 它接受消息列表而不是单个文本提示，并为您处理聊天模板。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

# Case 1: generate text
>>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
'100% open source and built to be easy to use.'

# Case 2: iterate over the generated tokens. Useful for large generation.
>>> for token in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
...     print(token)
100
%
open
source
and
built
to
be
easy
to
use
.

# Case 3: get more details about the generation process.
>>> client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
TextGenerationOutput(
    generated_text='100% open source and built to be easy to use.',
    details=TextGenerationDetails(
        finish_reason='length',
        generated_tokens=12,
        seed=None,
        prefill=[
            TextGenerationPrefillOutputToken(id=487, text='The', logprob=None),
            TextGenerationPrefillOutputToken(id=53789, text=' hugging', logprob=-13.171875),
            (...)
            TextGenerationPrefillOutputToken(id=204, text=' ', logprob=-7.0390625)
        ],
        tokens=[
            TokenElement(id=1425, text='100', logprob=-1.0175781, special=False),
            TokenElement(id=16, text='%', logprob=-0.0463562, special=False),
            (...)
            TokenElement(id=25, text='.', logprob=-0.5703125, special=False)
        ],
        best_of_sequences=None
    )
)

# Case 4: iterate over the generated tokens with more details.
# Last object is more complete, containing the full generated text and the finish reason.
>>> for details in client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
...     print(details)
...
TextGenerationStreamOutput(token=TokenElement(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(
    id=25,
    text='.',
    logprob=-0.5703125,
    special=False),
    generated_text='100% open source and built to be easy to use.',
    details=TextGenerationStreamOutputStreamDetails(finish_reason='length', generated_tokens=12, seed=None)
)

# Case 5: generate constrained output using grammar
>>> response = client.text_generation(
...     prompt="I saw a puppy a cat and a raccoon during my bike ride in the park",
...     model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
...     max_new_tokens=100,
...     repetition_penalty=1.3,
...     grammar={
...         "type": "json",
...         "value": {
...             "properties": {
...                 "location": {"type": "string"},
...                 "activity": {"type": "string"},
...                 "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
...                 "animals": {"type": "array", "items": {"type": "string"}},
...             },
...             "required": ["location", "activity", "animals_seen", "animals"],
...         },
...     },
... )
>>> json.loads(response)
{
    "activity": "bike riding",
    "animals": ["puppy", "cat", "raccoon"],
    "animals_seen": 3,
    "location": "park"
}
```

#### 文本到图像[[huggingface_hub.InferenceClient.文本到图像]]

```python
text_to_image(prompt: str, negative_prompt: str | None = None, height: int | None = None, width: int | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, model: str | None = None, scheduler: str | None = None, seed: int | None = None, extra_body: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L2439)

**参数：**

提示 (`str`) ：生成图像的提示。

negative_prompt (`str`, *可选*) ：一个提示，用于指导图像生成中不包含哪些内容。

height (`int`, *可选*) : 输出图像的高度（以像素为单位）

width (`int`, *可选*) : 输出图像的宽度（以像素为单位）

num_inference_steps (`int`, *可选*) ：去噪步骤的数量。更多的去噪步骤通常会带来更高质量的图像，但代价是推理速度变慢。

Guiding_scale (`float`, *可选*) ：较高的引导比例值会鼓励模型生成与文本提示紧密相关的图像，但值太高可能会导致饱和度和其他伪影。model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本到图像模型。默认为无。

调度程序（`str`，*可选*）：用兼容的调度程序覆盖调度程序。

种子（`int`，*可选*）：随机数生成器的种子。

extra_body (`dict[str, Any]`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**返回：** `Image`

生成的图像。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定模型根据给定文本生成图像。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

> [!提示]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> image = client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> image = client.text_to_image(
...     "An astronaut riding a horse on the moon.",
...     negative_prompt="low resolution, blurry",
...     model="stabilityai/stable-diffusion-2-1",
... )
>>> image.save("better_astronaut.png")
```

直接使用第三方提供商的示例。使用费用将通过您的 fal.ai 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="fal-ai",  # Use fal.ai provider
...     api_key="REDACTED",  # Pass your fal.ai API key
... )
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Use replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> image = client.text_to_image(
...     "An astronaut riding a horse on the moon.",
...     model="black-forest-labs/FLUX.1-dev",
... )
>>> image.save("astronaut.png")
```

使用具有额外参数的复制提供程序的示例

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Use replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> image = client.text_to_image(
...     "An astronaut riding a horse on the moon.",
...     model="black-forest-labs/FLUX.1-schnell",
...     extra_body={"output_quality": 100},
... )
>>> image.save("astronaut.png")
```

#### text_to_speech[[huggingface_hub.InferenceClient.text_to_speech]]

```python
text_to_speech(text: str, model: str | None = None, do_sample: bool | None = None, early_stopping: typing.Union[bool, ForwardRef('TextToSpeechEarlyStoppingEnum'), NoneType] = None, epsilon_cutoff: float | None = None, eta_cutoff: float | None = None, max_length: int | None = None, max_new_tokens: int | None = None, min_length: int | None = None, min_new_tokens: int | None = None, num_beam_groups: int | None = None, num_beams: int | None = None, penalty_alpha: float | None = None, temperature: float | None = None, top_k: int | None = None, top_p: float | None = None, typical_p: float | None = None, use_cache: bool | None = None, extra_body: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L2676)

**参数：**

text (`str`) ：要合成的文本。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本转语音模型。默认为无。

do_sample (`bool`, *可选*) ：生成新令牌时是否使用采样而不是贪婪解码。

Early_stopping（`Union[bool, "TextToSpeechEarlyStoppingEnum"]`，*可选*）：控制基于波束的方法的停止条件。

epsilon_cutoff (`float`, *可选*) ：如果设置为严格在 0 和 1 之间浮动，则仅对条件概率大于 epsilon_cutoff 的标记进行采样。在论文中，建议值范围从 3e-4 到 9e-4，具体取决于模型的大小。请参阅[Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)了解更多详情。eta_cutoff (`float`, *可选*) ：eta采样是局部典型采样和epsilon采样的混合。如果设置为严格在 0 和 1 之间浮动，则仅当令牌大于 eta_cutoff 或 sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))) 时才考虑该令牌。后一项直观地是预期的下一个标记概率，按 sqrt(eta_cutoff) 缩放。在论文中，建议值范围从 3e-4 到 2e-3，具体取决于模型的大小。更多详情请参见[Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)。

max_length (`int`, *可选*) ：生成文本的最大长度（以标记为单位），包括输入。

max_new_tokens (`int`, *可选*) ：要生成的最大令牌数量。优先于 max_length。

min_length (`int`, *可选*) ：生成文本的最小长度（以标记为单位），包括输入。

min_new_tokens (`int`, *可选*) ：要生成的最小令牌数量。优先于 min_length。

num_beam_groups (`int`, *可选*) ：将 num_beams 划分成的组数，以确保不同组波束之间的多样性。更多详情请参阅[this paper](https://hf.co/papers/1610.02424)。

num_beams (`int`, *可选*) ：用于波束搜索的波束数量。惩罚_alpha（`float`，*可选*）：该值平衡模型置信度和对比搜索解码中的退化惩罚。

温度（`float`，*可选*）：用于调节下一个令牌概率的值。

top_k (`int`, *可选*) ：为 top-k 过滤保留的最高概率词汇标记的数量。

top_p (`float`, *可选*) : 如果设置为 float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.

typical_p (⟦T475⟧, *optional*) : Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. See ⟦T1040⟧ for more details.

use_cache (⟦T476⟧, *optional*) : Whether the model should use the past last key/values attentions to speed up decoding

extra_body (⟦T477⟧, *optional*) : Additional provider-specific parameters to pass to the model. Refer to the provider's documentation for supported parameters.

**Returns:** ⟦T478⟧

The generated audio.

**Raises:** ⟦T1041⟧ or ⟦T479⟧

- ⟦T1042⟧ -- 
  If the model is unavailable or the request times out.
- ⟦T480⟧ -- 
  If the request fails with an HTTP error status code other than HTTP 503.

Synthesize an audio of a voice pronouncing a given text.

> [!TIP]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：
```py
>>> from pathlib import Path
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> audio = client.text_to_speech("Hello world")
>>> Path("hello_world.flac").write_bytes(audio)
```

直接使用第三方提供商的示例。使用量将在您的复制帐户中计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",
...     api_key="REDACTED",  # Pass your Replicate API key directly
... )
>>> audio = client.text_to_speech(
...     text="Hello world",
...     model="OuteAI/OuteTTS-0.3-500M",
... )
>>> Path("hello_world.flac").write_bytes(audio)
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",
...     api_key="hf_...",  # Pass your HF token
... )
>>> audio =client.text_to_speech(
...     text="Hello world",
...     model="OuteAI/OuteTTS-0.3-500M",
... )
>>> Path("hello_world.flac").write_bytes(audio)
```

使用具有额外参数的复制提供程序的示例

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Use replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> audio = client.text_to_speech(
...     "Hello, my name is Kororo, an awesome text-to-speech model.",
...     model="hexgrad/Kokoro-82M",
...     extra_body={"voice": "af_nicole"},
... )
>>> Path("hello.flac").write_bytes(audio)
```

fal.ai 上使用“YuE-s1-7B-anneal-en-cot”的音乐生成示例

```py
>>> from huggingface_hub import InferenceClient
>>> lyrics = '''
... [verse]
... In the town where I was born
... Lived a man who sailed to sea
... And he told us of his life
... In the land of submarines
... So we sailed on to the sun
... 'Til we found a sea of green
... And we lived beneath the waves
... In our yellow submarine

... [chorus]
... We all live in a yellow submarine
... Yellow submarine, yellow submarine
... We all live in a yellow submarine
... Yellow submarine, yellow submarine
... '''
>>> genres = "pavarotti-style tenor voice"
>>> client = InferenceClient(
...     provider="fal-ai",
...     model="m-a-p/YuE-s1-7B-anneal-en-cot",
...     api_key=...,
... )
>>> audio = client.text_to_speech(lyrics, extra_body={"genres": genres})
>>> with open("output.mp3", "wb") as f:
...     f.write(audio)
```

#### 文本到视频[[huggingface_hub.InferenceClient.文本到视频]]

```python
text_to_video(prompt: str, model: str | None = None, guidance_scale: float | None = None, negative_prompt: list[str] | None = None, num_frames: float | None = None, num_inference_steps: int | None = None, seed: int | None = None, extra_body: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L2579)

**参数：**

提示 (`str`) ：生成视频的提示。model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本到视频模型。默认为无。

Guiding_scale (`float`, *可选*) ：较高的指导比例值会鼓励模型生成与文本提示紧密相关的视频，但值太高可能会导致饱和和其他伪影。

negative_prompt（`list[str]`，*可选*）：一个或多个提示来指导视频生成中不包含的内容。

num_frames (`float`, *可选*) ：num_frames 参数决定生成多少视频帧。

num_inference_steps (`int`, *可选*) ：去噪步骤的数量。更多的去噪步骤通常会带来更高质量的视频，但代价是推理速度变慢。

种子（`int`，*可选*）：随机数生成器的种子。

extra_body (`dict[str, Any]`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**退货：** `bytes`

生成的视频。

根据给定的文本生成视频。> [!提示]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：

直接使用第三方提供商的示例。使用费用将通过您的 fal.ai 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="fal-ai",  # Using fal.ai provider
...     api_key="REDACTED",  # Pass your fal.ai API key
... )
>>> video = client.text_to_video(
...     "A majestic lion running in a fantasy forest",
...     model="tencent/HunyuanVideo",
... )
>>> with open("lion.mp4", "wb") as file:
...     file.write(video)
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Using replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> video = client.text_to_video(
...     "A cat running in a park",
...     model="genmo/mochi-1-preview",
... )
>>> with open("cat.mp4", "wb") as file:
...     file.write(video)
```

#### token_classification[[huggingface_hub.InferenceClient.token_classification]]

```python
token_classification(text: str, model: str | None = None, aggregation_strategy: typing.Optional[ForwardRef('TokenClassificationAggregationStrategy')] = None, ignore_labels: list[str] | None = None, stride: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L2884)

**参数：**

text (`str`) ：要分类的字符串。

model (`str`, *可选*) ：用于标记分类任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的令牌分类模型。默认为无。

aggregation_strategy (`"TokenClassificationAggregationStrategy"`, *可选*) ：用于基于模型预测融合令牌的策略

ignore_labels (`list[str`, *可选*) : 要忽略的标签列表

stride (`int`, *可选*) ：分割输入文本时块之间重叠标记的数量。

**退货：** `list[TokenClassificationOutputElement]`

[TokenClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TokenClassificationOutputElement) 项目列表，包含实体组、置信度得分、单词、开始和结束索引。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对给定文本执行标记分类。
通常用于句子解析，无论是语法解析还是命名实体识别 (NER)，以理解文本中包含的关键字。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
[
    TokenClassificationOutputElement(
        entity_group='PER',
        score=0.9971321225166321,
        word='Sarah Jessica Parker',
        start=11,
        end=31,
    ),
    TokenClassificationOutputElement(
        entity_group='PER',
        score=0.9773476123809814,
        word='Jessica',
        start=52,
        end=59,
    )
]
```

#### 翻译[[huggingface_hub.InferenceClient.translation]]

```python
translation(text: str, model: str | None = None, src_lang: str | None = None, tgt_lang: str | None = None, clean_up_tokenization_spaces: bool | None = None, truncation: typing.Optional[ForwardRef('TranslationTruncationStrategy')] = None, generate_parameters: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L2959)

**参数：**

text (`str`) ：要翻译的字符串。

model (`str`, *可选*) ：用于翻译任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的翻译模型。默认为无。

src_lang (`str`, *可选*) ：文本的源语言。对于可以翻译多种语言的模型是必需的。

tgt_lang (`str`, *可选*) ：要翻译的目标语言。对于可以翻译成多种语言的模型是必需的。

clean_up_tokenization_spaces (`bool`, *可选*) ：是否清理文本输出中潜在的额外空格。截断（`"TranslationTruncationStrategy"`，*可选*）：要使用的截断策略。

generate_parameters（`dict[str, Any]`，*可选*）：文本生成算法的附加参数化。

**退货：** [TranslationOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TranslationOutput)

生成的翻译文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError` 或 ``ValueError``

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。
- ``ValueError`` -- 
  如果仅提供 `src_lang` 和 `tgt_lang` 参数之一。

将文本从一种语言转换为另一种语言。

查看 https://huggingface.co/tasks/translation 了解有关如何选择最佳模型的更多信息
您的具体用例。源语言和目标语言通常取决于模型。
但是，可以为某些模型指定源语言和目标语言。如果您正在使用这些模型之一，
您可以使用 `src_lang` 和 `tgt_lang` 参数来传递相关信息。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.translation("My name is Wolfgang and I live in Berlin")
'Mein Name ist Wolfgang und ich lebe in Berlin.'
>>> client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
TranslationOutput(translation_text='Je m'appelle Wolfgang et je vis à Berlin.')
```

指定语言：
```py
>>> client.translation("My name is Sarah Jessica Parker but you can call me Jessica", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
"Mon nom est Sarah Jessica Parker mais vous pouvez m'appeler Jessica"
```

#### Visual_question_answering[[huggingface_hub.InferenceClient.visual_question_answering]]

```python
visual_question_answering(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], question: str, model: str | None = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L3048)

**参数：**image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：上下文的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

问题（`str`）：需要回答的问题。

model (`str`, *可选*) ：用于视觉问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的视觉问答模型。默认为无。

top_k (`int`, *可选*) ：返回的答案数量（将按可能性顺序选择）。请注意，如果上下文中没有足够的选项，我们将返回少于 topk 的答案。

**返回：** `list[VisualQuestionAnsweringOutputElement]`

包含预测标签和相关概率的[VisualQuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.VisualQuestionAnsweringOutputElement)项目列表。

**提高：** ``InferenceTimeoutError`` or `HfHubHTTPError`

- ``InferenceTimeoutError`` -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据图像回答开放式问题。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.visual_question_answering(
...     image="https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
...     question="What is the animal doing?"
... )
[
    VisualQuestionAnsweringOutputElement(score=0.778609573841095, answer='laying down'),
    VisualQuestionAnsweringOutputElement(score=0.6957435607910156, answer='sitting'),
]
```

#### Zero_shot_classification[[huggingface_hub.InferenceClient.zero_shot_classification]]

```python
zero_shot_classification(text: str, candidate_labels: list, multi_label: bool | None = False, hypothesis_template: str | None = None, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L3107)

**参数：**text (`str`) ：要分类的输入文本。

候选标签（`list[str]`）：将文本分类到的可能的类标签集。

labels (`list[str]`, *可选*) ：（已弃用）字符串列表。每个字符串都是输入文本的可能标签的语言表达。

multi_label (`bool`, *可选*) : 多个候选标签是否可以为真。如果为假，则对分数进行归一化，以使每个序列的标签似然之和为 1。如果为真，则将标签视为独立，并对每个候选者的概率进行归一化。

假设_模板（`str`，*可选*）：与`candidate_labels`结合使用的句子，通过用候选标签替换占位符来尝试文本分类。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。如果未提供，将使用默认推荐的零样本分类模型。

**返回：** `list[ZeroShotClassificationOutputElement]`

包含预测标签及其置信度的[ZeroShotClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ZeroShotClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

提供文本和一组候选标签作为输入，以对输入文本进行分类。

`multi_label=False` 的示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> text = (
...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
...     " mysteries when he went for a run up a hill in Nice, France."
... )
>>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
>>> client.zero_shot_classification(text, labels)
[
    ZeroShotClassificationOutputElement(label='scientific discovery', score=0.7961668968200684),
    ZeroShotClassificationOutputElement(label='space & cosmos', score=0.18570658564567566),
    ZeroShotClassificationOutputElement(label='microbiology', score=0.00730885099619627),
    ZeroShotClassificationOutputElement(label='archeology', score=0.006258360575884581),
    ZeroShotClassificationOutputElement(label='robots', score=0.004559356719255447),
]
>>> client.zero_shot_classification(text, labels, multi_label=True)
[
    ZeroShotClassificationOutputElement(label='scientific discovery', score=0.9829297661781311),
    ZeroShotClassificationOutputElement(label='space & cosmos', score=0.755190908908844),
    ZeroShotClassificationOutputElement(label='microbiology', score=0.0005462635890580714),
    ZeroShotClassificationOutputElement(label='archeology', score=0.00047131875180639327),
    ZeroShotClassificationOutputElement(label='robots', score=0.00030448526376858354),
]
```

使用 `multi_label=True` 和自定义 `hypothesis_template` 的示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()
>>> client.zero_shot_classification(
...    text="I really like our dinner and I'm very happy. I don't like the weather though.",
...    labels=["positive", "negative", "pessimistic", "optimistic"],
...    multi_label=True,
...    hypothesis_template="This text is {} towards the weather"
... )
[
    ZeroShotClassificationOutputElement(label='negative', score=0.9231801629066467),
    ZeroShotClassificationOutputElement(label='pessimistic', score=0.8760990500450134),
    ZeroShotClassificationOutputElement(label='optimistic', score=0.0008674879791215062),
    ZeroShotClassificationOutputElement(label='positive', score=0.0005250611575320363)
]
```

#### Zero_shot_image_classification[[huggingface_hub.InferenceClient.zero_shot_image_classification]]

```python
zero_shot_image_classification(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], candidate_labels: list, model: str | None = None, hypothesis_template: str | None = None, labels: list = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_client.py#L3210)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：标题的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

候选标签（`list[str]`）：该图像的候选标签

labels (`list[str]`, *可选*) ：（已弃用）字符串可能标签的列表。必须至少有 2 个标签。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。如果未提供，将使用默认推荐的零样本图像分类模型。假设_模板（`str`，*可选*）：与`candidate_labels`结合使用的句子，通过用候选标签替换占位符来尝试图像分类。

**返回：** `list[ZeroShotImageClassificationOutputElement]`

包含预测标签及其置信度的[ZeroShotImageClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ZeroShotImageClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

提供输入图像和文本标签来预测图像的文本标签。

示例：
```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient()

>>> client.zero_shot_image_classification(
...     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
...     labels=["dog", "cat", "horse"],
... )
[ZeroShotImageClassificationOutputElement(label='dog', score=0.956),...]
```

## 异步推理客户端[[huggingface_hub.AsyncInferenceClient]]

还提供了基于`asyncio`和`httpx`的异步版本客户端。

#### Huggingface_hub.AsyncInferenceClient[[huggingface_hub.AsyncInferenceClient]]

```python
huggingface_hub.AsyncInferenceClient(model: str | None = None, provider: typing.Union[typing.Literal['baseten', 'cerebras', 'cohere', 'deepinfra', 'fal-ai', 'featherless-ai', 'fireworks-ai', 'groq', 'hf-inference', 'novita', 'nscale', 'openai', 'ovhcloud', 'publicai', 'replicate', 'scaleway', 'together', 'wavespeed', 'zai-org'], typing.Literal['auto'], NoneType] = None, token: str | None = None, timeout: float | None = None, headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None, bill_to: str | None = None, base_url: str | None = None, api_key: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L113)

**参数：**model (`str`, `optional`) ：运行推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，例如`meta-llama/Meta-Llama-3-8B-Instruct` 或已部署推理端点的 URL。默认为“无”，在这种情况下，系统会自动为任务选择推荐模型。注：为了更好地兼容OpenAI客户端，`model`已别名为`base_url`。这两个论点是互斥的。如果将 URL 作为 `model` 或 `base_url` 传递以完成聊天，则 `(/v1)/chat/completions` 后缀路径将附加到 URL。

提供者（`str`，*可选*）：用于推理的提供者的名称。可以是 `"baseten"`、`"cerebras"`、`"cohere"`、`"deepinfra"`、`"fal-ai"`、`"featherless-ai"`、`"fireworks-ai"`、`"groq"`、`"hf-inference"`、 `"novita"`、`"nscale"`、`"openai"`、`"ovhcloud"`、`"publicai"`、`"replicate"`、`"scaleway"`、`"together"`、`"wavespeed"` 或`"zai-org"`。默认为“auto”：自动路由，默认为“最快”的提供商；您可以在 https://hf.co/settings/inference-providers 切换到“最便宜”或“首选”提供商订单。如果 model 是 URL 或传递了 `base_url`，则不使用 `provider`。令牌（`str`，*可选*）：拥抱脸部令牌。如果未提供，将默认为本地保存的令牌。注：为了更好地兼容OpenAI客户端，`token`已别名为`api_key`。这两个参数是互斥的并且具有完全相同的行为。

timeout (`float`, `optional`) ：等待服务器响应的最大秒数。默认为 None，这意味着它将循环直到服务器可用。

headers (`dict[str, str]`, `optional`) ：发送到服务器的附加标头。默认情况下，仅发送授权和用户代理标头。该字典中的值将覆盖默认值。

bill_to (`str`, `optional`) ：用于请求的计费帐户。默认情况下，请求在用户帐户上计费。请求只能向用户所属且已订阅 Enterprise Hub 的组织计费。

cookies (`dict[str, str]`, `optional`) ：发送到服务器的附加cookie。

base_url (`str`, `optional`) ：运行推理的基本 URL。这是来自 `model` 的重复参数，以使 [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 遵循与 `openai.OpenAI` 客户端相同的模式。如果设置了`model`则无法使用。默认为无。api_key (`str`, `optional`) ：用于身份验证的令牌。这是来自 `token` 的重复参数，以使 [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 遵循与 `openai.OpenAI` 客户端相同的模式。如果设置了`token`则无法使用。默认为无。

初始化一个新的推理客户端。

[InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 旨在提供统一的推理体验。客户端可以使用
与（免费）推理 API、自托管推理端点或第三方推理提供商无缝连接。

#### audio_classification[[huggingface_hub.AsyncInferenceClient.audio_classification]]

```python
audio_classification(audio: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, top_k: int | None = None, function_to_apply: typing.Optional[ForwardRef('AudioClassificationOutputTransform')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L319)

**参数：**

audio (Union[str, Path, bytes, BinaryIO]) ：要分类的音频内容。它可以是原始音频字节、本地音频文件或指向音频文件的 URL。

model（`str`，*可选*）：用于音频分类的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的音频分类模型。

top_k (`int`, *可选*) ：指定后，将输出限制为前 K 个最可能的类。function_to_apply (`"AudioClassificationOutputTransform"`, *可选*) ：应用于模型输出以检索分数的函数。

**返回：** `list[AudioClassificationOutputElement]`

包含预测标签及其置信度的[AudioClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.AudioClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对提供的音频内容执行音频分类。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.audio_classification("audio.flac")
[
    AudioClassificationOutputElement(score=0.4976358711719513, label='hap'),
    AudioClassificationOutputElement(score=0.3677836060523987, label='neu'),
    ...
]
```

#### audio_to_audio[[huggingface_hub.AsyncInferenceClient.audio_to_audio]]

```python
audio_to_audio(audio: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L377)

**参数：**

audio (Union[str, Path, bytes, BinaryIO]) ：模型的音频内容。它可以是原始音频字节、本地音频文件或指向音频文件的 URL。

model (`str`, *可选*) ：模型可以是任何接受一个音频文件并返回另一个音频文件的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的audio_to_audio模型。

**返回：** `list[AudioToAudioOutputElement]`

[AudioToAudioOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.AudioToAudioOutputElement) 项目列表，其中包含 blob 中的音频标签、内容类型和音频内容。

**提高：** ``InferenceTimeoutError`` or `HfHubHTTPError`- ``InferenceTimeoutError`` -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据模型执行与音频到音频相关的多项任务（例如：语音增强、源分离）。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> audio_output = await client.audio_to_audio("audio.flac")
>>> async for i, item in enumerate(audio_output):
>>>     with open(f"output_{i}.flac", "wb") as f:
            f.write(item.blob)
```

#### 自动语音识别[[huggingface_hub.AsyncInferenceClient.自动语音识别]]

```python
automatic_speech_recognition(audio: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, extra_body: dict | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L430)

**参数：**

audio (Union[str, Path, bytes, BinaryIO]) ：要转录的内容。它可以是原始音频字节、本地音频文件或音频文件的 URL。

model（`str`，*可选*）：用于 ASR 的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用 ASR 的默认推荐模型。

extra_body (`dict`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**返回：** [AutomaticSpeechRecognitionOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.AutomaticSpeechRecognitionOutput)

包含转录文本和可选的时间戳块的项目。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对给定的音频内容执行自动语音识别（ASR 或音频转文本）。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.automatic_speech_recognition("hello_world.flac").text
"hello world"
```

#### chat_completion[[huggingface_hub.AsyncInferenceClient.chat_completion]]

```python
chat_completion(messages: list, model: str | None = None, stream: bool = False, frequency_penalty: float | None = None, logit_bias: list[float] | None = None, logprobs: bool | None = None, max_tokens: int | None = None, n: int | None = None, presence_penalty: float | None = None, response_format: typing.Union[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputResponseFormatText, huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputResponseFormatJSONSchema, huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputResponseFormatJSONObject, NoneType] = None, seed: int | None = None, stop: list[str] | None = None, stream_options: huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputStreamOptions | None = None, temperature: float | None = None, tool_choice: typing.Union[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputToolChoiceClass, ForwardRef('ChatCompletionInputToolChoiceEnum'), NoneType] = None, tool_prompt: str | None = None, tools: list[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputTool] | None = None, top_logprobs: int | None = None, top_p: float | None = None, extra_body: dict | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L558)

**参数：**

消息（[ChatCompletionInputMessage](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputMessage)列表）：由角色和内容对组成的对话历史记录。

model (`str`, *可选*) ：用于聊天完成的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用基于聊天的文本生成的默认推荐模型。有关更多详细信息，请参阅 https://huggingface.co/tasks/text- Generation 。如果`model`是模型ID，它将作为`model`参数传递到服务器。如果您想在请求负载中设置`model`时定义自定义URL，则必须在初始化[InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)时设置`base_url`。

Frequency_penalty (`float`, *可选*) ：根据迄今为止文本中的现有频率对新标记进行惩罚。范围：[-2.0，2.0]。默认为 0.0。logit_bias (`list[float]`, *可选*) ：调整特定标记出现在生成的输出中的可能性。

logprobs (`bool`, *可选*) ：是否返回输出标记的对数概率。如果为 true，则返回消息内容中返回的每个输出标记的对数概率。

max_tokens (`int`, *可选*) ：响应中允许的最大令牌数。默认为 100。

n (`int`, *可选*) ：为每个提示生成的完成数。

Presence_penalty (`float`, *可选*) ：-2.0 到 2.0 之间的数字。正值根据新标记目前是否出现在文本中来对其进行惩罚，从而增加模型谈论新主题的可能性。

response_format (`ChatCompletionInputGrammarType()`, *可选*) ：语法约束。可以是 JSONSchema 或正则表达式。

种子（可选`int`，*可选*）：可重复控制流的种子。默认为无。

stop (`list[str]`, *可选*) ：最多四个字符串，触发响应结束。默认为无。

流（`bool`，*可选*）：启用实时响应流。默认为 False。

Stream_options ([ChatCompletionInputStreamOptions](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputStreamOptions), *可选*) ：流式完成的选项。温度（`float`，*可选*）：控制生成的随机性。较低的值可确保较少的随机完成。范围：[0, 2]。默认为 1.0。

top_logprobs (`int`, *可选*) ：0 到 5 之间的整数，指定每个标记位置最有可能返回的标记数量，每个标记都有一个关联的对数概率。如果使用此参数，则 logprobs 必须设置为 true。

top_p (`float`, *可选*) ：最有可能进行采样的下一个单词的分数。必须介于 0 和 1 之间。默认为 1.0。

tool_choice（[ChatCompletionInputToolChoiceClass](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputToolChoiceClass)或`ChatCompletionInputToolChoiceEnum()`，*可选*）：用于完成的工具。默认为“自动”。

tool_prompt (`str`, *可选*) ：附加在工具之前的提示。

工具（[ChatCompletionInputTool](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputTool)列表，*可选*）：模型可能调用的工具列表。目前，仅支持函数作为工具。使用它来提供模型可以为其生成 JSON 输入的函数列表。

extra_body (`dict`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**返回：** [ChatCompletionOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionOutput) 或 [ChatCompletionStreamOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionStreamOutput) 的可迭代从服务器返回的生成文本：
- 如果`stream=False`，生成的文本将作为[ChatCompletionOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionOutput)（默认）返回。
- 如果`stream=True`，则生成的文本将作为[ChatCompletionStreamOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionStreamOutput) 的序列逐个令牌返回。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

一种使用指定语言模型完成对话的方法。

> [!提示]
> 为了与 OpenAI 客户端兼容，`client.chat_completion` 方法别名为 `client.chat.completions.create`。
> 输入和输出完全相同，使用任一语法都会产生相同的结果。
> 查看[Inference guide](https://huggingface.co/docs/huggingface_hub/guides/inference#openai-compatibility)
> 有关 OpenAI 兼容性的更多详细信息。

> [!提示]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：

```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> messages = [{"role": "user", "content": "What is the capital of France?"}]
>>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
>>> await client.chat_completion(messages, max_tokens=100)
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason='eos_token',
            index=0,
            message=ChatCompletionOutputMessage(
                role='assistant',
                content='The capital of France is Paris.',
                name=None,
                tool_calls=None
            ),
            logprobs=None
        )
    ],
    created=1719907176,
    id='',
    model='meta-llama/Meta-Llama-3-8B-Instruct',
    object='text_completion',
    system_fingerprint='2.0.4-sha-f426a33',
    usage=ChatCompletionOutputUsage(
        completion_tokens=8,
        prompt_tokens=17,
        total_tokens=25
    )
)
```

使用流式传输的示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> messages = [{"role": "user", "content": "What is the capital of France?"}]
>>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
>>> async for token in await client.chat_completion(messages, max_tokens=10, stream=True):
...     print(token)
ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content='The', role='assistant'), index=0, finish_reason=None)], created=1710498504)
ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' capital', role='assistant'), index=0, finish_reason=None)], created=1710498504)
(...)
ChatCompletionStreamOutput(choices=[ChatCompletionStreamOutputChoice(delta=ChatCompletionStreamOutputDelta(content=' may', role='assistant'), index=0, finish_reason=None)], created=1710498504)
```

使用 OpenAI 语法的示例：
```py
# Must be run in an async context
# instead of `from openai import OpenAI`
from huggingface_hub import AsyncInferenceClient

# instead of `client = OpenAI(...)`
client = AsyncInferenceClient(
    base_url=...,
    api_key=...,
)

output = await client.chat.completions.create(
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
```

直接使用带有额外（特定于提供商的）参数的第三方提供商的示例。使用量将在您的 Together AI 帐户上计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="together",  # Use Together AI provider
...     api_key="REDACTED",  # Pass your Together API key directly
... )
>>> client.chat_completion(
...     model="meta-llama/Meta-Llama-3-8B-Instruct",
...     messages=[{"role": "user", "content": "What is the capital of France?"}],
...     extra_body={"safety_model": "Meta-Llama/Llama-Guard-7b"},
... )
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="novita",  # Use Novita provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> client.chat_completion(
...     model="meta-llama/Meta-Llama-3-8B-Instruct",
...     messages=[{"role": "user", "content": "What is the capital of France?"}],
... )
```使用图像 + 文本作为输入的示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient

# provide a remote URL
>>> image_url ="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
# or a base64-encoded image
>>> image_path = "/path/to/image.jpeg"
>>> with open(image_path, "rb") as f:
...     base64_image = base64.b64encode(f.read()).decode("utf-8")
>>> image_url = f"data:image/jpeg;base64,{base64_image}"

>>> client = AsyncInferenceClient("meta-llama/Llama-3.2-11B-Vision-Instruct")
>>> output = await client.chat.completions.create(
...     messages=[
...         {
...             "role": "user",
...             "content": [
...                 {
...                     "type": "image_url",
...                     "image_url": {"url": image_url},
...                 },
...                 {
...                     "type": "text",
...                     "text": "Describe this image in one sentence.",
...                 },
...             ],
...         },
...     ],
... )
>>> output
The image depicts the iconic Statue of Liberty situated in New York Harbor, New York, on a clear day.
```

使用工具示例：
```py
# Must be run in an async context
>>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
>>> messages = [
...     {
...         "role": "system",
...         "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
...     },
...     {
...         "role": "user",
...         "content": "What's the weather like the next 3 days in San Francisco, CA?",
...     },
... ]
>>> tools = [
...     {
...         "type": "function",
...         "function": {
...             "name": "get_current_weather",
...             "description": "Get the current weather",
...             "parameters": {
...                 "type": "object",
...                 "properties": {
...                     "location": {
...                         "type": "string",
...                         "description": "The city and state, e.g. San Francisco, CA",
...                     },
...                     "format": {
...                         "type": "string",
...                         "enum": ["celsius", "fahrenheit"],
...                         "description": "The temperature unit to use. Infer this from the users location.",
...                     },
...                 },
...                 "required": ["location", "format"],
...             },
...         },
...     },
...     {
...         "type": "function",
...         "function": {
...             "name": "get_n_day_weather_forecast",
...             "description": "Get an N-day weather forecast",
...             "parameters": {
...                 "type": "object",
...                 "properties": {
...                     "location": {
...                         "type": "string",
...                         "description": "The city and state, e.g. San Francisco, CA",
...                     },
...                     "format": {
...                         "type": "string",
...                         "enum": ["celsius", "fahrenheit"],
...                         "description": "The temperature unit to use. Infer this from the users location.",
...                     },
...                     "num_days": {
...                         "type": "integer",
...                         "description": "The number of days to forecast",
...                     },
...                 },
...                 "required": ["location", "format", "num_days"],
...             },
...         },
...     },
... ]

>>> response = await client.chat_completion(
...     model="meta-llama/Meta-Llama-3-70B-Instruct",
...     messages=messages,
...     tools=tools,
...     tool_choice="auto",
...     max_tokens=500,
... )
>>> response.choices[0].message.tool_calls[0].function
ChatCompletionOutputFunctionDefinition(
    arguments={
        'location': 'San Francisco, CA',
        'format': 'fahrenheit',
        'num_days': 3
    },
    name='get_n_day_weather_forecast',
    description=None
)
```

使用response_format的示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
>>> messages = [
...     {
...         "role": "user",
...         "content": "I saw a puppy a cat and a raccoon during my bike ride in the park. What did I see and when?",
...     },
... ]
>>> response_format = {
...     "type": "json",
...     "value": {
...         "properties": {
...             "location": {"type": "string"},
...             "activity": {"type": "string"},
...             "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
...             "animals": {"type": "array", "items": {"type": "string"}},
...         },
...         "required": ["location", "activity", "animals_seen", "animals"],
...     },
... }
>>> response = await client.chat_completion(
...     messages=messages,
...     response_format=response_format,
...     max_tokens=500,
... )
>>> response.choices[0].message.content
'{

y": "bike ride",
": ["puppy", "cat", "raccoon"],
_seen": 3,
n": "park"}'
```

#### 关闭[[huggingface_hub.AsyncInferenceClient.close]]

```python
close()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L240)

关闭客户端。

当使用客户端作为上下文管理器时，会自动调用此方法。

#### document_question_answering[[huggingface_hub.AsyncInferenceClient.document_question_answering]]

```python
document_question_answering(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], question: str, model: str | None = None, doc_stride: int | None = None, handle_impossible_answer: bool | None = None, lang: str | None = None, max_answer_len: int | None = None, max_question_len: int | None = None, max_seq_len: int | None = None, top_k: int | None = None, word_boxes: list[list[float] | str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L966)

**参数：**

image (`Union[str, Path, bytes, BinaryIO]`) ：上下文的输入图像。它可以是原始字节、图像文件或在线图像的 URL。

问题（`str`）：需要回答的问题。

model (`str`, *可选*) ：用于文档问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果不提供，将使用默认推荐的文档问答模型。默认为无。

doc_stride (`int`, *可选*) ：如果文档中的单词太长而无法满足模型的问题，它将被分成几个有重叠的块。该参数控制重叠的大小。

handle_impossible_answer (`bool`, *可选*) : 是否接受不可能作为答案lang (`str`, *可选*) ：运行 OCR 时使用的语言。默认为英文。

max_answer_len (`int`, *可选*) ：预测答案的最大长度（例如，仅考虑长度较短的答案）。

max_question_len (`int`, *可选*) ：标记化后问题的最大长度。如果需要，它将被截断。

max_seq_len (`int`, *可选*) ：传递给模型的每个块的标记中总句子（上下文+问题）的最大长度。如果需要，上下文将被分割成几个块（使用 doc_stride 作为重叠）。

top_k (`int`, *可选*) ：要返回的答案数量（将按可能性顺序选择）。如果上下文中没有足够的可用选项，则可以返回少于 top_k 的答案。

word_boxes (`list[Union[list[float], str`, *可选*) ：单词和边界框的列表（标准化 0->1000）。如果提供，推理将跳过 OCR 步骤并使用提供的边界框。

**返回：** `list[DocumentQuestionAnsweringOutputElement]`

[DocumentQuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.DocumentQuestionAnsweringOutputElement) 项目列表，其中包含预测标签、相关概率、单词 ID 和页码。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

回答有关文档图像的问题。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.document_question_answering(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")
[DocumentQuestionAnsweringOutputElement(answer='us-001', end=16, score=0.9999666213989258, start=16)]
```

#### feature_extraction[[huggingface_hub.AsyncInferenceClient.feature_extraction]]

```python
feature_extraction(text: str | list[str], normalize: bool | None = None, prompt_name: str | None = None, truncate: bool | None = None, truncation_direction: typing.Optional[typing.Literal['left', 'right']] = None, dimensions: int | None = None, encoding_format: typing.Optional[typing.Literal['float', 'base64']] = None, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1054)

**参数：**

text (*str* 或 *list[str]*) ：要嵌入的文本或文本列表。

model (*str*, *可选*) ：用于特征提取任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的特征提取模型。默认为无。

规范化（*bool*，*可选*）：是否规范化嵌入。仅在由文本嵌入推理支持的服务器上可用。Prompt_name (*str*, *可选*) ：用于编码的提示的名称。如果未设置，则不会应用任何提示。必须是 *Sentence Transformers* 配置 *prompts* 字典中的键。例如，如果 `prompt_name` 是“query”，`prompts` 是 &lcub;"query": "query: ",...}，则句子“法国的首都是什么？”将被编码为“查询：法国的首都是哪里？”因为提示文本将被添加到任何要编码的文本之前。

truncate (*bool*, *可选*) ：是否截断嵌入。仅在由文本嵌入推理支持的服务器上可用。

truncation_direction (*Literal["left", "right"]*, *可选*) ：当传递 *truncate=True* 时，应截断输入的哪一侧。

维度（*int*，*可选*）：生成的输出嵌入应具有的维度数。仅适用于兼容 OpenAI 的嵌入端点。

coding_format (*Literal["float", "base64"]*, *可选*) ：输出嵌入的格式。 “float”或“base64”。仅适用于兼容 OpenAI 的嵌入端点。

**退货：** `*np.ndarray*`

将输入文本表示为 float32 numpy 数组的嵌入。**引发：** [*InferenceTimeoutError*] 或 [*HfHubHTTPError*]

- [*推理超时错误*] -- 
  如果模型不可用或请求超时。
- [*HfHubHTTPError*] -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

为给定文本或一批文本生成嵌入。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.feature_extraction("Hi, who are you?")
array([[ 2.424802  ,  2.93384   ,  1.1750331 , ...,  1.240499, -0.13776633, -0.7889173 ],
[-0.42943227, -0.6364878 , -1.693462  , ...,  0.41978157, -2.4336355 ,  0.6162071 ],
...,
[ 0.28552425, -0.928395  , -1.2077185 , ...,  0.76810825, -2.1069427 ,  0.6236161 ]], dtype=float32)
```

#### fill_mask[[huggingface_hub.AsyncInferenceClient.fill_mask]]

```python
fill_mask(text: str, model: str | None = None, targets: list[str] | None = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1138)

**参数：**

text (`str`) ：要填充的字符串，必须包含 [MASK] 标记（检查模型卡以获取掩码的确切名称）。

model (`str`, *可选*) ：用于填充遮罩任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的填充掩模模型。

目标（`list[str`，*可选*）：通过时，模型会将分数限制为通过的目标，而不是在整个词汇表中查找。如果提供的目标不在模型词汇中，它们将被标记化，并且将使用第一个生成的标记（带有警告，并且可能会更慢）。

top_k (`int`, *可选*) ：传递时，覆盖要返回的预测数。

**返回：** `list[FillMaskOutputElement]`包含预测标签的[FillMaskOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.FillMaskOutputElement)项目列表，关联
概率、标记参考和完整的文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

用缺失的单词填补一个洞（准确地说是标记）。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.fill_mask("The goal of life is <mask>.")
[
    FillMaskOutputElement(score=0.06897063553333282, token=11098, token_str=' happiness', sequence='The goal of life is happiness.'),
    FillMaskOutputElement(score=0.06554922461509705, token=45075, token_str=' immortality', sequence='The goal of life is immortality.')
]
```

#### get_endpoint_info[[huggingface_hub.AsyncInferenceClient.get_endpoint_info]]

```python
get_endpoint_info(model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L3331)

**参数：**

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

**返回：** `dict[str, Any]`

有关端点的信息。

获取有关已部署端点的信息。

此端点仅在由文本生成推理 (TGI) 或文本嵌入推理 (TEI) 支持的端点上可用。
由`transformers`支持的端点返回空负载。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
>>> await client.get_endpoint_info()
{
    'model_id': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'model_sha': None,
    'model_dtype': 'torch.float16',
    'model_device_type': 'cuda',
    'model_pipeline_tag': None,
    'max_concurrent_requests': 128,
    'max_best_of': 2,
    'max_stop_sequences': 4,
    'max_input_length': 8191,
    'max_total_tokens': 8192,
    'waiting_served_ratio': 0.3,
    'max_batch_total_tokens': 1259392,
    'max_waiting_tokens': 20,
    'max_batch_size': None,
    'validation_workers': 32,
    'max_client_batch_size': 4,
    'version': '2.0.2',
    'sha': 'dccab72549635c7eb5ddb17f43f0b7cdff07c214',
    'docker_label': 'sha-dccab72'
}
```

#### health_check[[huggingface_hub.AsyncInferenceClient.health_check]]

```python
health_check(model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L3391)

**参数：**model（`str`，*可选*）：推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

**返回：** `bool`

如果一切正常，则属实。

检查已部署端点的运行状况。

运行状况检查仅适用于由文本生成推理 (TGI) 或文本嵌入推理 (TEI) 提供支持的推理端点。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient("https://jzgu0buei5.us-east-1.aws.endpoints.huggingface.cloud")
>>> await client.health_check()
True
```

#### image_classification[[huggingface_hub.AsyncInferenceClient.image_classification]]

```python
image_classification(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, function_to_apply: typing.Optional[ForwardRef('ImageClassificationOutputTransform')] = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1195)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：要分类的图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model（`str`，*可选*）：用于图像分类的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的图像分类模型。

function_to_apply (`"ImageClassificationOutputTransform"`, *可选*) ：应用于模型输出以检索分数的函数。

top_k (`int`, *可选*) ：指定后，将输出限制为前 K 个最可能的类。

**返回：** `list[ImageClassificationOutputElement]`包含预测标签和相关概率的[ImageClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ImageClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定的模型对给定图像执行图像分类。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
[ImageClassificationOutputElement(label='Blenheim spaniel', score=0.9779096841812134), ...]
```

#### image_segmentation[[huggingface_hub.AsyncInferenceClient.image_segmentation]]

```python
image_segmentation(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, mask_threshold: float | None = None, overlap_mask_area_threshold: float | None = None, subtask: typing.Optional[ForwardRef('ImageSegmentationSubtask')] = None, threshold: float | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1246)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：要分割的图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model（`str`，*可选*）：用于图像分割的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的图像分割模型。

mask_threshold (`float`, *可选*) ：将预测掩码转换为二进制值时使用的阈值。

overlap_mask_area_threshold（`float`，*可选*）：掩码重叠阈值，以消除小的、断开的段。

子任务（`"ImageSegmentationSubtask"`，*可选*）：要执行的分割任务，具体取决于模型功能。阈值（`float`，*可选*）：过滤掉预测掩模的概率阈值。

**退货：** `list[ImageSegmentationOutputElement]`

包含分段掩码和关联属性的[ImageSegmentationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ImageSegmentationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定的模型对给定图像执行图像分割。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.image_segmentation("cat.jpg")
[ImageSegmentationOutputElement(score=0.989008, label='LABEL_184', mask=<PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>), ...]
```

#### image_to_image[[huggingface_hub.AsyncInferenceClient.image_to_image]]

```python
image_to_image(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], prompt: str | None = None, negative_prompt: str | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, model: str | None = None, target_size: huggingface_hub.inference._generated.types.image_to_image.ImageToImageTargetSize | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1315)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：用于翻译的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

提示（`str`，*可选*）：指导图像生成的文本提示。

negative_prompt (`str`, *可选*) ：一个提示，用于指导图像生成中不包含哪些内容。

num_inference_steps (`int`, *可选*) ：用于扩散模型。去噪步数。更多的去噪步骤通常会带来更高质量的图像，但代价是推理速度变慢。guide_scale（`float`，*可选*）：用于扩散模型。较高的引导比例值会鼓励模型生成与文本提示紧密相关的图像，但代价是图像质量较低。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

target_size (`ImageToImageTargetSize`, *可选*) ：输出图像的大小（以像素为单位）。仅某些提供商和特定型号支持此参数。当不支持时它将被忽略。

**退货：** `Image`

翻译后的图像。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定模型执行图像到图像的转换。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> image = await client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
>>> image.save("tiger.jpg")
```

#### image_to_text[[huggingface_hub.AsyncInferenceClient.image_to_text]]

```python
image_to_text(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1472)

**参数：**image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：标题的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

**返回：** [ImageToTextOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ImageToTextOutput)

生成的文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

获取输入图像并返回文本。

根据您的用例（图像字幕、光学字符识别），模型可以有非常不同的输出
（OCR）、Pix2Struct 等）。请查看型号卡以了解有关型号特性的更多信息。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.image_to_text("cat.jpg")
'a cat standing in a grassy field '
>>> await client.image_to_text("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
'a dog laying on the grass next to a flower pot '
```

#### image_to_video[[huggingface_hub.AsyncInferenceClient.image_to_video]]

```python
image_to_video(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, prompt: str | None = None, negative_prompt: str | None = None, num_frames: float | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, seed: int | None = None, target_size: huggingface_hub.inference._generated.types.image_to_video.ImageToVideoTargetSize | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1392)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：用于生成视频的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

提示（`str`，*可选*）：指导视频生成的文本提示。

negative_prompt（`str`，*可选*）：一个提示，指导视频生成中不包含哪些内容。

num_frames (`float`, *可选*) ：num_frames 参数决定生成多少视频帧。

num_inference_steps (`int`, *可选*) ：用于扩散模型。去噪步数。更多的去噪步骤通常会带来更高质量的图像，但代价是推理速度变慢。

guide_scale（`float`，*可选*）：用于扩散模型。较高的指导比例值会鼓励模型生成与文本提示紧密相关的视频，但会降低图像质量。

种子（`int`，*可选*）：用于视频生成的种子。

target_size (`ImageToVideoTargetSize`, *可选*) ：输出视频帧的像素大小。num_inference_steps (`int`, *可选*) ：去噪步骤的数量。更多的去噪步骤通常会带来更高质量的视频，但代价是推理速度变慢。

种子（`int`，*可选*）：随机数生成器的种子。

**返回：** `bytes`

生成的视频。

从输入图像生成视频。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> video = await client.image_to_video("cat.jpg", model="Wan-AI/Wan2.2-I2V-A14B", prompt="turn the cat into a tiger")
>>> with open("tiger.mp4", "wb") as f:
...     f.write(video)
```

#### object_detection[[huggingface_hub.AsyncInferenceClient.object_detection]]

```python
object_detection(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], model: str | None = None, threshold: float | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1519)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) : 用于检测物体的图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

model (`str`，*可选*)：用于对象检测的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的对象检测模型 (DETR)。

阈值（`float`，*可选*）：进行预测所需的概率。

**返回：** `list[ObjectDetectionOutputElement]`

包含边界框和关联属性的 [ObjectDetectionOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ObjectDetectionOutputElement) 项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError` 或 ``ValueError``- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。
- ``ValueError`` -- 
  如果请求输出不是List。

使用指定的模型对给定图像执行对象检测。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.object_detection("people.jpg")
[ObjectDetectionOutputElement(score=0.9486683011054993, label='person', box=ObjectDetectionBoundingBox(xmin=59, ymin=39, xmax=420, ymax=510)), ...]
```

#### Question_answering[[huggingface_hub.AsyncInferenceClient.question_answering]]

```python
question_answering(question: str, context: str, model: str | None = None, align_to_words: bool | None = None, doc_stride: int | None = None, handle_impossible_answer: bool | None = None, max_answer_len: int | None = None, max_question_len: int | None = None, max_seq_len: int | None = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1568)

**参数：**

问题（`str`）：待回答的问题。

context (`str`) ：问题的上下文。

model (`str`) ：用于问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。

align_to_words (`bool`, *可选*) ：尝试将答案与真实单词对齐。提高空间分隔语言的质量。可能会对非空格分隔的语言（例如日语或中文）造成伤害

doc_stride (`int`, *可选*) ：如果上下文太长而无法满足模型的问题，它将被分成几个有重叠的块。该参数控制重叠的大小。handle_impossible_answer (`bool`, *可选*) ：是否接受不可能作为答案。

max_answer_len (`int`, *可选*) ：预测答案的最大长度（例如，仅考虑长度较短的答案）。

max_question_len (`int`, *可选*) ：标记化后问题的最大长度。如果需要，它将被截断。

max_seq_len (`int`, *可选*) ：传递给模型的每个块的标记中总句子（上下文+问题）的最大长度。如果需要，上下文将被分割成几个块（使用 docStride 作为重叠）。

top_k (`int`, *可选*) ：要返回的答案数量（将按可能性顺序选择）。请注意，如果上下文中没有足够的选项，我们将返回少于 topk 的答案。

**返回：**联盟[⟦T738⟧, list[QuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.QuestionAnsweringOutputElement)]

当top_k为1或未提供时，它返回单个`QuestionAnsweringOutputElement`。
当top_k大于1时，返回`QuestionAnsweringOutputElement`的列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

从给定文本中检索问题的答案。示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.question_answering(question="What's my name?", context="My name is Clara and I live in Berkeley.")
QuestionAnsweringOutputElement(answer='Clara', end=16, score=0.9326565265655518, start=11)
```

####句子相似度[[huggingface_hub.AsyncInferenceClient.sentence_similarity]]

```python
sentence_similarity(sentence: str, other_sentences: list, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1653)

**参数：**

句子 (`str`) ：与其他句子进行比较的主要句子。

other_sentences (`list[str]`) ：要比较的句子列表。

model (`str`, *可选*) ：用于句子相似度任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果不提供，将使用默认推荐的句子相似度模型。默认为无。

**返回：** `list[float]`

表示输入文本的嵌入。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

通过比较句子和其他句子列表的嵌入来计算它们之间的语义相似度。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.sentence_similarity(
...     "Machine learning is so easy.",
...     other_sentences=[
...         "Deep learning is so straightforward.",
...         "This is so difficult, like rocket science.",
...         "I can't believe how much I struggled with this.",
...     ],
... )
[0.7785726189613342, 0.45876261591911316, 0.2906220555305481]
```

#### 摘要[[huggingface_hub.AsyncInferenceClient.summarization]]

```python
summarization(text: str, model: str | None = None, clean_up_tokenization_spaces: bool | None = None, generate_parameters: dict[str, typing.Any] | None = None, truncation: typing.Optional[ForwardRef('SummarizationTruncationStrategy')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1707)

**参数：**

text (`str`) ：要摘要的输入文本。model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐模型进行汇总。

clean_up_tokenization_spaces (`bool`, *可选*) ：是否清理文本输出中潜在的额外空格。

generate_parameters（`dict[str, Any]`，*可选*）：文本生成算法的附加参数化。

截断（`"SummarizationTruncationStrategy"`，*可选*）：要使用的截断策略。

**返回：** [SummarizationOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.SummarizationOutput)

生成的摘要文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定模型生成给定文本的摘要。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.summarization("The Eiffel tower...")
SummarizationOutput(generated_text="The Eiffel tower is one of the most famous landmarks in the world....")
```

#### table_question_answering[[huggingface_hub.AsyncInferenceClient.table_question_answering]]

```python
table_question_answering(table: dict, query: str, model: str | None = None, padding: typing.Optional[ForwardRef('Padding')] = None, sequential: bool | None = None, truncation: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1766)

**参数：**

table (`str`) ：表示为列表字典的数据表，其中条目是标题，列表是所有值，所有列表必须具有相同的大小。query (`str`) ：要向表询问的纯文本查询。

model (`str`) ：用于表格问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。

填充（`"Padding"`，*可选*）：激活和控制填充。

顺序（`bool`，*可选*）：是否按顺序或批量进行推理。批处理速度更快，但考虑到序列的对话性质，像 SQA 这样的模型需要按顺序进行推理，以提取序列内的关系。

截断（`bool`，*可选*）：激活并控制截断。

**返回：** [TableQuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TableQuestionAnsweringOutputElement)

包含答案、坐标、单元格和使用的聚合器的表格问答输出。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

从表格中给出的信息中检索问题的答案。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> query = "How many stars does the transformers repository have?"
>>> table = {"Repository": ["Transformers", "Datasets", "Tokenizers"], "Stars": ["36542", "4512", "3934"]}
>>> await client.table_question_answering(table, query, model="google/tapas-base-finetuned-wtq")
TableQuestionAnsweringOutputElement(answer='36542', coordinates=[[0, 1]], cells=['36542'], aggregator='AVERAGE')
```

#### tabular_classification[[huggingface_hub.AsyncInferenceClient.tabular_classification]]

```python
tabular_classification(table: dict, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1829)

**参数：**

表 (`dict[str, Any]`) ：要分类的属性集。model（`str`，*可选*）：用于表格分类任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的表格分类模型。默认为无。

**返回：** `List`

标签列表，初始表中每行一个。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据一组属性对目标类别（一组）进行分类。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> table = {
...     "fixed_acidity": ["7.4", "7.8", "10.3"],
...     "volatile_acidity": ["0.7", "0.88", "0.32"],
...     "citric_acid": ["0", "0", "0.45"],
...     "residual_sugar": ["1.9", "2.6", "6.4"],
...     "chlorides": ["0.076", "0.098", "0.073"],
...     "free_sulfur_dioxide": ["11", "25", "5"],
...     "total_sulfur_dioxide": ["34", "67", "13"],
...     "density": ["0.9978", "0.9968", "0.9976"],
...     "pH": ["3.51", "3.2", "3.23"],
...     "sulphates": ["0.56", "0.68", "0.82"],
...     "alcohol": ["9.4", "9.8", "12.6"],
... }
>>> await client.tabular_classification(table=table, model="julien-c/wine-quality")
["5", "5", "5"]
```

#### tabular_regression[[huggingface_hub.AsyncInferenceClient.tabular_regression]]

```python
tabular_regression(table: dict, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1885)

**参数：**

table (`dict[str, Any]`) ：存储在表中的属性集。用于预测目标的属性可以是数值属性和分类属性。

model (`str`, *可选*) ：用于表格回归任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的表格回归模型。默认为无。

**返回：** `List`预测数字目标值的列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

给定表中的一组属性/特征，预测数字目标值。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> table = {
...     "Height": ["11.52", "12.48", "12.3778"],
...     "Length1": ["23.2", "24", "23.9"],
...     "Length2": ["25.4", "26.3", "26.5"],
...     "Length3": ["30", "31.2", "31.1"],
...     "Species": ["Bream", "Bream", "Bream"],
...     "Width": ["4.02", "4.3056", "4.6961"],
... }
>>> await client.tabular_regression(table, model="scikit-learn/Fish-Weight")
[110, 120, 130]
```

#### text_classification[[huggingface_hub.AsyncInferenceClient.text_classification]]

```python
text_classification(text: str, model: str | None = None, top_k: int | None = None, function_to_apply: typing.Optional[ForwardRef('TextClassificationOutputTransform')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L1936)

**参数：**

text (`str`) ：要分类的字符串。

model (`str`，*可选*)：用于文本分类任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本分类模型。默认为无。

top_k (`int`, *可选*) ：指定后，将输出限制为前 K 个最可能的类。

function_to_apply (`"TextClassificationOutputTransform"`, *可选*) ：应用于模型输出以检索分数的函数。

**退货：** `list[TextClassificationOutputElement]`

包含预测标签和相关概率的[TextClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对给定文本执行文本分类（例如情感分析）。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.text_classification("I like you")
[
    TextClassificationOutputElement(label='POSITIVE', score=0.9998695850372314),
    TextClassificationOutputElement(label='NEGATIVE', score=0.0001304351753788069),
]
```

#### text_ Generation[[huggingface_hub.AsyncInferenceClient.text_ Generation]]

```python
text_generation(prompt: str, details: bool | None = None, stream: bool | None = None, model: str | None = None, adapter_id: str | None = None, best_of: int | None = None, decoder_input_details: bool | None = None, do_sample: bool | None = None, frequency_penalty: float | None = None, grammar: huggingface_hub.inference._generated.types.text_generation.TextGenerationInputGrammarType | None = None, max_new_tokens: int | None = None, repetition_penalty: float | None = None, return_full_text: bool | None = None, seed: int | None = None, stop: list[str] | None = None, stop_sequences: list[str] | None = None, temperature: float | None = None, top_k: int | None = None, top_n_tokens: int | None = None, top_p: float | None = None, truncate: int | None = None, typical_p: float | None = None, watermark: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L2145)

**参数：**

提示符(`str`)：输入文字。

详细信息（`bool`，*可选*）：默认情况下，text_ Generation 返回一个字符串。如果您想要详细的输出（标记、概率、种子、完成原因等），请通过`details=True`。仅适用于运行 `text-generation-inference` 后端的型号。

流（`bool`，*可选*）：默认情况下，text_ Generation 返回完整的生成文本。如果您想要返回令牌流，请传递`stream=True`。仅适用于运行`text-generation-inference`后端的型号。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。默认为无。

adapter_id（`str`，*可选*）：Lora 适配器 ID。best_of (`int`, *可选*) ：生成 best_of 序列并返回最高 token logprobs 的序列。

Decoder_input_details (`bool`, *可选*) ：返回解码器输入令牌 logprobs 和 ids。您还必须设置 `details=True` 才能将其考虑在内。默认为`False`。

do_sample (`bool`, *可选*) : 激活 logits 采样

Frequency_penalty（`float`，*可选*）：-2.0 到 2.0 之间的数字。正值根据迄今为止文本中的现有频率对新标记进行惩罚，从而降低模型逐字重复同一行的可能性。

语法（[TextGenerationInputGrammarType](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextGenerationInputGrammarType)，*可选*）：语法约束。可以是 JSONSchema 或正则表达式。

max_new_tokens (`int`, *可选*) : 生成的令牌的最大数量。默认为 100。

Repeat_penalty (`float`, *可选*) ：重复惩罚的参数。 1.0 表示没有处罚。更多详情请参见[this paper](https://arxiv.org/pdf/1909.05858.pdf)。

return_full_text (`bool`, *可选*) : 是否将提示添加到生成的文本中

种子（`int`，*可选*）：随机采样种子

stop (`list[str]`, *可选*) ：如果生成了`stop`的成员，则停止生成令牌。stop_sequences (`list[str]`, *可选*) ：已弃用的参数。请使用`stop`代替。

温度（`float`，*可选*）：用于对 logits 分布进行建模的值。

top_n_tokens (`int`, *可选*) ：在每个生成步骤返回有关 `top_n_tokens` 最有可能的标记的信息，而不仅仅是采样的标记。

top_k (`int`, *可选`) ：为 top-k 过滤保留的最高概率词汇标记的数量。

top_p（`float`，*可选`) : If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p`或更高版本保留用于生成。

truncate (`int`, *可选`) ：将输入标记截断为给定大小。

典型_p（`float`，*可选`）：典型解码质量请参阅[Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666)了解更多信息

水印（`bool`，*可选*）：使用[A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)添加水印

**退货：** `Union[str, TextGenerationOutput, AsyncIterable[str], AsyncIterable[TextGenerationStreamOutput]]`

从服务器返回的生成文本：
- 如果`stream=False`和`details=False`，生成的文本将作为`str`返回（默认）
- 如果`stream=True`和`details=False`，生成的文本将逐个标记作为`AsyncIterable[str]`返回
- 如果`stream=False`和`details=True`，则返回生成的文本，其中包含更多详细信息作为[TextGenerationOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextGenerationOutput)
- 如果`details=True`和`stream=True`，则生成的文本将作为[TextGenerationStreamOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TextGenerationStreamOutput)的可迭代对象逐个返回

**提高：** ``ValidationError`` or [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) or `HfHubHTTPError`- ``ValidationError`` -- 
  如果输入值无效。不会对服务器进行 HTTP 调用。
- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

根据提示，生成以下文本。

> [!提示]
> 如果您想从聊天消息生成响应，您应该使用 [InferenceClient.chat_completion()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion) 方法。
> 它接受消息列表而不是单个文本提示，并为您处理聊天模板。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

# Case 1: generate text
>>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12)
'100% open source and built to be easy to use.'

# Case 2: iterate over the generated tokens. Useful for large generation.
>>> async for token in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, stream=True):
...     print(token)
100
%
open
source
and
built
to
be
easy
to
use
.

# Case 3: get more details about the generation process.
>>> await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True)
TextGenerationOutput(
    generated_text='100% open source and built to be easy to use.',
    details=TextGenerationDetails(
        finish_reason='length',
        generated_tokens=12,
        seed=None,
        prefill=[
            TextGenerationPrefillOutputToken(id=487, text='The', logprob=None),
            TextGenerationPrefillOutputToken(id=53789, text=' hugging', logprob=-13.171875),
            (...)
            TextGenerationPrefillOutputToken(id=204, text=' ', logprob=-7.0390625)
        ],
        tokens=[
            TokenElement(id=1425, text='100', logprob=-1.0175781, special=False),
            TokenElement(id=16, text='%', logprob=-0.0463562, special=False),
            (...)
            TokenElement(id=25, text='.', logprob=-0.5703125, special=False)
        ],
        best_of_sequences=None
    )
)

# Case 4: iterate over the generated tokens with more details.
# Last object is more complete, containing the full generated text and the finish reason.
>>> async for details in await client.text_generation("The huggingface_hub library is ", max_new_tokens=12, details=True, stream=True):
...     print(details)
...
TextGenerationStreamOutput(token=TokenElement(id=1425, text='100', logprob=-1.0175781, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=16, text='%', logprob=-0.0463562, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=1314, text=' open', logprob=-1.3359375, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=3178, text=' source', logprob=-0.28100586, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=273, text=' and', logprob=-0.5961914, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=3426, text=' built', logprob=-1.9423828, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-1.4121094, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=314, text=' be', logprob=-1.5224609, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=1833, text=' easy', logprob=-2.1132812, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=271, text=' to', logprob=-0.08520508, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(id=745, text=' use', logprob=-0.39453125, special=False), generated_text=None, details=None)
TextGenerationStreamOutput(token=TokenElement(
    id=25,
    text='.',
    logprob=-0.5703125,
    special=False),
    generated_text='100% open source and built to be easy to use.',
    details=TextGenerationStreamOutputStreamDetails(finish_reason='length', generated_tokens=12, seed=None)
)

# Case 5: generate constrained output using grammar
>>> response = await client.text_generation(
...     prompt="I saw a puppy a cat and a raccoon during my bike ride in the park",
...     model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
...     max_new_tokens=100,
...     repetition_penalty=1.3,
...     grammar={
...         "type": "json",
...         "value": {
...             "properties": {
...                 "location": {"type": "string"},
...                 "activity": {"type": "string"},
...                 "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
...                 "animals": {"type": "array", "items": {"type": "string"}},
...             },
...             "required": ["location", "activity", "animals_seen", "animals"],
...         },
...     },
... )
>>> json.loads(response)
{
    "activity": "bike riding",
    "animals": ["puppy", "cat", "raccoon"],
    "animals_seen": 3,
    "location": "park"
}
```

#### 文本到图像[[huggingface_hub.AsyncInferenceClient.文本到图像]]

```python
text_to_image(prompt: str, negative_prompt: str | None = None, height: int | None = None, width: int | None = None, num_inference_steps: int | None = None, guidance_scale: float | None = None, model: str | None = None, scheduler: str | None = None, seed: int | None = None, extra_body: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L2485)

**参数：**

提示（`str`）：生成图像的提示。

negative_prompt (`str`, *可选*) ：一个提示，用于指导图像生成中不包含哪些内容。

height (`int`, *可选*) : 输出图像的高度（以像素为单位）

width (`int`, *可选*) : 输出图像的宽度（以像素为单位）

num_inference_steps (`int`, *可选*) ：去噪步骤的数量。更多的去噪步骤通常会带来更高质量的图像，但代价是推理速度变慢。Guiding_scale (`float`, *可选*) ：较高的指导比例值会鼓励模型生成与文本提示紧密相关的图像，但值太高可能会导致饱和度和其他伪影。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本到图像模型。默认为无。

调度程序（`str`，*可选*）：用兼容的调度程序覆盖调度程序。

种子（`int`，*可选*）：随机数生成器的种子。

extra_body (`dict[str, Any]`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**退货：** `Image`

生成的图像。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

使用指定模型根据给定文本生成图像。

> [!警告]
> 如果您想使用图像 (`pip install Pillow`)，则必须安装`PIL`。> [!提示]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> image = await client.text_to_image("An astronaut riding a horse on the moon.")
>>> image.save("astronaut.png")

>>> image = await client.text_to_image(
...     "An astronaut riding a horse on the moon.",
...     negative_prompt="low resolution, blurry",
...     model="stabilityai/stable-diffusion-2-1",
... )
>>> image.save("better_astronaut.png")
```

直接使用第三方提供商的示例。使用费用将通过您的 fal.ai 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="fal-ai",  # Use fal.ai provider
...     api_key="REDACTED",  # Pass your fal.ai API key
... )
>>> image = client.text_to_image(
...     "A majestic lion in a fantasy forest",
...     model="black-forest-labs/FLUX.1-schnell",
... )
>>> image.save("lion.png")
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Use replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> image = client.text_to_image(
...     "An astronaut riding a horse on the moon.",
...     model="black-forest-labs/FLUX.1-dev",
... )
>>> image.save("astronaut.png")
```

使用具有额外参数的复制提供程序的示例

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Use replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> image = client.text_to_image(
...     "An astronaut riding a horse on the moon.",
...     model="black-forest-labs/FLUX.1-schnell",
...     extra_body={"output_quality": 100},
... )
>>> image.save("astronaut.png")
```

#### text_to_speech[[huggingface_hub.AsyncInferenceClient.text_to_speech]]

```python
text_to_speech(text: str, model: str | None = None, do_sample: bool | None = None, early_stopping: typing.Union[bool, ForwardRef('TextToSpeechEarlyStoppingEnum'), NoneType] = None, epsilon_cutoff: float | None = None, eta_cutoff: float | None = None, max_length: int | None = None, max_new_tokens: int | None = None, min_length: int | None = None, min_new_tokens: int | None = None, num_beam_groups: int | None = None, num_beams: int | None = None, penalty_alpha: float | None = None, temperature: float | None = None, top_k: int | None = None, top_p: float | None = None, typical_p: float | None = None, use_cache: bool | None = None, extra_body: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L2723)

**参数：**

text (`str`) ：要合成的文本。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本转语音模型。默认为无。

do_sample (`bool`, *可选*) ：生成新令牌时是否使用采样而不是贪婪解码。

Early_stopping（`Union[bool, "TextToSpeechEarlyStoppingEnum"]`，*可选*）：控制基于波束的方法的停止条件。epsilon_cutoff (`float`, *可选*) ：如果设置为严格在 0 和 1 之间浮动，则仅对条件概率大于 epsilon_cutoff 的标记进行采样。在论文中，建议值范围从 3e-4 到 9e-4，具体取决于模型的大小。有关更多详细信息，请参阅[Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)。

eta_cutoff (`float`, *可选*) ：eta采样是局部典型采样和epsilon采样的混合。如果设置为严格在 0 和 1 之间浮动，则仅当令牌大于 eta_cutoff 或 sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))) 时才考虑该令牌。后一项直观地是预期的下一个标记概率，按 sqrt(eta_cutoff) 缩放。在论文中，建议值范围从 3e-4 到 2e-3，具体取决于模型的大小。更多详情请参阅[Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191)。

max_length (`int`, *可选*) ：生成文本的最大长度（以标记为单位），包括输入。

max_new_tokens (`int`, *可选*) ：要生成的最大令牌数量。优先于 max_length。

min_length (`int`, *可选*) ：生成文本的最小长度（以标记为单位），包括输入。min_new_tokens (`int`, *可选*) ：要生成的最小令牌数量。优先于 min_length。

num_beam_groups (`int`, *可选*) ：将 num_beams 划分成的组数，以确保不同组波束之间的多样性。更多详情请参见[this paper](https://hf.co/papers/1610.02424)。

num_beams (`int`, *可选*) ：用于波束搜索的波束数量。

惩罚_alpha（`float`，*可选*）：该值平衡对比搜索解码中的模型置信度和退化惩罚。

温度（`float`，*可选*）：用于调节下一个令牌概率的值。

top_k (`int`, *可选*) ：为 top-k 过滤保留的最高概率词汇标记的数量。

top_p (`float`, *可选*) : 如果设置为 float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.

typical_p (⟦T860⟧, *optional*) : Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. See ⟦T1162⟧ for more details.

use_cache (⟦T861⟧, *optional*) : Whether the model should use the past last key/values attentions to speed up decoding

extra_body (⟦T862⟧, *optional*) : Additional provider-specific parameters to pass to the model. Refer to the provider's documentation for supported parameters.

**Returns:** ⟦T863⟧

The generated audio.

**Raises:** ⟦T1163⟧ or ⟦T864⟧

- ⟦T1164⟧ -- 
  If the model is unavailable or the request times out.
- ⟦T865⟧ -- 
  If the request fails with an HTTP error status code other than HTTP 503.

Synthesize an audio of a voice pronouncing a given text.

> [!TIP]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：
```py
# Must be run in an async context
>>> from pathlib import Path
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> audio = await client.text_to_speech("Hello world")
>>> Path("hello_world.flac").write_bytes(audio)
```

直接使用第三方提供商的示例。使用量将在您的复制帐户中计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",
...     api_key="REDACTED",  # Pass your Replicate API key directly
... )
>>> audio = client.text_to_speech(
...     text="Hello world",
...     model="OuteAI/OuteTTS-0.3-500M",
... )
>>> Path("hello_world.flac").write_bytes(audio)
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",
...     api_key="hf_...",  # Pass your HF token
... )
>>> audio =client.text_to_speech(
...     text="Hello world",
...     model="OuteAI/OuteTTS-0.3-500M",
... )
>>> Path("hello_world.flac").write_bytes(audio)
```

使用具有额外参数的复制提供程序的示例

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Use replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> audio = client.text_to_speech(
...     "Hello, my name is Kororo, an awesome text-to-speech model.",
...     model="hexgrad/Kokoro-82M",
...     extra_body={"voice": "af_nicole"},
... )
>>> Path("hello.flac").write_bytes(audio)
```fal.ai 上使用“YuE-s1-7B-anneal-en-cot”的音乐生成示例

```py
>>> from huggingface_hub import InferenceClient
>>> lyrics = '''
... [verse]
... In the town where I was born
... Lived a man who sailed to sea
... And he told us of his life
... In the land of submarines
... So we sailed on to the sun
... 'Til we found a sea of green
... And we lived beneath the waves
... In our yellow submarine

... [chorus]
... We all live in a yellow submarine
... Yellow submarine, yellow submarine
... We all live in a yellow submarine
... Yellow submarine, yellow submarine
... '''
>>> genres = "pavarotti-style tenor voice"
>>> client = InferenceClient(
...     provider="fal-ai",
...     model="m-a-p/YuE-s1-7B-anneal-en-cot",
...     api_key=...,
... )
>>> audio = client.text_to_speech(lyrics, extra_body={"genres": genres})
>>> with open("output.mp3", "wb") as f:
...     f.write(audio)
```

#### 文本到视频[[huggingface_hub.AsyncInferenceClient.文本到视频]]

```python
text_to_video(prompt: str, model: str | None = None, guidance_scale: float | None = None, negative_prompt: list[str] | None = None, num_frames: float | None = None, num_inference_steps: int | None = None, seed: int | None = None, extra_body: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L2626)

**参数：**

提示 (`str`) ：生成视频的提示。

model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的文本到视频模型。默认为无。

Guiding_scale (`float`, *可选*) ：较高的指导比例值会鼓励模型生成与文本提示紧密相关的视频，但值太高可能会导致饱和和其他伪影。

negative_prompt（`list[str]`，*可选*）：一个或多个提示，指导视频生成中不包含哪些内容。

num_frames (`float`, *可选*) ：num_frames 参数决定生成多少视频帧。

num_inference_steps (`int`, *可选*) ：去噪步骤的数量。更多的去噪步骤通常会带来更高质量的视频，但代价是推理速度变慢。

种子（`int`，*可选*）：随机数生成器的种子。extra_body (`dict[str, Any]`, *可选*) ：传递给模型的其他特定于提供者的参数。有关支持的参数，请参阅提供商的文档。

**退货：** `bytes`

生成的视频。

根据给定的文本生成视频。

> [!提示]
> 您可以使用 `extra_body` 参数将特定于提供者的参数传递给模型。

示例：

直接使用第三方提供商的示例。使用费用将通过您的 fal.ai 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="fal-ai",  # Using fal.ai provider
...     api_key="REDACTED",  # Pass your fal.ai API key
... )
>>> video = client.text_to_video(
...     "A majestic lion running in a fantasy forest",
...     model="tencent/HunyuanVideo",
... )
>>> with open("lion.mp4", "wb") as file:
...     file.write(video)
```

通过 Hugging Face Routing 使用第三方提供商的示例。使用量将通过您的 Hugging Face 帐户计费。

```py
>>> from huggingface_hub import InferenceClient
>>> client = InferenceClient(
...     provider="replicate",  # Using replicate provider
...     api_key="hf_...",  # Pass your HF token
... )
>>> video = client.text_to_video(
...     "A cat running in a park",
...     model="genmo/mochi-1-preview",
... )
>>> with open("cat.mp4", "wb") as file:
...     file.write(video)
```

#### token_classification[[huggingface_hub.AsyncInferenceClient.token_classification]]

```python
token_classification(text: str, model: str | None = None, aggregation_strategy: typing.Optional[ForwardRef('TokenClassificationAggregationStrategy')] = None, ignore_labels: list[str] | None = None, stride: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L2932)

**参数：**

text (`str`) ：要分类的字符串。

model (`str`, *可选*) ：用于标记分类任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的令牌分类模型。默认为无。

aggregation_strategy (`"TokenClassificationAggregationStrategy"`, *可选*) ：用于基于模型预测融合令牌的策略

ignore_labels (`list[str`, *可选*) : 要忽略的标签列表stride (`int`, *可选*) ：分割输入文本时块之间重叠标记的数量。

**退货：** `list[TokenClassificationOutputElement]`

[TokenClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TokenClassificationOutputElement) 项目列表，包含实体组、置信度得分、单词、开始和结束索引。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

对给定文本执行标记分类。
通常用于句子解析，无论是语法解析还是命名实体识别 (NER)，以理解文本中包含的关键字。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.token_classification("My name is Sarah Jessica Parker but you can call me Jessica")
[
    TokenClassificationOutputElement(
        entity_group='PER',
        score=0.9971321225166321,
        word='Sarah Jessica Parker',
        start=11,
        end=31,
    ),
    TokenClassificationOutputElement(
        entity_group='PER',
        score=0.9773476123809814,
        word='Jessica',
        start=52,
        end=59,
    )
]
```

#### 翻译[[huggingface_hub.AsyncInferenceClient.translation]]

```python
translation(text: str, model: str | None = None, src_lang: str | None = None, tgt_lang: str | None = None, clean_up_tokenization_spaces: bool | None = None, truncation: typing.Optional[ForwardRef('TranslationTruncationStrategy')] = None, generate_parameters: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L3008)

**参数：**

text (`str`) ：要翻译的字符串。

model (`str`, *可选*) ：用于翻译任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的翻译模型。默认为无。

src_lang (`str`, *可选*) ：文本的源语言。对于可以翻译多种语言的模型是必需的。tgt_lang (`str`, *可选*) ：要翻译的目标语言。对于可以翻译成多种语言的模型是必需的。

clean_up_tokenization_spaces (`bool`, *可选*) ：是否清理文本输出中潜在的额外空格。

截断（`"TranslationTruncationStrategy"`，*可选*）：要使用的截断策略。

generate_parameters（`dict[str, Any]`，*可选*）：文本生成算法的附加参数化。

**返回：** [TranslationOutput](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.TranslationOutput)

生成的翻译文本。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError` 或 ``ValueError``

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。
- ``ValueError`` -- 
  如果仅提供 `src_lang` 和 `tgt_lang` 参数之一。

将文本从一种语言转换为另一种语言。

查看 https://huggingface.co/tasks/translation 了解有关如何选择最佳模型的更多信息
您的具体用例。源语言和目标语言通常取决于模型。
但是，可以为某些模型指定源语言和目标语言。如果您正在使用这些模型之一，
您可以使用 `src_lang` 和 `tgt_lang` 参数来传递相关信息。示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.translation("My name is Wolfgang and I live in Berlin")
'Mein Name ist Wolfgang und ich lebe in Berlin.'
>>> await client.translation("My name is Wolfgang and I live in Berlin", model="Helsinki-NLP/opus-mt-en-fr")
TranslationOutput(translation_text='Je m'appelle Wolfgang et je vis à Berlin.')
```

指定语言：
```py
>>> client.translation("My name is Sarah Jessica Parker but you can call me Jessica", model="facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
"Mon nom est Sarah Jessica Parker mais vous pouvez m'appeler Jessica"
```

#### Visual_question_answering[[huggingface_hub.AsyncInferenceClient.visual_question_answering]]

```python
visual_question_answering(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], question: str, model: str | None = None, top_k: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L3098)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：上下文的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

问题（`str`）：要回答的问题。

model (`str`, *可选*) ：用于视觉问答任务的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。如果未提供，将使用默认推荐的视觉问答模型。默认为无。

top_k (`int`, *可选*) ：要返回的答案数量（将按可能性顺序选择）。请注意，如果上下文中没有足够的选项，我们将返回少于 topk 的答案。

**退货：** `list[VisualQuestionAnsweringOutputElement]`

包含预测标签和相关概率的[VisualQuestionAnsweringOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.VisualQuestionAnsweringOutputElement)项目列表。

**提高：** ``InferenceTimeoutError`` or `HfHubHTTPError`

- ``InferenceTimeoutError`` -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。根据图像回答开放式问题。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.visual_question_answering(
...     image="https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
...     question="What is the animal doing?"
... )
[
    VisualQuestionAnsweringOutputElement(score=0.778609573841095, answer='laying down'),
    VisualQuestionAnsweringOutputElement(score=0.6957435607910156, answer='sitting'),
]
```

#### Zero_shot_classification[[huggingface_hub.AsyncInferenceClient.zero_shot_classification]]

```python
zero_shot_classification(text: str, candidate_labels: list, multi_label: bool | None = False, hypothesis_template: str | None = None, model: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L3158)

**参数：**

text (`str`) ：要分类的输入文本。

候选标签（`list[str]`）：将文本分类到的可能的类标签集。

labels (`list[str]`, *可选*) ：（已弃用）字符串列表。每个字符串都是输入文本的可能标签的语言表达。

multi_label (`bool`, *可选*) ：多个候选标签是否可以为真。如果为假，则对分数进行归一化，以使每个序列的标签似然之和为 1。如果为真，则将标签视为独立，并对每个候选者的概率进行归一化。

假设_模板（`str`，*可选*）：与`candidate_labels`结合使用的句子，通过用候选标签替换占位符来尝试文本分类。model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。如果未提供，将使用默认推荐的零样本分类模型。

**退货：** `list[ZeroShotClassificationOutputElement]`

包含预测标签及其置信度的[ZeroShotClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ZeroShotClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

提供文本和一组候选标签作为输入，以对输入文本进行分类。

`multi_label=False` 的示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> text = (
...     "A new model offers an explanation for how the Galilean satellites formed around the solar system's"
...     "largest world. Konstantin Batygin did not set out to solve one of the solar system's most puzzling"
...     " mysteries when he went for a run up a hill in Nice, France."
... )
>>> labels = ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]
>>> await client.zero_shot_classification(text, labels)
[
    ZeroShotClassificationOutputElement(label='scientific discovery', score=0.7961668968200684),
    ZeroShotClassificationOutputElement(label='space & cosmos', score=0.18570658564567566),
    ZeroShotClassificationOutputElement(label='microbiology', score=0.00730885099619627),
    ZeroShotClassificationOutputElement(label='archeology', score=0.006258360575884581),
    ZeroShotClassificationOutputElement(label='robots', score=0.004559356719255447),
]
>>> await client.zero_shot_classification(text, labels, multi_label=True)
[
    ZeroShotClassificationOutputElement(label='scientific discovery', score=0.9829297661781311),
    ZeroShotClassificationOutputElement(label='space & cosmos', score=0.755190908908844),
    ZeroShotClassificationOutputElement(label='microbiology', score=0.0005462635890580714),
    ZeroShotClassificationOutputElement(label='archeology', score=0.00047131875180639327),
    ZeroShotClassificationOutputElement(label='robots', score=0.00030448526376858354),
]
```

使用 `multi_label=True` 和自定义 `hypothesis_template` 的示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()
>>> await client.zero_shot_classification(
...    text="I really like our dinner and I'm very happy. I don't like the weather though.",
...    labels=["positive", "negative", "pessimistic", "optimistic"],
...    multi_label=True,
...    hypothesis_template="This text is {} towards the weather"
... )
[
    ZeroShotClassificationOutputElement(label='negative', score=0.9231801629066467),
    ZeroShotClassificationOutputElement(label='pessimistic', score=0.8760990500450134),
    ZeroShotClassificationOutputElement(label='optimistic', score=0.0008674879791215062),
    ZeroShotClassificationOutputElement(label='positive', score=0.0005250611575320363)
]
```

#### Zero_shot_image_classification[[huggingface_hub.AsyncInferenceClient.zero_shot_image_classification]]

```python
zero_shot_image_classification(image: typing.Union[bytes, typing.BinaryIO, str, pathlib.Path, ForwardRef('Image'), bytearray, memoryview], candidate_labels: list, model: str | None = None, hypothesis_template: str | None = None, labels: list = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/inference/_generated/_async_client.py#L3263)

**参数：**

image (`Union[str, Path, bytes, BinaryIO, PIL.Image.Image]`) ：标题的输入图像。它可以是原始字节、图像文件、在线图像的 URL 或 PIL 图像。

候选标签（`list[str]`）：该图像的候选标签

labels (`list[str]`, *可选*) ：（已弃用）字符串可能标签的列表。必须至少有 2 个标签。model (`str`, *可选*) ：用于推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，也可以是已部署的推理端点的 URL。此参数覆盖在实例级别定义的模型。如果未提供，将使用默认推荐的零样本图像分类模型。

假设_模板（`str`，*可选*）：与`candidate_labels`结合使用的句子，通过用候选标签替换占位符来尝试图像分类。

**退货：** `list[ZeroShotImageClassificationOutputElement]`

包含预测标签及其置信度的[ZeroShotImageClassificationOutputElement](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_types#huggingface_hub.ZeroShotImageClassificationOutputElement)项目列表。

**加薪：** [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) 或 `HfHubHTTPError`

- [InferenceTimeoutError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceTimeoutError) -- 
  如果模型不可用或请求超时。
- `HfHubHTTPError` -- 
  如果请求失败并显示 HTTP 503 以外的 HTTP 错误状态代码。

提供输入图像和文本标签来预测图像的文本标签。

示例：
```py
# Must be run in an async context
>>> from huggingface_hub import AsyncInferenceClient
>>> client = AsyncInferenceClient()

>>> await client.zero_shot_image_classification(
...     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
...     labels=["dog", "cat", "horse"],
... )
[ZeroShotImageClassificationOutputElement(label='dog', score=0.956),...]
```

## InferenceTimeoutError[[huggingface_hub.InferenceTimeoutError]]

#### Huggingface_hub.InferenceTimeoutError[[huggingface_hub.InferenceTimeoutError]]

```python
huggingface_hub.InferenceTimeoutError(message: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/errors.py#L155)

当模型不可用或请求超时时引发错误。### 缓存系统参考
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/cache.md