<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 推理提供者

Hugging Face 的模型页面具有数千个模型的即用即付推理，因此您可以直接在浏览器中尝试所有模型。该服务由推理提供商提供支持，并包含免费套餐。

推理提供商为开发人员提供了对数百种机器学习模型的简化、统一访问，这些模型由最好的无服务器推理合作伙伴提供支持。 👉 **如需完整文档，请访问 [Inference Providers Documentation](https://huggingface.co/docs/inference-providers)**。

## Hub 上的推理提供程序

Inference Providers 与 Hugging Face Hub 深度集成，您可以通过几种不同的方式使用它：

- **交互式小部件** - 使用在幕后使用推理提供程序的交互式小部件直接在模型页面上测试模型。查看 [DeepSeek-R1-0528 model page](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 的示例。
- **推理游乐场** - 轻松测试聊天完成模型并将其与提示进行比较。查看 [Inference Playground](https://huggingface.co/playground) 开始使用。
- **搜索** - 在 [models page](https://huggingface.co/models?inference_provider=all) 上按推理提供程序过滤模型，以查找通过特定提供程序可用的模型。
- **Data Studio** - 使用 AI 探索 Hub 上的数据集。查看您最喜欢的数据集上的 [Data Studio](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/viewer?views%5B%5D=train)。

## 使用推理提供程序进行构建您可以使用我们的 SDK 或 HTTP 客户端将推理提供程序集成到您自己的应用程序中。这是 Python 和 JavaScript 的快速入门，有关更多详细信息，请查看 [Inference Providers Documentation](https://huggingface.co/docs/inference-providers)。

您可以使用我们的 Python SDK 与推理提供程序进行交互。

```python
from huggingface_hub import InferenceClient

import os

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto",   # Automatically selects best provider
)

# Chat completion
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3-0324",
    messages=[{"role": "user", "content": "A story about hiking in the mountains"}]
)

# Image generation
image = client.text_to_image(
    prompt="A serene lake surrounded by mountains at sunset, photorealistic style",
    model="black-forest-labs/FLUX.1-dev"
)

```

或者，您可以只使用 OpenAI API 兼容客户端。

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3-0324",
    messages=[
        {
            "role": "user",
            "content": "A story about hiking in the mountains"
        }
    ],
)
```

> [!警告]
> OpenAI API 兼容客户端不支持图像生成。

您可以使用我们的 JavaScript SDK 与推理提供程序进行交互。

```javascript
import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient(process.env.HF_TOKEN);

const chatCompletion = await client.chatCompletion({
    provider: "auto",  // Automatically selects best provider  
    model: "deepseek-ai/DeepSeek-V3-0324",
    messages: [{ role: "user", content: "Hello!" }]
});

const imageBlob = await client.textToImage({
  model: "black-forest-labs/FLUX.1-dev",
  inputs:
    "A serene lake surrounded by mountains at sunset, photorealistic style",
});
```

或者，您可以只使用 OpenAI API 兼容客户端。

```javascript
import { OpenAI } from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: REDACTED,
});

const completion = await client.chat.completions.create({
  model: "meta-llama/Llama-3.1-8B-Instruct",
  messages: [{ role: "user", content: "A story about hiking in the mountains" }],
});

```

> [!警告]
> OpenAI API 兼容客户端不支持图像生成。

您需要具有推理权限的 Hugging Face 令牌。在 [Settings > Tokens](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained) 创建一个。

### 推理提供者如何工作

要深入了解推理提供程序，请查看 [Inference Providers Documentation](https://huggingface.co/docs/inference-providers)。以下是一些关键资源：

- **[Quick Start](https://huggingface.co/docs/inference-providers)** 
- **[Pricing & Billing Guide](https://huggingface.co/docs/inference-providers/pricing)**
- **[Hub Integration Details](https://huggingface.co/docs/inference-providers/hub-integration)**

### HF-Inference API 是什么？

HF-Inference API 是通过推理提供程序提供的提供程序之一。它以前被称为“推理 API（无服务器）”，并由 [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) 提供支持。

有关 HF-Inference 提供程序的更多详细信息，请查看其 [dedicated page](https://huggingface.co/docs/inference-providers/providers/hf-inference)。### 在拥抱脸部使用 BERTopic
https://huggingface.co/docs/hub/bertopic.md