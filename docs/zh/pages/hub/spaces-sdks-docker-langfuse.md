<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Langfuse 的空间

本指南向您展示如何在 Hugging Face Spaces 上部署 Langfuse 并开始检测您的 LLM 应用程序以实现可观察性。此集成可帮助您在 Hugging Face Hub 上试验 LLM API、在一个位置管理提示并评估模型输出。

## 朗芙斯是什么？

[Langfuse](https://langfuse.com) 是一个开源 LLM 工程平台，可帮助团队协作调试、评估和迭代其 LLM 应用程序。 

Langfuse 的主要功能包括用于捕获应用程序执行流完整上下文的 LLM 跟踪、用于集中和协作提示迭代的提示管理、用于评估输出质量的评估指标、用于测试和基准测试的数据集创建，以及用于试验提示和模型配置的游乐场。

_该视频是 Langfuse 功能的 10 分钟演练：_

## 为什么选择法学硕士可观察性？- 随着语言模型变得越来越流行，理解它们的行为和性能非常重要。
- **LLM 可观察性**涉及通过其输出监控和了解 LLM 申请的内部状态。
- 这对于应对以下挑战至关重要：
  - **复杂的控制流**具有重复或链式调用，使得调试具有挑战性。
  - **非确定性输出**，增加了一致质量评估的复杂性。
  - **不同的用户意图**，需要深入理解才能改善用户体验。
- 构建 LLM 应用程序涉及复杂的工作流程，可观察性有助于管理这些复杂性。

## 步骤 1：在 Spaces 上设置 Langfuse

Langfuse Hugging Face Space 允许您只需点击几下即可启动并运行已部署的 Langfuse 版本。

    

要开始，请单击上面的按钮或按照以下步骤操作：1. 创建一个[**new Hugging Face Space**](https://huggingface.co/new-space)
2. 选择 **Docker** 作为 Space SDK
3. 选择 **Langfuse** 作为空间模板
4. 附加 **[Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)** 以确保您的 Langfuse 数据在重新启动后仍保留
5. 确保空间设置为 **公共** 可见性，以便 Langfuse API/SDK 可以访问该应用程序（有关更多详细信息，请参阅下面的注释）
6. [可选但建议]为了安全部署，请替换**环境变量**的默认值：
   - `NEXTAUTH_SECRET`：用于验证登录会话cookie，使用`openssl rand -base64 32`生成至少具有256熵的秘密。
   - `SALT`：用于加盐哈希API密钥，使用`openssl rand -base64 32`生成至少具有256熵的秘密。
   - `ENCRYPTION_KEY`：用于加密敏感数据。必须是256位，64个十六进制格式的字符串字符，通过：`openssl rand -hex 32`生成。
7. 单击**创建空间**！

![Clone the Langfuse Space](https://langfuse.com/images/cookbook/huggingface/huggingface-space-setup.png)

### 用户访问

您的 Langfuse Space 已预先配置了 Hugging Face OAuth 以进行安全身份验证，因此您需要在首次登录时按照弹出窗口中的说明授权 `read` 访问您的 Hugging Face 帐户。

进入应用程序后，您可以使用[the native Langfuse features](https://langfuse.com/docs/rbac)来管理组织、项目和用户。Langfuse 空间_必须_设置为**公共**可见性，以便 Langfuse API/SDK 可以访问应用程序。这意味着默认情况下，任何登录的 Hugging Face 用户都可以访问 Langfuse 空间。

您可以通过两种不同的方法阻止新用户注册和访问该空间：

#### 1.（推荐）Hugging Face 本机组织级 OAuth 限制

如果您想限制仅对指定组织的成员进行访问，您只需在空间的`README.md`文件中设置`hf_oauth_authorized_org`元数据字段，如[here](https://huggingface.co/docs/hub/spaces-oauth#create-an-oauth-app)所示。

配置后，只有属于指定组织的成员的用户才能访问该空间。

#### 2.手动访问控制

您还可以通过将 `AUTH_DISABLE_SIGNUP` 环境变量设置为 `true` 来限制每个用户的访问。在设置此变量之前，请确保您首先登录并通过了空间的身份验证，否则您自己的用户个人资料将无法进行身份验证。> [!提示]
> **注意：** 如果您已将 `AUTH_DISABLE_SIGNUP` 环境变量设置为 `true` 以限制访问，并且想要授予新用户访问该空间的权限，则需要首先将其设置回 `false`（等待重建完成），添加用户并让他们使用 OAuth 进行身份验证，然后将其设置回 `true`。

## 步骤2：使用Langfuse

现在您已经运行了 Langfuse，您可以开始检测您的 LLM 应用程序以捕获跟踪并管理您的提示。让我们看看如何！

### 监控任何应用程序 

Langfuse 与模型无关，可用于跟踪任何应用程序。请按照 Langfuse 文档中的 [get-started guide](https://langfuse.com/docs) 操作，了解如何检测代码。

Langfuse 与许多流行的 LLM 框架保持原生集成，包括 [Langchain](https://langfuse.com/docs/integrations/langchain/tracing)、[LlamaIndex](https://langfuse.com/docs/integrations/llama-index/get-started) 和 [OpenAI](https://langfuse.com/docs/integrations/openai/python/get-started)，并提供 Python 和 JS/TS SDK 来检测您的代码。 Langfuse 还提供各种 API 端点来提取数据，并已被其他开源项目集成，例如[Langflow](https://langfuse.com/docs/integrations/langflow)、[Dify](https://langfuse.com/docs/integrations/dify) 和 [Haystack](https://langfuse.com/docs/integrations/haystack/get-started)。

### 示例 1：跟踪对推理提供程序的调用

作为一个简单的示例，以下是如何使用 Langfuse Python SDK 跟踪对 [Inference Providers](https://huggingface.co/docs/inference-providers/en/index) 的 LLM 调用。请务必首先配置您的 `LANGFUSE_HOST`、`LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY` 环境变量，并确保您有 [authenticated with your Hugging Face account](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)。

```python
from langfuse.openai import openai
from huggingface_hub import get_token

client = openai.OpenAI(
    base_url="https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1",
    api_key=get_token(),
)

messages = [{"role": "user", "content": "What is observability for LLMs?"}]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=messages,
    max_tokens=100,
)
```

### 示例 2：监控 Gradio 应用程序

我们创建了一个 Gradio 模板空间，展示了如何使用 Hugging Face 模型创建简单的聊天应用程序，并在 Langfuse 中跟踪模型调用和用户反馈 - 无需离开 Hugging Face。

    

首先，[duplicate this Gradio template space](https://huggingface.co/spaces/langfuse/langfuse-gradio-example-template?duplicate=true) 并按照[README](https://huggingface.co/spaces/langfuse/langfuse-gradio-example-template/blob/main/README.md) 中的说明进行操作。

## 步骤3：在Langfuse中查看痕迹

一旦您检测了您的应用程序，并将跟踪或用户反馈引入 Langfuse，您就可以在 Langfuse 中查看您的跟踪。

![Example trace with Gradio](https://langfuse.com/images/cookbook/huggingface/huggingface-gradio-example-trace.png)

_[Example trace in the Langfuse UI](https://langfuse-langfuse-template-space.hf.space/project/cm4r1ajtn000a4co550swodxv/traces/9cdc12fb-71bf-4074-ab0b-0b8d212d839f?timestamp=2024-12-20T12%3A12%3A50.089Z&view=preview)_

## 其他资源和支持

- [Langfuse documentation](https://langfuse.com/docs)
- [Langfuse GitHub repository](https://github.com/langfuse/langfuse)
- [Langfuse Discord](https://langfuse.com/discord)
- [Langfuse template Space](https://huggingface.co/spaces/langfuse/langfuse-template-space)

如需更多帮助，请在 [GitHub discussions](https://langfuse.com/discussions) 或 [open an issue](https://github.com/langfuse/langfuse/issues) 上打开支持线程。

### 如何使用 Microsoft Entra ID (Azure AD) 配置 SCIM
https://huggingface.co/docs/hub/security-sso-entra-id-scim.md