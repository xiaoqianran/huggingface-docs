<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在本地使用 AI 模型

您可以在计算机上本地运行 Hub 中的 AI 模型。这意味着您可以受益于以下优势：

- **隐私**：您不会将数据发送到远程服务器。
- **速度**：您的硬件是限制因素，而不是服务器或连接速度。
- **控制**：您可以根据自己的喜好配置模型。
- **成本**：您可以在本地运行模型，而无需向 API 提供商付费。

## 如何使用本地应用程序

本地应用程序是可以直接在您的计算机上运行 Hugging Face 模型的应用程序。开始使用：

1. **在[Local Apps settings](https://huggingface.co/settings/local-apps)中启用本地应用程序**。

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/settings.png)

1. **通过搜索从 Hub 中选择支持的型号**。您可以在导航栏的`Other`部分中按`app`进行过滤： 

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/search_llamacpp.png)

3. **从模型页面的“使用此模型”下拉列表中选择本地应用程序**。

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/button.png)

4. **复制并运行** 在终端中提供的命令。

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/command.png)

## 支持的本地应用程序

检查本地应用程序是否受支持的最佳方法是转到本地应用程序设置并查看该应用程序是否已列出。以下是一些最流行的本地应用程序的快速概述：> [!提示]
> 👨‍💻 要使用这些本地应用程序，请复制上述模型卡中的片段。
>
> 👷 如果您正在构建本地应用程序，您可以在 [this guide](https://huggingface.co/docs/hub/en/models-adding-libraries) 中了解如何与 Hub 集成。

### 骆驼.cpp

Llama.cpp 是一个高性能 C/C++ 库，用于在本地运行 LLM，并跨许多不同的硬件（包括 CPU、CUDA 和 Metal）进行优化推理。

**优点：**
- 多个 CPU 系列上基于 CPU 的模型具有极快的性能
- 资源使用率低
- 多种界面选项（CLI、服务器、Python 库）
- 针对 CPU 和 GPU 进行硬件优化

要使用 Llama.cpp，请导航到模型卡并单击“使用此模型”并复制命令。

```sh
# Load and run the model:
./llama-server -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

请阅读我们的专用[llama.cpp + HF doc page](./gguf-llamacpp)。

### 奥拉马

Ollama 是一个应用程序，可让您通过简单的命令行界面在计算机上本地运行大型语言模型。

**优点：**
- 轻松安装和设置
- 与 Hugging Face Hub 直接集成

要使用 Ollama，请导航至模型卡并单击“使用此模型”并复制命令。

```sh
ollama run hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

### 一月

Jan 是一个开源 ChatGPT 替代方案，完全离线运行，具有用户友好的界面。**优点：**
- 用户友好的图形用户界面
- 与文档和文件聊天
- OpenAI 兼容的 API 服务器，因此您可以运行模型并从其他应用程序使用它们

要使用 Jan，请导航至模型卡并单击“使用此模型”。 Jan 将打开，您可以通过该界面开始聊天。

### LM工作室

> [!注意]
> 阅读我们的专用[LM Studio doc page](./lmstudio)

LM Studio 是一个桌面应用程序，提供了一种下载、运行和试验本地 LLM 的简单方法。

**优点：**
- 直观的图形界面
- 内置模型浏览器
- 开发者工具和API
- 免费供个人和商业用途

导航至模型卡并单击“使用此模型”。 LM Studio 将打开，您可以通过界面开始聊天。

### 单点登录配置指南
https://huggingface.co/docs/hub/security-sso-configuration-guides.md