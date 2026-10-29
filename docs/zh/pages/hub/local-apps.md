<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在本地使用 AI 模型

您可以在计算机上本地运行来自 Hub 的 AI 模型。这意味着您可以受益于以下优势：

- **隐私**：您不会将数据发送到远程服务器。
- **速度**：您的硬件是限制因素，而不是服务器或连接速度。
- **控制**：您可以根据自己的喜好配置模型。
- **成本**：您可以在本地运行模型，而无需向 API 提供商付费。

## 如何使用本地应用程序

本地应用程序是可以直接在您的计算机上运行 Hugging Face 模型的应用程序。开始使用：

1. **在您的[Local Apps settings](https://huggingface.co/settings/local-apps)中启用本地应用程序**。

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/settings.png)

2. **通过搜索从 Hub 中选择支持的型号**。您可以在过滤器侧边栏`Other`部分的`Apps`组中按应用程序进行过滤，例如[models compatible with llama.cpp](https://huggingface.co/models?apps=llama.cpp)：

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/search_llamacpp.png)

3. **从模型页面的“使用此模型”下拉列表中选择本地应用程序**。

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/button.png)

4. **复制并运行** 在终端中提供的命令。

![Local Apps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local-apps/command.png)

## 支持的本地应用程序

支持的本地应用程序列在您的[Local Apps settings](https://huggingface.co/settings/local-apps)中。

在模型页面上，“使用此模型”下拉列表仅在模型与其兼容时显示应用程序（例如，llama.cpp 或 Ollama 的 GGUF 模型）。Hub 目前提供以下应用程序：

|应用程序 |网站 |兼容型号 |
| ---| ---| ---|
|  [Llama](#llama) | [llama.app](https://llama.app) | GGUF |
|  [llama.cpp](#llamacpp) | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF |
|  [LM Studio](#lm-studio) | [lmstudio.ai](https://lmstudio.ai) | GGUF、MLX |
|  [Jan](#jan) | [jan.ai](https://jan.ai) | GGUF |
|  [Atomic Chat](#atomic-chat) | [atomic.chat](https://atomic.chat) | GGUF、MLX |
|  [Unsloth Desktop](#unsloth-desktop) | [unsloth.ai/docs](https://unsloth.ai/docs) | GGUF，Unsloth 模型 |
|  [Ollama](#ollama) | [ollama.com](https://ollama.com) | GGUF |
|  [vLLM](#vllm) | [docs.vllm.ai](https://docs.vllm.ai) | Transformers、GGUF、AWQ、GPTQ 文本生成和图像文本到文本模型 |
|  [SGLang](#sglang) | [docs.sglang.io](https://docs.sglang.io) | Transformers、AWQ、GPTQ 文本生成和图像文本到文本模型 |
|  [MLX LM](#mlx-lm) | [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | MLX 文本生成模型 |
|  [Docker Model Runner](#docker-model-runner) | [docs.docker.com/ai/model-runner](https://docs.docker.com/ai/model-runner/) | GGUF、vLLM 兼容型号 |
|  [Lemonade](#lemonade) | [lemonade-server.ai](https://lemonade-server.ai) | GGUF、AMD Ryzen AI 模型 |
|  [Pi](#pi) | [github.com/earendil-works/pi](https://github.com/earendil-works/pi) |具有工具调用功能的 GGUF 和 MLX 聊天模型 |
|  [Hermes Agent](#hermes-agent) | [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com/) |具有工具调用功能的 GGUF 和 MLX 聊天模型 |
|  [OpenClaw](#openclaw) | [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) |具有工具调用功能的 GGUF 和 MLX 聊天模型 |
|  [Draw Things](#draw-things) | [drawthings.ai](https://drawthings.ai) |扩散器文本到图像模型和 LoRA |
|  [DiffusionBee](#diffusionbee) | [diffusionbee.com](https://diffusionbee.com) |扩散器文本到图像模型 |
|  [JoyFusion](#joyfusion) | [joyfusion.app](https://joyfusion.app) |标记为 `joyfusion` 的核心 ML 文本到图像模型 |> [!提示]
> 👨‍💻 要使用这些本地应用程序，请复制上述模型卡中的片段。
>
> 👷 如果您正在构建本地应用程序，您可以在 [this guide](https://huggingface.co/docs/hub/en/models-adding-libraries) 中了解如何与 Hub 集成。

### 骆驼

[Llama](https://llama.app) 是一个本地 macOS 和 Windows 应用程序，用于在本地运行 GGUF 模型。

**优点：**
- 位于菜单栏或任务栏上的轻量级本机应用程序
- 从 Hub 一键安装 GGUF 模型
- 与已安装模型的本地私人聊天
- 获取适合您的计算机的推荐型号
- 轻松与 Pi、Hermes、OpenClaw 等代理集成

要使用 Llama，请导航至 GGUF 模型的模型卡，然后单击“使用此模型”。 Llama 将打开并指导您安装模型。

> [!注意]
> Llama 目前适用于 macOS 和 Windows。

### Llama.cpp[[llamacpp]]

Llama.cpp 是一个高性能 C/C++ 库，用于在本地运行 LLM，并跨许多不同的硬件（包括 CPU、CUDA 和 Metal）进行优化推理。

**优点：**
- 多个 CPU 系列上基于 CPU 的模型具有极快的性能
- 资源使用率低
- 多个界面选项（CLI、带有 Web UI 的 OpenAI 兼容服务器）
- 针对 CPU 和 GPU 进行硬件优化要使用 Llama.cpp，请导航到模型卡并单击“使用此模型”并复制命令。该代码片段涵盖了安装 llama.cpp（在 macOS 和 Linux 上安装脚本，在 Windows 上安装 WinGet，或从源代码构建）并运行模型：

```sh
# Install llama.cpp (macOS, Linux):
curl -LsSf https://llama.app/install.sh | sh

# Start a local OpenAI-compatible server with a web UI:
llama serve -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M

# Or run inference directly in the terminal:
llama cli -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

请阅读我们的专用[llama.cpp + HF doc page](./gguf-llamacpp)。

### LM工作室

> [!注意]
> 阅读我们的专用[LM Studio doc page](./lmstudio)

LM Studio 是一个桌面应用程序，提供了一种下载、运行和试验本地 LLM 的简单方法。

**优点：**
- 直观的图形界面
- 内置模型浏览器
- 开发者工具和API
- 免费供个人和商业用途

导航至 GGUF 或 MLX 模型的模型卡，然后单击“使用此模型”。 LM Studio 将打开，您可以通过界面开始聊天。

### 一月

Jan 是一个开源 ChatGPT 替代方案，完全离线运行，具有用户友好的界面。

**优点：**
- 用户友好的图形用户界面
- 与文档和文件聊天
- OpenAI 兼容的 API 服务器，因此您可以运行模型并从其他应用程序使用它们

要使用 Jan，请导航至 GGUF 模型的模型卡，然后单击“使用此模型”。 Jan 将打开，您可以通过该界面开始聊天。

### 原子聊天[Atomic Chat](https://atomic.chat)是一款用于运行本地模型的桌面聊天应用程序。

**优点：**
- 图形化聊天界面
- 支持GGUF和MLX模型

要使用原子聊天，请导航至 GGUF 或 MLX 模型的模型卡，然后单击“使用此模型”。原子聊天将打开，模型可供下载。

### 卸载桌面

[Unsloth Desktop](https://unsloth.ai/docs) 是 Unsloth 的一款桌面应用程序，用于运行本地模型。

**优点：**
- 图形界面
- 适用于 GGUF 模型和来自 Unsloth 生态系统的模型

要使用 Unsloth Desktop，请导航至 GGUF 或 Unsloth 模型的模型卡，然后单击“使用此模型”。 Unsloth Desktop 将打开模型，包括您选择的 GGUF 文件（如果您从模型文件中选择了一个）。

### 奥拉马

Ollama 是一个应用程序，可让您通过简单的命令行界面在计算机上本地运行大型语言模型。它适用于 GGUF 型号。

**优点：**
- 轻松安装和设置
- 与 Hugging Face Hub 直接集成

要使用 Ollama，请导航至模型卡并单击“使用此模型”并复制命令。

```sh
ollama run hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

### 法学硕士

[vLLM](https://docs.vllm.ai) 是一款适用于法学硕士的高吞吐量推理和服务引擎，具有兼容 OpenAI 的 API 服务器。**优点：**
- 高吞吐量，针对 GPU 进行了优化
- 支持 Transformers、GGUF 和量化（AWQ、GPTQ）模型
- OpenAI兼容的API服务器

vLLM 可用于文本生成和图像文本到文本模型。导航到模型卡，单击“使用此模型”并复制命令：

```sh
# Install vLLM from pip:
pip install vllm

# Start the vLLM server:
vllm serve "openai/gpt-oss-20b"

# Call the server using curl (OpenAI-compatible API):
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "openai/gpt-oss-20b",
		"messages": [{"role": "user", "content": "What is the capital of France?"}]
	}'
```

该代码片段还展示了如何使用 Docker Model Runner 运行模型。

### SGLang

[SGLang](https://docs.sglang.io)是一个针对法学硕士和视觉语言模型的快速服务框架。

**优点：**
- GPU 上的高性能服务
- 支持 Transformers 和量化（AWQ、GPTQ）模型
- OpenAI兼容的API服务器

SGLang 可用于文本生成和图像文本到文本模型。导航到模型卡，单击“使用此模型”并复制命令：

```sh
# Install SGLang from pip:
pip install sglang

# Start the SGLang server:
python3 -m sglang.launch_server \
    --model-path "openai/gpt-oss-20b" \
    --host 0.0.0.0 \
    --port 30000
```

服务器在端口 30000 上公开 OpenAI 兼容的 API。该代码片段还包括使用 `lmsysorg/sglang` 映像的 Docker 命令。

### MLX LM

[MLX LM](https://github.com/ml-explore/mlx-lm) 使用 Apple 的 MLX 框架在 Apple 芯片上运行 LLM。

**优点：**
- 针对 Apple 芯片进行了优化
- 从命令行生成交互式聊天和文本
- OpenAI兼容服务器MLX LM 可用于 MLX 文本生成模型。导航到模型卡，单击“使用此模型”并复制命令：

```sh
# Install MLX LM
uv tool install mlx-lm

# Interactive chat REPL
mlx_lm.chat --model "mlx-community/Qwen3-4B-4bit"

# Or start an OpenAI-compatible server
mlx_lm.server --model "mlx-community/Qwen3-4B-4bit"
```

### Docker Model Runner

[Docker Model Runner](https://docs.docker.com/ai/model-runner/) 允许您使用 Docker CLI 拉取和运行模型。

**优点：**
- 使用熟悉的 Docker 工具
- 支持GGUF模型和与vLLM兼容的模型

导航到模型卡，单击“使用此模型”并复制命令：

```sh
docker model run hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

### Lemonade

[Lemonade](https://lemonade-server.ai) 是本地 LLM 服务器，针对 AMD 硬件进行了优化。

**优点：**
- Supports GGUF models
- 运行为 AMD Ryzen AI NPU 构建的模型（标记为 `ryzenai-npu` 或 `ryzenai-hybrid` 的模型）

在[downloading Lemonade](https://lemonade-server.ai)之后，导航到模型卡，单击“使用此模型”并复制命令：

```sh
# Pull the model
lemonade pull unsloth/gpt-oss-20b-GGUF:Q4_K_M

# Run and chat with the model
lemonade run user.gpt-oss-20b-GGUF-Q4_K_M

# List all available models
lemonade list
```

### Pi

[Pi](https://github.com/earendil-works/pi) 是在您的终端中运行的编码代理。

Pi适用于GGUF和MLX聊天模型，其聊天模板支持工具调用。该代码片段首先启动本地 OpenAI 兼容服务器（`llama serve` 对于 GGUF 模型，`mlx_lm.server` 对于 MLX 模型），然后配置 Pi 使用它：

```sh
# Start a local OpenAI-compatible server:
llama serve -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M

# Install Pi:
npm install -g @earendil-works/pi-coding-agent

# start Pi in your project directory:
pi

# configure Pi to use your llama server:
/login llama.cpp
```

### Hermes Agent

[Hermes Agent](https://hermes-agent.nousresearch.com/)是Nous Research的AI代理，可以与本地模型一起运行。Hermes Agent 适用于 GGUF 和 MLX 聊天模型，其聊天模板支持工具调用。该代码片段首先启动本地 OpenAI 兼容服务器（对于 GGUF 模型为`llama serve`，对于 MLX 模型为`mlx_lm.server`），然后配置 Hermes Agent 以使用它：

```sh
# Start a local OpenAI-compatible server:
llama serve -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M

# Install Hermes:
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes setup

# Point Hermes at the local server:
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8080/v1
hermes config set model.default unsloth/gpt-oss-20b-GGUF:Q4_K_M

# Run Hermes
hermes
```

### 开爪

[OpenClaw](https://github.com/openclaw/openclaw)是一个开源的个人人工智能代理，可以与本地模型一起运行。

OpenClaw 适用于 GGUF 和 MLX 聊天模型，其聊天模板支持工具调用。该代码片段首先启动本地 OpenAI 兼容服务器（`llama serve` 对于 GGUF 模型，`mlx_lm.server` 对于 MLX 模型），然后配置 OpenClaw 使用它：

```sh
# Start a local OpenAI-compatible server:
llama serve -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M

# Install OpenClaw:
npm install -g openclaw@latest

# Register the local server and set it as the default model:
openclaw onboard --non-interactive --mode local \
  --auth-choice custom-api-key \
  --custom-base-url http://127.0.0.1:8080/v1 \
  --custom-model-id "unsloth/gpt-oss-20b-GGUF:Q4_K_M" \
  --custom-provider-id llama-cpp \
  --custom-compatibility openai \
  --custom-text-input \
  --accept-risk \
  --skip-health

# Run OpenClaw
openclaw agent --local --agent main --message "Hello from Hugging Face"
```

### 画东西

[Draw Things](https://drawthings.ai)是一款用于在Apple设备上本地生成图像的应用程序。

**优点：**
- 完全在设备上运行文本到图像模型
- 可以导入基础模型和 LoRA

要使用 Draw Things，请导航至 Diffusers 文本到图像模型或 LoRA 的模型卡，然后单击“使用此模型”。 Draw Things 将打开并导入模型。

> [!注意]
> Draw Things 仅适用于 macOS。

### 扩散蜜蜂

[DiffusionBee](https://diffusionbee.com)是一款桌面应用程序，用于使用稳定扩散模型在本地生成图像。

**优点：**
- 简单的图形界面
- 无需设置或命令行要使用 DiffusionBee，请导航至 Diffusers 文本到图像模型的模型卡，然后单击“使用此模型”。 DiffusionBee 将打开并导入模型。

> [!注意]
> DiffusionBee 仅适用于 macOS。

### 欢乐融合

[JoyFusion](https://joyfusion.app)是一款在Apple设备上使用Core ML模型本地生成图像的应用程序。

要使用 JoyFusion，请导航到标记为 `joyfusion` 的 Core ML 文本到图像模型的模型卡，然后单击“使用此模型”。 JoyFusion 将打开并导入模型。

> [!注意]
> JoyFusion 仅适用于 macOS。

### 将您的库与 Hub 集成
https://huggingface.co/docs/hub/models-adding-libraries.md