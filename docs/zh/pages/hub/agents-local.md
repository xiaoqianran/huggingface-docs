<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 带有 llama.cpp 的本地代理

您可以完全在自己的硬件上运行编码代理。多个开源代理可以连接到本地 [llama.cpp](https://github.com/ggerganov/llama.cpp) 服务器，为您提供类似于 Claude Code 或 Codex 的体验 - 但一切都在您的计算机上运行。

## 开始使用

### 1. 设置本地硬件

设置您的本地硬件，以便它可以显示哪些型号与您的设置兼容。

转到 [huggingface.co/settings/hardware](https://huggingface.co/settings/hardware) 并配置您的本地硬件配置文件。然后在 [Local Apps settings](https://huggingface.co/settings/local-apps) 中选择 `llama.cpp`，因为这将是您将使用的引擎。

### 2. 寻找兼容型号

Browse for [Llama.cpp-compatible models](https://huggingface.co/models?apps=llama.cpp&sort=trending).

### 3. Launch the llama.cpp Server

在模型页面上，单击**“使用此模型”**按钮并选择`llama.cpp`。它将向您显示设置的确切命令。第一步是启动 llama.cpp 服务器，例如

```bash
llama-server -hf ggml-org/gemma-4-26b-a4b-it-GGUF:Q4_K_M
```

这将下载模型并在您的计算机上启动 OpenAI 兼容的 API 服务器。有关安装说明，请参阅[llama.cpp guide](./gguf-llamacpp)。

### 4. 连接您的代理

选择下面的代理之一并按照设置说明进行操作。

## Pi

[Pi](https://pi.dev) 是[OpenClaw](https://github.com/openclaw) 背后的代理，现在直接集成到 Hugging Face 中，让您可以访问数千种兼容模型。

Install Pi:

```bash
npm install -g @mariozechner/pi-coding-agent
```然后将本地模型添加到 Pi 的配置文件`~/.pi/agent/models.json`：

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "ggml-org-gemma-4-26b-4b-gguf"
        }
      ]
    }
  }
}
```

Start Pi in your project directory:

```bash
pi
```

Pi 连接到您的本地 llama.cpp 服务器并为您提供交互式代理会话。

![Demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/pi-llama-cpp-demo.gif)

### Enabling vision support

对于支持视觉的模型，将 `"input": ["text", "image"]` 添加到 `~/.pi/agent/models.json` 中的模型条目：

```json
"models": [
  {
    "id": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL",
    "input": ["text", "image"]
  }
]
```

Browse [vision-language models compatible with Pi](https://huggingface.co/models?pipeline_tag=image-text-to-text&apps=pi).

## OpenClaw

[OpenClaw](https://github.com/openclaw) works locally with llama.cpp. You can set your model via the onboard command:

```bash
openclaw onboard --non-interactive \
  --auth-choice custom-api-key \
  --custom-base-url "http://127.0.0.1:8080/v1" \
  --custom-model-id "ggml-org-gemma-4-26b-a4b-gguf" \
  --custom-api-key "llama.cpp" \
  --secret-input-mode plaintext \
  --custom-compatibility openai \
  --accept-risk
```

您还可以交互运行`openclaw onboard`，使用`openai`选择`custom-compatibility`，并传递相同的配置。

### Local Memory Search for OpenClaw

您可以使用 Llama.cpp 运行本地嵌入模型以进行代理的内存搜索。 To do so, make sure to have node-llama-cpp. 

```bash
npm i node-llama-cpp 
```

这是一个在本地运行 [quantized EmbeddingGemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF?show_file_info=embeddinggemma-300M-Q8_0.gguf) 进行内存搜索的示例片段。 OpenClaw 使用以下命令自动下载并提供模型。

```bash
openclaw config set agents.defaults.memorySearch.provider local
openclaw config set agents.defaults.memorySearch.local.modelPath "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf"
```

Restart the gateway and validate.

```bash
openclaw gateway restart
openclaw memory status
# Memory Search (main)
# Provider: local (requested: local)
# Model: hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf
```

## Hermes Agent

[Hermes Agent](https://hermes-agent.nousresearch.com/) works locally with llama.cpp. Define a default config as:

```yaml
model:
  provider: custom
  default: ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M
  base_url: http://127.0.0.1:8080/v1
  api_key: llama.cpp

custom_providers:
  - name: Local (127.0.0.1:8080)
    base_url: http://127.0.0.1:8080/v1
    api_key: llama.cpp
    model: ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M
```

### Local Memory Search for Hermes AgentHermes Agent consumes semantic search models through endpoints.使用 llama.cpp 或您选择的推理引擎在端点 8080 上获得首选嵌入模型后，请将以下内容添加到 `~/.hermes/config.yaml`。

```bash
auxiliary:
  session_search:
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "REDACTED"
    model: "local-llama" # your model alias
    timeout: 90
    max_concurrency: 1
```

Check if this works, `none - built-in only` shows that no other memory plug-ins are used. The output below shows that local serving is active.

```bash
$ hermes memory status
# Memory status
#────────────────────────────────────────
#  Built-in:  always active
#  Provider:  (none — built-in only)
```

## OpenCode

[OpenCode](https://opencode.ai) works locally with llama.cpp. Define a `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "gemma-4-26b-4b-it": {
          "name": "Gemma 4 (local)",
          "limit": {
            "context": 128000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

## 它是如何工作的

The setup has two components running locally:

1. **llama.cpp server** — Serves the model as an OpenAI-compatible API on `localhost`.
2. **Your agent** — The agent process that sends prompts to the local server, reasons about tasks, and executes actions.

```
┌─────────┐     API calls     ┌──────────────────┐
│  Agent  │ ───────────────▶  │  llama.cpp server │
│         │ ◀───────────────  │  (local model)    │
└─────────┘    responses      └──────────────────┘
     │
     ▼
  Your files,
  terminal, etc.
```

## Alternative: llama-agent

[llama-agent](https://github.com/gary149/llama-agent) takes a different approach — it builds the agent loop directly into [llama.cpp](https://github.com/ggerganov/llama.cpp) as a single binary with zero external dependencies. No Node.js, no Python, just compile and run:

```bash
git clone https://github.com/gary149/llama-agent.git
cd llama-agent

# Build
cmake -B build
cmake --build build --target llama-agent

# Run (downloads the model automatically)
./build/bin/llama-agent -hf ggml-org/gemma-4-26b-a4b-it-GGUF:Q4_K_M
```

Because tool calls happen in-process rather than over HTTP, there is no network overhead between the model and the agent. It also supports subagents, MCP servers, and an HTTP API server mode.

## 后续步骤- [Use AI Models Locally](./local-apps) — 了解有关在机器上运行模型的更多信息
- [llama.cpp Guide](./gguf-llamacpp)——llama.cpp安装及使用详解
- [Agents on the Hub](./agents-overview) — 将代理连接到 Hugging Face 生态系统

### 门控组集合
https://huggingface.co/docs/hub/enterprise-gating-group-collections.md