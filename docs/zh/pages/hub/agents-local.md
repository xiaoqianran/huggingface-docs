<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 带有 llama.cpp 的本地代理

您可以完全在您自己的硬件上本地运行编码代理。多个开源代理可以连接到本地 [llama.cpp](https://llama.app/) 服务器，为您提供类似于 Claude Code 或 Codex 的体验 - 但一切都在您的计算机上运行。

## 它是如何工作的

该设置有两个本地运行的组件：

1. **llama.cpp 服务器** — 在 `localhost` 上将模型作为 OpenAI 兼容 API 提供服务。
2. **您的代理** — 向本地服务器发送提示、任务原因并执行操作。

```shell
┌─────────┐     API calls     ┌───────────────────┐
│  Agent  │ ───────────────▶  │  llama.cpp server │
│         │ ◀───────────────  │  (local model)    │
└─────────┘    responses      └───────────────────┘
     │
     ▼
  Your files,
  terminal, etc.
```

## 设置 llama.cpp 服务器

### 1. 设置本地硬件

设置您的本地硬件，以便它可以显示哪些型号与您的设置兼容。

转到 [huggingface.co/settings/hardware](https://huggingface.co/settings/hardware) 并配置您的本地硬件配置文件。然后在 [Local Apps settings](https://huggingface.co/settings/local-apps) 中选择 `llama.cpp`，因为这将是您将使用的引擎。

### 2. 寻找兼容型号

浏览[GGUF models compatible with Llama](https://huggingface.co/models?apps=llama-app&sort=trending)。选择后，将模型下载到您的硬件上。

如果您需要灵感，Llama 应用程序会建议一些适合您的硬件设备的模型（请参阅步骤 3）。

### 3.安装Llama应用程序

您可以轻松地在您的计算机上安装 llama.cpp：- **Mac 和 Windows**：安装 [Llama app](https://llama.app)，一个位于系统托盘上的极简桌面应用程序
     安装应用程序后，请按照以下步骤开始使用为您的系统推荐的您最喜欢的模型。
     ![llama app](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/decision/llama-app.png)
- **Linux（或者如果您只需要命令行）**：一行安装命令：
     ```bash
     curl -LsSf https://llama.app/install.sh | sh
     ```

验证安装：

```bash
llama --version
```

### 4.启动路由器

如果您安装了桌面应用程序，`llama serve` 已经在运行。您可以通过浏览[localhost:9931](http://localhost:9931/#/models)上的型号列表来查看。

如果没有，请启动`llama serve`。传递模型会启动单模型模式而不是路由器模式。路由器配置请参见[llama.cpp server guide](https://llama.app/docs/serve#serving-multiple-models-router-mode)。

您有两个选择：
- 自动加载：默认情况下，模型按需加载。
- 手动加载：添加`--no-models-autoload`，然后使用`/llama`加载模型，然后再选择模型。

对于自动加载，只需启动：

```bash
llama serve
```

在大多数情况下，您不需要任何额外的标志，因为默认值应该是合理的。对于高级用法，以下是您可能感兴趣的可能标志的示例：- `-ngl all` 将所有层卸载到 GPU。省略它以使用自动 GPU 卸载。
- `-c 32768` 设置标记中的上下文大小。使用 `-c 0` 作为模型的完整上下文窗口，这可能需要更多的内存。上下文在并行服务器插槽之间共享。
- `--models-dir ~/models` 从自定义本地目录发现本地 GGUF 文件。

虽然上面的命令应该是您所需要的，但这里有一些与 `llama serve` 一起使用的常见标志：

|旗帜|它有什么作用 |
| :--: | :--: |
| `-ngl all` |将所有层卸载到 GPU |
| `-c 32768` |设置标记中的上下文大小 |
| `--models-dir ~/models` |从自定义本地目录发现本地 GGUF 文件 |

### 5. 连接您的代理

选择下面的代理之一并按照设置说明进行操作。

## 圆周率

[Pi](https://pi.dev) 是 [OpenClaw](https://github.com/openclaw) 背后的代理，现在直接集成到 Hugging Face 中，让您可以访问数千种兼容型号。

安装树莓派：

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```

在项目目录中启动 Pi：

```bash
pi
```

配置llama.cpp：

```bash
/login llama.cpp
```

运行`/llama`加载模型，然后运行`/model`为当前会话选择加载的模型。

## 开爪

[OpenClaw](https://github.com/openclaw) 在本地与 llama.cpp 一起工作。您可以通过板载命令设置模型：

```bash
openclaw plugins install @openclaw/llama-cpp-provider
openclaw onboard
```

### OpenClaw 的本地内存搜索您可以使用 Llama.cpp 运行本地嵌入模型以进行代理的内存搜索。为此，请确保拥有 node-llama-cpp。

```bash
npm i node-llama-cpp 
```

这是一个在本地运行 [quantized EmbeddingGemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF?show_file_info=embeddinggemma-300M-Q8_0.gguf) 进行内存搜索的示例片段。 OpenClaw 使用以下命令自动下载并提供模型。

```bash
openclaw config set agents.defaults.memorySearch.provider local
openclaw config set agents.defaults.memorySearch.local.modelPath "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf"
```

重新启动网关并验证。

```bash
openclaw gateway restart
openclaw memory status
# Memory Search (main)
# Provider: local (requested: local)
# Model: hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf
```

## 爱马仕代理

[Hermes Agent](https://hermes-agent.nousresearch.com/) 在本地与 llama.cpp 一起工作。在`~/.hermes/config.yaml`中添加以下内容。

```yaml
custom_providers:
  - name: llama cpp (127.0.0.1:8080)
    base_url: http://127.0.0.1:8080/v1
```

### 本地内存搜索 Hermes 代理

Hermes Agent 通过端点使用语义搜索模型。使用 llama.cpp 或您选择的推理引擎在端点 8080 上获得首选嵌入模型后，请将以下内容添加到 `~/.hermes/config.yaml`。

```bash
auxiliary:
  session_search:
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "REDACTED"
    model: "local-llama" # your model alias
    timeout: 90
    max_concurrency: 1
```

检查一下是否有效，`none - built-in only`显示没有使用其他内存插件。下面的输出显示本地服务处于活动状态。

```bash
$ hermes memory status
# Memory status
#────────────────────────────────────────
#  Built-in:  always active
#  Provider:  (none — built-in only)
```

## 开放代码

[OpenCode](https://opencode.ai) 在本地与 llama.cpp 一起工作。定义`~/.config/opencode/opencode.json`：

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

## 替代方案：llama-agent

[llama-agent](https://github.com/gary149/llama-agent) 采用了不同的方法 - 它将代理循环直接构建到 [llama.cpp](https://github.com/ggerganov/llama.cpp) 作为具有零外部依赖性的单个二进制文件。没有 Node.js，没有 Python，只需编译并运行：

```bash
git clone https://github.com/gary149/llama-agent.git
cd llama-agent

# Build
cmake -B build
cmake --build build --target llama-agent

# Run (downloads the model automatically)
./build/bin/llama-agent -hf ggml-org/gemma-4-26b-a4b-it-GGUF:Q4_K_M
```由于工具调用发生在进程内而不是通过 HTTP，因此模型和代理之间没有网络开销。它还支持子代理、MCP 服务器和 HTTP API 服务器模式。

## 后续步骤

- [Use AI Models Locally](./local-apps) — 了解有关在机器上运行模型的更多信息
- [llama.cpp Guide](./gguf-llamacpp)——llama.cpp安装及使用详解
- [Agents on the Hub](./agents-overview) — 将代理连接到 Hugging Face 生态系统

### 如何使用 Okta 配置 OIDC SSO
https://huggingface.co/docs/hub/security-sso-okta-oidc.md