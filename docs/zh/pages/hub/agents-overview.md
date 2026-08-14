<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 代理

Hugging Face 提供了将 AI 代理直接连接到 Hub 的工具和协议。无论您是与 Claude 聊天、使用 Codex 进行构建，还是开发自定义代理，您都可以访问模型、数据集、空间和社区工具。本页介绍如何将 [chat agents](#chat-with-hugging-face) 和 [coding agents](#coding-agents) 连接到集线器。 

|页 |描述 |
| ---- | ----------- |
| [CLI](./agents-cli) |为您的代理提供具有内置技能的 `hf` CLI |
| [MCP Server](./agents-mcp) | Connect any MCP-compatible client to the Hub |
| [Skills](./agents-skills) | Task-specific guidance for AI/ML workflows |
| [SDK](./agents-sdk) |使用 Python 或 JavaScript 以编程方式构建代理 |
| [Local Agents](./agents-local) | Run fully local agents with llama.cpp and Pi |
| [Agent Traces](./agent-traces) |在 Hub 上查看 Claude Code、Codex 和 Pi 会议 |

## Chat with Hugging Face

使用模型上下文协议 (MCP) 将您的 AI 助手直接连接到 Hugging Face Hub。连接后，您可以搜索模型、探索数据集、生成图像并使用社区工具——所有这些都可以在聊天界面中进行。

### 支持的助手

HF MCP 服务器可与任何 MCP 兼容的客户端配合使用：
- **ChatGPT**（通过插件）
- **克劳德桌面**
- **自定义 MCP 客户端**

### 设置

#### 1. Open MCP Settings

![MCP Settings Example](https://huggingface.co/huggingface/documentation-images/resolve/main/agents-docs/mcp-settings.png)Visit [huggingface.co/settings/mcp](https://huggingface.co/settings/mcp) while logged in.

#### 2. Select Your Client

Choose your MCP-compatible client from the list.该页面显示特定于客户端的说明和可供复制的配置片段。

#### 3. Configure and Restart

将配置片段复制到客户端的 MCP 设置中，保存并重新启动客户端。

> [!提示]
> 设置页面生成您的客户期望的确切配置。使用它而不是手动编写配置。

### 你能做什么

Once connected, ask your assistant to use Hugging Face tools among the ones you selected in your configuration:

|任务|示例提示 |
| ---- | -------------- |
|搜索型号 | "Find Qwen 3 quantizations on Hugging Face" |
|探索数据集 | "Show datasets about weather time-series" |
|寻找空间 | "Find a Space that can transcribe audio files" |
|生成图像 | "Create a 1024x1024 image of a cat in Ghibli style" |
|搜索论文 | "Find recent papers on vision-language models" |

您的助手调用 Hugging Face 服务器公开的 MCP 工具，并返回包含元数据、链接和上下文的结果。

### 添加社区工具

使用 MCP 兼容的 Gradio Spaces 扩展您的设置：1. Browse [Spaces with MCP support](https://huggingface.co/spaces?filter=mcp-server)
2. 将它们添加到您的[MCP settings](https://huggingface.co/settings/mcp)
3. 重新启动客户端以获取新工具

Gradio MCP 应用程序将其函数公开为带有参数和描述的工具，以便您的助手可以直接调用它们。

### Learn More

- [MCP Server Guide](./agents-mcp) - 详细设置和配置
- [HF MCP Settings](https://huggingface.co/settings/mcp) - 配置您的客户端
- [MCP-compatible Spaces](https://huggingface.co/spaces?filter=mcp-server) - 社区工具

## 编码代理

通过 MCP 服务器和技能将 Hugging Face 集成到您的编码工作流程中。直接从 IDE 或编码代理访问模型、数据集和 ML 工具。例如，我们通过 MCP 和/或技能涵盖了这些编码代理及更多内容：

| Coding Agent |积分方法|
| ------------ | ------------------ |
| [Claude Code](https://code.claude.com/docs) | MCP服务器+技能|
| [OpenAI Codex](https://openai.com/codex/) | MCP Server + Skills |
| [Open Code](https://opencode.ai/) | MCP Server + Skills |
| [Cursor](https://www.cursor.com/) | MCP Server + Skills |
| [VS Code](https://code.visualstudio.com/) | MCP 服务器 |
| [Gemini CLI](https://geminicli.com/) | MCP 服务器 |
| [Zed](https://zed.dev/) | MCP 服务器 |

### 快速设置

#### MCP Server

MCP 服务器使您的编码代理可以访问 Hub 搜索、空间和社区工具。

**光标/VS代码/Zed：**

1. 参观[huggingface.co/settings/mcp](https://huggingface.co/settings/mcp)
2. 从列表中选择您的 IDE
3.复制配置片段
4. 将其添加到 IDE 的 MCP 设置中
5. 重新启动IDE

**克劳德代码：**

```bash
claude mcp add hf-mcp-server -t http "https://huggingface.co/mcp?login"
```

#### 技能Skills provide task-specific guidance for AI/ML workflows.它们与 MCP 一起工作或独立工作。

```bash
# start claude 
claude

# install the skills marketplace plugin
/plugin marketplace add huggingface/skills
```

然后，安装技能规范：
```bash
/plugin install hf-cli@huggingface/skills
```

See the [Skills Guide](./agents-skills) for available skills and usage.

### 你能做什么

配置完成后，您的编码代理可以：

|能力|示例|
| ---------- | -------- |
|搜索中心 | "Find a code generation model under 7B parameters" |
|生成图像 | "Create a diagram of a transformer architecture" |
|探索数据集 | "What datasets are available for sentiment analysis?" |
|运行空间| "Use the Whisper Space to transcribe this audio file" |
|获取文档 | “如何使用 Transformer 微调模型？” |

### 环境配置

#### 身份验证

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="hf_..."
```

或者通过[CLI](./agents-cli)进行身份验证：

```bash
hf auth login
```

#### 添加社区工具

使用 MCP 兼容的 Gradio Spaces 扩展您的设置：

1.浏览[Spaces with MCP support](https://huggingface.co/spaces?filter=mcp-server)
2. 将它们添加到您的[MCP settings](https://huggingface.co/settings/mcp)
3. 重新启动您的IDE

### 示例工作流程

```text
You: Find a text classification model that works well on short texts

Agent: [Searches Hugging Face Hub]
       Found several options:
       - distilbert-base-uncased-finetuned-sst-2-english (sentiment)
       - facebook/bart-large-mnli (zero-shot)
       ...

You: Show me how to use the first one

Agent: [Fetches documentation]
       Here's how to use it with transformers:
       
       from transformers import pipeline
       classifier = pipeline("sentiment-analysis", 
                            model="distilbert-base-uncased-finetuned-sst-2-english")
       result = classifier("I love this product!")
```

## 注册您的代理线束Hugging Face 维护着代理工具、与 Hub 交互的编码代理和工具（Claude Code、Codex、Cursor 等）的公共注册表。 When `huggingface_hub` detects it is running inside a registered harness, it reports it via the user agent on Hub requests.注册您的工具意味着此活动按名称归因于您的工具，并为您的项目提供友好的显示标签、返回文档和存储库的链接以及公共位置[agent usage dataset](https://huggingface.co/datasets/huggingface/agent-usage) — 未注册的工具仅计入其总计 `unknown` 份额。

To register a harness, open a Pull Request adding an entry to [⟦T9⟧](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/agent-harnesses.ts) in the `@huggingface/tasks` package:- 线束 ID（输入键）应小写并用连字符分隔（例如：`"claude-code"`）。
- 使用用户友好的外壳设置`prettyLabel`（例如：`Claude Code`）。
- （可选）设置 `repoUrl` 并带有指向线束源代码（通常是 GitHub 存储库）的链接。
- （可选）设置 `docsUrl` 并提供线束文档或网站的链接。
-（可选）设置`description`，其中包含安全带的简短单行说明。
- 定义如何从环境中检测到您的安全带：
  - 如果您的线束设置了标准环境变量之一（`AI_AGENT`或`AGENT`），则其值将直接用作标识符，不需要额外的配置。
  - 否则，设置 `envVars` 将环境变量名称映射到值模式。使用 `"*"` 匹配任何非空值，使用精确字符串进行精确匹配，或使用 `"<prefix>*"` 进行前缀匹配。

就是这样！合并 PR 后，来自代理工具的 `huggingface_hub` 流量（包括 `hf` CLI）将按名称归因于它，并显示在下一次每月更新的 [agent usage dataset](https://huggingface.co/datasets/huggingface/agent-usage) 中。

## 后续步骤- [CLI](./agents-cli) - 用于集线器操作的命令行界面
- [MCP Server](./agents-mcp) - 将任何兼容 MCP 的 AI 助手连接到 Hub
- [Skills](./agents-skills) - 编码代理的预构建功能
- [SDK](./agents-sdk) - 用于构建代理的 Python 和 JavaScript 库

### 数字对象标识符 (DOI)
https://huggingface.co/docs/hub/doi.md