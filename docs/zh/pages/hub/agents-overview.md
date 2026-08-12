<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 代理

Hugging Face 提供了将 AI 代理直接连接到 Hub 的工具和协议。无论您是与 Claude 聊天、使用 Codex 进行构建，还是开发自定义代理，您都可以访问模型、数据集、空间和社区工具。本页介绍如何将 [chat agents](#chat-with-hugging-face) 和 [coding agents](#coding-agents) 连接到集线器。 

|页 |描述 |
| ---- | ----------- |
| [CLI](./agents-cli) |为您的代理提供具有内置技能的 `hf` CLI |
| [MCP Server](./agents-mcp) |将任何 MCP 兼容客户端连接到集线器 |
| [Skills](./agents-skills) | AI/ML 工作流程的特定任务指南 |
| [SDK](./agents-sdk) |使用 Python 或 JavaScript 以编程方式构建代理 |
| [Local Agents](./agents-local) |使用 llama.cpp 和 Pi 运行完全本地代理 |
| [Agent Traces](./agent-traces) |在 Hub 上查看 Claude Code、Codex 和 Pi 会议 |

## 抱脸聊天

使用模型上下文协议 (MCP) 将您的 AI 助手直接连接到 Hugging Face Hub。连接后，您可以搜索模型、探索数据集、生成图像并使用社区工具——所有这些都可以在聊天界面中进行。

### 支持的助手

HF MCP 服务器可与任何 MCP 兼容的客户端配合使用：
- **ChatGPT**（通过插件）
- **克劳德桌面**
- **自定义 MCP 客户端**

### 设置

#### 1. 打开 MCP 设置

![MCP Settings Example](https://huggingface.co/huggingface/documentation-images/resolve/main/agents-docs/mcp-settings.png)登录后访问[huggingface.co/settings/mcp](https://huggingface.co/settings/mcp)。

#### 2. 选择您的客户

从列表中选择与 MCP 兼容的客户端。该页面显示特定于客户端的说明和可供复制的配置片段。

#### 3. 配置并重启

将配置片段复制到客户端的 MCP 设置中，保存并重新启动客户端。

> [!提示]
> 设置页面生成您的客户期望的准确配置。使用它而不是手动编写配置。

### 你能做什么

连接后，请让您的助手使用您在配置中选择的工具中的 Hugging Face 工具：

|任务|示例提示 |
| ---- | -------------- |
|搜索型号 | “在 Hugging Face 上查找 Qwen 3 量化”|
|探索数据集 | “显示有关天气时间序列的数据集” |
|寻找空间 | “找到一个可以转录音频文件的空间”|
|生成图像 | “创建吉卜力风格的 1024x1024 猫图像”|
|搜索论文 | “查找有关视觉语言模型的最新论文”|

您的助手调用 Hugging Face 服务器公开的 MCP 工具，并返回包含元数据、链接和上下文的结果。

### 添加社区工具

使用 MCP 兼容的 Gradio Spaces 扩展您的设置：1.浏览[Spaces with MCP support](https://huggingface.co/spaces?filter=mcp-server)
2. 将它们添加到您的[MCP settings](https://huggingface.co/settings/mcp)
3. 重新启动客户端以获取新工具

Gradio MCP 应用程序将其函数公开为带有参数和描述的工具，以便您的助手可以直接调用它们。

### 了解更多

- [MCP Server Guide](./agents-mcp) - 详细设置和配置
- [HF MCP Settings](https://huggingface.co/settings/mcp) - 配置您的客户端
- [MCP-compatible Spaces](https://huggingface.co/spaces?filter=mcp-server) - 社区工具

## 编码代理

通过 MCP 服务器和技能将 Hugging Face 集成到您的编码工作流程中。直接从 IDE 或编码代理访问模型、数据集和 ML 工具。例如，我们通过 MCP 和/或技能涵盖了这些编码代理及更多内容：

|编码代理|积分方法|
| ------------ | ------------------ |
| [Claude Code](https://code.claude.com/docs) | MCP服务器+技能|
| [OpenAI Codex](https://openai.com/codex/) | MCP服务器+技能|
| [Open Code](https://opencode.ai/) | MCP服务器+技能|
| [Cursor](https://www.cursor.com/) | MCP服务器+技能|
| [VS Code](https://code.visualstudio.com/) | MCP 服务器 |
| [Gemini CLI](https://geminicli.com/) | MCP 服务器 |
| [Zed](https://zed.dev/) | MCP 服务器 |

### 快速设置

#### MCP 服务器

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

#### 技能技能为 AI/ML 工作流程提供特定于任务的指导。它们与 MCP 一起工作或独立工作。

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

有关可用技能和用法，请参阅[Skills Guide](./agents-skills)。

### 你能做什么

配置完成后，您的编码代理可以：

|能力|示例|
| ---------- | -------- |
|搜索中心 | “寻找7B参数下的代码生成模型”|
|生成图像 | “创建变压​​器架构图”|
|探索数据集 | “哪些数据集可用于情绪分析？” |
|运行空间| “使用 Whisper Space 转录此音频文件”|
|获取文档 | “如何使用 Transformer 微调模型？” |

### 环境配置

#### 身份验证

将您的 Hugging Face 令牌设置为环境变量：

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

## 注册您的代理线束Hugging Face 维护着代理工具、与 Hub 交互的编码代理和工具（Claude Code、Codex、Cursor 等）的公共注册表。当 `huggingface_hub` 检测到它正在注册的线束内运行时，它会通过集线器请求的用户代理进行报告。注册您的工具意味着此活动按名称归因于您的工具，并为您的项目提供友好的显示标签、返回文档和存储库的链接以及公共位置[agent usage dataset](https://huggingface.co/datasets/huggingface/agent-usage) — 未注册的工具仅计入其总计 `unknown` 份额。

要注册线束，请打开拉取请求，将条目添加到 `@huggingface/tasks` 包中的 [⟦T9⟧](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/agent-harnesses.ts)：- 线束 ID（输入键）应小写并用连字符分隔（例如：`"claude-code"`）。
- 使用用户友好的外壳设置`prettyLabel`（例如：`Claude Code`）。
- （可选）设置 `repoUrl` 并带有指向线束源代码（通常是 GitHub 存储库）的链接。
- （可选）设置 `docsUrl` 并提供线束文档或网站的链接。
-（可选）设置`description`，其中包含安全带的简短单行说明。
- 定义如何从环境中检测到您的安全带：
  - 如果您的线束设置了标准环境变量之一（`AI_AGENT`或`AGENT`），则其值将直接用作标识符，不需要额外的配置。
  - 否则，设置 `envVars` 将环境变量名称映射到值模式。使用 `"*"` 匹配任何非空值，使用精确字符串进行精确匹配，或使用 `"<prefix>*"` 进行前缀匹配。

就是这样！合并 PR 后，来自代理工具的 `huggingface_hub` 流量（包括 `hf` CLI）将按名称归因于它，并显示在下一次每月更新的 [agent usage dataset](https://huggingface.co/datasets/huggingface/agent-usage) 中。

## 后续步骤- [CLI](./agents-cli) - 用于集线器操作的命令行界面
- [MCP Server](./agents-mcp) - 将任何兼容 MCP 的 AI 助手连接到集线器
- [Skills](./agents-skills) - 编码代理的预构建功能
- [SDK](./agents-sdk) - 用于构建代理的 Python 和 JavaScript 库

### 数字对象标识符 (DOI)
https://huggingface.co/docs/hub/doi.md