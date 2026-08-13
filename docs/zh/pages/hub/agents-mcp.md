<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 拥抱脸部 MCP 服务器

Hugging Face MCP（模型上下文协议）服务器将兼容 MCP 的 AI 助手（例如 Codex、Cursor、VS Code 扩展、Zed、ChatGPT 或 Claude Desktop）直接连接到 Hugging Face Hub。连接后，您的助手可以在编辑器、聊天或 CLI 中搜索和探索 Hub 资源并使用社区工具。

## 你可以做什么

- 搜索和探索 Hub 资源：模型、数据集、空间、论文等。
- 使用自然语言查询搜索 Hugging Face 文档。
- 安排和运行作业并使用沙箱
- 通过[Spaces](https://hf.co/spaces)上托管的 MCP 兼容 Gradio 应用程序运行社区工具。
- 通过元数据、链接和上下文将结果带回您的助手。

## 开始吧

1. 登录后打开[MCP settings](https://huggingface.co/settings/mcp)。

2. 选择您的客户端：选择与 MCP 兼容的客户端（例如 Cursor、VS Code、Zed、Claude Desktop）。该页面显示客户特定的说明和可供复制的配置片段。

3. 粘贴并重新启动：将代码片段复制到客户端的 MCP 配置中，保存并重新启动/重新加载客户端。您应该会在客户端中看到“Hugging Face”（或类似内容）列为已连接的 MCP 服务器。> [!提示]
> 设置页面生成您的客户期望的确切配置。使用它而不是手动编写配置。

![MCP Settings Example](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf-mcp-settings-new.png)

## 使用服务器

连接后，请您的助手使用拥抱脸部工具。提示示例：

- “搜索 Qwen 3.6 量化的拥抱脸部模型。”
- “找到一个可以转录音频文件的空间。”
- “显示有关天气时间序列的数据集。”
- “创建吉卜力猫风格的 1024 x 1024 图像。”
- “如何将 LoRA 适配器与 PEFT 结合使用？” （使用文档语义搜索）
- “查找有关视觉语言模型的论文。”

您的助手将使用 MCP 服务器公开的工具（包括您选择的空间，如下一节所示）导航 Hub 并返回结果（标题、所有者、下载、链接等）。然后，您可以在 Hub 上打开资源或在同一个聊天中继续迭代。

![HF MCP with Spaces in VS Code](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf-mcp-vscode.png)

## 内置工具

大多数 Hub 任务可以通过内置的 `hf_fs` 工具高效完成。它可以让您的助手高效地导航中心，包括文档和空间的语义搜索。

您可以从 [MCP settings](https://huggingface.co/settings/mcp) 页面配置额外的工具。|工具|描述 |
|------|-------------|
| **贡献存储库** |创建文件并将其写入这些已创建的存储库 |
| **沙箱** |创建和使用沙箱：包括沙箱文件管理 |
| **运行和管理作业** |在 Hugging Face 基础设施上运行、监控和安排作业。 |

> [!提示]
> 不时添加实验工具。尝试并反馈
> 它们是否能改善您的体验 [here](https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/69)

## 添加社区工具（Spaces）

您可以使用社区构建的兼容 MCP 的 Gradio Spaces 来扩展您的设置：

- 通过 MCP 支持[here](https://huggingface.co/spaces?filter=mcp-server) 探索空间。
- 在 Hugging Face [here](https://huggingface.co/settings/mcp) 的 MCP 设置中添加相关空间。

Gradio MCP 应用程序将其功能公开为工具（带有参数和描述），以便您的助手可以直接调用它们。请重新启动或刷新您的客户端，以便它获取您添加的新工具。

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f17f0a0925b9863e28ad517/ex9KRpvamn84ZaOlSp_Bj.png)

查看我们关于作为 MCP 服务器的空间的专用指南 [here](https://huggingface.co/docs/hub/spaces-mcp-servers#add-an-existing-space-to-your-mcp-tools)。

### 空格选项

您的[MCP settings](https://huggingface.co/settings/mcp)提供了多个选项来自定义空间的工作方式：|选项 |描述 |
|--------|-------------|
| **动态空间** |在运行时动态调用 MCP 空间。启用后，您的助手可以即时发现和使用 MCP 兼容的空间，而无需手动添加它们。 |
| **删除嵌入图像** |删除由 Gradio Spaces 生成的嵌入图像。如果您的 MCP 客户端的图像支持有限或您需要纯文本响应，则非常有用。 |

## 了解更多

- 设置和客户端设置：https://huggingface.co/settings/mcp
- 变更日志公告：https://huggingface.co/changelog/hf-mcp-server
- Hugging Face MCP 服务器：https://huggingface.co/mcp
- 使用 Gradio Spaces 构建您自己的 MCP 服务器：https://www.gradio.app/guides/building-mcp-server-with-gradio

### 使用空格作为组织卡
https://huggingface.co/docs/hub/spaces-organization-cards.md