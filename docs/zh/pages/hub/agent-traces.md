<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 代理追踪

Hugging Face Hub 原生支持来自 Claude Code、Codex、Pi、Hermes Agent 和 Factory Droid 的代理跟踪。将原始 JSONL 会话上传到 [Dataset](https://huggingface.co/datasets?format=format%3Aagent-traces) 或 [Storage Bucket](./storage-buckets) 以在专用跟踪查看器中打开它们。数据集在Data Studio中显示踪迹；存储桶可让您直接打开单独的 `.jsonl` 文件。

通过 OpenRouter 路由的请求也可以直接导出到存储桶并在同一查看器中打开，请参阅[OpenRouter Broadcast traces](#openrouter-broadcast-traces)。

## 寻找你的踪迹

大多数受支持的代理将 JSONL 会话写入本地目录：

|代理|本地会话目录 |
| ------------- | ---------------------------- |
|克劳德·代码 | `~/.claude/projects` |
|法典| `~/.codex/sessions` |
|法典档案 | `~/.codex/archived_sessions` |
|工厂/机器人| `~/.factory/sessions` |
|圆周率 | `~/.pi/agent/sessions` |

这些跟踪文件是开箱即用的，因此您可以上传它们，而无需先修改或转换它们。

Hermes Agent 将当前会话存储在其会话数据库中。上传之前将要共享的会话导出为 JSONL：

```bash
hermes sessions export session.jsonl --session-id <session_id>
```

您正在构建自己的安全带吗？如果您的代理未在上面列出，您可以通过发出 [Session Traces Format](./session-traces-format) 使其会话在跟踪查看器中呈现。跟踪文件可以包括提示、工具输入、命令输出、本地路径、屏幕截图、秘密、私有代码和个人数据。在公开发布之前检查并编辑跟踪记录，或者如果您不确定其中的内容，请将数据集或存储桶保留为私有。

对于 Pi Agent 会话，[⟦T10⟧](https://github.com/badlogic/pi-share-hf) 可以帮助收集项目会话、编辑已知机密、运行 TruffleHog 和 LLM 审查，以及仅上传通过检查的会话。

最简单的上传方法是询问您的代理本身：将其指向上面的目录或导出的 JSONL 文件，并告诉其将 `.jsonl` 文件上传到 Hub 数据集或存储桶。

如果您希望在新会话登陆时保持同步跟踪，则存储桶特别有用。

## 上传你的痕迹

创建数据集或存储桶后，使用 [recommended standalone installer](/docs/huggingface_hub/guides/cli#getting-started) 安装 `hf` CLI 并登录。安装程序还会设置 [Hugging Face CLI Skill](./agents-cli)，以便您的编码代理可以为您运行 `hf` 命令。

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login

hf upload <username>/<dataset-name> ~/.codex/sessions . --repo-type dataset
hf buckets sync ~/.codex/sessions hf://buckets/<username>/<bucket-name>/codex
```

将 `~/.codex/sessions` 替换为代理的匹配会话目录，或者包含从 Hermes Agent 导出的 JSONL 文件的目录。对数据集使用`hf upload`，对要在写入新跟踪时更新的存储桶使用`hf buckets sync`。较短的 `hf sync` 命令是 `hf buckets sync` 的别名。

## 查看您的痕迹在数据集中获得跟踪后，打开 Data Studio 并单击一行。对于存储桶中的跟踪，导航到要检查的 `.jsonl` 文件并将其打开。跟踪查看器显示会话时间线、提示、辅助消息、工具调用、shell 输出、推理块和结果。

对于公共示例，请打开 [⟦T20⟧](https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent)。您还可以浏览更多标记为[⟦T21⟧](https://huggingface.co/datasets?format=format%3Aagent-traces)的数据集。

## OpenRouter 广播跟踪

[OpenRouter Broadcast](https://openrouter.ai/docs/guides/features/broadcast/overview) 可以将您通过 OpenRouter 路由的请求的跟踪发送到 [Storage Bucket](./storage-buckets)，并且 Hub 在跟踪查看器中打开这些文件。您的应用程序或代理中没有任何变化，因此它适用于通过 OpenRouter 的任何内容：Claude Code 或 Codex 指向 OpenRouter、OpenCode 或普通的聊天完成应用程序。

要进行设置，请从具有写入权限的令牌创建一个存储桶 [generate S3 credentials](./storage-buckets-s3#generating-s3-credentials)，然后在 OpenRouter 中的 **设置 > 可观察性** 下添加 **Hugging Face Storage Buckets** 作为目标。 OpenRouter 的 [Hugging Face Storage Buckets guide](https://openrouter.ai/docs/guides/features/broadcast/hugging-face) 有完整的步骤。

将存储桶创建为私有：

```bash
hf buckets create <username>/openrouter-traces --private
```这些文件包含完整的提示、系统提示和工具输出（例如文件路径和 shell 输出），以及您的 OpenRouter 帐户和用户 ID 以及 API 密钥的名称。 OpenRouter 的指南不包括存储桶可见性，任何人都可以读取公共存储桶。

每个跟踪都保存为名为 `{traceId}-{timestamp}.json` 的单独 JSON 文件。使用默认路径模板 `openrouter-traces/{date}`，它们每天都会进入一个文件夹：

```text
openrouter-traces/<date>/gen-<unix>-<id>-<timestamp>.json
```

跟踪是一个模型请求，而不是整个会话。单个代理轮次通常会发出多个模型请求，因此三轮次会话可以生成大约十个文件。每个文件都包含该请求之前的对话，这意味着会话的最后一个文件保存整个会话。来自同一会话的文件共享相同的`trace.sessionId`。

打开存储桶中的任何 `.json` 文件，将其视为 **OpenRouter 跟踪**，并在跟踪视图和原始 JSON 之间切换。查看器显示消息、工具调用和结果、模型、令牌计数（包括缓存的令牌）和成本。页脚中的费用是针对该一次请求的费用，而不是针对会话的费用。

阅读这些痕迹时需要了解两件事：- 不存储图像。导出前从消息中打开路由器[strips multimodal content](https://openrouter.ai/docs/guides/features/broadcast/overview)，查看器在其位置显示`[image]`。
- 某些请求可能丢失或没有内容。 OpenRouter 的采样率和 API 密钥过滤器限制发送到目的地的请求，而隐私模式会删除提示和完成内容。

### 使用空格作为组织卡
https://huggingface.co/docs/hub/spaces-organization-cards.md