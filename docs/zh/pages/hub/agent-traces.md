<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 代理追踪

Hugging Face Hub 原生支持来自 Claude Code、Codex 和 Pi Agent 的代理跟踪。将原始 JSONL 会话上传到 [Dataset](https://huggingface.co/datasets?format=format%3Aagent-traces) 或 [Storage Bucket](./storage-buckets) 以在专用跟踪查看器中打开它们。数据集在Data Studio中显示踪迹；存储桶可让您直接打开单独的 `.jsonl` 文件。

## 寻找你的踪迹

每个支持的代理将 JSONL 会话写入以下目录：

|代理|本地会话目录 |
| ----------- | ----------------------- |
|克劳德·代码 | `~/.claude/projects` |
|法典| `~/.codex/sessions` |
|圆周率 | `~/.pi/agent/sessions` |

这些跟踪文件是开箱即用的，因此您可以上传它们，而无需先修改或转换它们。

您正在构建自己的安全带吗？如果您的代理未在上面列出，您可以通过发出 [Session Traces Format](./session-traces-format) 使其会话在跟踪查看器中呈现。

跟踪文件可以包括提示、工具输入、命令输出、本地路径、屏幕截图、秘密、私有代码和个人数据。在公开发布之前检查并编辑跟踪记录，或者如果您不确定其中的内容，请将数据集或存储桶保留为私有。对于 Pi Agent 会话，[⟦T5⟧](https://github.com/badlogic/pi-share-hf) 可以帮助收集项目会话、编辑已知机密、运行 TruffleHog 和 LLM 审查，以及仅上传通过检查的会话。

最简单的上传方法是询问您的代理本身：将其指向上面的目录并告诉其将 `.jsonl` 文件上传到 Hub 数据集或存储桶。

如果您希望在新会话登陆时保持同步跟踪，则存储桶特别有用。

## 上传你的痕迹

创建数据集或存储桶后，使用 [recommended standalone installer](/docs/huggingface_hub/guides/cli#getting-started) 安装 `hf` CLI 并登录。如果您希望编码代理为您运行 `hf` 命令，请使用 `hf skills add` 安装 Hugging Face CLI 技能。

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
hf skills add

hf upload <username>/<dataset-name> ~/.codex/sessions . --repo-type dataset
hf buckets sync ~/.codex/sessions hf://buckets/<username>/<bucket-name>/codex
```

将 `~/.codex/sessions` 替换为与您的代理匹配的会话目录。对数据集使用 `hf upload`，对要在写入新跟踪时更新的存储桶使用 `hf buckets sync`。较短的 `hf sync` 命令是 `hf buckets sync` 的别名。

## 查看您的痕迹

在数据集中获得跟踪后，打开 Data Studio 并单击一行。对于存储桶中的跟踪，导航到要检查的 `.jsonl` 文件并将其打开。跟踪查看器显示会话时间线、提示、助理消息、工具调用和结果。

对于公共示例，请打开 [⟦T16⟧](https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent)。您还可以浏览更多标记为[⟦T17⟧](https://huggingface.co/datasets?format=format%3Aagent-traces)的数据集。### 恶意软件扫描
https://huggingface.co/docs/hub/security-malware.md