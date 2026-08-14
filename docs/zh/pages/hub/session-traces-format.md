<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 会话跟踪格式

开箱即用地支持来自 Claude Code、Codex 和 Pi 的代理跟踪（请参阅[Agent Traces](./agent-traces)）。如果您构建自己的工具，则可以通过将其写入下面描述的 **会话跟踪简单格式（STS 格式）** 来使其会话在同一查看器中呈现。

**会话跟踪简单格式（STS 格式）** 是 Hugging Face Hub 检测并在其跟踪查看器中呈现的 [JSONL](https://jsonlines.org) 文件（每行一个 JSON 对象）。它将单个代理/聊天会话捕获为标题行，后跟消息行。

## 1. 会话头（第一行）

```json
{ "type": "session", "harness": "my-agent", "id": "b1a2c3", "name": "Implementing a new API" }
```

|领域 |必填|笔记|
| --------- | -------- | ------------------------------------------------- |
| `type` |是的 |必须是`"session"`|
| `harness` |是的 | **产生跟踪的线束的 ID** |
| `id` |是的 |唯一的会话 ID |
| `name` |没有|人类可读的标题 |
| …… |没有|允许并忽略任何额外的元数据 |`harness` 是关键字段：它是 **harness 的 id**，它告诉 Hub 会话要使用哪个渲染器、图标和标签。目前识别的 id：`llama.app`。在这里添加新的线束标识符非常容易。

> 打造自己的安全带？请联系我们，我们将为它添加一个图标和标签，以便您的会话呈现您的品牌。

## 2. 消息（接下来的每一行）

每一行都是信封中的一条消息：

```json
{ "type": "message", "message": { "role": "assistant", "content": "…" } }
```

`message` 对象：

|领域 |必填|笔记|
| ------------------ | -------- | ---------------------------------------------------------------------------------------------------------- |
| `role` |是的 | `"user"`·`"assistant"`·`"system"`·`"tool"` |
| `content` |是的 |文本（可能为空）|
| `reasoningContent` |没有|模型推理，显示为单独的思维块 |
| `toolCalls` |没有|辅助工具调用：`[{ "id", "function": { "name", "arguments" } }]`（`arguments`为JSON字符串）|| `toolCallId` |没有|在 `role: "tool"` 消息上，将结果链接到它应答的 `toolCalls[].id` |
| `timestamp` |没有|纪元毫秒 |
| `model` |没有|型号名称 |

工具结果是`role: "tool"`携带`toolCallId`的消息；查看器将每个结果拼接到生成该结果的调用旁边。

### 示例

```jsonl
{"type":"session","harness":"my-agent","id":"abc123","name":"what time is it"}
{"type":"message","message":{"role":"user","content":"what time is it?"}}
{"type":"message","message":{"role":"assistant","content":"","toolCalls":[{"id":"t1","function":{"name":"get_time","arguments":"{}"}}]}}
{"type":"message","message":{"role":"tool","toolCallId":"t1","content":"2026-07-01T15:00:00Z"}}
{"type":"message","message":{"role":"assistant","content":"it is 15:00 UTC"}}
```

将生成的 `.jsonl` 上传到 [Dataset](https://huggingface.co/datasets) 或 [Storage Bucket](./storage-buckets) 以在跟踪查看器中打开它。

## 替代方案：Pi 的会话格式

如果您不想采用上面的形状，您可以发出 **Pi 的会话格式** ([session-format.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md))，Hub 已经支持该格式。在这种情况下，还要向 Pi 的会话标头行添加一个 `harness: "..."` 字段，以便 Hub 可以将跟踪归因于您的线束。

### 在拥抱脸部时使用 ML-Agents
https://huggingface.co/docs/hub/ml-agents.md