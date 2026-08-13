<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 代理库

## 小代理

一个轻量级工具包，用于在 Hugging Face Inference 之上运行 MCP 支持的代理。提供[JavaScript](https://huggingface.co/docs/huggingface.js/en/tiny-agents/README) (`@huggingface/tiny-agents`) 和[Python](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mcp) (`huggingface_hub`)。

```bash
# JavaScript
npx @huggingface/tiny-agents run "agent/id"

# Python
pip install "huggingface_hub[mcp]"
tiny-agents run "agent/id"
```

使用 `agent.json` 配置创建您自己的代理：

```json
{
	"model": "Qwen/Qwen2.5-72B-Instruct",
	"provider": "together",
	"servers": [
		{
			"type": "stdio",
			"command": "npx",
			"args": ["@playwright/mcp@latest"]
		}
	]
}
```

对于本地 LLM，请添加指向您的服务器的 `endpointUrl`（例如 `http://localhost:1234/v1`）。在[SDK guide](./agents-sdk)了解更多信息。

## 广播 MCP 服务器

通过单行更改即可将任何 Gradio 应用程序转变为 MCP 服务器：

```python
demo.launch(mcp_server=True)
```

服务器将每个函数公开为一个工具，并从文档字符串自动生成描述。将其连接到任何 MCP 客户端。 [Hub](https://huggingface.co/spaces?filter=mcp-server) 上提供了数千个 MCP 兼容空间。了解更多信息[Gradio MCP guide](https://www.gradio.app/guides/building-mcp-server-with-gradio)。

## 烟熏剂

[smolagents](https://github.com/huggingface/smolagents) 是一个轻量级的 Python 库，用于通过几行代码构建代理。它支持`CodeAgent`（用Python编写动作）和`ToolCallingAgent`（使用JSON工具调用），通过[Inference Providers](../inference-providers/index)与任何模型配合使用，并与MCP服务器集成。

```bash
smolagent "Plan a trip to Tokyo, Kyoto and Osaka between Mar 28 and Apr 7." \
--model-type "InferenceClientModel" \
--model-id "Qwen/Qwen2.5-Coder-32B-Instruct" \
--tools "web_search"
```

代理可以作为空间推送到中心。浏览社区代理[here](https://huggingface.co/spaces?filter=smolagents&sort=likes)。了解更多信息[smolagents documentation](https://huggingface.co/docs/smolagents/tutorials/tools#use-mcp-tools-with-mcpclient-directly)。

### 在拥抱脸部时使用 Stanza
https://huggingface.co/docs/hub/stanza.md