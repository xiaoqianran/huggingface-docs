<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 SDK 构建

使用 Hugging Face 代理 SDK 构建 MCP 支持的代理。 `huggingface_hub` (Python) 和 `@huggingface/tiny-agents` (JavaScript) 库提供了将 LLM 连接到 MCP 工具所需的一切。

## 安装

```bash
pip install "huggingface_hub[mcp]"
```

```bash
npm install @huggingface/tiny-agents
# or
pnpm add @huggingface/tiny-agents
```

## 快速入门：运行代理

最快的入门方法是使用 `tiny-agents` CLI：

```bash
tiny-agents run julien-c/flux-schnell-generator
```

```bash
npx @huggingface/tiny-agents run "julien-c/flux-schnell-generator"
```

这将从[tiny-agents collection](https://huggingface.co/datasets/tiny-agents/tiny-agents)加载代理，连接到其MCP服务器，并开始交互式聊天。

## 使用代理类

`Agent` 类管理聊天循环和 MCP 工具执行。它使用 [Inference Providers](https://huggingface.co/docs/inference-providers) 来运行 LLM。

```python
from huggingface_hub import Agent
import asyncio

agent = Agent(
    model="Qwen/Qwen2.5-72B-Instruct",
    provider="novita",
    servers=[
        {
            "type": "sse",
            "url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
        }
    ]
)

async def main():
    async for chunk in agent.run("Generate an image of a sunset"):
        if hasattr(chunk, 'choices'):
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="")

asyncio.run(main())
```

有关所有选项，请参阅[Agent reference](https://huggingface.co/docs/huggingface_hub/package_reference/mcp#huggingface_hub.Agent)。

```typescript
import { Agent } from "@huggingface/tiny-agents";

const agent = new Agent({
    model: "Qwen/Qwen2.5-72B-Instruct",
    provider: "novita",
    apiKey: REDACTED,
    servers: [
        {
            type: "sse",
            url: "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
        }
    ]
});

await agent.loadTools();

for await (const chunk of agent.run("Generate an image of a sunset")) {
    if ("choices" in chunk) {
        const delta = chunk.choices[0]?.delta;
        if (delta.content) {
            console.log(delta.content);
        }
    }
}
```

有关所有选项，请参阅[tiny-agents documentation](https://huggingface.co/docs/huggingface.js/tiny-agents/README)。

## 直接使用MCPClient

如需更多控制，请使用`MCPClient`直接管理MCP服务器和工具调用。

```python
import asyncio
from huggingface_hub import MCPClient

async def main():
    async with MCPClient(
        model="Qwen/Qwen2.5-72B-Instruct",
        provider="novita",
    ) as client:
        # Connect to an MCP server
        await client.add_mcp_server(
            type="sse", 
            url="https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
        )
        
        # Process a request with tools
        messages = [{"role": "user", "content": "Generate an image of a sunset"}]
        
        async for chunk in client.process_single_turn_with_tools(messages):
            if hasattr(chunk, 'choices'):
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="")

asyncio.run(main())
```

有关所有选项，请参阅[MCPClient reference](https://huggingface.co/docs/huggingface_hub/package_reference/mcp#huggingface_hub.MCPClient)。

JavaScript SDK 使用 `Agent` 类进行 MCP 交互。对于较低级别的控制，请参阅[@huggingface/mcp-client](https://huggingface.co/docs/huggingface.js/mcp-client/README)包。

## 分享你的经纪人

向 Hub 上的[tiny-agents collection](https://huggingface.co/datasets/tiny-agents/tiny-agents) 贡献代理。包括：

- `agent.json` - 代理配置（必填）
- `PROMPT.md` 或 `AGENTS.md` - 系统提示（可选）
- `EXAMPLES.md` - 示例提示和用例（可选）

## 了解更多- [huggingface_hub MCP Reference](https://huggingface.co/docs/huggingface_hub/package_reference/mcp) - Python API 参考
- [tiny-agents Documentation](https://huggingface.co/docs/huggingface.js/tiny-agents/README) - JavaScript API 参考
- [Inference Providers](https://huggingface.co/docs/inference-providers) - 可用的法学硕士提供商
- [tiny-agents Collection](https://huggingface.co/datasets/tiny-agents/tiny-agents) - 浏览社区代理
- [MCP Server Guide](./agents-mcp) - 连接到 Hugging Face MCP 服务器

### 适度
https://huggingface.co/docs/hub/moderation.md