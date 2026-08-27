<!-- huggingface-docs: machine-translated zh-CN from English source -->

#MCP 客户端

`huggingface_hub` 库现在包含一个 [MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient)，旨在使大型语言模型 (LLM) 能够通过 [Model Context Protocol](https://modelcontextprotocol.io) (MCP) 与外部工具交互。该客户端扩展了[AsyncInferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)以无缝集成工具使用。

[MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient) 连接到公开工具的 MCP 服务器（本地 `stdio` 脚本或远程 `http`/`sse` 服务）。它将这些工具提供给法学硕士（通过[AsyncInferenceClient](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)）。如果 LLM 决定使用工具，[MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient) 管理对 MCP 服务器的执行请求，并将工具的输出转发回 LLM，通常实时传输结果。

我们还提供更高级别的[Agent](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.Agent)课程。这个“小代理”通过管理聊天循环和状态来简化会话代理的创建，充当[MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient)的包装器。

## MCP 客户端[[huggingface_hub.MCPClient]]

#### Huggingface_hub.MCPClient[[huggingface_hub.MCPClient]]

```python
huggingface_hub.MCPClient(model: typing.Optional[str] = None, provider: typing.Union[typing.Literal['baseten', 'cerebras', 'cohere', 'deepinfra', 'fal-ai', 'featherless-ai', 'fireworks-ai', 'groq', 'hf-inference', 'novita', 'nscale', 'openai', 'ovhcloud', 'publicai', 'replicate', 'scaleway', 'together', 'wavespeed', 'zai-org'], typing.Literal['auto'], NoneType] = None, base_url: typing.Optional[str] = None, api_key: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/inference/_mcp/mcp_client.py#L55)

**参数：**

model (`str`, `optional`) ：运行推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，例如`meta-llama/Meta-Llama-3-8B-Instruct` 或已部署推理端点或其他本地或远程端点的 URL。提供者（`str`，*可选*）：用于推理的提供者的名称。默认为“auto”，即模型可用的第一个提供程序，按 https://hf.co/settings/inference-providers 中用户的顺序排序。如果 model 是 URL 或传递了 `base_url`，则不使用 `provider`。

base_url (`str`, *可选*) ：运行推理的基本 URL。默认为无。

api_key (`str`, `optional`) ：用于身份验证的令牌。如果未提供，将默认为本地 Hugging Face 保存的令牌。您还可以使用自己的提供商 API 密钥直接与提供商的服务进行交互。

用于连接到一台或多台 MCP 服务器并使用工具处理聊天完成的客户端。

> [!警告]
> 该课程是实验性的，将来可能会发生重大变化，恕不另行通知。

#### add_mcp_server[[huggingface_hub.MCPClient.add_mcp_server]]

```python
add_mcp_server(type: typing.Literal['stdio', 'sse', 'http'], **params: typing.Any)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/inference/_mcp/mcp_client.py#L123)

**参数：**

type (`str`) ：要连接的服务器的类型。可以是以下之一： - “stdio”：标准输入/输出服务器（本地） - “sse”：服务器发送事件 (SSE) 服务器 - “http”：StreamableHTTP 服务器- ****params** (`dict[str, Any]`) ：服务器参数可以是： - 对于 stdio 服务器： - command (str)：运行 MCP 服务器的命令 - args (list[str]，可选)：命令的参数 - env (dict[str, str]，可选)：命令的环境变量 - cwd (Union[str, Path, None]，可选)：命令的工作目录 - allowed_tools (list[str], 可选): 允许来自该服务器的工具名称列表 - 对于 SSE 服务器: - url (str): SSE 服务器的 URL - headers (dict[str, Any], 可选): SSE 连接的标头 - timeout (float, 可选): 连接超时 - sse_read_timeout (float, 可选): SSE 读取超时 - allowed_tools (list[str], 可选): 允许来自该服务器的工具名称列表- 对于 StreamableHTTP 服务器： - url (str)：StreamableHTTP 服务器的 URL - headers (dict[str, Any], 可选)：StreamableHTTP 连接的标头 - timeout (timedelta，可选)：连接超时 - sse_read_timeout (timedelta，可选)：SSE 读取超时 - Terminate_on_close (bool，可选)：是否在关闭时终止 - allowed_tools (list[str]，可选)：允许使用的工具名称列表服务器

连接到 MCP 服务器#### 清理[[huggingface_hub.MCPClient.cleanup]]

```python
cleanup()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/inference/_mcp/mcp_client.py#L109)

清理资源

#### process_single_turn_with_tools[[huggingface_hub.MCPClient.process_single_turn_with_tools]]

```python
process_single_turn_with_tools(messages: list, exit_loop_tools: typing.Optional[list[huggingface_hub.inference._generated.types.chat_completion.ChatCompletionInputTool]] = None, exit_if_first_chunk_no_tool: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/inference/_mcp/mcp_client.py#L248)

**参数：**

messages (`list[dict]`) : 代表对话历史的消息对象列表

exit_loop_tools (`list[ChatCompletionInputTool]`, *可选*) ：调用时应退出生成器的工具列表

exit_if_first_chunk_no_tool (`bool`, *可选*) ：如果第一个块中不存在工具则退出。默认为 False。

**产量：**

[ChatCompletionStreamOutput](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionStreamOutput) 块或 [ChatCompletionInputMessage](/docs/huggingface_hub/v1.29.0/en/package_reference/inference_types#huggingface_hub.ChatCompletionInputMessage) 对象

使用 `self.model` 和可用工具处理查询，生成块和工具输出。

## 代理[[huggingface_hub.Agent]]

#### Huggingface_hub.Agent[[huggingface_hub.Agent]]

```python
huggingface_hub.Agent(model: Optional[str] = None, servers: Iterable[ServerConfig], provider: Optional[PROVIDER_OR_POLICY_T] = None, base_url: Optional[str] = None, api_key: Optional[str] = None, prompt: Optional[str] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/inference/_mcp/agent.py#L13)

**参数：**

model (`str`, *可选*) ：运行推理的模型。可以是 Hugging Face Hub 上托管的模型 ID，例如`meta-llama/Meta-Llama-3-8B-Instruct` 或已部署推理端点或其他本地或远程端点的 URL。

服务器 (`Iterable[dict]`) ：要连接的 MCP 服务器。每个服务器都是一个包含 `type` 键和 `config` 键的字典。 `type` 键可以是 `"stdio"` 或 `"sse"`，`config` 键是服务器参数的字典。提供者（`str`，*可选*）：用于推理的提供者的名称。默认为“auto”，即模型可用的第一个提供程序，按 https://hf.co/settings/inference-providers 中用户的顺序排序。如果 model 是 URL 或传递了 `base_url`，则不使用 `provider`。

base_url (`str`, *可选*) ：运行推理的基本 URL。默认为无。

api_key (`str`, *可选*) ：用于身份验证的令牌。如果未提供，将默认为本地 Hugging Face 保存的令牌。您还可以使用自己的提供商 API 密钥直接与提供商的服务进行交互。

提示（`str`，*可选*）：用于代理的系统提示。默认为`constants.py`中的默认系统提示符。

简单代理的实现，这是一个构建在 [MCPClient](/docs/huggingface_hub/v1.29.0/en/package_reference/mcp#huggingface_hub.MCPClient) 之上的简单 while 循环。

> [!警告]
> 该课程是实验性的，将来可能会发生重大变化，恕不另行通知。

#### 运行[[huggingface_hub.Agent.run]]

```python
run(user_input: str, abort_event: Optional[asyncio.Event] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/inference/_mcp/agent.py#L57)

**参数：**

user_input (`str`) ：用于运行代理的用户输入。

abort_event (`asyncio.Event`, *可选*) ：可用于中止代理的事件。如果设置了该事件，代理将停止运行。使用给定的用户输入运行代理。

### 下载文件
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/file_download.md