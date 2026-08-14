<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 作为 MCP 服务器的空间

您可以**将任何具有可见 `MCP` 徽章的公共空间公开到可调用工具中**，该工具可在任何 MCP 兼容客户端中使用，您可以根据需要添加任意数量的空间，而无需编写任何代码。

## 设置您的 MCP 客户端

从 [Hub MCP settings](https://huggingface.co/settings/mcp) 中，选择 MCP 客户端（VSCode、Cursor、Claude Code 等），然后按照设置说明进行操作。 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f17f0a0925b9863e28ad517/wWm_GeuWF17OrMyJT4tMx.png)

> [!警告]
> 您需要具有读取权限的有效 Hugging Face 令牌才能使用 MCP 工具。如果您没有，请在此处创建一个新的“读取”访问令牌。

## 将现有空间添加到您的 MCP 工具中

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f17f0a0925b9863e28ad517/ex9KRpvamn84ZaOlSp_Bj.png)

1. 浏览兼容的[Spaces](https://huggingface.co/spaces?filter=mcp-server)以查找可通过MCP使用的空间。您还可以在任何 Spaces 卡上查找灰色 **MCP** 徽章。
2. 单击徽章并选择 **添加到 MCP 工具**，然后在询问时确认。
3. 空间应列在“空间工具”部分的 MCP 服务器设置中。

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f17f0a0925b9863e28ad517/uI4PsneUZoWn_TExhNJyt.png)

## 使用 MCP 客户端中的 Spaces如果您的 MCP 客户端配置正确，您添加的空间将立即可用，无需更改任何内容（如果它没有重新启动您的客户端并且应该出现）。大多数 MCP 客户端都会列出当前加载的工具，以便您可以确保空间可用。

> [!提示]
> 对于 ZeroGPU Spaces，您的配额将在调用该工具时使用，如果配额用完，您可以订阅 PRO 以获得 40 分钟的每日配额（比免费用户多 8 倍的配额）。例如，您的 PRO 帐户允许您使用 FLUX.1-schnell 每天生成最多 600 张图像。

## 构建您自己的 MCP 兼容的 Gradio Space

要创建您自己的启用 MCP 的空间，您需要[Create a new Gradio Space](https://huggingface.co/new-space?sdk=gradio)，然后确保在代码中启用 MCP 支持。开始使用 [Gradio Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-gradio) 并确保检查 [detailed MCP guide](https://www.gradio.app/guides/building-mcp-server-with-gradio) 了解更多详细信息。

首先，安装支持 MCP 的 Gradio：
```bash
pip install "gradio[mcp]"
```

然后使用清晰的类型提示和文档字符串创建您的应用程序：

```python
import gradio as gr

def letter_counter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word.
    
    Args:
        word: The word to search in
        letter: The letter to count
        
    Returns:
        Number of times the letter appears in the word
    """
    return word.lower().count(letter.lower())

demo = gr.Interface(fn=letter_counter,
                    inputs=["text", "text"],
                    outputs="number")
demo.launch(mcp_server=True)   # exposes an MCP schema automatically
```

将应用程序推送到 **Gradio Space**，它将自动接收 **MCP** 徽章。然后任何人都可以通过单击将其添加为工具。> [!提示]
> 将现有的 Gradio Space 转换为 MCP 服务器也非常容易。从上下文菜单中复制它，然后只需将 mcp_server=True 参数添加到您的 launch() 方法中，并确保您的函数具有清晰的类型提示和文档字符串 - 您可以使用 AI 工具轻松地自动执行此操作（AI 生成文档字符串的示例）。

## 通过混合空间发挥创意！

由于 Hugging Face Spaces 是最大的 AI 应用程序目录，您可以找到许多可用作 MCP 工具的创意工具。混合和匹配不同的空间可以带来强大且富有创意的工作流程。

  
    
    
  
    该视频演示了使用
    Lightricks/ltx-视频蒸馏和
    类似于 Claude Code 中的 AI/Chatterbox，生成带音频的视频。

### 更多创建空间的方法
https://huggingface.co/docs/hub/spaces-more-ways-to-create.md