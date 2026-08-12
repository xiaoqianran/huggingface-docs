<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空格作为 API 端点

Hugging Face 上的每个 Gradio Space 都会自动用作 API 端点。您可以从 Python、JavaScript 或任何 HTTP 客户端调用它。如果您可以在浏览器中使用 Space，则可以将其作为 API 进行调用。

## 快速开始

安装Python客户端并调用任意公共空间：

```bash
pip install --upgrade gradio_client
```

```python
from gradio_client import Client

client = Client("abidlabs/en2fr", token="hf_...")
result = client.predict("Hello, world!", api_name="/predict")
print(result)  # "Bonjour, le monde!"
```

## 查看可用的API端点

每个 Gradio Space 页脚中都有一个“通过 API 使用”链接。点击查看：

- 所有可用端点及其名称
- 参数类型及说明
- 自动生成 Python 和 JavaScript 代码片段
- 一个 API 记录器，可根据您的 UI 交互生成代码

每个 Space 还公开了 OpenAPI 规范：

```
https://<space-subdomain>.hf.space/gradio_api/openapi.json
```

例如：`https://abidlabs-en2fr.hf.space/gradio_api/openapi.json`

这对于理解完整的 API 架构并将其集成到您自己的应用程序中非常有用。

您还可以通过编程方式检查端点：

```python
from gradio_client import Client

client = Client("abidlabs/whisper", token="hf_...")
client.view_api()  # Prints all endpoints with parameters
```

## Python 客户端

### 安装

```bash
pip install --upgrade gradio_client
```

需要 Python 3.10+。

### 连接到空间

```python
from gradio_client import Client

# Public Space
client = Client("username/space-name")

# Private Space (requires token)
client = Client("username/private-space", token="hf_xxxxx")
```

> [!提示]
> 在 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 获取你的拥抱脸令牌。对于私人空间，您需要具有 **READ** 权限的令牌。

### 做出预测

**同步（阻塞）：**

```python
result = client.predict("Hello", api_name="/predict")
```

**异步（非阻塞）：**

```python
job = client.submit("Hello", api_name="/predict")
# Do other work...
result = job.result()  # Get result when ready
```

### 处理文件对任何文件输入使用 `handle_file()`：

```python
from gradio_client import Client, handle_file

client = Client("abidlabs/whisper", token="hf_...")

# From local file
result = client.predict(audio=handle_file("audio.wav"), api_name="/predict")

# From URL
result = client.predict(audio=handle_file("https://example.com/audio.wav"), api_name="/predict")
```

### 监控作业状态

```python
job = client.submit("Hello", api_name="/predict")

# Check status
status = job.status()
print(f"Queue position: {status.rank}, ETA: {status.eta}")

# Check if complete
if job.done():
    result = job.result()

# Cancel a pending job
job.cancel()
```

### 流/生成器端点

对于产生多个输出的端点：

```python
job = client.submit(prompt="Write a story", api_name="/generate")

# Iterate over streaming outputs
for output in job:
    print(output)
```

## JavaScript 客户端

### 安装

```bash
npm i @gradio/client
```

或者通过 CDN 使用：

```html
<script type="module">
  import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
</script>
```

### 连接和预测

```javascript
import { Client } from "@gradio/client";

const app = await Client.connect("abidlabs/en2fr", { token: "hf_..." });
const result = await app.predict("/predict", ["Hello"]);
console.log(result.data);
```

### 处理文件

```javascript
import { Client, handle_file } from "@gradio/client";

const app = await Client.connect("abidlabs/whisper", { token: "hf_..." });
const result = await app.predict("/predict", [
  handle_file("https://example.com/audio.wav")
]);
```

### 流结果

```javascript
const job = app.submit("/predict", ["Hello"]);

for await (const message of job) {
  if (message.type === "data") {
    console.log("Result:", message.data);
  }
  if (message.type === "status") {
    console.log("Queue position:", message.position);
  }
}
```

## REST API（卷曲）

您还可以直接通过 HTTP 调用 Gradio Spaces，无需任何客户端库。

### 基于队列的 API（推荐）

大多数 Spaces 使用两步过程：

**第 1 步：提交您的请求**

```bash
curl -X POST "https://abidlabs-en2fr.hf.space/gradio_api/call/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"data": ["Hello, world"]}'
```

回应：
```json
{"event_id": "abc123"}
```

**第2步：获取结果**

```bash
curl -N "https://abidlabs-en2fr.hf.space/gradio_api/call/predict/abc123" \
  -H "Authorization: Bearer $HF_TOKEN"
```

响应（服务器发送的事件）：
```
event: complete
data: ["Bonjour, le monde!"]
```

私人空间需要 `Authorization` 标头，并为公共空间提供更好的速率限制。

## ZeroGPU 空间

ZeroGPU 空间根据您的帐户类型具有使用配额：

|账户类型 |包含每日 GPU 配额 |
|----------|------------------------|
|未经验证 | 2 分钟 |
|免费帐户 | 5 分钟 |
|专业帐户 | 40 分钟 |

当您使用令牌进行身份验证时，您帐户的 GPU 配额将被消耗。未经身份验证的请求使用具有更严格限制的共享池。PRO、团队和企业用户可以使用预付费积分，以每 10 分钟 GPU 时间 **1 美元**的价格超出每日配额。

> [!提示]
> 您可以[subscribe to PRO](https://huggingface.co/subscribe/pro) 获得 40 分钟的每日 GPU 配额、更高的队列优先级以及通过积分延长配额的能力。

## 常见模式

### FastAPI 集成

```python
from fastapi import FastAPI
from gradio_client import Client, handle_file

app = FastAPI()
client = Client("abidlabs/whisper", token="hf_...")

@app.post("/transcribe/")
async def transcribe(file_url: str):
    result = client.predict(audio=handle_file(file_url), api_name="/predict")
    return {"transcription": result}
```

### 重试错误处理

```python
import time
from gradio_client import Client

def predict_with_retry(client, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.predict(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

client = Client("username/space", token="hf_...")
result = predict_with_retry(client, "input", api_name="/predict")
```

### 从另一个空间调用空间

从您自己的 Gradio 应用程序调用 ZeroGPU Space 时，转发用户的身份验证：

```python
import gradio as gr
from gradio_client import Client

def process(prompt, request: gr.Request):
    x_ip_token = request.headers.get('x-ip-token', '')
    client = Client("owner/zerogpu-space", headers={"x-ip-token": x_ip_token})
    return client.predict(prompt, api_name="/predict")

demo = gr.Interface(fn=process, inputs="text", outputs="text")
demo.launch()
```

## 通过语义搜索查找空间

由于有数千个可用的渐变空间，您有时希望找到一个来完成特定任务：

```bash
curl -s "https://huggingface.co/api/spaces/semantic-search?q=text+to+speech&sdk=gradio"
```

这将返回按语义相关性排名的空间，其中元数据包括空间 ID、喜欢和简短描述。使用 `sdk=gradio` 参数来过滤公开 API 的空间。

## 了解更多

- [Gradio Python Client Guide](https://www.gradio.app/guides/getting-started-with-the-python-client)
- [Gradio JavaScript Client Guide](https://www.gradio.app/guides/getting-started-with-the-js-client)
- [Querying Gradio Apps with curl](https://www.gradio.app/guides/querying-gradio-apps-with-curl)
- [Spaces ZeroGPU](./spaces-zerogpu)

### 用户配置 (SCIM)
https://huggingface.co/docs/hub/enterprise-scim.md