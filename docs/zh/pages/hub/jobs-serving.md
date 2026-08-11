<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 为模特提供工作服务

使用[exposed ports](./jobs-configuration#expose-ports)，作业可以充当临时推理服务器：在 GPU 风格上启动 vLLM，将任何 OpenAI 兼容客户端指向作业的 URL，并在完成后取消作业。您在作业运行时按分钟付费，端点会随作业一起消失。

当端点是一种手段而不是产品时，这是一个很好的选择：评估运行、数据标记会话、针对热门模型的提示迭代，或者只需要运行一个下午的演示。

> [!提示]
> 如果您想要一个不会消失的更永久的端点，您需要 [Inference Endpoints](https://huggingface.co/docs/inference-endpoints)，它提供具有自动扩展、监控和稳定 URL 的托管基础设施。

## 启动 vLLM 服务器

`vllm/vllm-openai` 映像已预安装所有内容。一条命令启动 OpenAI 兼容服务器：

```bash
>>> hf jobs run --detach --expose 8000 --flavor a10g-small -s HF_TOKEN \
...   vllm/vllm-openai \
...   vllm serve LiquidAI/LFM2.5-8B-A1B --max-model-len 8192
✓ Job started
  id: 6a2b137a59bbdade52d4a58c
  url: https://huggingface.co/jobs/davanstrien/6a2b137a59bbdade52d4a58c
Hint: Exposed ports are reachable at (requires an HF token with read access to the job):
  https://6a2b137a59bbdade52d4a58c--8000.hf.jobs
```

`-s HF_TOKEN` 将您的 Hugging Face 令牌作为秘密转发给作业，因此模型下载经过身份验证 - 对于门控或私有模型来说是必需的，并为您提供更高的速率限制和更快的下载速度。

> [!注意]
> Jobs 直接运行您提供的命令 - 它不会像 `docker run` 那样将参数传递给图像的入口点。始终拼写出完整的命令（`vllm serve ...`，而不仅仅是`--model ...`）。服务器需要几分钟时间才能准备就绪（图像拉取、模型下载、模型加载）。通过`hf jobs logs -f <job_id>`跟踪进度；当日志显示 `Application startup complete` 时，服务器已准备就绪。

## 连接客户端

公开端口需要 HF 令牌，该令牌具有对作业命名空间的`read` 访问权限，并作为承载令牌传递。对于 OpenAI 兼容服务器，它直接插入客户端的 API 密钥 — 基本 URL 是公开的端口 URL 加上 `/v1`：

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://6a2b137a59bbdade52d4a58c--8000.hf.jobs/v1",
    api_key=os.environ["HF_TOKEN"],  # any token with read access to the job's namespace
)

response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-8B-A1B",
    messages=[{"role": "user", "content": "Write a haiku about ephemeral compute."}],
)
print(response.choices[0].message.content)
```

LFM2.5 是一个推理模型，因此响应将其推理包含在最终答案之前的 `<think>` 标签内。

或者使用`curl`：

```bash
>>> curl https://6a2b137a59bbdade52d4a58c--8000.hf.jobs/v1/chat/completions \
...   -H "Authorization: Bearer $HF_TOKEN" \
...   -H "Content-Type: application/json" \
...   -d '{"model": "LiquidAI/LFM2.5-8B-A1B", "messages": [{"role": "user", "content": "Hello!"}]}'
```

由于令牌在 `Authorization` 标头中传输，因此这些 URL 可在脚本、笔记本和代理中工作 — 无论您在哪里使用兼容 OpenAI 的 API，都可以使用这些 URL。它们无法直接在浏览器中打开。

## 使用 llama.cpp 提供 GGUF 模型

相同的模式适用于任何使用 HTTP 的服务器。 llama.cpp 的 `llama serve` 使用 `-hf` 标志直接从集线器中提取 GGUF 文件，并提供相同的 OpenAI 兼容 API。例如，要使用 Gemma 推荐的采样设置来服务 [Gemma 4 E4B](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)：

```bash
>>> hf jobs run --detach --expose 8080 --flavor a10g-small -s HF_TOKEN \
...   ghcr.io/ggml-org/llama.cpp:server-cuda -- \
...   /app/llama serve -hf ggml-org/gemma-4-E4B-it-GGUF \
...   --host 0.0.0.0 --port 8080 -ngl 99 \
...   --temp 1.0 --top-p 0.95 --top-k 64
```

`--` 将作业的命令与 `hf jobs run` 自己的选项分开 - 此处需要，因为 `llama serve` 的标志否则将由 CLI 本身解析。> [!提示]
> 您可以通过将模型存储库安装为只读卷并将服务器直接指向该文件来完全跳过模型下载：
>
> ```bash
> >>> hf jobs run --detach --expose 8080 --flavor a10g-small -s HF_TOKEN \
> ...   -v hf://ggml-org/gemma-4-E4B-it-GGUF:/model:ro \
> ...   ghcr.io/ggml-org/llama.cpp:server-cuda -- \
> ...   /app/llama serve --model /model/gemma-4-E4B-it-Q4_K_M.gguf \
> ...   --host 0.0.0.0 --port 8080 -ngl 99
> ```
>
> 服务器启动速度要快得多，因为无需下载任何内容 - 模型在加载时从已安装的存储库中进行流式传输。

> [!警告]
> 您的服务器必须侦听 `0.0.0.0`。默认情况下，`llama serve`绑定到`127.0.0.1`，作业代理无法访问它——显式传递`--host 0.0.0.0`。

这同样适用于任何其他 OpenAI 兼容服务器（SGLang，...）：在公开端口上启动服务器，侦听 `0.0.0.0`，并使用 HF 令牌作为 API 密钥进行连接。

## 停止服务器

当您取消作业或达到超时时（默认为 30 分钟；为较长的会话明确设置 `--timeout`），作业及其计费将停止：

```bash
>>> hf jobs cancel <job_id>
```

> [!注意]
> 暴露端口需要 `huggingface_hub` >= 1.19.0，并且在作业硬件价格的基础上计费 — 请参阅 [Jobs pricing](./jobs-pricing)。

### 带注释的模型卡模板
https://huggingface.co/docs/hub/model-card-annotated.md