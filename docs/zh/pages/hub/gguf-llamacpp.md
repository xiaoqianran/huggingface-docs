<!-- huggingface-docs: machine-translated zh-CN from English source -->

# GGUF 与 llama.cpp 的用法

> [!提示]
> 您现在可以在 Hugging Face Endpoints 上部署任何 llama.cpp 兼容的 GGUF，了解更多信息 [here](https://huggingface.co/docs/inference-endpoints/en/others/llamacpp_container)

Llama.cpp 允许您下载并在 GGUF 上运行推理，只需提供 Hugging Face 存储库路径和文件名即可。 llama.cpp 下载模型检查点并自动缓存它。缓存的位置由`LLAMA_CACHE`环境变量定义；了解更多相关信息[here](https://github.com/ggerganov/llama.cpp/pull/7826)。

您可以通过brew（适用于Mac和Linux）安装llama.cpp，也可以从源代码构建它。还有预先构建的二进制文件和 Docker 镜像，您可以[check in the official documentation](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage)。

 ### 选项1：使用brew/winget安装

```bash
brew install llama.cpp
```

或者，在 Windows 上通过 winget

```bash
winget install llama.cpp
```

### 选项 2：从源代码构建

第 1 步：从 GitHub 克隆 llama.cpp。

```
git clone https://github.com/ggerganov/llama.cpp
```

第 2 步：进入 llama.cpp 文件夹并构建它。您还可以添加特定于硬件的标志（例如：Nvidia GPU 的`-DGGML_CUDA=1`）。

```
cd llama.cpp
cmake -B build   # optionally, add -DGGML_CUDA=ON to activate CUDA
cmake --build build --config Release
```

注：其他硬件支持（例如：AMD ROCm、Intel SYCL）请参考[llama.cpp's build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

安装后，您可以按如下方式使用`llama-cli`或`llama-server`：

```bash
llama-cli -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

注意：您可以显式添加 `-no-cnv` 以在原始完成模式（非聊天模式）下运行 CLI。此外，您可以使用 llama.cpp 服务器直接调用 OpenAI 规范聊天完成端点：

```bash
llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

运行服务器后，您可以简单地使用端点，如下所示：

```bash
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"messages": [
    {
        "role": "system",
        "content": "You are an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
    },
    {
        "role": "user",
        "content": "Write a limerick about Python exceptions"
    }
  ]
}'
```

将 `-hf` 替换为任何有效的 Hugging Face 中心存储库名称 - 开始吧！ 🦙

### 托管单点登录
https://huggingface.co/docs/hub/enterprise-advanced-sso.md