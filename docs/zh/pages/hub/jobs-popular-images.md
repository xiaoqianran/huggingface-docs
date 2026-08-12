<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 热门图片

以下是来自流行框架的现成 Docker 映像列表，您可以在作业中使用 uv 来使用这些映像。

这些 Docker 镜像已经安装了 uv，但是如果您想对未安装 uv 的镜像使用镜像 + uv，则需要确保首先安装了 uv。这在许多情况下都可以很好地工作，但对于可能有非常具体要求的 LLM 推理库，使用安装了该库的特定映像可能会很有用。

> [!提示]
> 对于 GPU 推理库，传递 `--image` 以便运行获得匹配的 CUDA 系统堆栈
>（工具包、`nvcc`、库）— 请参阅下面的[Using framework images for GPU
> libraries](#using-framework-images-for-gpu-libraries)。

## 法学硕士

vLLM 是一种非常知名且广泛使用的推理引擎。它以其扩展法学硕士推理能力而闻名。
他们提供了 vLLM 和 UV 就绪的 `vllm/vllm-openai` Docker 镜像。该图像非常适合运行批量推理。

使用 `--image` 参数来使用此 Docker 映像：

```bash
>>> hf jobs uv run --image vllm/vllm-openai --flavor l4x4 generate-responses.py
```

> [!提示]
> `vllm/vllm-openai` 映像捆绑了 CUDA 工具包和预构建的 vLLM/FlashInfer 内核。
> 通常您只需要使用它即可；您还可以重用其预构建的 Python 版本 - 请参阅下面的[Using
> framework images for GPU libraries](#using-framework-images-for-gpu-libraries)。

您可以在 [Daniel Van Strien's blog post](https://danielvanstrien.xyz/posts/2025/hf-jobs/vllm-batch-inference.html) 中找到有关作业上 vLLM 批量推理的更多信息。

## TRLTRL 是一个专为训练后模型而设计的库，使用监督微调 (SFT)、组相对策略优化 (GRPO) 和直接偏好优化 (DPO) 等技术。 `huggingface/trl` 提供了具有 UV 和所有 TRL 依赖项的最新 Docker 映像，并且可以直接与 Hugging Face Jobs 一起使用。

使用 `--image` 参数来使用此 Docker 镜像：

```bash
>>> hf jobs uv run --image huggingface/trl --flavor a100-large -s HF_TOKEN train.py
```

## 使用 GPU 库的框架图像

像 vLLM 这样的 GPU 库需要的不仅仅是 Python 包——它们还需要一个匹配系统
环境：CUDA工具包（包括`nvcc`）、NCCL、cuDNN等系统库等
上。如果省略 `--image`，`hf jobs uv run` 使用默认的 uv 图像
(`ghcr.io/astral-sh/uv:python3.12-bookworm`)，一个没有 CUDA 工具包的裸 Python 基础。你的
依赖项仍然从 PyPI 安装，但在运行时，需要该工具包的库可能会失败 -
例如 FlashInfer 的采样器 JIT 编译内核并中止：

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

传递框架镜像修复了这个问题——它提供了 CUDA 工具包，`nvcc`，以及匹配的
库——这通常就是你所需要的：

```bash
hf jobs uv run --image vllm/vllm-openai --flavor l4x4 -s HF_TOKEN generate-responses.py
```

UV 仍然会从 PyPI 重新安装脚本依赖项，但它们现在针对
图像的系统堆栈，因此框架可以工作。### 可选：重用图像的预构建 Python 版本

这些镜像还提供了框架本身的预构建、与 CUDA 匹配的构建。导入那些
而不是 UV 的全新 PyPI 安装 — 如果 PyPI 构建与图像的 CUDA 堆栈不一致，则很方便
（ABI 不匹配，或者无法构建的内核）- 将 UV 指向图像的解释器并添加
将其站点包添加到导入路径：

```bash
hf jobs uv run \
    --image vllm/vllm-openai \
    --flavor l4x4 \
    --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    -s HF_TOKEN \
    generate-responses.py
```

- `--python` 使用**图像的**解释器，保持 ABI 与其编译扩展的兼容性。
- `-e PYTHONPATH=...` 使 `import vllm` 解析为该运行的图像的预构建版本。

每个图像的路径不同，因此在 `cpu-basic` 上探测它们，而不是硬编码：

```bash
hf jobs run --flavor cpu-basic vllm/vllm-openai bash -c 'which python3; which uv; python3 -m pip show vllm | grep Location'
```

```text
/usr/bin/python3                              # pass to --python
/usr/local/bin/uv                             # uv is present, so `uv run` works
Location: /usr/local/lib/python3.12/dist-packages   # pass to PYTHONPATH
```

将 `vllm` 替换为您要重用的库。布局有所不同 - `vllm/vllm-openai` 和
`lmsysorg/sglang` 使用上面的系统`dist-packages`，而`unsloth/unsloth` 使用
虚拟环境（`/opt/venv/...`）。

> [!提示]
> 此引脚导入到图像的构建中； UV 仍然安装您声明的依赖项，因此它
> 不是加速。一个`uv run --system-site-packages`将重用图像的包和
> 跳过重新安装是[requested upstream](https://github.com/astral-sh/uv/issues/7999)。

### 数据工作室
https://huggingface.co/docs/hub/data-studio.md