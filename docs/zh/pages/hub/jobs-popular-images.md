<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 热门图片

以下是来自流行框架的现成 Docker 映像列表，您可以在作业中使用这些映像。

选择如何运行您的代码：

- [⟦T9⟧](./jobs-configuration#docker-jobs)直接在图像中执行命令，
  所以你可以使用它预装的Python包。
- [⟦T10⟧](./jobs-configuration#uv-jobs) 使用 UV 运行您的脚本。对于一个脚本
  `# /// script` 依赖标头，UV 创建一个隔离环境。 `--image`耗材
  系统环境，但不会自动公开图像的Python包。

> [!提示]
> 使用`hf jobs run`，当命令使用选项时，在图像和命令之间添加`--`
> 乔布斯也认可的，例如`--help`。参见[Docker Jobs configuration](./jobs-configuration#docker-jobs)。

下面的 UV 示例需要安装了 `uv` 的图像。组合 UV 管理的依赖项
与图像的预安装包，请参阅[Reuse the image's packages and add dependencies with UV](#reuse-the-images-packages-and-add-dependencies-with-uv)。

> [!提示]
> 对于 GPU 推理库，传递 `--image`，以便运行获得匹配的 CUDA 系统堆栈
>（工具包、`nvcc`、库）— 请参阅下面的[Using framework images for GPU
> libraries](#using-framework-images-for-gpu-libraries)。

## 法学硕士

vLLM 是一种非常知名且广泛使用的推理引擎。它以其扩展法学硕士推理能力而闻名。
他们提供了 vLLM 和 UV 就绪的 `vllm/vllm-openai` Docker 镜像。该图像非常适合运行批量推理。

使用 `--image` 参数来使用此 Docker 映像：

```bash
>>> hf jobs uv run --image vllm/vllm-openai --flavor l4x4 generate-responses.py
```> [!提示]
> 使用 `hf jobs uv run`，此图像提供 CUDA 工具，但 UV 从您的
> 脚本的依赖项，而不是使用映像的预安装 vLLM 构建及其内核。
> 要使用图像的构建，请使用 `hf jobs run` 运行命令，或使用
> [reuse approach](#reuse-the-images-packages-and-add-dependencies-with-uv) 下面。

您可以在 [Daniel Van Strien's blog post](https://danielvanstrien.xyz/posts/2025/hf-jobs/vllm-batch-inference.html) 中找到有关作业上 vLLM 批量推理的更多信息。

## TRL

TRL 是一个专为训练后模型而设计的库，使用监督微调 (SFT)、组相对策略优化 (GRPO) 和直接偏好优化 (DPO) 等技术。 `huggingface/trl` 提供了具有 UV 和所有 TRL 依赖项的最新 Docker 映像，并且可以直接与 Hugging Face Jobs 一起使用。

使用 `--image` 参数来使用此 Docker 镜像：

```bash
>>> hf jobs uv run --image huggingface/trl --flavor a100-large -s HF_TOKEN train.py
```

这为您的脚本提供了图像的 CUDA 堆栈，但不是其 Python 包，包括 TRL 和
火炬。具有 `# /// script` 标头的脚本使用 UV 从该标头解析的依赖项。
要直接使用图像的 TRL，请使用 `hf jobs run`。例如，检查其安装的版本：

```bash
>>> hf jobs run --flavor cpu-basic huggingface/trl python -c 'import trl; print(trl.__version__)'
```

如果您在重用图像包时需要 UV 的脚本工作流程，请选择其解释器
并按照 [Reuse the image's packages and add dependencies with UV](#reuse-the-images-packages-and-add-dependencies-with-uv) 中的描述公开其站点包：

```bash
hf jobs uv run \
    --image huggingface/trl \
    --flavor a100-large \
    --python /opt/conda/bin/python3 \
    -e PYTHONPATH=/opt/conda/lib/python3.11/site-packages \
    -s HF_TOKEN \
    train.py
```

## 使用 GPU 库的框架图像像 vLLM 这样的 GPU 库需要的不仅仅是 Python 包——它们还需要一个匹配系统
环境：CUDA工具包（包括`nvcc`）、NCCL、cuDNN等系统库等
上。如果省略 `--image`，`hf jobs uv run` 使用默认的 uv 图像
(`ghcr.io/astral-sh/uv:python3.12-bookworm`)，一个没有 CUDA 工具包的裸 Python 基础。你的
依赖项仍然从 PyPI 安装，但在运行时，需要该工具包的库可能会失败 -
例如 FlashInfer 的采样器 JIT 编译内核并中止：

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

使用所需的 CUDA 工具传递框架映像可以解决此缺少工具包的错误：

```bash
hf jobs uv run --image vllm/vllm-openai --flavor l4x4 -s HF_TOKEN generate-responses.py
```

UV 仍然会单独解析并安装您的脚本依赖项。影像供应系统
工具，但不保证解析后的Python包与其兼容。

### 重用图像的包并添加 UV 的依赖项

框架映像提供预装的软件包，这些软件包可能会很慢或难以构建，例如
PyTorch、vLLM 及其 CUDA 扩展。您可能仍然需要额外的 Python 包
脚本 - 例如，加载特定的数据格式或跟踪实验。您可以在使用 UV 安装这些附加功能时重复使用图像的预安装堆栈。声明
在脚本的 `# /// script` 依赖项标头中添加其他包，然后将 UV 指向
图像的解释器并将其站点包添加到导入路径：

```bash
hf jobs uv run \
    --image vllm/vllm-openai \
    --flavor l4x4 \
    --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    -s HF_TOKEN \
    generate-responses.py
```

- `--python` 使用**图像的**解释器创建 UV 环境，与 Python 匹配
  其编译扩展使用的版本。它本身不会公开图像的包。
- `-e PYTHONPATH=...` 使 `import vllm` 解析为该运行的图像的预构建版本。
- 修剪您的 `# /// script` 依赖关系以适应图像*缺乏*的内容。 `PYTHONPATH` 被搜索
  在 UV 环境之前，因此图像会遮蔽标头声明的任何内容
  包 - 包括您固定的较新版本。您保留的依赖关系仍然可以引入
  这些包是传递性的； UV 不使用`PYTHONPATH` 来满足依赖解析。

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
`lmsysorg/sglang`使用上面的系统`dist-packages`，`unsloth/unsloth`使用virtualenv
(`/opt/venv/...`)，并且`huggingface/trl`使用conda
（`/opt/conda/lib/python3.11/site-packages`，继承自`pytorch/pytorch`）。> [!提示]
> 这会选择用于导入的图像构建，而不是用于 UV 的依赖解析器。修剪
> header 可以减少冗余安装，但不保证它们被消除。紫外线运行
> --system-site-packages` that would skip the `PYTHONPATH`步骤是[requested
> upstream](https://github.com/astral-sh/uv/issues/7999)。

### 数据工作室
https://huggingface.co/docs/hub/data-studio.md