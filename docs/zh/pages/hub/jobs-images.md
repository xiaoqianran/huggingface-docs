<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 Docker 镜像

每个作业都在 Docker 映像内运行，该映像提供其软件环境，包括系统工具和安装的库。使用`hf jobs run`，您可以提供图像和命令。使用 `hf jobs uv run`，uv 为您的脚本准备 Python 环境，并且乔布斯会选择默认图像，除非您指定图像。

## 选择如何运行您的工作负载

当你的Python脚本的依赖可以用uv安装时，从`hf jobs uv run`开始，如[Quickstart](./jobs-quickstart)所示。当您需要额外的系统工具或想要使用准备好的软件环境时，请选择镜像。

|您需要什么 |起点|
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
|运行带有声明的依赖项的 Python 脚本 | `hf jobs uv run script.py` |
|使用镜像中已安装的工具或Python包 | `hf jobs run IMAGE COMMAND` |
|使用 uv 管理的依赖项和其他系统工具运行脚本 | `hf jobs uv run --image IMAGE script.py` |

默认图像包括Python和uv。有关当前默认图像和可用选项，请参阅[UV Jobs configuration](./jobs-configuration#uv-jobs)。您可以使用 Docker Hub 或 [build and host your own image with a Docker Space](#build-your-own-image-with-a-docker-space) 等注册表中的现有映像。

## 使用现成的图像

将 `hf jobs run` 与图像和在其中运行的命令一起使用。来自 Docker Hub 或其他注册表的任何公共映像都可以使用，包括特定标签。从小事开始：

```bash
hf jobs run ubuntu echo 'Hello from the cloud!'
```

该作业拉取映像，运行命令并退出。要使用映像已提供的软件，请选择附带该映像的映像。例如，使用 PyTorch 图像在 GPU 上创建张量并将其值加倍：

```bash
hf jobs run --flavor t4-small --timeout 5m \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- python -c 'import torch; x = torch.ones(3, device="cuda"); print(x * 2)'
```

该映像提供 PyTorch 及其 CUDA 软件堆栈。 `--flavor t4-small` 选择 GPU，`--timeout 5m` 限制作业的运行时间，Python 命令使用图像已安装的 PyTorch 包。其输出显示在作业日志中。

> [!提示]
> `--` 将作业选项与命令及其参数分开。例如，将此分隔符传递给您的命令后的 `--help`。参见[Passing arguments](./jobs-configuration#passing-arguments)。

### 使用输入和输出文件运行您自己的脚本

由 `hf jobs run` 运行的脚本必须在作业中可用。您可以使用图像中包含的脚本，或[mount a local directory](./jobs-configuration#local-directories)。例如，假设您有一个使用 `huggingface/trl` 图像中的 Transformers 和 PyTorch 包的推理脚本。您的本地 `work` 目录包含 `inference.py` 和 `inputs.jsonl`，并且脚本接受 `--input` 和 `--output` 路径。为结果创建 [Storage Bucket](./storage-buckets) 后，运行它：

```bash
hf jobs run --flavor t4-small --timeout 10m \
    -v ./work:/work \
    -v hf://buckets/YOUR_USERNAME/results:/output \
    huggingface/trl \
    -- python /work/inference.py --input /work/inputs.jsonl --output /output/results.jsonl
```

将 `YOUR_USERNAME` 替换为存储桶所有者的命名空间。 CLI上传本地目录并将其挂载到`/work`。您的脚本读取那里的输入并将结果写入安装在 `/output` 的存储桶，在作业完成后它们仍然可用。根据您的脚本调整命令的参数、硬件和超时；其他输入和输出选项请参见[Volumes](./jobs-configuration#volumes)。

## 使用带有 uv 的图像

当您的脚本需要默认图像之外的系统工具，同时保持 uv 的 Python 依赖工作流程时，请使用 `hf jobs uv run --image`。所选镜像必须安装`uv`。

环境分为三个部分：- **镜像中的系统工具和库**，例如`ffmpeg`或CUDA工具包，根据镜像的配置可供作业使用。
- **镜像中安装的Python包**属于其现有的Python环境。
- **脚本依赖项。** 使用 [inline dependency header](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)，uv 将您的 Python 包安装在隔离的环境中。映像中预安装的软件包不会自动可用。

例如，vLLM 脚本可能需要带有 CUDA 工具的映像，同时使用 uv 安装其自己声明的 Python 依赖项。系统工具仍然必须与 uv 安装的软件包兼容。

如果您想直接使用图像预装的 PyTorch、TRL 或 vLLM，请使用 `hf jobs run`。如果您还需要 uv 安装额外的依赖项，请参阅[Reuse the image's packages and add dependencies with uv](#reuse-the-images-packages-and-add-dependencies-with-uv)。

## 使用 Docker Space 构建您自己的镜像

Docker Space 可以使用您的工作负载所需的工具（包括您自己的脚本）构建和托管映像，并且该映像可以是私有的，无需注册帐户。例如，在Python镜像中添加FFmpeg，准备处理音频或视频的环境。

创建一个 [Docker Space](./spaces-sdks-docker#setting-up-docker-spaces) 并将此 `Dockerfile` 添加到其存储库的根目录中：

```dockerfile
FROM python:3.12-slim-bookworm

# Install FFmpeg for audio and video processing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Match the non-root user ID used by Docker Spaces.
RUN useradd --create-home --uid 1000 user
USER user

# Check that FFmpeg is available.
CMD ["ffmpeg", "-version"]
````FROM` 提供 Python。 FFmpeg 不包含在此基础映像中，因此 `RUN` 在构建时安装它。 `useradd` 创建一个用户，其主目录和 UID `1000`，与 [Docker Spaces](./spaces-sdks-docker#permissions) 匹配； `USER` 以该用户身份运行命令。 `CMD` 设置一个默认命令，这里检查FFmpeg的安装情况。

提交文件并等待空间的构建日志显示镜像已推送。然后传递您通常放置图像名称的空间 URL，将 `YOUR_USERNAME/video-tools` 替换为您的空间 ID：

```bash
hf jobs run --flavor cpu-basic --timeout 5m \
    hf.co/spaces/YOUR_USERNAME/video-tools -- ffmpeg -version
```

该版本显示在作业日志中。在空间 URL 之后提供不同的命令，以将 FFmpeg 或 Python 用于您的工作负载。要在映像中包含您自己的脚本，请将一行（例如 `COPY --chown=user:user process.py /app/process.py`）添加到 Dockerfile 并运行 `python /app/process.py`。有关安装数据和保存结果，请参阅[input and output files](#run-your-own-script-with-input-and-output-files)。

> [!注意]
> 此示例打印版本并退出，因此 Space 在构建后显示运行时错误。这很好：作业只需要构建的映像，而不是正在运行的空间。> [!提示]
> 空间的构建图像不保证保持可拉动。它可能会因注册表维护或区域移动而丢失。如果作业报告未找到映像，请在工厂重新启动空间以重建它。作业始终使用最新版本，因此新的提交也会替换该映像。

## 示例图像

注册表映像由其发布者维护，并且可以固定到特定标签。

|图片|它提供什么 |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` |上例中使用的 PyTorch 和 CUDA 开发环境。            |
| `huggingface/trl` |用于训练后和相关 Python 工作负载的 TRL、Transformers、PyTorch 和 uv。 |
| `vllm/vllm-openai` |用于 LLM 推理的 vLLM、uv 和 CUDA 工具。                                      |

使用 `hf jobs run` 来运行带有镜像已安装软件包的命令。对于`hf jobs uv run`，检查图像是否安装了uv并按照上面的[environment guidance](#use-an-image-with-uv)进行操作。浏览 [Examples & Tutorials](./jobs-examples) 以了解要运行和适应的工作负载。## 重用图像的包并使用 uv 添加依赖项

框架映像提供预安装的软件包，这些软件包可能会很慢或难以构建，例如 PyTorch、vLLM 及其 CUDA 扩展。您的脚本可能仍然需要额外的 Python 包 - 例如，加载特定的数据格式或跟踪实验。

您可以在使用 uv 安装这些附加功能时重用图像的预安装堆栈。在脚本的 `# /// script` 依赖项标头中声明附加包，然后将 uv 指向图像的解释器并将其站点包添加到导入路径：

```bash
hf jobs uv run \
    --image vllm/vllm-openai \
    --flavor l4x4 \
    --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    -s HF_TOKEN \
    generate-responses.py
```

- `--python` 使用图像解释器创建 uv 环境，匹配其编译扩展使用的 Python 版本。它本身不会公开图像的包。
- `-e PYTHONPATH=...` 使 `import vllm` 解析为该运行的图像的预构建版本。
- 修剪您的 `# /// script` 依赖项以适应图像_缺乏_的内容。 `PYTHONPATH` 在 uv 环境之前被搜索，因此图像会遮蔽标头为同一包声明的任何内容 - 包括您固定的较新版本。您保留的依赖项仍然可以传递地引入这些包； uv 不使用 `PYTHONPATH` 来满足依赖解析。每个图像的路径不同，因此在 `cpu-basic` 上探测它们，而不是硬编码：

```bash
hf jobs run --flavor cpu-basic vllm/vllm-openai bash -c 'which python3; which uv; python3 -m pip show vllm | grep Location'
```

```text
/usr/bin/python3                              # pass to --python
/usr/local/bin/uv                             # uv is present, so `uv run` works
Location: /usr/local/lib/python3.12/dist-packages   # pass to PYTHONPATH
```

将 `vllm` 替换为您要重用的库。布局有所不同 - `vllm/vllm-openai`和`lmsysorg/sglang`使用上面的系统`dist-packages`，`unsloth/unsloth`使用virtualenv（`/opt/venv/...`），`huggingface/trl`使用conda（`/opt/conda/lib/python3.11/site-packages`，继承自`pytorch/pytorch`）。

> [!提示]
> 这会选择用于导入的图像构建，而不是用于 uv 的依赖解析器。修剪标头可以减少冗余安装，但不能保证消除它们。跳过 `PYTHONPATH` 步骤的 `uv run --system-site-packages` 是 [requested upstream](https://github.com/astral-sh/uv/issues/7999)。

对于`huggingface/trl`镜像，对应的解释器和包路径为：

```bash
hf jobs uv run \
    --image huggingface/trl \
    --flavor a100-large \
    --python /opt/conda/bin/python3 \
    -e PYTHONPATH=/opt/conda/lib/python3.11/site-packages \
    -s HF_TOKEN \
    train.py
```

这些命令使用 `generate-responses.py` 和 `train.py` 作为您自己的脚本的名称。

## 故障排除

### 使用 GPU 库的框架图像

一些 GPU 库需要额外的系统工具，例如 CUDA 编译器 (`nvcc`)、NCCL 或 cuDNN。安装他们的 Python 包并不能保证映像包含他们需要的所有工具。例如，FlashInfer 的采样器在尝试在没有 CUDA 工具包的情况下编译映像中的内核时可能会失败：

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

使用所需的 CUDA 工具传递框架映像可以解决此缺少工具包的错误：

```bash
hf jobs uv run --image vllm/vllm-openai --flavor l4x4 -s HF_TOKEN generate-responses.py
```uv 仍然会单独解析并安装您的脚本依赖项。该映像提供系统工具，但不保证解析的 Python 包与其兼容。

### 小部件示例
https://huggingface.co/docs/hub/models-widgets-examples.md