<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Docker 空间

空间可容纳 Streamlit 和 Gradio 范围之外的应用程序的自定义 [Docker containers](https://docs.docker.com/get-started/)。 Docker Spaces 允许用户超越以前使用标准 SDK 所能实现的限制。从 FastAPI 和 Go 端点到 Phoenix 应用程序和 ML Ops 工具，Docker Spaces 可以在许多不同的设置中提供帮助。

## 设置 Docker 空间

当 [creating a new Space](https://huggingface.co/new-space) 时选择 **Docker** 作为 SDK，将通过在 `README.md` 文件的 YAML 块中将 `sdk` 属性设置为 `docker` 来初始化您的空间。或者，给定现有的 Space 存储库，在 Spaces **README.md** 文件顶部的 `YAML` 块内设置 `sdk: docker`。您还可以通过设置 `app_port: 7860` 来更改默认暴露的端口 `7860`。之后，您可以创建一个普通的`Dockerfile`。

```Yaml
---
title: Basic Docker SDK Space
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---
```

在内部，您可以根据需要拥有任意数量的开放端口。例如，您可以在 Space 中安装 Elasticsearch 并在其默认端口 9200 上内部调用它。

如果您想向外界公开在多个端口上提供服务的应用程序，一种解决方法是使用 Nginx 等反向代理将请求从更广泛的互联网（在单个端口上）分派到不同的内部端口。

## 秘密和变量管理您可以在空间设置中管理空间的环境变量。了解更多[here](./spaces-overview#managing-secrets)。

### 变量

#### 构建时间

构建 Docker 空间时，变量作为 `build-arg` 传递。阅读 [Docker's dedicated documentation](https://docs.docker.com/engine/reference/builder/#arg) 了解如何在 Dockerfile 中使用它的完整指南。

```Dockerfile
	# Declare your environment variables with the ARG directive
	ARG MODEL_REPO_NAME

	FROM python:latest
	# [...]
	# You can use them like environment variables
	RUN predict.py $MODEL_REPO_NAME
```

#### 运行时

变量在运行时注入容器的环境中。 

### 秘密

#### 构建时间

在 Docker Spaces 中，出于安全原因，秘密管理有所不同。在 [Settings tab](./spaces-overview#managing-secrets) 中创建机密后，您可以通过在 Dockerfile 中添加以下行来公开该机密：

例如，如果 `SECRET_EXAMPLE` 是您在“设置”选项卡中创建的密钥的名称，则您可以在构建时通过将其安装到文件中来读取它，然后使用 `$(cat /run/secrets/SECRET_EXAMPLE)` 读取它。

请参阅下面的示例：
```Dockerfile
# Expose the secret SECRET_EXAMPLE at buildtime and use its value as git remote URL
RUN --mount=type=secret,id=SECRET_EXAMPLE,mode=0444,required=true \
 git init && \
 git remote add origin $(cat /run/secrets/SECRET_EXAMPLE)
```

```Dockerfile
# Expose the secret SECRET_EXAMPLE at buildtime and use its value as a Bearer token for a curl request
RUN --mount=type=secret,id=SECRET_EXAMPLE,mode=0444,required=true \
	curl test -H 'Authorization: Bearer $(cat /run/secrets/SECRET_EXAMPLE)'
```

#### 运行时

与公共变量相同，在运行时，您可以将机密作为环境变量进行访问。例如，在 Python 中，您将使用 `os.environ.get("SECRET_EXAMPLE")`。查看这个使用秘密的 Docker Space 的 [example](https://huggingface.co/spaces/DockerTemplates/secret-example)。

## 权限

容器以用户 ID 1000 运行。为了避免权限问题，您应该创建一个用户并在任何 `COPY` 或下载之前设置其 `WORKDIR`。

```Dockerfile
# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Download a checkpoint
RUN mkdir content
ADD --chown=user https://<SOME_ASSET_URL> content/<SOME_ASSET_NAME>
```始终指定 `--chown=user` 与 `ADD` 和 `COPY` 以确保新文件归您的用户所有。

如果您仍然面临权限问题，您可能需要在 `Dockerfile` 中使用 `chmod` 或 `chown` 来授予正确的权限。例如，如果你想使用目录`/data`，你可以这样做：

```Dockerfile
RUN mkdir -p /data
RUN chmod 777 /data
```

您应该始终避免多余的 chown。
> [!警告]
> 更新文件的元数据会创建存储在新层中的新副本。因此，由于所有受影响文件的重复，递归 chown 可能会导致非常大的图像。

而不是通过运行 `chown` 来修复权限：
```
COPY checkpoint .
RUN chown -R user checkpoint
```
你应该始终这样做：
```
COPY --chown=user checkpoint .
```
（同样适用于`ADD`命令）

## 数据持久化

只要 Docker Space 重新启动，写入磁盘上的数据就会丢失。要在重新启动后保留数据，您可以将 [Storage Bucket](./storage-buckets) 连接到您的空间。

目前，`/data`卷仅在运行时可用，即您无法在Dockerfile的构建步骤中使用`/data`。您还可以在特定情况下使用我们的数据集中心，您可以将状态和数据存储在 git LFS 存储库中。您可以找到持久性 [here](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver) 的示例，它使用 [⟦T33⟧ library](https://huggingface.co/docs/huggingface_hub/index) 以编程方式将文件上传到数据集存储库。此空间示例与 [this guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#scheduled-uploads) 将帮助您定义最适合您的数据类型的解决方案。

最后，在某些情况下，您可能希望使用 Space 代码中的外部存储解决方案，例如外部托管数据库、S3 等。

### 带 GPU 的 Docker 容器

您可以使用我们的 GPU 风格的 [Spaces Hardware](./spaces-gpus) 之一来运行具有 GPU 支持的 Docker 容器。

我们建议使用 Docker Hub 中的 [⟦T34⟧](https://hub.docker.com/r/nvidia/cuda) 作为基础镜像，它预装了 CUDA 和 cuDNN。

在 Docker 构建期间，您无权访问 GPU 硬件。因此，您不应在 Dockerfile 的构建步骤中尝试运行任何与 GPU 相关的命令。例如，您无法运行 `nvidia-smi` 或 `torch.cuda.is_available()` 构建映像。了解更多[here](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#description)。

## 阅读更多

- [Full Docker demo example](spaces-sdks-docker-first-demo)
- [List of Docker Spaces examples](spaces-sdks-docker-examples)
- [Spaces Examples](https://huggingface.co/SpacesExamples)

### 存储库入门
https://huggingface.co/docs/hub/repositories-getting-started.md