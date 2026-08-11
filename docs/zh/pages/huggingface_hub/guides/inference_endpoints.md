<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 推理端点

Inference Endpoints 提供了一个安全的生产解决方案，可以在由 Hugging Face 管理的专用自动扩展基础设施上轻松部署任何 `transformers`、`sentence-transformers` 和 `diffusers` 模型。推理端点是根据 [Hub](https://huggingface.co/models) 的模型构建的。
在本指南中，我们将学习如何使用`huggingface_hub`以编程方式管理推理端点。有关推理端点产品本身的更多信息，请查看其[official documentation](https://huggingface.co/docs/inference-endpoints/index)。

本指南假设 `huggingface_hub` 已正确安装并且您的计算机已登录。如果情况尚未如此，请查看 [Quick Start guide](https://huggingface.co/docs/huggingface_hub/quick-start#quickstart)。支持推理端点 API 的最低版本是`v0.19.0`。> [!提示]
> **新功能：** 现在可以通过简单的 API 调用从 [HF model catalog](https://endpoints.huggingface.co/catalog) 部署推理端点。该目录是精心策划的模型列表，可以使用优化设置进行部署。您无需配置任何东西，所有繁重的事情都由我们承担！所有型号和设置均保证经过测试，以提供最佳的成本/性能平衡。  [create_inference_endpoint_from_catalog()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint_from_catalog) 的工作方式与 [create_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint) 相同，需要传递的参数少得多。您可以选择指定 `accelerator`（`"cpu"`、`"gpu"` 或 `"neuron"`）来覆盖默认硬件选择。您可以使用 [list_inference_catalog()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_inference_catalog) 以编程方式检索目录。
>
> 请注意，这仍然是一个实验性功能。如果您使用它，请告诉我们您的想法！

## 创建推理端点

第一步是使用 [create_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint) 创建推理端点：

```py
>>> from huggingface_hub import create_inference_endpoint

>>> endpoint = create_inference_endpoint(
...     "my-endpoint-name",
...     repository="gpt2",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="cpu",
...     vendor="aws",
...     region="us-east-1",
...     type="authenticated",
...     instance_size="x2",
...     instance_type="intel-icl"
... )
```

或者通过 CLI：

```bash
hf endpoints deploy my-endpoint-name --repo gpt2 --framework pytorch --accelerator cpu --vendor aws --region us-east-1 --instance-size x2 --instance-type intel-icl --task text-generation

# Deploy from the catalog with a single command
hf endpoints catalog deploy --repo openai/gpt-oss-120b

# Deploy from the catalog with a specific accelerator
hf endpoints catalog deploy --repo openai/gpt-oss-120b --accelerator gpu
```在此示例中，我们创建了一个名为 `"my-endpoint-name"` 的 `authenticated` 推理端点，为 `text-generation` 提供 [gpt2](https://huggingface.co/gpt2) 服务。 `authenticated` 推理端点意味着需要您的令牌才能访问 API。我们还需要提供其他信息来配置硬件要求，例如供应商、区域、加速器、实例类型和大小。您可以查看可用资源列表[here](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aprovider/list_vendors)。或者，为了方便起见，您可以使用 [Web interface](https://ui.endpoints.huggingface.co) 手动创建推理端点。有关高级设置及其使用的详细信息，请参阅此[guide](https://huggingface.co/docs/inference-endpoints/guides/advanced)。

[create_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint)返回的值是一个[InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)对象：

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

或者通过 CLI：

```bash
hf endpoints describe my-endpoint-name
```

它是一个保存有关端点信息的数据类。您可以访问重要属性，例如`name`，`repository`，`status`，`task`，`created_at`，`updated_at`等。如果需要，您还可以使用`endpoint.raw`访问服务器的原始响应。

创建推理端点后，您可以在 [personal dashboard](https://ui.endpoints.huggingface.co/) 上找到它。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/inference_endpoints_created.png)

#### 使用自定义图像默认情况下，推理端点是根据 Hugging Face 提供的 Docker 映像构建的。但是，可以使用 `custom_image` 参数指定任何 docker 镜像。一个常见的用例是使用 [text-generation-inference](https://github.com/huggingface/text-generation-inference) 框架运行法学硕士。这可以这样做：

```python
# Start an Inference Endpoint running Zephyr-7b-beta on TGI
>>> from huggingface_hub import create_inference_endpoint
>>> endpoint = create_inference_endpoint(
...     "aws-zephyr-7b-beta-0486",
...     repository="HuggingFaceH4/zephyr-7b-beta",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="gpu",
...     vendor="aws",
...     region="us-east-1",
...     type="authenticated",
...     instance_size="x1",
...     instance_type="nvidia-a10g",
...     custom_image={
...         "healthRoute": "/health",
...         "env": {
...             "MAX_BATCH_PREFILL_TOKENS": "2048",
...             "MAX_INPUT_LENGTH": "1024",
...             "MAX_TOTAL_TOKENS": "1512",
...             "MODEL_ID": "/repository"
...         },
...         "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
...     },
... )
```

作为 `custom_image` 传递的值是一个字典，其中包含 docker 容器的 url 和运行它的配置。有关它的更多详细信息，请查看[Swagger documentation](https://api.endpoints.huggingface.cloud/#/v2%3A%3Aendpoint/create_endpoint)。

对于需要自定义入口点或运行时标志的容器，请传递 `container_command` 和/或 `container_args` （每个都是令牌列表）。它们映射到 API 负载中的 `model.command` 和 `model.args`。它们不依赖于自定义镜像：托管引擎镜像（例如 vLLM、SGLang）也通过 `container_args` 接受引擎标志。同样可以通过 CLI 通过 `hf endpoints deploy ... --container-command "..." --container-args "..."` 获得。

### 获取或列出现有的推理端点

在某些情况下，您可能需要管理之前创建的推理端点。如果您知道名称，则可以使用 [get_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_inference_endpoint) 获取它，它返回一个 [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 对象。或者，您可以使用 [list_inference_endpoints()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_inference_endpoints) 检索所有推理端点的列表。两种方法都接受可选的 `namespace` 参数。您可以将 `namespace` 设置为您所属的任何组织。否则，它默认为您的用户名。```py
>>> from huggingface_hub import get_inference_endpoint, list_inference_endpoints

# Get one
>>> get_inference_endpoint("my-endpoint-name")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# List all endpoints from an organization
>>> list_inference_endpoints(namespace="huggingface")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]

# List all endpoints from all organizations the user belongs to
>>> list_inference_endpoints(namespace="*")
[InferenceEndpoint(name='aws-starchat-beta', namespace='huggingface', repository='HuggingFaceH4/starchat-beta', status='paused', url=None), ...]
```

或者通过 CLI： 

```bash
hf endpoints describe my-endpoint-name
hf endpoints ls --namespace huggingface
hf endpoints ls --namespace '*'
```

## 检查部署状态

在本指南的其余部分中，我们假设有一个名为 `endpoint` 的 [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 对象。您可能已经注意到端点具有类型为 [InferenceEndpointStatus](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointStatus) 的 `status` 属性。当推理端点部署并可访问时，状态应为 `"running"` 并设置 `url` 属性：

```py
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

在达到 `"running"` 状态之前，推理端点通常会经历 `"initializing"` 或 `"pending"` 阶段。您可以通过运行[fetch()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.fetch)来获取端点的新状态。与 [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 中向服务器发出请求的所有其他方法一样，`endpoint` 的内部属性也发生了变化：

```py
>>> endpoint.fetch()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
```

或者通过 CLI：

```bash
hf endpoints describe my-endpoint-name
```

您可以直接调用[wait()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.wait)，而不是在等待推理端点运行时获取推理端点状态。该助手将 `timeout` 和 `fetch_every` 参数（以秒为单位）作为输入，并将阻塞线程，直到部署推理端点。默认值分别为`None`（无超时）和`5`秒。

```py
# Pending endpoint
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)

# Wait 10s => raises a InferenceEndpointTimeoutError
>>> endpoint.wait(timeout=10)
    raise InferenceEndpointTimeoutError("Timeout while waiting for Inference Endpoint to be deployed.")
huggingface_hub._inference_endpoints.InferenceEndpointTimeoutError: Timeout while waiting for Inference Endpoint to be deployed.

# Wait more
>>> endpoint.wait()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='running', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
```

如果设置了 `timeout` 并且推理端点加载时间过长，则会引发 `InferenceEndpointTimeoutError` 超时错误。

## 运行推理

一旦您的推理端点启动并运行，您终于可以在其上运行推理了！[InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 有两个属性 `client` 和 `async_client` 分别返回 [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 和 [AsyncInferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient) 对象。

```py
# Run text_generation task:
>>> endpoint.client.text_generation("I am")
' not a fan of the idea of a "big-budget" movie. I think it\'s a'

# Or in an asyncio context:
>>> await endpoint.async_client.text_generation("I am")
```

如果推理端点未运行，则会引发 [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError) 异常：

```py
>>> endpoint.client
huggingface_hub._inference_endpoints.InferenceEndpointError: Cannot create a client for this Inference Endpoint as it is not yet deployed. Please wait for the Inference Endpoint to be deployed using `endpoint.wait()` and try again.
```

有关如何使用[InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)的更多详细信息，请查看[Inference guide](../guides/inference)。

## 管理生命周期

现在我们已经了解了如何创建推理端点并在其上运行推理，接下来让我们看看如何管理其生命周期。

> [!提示]
> 在本节中，我们将看到 [pause()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.pause)、[resume()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.resume)、[scale_to_zero()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.scale_to_zero)、[update()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.update) 和 [delete()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.delete) 等方法。为了方便起见，所有这些方法都是添加到 [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 的别名。如果您愿意，还可以使用`HfApi`中定义的通用方法：[pause_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint)、[resume_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint)、[scale_to_zero_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint)、[update_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_inference_endpoint)和[delete_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_inference_endpoint)。

### 暂停或缩放至零

为了在推理端点未使用时降低成本，您可以选择使用 [pause()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.pause) 暂停它，或使用 [scale_to_zero()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.scale_to_zero) 将其缩放为零。> [!提示]
> *暂停*或*缩放至零*的推理端点不会产生任何费用。这两者之间的区别在于，“暂停”端点需要使用[resume()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.resume)显式“恢复”。相反，如果对其进行推理调用，“缩放至零”端点将自动启动，并具有额外的冷启动延迟。推理端点还可以配置为在闲置一段时间后自动缩放为零。

```py
# Pause and resume endpoint
>>> endpoint.pause()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='paused', url=None)
>>> endpoint.resume()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='pending', url=None)
>>> endpoint.wait().client.text_generation(...)
...

# Scale to zero
>>> endpoint.scale_to_zero()
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2', status='scaledToZero', url='https://jpj7k2q4j805b727.us-east-1.aws.endpoints.huggingface.cloud')
# Endpoint is not 'running' but still has a URL and will restart on first call.
```

或者通过 CLI：

```bash
hf endpoints pause my-endpoint-name
hf endpoints resume my-endpoint-name
hf endpoints scale-to-zero my-endpoint-name
```

### 更新型号或硬件要求

在某些情况下，您可能还想更新推理端点而不创建新的推理端点。您可以更新托管模型或运行模型的硬件要求。您可以使用 [update()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.update) 来做到这一点：

```py
# Change target model
>>> endpoint.update(repository="gpt2-large")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Update number of replicas
>>> endpoint.update(min_replica=2, max_replica=6)
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Update to larger instance
>>> endpoint.update(accelerator="cpu", instance_size="x4", instance_type="intel-icl")
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)

# Update the container arguments, e.g. engine flags on an endpoint running vLLM or SGLang.
# Replaces the current value, it is not appended.
>>> endpoint.update(container_args=["--enable-auto-tool-choice", "--tool-call-parser", "lfm2"])
InferenceEndpoint(name='my-endpoint-name', namespace='Wauplin', repository='gpt2-large', status='pending', url=None)
```

或者通过 CLI：

```bash
hf endpoints update my-endpoint-name --repo gpt2-large
hf endpoints update my-endpoint-name --min-replica 2 --max-replica 6
hf endpoints update my-endpoint-name --accelerator cpu --instance-size x4 --instance-type intel-icl
hf endpoints update my-endpoint-name --container-args "--enable-auto-tool-choice --tool-call-parser lfm2"
```

### 删除端点

最后，如果您不再使用推理端点，您可以简单地调用`~InferenceEndpoint.delete()`。

> [!警告]
> 这是不可恢复的操作，将完全删除端点，包括其配置、日志和使用指标。您无法恢复已删除的推理端点。

## 一个端到端的例子推理端点的一个典型用例是一次处理一批作业以限制基础设施成本。您可以使用我们在本指南中看到的内容自动执行此过程：

```py
>>> import asyncio
>>> from huggingface_hub import create_inference_endpoint

# Start endpoint + wait until initialized
>>> endpoint = create_inference_endpoint(name="batch-endpoint",...).wait()

# Run inference
>>> client = endpoint.client
>>> results = [client.text_generation(...) for job in jobs]

# Or with asyncio
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# Pause endpoint
>>> endpoint.pause()
```

或者，如果您的推理端点已存在并且已暂停：

```py
>>> import asyncio
>>> from huggingface_hub import get_inference_endpoint

# Get endpoint + wait until initialized
>>> endpoint = get_inference_endpoint("batch-endpoint").resume().wait()

# Run inference
>>> async_client = endpoint.async_client
>>> results = asyncio.gather(*[async_client.text_generation(...) for job in jobs])

# Pause endpoint
>>> endpoint.pause()
```

### 桶
https://huggingface.co/docs/huggingface_hub/v1.27.0/guides/buckets.md