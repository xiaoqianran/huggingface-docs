<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 推理端点

Inference Endpoints 提供了一种安全的生产解决方案，可以在由 Hugging Face 管理的专用自动扩展基础设施上轻松部署模型。推理端点是根据 [Hub](https://huggingface.co/models) 的模型构建的。本页面是 `huggingface_hub` 与推理端点集成的参考。有关推理端点产品的更多信息，请查看其[official documentation](https://huggingface.co/docs/inference-endpoints/index)。

> [!提示]
> 查看 [related guide](../guides/inference_endpoints) 了解如何使用 `huggingface_hub` 以编程方式管理您的推理端点。

推理端点可以通过 API 完全管理。端点用 [Swagger](https://api.endpoints.huggingface.cloud/) 记录。 [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 类是一个在此 API 之上构建的简单包装器。

## 方法

推理端点功能的子集在 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 中实现：

- [get_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_inference_endpoint) 和 [list_inference_endpoints()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_inference_endpoints) 获取有关您的推理端点的信息
- [create_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint)、[update_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_inference_endpoint) 和 [delete_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_inference_endpoint) 用于部署和管理推理端点
- [pause_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint) 和 [resume_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint) 暂停和恢复推理端点
- [scale_to_zero_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint) 手动将端点扩展到 0 个副本

## InferenceEndpoint[[huggingface_hub.InferenceEndpoint]]主要数据类是[InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)。它包含有关已部署的`InferenceEndpoint`的信息，包括其配置和当前状态。部署后，您可以使用 [InferenceEndpoint.client](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.client) 和 [InferenceEndpoint.async_client](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.async_client) 属性在端点上运行推理，它们分别返回 [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient) 和 [AsyncInferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient) 对象。

#### Huggingface_hub.InferenceEndpoint[[huggingface_hub.InferenceEndpoint]]

```python
huggingface_hub.InferenceEndpoint(namespace: str, raw: dict, _token: str | bool | None, _api: HfApi)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L44)

**参数：**

name (`str`) ：推理端点的唯一名称。

namespace (`str`) ：推理端点所在的命名空间。

存储库 (`str`) ：部署在此推理端点上的模型存储库的名称。

status ([InferenceEndpointStatus](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointStatus)) ：推理端点的当前状态。

url (`str`, *可选*) ：推理端点的 URL（如果可用）。只有部署的推理端点才会有 URL。

框架（`str`）：模型使用的机器学习框架。

revision (`str`) ：部署在推理端点上的特定模型修订版。

任务 (`str`) ：与已部署模型关联的任务。

created_at (`datetime.datetime`) ：创建推理端点时的时间戳。

Updated_at (`datetime.datetime`) ：推理端点上次更新的时间戳。type ([InferenceEndpointType](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointType)) ：推理端点的类型（公共、经过身份验证、私有）。

raw (`dict`) ：从API返回的原始字典数据。

token（`str` 或 `bool`，*可选*）：推理端点的身份验证令牌（如果在请求 API 时设置）。如果未提供，将默认为本地保存的令牌。如果您不想将令牌发送到服务器，请传递 `token=False`。

包含有关已部署的推理端点的信息。

示例：
```python
>>> from huggingface_hub import get_inference_endpoint
>>> endpoint = get_inference_endpoint("my-text-to-image")
>>> endpoint
InferenceEndpoint(name='my-text-to-image', ...)

# Get status
>>> endpoint.status
'running'
>>> endpoint.url
'https://my-text-to-image.region.vendor.endpoints.huggingface.cloud'

# Run inference
>>> endpoint.client.text_to_image(...)

# Pause endpoint to save $$$
>>> endpoint.pause()

# ...
# Resume and wait for deployment
>>> endpoint.resume()
>>> endpoint.wait()
>>> endpoint.client.text_to_image(...)
```

#### from_raw[[huggingface_hub.InferenceEndpoint.from_raw]]

```python
from_raw(raw: dict, namespace: str, token: str | bool | None = None, api: typing.Optional[ForwardRef('HfApi')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L127)

从原始字典初始化对象。

#### 客户端[[huggingface_hub.InferenceEndpoint.client]]

```python
client()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L146)

**返回：** [InferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.InferenceClient)

指向已部署端点的推理客户端。

**加薪：** [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError)

- [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError) -- 如果推理端点尚未部署。

返回客户端以对此推理端点进行预测。

#### async_client[[huggingface_hub.InferenceEndpoint.async_client]]

```python
async_client()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L168)

**返回：** [AsyncInferenceClient](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)

指向已部署端点的异步兼容推理客户端。

**加薪：** [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError)

- [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError) -- 如果推理端点尚未部署。

返回客户端以对此推理端点进行预测。#### 删除[[huggingface_hub.InferenceEndpoint.delete]]

```python
delete()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L401)

删除推理端点。

此操作不可逆。如果您不想为推理端点付费，最好选择
使用[InferenceEndpoint.pause()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.pause)暂停它或使用[InferenceEndpoint.scale_to_zero()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.scale_to_zero)将其缩放到零。

这是 [HfApi.delete_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_inference_endpoint) 的别名。

#### 获取[[huggingface_hub.InferenceEndpoint.fetch]]

```python
fetch()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L243)

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

相同的推理端点，使用最新数据进行了突变。

获取有关推理端点的最新信息。

#### 暂停[[huggingface_hub.InferenceEndpoint.pause]]

```python
pause()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L344)

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

相同的推理端点，使用最新数据进行了突变。

暂停推理端点。

暂停的推理端点不会被计费。它可以随时使用[InferenceEndpoint.resume()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.resume)恢复。
这与使用 [InferenceEndpoint.scale_to_zero()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.scale_to_zero) 将推理端点缩放为零不同，后者
当有请求时会自动重新启动。

这是 [HfApi.pause_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint) 的别名。当前对象已就地变异
来自服务器的最新数据。

#### 简历[[huggingface_hub.InferenceEndpoint.resume]]

```python
resume(running_ok: bool = True)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L362)

**参数：**running_ok (`bool`, *可选*) ：如果为 `True`，则如果推理端点已在运行，则该方法不会引发错误。默认为`True`。

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

相同的推理端点，使用最新数据进行了突变。

恢复推理端点。

这是 [HfApi.resume_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint) 的别名。当前对象已就地变异
来自服务器的最新数据。

####scale_to_zero[[huggingface_hub.InferenceEndpoint.scale_to_zero]]

```python
scale_to_zero()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L383)

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

相同的推理端点，使用最新数据进行了突变。

将推理端点缩放为零。

缩放为零的推理端点不会被收取费用。它将在下一个请求时恢复，并带有
冷启动延迟。这与使用 [InferenceEndpoint.pause()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.pause) 暂停推理端点不同，后者
需要使用[InferenceEndpoint.resume()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.resume)手动恢复。

这是 [HfApi.scale_to_zero_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint) 的别名。当前对象已就地变异
来自服务器的最新数据。

####更新[[huggingface_hub.InferenceEndpoint.update]]

```python
update(accelerator: str | None = None, instance_size: str | None = None, instance_type: str | None = None, min_replica: int | None = None, max_replica: int | None = None, scale_to_zero_timeout: int | None = None, repository: str | None = None, framework: str | None = None, revision: str | None = None, task: str | None = None, custom_image: dict | None = None, container_command: list[str] | None = None, container_args: list[str] | None = None, secrets: dict[str, str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L254)

**参数：**

加速器（`str`，*可选*）：用于推理的硬件加速器（例如`"cpu"`）。instance_size (`str`, *可选*) ：用于托管模型的实例的大小或类型（例如`"x4"`）。

instance_type（`str`，*可选*）：将部署推理端点的云实例类型（例如`"intel-icl"`）。

min_replica (`int`, *可选*) ：推理端点保持运行的最小副本（实例）数量。

max_replica（`int`，*可选*）：推理端点可扩展的最大副本（实例）数量。

scale_to_zero_timeout（`int`，*可选*）：非活动端点缩放为零之前的持续时间（以分钟为单位）。 

存储库（`str`，*可选*）：与推理端点关联的模型存储库的名称（例如`"gpt2"`）。

框架（`str`，*可选*）：用于模型的机器学习框架（例如`"custom"`）。

修订版（`str`，*可选*）：要在推理端点上部署的特定模型修订版（例如`"6c0e6080953db56375760c0471a8c5f2929baf11"`）。

任务（`str`，*可选*）：部署模型的任务（例如`"text-classification"`）。custom_image（`dict`，*可选*）：用于推理端点的自定义 Docker 映像。如果您想部署在 `text-generation-inference` (TGI) 框架上运行的推理端点（请参阅示例），这非常有用。

container_command (`list[str]`, *可选*) ：覆盖容器入口点命令（映射到 API 有效负载中的 `model.command`）。适用于托管引擎映像（例如 vLLM、SGLang）和自定义映像。

container_args (`list[str]`, *可选*) ：附加到容器入口点的参数（映射到 API 负载中的 `model.args`）。适用于托管引擎映像（例如 vLLM、SGLang）和自定义映像。

Secrets (`dict[str, str]`, *可选*) ：要注入到容器环境中的秘密值。

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

相同的推理端点，使用最新数据进行了突变。

更新推理端点。

此方法允许更新计算配置、部署模型或两者。所有参数都是
可选，但必须至少提供一项。

这是 [HfApi.update_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_inference_endpoint) 的别名。当前对象已就地变异
来自服务器的最新数据。

#### 等待[[huggingface_hub.InferenceEndpoint.wait]]

```python
wait(timeout: int | None = None, refresh_every: int = 5)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L190)

**参数：**timeout (`int`, *可选*) ：等待部署推理端点的最长时间，以秒为单位。如果`None`，就会无限期地等待。

fresh_every (`int`, *可选*) ：每次获取推理端点状态之间等待的时间，以秒为单位。默认为 5 秒。

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

相同的推理端点，使用最新数据进行了突变。

**加薪：** [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError) 或 `InferenceEndpointTimeoutError`

- [InferenceEndpointError](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpointError) -- 
  如果推理端点最终处于失败状态。
- `InferenceEndpointTimeoutError` -- 
  如果推理端点在 `timeout` 秒后未部署。

等待部署推理端点。

每隔1s就会从服务器获取一次信息。如果推理端点在`timeout`之后未部署
秒，将引发`InferenceEndpointTimeoutError`。 [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint) 将根据最新版本进行适当突变
数据。

## InferenceEndpointStatus[[huggingface_hub.InferenceEndpointStatus]]

#### Huggingface_hub.InferenceEndpointStatus[[huggingface_hub.InferenceEndpointStatus]]

```python
huggingface_hub.InferenceEndpointStatus(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L20)

一个枚举。

## InferenceEndpointType[[huggingface_hub.InferenceEndpointType]]

#### Huggingface_hub.InferenceEndpointType[[huggingface_hub.InferenceEndpointType]]

```python
huggingface_hub.InferenceEndpointType(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_inference_endpoints.py#L31)

一个枚举。## InferenceEndpointError[[huggingface_hub.InferenceEndpointError]]

#### Huggingface_hub.InferenceEndpointError[[huggingface_hub.InferenceEndpointError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/errors.py#L162)

处理推理端点时的一般异常。

### HfApi 客户端
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/hf_api.md