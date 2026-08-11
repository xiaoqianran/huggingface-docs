<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 管理您的 Space 运行时

查看 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 文档页面，获取管理 Hub 上空间的方法参考。

- 复制空间：[duplicate_space()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.duplicate_space)
- 获取当前运行时间：[get_space_runtime()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_space_runtime)
- 获取构建或运行日志：[fetch_space_logs()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_space_logs)
- 管理秘密：[add_space_secret()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.add_space_secret)和[delete_space_secret()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_space_secret)
- 管理硬件：[request_space_hardware()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.request_space_hardware)
- 管理状态：[pause_space()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_space)、[restart_space()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.restart_space)、[set_space_sleep_time()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.set_space_sleep_time)
- 等待空间准备好：[wait_for_space()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.wait_for_space)

## 数据结构

### SpaceRuntime[[huggingface_hub.SpaceRuntime]]

#### Huggingface_hub.SpaceRuntime[[huggingface_hub.SpaceRuntime]]

```python
huggingface_hub.SpaceRuntime(data: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L195)

**参数：**

stage (`str`) ：空间的当前阶段。示例：跑步。

hardware (`str` or `None`) : 空间的当前硬件。示例：“cpu 基本”。如果空间第一次是`BUILDING`，可以是`None`。

request_hardware（`str`或`None`）：请求的硬件。可能与`hardware`不同，尤其是在刚刚提出请求的情况下。示例：“t4-中”。如果尚未请求任何硬件，则可以是`None`。sleep_time (`int` 或 `None`) ：上次请求后空间将保持活动状态的秒数。默认情况下（如果值为`None`），如果 Space 在升级的硬件上运行，则它永远不会进入睡眠状态，而在空闲的“cpu-basic”硬件上它将在 48 小时后进入睡眠状态。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-gpus#sleep-time。

volumes（`list[Volume]`或`None`）：空间中安装的卷列表。每个卷都是一个 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 对象，描述其类型、源、安装路径和可选设置。 `None` 如果未附加任何卷。

raw (`dict`) ：来自服务器的原始响应。包含有关 Space 运行时的更多信息，例如副本数量、CPU 数量、内存大小……

包含有关空间当前运行时间的信息。

### SpaceHardware[[huggingface_hub.SpaceHardware]]

#### Huggingface_hub.SpaceHardware[[huggingface_hub.SpaceHardware]]

```python
huggingface_hub.SpaceHardware(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L68)

可用于在 Hub 上运行 Space 的硬件的枚举。

值可以与字符串进行比较：
```py
assert SpaceHardware.CPU_BASIC == "cpu-basic"
```

取自https://github.com/huggingface-internal/moon-landing/blob/main/server/repo_types/SpaceHardwareFlavor.ts（私人网址）。

### SpaceStage[[huggingface_hub.SpaceStage]]#### Huggingface_hub.SpaceStage[[huggingface_hub.SpaceStage]]

```python
huggingface_hub.SpaceStage(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L22)

枚举 Hub 上空间的可能阶段。

值可以与字符串进行比较：
```py
assert SpaceStage.BUILDING == "BUILDING"
```

取自https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceInfo.ts#L61（私人网址）。

### SpaceStorage[[huggingface_hub.SpaceStorage]]

#### Huggingface_hub.SpaceStorage[[huggingface_hub.SpaceStorage]]

```python
huggingface_hub.SpaceStorage(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L104)

Hub 上的空间可用的持久存储的枚举。

值可以与字符串进行比较：
```py
assert SpaceStorage.SMALL == "small"
```

取自https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceHardwareFlavor.ts#L24（私人网址）。

### SpaceVariable[[huggingface_hub.SpaceVariable]]

#### Huggingface_hub.SpaceVariable[[huggingface_hub.SpaceVariable]]

```python
huggingface_hub.SpaceVariable(key: str, values: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L273)

**参数：**

key (`str`) : 可变键。示例：`"MODEL_REPO_ID"`

value (`str`) ：变量值。示例：`"the_model_repo_id"`。

描述（`str`或无）：变量的描述。示例：`"Model Repo ID of the implemented model"`。

UpdatedAt（`datetime`或None）：变量上次更新的日期时间（如果变量已至少更新一次）。

包含有关空间当前变量的信息。### SpaceTemplate[[huggingface_hub.SpaceTemplate]]

#### Huggingface_hub.SpaceTemplate[[huggingface_hub.SpaceTemplate]]

```python
huggingface_hub.SpaceTemplate(data: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L368)

**参数：**

name (`str`) ：模板的人类友好名称（例如`"JupyterLab"`、`"chatbot"`）。

repo_id (`str`) ：模板空间的 Repo id（例如`"SpacesExamples/jupyterlab"`）。

sdk (`str`) ：构建模板所用的 SDK（例如 `"gradio"`、`"docker"`、`"static"`）。

Preferred_private (`bool`) ：是否建议从此模板创建的空间为私有。

包含有关 Hub 上可用的空间模板的信息。

由[HfApi.list_space_templates()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_space_templates)返回。 `repo_id` 可以作为 `space_template` 传递
到 [HfApi.create_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 从该模板播种一个新空间。

### 警告
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/cli.md