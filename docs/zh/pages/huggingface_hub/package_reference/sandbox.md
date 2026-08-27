<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 沙箱

查看 [Sandboxes guide](../guides/sandbox) 了解如何使用它们。

## 沙箱[[huggingface_hub.Sandbox]]

#### Huggingface_hub.Sandbox[[huggingface_hub.Sandbox]]

```python
huggingface_hub.Sandbox(id: str, server: _SandboxServer, local_id: str | None, owns_sandbox: bool, owns_server: bool)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L473)

运行在 Hugging Face Jobs 上的隔离云机器。

使用 [Sandbox.create()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.create) 创建一个专用的作业（每个沙箱一个作业），或者从 [SandboxPool](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxPool) 获得许多廉价的共享作业。

使用 [Sandbox.connect()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.connect) 从任何地方重新连接到正在运行的沙箱。用作上下文管理器以在退出时终止它：

```python
>>> from huggingface_hub import Sandbox
>>> with Sandbox.create(image="python:3.12") as sbx:
...     print(sbx.run("python --version").stdout)
```

#### 关闭[[huggingface_hub.Sandbox.close]]

```python
close()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L682)

释放本地 HTTP 客户端而不终止沙箱。幂等。

池沙箱无操作（客户端属于池的主机）。

#### 连接[[huggingface_hub.Sandbox.connect]]

```python
connect(sandbox_id: str, namespace: str | None = None, token: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L616)

仅使用其 ID 从任何地方重新附加到正在运行的沙箱。

#### 创建[[huggingface_hub.Sandbox.create]]

```python
create(image: str = 'python:3.12', flavor: str = 'cpu-basic', idle_timeout: int | float | str | None = 600, env: dict[str, typing.Any] | None = None, secrets: dict[str, typing.Any] | None = None, volumes: typing.Optional[typing.List[huggingface_hub._space_api.Volume]] = None, namespace: str | None = None, forward_hf_token: bool = False, start_timeout: float = 120.0, token: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L515)

**参数：**

镜像（`str`，*可选*，默认为`"python --3.12"`）：任何带有`/bin/sh`（Docker Hub或`hf.co/spaces/...`）的Docker镜像。

flavor (`str`, *可选*, 默认为 `"cpu-basic"`) : 硬件风格，例如`"cpu-basic"`，`"a10g-small"`。参见`hf jobs hardware`。idle_timeout（`int`或`float`或`str`，*可选*，默认为`600`）：在长时间不活动（没有API调用，没有正在运行的进程）后自动关闭。默认为 10 分钟；通过 `None` 禁用。

env (`dict[str, Any]`, *可选*) ：沙箱中可用的环境变量。

Secrets (`dict[str, Any]`, *可选*) ：秘密环境变量（加密的服务器端）。

卷（`List[Volume]`，*可选*）：要安装的 HF 存储库/存储桶，请参阅[Volume](/docs/huggingface_hub/v1.29.0/en/package_reference/jobs#huggingface_hub.Volume)。

命名空间（`str`，*可选*）：要在其下运行的用户或组织命名空间（默认为当前用户）。

forward_hf_token（`bool`，*可选*，默认为`False`）：如果为 True，您的 HF 令牌将作为 `HF_TOKEN`（选择加入）注入。

start_timeout (`float`，*可选*，默认为`120.0`)：等待沙箱准备就绪的最大秒数。

令牌（`str`，*可选*）：HF 令牌覆盖。

创建一个专用沙箱（一个 HF 作业）并进行阻止，直到其准备就绪（在 cpu-basic 上约为 7 秒）。

每个沙箱都是一个完全隔离的虚拟机，因此这是 GPU 的正确选择
工作负载或不受信任的代码。要改为扇出许多廉价的 CPU 沙箱，请使用
[SandboxPool](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxPool)。

作业以固定的 24 小时最大生命周期运行； `idle_timeout`是真的
keeper——闲置的沙箱在此之前就会自行关闭。图像只需要`/bin/sh`。沙盒服务器在启动时下载
`wget`/`curl`（如果可用），否则读取始终安装的服务器存储桶（其中
冷启动时间增加约 2-3 秒，因此运输 `wget`/`curl` 可以保持快速）。

#### 进程[[huggingface_hub.Sandbox.processes]]

```python
processes()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L821)

列出该沙箱的后台进程。

返回以[Sandbox.run()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.run)`(..., background=True)`启动的进程；停一
与[SandboxProcess.kill()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess.kill)。已完成的流程保持列出（带有 `running=False` 和
他们的`exit_code`）直到沙箱被删除。

#### proxy_url_for[[huggingface_hub.Sandbox.proxy_url_for]]

```python
proxy_url_for(port: int | str, path: str = '/', scheme: str = 'https://')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L855)

**参数：**

port (`int` 或 `str`) ：内部服务器监听的端口（池：unix 套接字的 `<port>`）。

路径（`str`，*可选*，默认为`"/"`）：内部服务器上指向的路径，例如`"/ws"`。

方案（`str`，*可选*，默认为`"https --//"`）：用于构建链接的 URL 方案。默认为`"https://"`；为 WebSocket 客户端传递`"wss://"`（代理与协议无关，因此只有客户端方案发生变化）。

**返回：** `str`

像 `https://<job_id>--49983.hf.jobs/v1/.../proxy/8000/ws` 这样的 URL（或者
`wss://...` 与 `scheme="wss://"`）。

代理到在此沙箱*内部*运行的服务器的公共 URL。对返回 URL 的请求由工作中的沙箱服务器转发到
您在 `port` 上的沙箱中启动的服务器，包括 WebSocket (`ws(s)://`)
升级和流式响应。将其与 `proxy_headers` 配对以进行身份​​验证。

沙箱必须如何监听 `port`：

- **池/共享沙箱**：它无法绑定TCP端口（Landlock），因此绑定一个
  **unix 套接字** 位于 `$SBX_PROXY_DIR/<port>.sock` （`SBX_PROXY_DIR` 环境变量
  在每个沙箱中都设置）。例如。 `uvicorn app:app --uds $SBX_PROXY_DIR/8000.sock`。
- **专用沙箱**：在`127.0.0.1:<port>`上绑定普通TCP端口。 （你可以
  也可以直接通过作业代理公开端口，而无需经过此处。）

示例：
```python
>>> url = sandbox.proxy_url_for(8000, "/ws", scheme="wss://")
>>> import websockets
>>> async with websockets.connect(url, additional_headers=sandbox.proxy_headers) as ws:
...     await ws.send("hello")
```

#### 运行[[huggingface_hub.Sandbox.run]]

```python
run(cmd: typing.Union[str, typing.List[str]], shell: bool | None = None, env: dict[str, typing.Any] | None = None, cwd: str | None = None, timeout: float | None = None, stdin: str | None = None, on_stdout: typing.Optional[typing.Callable[[str], NoneType]] = None, on_stderr: typing.Optional[typing.Callable[[str], NoneType]] = None, check: bool = True, background: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L728)

**参数：**

cmd (`str` 或 `List[str]`) ：shell 命令字符串（使用 `/bin/sh -c` 运行）或 argv 列表（直接执行）。

shell (`bool`, *可选*) ：强制执行模式，而不是从`cmd`的类型推断。 `True`贯穿`/bin/sh -c`并且要求`cmd`是一个字符串； `False` 直接执行 `cmd` 并要求它是一个 argv 列表。 `None`（默认）从类型推断。显式设置它以避免类型驱动的 footgun（例如 `["echo hi"]` 作为名为 `"echo hi"` 的单个程序执行）。

env (`dict[str, Any]`, *可选*) ：此命令的额外环境变量。cwd（`str`，*可选*）：工作目录。

timeout (`float`, *可选*) ：在这么多秒后终止命令（整个进程组）。

stdin (`str`, *可选*) ：要写入命令的 stdin 的数据。

on_stdout (`Callable[[str], None]`, *可选*) ：当 stdout 块到达时调用回调。

on_stderr (`Callable[[str], None]`, *可选*) ：当 stderr 块到达时调用回调。

check (`bool`，*可选*，默认为`True`)：如果为True，则在非零退出时提高`SandboxCommandError`。

背景（`bool`，*可选*，默认为`False`）：如果为True，则启动分离命令并立即返回[SandboxProcess](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess)，而不是等待它并返回[SandboxCommandResult](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxCommandResult)。

在沙箱中运行命令并等待它，实时流式传输输出。

使用 `background=True` 命令将独立启动，并且 `run` 返回
[SandboxProcess](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess) 立即，无需等待它完成 — 方便
服务器和其他长时间运行的进程。稍后用 [Sandbox.processes()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.processes) 列出它们
然后用 [SandboxProcess.kill()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess.kill) 停止一个。流式传输/仅等待选项
（`timeout`、`stdin`、`on_stdout`、`on_stderr`、`check`）不适用于该模式。

返回：a [SandboxCommandResult](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxCommandResult)（带有 `exit_code`、`stdout`、`stderr`，
`duration_ms`)，或`background=True`时为[SandboxProcess](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess)。

## SandboxPool[[huggingface_hub.SandboxPool]]#### Huggingface_hub.SandboxPool[[huggingface_hub.SandboxPool]]

```python
huggingface_hub.SandboxPool(image: str = 'python:3.12', flavor: str = 'cpu-basic', sandboxes_per_host: int = 50, warm_up: int = 1, max_hosts: int | None = None, name: str | None = None, idle_timeout: int | float | str | None = 600, namespace: str | None = None, start_timeout: float = 120.0, token: str | None = None, _connect_mode: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L919)

一组共享的“主机”作业，每个作业都包含许多内陆隔离的沙箱。

一台主机是一个计费的 HF 作业（一台 VM）；它运行沙箱服务器和多路复用
最多 `sandboxes_per_host` 轻量级沙箱，彼此隔离
uid + Landlock LSM。这使得大扇出变得便宜（VM 成本被共享
跨所有沙箱）并且快速（创建沙箱是〜一个代理往返
一旦主人热情）。最适合许多并行 CPU 沙箱，例如 RL 部署；
对于相互不信任的工作负载之间的 GPU 或强大的 VM 级隔离，请使用
[Sandbox.create()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.create) 代替。

构造函数预先配置 `warm_up` 主机（默认 1）并阻止直到它们被配置
准备好了；然后，当请求沙箱时，可以按需配置更多主机，并且所有主机

在 `close()` 被拆除（或闲置时，通过 `idle_timeout`）。用户从不管理作业：

```python
>>> from huggingface_hub import SandboxPool
>>> with SandboxPool(image="python:3.12", flavor="cpu-basic", warm_up=2) as pool:
...     boxes = [pool.create() for _ in range(100)]   # packed across the warm hosts
...     print(boxes[0].run("echo hi").stdout)
hi
````create()` 一次创建**一个**沙箱：它重用仍然有空闲的主机
启动新的容量之前，您可以随着工作的到来而按需增长。为了避免
前几次调用冷启动，为主机预先配置`warm_up`（或
`warm`）。温暖的主机是通过工作标签发现的，因此可以在不同的地方重复使用工作**
也处理**（具有相同 `image`/`flavor`/`name` 的新池附加到

举办了一场较早的比赛，落后了）：

```python
>>> pool = SandboxPool(image="python:3.12")
>>> sbx = pool.create()    # finds a warm host (here or in another process), else boots one
```

#### 关闭[[huggingface_hub.SandboxPool.close]]

```python
close()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L1258)

释放池。幂等。

对于我们创建的池，这会终止所有主机作业（因此它们的所有作业）
沙箱）。对于 `connect()` 的处理，它仅释放本地 HTTP 客户端：
共享主机可能正在为其他客户端提供服务，因此 - 就像 [Sandbox.connect()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.connect) - 留下一个
`with` 块不得将其撕毁。显式终止连接池的主机
与`hf sandbox pool delete <id>`。

#### 连接[[huggingface_hub.SandboxPool.connect]]

```python
connect(pool_id: str, namespace: str | None = None, token: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L1055)

**参数：**

pool_id (`str`) ：首次创建池时返回的 id。

命名空间（`str`，*可选*）：用于搜索池主机的命名空间（默认为您的主机）。

令牌（`str`，*可选*）：HF 令牌覆盖。从任何机器通过 ID 重新连接到正在运行的池 - 不需要本地状态。

查找标有 `pool_id` 的正在运行的主机并重建池的配置
（图像/风味/密度/主机空闲）来自该主机作业的规范和环境变量，返回
[SandboxPool](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxPool) 准备好 `create()` 更多沙箱 — 打包到运行中
主机，或在主机已满时启动重复的（相同配置）。

如果未找到正在运行的主机，则引发 `SandboxError`（一旦池停止存在
它的所有主机都消失了——空闲超时或被杀死）。

#### 创建[[huggingface_hub.SandboxPool.create]]

```python
create(env: dict[str, typing.Any] | None = None, idle_timeout: int | float | str | None = 600, forward_hf_token: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L1137)

**参数：**

env (`dict[str, Any]`, *可选*) ：此沙箱的环境变量（每个沙箱都有自己的环境变量）。

idle_timeout（`int`或`float`或`str`，*可选*，默认为`600`）：每个沙箱空闲超时 - 在如此长时间的不活动（没有API调用，没有正在运行的进程）之后，沙箱将从其主机中逐出。与主机空闲超时不同。通过 `None` 禁用。forward_hf_token（`bool`，*可选*，默认为`False`）：如果为True，则将您的HF令牌作为`HF_TOKEN`注入沙箱中（选择加入）。与专用沙箱的`secrets`不同，池化沙箱的环境在创建时传递到主机服务器（从未存储在主机作业中），因此它不会出现在任何作业的元数据中。

创建一个沙箱，根据需要配置一台主机。

重用具有空闲容量的主机（此池的主机，或通过作业标签找到的热主机）
/ 本地缓存）在启动新主机之前，因此针对热主机的`create()`
费用〜一次往返。反复调用即可扇出；使用`warm_up`（或`warm`）
预先配置主机并避免第一次调用时冷启动。如果主机填满
在我们下面（另一个进程打包了它）或者缓存的主机消失了，沙箱就消失了
重新放置在另一台主机（或新主机）上。

#### 温暖[[huggingface_hub.SandboxPool.warm]]

```python
warm(num_hosts: int = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L1110)

确保 `num_hosts` 空主机正在运行并让它们保持运行。返回
池的主机作业 ID。用于预先“创建”池：主机携带池标签和配置（在
他们的环境变量），所以后来的`SandboxPool.connect(pool_id)`（甚至来自另一个
机）找到它们并生成沙箱而无需冷启动。主办方保留
计费直到被杀或闲置。

在启动之前采用已为此池运行的主机（通过作业标签找到），
因此，`connect()` 之后的 `warm()` — 或重复的 `warm()` — 最多可达 `num_hosts`
而不是复制现场主机并吹过`max_hosts`。

## 数据结构

### SandboxCommandResult[[huggingface_hub.SandboxCommandResult]]

#### Huggingface_hub.SandboxCommandResult[[huggingface_hub.SandboxCommandResult]]

```python
huggingface_hub.SandboxCommandResult(exit_code: int | None, stdout: str, stderr: str, signal: int | None = None, timed_out: bool = False, duration_ms: int = 0)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L114)

在具有 [Sandbox.run()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.run) 的沙箱中执行命令的结果。

### SandboxProcess[[huggingface_hub.SandboxProcess]]

#### Huggingface_hub.SandboxProcess[[huggingface_hub.SandboxProcess]]

```python
huggingface_hub.SandboxProcess(pid: int, cmd: typing.Union[str, typing.List[str]], _sandbox: Sandbox, tag: str | None = None, started_at_ms: int | None = None, running: bool = True, exit_code: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L134)

后台进程在沙箱中以 [Sandbox.run()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.run)`(..., background=True)` 启动。

使用 [Sandbox.processes()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.Sandbox.processes) 列出沙箱的进程，并使用 [SandboxProcess.kill()](/docs/huggingface_hub/v1.29.0/en/package_reference/sandbox#huggingface_hub.SandboxProcess.kill) 停止进程。
已完成的进程将保留在列表中，直到沙箱被删除，因此 `running` 和
`exit_code` 判断进程是否仍然存在或已经退出（截至其列出时）。

#### 杀死[[huggingface_hub.SandboxProcess.kill]]

```python
kill()
```[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L152)

终止后台进程（幂等服务器端）。

### FileEntry[[huggingface_hub._sandbox.FileEntry]]

#### Huggingface_hub._sandbox.FileEntry[[huggingface_hub._sandbox.FileEntry]]

```python
huggingface_hub._sandbox.FileEntry(name: str, path: str, type: typing.Literal['file', 'dir', 'symlink'], size: int, mtime_ms: int | None = None, mode: str = '')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_sandbox.py#L158)

沙箱内的文件或目录。

## 错误

### SandboxError[[huggingface_hub.errors.SandboxError]]

#### Huggingface_hub.errors.SandboxError[[huggingface_hub.errors.SandboxError]]

```python
huggingface_hub.errors.SandboxError(message: str, status_code: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/errors.py#L586)

**参数：**

status_code ：如果错误源自 API 响应，则沙箱内服务器返回的 HTTP 状态（例如，`404` 表示丢失文件）。 `None` 否则。

沙箱操作的基本异常（请参阅`huggingface_hub.Sandbox`）。

### SandboxCommandError[[huggingface_hub.errors.SandboxCommandError]]

#### Huggingface_hub.errors.SandboxCommandError[[huggingface_hub.errors.SandboxCommandError]]

```python
huggingface_hub.errors.SandboxCommandError(cmd, result)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/errors.py#L599)

**参数：**

cmd ：失败的命令。

结果：完整的`SandboxCommandResult`（exit_code，stdout，stderr，...）。

当沙箱中运行的命令以非零代码退出时引发。

### 存储卡
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/cards.md